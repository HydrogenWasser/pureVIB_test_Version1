import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt

def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5

class HydrogenIB(nn.Module):
    """
    Hydrogen IB model (FC).
    """
    def __init__(self, device):
        super(HydrogenIB, self).__init__()
        self.beta = 0.12
        self.z_dim = 256
        self.num_sample = 12
        self.device = device
        self.batch_size = 100
        self.mask_center = [5, 14, 23]
        self.aug_weight = 0.9
        self.w_ce = 1
        self.w_reg = 0.1
        self.train_flag = True
        self.relu = nn.ReLU(inplace=True)
        self.prior = Normal(torch.zeros(1, self.z_dim).to(self.device),torch.ones(1, self.z_dim).to(self.device))
        self.Encoder = nn.Sequential(nn.Linear(in_features=784, out_features=1024),
                                     nn.ReLU(),
                                     nn.Linear(in_features=1024, out_features=1024),
                                     nn.ReLU(),
                                     nn.Linear(in_features=1024, out_features=2 * self.z_dim))

        self.Decoder = nn.Sequential(nn.Linear(in_features=self.z_dim, out_features=10),
                                     nn.ReLU()
                                     )
        # self.linear = nn.Linear(in_features=128, out_features=10)

    def gaussian_noise(self, num_samples, K):
        return torch.normal(torch.zeros(*num_samples, K), torch.ones(*num_samples, K)).to(self.device)

    def sample_prior_Z(self, num_samples):
        return self.gaussian_noise(num_samples=num_samples, K=self.dimZ)

    def encoder_result(self, x):
        encoder_output = self.Encoder(2*x-1)
        mean_C = encoder_output[:, :self.z_dim]
        std_C = torch.nn.functional.softplus(encoder_output[:, self.z_dim:])
        return mean_C, std_C

    def sample_Z(self, num_samples, x):
        mean_C, std_C = self.encoder_result(x)
        return mean_C, std_C, mean_C + std_C * self.gaussian_noise(num_samples=(num_samples, self.batch_size), K=self.z_dim)
               # mean_S, std_S, mean_S + std_S * self.gaussian_noise(num_samples=(num_samples, batch_size), K=self.z_dim)

    def get_logits(self, x):
        #encode
        mean_C, std_C, z_C = self.sample_Z(num_samples=self.num_sample, x=x)
        # z_C (12, 100, 256)
        #decode
        y_logits = self.Decoder(z_C)
        # y_logits = self.linear(z_C) # y_logits (12, 100, 10)

        z_C = torch.mean(z_C, dim=0) # z_C (100, 128)

        return mean_C, std_C, y_logits, z_C
        # mean_C (100, 256)
        # std_C (100, 256)
        # y_logits (12, 100, 10)
        # z_C (100, 128)

    def forward(self, x):
        # if self.train_flag:
        final_pred, z_scores, features, outputs, mean_Cs, std_Cs, y_logits_s = self.causal_forward(x)
        return final_pred, z_scores, features, outputs, mean_Cs, std_Cs, y_logits_s
        #
        # else:
        #     x = x.view(-1, 28 * 28)
        #     mean_C, std_C, y_logits, z_C = self.get_logits(x)
        #     y_pre = torch.mean(y_logits, dim=0)
        #     return y_pre, z_C
        #     # y_pre (100, 10)
        #     # z_C (100, 128)

    def smooth_l1_loss(self, x, y):
        diff = F.smooth_l1_loss(x, y, reduction='none')
        diff = diff.sum(1)
        diff = diff.mean(0)
        return diff

    def get_mean_wo_i(self, inputs, i):
        return (sum(inputs) - inputs[i]) / float(len(inputs) - 1)

    def causal_forward(self, x):

        b, w, h = x.shape
        samples = []
        masks = []
        NUM_LOOP = 9
        NUM_INNER_SAMPLE = 3
        NUM_TOTAL_SAMPLE = NUM_LOOP * NUM_INNER_SAMPLE
        for i in range(NUM_TOTAL_SAMPLE):
            # differentiable sampling
            sample = self.relu(x + x.detach().clone().uniform_(-1,1) * self.aug_weight)
            sample = sample / (sample + 1e-5)
            #on_sample = torch.clamp(x + torch.randn_like(x) * 0.1, min=0, max=1)
            if i % NUM_INNER_SAMPLE == 0:
                idx = int(i // NUM_INNER_SAMPLE)
                x_idx = int(idx // 3)
                y_idx = int(idx % 3)
                center_x = self.mask_center[x_idx]
                center_y = self.mask_center[y_idx]
            # attention
            mask = self.create_mask(w, h, center_x, center_y, alpha=10.0).to(x.device)
            sample = sample * mask.float()
            samples.append(sample)
            masks.append(mask)

        outputs = []
        features = []
        z_scores = []
        mean_Cs = []
        std_Cs = []
        y_logits_s = []
        for i in range(NUM_LOOP):
            # Normalized input
            inputs = sum(samples[NUM_INNER_SAMPLE * i : NUM_INNER_SAMPLE * (i+1)]) / NUM_INNER_SAMPLE
            z_score = (sum(masks[NUM_INNER_SAMPLE * i : NUM_INNER_SAMPLE * (i+1)]).float() / NUM_INNER_SAMPLE).mean()
            # forward modules
            inputs = inputs.view(-1, 28*28)

            mean_C, std_C, y_logits, z_C = self.get_logits(inputs)
            y_pre = torch.mean(y_logits, dim=0)
            # z_C (100, 128)
            # y_pre (100, 10)
            feats = z_C
            preds = y_pre

            z_scores.append(z_score.view(1,1).repeat(b, 1))
            features.append(feats)
            outputs.append(preds)
            mean_Cs.append(mean_C)
            std_Cs.append(std_C)
            y_logits_s.append(y_logits)
        final_pred = sum([pred / (z + 1e-9) for pred, z in zip(outputs, z_scores)]) / NUM_LOOP

        return final_pred, z_scores, features, outputs, mean_Cs, std_Cs, y_logits_s
        # final_pred (100, 10)
        # z_scores (9, 100, 1)
        # features (9, 100, 128)
        # outputs (9, 100, 10)



    def compute_IB_loss(self, mean, std, y_logits, y_batch, num_samples):
        prior_Z_distr = torch.zeros(self.batch_size, self.z_dim).to(self.device), torch.ones(self.batch_size, self.z_dim).to(self.device)
        enc_dist = mean, std
        I_X_T_bound = torch.mean(KL_between_normals(enc_dist, prior_Z_distr)) / math.log(2)

        # compute I(Y,T)
        # y_logits (12, 100, 10)
        loss_func = nn.CrossEntropyLoss(reduce=False)
        y_logits = y_logits.permute(1, 2, 0)    # y_logits (100, 10, 12)
        y_label = y_batch[:, None].expand(-1, num_samples)      # y_label (100, 12)
        cross_entropy_loss = loss_func(y_logits, y_label)
        cross_entropy_loss_montecarlo = torch.mean(cross_entropy_loss, dim=-1)
        I_Y_T_bound = torch.mean(cross_entropy_loss_montecarlo, dim=0) / math.log(2)

        # compute Loss
        Ibloss = I_Y_T_bound + self.beta*I_X_T_bound
        return Ibloss, I_X_T_bound, math.log(10, 2) - I_Y_T_bound

    def eval_batch_loss(self, x_batch, y_batch, num_samples):
        x_batch = x_batch.view(-1, 28*28)
        mean, std, y_logits, z_C = self.get_logits(x_batch)
        Ibloss, I_X_T_bound, I_Y_T_bound = self.compute_IB_loss(mean, std, y_logits, y_batch, num_samples)

        return Ibloss, I_X_T_bound, I_Y_T_bound

    def train_batch_loss(self, logits, features, z_scores, y_logits_s, mean_Cs, std_Cs, y_batch, num_samples):

        all_Ibloss = []
        all_regs = []
        all_I_XT = []
        all_I_YT = []

        for i in range(len(logits)):

            mean = mean_Cs[i]
            std = std_Cs[i]
            y_logits = y_logits_s[i]
            Ibloss, I_X_T_bound, I_Y_T_bound = self.compute_IB_loss(mean, std, y_logits, y_batch, num_samples)

            all_Ibloss.append(Ibloss)
            all_I_XT.append(I_X_T_bound)
            all_I_YT.append(I_Y_T_bound)

        for i in range(len(features)):
            reg_loss = self.smooth_l1_loss(features[i] * self.get_mean_wo_i(z_scores, i),
                                           self.get_mean_wo_i(features, i) * z_scores[i])
            # iter_info_print['ciiv_l1loss_{}'.format(i)] = reg_loss.sum().item()
            all_regs.append(reg_loss)

        loss = self.w_ce * sum(all_Ibloss) / len(all_Ibloss) + self.w_reg * sum(all_regs) / len(all_regs)
        I_XT = sum(all_I_XT) / len(all_I_XT)
        I_YT = sum(all_I_YT) / len(all_I_YT)

        return loss, I_XT, I_YT


    def give_beta(self, beta):
        self.beta = beta

    def create_mask(self, w, h, center_x, center_y, alpha=10.0):
        widths = torch.arange(w).view(1, -1).repeat(h,1)
        heights = torch.arange(h).view(-1, 1).repeat(1,w)
        mask = ((widths - center_x)**2 + (heights - center_y)**2).float().sqrt()
        # non-linear
        mask = (mask.max() - mask + alpha) ** 0.3
        mask = mask / mask.max()
        # sampling
        mask = (mask + mask.clone().uniform_(0, 1)) > 0.9
        mask.float()
        return mask.unsqueeze(0)





class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average
