import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanCNN(nn.Module):
    """
    Hydrogen IB model (FC).
    """
    def __init__(self, device):
        super(CleanCNN, self).__init__()
        self.device = device
        self.batch_size = 100
        self.mask_center = [5, 14, 23]
        self.aug_weight = 0.9
        self.w_ce = 1
        self.w_reg = 0.1
        self.train_flag = True

        self.relu = nn.ReLU(inplace=True)
        self.Encoder = nn.Sequential(nn.Linear(in_features=784, out_features=1024),
                                               nn.ReLU(),
                                               nn.Linear(in_features=1024, out_features=1024),
                                               nn.ReLU(),
                                               nn.Linear(in_features=1024, out_features=256),
                                               nn.ReLU()
                                     )
        self.decoder = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        if self.train_flag:
            final_pred, z_scores, features, outputs = self.causal_forward(x)
            return final_pred, z_scores, features, outputs
        else:
            x = x.view(-1, 28 * 28)
            output = self.Encoder(x)
            output = self.decoder(output)
            return output



    def smooth_l1_loss(self, x, y):
        diff = F.smooth_l1_loss(x, y, reduction='none')
        diff = diff.sum(1)
        diff = diff.mean(0)
        return diff

    def get_mean_wo_i(self, inputs, i):
        return (sum(inputs) - inputs[i]) / float(len(inputs) - 1)

    def batch_loss(self, logits, features, z_scores, y_batch):
        all_ces = []
        all_regs = []

        loss_func = nn.CrossEntropyLoss(reduce=False)

        for i, logit in enumerate(logits):
            ce_loss = loss_func(logit, y_batch)
            # iter_info_print['ce_loss_{}'.format(i)] = ce_loss.sum().item()
            all_ces.append(ce_loss)

        for i in range(len(features)):
            reg_loss = self.smooth_l1_loss(features[i] * self.get_mean_wo_i(z_scores, i), self.get_mean_wo_i(features, i) * z_scores[i])
            # iter_info_print['ciiv_l1loss_{}'.format(i)] = reg_loss.sum().item()
            all_regs.append(reg_loss)

        loss = self.w_ce * sum(all_ces) / len(all_ces) + self.w_reg * sum(all_regs) / len(all_regs)

        return loss

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
        for i in range(NUM_LOOP):
            # Normalized input
            inputs = sum(samples[NUM_INNER_SAMPLE * i : NUM_INNER_SAMPLE * (i+1)]) / NUM_INNER_SAMPLE
            z_score = (sum(masks[NUM_INNER_SAMPLE * i : NUM_INNER_SAMPLE * (i+1)]).float() / NUM_INNER_SAMPLE).mean()
            # forward modules
            inputs = inputs.view(-1, 28*28)

            feats = self.Encoder(inputs)
            preds = self.decoder(feats)

            z_scores.append(z_score.view(1,1).repeat(b, 1))
            features.append(feats)
            outputs.append(preds)

        final_pred = sum([pred / (z + 1e-9) for pred, z in zip(outputs, z_scores)]) / NUM_LOOP


        return final_pred, z_scores, features, outputs
