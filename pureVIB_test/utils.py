import math, random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torch.utils.data import TensorDataset, Dataset, DataLoader
import matplotlib as plt
from HydrogenIB import *
from Clean_CNN import *
import os
import fgsm
import pgd
os.environ["GIT_PYTHON_REFRESH"] = "quiet"




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 100
samples_amount = 12


"                                                     "
"               Daten Vorbereitung                    "
"                                                     "

train_data = FashionMNIST(root='./FashionMnist_data',train=True,download=True)
train_dataset_causal = TensorDataset(train_data.train_data.float() / 255, train_data.train_labels)
train_loader_causal = DataLoader(train_dataset_causal, batch_size=batch_size, shuffle=True)

test_data = FashionMNIST(root='./FashionMnist_data',train=True,download=False)
test_dataset_causal = TensorDataset(test_data.test_data.float() / 255, test_data.test_labels)
test_loader_causal = DataLoader(test_dataset_causal, batch_size=batch_size)








# train_data = MNIST('./data', download=True, train=True)
# train_dataset_causal = TensorDataset(train_data.train_data.float() / 255, train_data.train_labels)
# train_loader_causal = DataLoader(train_dataset_causal, batch_size=batch_size)
#
#
# test_data = MNIST('./data', download=True, train=False)
# test_dataset_causal = TensorDataset(test_data.test_data.float() / 255, test_data.test_labels)
# test_loader_causal = DataLoader(test_dataset_causal, batch_size=batch_size)


"                                                     "
"               Modell Speichung                      "
"                                                     "

def save(net, name, dataset):
    path = './model'
    if not os.path.exists(path):
        os.mkdir(path)
    net_path = path + '/' + dataset + name +'.pkl'
    net = net.cpu()
    torch.save(net.state_dict(), net_path)
    net.to(device)

def load(net, name, dataset):
    net_path = './model/' + dataset + name +'.pkl'
    net.load_state_dict(torch.load(net_path))
    net.to(device)
    return net

"                                                     "
"               Modell Trainierung                    "
"                                                     "

def HydrogenIB_Train(model, ema, num_epoch, dataset):

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    model.train_flag = True


    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    for epoch in range(num_epoch):
        loss_bei_epoch = []
        accuracy_bei_epoch = []
        I_X_T_bei_epoch = []
        I_Y_T_bei_epoch = []

        if epoch % 2 == 0 and epoch > 0:
            scheduler.step()

        for x_batch, y_batch in train_loader_causal:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pre, z_scores, features, logits, mean_Cs, std_Cs, y_logits_s = model(x_batch)
            y_prediction = torch.max(y_pre, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            accuracy_bei_epoch.append(accuracy.item())

            loss, I_X_T, I_Y_T = model.train_batch_loss(logits, features, z_scores, y_logits_s, mean_Cs, std_Cs, y_batch, num_samples=12)
            loss_bei_epoch.append(loss.item())
            I_X_T_bei_epoch.append(I_X_T.item())
            I_Y_T_bei_epoch.append(I_Y_T.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for name, param in model.named_parameters():
                if (param.requires_grad):
                    ema(name, param.data)


        # if(epoch%5 == 0):
        print("EPOCH: ", epoch, ", loss: ", np.mean(loss_bei_epoch), ", Accuracy: ", np.mean(accuracy_bei_epoch), ", I_X_T: ",  np.mean(I_X_T_bei_epoch), ", I_Y_T: ", np.mean(I_Y_T_bei_epoch))
    save(model, "HydrogenIB", dataset)

def HydrogenIB_eval(model, dataset):
    model = load(model, "HydrogenIB", dataset)
    model.eval()
    model.train_flag = False
    accuracy_ = []
    loss_ = []
    I_X_T_ = []
    I_Y_T_ = []

    for x_batch, y_batch in test_loader_causal:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pre, z_scores, features, logits, mean_Cs, std_Cs, y_logits_s = model(x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_.append(accuracy.item())


        loss, I_X_T, I_Y_T = model.train_batch_loss(logits, features, z_scores, y_logits_s, mean_Cs, std_Cs, y_batch, num_samples=12)
        loss_.append(loss.item())
        I_X_T_.append(I_X_T.item())
        I_Y_T_.append(I_Y_T.item())

    # if(epoch%5 == 0):
    print("TEST, Accuracy: ", np.mean(accuracy_), ", I_X_T: ",  np.mean(I_X_T_), ", I_Y_T: ", np.mean(I_Y_T_))

def HydrogenIB_fgsm(model, epsilon, advType, dataset):
    model = load(model, "HydrogenIB", dataset)
    model.eval()
    model.train_flag = False
    if advType == "fgsm":
        adver_image_obtain = fgsm.attack_model(model=model)
    elif advType == "pgd":
        adver_image_obtain = pgd.attack_model(model=model)
    accuracy_clean = []
    accuracy_adver = []
    # fuck = 0
    for x_batch, y_batch in test_loader_causal:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pre, z_scores, features, logits, mean_Cs, std_Cs, y_logits_s = model(x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_clean.append(accuracy.item())


        perturbed_x_batch = adver_image_obtain.generate(x_batch, eps=epsilon, y=y_batch)

        y_pre, z_scores, features, logits, mean_Cs, std_Cs, y_logits_s = model(perturbed_x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_adver.append(accuracy.item())


    # if(epoch%5 == 0):
    print("TEST, Clean Accuracy: ", np.mean(accuracy_clean), ", Adversial Accuracy: ",  np.mean(accuracy_adver))

def Clean_CNN_Train(model, num_epoch, dataset):
    model.train_flag = True
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print(device)
    for epoch in range(num_epoch):
        loss_bei_epoch = []
        accuracy_bei_epoch = []

        for x_batch, y_batch in train_loader_causal:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # print(x_batch.shape)
            preds, z_scores, features, logits = model(x_batch)

            loss = torch.mean(model.batch_loss(logits, features, z_scores, y_batch))
            loss_bei_epoch.append(loss.item())

            y_prediction = torch.max(preds, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            accuracy_bei_epoch.append(accuracy.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # if(epoch%5 == 0):
        print("EPOCH: ", epoch, ", loss: ", np.mean(loss_bei_epoch), ", Accuracy: ", np.mean(accuracy_bei_epoch))
    save(model, "CNN", dataset)


def Clean_CNN_eval(model):
    model = load(model, "CNN", dataset)
    model.eval()
    model.train_flag = False
    accuracy_ = []
    loss = []

    for x_batch, y_batch in test_loader_causal:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pre = model(x_batch)
        loss_func = nn.CrossEntropyLoss(reduce=False)
        ce_loss = torch.mean(loss_func(y_pre, y_batch))

        loss.append(ce_loss.item())

        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_.append(accuracy.item())


    # if(epoch%5 == 0):
    print("TEST, Accuracy: ", np.mean(accuracy_), ", Loss: ",  np.mean(loss))

def Clean_CNN_fgsm(model, epsilon, advType):

    model = load(model, "CNN", dataset)
    model.eval()
    model.train_flag = False
    if advType == "fgsm":
        adver_image_obtain = fgsm.attack_model(model=model)
        adver_image_obtain.is_CNN = True
    elif advType == "pgd":
        adver_image_obtain = pgd.attack_model(model=model)
        adver_image_obtain.is_CNN = True


    accuracy_clean = []
    accuracy_adver = []
    # fuck = 0
    for x_batch, y_batch in test_loader_causal:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pre = model(x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_clean.append(accuracy.item())

        perturbed_x_batch = adver_image_obtain.generate(x_batch, eps=epsilon, y=y_batch)

        y_pre = model(perturbed_x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_adver.append(accuracy.item())

    # if(epoch%5 == 0):
    print("TEST, Clean Accuracy: ", np.mean(accuracy_clean), ", Adversial Accuracy: ", np.mean(accuracy_adver))
