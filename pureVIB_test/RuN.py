from utils import *


"                                                     "
"               Los Geht's!                           "
"                                                     "


num_epoch = 50
HydrogenIB_train_tag = True
cnn_trian_tag = True
HydrogenIB_attac_tag = True
cnn_attack_tag = True
# adverType = ["fgsm", "pgd"]
# adverType_ = adverType[0]
dataset = "FaschionMNIST"
model = HydrogenIB(device=device).to(device)
cnn_model = CleanCNN(device=device).to(device)
ema = EMA(0.999)
if HydrogenIB_train_tag:
    print("------------------------------------Begin HydrogenIB Train-------------------------")
    # HydrogenIB_Train(model=model, ema=ema, num_epoch=num_epoch, dataset=dataset)
    HydrogenIB_eval(model=model, dataset=dataset)
    print("------------------------------------End HydrogenIB Train-------------------------")
if cnn_trian_tag:
    print("------------------------------------Begin CleanCNN Train-------------------------")
    Clean_CNN_Train(model=cnn_model, num_epoch=num_epoch, dataset=dataset)
    print("------------------------------------End CleanCNN Train-------------------------")
if cnn_attack_tag:
    print("------------------------------------Begin CleanCNN Test & Adversial-------------------------")
    # if adverType_ == adverType[1]:
    print("-------------------PGD TEST------------------")
    Clean_CNN_fgsm(cnn_model, 0.031, "pgd", dataset)
    Clean_CNN_fgsm(cnn_model, 0.1, "pgd", dataset)
    Clean_CNN_fgsm(cnn_model, 0.2, "pgd", dataset)
    Clean_CNN_fgsm(cnn_model, 0.3, "pgd", dataset)
    # else:
    print("-------------------FGSM TEST------------------")
    Clean_CNN_fgsm(cnn_model, 0.031, "fgsm", dataset)
    Clean_CNN_fgsm(cnn_model, 0.1, "fgsm", dataset)
    Clean_CNN_fgsm(cnn_model, 0.2, "fgsm", dataset)
    Clean_CNN_fgsm(cnn_model, 0.3, "fgsm", dataset)
    print("------------------------------------End CleanCNN Test & Adversial-------------------------")
if HydrogenIB_attac_tag:
    print("------------------------------------Begin HydrogenIB Test & Adversial-------------------------")
    # if adverType_ == adverType[1]:
    print("-------------------PGD TEST------------------")
    HydrogenIB_fgsm(model, 0.031, "pgd", dataset)
    HydrogenIB_fgsm(model, 0.1, "pgd", dataset)
    HydrogenIB_fgsm(model, 0.2, "pgd", dataset)
    HydrogenIB_fgsm(model, 0.3, "pgd", dataset)
    # else:
    print("-------------------FGSM TEST------------------")
    HydrogenIB_fgsm(model, 0.031, "fgsm", dataset)
    HydrogenIB_fgsm(model, 0.1, "fgsm", dataset)
    HydrogenIB_fgsm(model, 0.2, "fgsm", dataset)
    HydrogenIB_fgsm(model, 0.3, "fgsm", dataset)
    print("------------------------------------End HydrogenIB Test & Adversial-------------------------")
