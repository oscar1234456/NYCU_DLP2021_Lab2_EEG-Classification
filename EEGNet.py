##
#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab2 EEG Classification
#Date: 2021/07/24
#Subject: Implement two CNN Model: EEGNet, DeepConvNet using Pytorch
#Email: oscarchen.cs10@nycu.edu.tw
import dataloader
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from torch import nn, optim
from model import EEGnet
from tester import test
from trainner import train
import pickle


## NN
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


EEGnetModel = EEGnet("ELU").to(device)
print(EEGnetModel)

## Parameters
Batch_size = 64
Learning_rate = 0.05
Epochs = 300
optimizer = optim.Adam(EEGnetModel.parameters())
loss_fn = nn.CrossEntropyLoss()

## Data loader
X_train, Y_train, X_test, Y_test = dataloader.read_bci_data()
Train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).type(torch.FloatTensor),  torch.from_numpy(Y_train).type(torch.LongTensor)), batch_size=Batch_size)
Test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).type(torch.FloatTensor),  torch.from_numpy(Y_test).type(torch.LongTensor)), batch_size=Batch_size)
# X_train:(1080, 1, 2, 750) Y_train:(1080,)  X_test:(1080, 1, 2, 750) Y_test:(1080,)
# X: [ [ [     [ 750 ]   ,   [ 750 ]      ]    ], ......, []    ]
for X, y in Train_loader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

## Training
EEGnetModelTrainingRes = list()
EEGnetModelTestingRes = list()
for t in range(Epochs):
    print(f"-----------Epoch | {t+1} |------------")
    train(Train_loader, EEGnetModel, loss_fn, optimizer, device)
    trainingRes = test(EEGnetModel, Train_loader, loss_fn, device, "Train")
    testingRes = test(EEGnetModel, Test_loader, loss_fn, device, "Test")
    EEGnetModel.train()
    EEGnetModelTrainingRes.append(trainingRes)
    EEGnetModelTestingRes.append(testingRes)
print("Done!")

test(EEGnetModel,Test_loader,loss_fn, device,"Test")

##Save Training & Testing Accuracy Result
with open('EEGnet_ELU_Training.pickle', 'wb') as f:
    pickle.dump(EEGnetModelTrainingRes, f)
with open('EEGnet_ELU_Testing.pickle', 'wb') as f:
    pickle.dump(EEGnetModelTestingRes, f)

## Save my model
torch.save(EEGnetModel.state_dict(), 'EEGnet_ELU_weight.pth') #Save weight