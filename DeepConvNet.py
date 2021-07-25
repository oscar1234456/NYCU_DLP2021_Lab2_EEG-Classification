##
import dataloader
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from torch import nn, optim
from model import DeepConvNet
from tester import test
from trainner import train
import matplotlib.pyplot as plt

## NN
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

DeepConvNetModel = DeepConvNet().to(device)
print(DeepConvNetModel)

## Parameters
Batch_size = 64
Learning_rate = 0.001
Epochs = 300
optimizer = optim.Adam(DeepConvNetModel.parameters())
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

for t in range(Epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(Train_loader,DeepConvNetModel, loss_fn, optimizer, device)
    test(DeepConvNetModel, Train_loader, loss_fn, device)
    print("Test_loader:")
    test(DeepConvNetModel, Test_loader, loss_fn, device)
    DeepConvNetModel.train()
print("Done!")

test(DeepConvNetModel,Test_loader,loss_fn, device)

## Save my model
torch.save(DeepConvNetModel.state_dict(), 'DeepConvNetModel_ELu_weight.pth') #Save weight