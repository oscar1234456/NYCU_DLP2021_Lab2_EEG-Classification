##
#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab2 EEG Classification
#Date: 2021/07/24
#Subject: Implement two CNN Model: EEGNet, DeepConvNet using Pytorch
#Email: oscarchen.cs10@nycu.edu.tw
import pickle
import dataloader
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn, optim
from model import DeepConvNet
from tester import test
from trainner import train

## NN
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

DeepConvNetModel = DeepConvNet('LeakyReLU').to(device)  #TODO: Change the Activation Function
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
DeepConvNetModelTrainingRes = list()
DeepConvNetModelTestingRes = list()
for t in range(Epochs):
    print(f"-----------Epoch | {t+1} |------------")
    train(Train_loader,DeepConvNetModel, loss_fn, optimizer, device)
    trainingRes = test(DeepConvNetModel, Train_loader, loss_fn, device, "Train")
    testingRes = test(DeepConvNetModel, Test_loader, loss_fn, device, "Test")
    DeepConvNetModel.train()
    DeepConvNetModelTrainingRes.append(trainingRes)
    DeepConvNetModelTestingRes.append(testingRes)
print("Done!")

test(DeepConvNetModel,Test_loader,loss_fn, device, "Test")

##Save Training & Testing Accuracy Result
with open('DeepConvNet_LeakyReLU_Training.pickle', 'wb') as f:
    pickle.dump(DeepConvNetModelTrainingRes, f)
with open('DeepConvNet_LeakyReLU_Testing.pickle', 'wb') as f:
    pickle.dump(DeepConvNetModelTestingRes, f)

## Save my model
torch.save(DeepConvNetModel.state_dict(), 'DeepConvNetModel_LeakyReLU_weight.pth') #Save weight