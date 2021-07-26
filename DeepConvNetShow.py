#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab2 EEG Classification
#Date: 2021/07/24
#Subject: Implement two CNN Model: EEGNet, DeepConvNet using Pytorch
#Email: oscarchen.cs10@nycu.edu.tw
import torch
from torch import nn
import dataloader
from torch.utils.data import DataLoader, TensorDataset
from model import DeepConvNet
from tester import test

if __name__ == '__main__':
    loss_fn = nn.CrossEntropyLoss()
    X_train, Y_train, X_test, Y_test = dataloader.read_bci_data()
    Test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).type(torch.FloatTensor),
                                           torch.from_numpy(Y_test).type(torch.LongTensor)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    DeepConvNetFinal = DeepConvNet("ELU").to(device)
    DeepConvNetFinal.load_state_dict(torch.load('DeepConvNetModel_ELu_weight1.pth'))
    print( DeepConvNetFinal)
    print("DeepConvNetFinal(Modified) (ELU):")
    test( DeepConvNetFinal, Test_loader,loss_fn,device, "Test")

