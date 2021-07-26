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
from model import EEGnet
from tester import test


if __name__ == '__main__':
    loss_fn = nn.CrossEntropyLoss()
    X_train, Y_train, X_test, Y_test = dataloader.read_bci_data()
    Test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).type(torch.FloatTensor),
                                           torch.from_numpy(Y_test).type(torch.LongTensor)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    EEGnetModelFinal_relu = EEGnet("ReLU").to(device)
    EEGnetModelFinal_relu.load_state_dict(torch.load('EEGnet_ReLu_weight3.pth'))
    # print(EEGnetModelFinal_relu)
    print("EEGnet(Modified) (ReLU):")
    test(EEGnetModelFinal_relu, Test_loader,loss_fn,device, "Test")
    print()
    EEGnetModelFinal_LeakyReLU= EEGnet("LeakyReLU").to(device)
    EEGnetModelFinal_LeakyReLU.load_state_dict(torch.load('EEGnet_LeakyReLU_weight.pth'))
    # print(EEGnetModelFinal_LeakyReLU)
    print("EEGnet(Modified) (LeakyReLU):")
    test(EEGnetModelFinal_LeakyReLU, Test_loader, loss_fn, device, "Test")
    print()
    EEGnetModelFinal_ELU = EEGnet("ELU").to(device)
    EEGnetModelFinal_ELU.load_state_dict(torch.load('EEGnet_ELU_weight.pth'))
    # print(EEGnetModelFinal_ELU)
    print("EEGnet(Modified) (ELU):")
    test(EEGnetModelFinal_ELU, Test_loader, loss_fn, device, "Test")
