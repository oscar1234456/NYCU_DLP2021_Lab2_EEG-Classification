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

    DeepConvNetFinal_ReLU = DeepConvNet("ReLU").to(device)
    DeepConvNetFinal_ReLU.load_state_dict(torch.load('DeepConvNetModel_ReLU_weight2.pth'))
    # print(DeepConvNetFinal_ReLU)
    print("DeepConvNetFinal(Modified) (ReLU):")
    test( DeepConvNetFinal_ReLU, Test_loader,loss_fn,device, "Test")
    print()

    DeepConvNetFinal_LeakyReLU = DeepConvNet("LeakyReLU").to(device)
    DeepConvNetFinal_LeakyReLU.load_state_dict(torch.load('DeepConvNetModel_LeakyReLu_weight.pth'))
    # print(DeepConvNetFinal_LeakyReLU)
    print("DeepConvNetFinal(Modified) (LeakyReLU):")
    test(DeepConvNetFinal_LeakyReLU, Test_loader, loss_fn, device, "Test")
    print()

    DeepConvNetFinal_ELU = DeepConvNet("ELU").to(device)
    DeepConvNetFinal_ELU.load_state_dict(torch.load('DeepConvNetModel_ELu_weight2.pth'))
    # print(DeepConvNetFinal_ELU)
    print("DeepConvNetFinal(Modified) (ELU):")
    test(DeepConvNetFinal_ELU, Test_loader, loss_fn, device, "Test")

