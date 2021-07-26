#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab2 EEG Classification
#Date: 2021/07/24
#Subject: Implement two CNN Model: EEGNet, DeepConvNet using Pytorch
#Email: oscarchen.cs10@nycu.edu.tw

from torch import nn
import torch
def test(model, test_loader, loss_fn,device, choose):
    print("-----Testing Start(Testing Set)-----") if choose=="Test" else print("-----Testing Start(Training Set)-----")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            test_loss += loss_fn(output,  target.to(device) ).item() #conver to Python float
            pred = output.argmax(dim=1, keepdim=True) #choose the highest value index
            correct += pred.eq(target.to(device).view_as(pred)).sum().item() #count the amount of correct

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\n Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(test_loader.dataset),accuracy))
    print("-----Testing Over-----")
    return  accuracy