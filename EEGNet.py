##
import dataloader
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt


## NN
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class EEGnet(nn.Module):
    def __init__(self):
        super(EEGnet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),  # TODO:Waiting for substitution 1
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),  # TODO:Waiting for substitution 2
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.flatten = nn.Flatten()
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)  # Question about in_features
        )

    def forward(self, x):
        # print(f"data_in:{x.shape}")
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        # print("bow!")
        # print(f"x infor:{x.shape}")
        x = self.flatten(x)
        # print(f"out infor:{x.shape}")
        finalX = self.classify(x)
        return finalX


EEGnetModel = EEGnet().to(device)
print(EEGnetModel)

## Parameters
Batch_size = 64
Learning_rate = 1e-2
Epochs = 1000
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
## Trainner
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print("size:", size)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # print(f"X :{X.shape}")
        # print(f"batch:{batch}")
        # print("y length:",y.shape)
        # Compute prediction error
        pred = model(X)
        print(f"pred:{pred.shape}")
        # print(f"y shape:{y.shape}")
        loss = loss_fn(pred,  y )
        # #
        # # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
## Tester
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # disable gradient calculation for efficiency
        for data, target in test_loader:
            # Prediction
            output = model(data.to(device))

            # Compute loss & accuracy
            test_loss += loss_fn(output,  target.to(device) ).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.to(device).view_as(pred)).sum().item() # how many predictions in this batch are correct

    test_loss /= len(test_loader.dataset)

    # Log testing info
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


## Training

for t in range(Epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(Train_loader, EEGnetModel, loss_fn, optimizer)
print("Done!")

test(EEGnetModel,Test_loader)
