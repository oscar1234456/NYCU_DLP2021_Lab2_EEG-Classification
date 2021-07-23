from torch import nn

class EEGnet(nn.Module):
    def __init__(self):
        super(EEGnet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 18), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 10), stride=(1, 2), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),  # TODO:Waiting for substitution 1
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.5)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),  # TODO:Waiting for substitution 2
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.5)
        )
        self.flatten = nn.Flatten()
        self.classify = nn.Sequential(
            nn.Linear(in_features=352, out_features=2, bias=True)  # Question about in_features
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


class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
        self.firstDoubleConv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), padding="valid"),
            nn.Conv2d(25, 25, kernel_size=(2, 1), padding="valid"),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.secondConv = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), padding="valid"),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.thirdConv = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), padding="valid"),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.finalConv = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), padding="valid"),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.flatten = nn.Flatten()
        self.classify = nn.Sequential(
            nn.Linear(in_features=9000, out_features=2, bias=True)  # Question about in_features
        )

    def forward(self, x):
        firstFeature = self.firstDoubleConv(x)
        secondFeature = self.secondConv(firstFeature)
        thirdFeature = self.thirdConv(secondFeature)
        flatten = self.flatten(thirdFeature)
        finalX = self.classify(flatten)
        return finalX