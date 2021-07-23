from torch import nn

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