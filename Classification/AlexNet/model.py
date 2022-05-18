import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_cls=1000):
        # input: (3, 224, 224)
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),      # (96, 55, 55)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # (96, 27, 27)
            nn.Conv2d(96, 256, kernel_size=5, padding=2),               # (256, 27, 27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # (256, 13, 13)
            nn.Conv2d(256, 384, kernel_size=3, padding=1),              # (384, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),              # (384, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),              # (256, 13, 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)                       # (256, 6, 6)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_cls)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
        