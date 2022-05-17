import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module): 
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))       # (6, 28, 28)
        x = self.pool1(x)               # (6, 14, 14)
        x = F.relu(self.conv2(x))       # (16, 10, 10)
        x = self.pool2(x)               # (16, 5, 5)
        x = x.view(-1, 16*5*5)          # 展平
        x = F.relu(self.fc1(x))         # (120)
        x = F.relu(self.fc2(x))         # (84)
        x = self.fc3(x)                 # (10)
        # 计算交叉熵时会softmax          
        return x