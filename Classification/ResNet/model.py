import torch.nn as nn
import torch

# For ResNet18, ResNet34
class BasicBlock(nn.Module):
    """
    ResNet18, ResNet34的残差块中卷积核个数是相同的，所以expansion=1
    |3x3, 64|
    |3x3, 64|
    """
    expansion = 1
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        in_channel:     残差块第一个卷积核输入通道数
        out_channel:    残差块第一个卷积核输出通道数，对应主分支卷积核个数
        stride:         默认为1，conv2及以后的第一个残差块需要下采样，设置为2
        downsample:     虚线残差结构，残差分支需要降采样             
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        # 虚线分支需要降采样
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 主线和支线分支相加再relu
        out += identity
        out = self.relu(out)
        
        return out
    

# For ResNet50, ResNet101, ResNet152
class Bottleneck(nn.Module):
    """
    ResNet50, ResNet101, ResNet152的残差块中卷积核个数是倍增的，所以expansion=4
    |3x3,  64|
    |3x3,  64|
    |3x3, 256|
    """
    expansion = 4
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        in_channel:     残差块第一个卷积核输入通道数
        out_channel:    残差块第一个卷积核输出通道数，对应主分支卷积核个数
        stride:         默认为1，conv2及以后的第一个残差块需要下采样，设置为2
        downsample:     虚线残差结构，残差分支需要降采样
        即残差块的卷积核个数分别为[out_channel, out_channel, out_channel * expansion]             
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, 
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, block_num, num_cls=1000, include_top=True):
        """
        block:          BasicBlock/Bottleneck
        block_num:      列表，代表每个Stage的残差块数目，例如ResNet50为[3, 4, 6, 3] 
        num_cls:        下游分类任务类别数目
        include_top:    扩展用   
        """
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64    # 最大池化层后的通道数
        
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 只有第一个Stage的步长为1
        self.layer1 = self._make_layer(block, 64, block_num[0], stride=1)
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
        
        # 全连接层
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.in_channel, num_cls)
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def _make_layer(self, block, channel, block_num, stride=1):
        """
        block:          BasicBlock/Bottleneck
        channel:        残差块第一个卷积层的输入通道数
        block_num:      列表，代表每个Stage的残差块数目
        stride:         默认为1，虚线结构残差块需要减小尺寸，设置为2
        """
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
            
        layers = []
        # 每个Stage的第一个残差块特殊处理
        layers.append(block(self.in_channel, channel, stride=stride, downsample=downsample))
        
        # 之后当前Stage剩下残差块和下一个Stage第一个残差块输入通道均为为channel * block.expansion
        self.in_channel = channel * block.expansion
        
        # 当前Stage剩余残差块没有特殊处理
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, stride=1))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)
            
        return x

# https://download.pytorch.org/models/resnet18-f37072fd.pth
def resnet18(num_cls=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_cls, include_top)

# https://download.pytorch.org/models/resnet34-b627a593.pth
def resnet34(num_cls=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_cls, include_top)

# https://download.pytorch.org/models/resnet50-0676ba61.pth
def resnet50(num_cls=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_cls, include_top)

# https://download.pytorch.org/models/resnet101-63fe2227.pth
def resnet101(num_cls=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_cls, include_top)

# https://download.pytorch.org/models/resnet152-394f9c45.pth
def resnet152(num_cls=1000, include_top=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_cls, include_top)   
    
        
            
        