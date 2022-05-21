from numpy import isin
import torch
import torch.nn as nn

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    ch:         输入通道数
    divisor:    基数，函数输出的通道数应为它的倍数
    min_ch:     默认为None, 输出最小通道数
    """
    if min_ch is None:
        min_ch = divisor
    # 求最近的divisor倍数，从向下取整演变而来
    # 8-----12*****16，其中-----部分应该输出8，*****部分应该输出16
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # 保证输出通道数不小于输入通道的90%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

# 封装，实例化时使用nn.Sequential创建对象
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        """
        in_channel:     卷积核输入通道数
        out_channel:    卷积核输出通道数
        kernel_size:    卷积核大小
        stride:         卷积核步长
        groups:         分组卷积组数, 为1是传统卷积，为in_channel是深度可分离卷积
        """
        padding = (kernel_size - 1) // 2        # 卷积后尺寸不变
        # 使用父类nn.Sequential的__init__函数构造对象
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

# 倒残差结构，两头细中间粗
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        """
        in_channel:     输入通道数
        out_channel:    输出通道数
        stride:         中间卷积核的步长
        expand_ratio:   第一个1x1卷积核升维倍率
        """
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        # 步长为1且输入输出通道数相同时才有shortcut
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # MobileNetV2第一个倒残差结构没有先升维，因此不需要第一个1x1卷积
        # 此外的倒残差结构都有第一个1x1卷积
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3深度卷积
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1逐点卷积
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            # 线性激活，即y=x啥也不做
            nn.BatchNorm2d(out_channel)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class MobileNetV2(nn.Module):
    def __init__(self, num_cls=1000, alpha=1.0, round_nearest=8):
        """
        num_cls:        分类数量
        alpha:          调整网络卷积核个数的倍率
        round_nearest:  调整后的个数应为round_nearest的倍数
        """
        super(MobileNetV2, self).__init__()
        # 第一个卷积层卷积核个数
        in_channel = _make_divisible(32 * alpha, round_nearest)  
        # 最后一个卷积层卷积核个数
        last_channel = _make_divisible(1280 * alpha, round_nearest)
        block = InvertedResidual
        
        inverted_residual_setting = [
            # t, c, n, s = 扩展倍率，输出通道数，残差块个数，步长
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        features = []
        # conv1
        features.append(ConvBNReLU(3, in_channel, kernel_size=3, stride=2))
        # 创建倒残差结构
        for t, c, n, s in inverted_residual_setting:
            out_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                # 每层只有第一个倒残差结构步长可能为2
                stride = s if i == 0 else 1  
                features.append(block(in_channel, out_channel, stride, t))
                # 没经过一个倒残差结构，下一个倒残差结构的输入通道数是当前倒残差结构输出通道数
                in_channel = out_channel 
        # 特征提取部分最后一个卷积核
        features.append(ConvBNReLU(in_channel, last_channel, kernel_size=1))
        
        self.features = nn.Sequential(*features)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(last_channel, num_cls)
        )
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 全局平均池化
        x = self.avgpool(x)
        # 展平
        x = torch.flatten(x, start_dim=1)
        # 分类
        x = self.classifier(x)
        return x
        