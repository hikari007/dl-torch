from functools import partial
from turtle import width
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Callable, Optional

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    ch:             输入通道数
    divisor:        基数，函数输出的通道数应为它的倍数
    min_ch:         默认为None, 输出最小通道数
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

# 卷积+批归一化+激活
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        """
        in_planes:              输入通道数
        out_planes:             输出通道数
        kernel_size:            卷积核大小
        stride:                 卷积核步长
        groups:                 用于分组卷积
        norm_layer:             归一化层
        activation_layer:       激活层
        """
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            # 卷积层
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            # 归一化层
            norm_layer(out_planes),
            # 激活层
            activation_layer(inplace=True)
        )


# SE模块
class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        """
        input_c:            输入通道数
        squeeze_factor:     第一个fc层的输出通道衰减倍数          
        """
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, kernel_size=1)
        
    def forward(self, x: Tensor) -> Tensor:
        # 全局平均池化
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x

# MobileNetV3中bneck配置
class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expand_c: int,
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 width_multi: float):
        """
        input_c:        输入通道数
        kernel:         中间层卷积核大小
        expand_c:       第一层1x1卷积核个数(升维后的通道数)
        out_c:          最后一层1x卷积核个数(降维后的通道数)
        use_se:         是否使用SE模块
        activation:     使用的激活函数
        stride:         中间层卷积核步长
        width_multi:    MobilNetV2中的alpha倍率因子
        """
        # 利用倍率因子，对通道数调整
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expand_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        # 是否使用h-swish激活函数
        self.use_hs = activation == 'HS'
        self.stride = stride
    
    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)

# 倒残差结构
class InvertedResidual(nn.Module):
    def __init__(self,
                 cfg: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        """
        cfg:            配置文件
        norm_layer:     使用的归一化方式
        """
        super(InvertedResidual, self).__init__()
        # 判断步长是否非法
        if cfg.stride not in [1, 2]:
            raise ValueError('illegal stride value.')
        
        # 判断是否使用shortcut
        self.use_res_connect = (cfg.stride == 1 and cfg.input_c == cfg.out_c)
        
        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cfg.use_hs else nn.ReLU
        
        # 需要升维时才有第一个1x1卷积层
        if cfg.input_c != cfg.expanded_c:
            layers.append(ConvBNActivation(cfg.input_c,
                                           cfg.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))
        
        # 深度卷积
        layers.append(ConvBNActivation(cfg.expanded_c,
                                       cfg.expanded_c,
                                       kernel_size=cfg.kernel,
                                       stride=cfg.stride,
                                       groups=cfg.expanded_c,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))
        self.kernel = cfg.kernel
        self.stride = cfg.stride
        # SE模块
        if cfg.use_se:
            layers.append(SqueezeExcitation(cfg.expanded_c))
        # 逐点卷积(使用线性激活)
        layers.append(ConvBNActivation(cfg.expanded_c,
                                       cfg.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))
        
        self.block = nn.Sequential(*layers)
        self.out_channels = cfg.out_c
        
    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result    
    
class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int,
                 num_cls: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        """
        inverted_residual_setting:      各层bneck配置
        last_channel:                   倒数第二个1x1卷积层输出通道(fc1)
        num_cls:                        分类数目
        block:                          bneck使用的残差结构
        norm_layer:                     使用的归一化方式
        """
        super(MobileNetV3, self).__init__()
        
        # 判断inverted_residual_setting是否合法
        if not inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty.')
        elif not (isinstance(inverted_residual_setting, List) and 
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError('The inverted_residual_setting should be List[InvertedResidualConfig]')
        
        if block is None:
            block = InvertedResidual
            
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        
        layers: List[nn.Module] = []
        
        # 第一个卷积层
        first_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3,
                                       first_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # 构建bnecks
        for cfg in inverted_residual_setting:
            layers.append(block(cfg, norm_layer))
            
        # 构建特征提取部分最后一个1x1卷积
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        
        self.features = nn.Sequential(*layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_c, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_cls)
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    

def mobilenet_v3_large(num_cls: int = 1000, 
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    num_cls:        分类数目
    reduced_tail:   是否进一步减小网络规模C4/C5
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)
    
    reduce_divider = 2 if reduced_tail else 1
    
    # bneck配置
    inverted_residual_setting = [
        # input_c, kernel, expand_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    
    last_channel = adjust_channels(1280 // reduce_divider)
    
    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_cls=num_cls)

def mobilenet_v3_small(num_cls: int = 1000, 
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    num_cls:        分类数目
    reduced_tail:   是否进一步减小网络规模C4/C5
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)
    
    reduce_divider = 2 if reduced_tail else 1
    
    # bneck配置
    inverted_residual_setting = [
        # input_c, kernel, expand_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    
    last_channel = adjust_channels(1024 // reduce_divider)  # c5
    
    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_cls=num_cls)       