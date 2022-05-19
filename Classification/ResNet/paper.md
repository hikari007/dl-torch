#### Batch Normalization

- 用于加速训练；推理时不进行归一化，直接使用训练时学习到的分布`model.eval()`
- BN层放在卷积层和激活层之间
- 使用BN时卷积层偏置参数无效无需设置`bias=False`

#### 迁移学习

- 直接训练整个网络
- 修改分类网络，只训练全连接层
- 在网络最后添加一个全连接层，只训练该层

#### 网络结构

对于ResNet18、ResNet34，conv3_1, conv4_1, conv5_1的第一个卷积层要做通道、宽高调整

对于ResNet50、ResNet101、ResNet152，conv2_1的第一个卷积层要做通道调整，conv3_1,conv4_1和conv5_1的第一个卷积层做通道、宽高调整

记每个Stage中的残差块内第一个1x1卷积核个数为channel
总结得到，如果当前Stage步长不为1(要做宽高调整)或者输入通道数self.in_channel != channel * expansion(要做通道调整)，则需要downsample
