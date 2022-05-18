import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from model import LeNet
import torch.optim as optim

data_root = '/common/Dataset/CIFAR10'
batch_size = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 准备训练数据集(50000张)
train_set = torchvision.datasets.CIFAR10(root=data_root, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

# 准备测试数据集(10000张)
val_set = torchvision.datasets.CIFAR10(root=data_root, train=False, transform=transform, download=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

# 定义类别
classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
)

def main():
    # 创建模型
    model = LeNet().to(device)

    # 定义分类损失函数
    loss_func = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    for epoch in range(5):
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # 获取标签和图像
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 正向传播
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 累加损失
            epoch_loss += loss.item()
            running_loss += loss.item()
        
            if step % 500 == 499:   # 每500次迭代打印和测试
                acc = 0.0
                with torch.no_grad():   # 不计算梯度
                    for data in val_loader:
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        predict_y = torch.max(outputs, dim=1)[1]
                        acc += torch.eq(predict_y, labels).sum().item()
                    acc = acc / 10000.0
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                            (epoch + 1, step + 1, running_loss / 500, acc))  
                    running_loss = 0.0
            
    print('Finished Training')
    
    save_path = './LeNet.pth'
    torch.save(model.state_dict(), save_path)
        
if __name__ == '__main__':
    main()