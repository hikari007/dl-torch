from cProfile import label
import torch
from zmq import device
from model import LeNet
import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 创建模型
model = LeNet().to(dev)

# 读取权重
model.load_state_dict(torch.load('LeNet.pth'))

# 测试
img = Image.open('cat.jpg')
img = transform(img)   # (H, W, C) -> (C, H, W)
img = img.to(dev)
img = torch.unsqueeze(img, dim=0)      # 增加batch维度

with torch.no_grad():
    outputs = model(img)
    predict = torch.softmax(outputs, dim=1)
    pred_cls = torch.max(predict, dim=1)[1].data.cpu().numpy()
print(classes[int(pred_cls)], predict[0][pred_cls].item())
    




