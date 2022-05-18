from torchvision.transforms import transforms
import sys
from PIL import Image
import torch
import json
from model import AlexNet

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

img_path = sys.argv[1]

# 读取图片并预处理
img = Image.open(img_path)
img = transform(img)
img = torch.unsqueeze(img, dim=0)

# 读取json类别文件
try:
    json_file = open('./class_indices.json', 'r')
    cls_dict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 创建模型，读取参数
model_weight_path = './AlexNet.pth'
model = AlexNet(len(cls_dict))
model.load_state_dict(torch.load(model_weight_path))
model.to(device)
model.eval()

# 推理
with torch.no_grad():
    output = torch.squeeze(model(img.to(device)))
    predict = torch.softmax(output, dim=0)
    predict_cls = torch.argmax(predict).cpu().item()

print(cls_dict[str(predict_cls)], predict[predict_cls].item())
