import torch
import os
from PIL import Image
from model import VGGNet
import argparse
from torchvision import transforms
import json

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 读取类别文件
    json_file = args.cls_file
    assert os.path.exists(json_file), "file '{}' not exit.".format(json_file)
    with open(json_file, 'r') as f:
        idx2cls = json.load(f)

    # 读取预测图片并预处理
    img_path = args.img_path
    assert os.path.exists(img_path), "file '{}' not exit.".format(img_path)
    img = Image.open(img_path)
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)

    # 读取模型
    model_name = args.model
    weights_path = './save_weights/{}.pth'.format(model_name)
    assert os.path.exists(weights_path), "file '{}' not exist.".format(weights_path)
    model = VGGNet(model_name=model_name, num_cls=len(idx2cls))
    model.load_state_dict(torch.load(weights_path))
    model.to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(img.to(device))
        outputs = torch.squeeze(outputs).cpu()
        predict = torch.softmax(outputs, dim=0)
        predict_prob, predict_cls = torch.max(predict, dim=0)
    
    print(idx2cls[str(predict_cls.item())], predict_prob.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='path of image to predict')
    parser.add_argument('--model', default='vgg19', help='model path')
    parser.add_argument('--cls_file', default='./cls_indices.json', help='json file of class')

    args = parser.parse_args()
    main(args)
