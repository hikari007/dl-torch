import argparse
from torchvision import transforms
import os
import json
from PIL import Image
import torch
from model_v2 import MobileNetV2
from model_v3 import mobilenet_v3_large, mobilenet_v3_small

def create_model(model_name, num_cls=1000):
    if model_name == 'mobilenet_v2':
        return MobileNetV2(num_cls)
    elif model_name == 'mobilenet_v3_large':
        return mobilenet_v3_large(num_cls)
    elif model_name == 'mobilenet_v3_small':
        return mobilenet_v3_small(num_cls)
    else:
        raise ValueError('Undefined model {}.'.format(model_name))

def main(args):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 读取类别文件
    json_file = args.cls_file
    assert os.path.exists(json_file), "file '{}' not exist.".format(json_file)
    with open(json_file, 'r') as f:
        idx2cls = json.load(f)
    num_cls = len(idx2cls)
    
    # 读取图片并预处理
    img_path = args.img_path
    assert os.path.exists(img_path), "file '{}' not exits.".format(img_path)
    img = Image.open(img_path)
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    
    # 读取模型
    weight_path = args.model_path
    assert os.path.exists(weight_path), "file '{}' not exits.".format(weight_path)
    model = create_model(args.model_name, num_cls).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    # 推理
    model.eval()    # 推理不用BN，否则结果特别差
    with torch.no_grad():
        outputs = model(img.to(device))
        outputs = torch.squeeze(outputs).cpu()
        predict = torch.softmax(outputs, dim=0)
        predict_prob, predict_cls = torch.max(predict, dim=0)
    
    print(idx2cls[str(predict_cls.item())], predict_prob.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict config')
    parser.add_argument('img_path', type=str, help='path of image to predict')
    parser.add_argument('--model_name', default='mobilenet_v3_small', choices=['mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small'], help='create model name')
    parser.add_argument('--model_path', default='./save_weights/mobilenet_v3_small_pretrain_freeze.pth', type=str, help='saved model path')
    parser.add_argument('--cls_file', default='./class_indices.json', type=str, help='path of json file which record idx-cls mapping')
    args = parser.parse_args()
    
    main(args)