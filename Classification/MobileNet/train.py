from model_v2 import MobileNetV2
from model_v3 import mobilenet_v3_large, mobilenet_v3_small
from torchvision import datasets, transforms
import json
import torch
import argparse
import os
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm
import sys
import wandb

def create_model(model_name, num_cls=1000):
    if model_name == 'mobilenet_v2':
        return MobileNetV2(num_cls)
    elif model_name == 'mobilenet_v3_large':
        return mobilenet_v3_large(num_cls)
    elif model_name == 'mobilenet_v3_small':
        return mobilenet_v3_small(num_cls)

def main(args):
    suffix = ''
    if args.pretrain:
        suffix += '_pretrain'
        if args.freeze:
            suffix += '_freeze'
    save_model_name = args.model_name + suffix
    wandb.init(config=vars(args), project='MobileNet Classification', name=save_model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
    save_path = os.path.join(args.save_path, '{}.pth'.format(save_model_name))
    # 预处理方式
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])  
    }
    
    data_root = args.data_root
    train_set = datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=transform['train'])
    val_set = datasets.ImageFolder(root=os.path.join(data_root, 'val'), transform=transform['val'])
    train_set_size = len(train_set)
    val_set_size = len(val_set)
    print('Use {} images for training, {} for validation.'.format(train_set_size, val_set_size))
    
    # 获取类别，写入json文件
    cls2idx = train_set.class_to_idx
    num_cls = len(cls2idx)
    idx2cls = dict((v, k) for k, v in cls2idx.items())
    json_str = json.dumps(idx2cls, indent=4)
    with open('./class_indices.json', 'w') as json_file:
        json_file.write(json_str)
        
    # 定义DataLoader
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)
    
    # 定义模型
    model = create_model(args.model_name, num_cls)
    
    # 载入预训练权重
    if args.pretrain:
        model_weight_path = os.path.join('./pretrained_model', '{}.pth'.format(args.model_name))
        pre_weights = torch.load(model_weight_path)
        pre_dict = {k: v for k, v in pre_weights.items() if 'classifier' not in k}
        missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)
        
        # 冻结特征提取部分参数权重
        if args.freeze:
            for param in model.features.parameters():
                param.requires_grad = False
        
    model.to(device)
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    best_acc = 0.0
    
    # 训练
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for data in train_bar:
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = loss_func(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.desc = 'epoch [{}/{}], loss {:.3f}'.format(epoch + 1, args.epochs, loss.item())
        
        mean_loss = running_loss / len(train_loader)
        
        # 验证
        model.eval()
        acc = 0.0
        val_bar = tqdm(val_loader, file=sys.stdout)
        with torch.no_grad():
            for data in val_bar:
                images, labels = data
                outputs = model(images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, labels.to(device)).sum().item()
        
        acc = acc / val_set_size
        print('[epoch {}]: mean_loss {:.3f}, acc {:.3f}.'.format(epoch + 1, mean_loss, acc))
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            
        wandb.log({'loss': mean_loss, 'acc': acc})
    
    print('Finsh Training')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train config')
    parser.add_argument('--device', default='gpu', type=str, help='training device', choices=['gpu', 'cpu'])
    parser.add_argument('--pretrain', action='store_true', help='user pretrained model if set true')
    parser.add_argument('--freeze', action='store_true', help="freeze feature extraction part if set true")
    parser.add_argument('--data_root', default='/common/Dataset/flower_data', type=str, help='dataset path')
    parser.add_argument('--num_workers', default=8, type=int, help='num of worker use to load data')
    parser.add_argument('-save_path', default='./save_weights', type=str, help='path to save model')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='total train epochs')
    parser.add_argument('--model_name', default='mobilenet_v3_large', type=str, choices=['mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small'], help='model name')
    
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        
    main(args) 