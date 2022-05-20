from pickletools import optimize
import torch
from torchvision import transforms
import wandb
import model
import argparse
import os
from torchvision import datasets
import json
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import sys

def create_model(model_name, num_cls=1000):
    assert model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], 'Undefined model {}'.format(model_name)
    return model.__dict__[model_name](num_cls, True)

def main(args):
    wandb.init(config=vars(args), 
               project='ResNet Classification',
               name=args.model_name)
    
    # 迁移学习，使用了预训练模型，预处理一致
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
    train_set = datasets.ImageFolder(os.path.join(data_root, 'train'), transform['train'])
    train_set_size = len(train_set)
    val_set = datasets.ImageFolder(os.path.join(data_root, 'val'), transform['val'])
    val_set_size = len(val_set)
    print('Using {} images for training, {} images for validation'.format(train_set_size, val_set_size))
    
    # 写入类别文件
    cls2idx = train_set.class_to_idx
    idx2cls = dict((v, k) for k, v in cls2idx.items())
    num_cls = len(idx2cls)
    json_str = json.dumps(idx2cls, indent=4)
    with open('cls_indices.json', 'w') as json_file:
        json_file.write(json_str)
        
    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers)
    
    # 创建模型
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
    model = create_model(args.model_name)
    pretrained_model_path = './pretrained_model/{}.pth'.format(args.model_name)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(pretrained_model_path), strict=False)
    save_path = os.path.join(args.save_path, '{}.pth'.format(args.model_name))

    # 修改全连接层
    inchannels = model.fc.in_features
    model.fc = nn.Linear(inchannels, num_cls)
    model.to(device)
    
    # 定义回归损失
    loss_func = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    best_acc = 0.0
    
    # 训练
    for epoch in range(args.epochs):
        model.train()   # 训练时使用BN
        train_bar = tqdm(train_loader, file=sys.stdout)
        running_loss = 0.0
        
        for train_data in train_bar:
            train_images, train_labels = train_data
            optimizer.zero_grad()
            outputs = model(train_images.to(device))
            loss = loss_func(outputs, train_labels.to(device))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            train_bar.desc = 'train epoch [{}/{}], loss {:.3f}'.format(epoch + 1, args.epochs, loss)
        
        mean_loss = running_loss / len(train_loader)
        
        # 验证
        model.eval()    # 不使用BN
        acc = 0.0
        val_bar = tqdm(val_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            with torch.no_grad():
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        
        acc = acc / val_set_size
        
        print('[epoch %d], train_loss %.3f, acc %.3f' % (epoch + 1, mean_loss, acc))
        wandb.log({'loss': mean_loss, 'acc': acc})
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
    
    print('Finish Training')
        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train configs')
    parser.add_argument('--device', default='gpu', type=str, help='use gpu or cpu')
    parser.add_argument('--epochs', default=50, type=int, help='training epochs')
    parser.add_argument('--save_path', default='./save_weights', type=str, help='path to save model')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--model_name', default='resnet50', type=str, help='model name')
    parser.add_argument('--data_root', default='/common/Dataset/flower_data', type=str, help='dataset path')
    parser.add_argument('--num_workers', default=8, type=int, help='workers use to load data')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        
    main(args)