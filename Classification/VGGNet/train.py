import torch
from torchvision import datasets
import os
from torchvision import transforms
import json
import argparse
from model import VGGNet
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import sys
import wandb

def main(args):
    
    wandb.init(config=vars(args), project='VGGNet Classification',name='vgg16')
    
    # 训练集和验证集预处理
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }


    data_root = '/common/Dataset/flower_data'
    train_set = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=transform['train'])
    train_set_size = len(train_set)
    val_set = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=transform['val'])
    val_set_size = len(val_set)

    # 将索引-类别映射写入json文件
    cls2idx = train_set.class_to_idx
    idx2cls = dict((v, k) for k, v in cls2idx.items())
    json_str = json.dumps(idx2cls, indent=4)
    with open('cls_indices.json', 'w') as json_file:
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
    
    print('using {} images for training, {} images for validation.'.format(train_set_size, val_set_size))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 创建模型
    model = VGGNet(model_name=args.model, num_cls=len(cls2idx), init_weights=True)
    model.to(device)
    # 分类损失
    loss_func = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()       # 训练模式，dropout生效
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        
        # 训练
        for train_data in train_bar:
            train_images, train_labels = train_data
            # 梯度清零
            optimizer.zero_grad()
            # 正向传播
            outputs = model(train_images.to(device))
            # 计算损失
            loss = loss_func(outputs, train_labels.to(device))
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.desc = 'train epoch [{}/{}], loss: {}'.format(epoch + 1, args.epochs, loss)
        
        mean_loss = running_loss / len(train_loader)
        
        # 验证
        model.eval()        # 验证模式，dropout无效
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        
        acc = acc / val_set_size
        print('[epoch %d] train_loss: %.3f, acc: %.3f' % (epoch + 1, mean_loss, acc))
        
        # 保存准确率最高的模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.save_path, '{}.pth'.format(args.model)))
        
        wandb.log({'loss': mean_loss, 'acc': acc})
    
    print('Finish Training!')        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config of train')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='num of workers to load data')
    parser.add_argument('--epochs', default=10, type=int, help='total training epochs')
    parser.add_argument('--model', default='vgg16', type=str, help='model name')
    parser.add_argument('--save_path', default='./save_weights', type=str, help='where to save model')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    main(args)
    

