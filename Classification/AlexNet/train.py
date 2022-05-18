import torch
from torchvision import transforms, utils, datasets
import json
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
from model import AlexNet

def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_transform = {
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

    # 定义训练集
    train_dataset = datasets.ImageFolder(root=data_root+'/train', transform=data_transform['train'])
    print('train dataset size: {:d}'.format(len(train_dataset)))

    # 获取类别 {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    flower_list = train_dataset.class_to_idx
    # {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    cls_dict = dict((v, k) for k, v in flower_list.items())
    # 写入json文件
    json_str = json.dumps(cls_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    epochs = 10
    batch_size = 32
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=8)

    val_dataset = datasets.ImageFolder(root=data_root+'/val', transform=data_transform['val'])
    print('validation dataset size: {:d}'.format(len(val_dataset)))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=8)
    
    # 创建模型
    model = AlexNet(num_cls=5)
    
    model.to(device)

    # 分类损失
    loss_func = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    
    save_path = './AlexNet.pth'
    best_acc = 0.0
    
    for epoch in range(epochs):
        # train
        model.train()   # 训练过程中使用dropout
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        
        for data in train_bar:
            imgs, labels = data
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(imgs.to(device))
            # 计算损失
            loss = loss_func(outputs, labels.to(device))
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            
            running_loss += loss.item()

            # 定义进度条前面文字信息
            train_bar.desc = 'train epoch[{}/{}] loss:{:.3f}'.format(epoch + 1, epochs, loss)
        
        # 训练完一轮，开始验证
        model.eval()    # 取消dropout
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_imgs, val_labels = val_data
                outputs = model(val_imgs.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum()
        
        val_acc = acc / len(val_dataset)
        print('[epoch %d] train_loss: %.3f val_acc: %.3f' % 
              (epoch + 1, running_loss / len(train_loader), val_acc))
        
        # 保存效果最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            
    print('Finish Training')
    
if __name__ == '__main__':
    main()



