import os
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from resnet50 import ResNet50

def main():

    image_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  # 随机裁剪到256*256
            transforms.RandomRotation(degrees=15),  # 随机旋转
            transforms.RandomHorizontalFlip(p=0.5),  # 依概率水平旋转
            transforms.CenterCrop(size=224),  # 中心裁剪到224*224符合resnet的输入要求
            transforms.ToTensor(),  # 填充
            transforms.Normalize([0.485, 0.456, 0.406],  # 转化为tensor，并归一化至[0，-1]
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),  # 图像变换至256
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),  # 填充
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # 2加载数据集
    dataset = 'D:/PyCharmPythonProject/ResNet50/test1/datatest/data'
    train_directory = os.path.join(dataset, 'train')  # 训练集的路径，os.path.join()函数是路径拼接函数
    valid_directory = os.path.join(dataset, 'valid')  # 验证集的路径
    test_directory = os.path.join(dataset, 'test')  # 测试集路径
    batch_size = 32  # 分成32组

    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        # imagefolder（root, transform），前者是图片路径，后者是对图片的变换，生成的数据类型是dataset
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
        'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['valid'])
    }  # 把dataset类型的数据放在数组里，便于通过键值调用

    train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
    # DataLoader(dataset, batch_size, shuffle) dataset数据类型；分组数；是否打乱
    valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
    test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)

    x, label = iter(train_data).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    # model = Lenet5().to(device)
    model = ResNet50().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # print(model)

    for epoch in range(1000):

        model.train()
        for batchidx, (x, label) in enumerate(train_data):
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in test_data:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)


if __name__ == '__main__':
    main()