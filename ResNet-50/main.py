import os
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from resnet import resnet50
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, classification_report



def main():

    image_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  # 随机裁剪到256*256
            transforms.RandomRotation(degrees=15),  # 随机旋转
            transforms.RandomHorizontalFlip(p=0.5),  # 依概率水平旋转
            transforms.CenterCrop(size=224),  # 中心裁剪到224*224符合输入要求
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
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),  # 图像变换至256
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),  # 填充
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # 加载数据集
    dataset = 'D:/EyePACS/1000data/'
    train_directory = os.path.join(dataset, 'train')  # 训练集的路径
    valid_directory = os.path.join(dataset, 'valid')  # 验证集的路径
    test_directory = os.path.join(dataset, 'test')  # 测试集路径
    batch_size = 64  # 分成64组

    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        # imagefolder（root, transform），root是图片路径，transform是对图片的变换，生成的数据类型是dataset
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
        'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
    }  # 把dataset类型的数据放在数组里，便于通过键值调用

    train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
    # DataLoader(dataset, batch_size, shuffle) dataset数据类型；分组数；是否打乱
    valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
    test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)

    x, label = iter(train_data).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    model = resnet50().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    # 初始学习率设为0.1，每200个Epoch后使学习率缩小为原来的0.1倍

    # 开始训练
    istrain = 1
    if istrain:
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(700):
            # 轮数为700轮
            model.train()
            for batchidx, (x, label) in enumerate(train_data):

                x, label = x.to(device), label.to(device)

                logits = model(x) # 前向传播
                loss = criteon(logits, label) # 计算损失值loss

                optimizer.zero_grad() # 梯度清零
                loss.backward() # 反向传播
                optimizer.step() # 更新

            print('epoch:',epoch+1,',','loss:',loss.item())

            # 开始valid验证
            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_num = 0
                for x, label in valid_data:

                    x, label = x.to(device), label.to(device)

                    logits = model(x) # 前向传播

                    pred = logits.argmax(dim=1) # pred为预测结果

                    correct = torch.eq(pred, label).float().sum().item()
                    total_correct += correct
                    total_num += x.size(0)

                acc = total_correct / total_num
                # 计算准确率

                if best_acc < acc:
                    best_acc = acc
                    best_epoch = epoch + 1

                print('epoch:',epoch+1,',','valid acc:',acc,',','best epoch:',best_epoch)

            scheduler.step()

    # 开始测试
    istest = 1
    if istest:
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            all_pred = []
            all_label = []
            for x, label in test_data:

                x, label = x.to(device), label.to(device)

                logits = model(x) # 前向传播

                pred = logits.argmax(dim=1) # 预测结果

                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)

                pred = torch.as_tensor(pred, device='cpu')
                label = torch.as_tensor(label, device='cpu')
                n_pred = pred.numpy()
                n_label = label.numpy()
                all_pred.extend(n_pred)
                all_label.extend(n_label)
                # 收集pred与label，放入数组中

            acc = total_correct / total_num # 准确率
            precision = precision_score(all_label, all_pred, average='micro') # 精确率
            recall = recall_score(all_label, all_pred, average='micro') # 召回值
            f1 = f1_score(all_label, all_pred, average='micro') # f1值
            cm = confusion_matrix(all_label, all_pred) # 混淆矩阵
            target_names = ['0','1','2','3','4']
            cr = classification_report(all_label, all_pred, target_names=target_names)

            print('test acc:',acc)
            print('micro precision:',precision)
            print('micro recall:',recall)
            print('micro f1:',f1)
            print('confusion matrix:')
            print(cm)
            print(cr)

            pdf_result_arr = pd.DataFrame(acc, columns=["result"], index=["acc"])
            path = "D:/PyCharmPythonProject/ResNet50/test5/test/"
            filename = "result.csv"
            pdf_result_arr.to_csv(path + filename)


if __name__ == '__main__':
    main()