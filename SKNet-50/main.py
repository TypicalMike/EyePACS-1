import os
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sknet import SKNet50
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, classification_report



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
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),  # 图像变换至256
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),  # 填充
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # 2加载数据集
    dataset = 'D:/PyCharmPythonProject/SeNet50/test1/datatest/data/'
    train_directory = os.path.join(dataset, 'train')  # 训练集的路径，os.path.join()函数是路径拼接函数
    valid_directory = os.path.join(dataset, 'valid')  # 验证集的路径
    test_directory = os.path.join(dataset, 'test')  # 测试集路径
    batch_size = 32  # 分成32组

    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        # imagefolder（root, transform），前者是图片路径，后者是对图片的变换，生成的数据类型是dataset
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
        'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
    }  # 把dataset类型的数据放在数组里，便于通过键值调用

    train_data_size = len(data['train'])  # 训练集的大小
    valid_data_size = len(data['valid'])  # 验证集的大小
    test_data_size = len(data['test'])  # 验证集的大小

    train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
    # DataLoader(dataset, batch_size, shuffle) dataset数据类型；分组数；是否打乱
    valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
    test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)

    x, label = iter(train_data).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    # model = Lenet5().to(device)
    model = SKNet50().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    # print(model)

    istrain = 0
    if istrain:
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(700):
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

            print('epoch:',epoch+1,',','loss:',loss.item())

            model.eval()
            with torch.no_grad():
                # test
                total_correct = 0
                total_num = 0
                for x, label in valid_data:
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

                if best_acc < acc:
                    best_acc = acc
                    best_epoch = epoch + 1

                print('epoch:',epoch+1,',','valid acc:',acc,',','best epoch:',best_epoch)
                torch.save(model.state_dict(), 'D:/PyCharmPythonProject/SkNet50/test1/model/' + 'dataset' + '_model_' + str(epoch + 1) + '.pt')
            scheduler.step()

    istest = 1
    if istest:
        model.load_state_dict(torch.load('D:/PyCharmPythonProject/SkNet50/test1/model/' + 'dataset' + '_model_' + '1' + '.pt'))
        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            all_pred = []
            all_label = []
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

                pred = torch.as_tensor(pred, device='cpu')
                label = torch.as_tensor(label, device='cpu')
                n_pred = pred.numpy()
                n_label = label.numpy()
                all_pred.extend(n_pred)
                all_label.extend(n_label)

            acc = total_correct / total_num
            precision = precision_score(all_label, all_pred, average='micro')
            recall = recall_score(all_label, all_pred, average='micro')
            f1 = f1_score(all_label, all_pred, average='micro')
            cm = confusion_matrix(all_label, all_pred)
            target_names = ['0', '1', '2', '3', '4']
            cr = classification_report(all_label, all_pred, target_names=target_names)

            print('test acc:', acc)
            print('test acc:', acc)
            print('micro precision:', precision)
            print('micro recall:', recall)
            print('micro f1:', f1)
            print('confusion matrix:')
            print(cm)
            print(cr)

            pdf_result_arr = pd.DataFrame(acc, columns=["result"], index=["acc"])
            path = "D:/PyCharmPythonProject/SkNet50/test1/test/"
            filename = "result.csv"
            pdf_result_arr.to_csv(path + filename)


if __name__ == '__main__':
    main()