import os
import shutil
import pandas as pd
import random

# 打开表格文件并读取
f = open("D:/EyePACS/DR/trainLabels.csv", "rb")  # 打开csv文件
list = pd.read_csv(f)  # 这句不能少
# print(list)
# 创建文件夹
for i in range(5):  # "6"指的是0-5总共6个类别
    if not os.path.exists('D:/EyePACS/alltrainfenlei/' + str(i)):  # 最后一个 / 不要漏
        os.mkdir('D:/EyePACS/alltrainfenlei/' + str(i))
# 进行分类
for i in range(5):

    listnew = list[list["level"] == i]  # 对应csv文件标签 那一栏的标题

    l = listnew["image"].tolist()  # 对应csv文件图片那一栏的标题

    j = str(i)
    for each in l:
        shutil.move('D:/EyePACS/AllTrain/' + each + '.jpeg', 'D:/EyePACS/alltrainfenlei/' + j)
print("完成")