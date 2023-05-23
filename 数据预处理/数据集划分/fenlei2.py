import splitfolders

# 将数据集按照8:1:1划分为训练集、验证集和测试集
splitfolders.ratio(input='D:/EyePACS/1000test/', output='D:/EyePACS/1000data/', seed=1337, ratio=(0.8, 0.1, 0.1))