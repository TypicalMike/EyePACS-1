import os
import torchvision.transforms as transforms
from PIL import Image

data_path = 'D:/EyePACS/alltrainfenlei/4/'
img_list = os.listdir(data_path)

if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

for img_name in img_list:
    if img_name == '.gitkeep':
        continue

    img = Image.open(os.path.join(data_path, img_name)) # 读取图像
    print(img_name)

    caijian = transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)) # 随机裁剪
    img_1 = caijian(img)

    fanzhuan = transforms.RandomHorizontalFlip(p=0.5) # 随机翻转
    img_2 = fanzhuan(img)

    xuanzhuan = transforms.RandomRotation((30,50)) # 随机旋转30到50度
    img_3 = xuanzhuan(img_2)

    savepath = "D:/EyePACS/alltrainfenlei3/4/" + "nine_" + img_name # 保存图像
    img_3.save(savepath)