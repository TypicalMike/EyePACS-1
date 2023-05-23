import os
import cv2
import retinex

data_path = 'D:/PyCharmPythonProject/MNIST/MSRCR2/2/'
img_list = os.listdir(data_path)

if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

for img_name in img_list:
    if img_name == '.gitkeep':
        continue

    img = cv2.imread(os.path.join(data_path, img_name))
    print(img_name)

    img_msrcr = retinex.MSRCR(img)  #对图像使用MSRCR增强算法

    graymsrcrImage = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2GRAY)  # 将MSRCR的图像灰度化

    clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=(8, 8))  # 对图像使用CLAHE图像增强算法，得到imgEquA
    imgEquA = clahe.apply(graymsrcrImage)

    savepath = "D:/PyCharmPythonProject/MNIST/MSRCR2/3/" + img_name
    cv2.imwrite(savepath, imgEquA)
    cv2.waitKey()