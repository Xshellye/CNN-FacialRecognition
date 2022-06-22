import os
import numpy as np
import cv2
import random
# from torch.utils.data import DataLoader 学习中...

TRAIN_RATE = 0.8
VALID_RATE = 0.1
SAMPLE_QUANTITY = 2000

IMAGE_SIZE = 227
MYPATH = ".\data"
 
#读取训练数据
images = []
labels = []
def read_path(path_name):    
    for dir_item in os.listdir(path_name):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))        
        if os.path.isdir(full_path):    #如果是文件夹，继续递归调用
            read_path(full_path)
        else:   #文件
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)                
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA) #修改图片的尺寸为64*64                
                images.append(image)
                labels.append(path_name)                                                   
    return images,labels
    

#从指定路径读取训练数据
def load_dataset(path_name):
    images,labels = read_path(path_name)

    #两个人共2000张图片，IMAGE_SIZE为64，故尺寸为2000 * 64 * 64 * 3
    #图片为64 * 64像素,一个像素3个通道(RGB)
    #标注数据，'pic'文件夹下都是我的脸部图像，全部指定为0，另外一个文件夹下是同学的，全部指定为1
    labels = np.array([0 if label.endswith('pic') else 1 for label in labels])        
    #简单交叉验证
    images = list(images)
    for i in range(len(images)):
        images[i] = [images[i],labels[i]]
    random.shuffle(images)
    train_data = []
    test_data = []
    valid_data = []
    #训练集
    for i in range(int(SAMPLE_QUANTITY * TRAIN_RATE)):
        train_data.append(images[i])       
    #验证集
    for i in range(int(SAMPLE_QUANTITY * TRAIN_RATE),int(SAMPLE_QUANTITY * (TRAIN_RATE + VALID_RATE))):
        valid_data.append(images[i])       
    #测试集
    for i in range(int(SAMPLE_QUANTITY * (TRAIN_RATE + VALID_RATE)), SAMPLE_QUANTITY):
        test_data.append(images[i])

    return train_data, test_data , valid_data

# if __name__ == "__main__":
#     train_loader = load_dataset(MYPATH)
     