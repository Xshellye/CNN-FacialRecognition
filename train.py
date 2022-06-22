import torch
import numpy as np
from torch.autograd import Variable
from prepare import load_dataset, MYPATH
from alexnet import AlexNet
from torch import nn, optim

#定义学习率
learning_rate = 0.001
#是否使用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
model = AlexNet().to(device)

#定义交叉熵损失函数与SGD随机梯度下降
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#数据载入
train_data, test_data , valid_data = load_dataset(MYPATH)
epoch = 0

for data in train_data:
    img, label = data
    
    img = torch.LongTensor(img)
    #升维,因为pytorch规定输入卷积层的张量至少为4纬,故在此加一个batch的维度
    img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)    
    #改变张量维度的顺序，pytorch规定卷积层的张量为[batch_size,channel,image_height,image_width],即此处要求2000*3*64*64，而我们原来为2000*64*64*3。
    img= np.transpose(img, (0,3,1,2)) 
    
    #label不能直接转换为LongTensor否则会报错，原因未知-_-
    label = torch.tensor(label)
    label = label.long()    
    #转换为向量，否则无法进行比较
    label = torch.flatten(label) 
    label = Variable(label)
    
    img = img.to(device)
    label = label.to(device)
    
    out = model(img)
    
    loss = criterion(out, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch+=1
    if epoch%50 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
        
model.eval()
eval_loss = 0
eval_acc = 0
for data in valid_data:
    img, label = data
    img = torch.LongTensor(img)
    img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    img= np.transpose(img, (0,3,1,2)) 
    
    label = torch.tensor(label)
    label = label.long()
    label =  torch.flatten(label)
    label = Variable(label)
    
    img = img.to(device)
    label = label.to(device)
    
    out = model(img)
    
    loss = criterion(out, label)
    eval_loss += loss.data.item()*label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
    
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(test_data)),
    eval_acc / (len(test_data))
))
#保存训练好的模型
torch.save(model, 'net.pkl')
