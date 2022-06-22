import torch
# 定义网络结构
class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=96,
                            kernel_size=11,
                            stride=4),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),        
            torch.nn.MaxPool2d(3,2),
            torch.nn.Conv2d(96,256,5,padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2),
            torch.nn.Conv2d(256,384,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384,384,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384,256,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256*6*6,4096),
            torch.nn.ReLU(),     
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2),
            torch.nn.Softmax(dim=1)         #dim=1是按行softmax——降到（0,1）区间内相当于概率
        )
    def forward(self, x):
        x = self.conv(x)
        #print(x.size())
        x = x.contiguous().view(-1,256*6*6)  #使用.contiguous()防止用多卡训练的时候tensor不连续，即tensor分布在不同的内存或显存中
        x = self.fc(x) 
        return x