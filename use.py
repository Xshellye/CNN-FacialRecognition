import numpy as np
import cv2
import sys
import torch
from torch.autograd import Variable

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
        
    #加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
    model = torch.load('net.pkl').to(device)  
              
    #框住人脸的矩形边框颜色       
    color = (0, 255, 0)
    
    #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)
    
    #人脸识别分类器本地存储路径
    cascade_path = "D:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml"    
    
    #循环检测识别人脸
    while True:
        ret, frame = cap.read()   #读取一帧视频
        
        if ret is True:
            
            #图像灰化，降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)                
 
        #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                
                #截取脸部图像提交给模型识别这是谁
                img = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
                img = torch.from_numpy(img)
                img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
                img = np.transpose(img, (0,3,1,2)) 
                img = img.to(device)
                faceID = model(img)
                ACC = str(faceID[0][0].item())
                cv2.putText(frame,"ACC:" + ACC[:6], 
            (x - 40, y - 40),                      #坐标
            cv2.FONT_HERSHEY_SIMPLEX,              #字体
            1,                                     #字号
            (255,0,255),                           #颜色
            2)                                     #字的线宽
                
                #如果是“我”
                if faceID[0][0] > faceID[0][1] and faceID[0][0] > 0.9:                                                        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    
                    #文字提示是谁
                    cv2.putText(frame,'It\'s me!', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
                else:
                    pass
                            
        cv2.imshow("me", frame)
        
        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break
 
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()