import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import cv2
import os
import time

class Net(nn.Module):
    def __init__(self, im_size):
        self.im_size = im_size
        self.kernels  = [150, 125, 75]
        self.strides = [4,4,6]
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=self.kernels[0], stride=self.strides[0], padding = 0)#, padding=1) # 32
        # self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1,1,  kernel_size=self.kernels[1], stride=self.strides[0])
        self.maxpool2 = nn.MaxPool2d(kernel_size = 8, stride = 1)
        self.conv3 = nn.Conv2d(1,1,  kernel_size=self.kernels[2], stride=self.strides[0], padding = 0)
        self.maxpool3 = nn.MaxPool2d(kernel_size= 2, stride = 1)
        #self.conv4 = nn.Conv2d(1,1,  kernel_size=50, stride=7, padding = 4)
        #self.conv5 = nn.Conv2d(1,1,  kernel_size=25, stride=8,padding = 8)
        #self.conv6 = nn.Conv2d(1,1,  kernel_size=10, stride=8)
        # self.conv7 = nn.Conv2d(1,1, kernel_size = 8, stride = 4)
        # self.conv8 = nn.Conv2d(1,1,kernel_size = 5, stride = 3)

        #self.FC1 = nn.Linear(3,4, bias = True)
        #self.FC2 = nn.Linear(200, 4)
        #self.softmax = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                # m.weight.data.normal_(0, np.sqrt(2. / n))
                m.weight.data.fill_(1)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x1 = self.conv1(x)
        #print(self.conv1.weight)
        x1 = self.maxpool3(x1)
        y1 = torch.argmax(x1).float()
        x1_sum = torch.sum(x1).detach().numpy()
        #print(x1.shape)
        x2 = self.conv2(x)
        x2 = self.maxpool2(x2)
        y2 = torch.argmax(x2).float()
        x2_sum = torch.sum(x2).detach().numpy()
        #print(x2.shape)
        x3 = self.conv3(x)
        x3 = self.maxpool3(x3)
        x3_sum = torch.sum(x3).detach().numpy()
        y3 = torch.argmax(x3).float()

        y = torch.tensor((y1,y2,y3))
        n = torch.argmax(y)
        out = torch.zeros(4)
        out[0] = (y[n]%im_size)*self.kernels[n]
        out[1] = (int(y[n]/im_size))*self.kernels[n]
        out[2] = (y[n]%im_size+1)*self.kernels[n]
        out[3] = (int(y[n]/im_size)+1)*self.kernels[n]

        return out
if __name__ == '__main__':
    text_file = open(r'./annotations_full.txt', "r").read().splitlines()
    #text = text_file.readlines()
    batch_size = 1
    im_size = 300

    net  = Net(im_size)
#    net.load_state_dict(torch.load(r'.\saves\model_class_epoch_ 0.pth'))
    print_step  =1
    losses = np.zeros(print_step)
    print("build net:", net)
    t1 = time.time()
    print_step = 10
    #log_path = r'.\logs\\'
    #writer = SummaryWriter(log_path)
    for epoch in range(50):
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        for i in range(int(len(text_file)/batch_size)):
            images = torch.zeros((batch_size,1, im_size,im_size),dtype = torch.float32)
            targets = torch.zeros(batch_size, 4)
            for s in range(batch_size):
                m = np.random.randint(0,len(text_file)-4)
                m=30
                if text_file[m][-3:]!='png':
                    m+=1
                line = text_file[m]
                name = line
                image = torch.from_numpy(cv2.resize(cv2.imread(name, 0), (im_size, im_size))).float()/255
                images[int(s/2)] = image#.transpose(0,2).transpose(1,2)
                m+=1
                line = text_file[m]
                target = np.fromstring(line, dtype = float, sep = ' ')
                target[0]/=720
                target[1]/=576
                target[2]/=720
                target[3]/=576
                targets[int(s/2)]=torch.from_numpy(target)
            net.train()

            images, targets = Variable(images), Variable(targets)
            # train the network
            optimizer.zero_grad()
            scores = net.forward(images)
            scores = scores.view(batch_size, 4)
            #print(scores, targets)
            loss = nn.functional.mse_loss(scores, targets)
            #writer.add_scalar('Loss', loss.item(), i)
            #losses[i % print_step]=loss
            #loss.backward()
            optimizer.step()
            if i % print_step == 0:
                t2 = time.time()
                print('time:', t2-t1, 'loss:', np.mean(losses))
                t1 = t2
                image = cv2.imread(name)
                cv2.rectangle(image, (int(scores[-1][0]*720), int(scores[-1][1]*576)),(int(scores[-1][2]*720),int(scores[-1][3]*720)),(255,0,0),3)
                cv2.rectangle(image, (int(target[0]*720),int(target[1]*576)), (int(target[2]*720),int(target[3]*576)),(0,255,0),  3)
                cv2.imshow('im',image)
                cv2.waitKey(1000)
                print("saving image"+r'.\logs\ '+str(epoch)+str(i/2)+'.png')

                cv2.imwrite(r'.\logs\ '+str(epoch)+str(i/2)+'.png',image)
        torch.save(net.state_dict(), r'./saves/model_class_epoch_ '+ str(epoch) + '.pth')
        print('model saved, epoch = ', epoch)
