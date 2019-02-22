import torch
import cv2
import numpy as np
import torchvision
import torch.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from tensorboardX import SummaryWriter
import time

# class fire(nn.Module):
#     def __init__(self, inplanes, squeeze_planes, expand_planes):
#         super(fire, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
#         self.bn1 = nn.BatchNorm2d(squeeze_planes)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
#         self.bn2 = nn.BatchNorm2d(expand_planes)
#         self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(expand_planes)
#         self.relu2 = nn.ReLU(inplace=True)
#
#         # using MSR initilization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
#                 m.weight.data.normal_(0, np.sqrt(2./n))
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         out1 = self.conv2(x)
#         out1 = self.bn2(out1)
#         out2 = self.conv3(x)
#         out2 = self.bn3(out2)
#         out = torch.cat([out1, out2], 1)
#         out = self.relu2(out)
#         return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=150, stride=4, padding = 0)#, padding=1) # 32
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(3,1,  kernel_size=125, stride=4)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 8, stride = 1)
        self.conv3 = nn.Conv2d(3,1,  kernel_size=75, stride=6, padding = 0)
        self.maxpool3 = nn.MaxPool2d(kernel_size= 2, stride = 1)
        self.conv4 = nn.Conv2d(3,1,  kernel_size=50, stride=7, padding = 4)
        self.conv5 = nn.Conv2d(3,1,  kernel_size=25, stride=8,padding = 8)
        self.conv6 = nn.Conv2d(3,1,  kernel_size=10, stride=8)
        self.conv7 = nn.Conv2d(3,1, kernel_size = 8, stride = 5)
        self.conv8 = nn.Conv2d(3,1,kernel_size = 5, stride = 3)

        self.FC1 = nn.Linear(10952,4, bias = True)
        #self.FC2 = nn.Linear(200, 4)
        #self.softmax = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool3(x1)
        #print(x1.shape)
        x2 = self.conv2(x)
        x2 = self.maxpool2(x2)
        #print(x2.shape)
        x3 = self.conv3(x)
        x3 = self.maxpool3(x3)
        #print(x3.shape)
        x4 = self.conv4(x)
        #x4 = self.maxpool4(x4)
        #print(x4.shape)
        x5 = self.conv5(x)
        #print(x5.shape)
        x6 = self.conv6(x)
        #print(x6.shape)
        x7 = self.conv7(x)
        #print(x7.shape)
        x8 = self.conv8(x)
        #print(x8.shape)
        x = torch.cat((x1,x2,x3,x4, x5, x6, x7, x8), 1)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view((batch_size, 1, -1))
        #print(x.shape)

        x = self.FC1(x)
        x = self.relu(x)
        #x = self.FC2(x)
        # print(x.shape)

        #x = self.softmax(x)

        return x

def fire_layer(inp, s, e):
    f = fire(inp, s, e)
    return f

def squeezenet(pretrained=False):
    net = SqueezeNet()
    # inp = Variable(torch.randn(64,3,32,32))
    # out = net.forward(inp)
    # print(out.size())
    return net

#log_path = r'.\logs\\'
#writer = SummaryWriter(log_path)

text_file = open(r'./annotations_full.txt', "r").read().splitlines()
#text = text_file.readlines()
batch_size = 1
num_classes = 4
net  = Net()
#net.load_state_dict(torch.load(r'.\saves\model_conv_FC_ 0.pth'))
epoches = 500
times = []
print("build net:", net)
losses = []
t2 = time.time()

for epoch in range(epoches):
    if epoch%20 == 0:
        optimizer = optim.SGD(net.parameters(), lr=0.05/(epoch+1)*2, momentum=0.9, weight_decay=5e-4)
    #i = 0
    #while i < (len(text_file)-2*batch_size):
    for i in range(int(len(text_file)/batch_size/6)):
        images = torch.zeros((batch_size,3, 300,300),dtype = torch.float32)
        targets = torch.zeros(batch_size, 4)
        for s in range(batch_size):
            m = np.random.randint(0,len(text_file)-4)
            if text_file[m][-3:]!='png':
                m+=1
            line = text_file[m]
            #if line[-3:] == 'png':
            name = line
            image = torch.from_numpy(cv2.resize(cv2.imread(name), (300, 300))).float()
            images[int(s/2)] = image.transpose(0,2).transpose(1,2)
            m+=1
            line = text_file[m]

            #else:
            target = np.fromstring(line, dtype = float, sep = ' ')
            target[0]=target[0]/720*300
            target[1]=target[1]/576*300
            target[2]=target[2]/720*300
            target[3]=target[3]/576*300
            targets[int(s/2)]=torch.from_numpy(target)
            #i+=1
        net.train()
        images, targets = Variable(images), Variable(targets)
        # train the network
        for s in range(1):#25-epoch*5
            optimizer.zero_grad()
            t1 = time.time()
            scores = net.forward(images)
            scores = scores.view(batch_size, num_classes)
            #targets = torch.LongTensor([[targets[0][0],targets[0][1],targets[0][2],targets[0][3]]])
            loss = nn.functional.smooth_l1_loss(scores, targets)
            losses.append(float(loss))
            times.append(int(len(text_file)/batch_size)*epoch + i)
            # compute the accuracy
            # pred = scores.data.max(1)[1] # get the index of the max log-probability
            # correct += pred.eq(targets.data).cpu().sum()

            #avg_loss.append(loss.data[0])
            loss.backward()
            optimizer.step()

            # print('time:', t2-t1, 'loss:', float(loss))
        #writer.add_scalar('Loss', loss.item(), i)
        if i %20 == 0:
            t1 = time.time()

            l = np.asarray(losses[-20:])
            print('time:', -t2+t1, 'loss:', np.mean(l))
            print(scores, targets)

            image = cv2.imread(name)
            cv2.rectangle(image, (int(scores[-1][0]*720/300),int(scores[-1][1]*576/300)),(int(scores[-1][2]*720/300),int(scores[-1][3]*576/300)),(0,255,0),3)
            #cv2.rectangle(image, (int(scores[0][0]*720/576),int(scores[0][1])),(int(scores[0][2]*720/576),int(scores[0][3])),(0,255,0),3)
            plt.plot(times, losses)
            plt.grid()
            fig_name = '.\logs_small\\'+str(i) + '_'+str(epoch)+'.png'
            plt.savefig('.\logs_small\\'+str(i) + '_'+str(epoch)+'.png')
            cv2.rectangle(image, (int(target[0]*720/300),int(target[1]*576/300)), (int(target[2]*720/300),int(target[3]*576/300)), (255,0,0), 3)
            #cv2.rectangle(image, (int(target[0]*720/576),int(target[1])), (int(target[2]*720/576),int(target[3])), (255,0,0), 3)
            if i > 10 or epoch != 0:
                t2 = t1

                if os.path.exists(fig_name_old):
                    os.remove(fig_name_old)
            print("saving image"+r'.\logs_small\ '+str(epoch)+str(i/2)+'.png')
            cv2.imwrite(r'.\logs_small\ '+str(epoch)+str(i/2)+'.png',image)
            fig_name_old = fig_name
    if epoch%10 == 0:
        torch.save(net.state_dict(), r'./saves/model_conv_FC_ '+ str(epoch) + '.pth')
        print('model saved, epoch = ', epoch)
