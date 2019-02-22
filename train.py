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

class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, np.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)#, padding=1) # 32
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.fire2 = fire(16, 16, 64)
        self.fire3 = fire(128, 16, 64)
        self.fire4 = fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire5 = fire(256, 32, 128)
        self.fire6 = fire(256, 48, 192)
        self.fire7 = fire(384, 48, 192)
        self.fire8 = fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire9 = fire(512, 64, 256)
        self.fire10 = fire(512, 80, 320)
        self.fire11 = fire(640, 80, 320)
        self.fire12 = fire(640, 96, 384)
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(768, 4, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=17)
    #    self.FC1 = nn.Linear(289, 100)
    #    self.FC2 = nn.Linear(100,1)
        #self.FC2 = nn.Linear(100, 1)
        #self.softmax = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        #print(x.shape)
        x = self.fire2(x)
        #print(x.shape)
        x = self.fire3(x)
        #print(x.shape)
        x = self.fire4(x)
        #print(x.shape)

        x = self.maxpool2(x)
        #print(x.shape)

        x = self.fire5(x)
        #print(x.shape)

        x = self.fire6(x)
        #print(x.shape)

        x = self.fire7(x)
        #print(x.shape)

        x = self.fire8(x)
        #print(x.shape)

        x = self.maxpool3(x)
        #print(x.shape)

        x = self.fire9(x)
        #print(x.shape)
        x = self.fire10(x)
        x= self.fire11(x)
        x = self.fire12(x)
        x = self.maxpool4(x)
        x = self.conv2(x)
        #print(x.shape)

        #x = self.avg_pool(x)
        #print(x.shape)

        #x = x.view((batch_size, num_classes, -1))
        #print(x.shape)

        x = self.avg_pool(x)
        # x = self.relu(x)
        #
        # x = self.FC2(x)
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
batch_size = 2
num_classes = 4
net  = SqueezeNet()
#net.load_state_dict(torch.load(r'.\saves\model_FC_epoch_ 30.pth'))


print("build net:", net)

for epoch in range(50):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    #i = 0
    #while i < (len(text_file)-2*batch_size):
    for i in range(int(len(text_file)/batch_size)):
        images = torch.zeros((batch_size,3, 576,576),dtype = torch.float32)
        targets = torch.zeros(batch_size, 4)
        for s in range(batch_size):
            m = np.random.randint(0,len(text_file)-4)
            if text_file[m][-3:]!='png':
                m+=1
            line = text_file[m]
            #if line[-3:] == 'png':
            name = line
            image = torch.from_numpy(cv2.resize(cv2.imread(name), (576, 576))).float()/255
            images[int(s/2)] = image.transpose(0,2).transpose(1,2)
            m+=1
            line = text_file[m]

            #else:
            target = np.fromstring(line, dtype = float, sep = ' ')
            target[0]/=720
            target[1]/=576
            target[2]/=720
            target[3]/=576
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
            print(scores)
            #targets = torch.LongTensor([[targets[0][0],targets[0][1],targets[0][2],targets[0][3]]])
            loss = nn.functional.mse_loss(scores, targets)
            #print(loss)
            # compute the accuracy
            # pred = scores.data.max(1)[1] # get the index of the max log-probability
            # correct += pred.eq(targets.data).cpu().sum()

            #avg_loss.append(loss.data[0])
            loss.backward()
            optimizer.step()
            t2 = time.time()
            print('time:', t2-t1, 'loss:', float(loss))
        #writer.add_scalar('Loss', loss.item(), i)
        if i %20 == 0:
            image = cv2.imread(name)
            cv2.rectangle(image, (int(scores[-1][0]*720),int(scores[-1][1]*576)),(int(scores[-1][2]*720),int(scores[-1][3]*576)),(0,255,0),3)
            #cv2.rectangle(image, (int(scores[0][0]*720/576),int(scores[0][1])),(int(scores[0][2]*720/576),int(scores[0][3])),(0,255,0),3)

            cv2.rectangle(image, (int(target[0]*720),int(target[1]*576)), (int(target[2]*720),int(target[3]*576)), (255,0,0), 3)
            #cv2.rectangle(image, (int(target[0]*720/576),int(target[1])), (int(target[2]*720/576),int(target[3])), (255,0,0), 3)

            print("saving image"+r'.\logs\ '+str(epoch)+str(i/2)+'.png')
            cv2.imwrite(r'.\logs\ '+str(epoch)+str(i/2)+'.png',image)
    torch.save(net.state_dict(), r'./saves/model_FC_epoch_ '+ str(epoch) + '.pth')
    print('model saved, epoch = ', epoch)
