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
from squeezenet import SqueezeNet


if __name__ == '__main__':
    text_file = open(r'./annotations_full.txt', "r").read().splitlines()
    #text = text_file.readlines()
    batch_size = 1
    im_size = 150
    num_classes = im_size**2
    thresh = 0.5
    net  = SqueezeNet(num_classes = num_classes)
    net.load_state_dict(torch.load(r'.\saves\model_class_epoch_ 0.pth'))

    losses = np.zeros(20)
    print("build net:", net)
    t1 = time.time()
    print_step = 10
    for epoch in range(50):
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        for i in range(int(len(text_file)/batch_size)):
            images = torch.zeros((batch_size,1, im_size,im_size),dtype = torch.float32)
            targets = torch.zeros(batch_size, num_classes)
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
                target*=im_size
                for k in range(int(target[0]),int(target[2])):
                    for p in range(int(target[1]),int(target[3])):
                        targets[int(s/2)][im_size*k+p] = 1
                #targets[int(s/2)]=torch.from_numpy(target)
            net.train()

            images, targets = Variable(images), Variable(targets)
            # train the network
            for s in range(1):#25-epoch*5
                optimizer.zero_grad()
                scores = net.forward(images)
                scores = scores.view(batch_size, num_classes)
                #targets = torch.LongTensor([[targets[0][0],targets[0][1],targets[0][2],targets[0][3]]])
                loss = nn.functional.mse_loss(scores, targets)
                losses[i % print_step]=loss

                # compute the accuracy
                # pred = scores.data.max(1)[1] # get the index of the max log-probability
                # correct += pred.eq(targets.data).cpu().sum()

                #avg_loss.append(loss.data[0])
                loss.backward()
                optimizer.step()
            #writer.add_scalar('Loss', loss.item(), i)
            if i % print_step == 0:
                t2 = time.time()
                print('time:', t2-t1, 'loss:', np.mean(losses))
                t1 = t2
                image = cv2.imread(name)
                target/=im_size
                #im = torch.reshape(targets[0],(im_size, im_size))
                x1, y1,x2,y2 = 0,0,0,0
                x = []
                y = []
                for p in range(num_classes):
                    if scores[-1][p] > thresh:
                        y.append(int(p/im_size))
                        x.append(p%im_size)
                if x == []:
                    continue
                x1 = min(x)
                x2 = max(x)
                y1 = min(y)
                y2 = max(y)
                print(x1, y1,x2,y2)
                cv2.rectangle(image, (int(x1*720/im_size), int(y1*576/im_size)),(int(x2*720/im_size),int(y2*720/im_size)),3)
                cv2.rectangle(image, (int(target[0]*720),int(target[1]*576)), (int(target[2]*720),int(target[3]*576)),  3)
                print("saving image"+r'.\logs\ '+str(epoch)+str(i/2)+'.png')

                cv2.imwrite(r'.\logs\ '+str(epoch)+str(i/2)+'.png',image)
        torch.save(net.state_dict(), r'./saves/model_class_epoch_ '+ str(epoch) + '.pth')
        print('model saved, epoch = ', epoch)
