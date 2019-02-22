import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy
import argparse
import cmath
import torch
from torch.autograd import Variable
from math import pi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smooth', default=21, type=int, help='odd number, output graph smoothing')
    parser.add_argument('--name', default='R851_LT-Ge growth at Tt=150C.avi', type=str, help='video file path')
    args = parser.parse_args()
    print('.\RHEED\\'+ args.name)
    cap = cv2.VideoCapture('.\RHEED\\'+ args.name)
    brightness = []
    time = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(length, fps)
    t = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:

            N = np.mean(frame, axis = (0,1,2))
            brightness.append(N)
            time.append(float(t)/fps)

            t+=1
            if t %100 ==0:
                print(float(t)/fps)
        else:
            break
    #brightness = np.sin(time)

    yhat = savgol_filter(brightness, args.smooth, 3)#3 for polynom order
    y_f = np.asarray(scipy.fftpack.rfft(brightness))
    #yhat = brightness - yhat
    # w = torch.Tensor([20.])
    # yhat = Variable(torch.from_numpy(np.asarray(yhat, dtype = np.float32)), requires_grad=True)
    # w = Variable(w, requires_grad=True)
    # a = Variable(torch.Tensor([1.]), requires_grad=True)
    # time=Variable(torch.from_numpy(np.asarray(time, dtype = np.float32)), requires_grad=True)
    # learning_rate = 0.004
    # for epoch in range(1000):
    #     y = a*torch.sin(2*pi*w*t)
    #     loss = torch.mean((yhat-y)**2)
    #     loss.backward()
    #     a.data-=learning_rate*a.grad.data
    #     w.data-=learning_rate*w.grad.data
    #     a.grad.data.zero_()
    #     w.grad.data.zero_()
    #     print(loss.item())
    # print(w.item())
    # plt.plot(time.detach().numpy(), yhat.detach().numpy())
    # plt.plot(time.detach().numpy(), a.item()*np.sin(w.item()*time.detach().numpy()))
    # plt.grid()
    # plt.xlabel('Time, sec')
    # plt.ylabel('Mean pixel smoothed')
    # plt.savefig('.\plots\\'+ args.name + 'smoothed'+ str(args.smooth)+ '.jpg')
    # plt.show()


    # time = np.asarray(time)
    # plt.plot(np.flip(1/time[1:len(time)], 0), y_f[1:len(time)])
    # plt.grid()
    # plt.xlabel('Time, sec')
    # plt.ylabel('Fourier mean pixel')
    # plt.savefig('.\plots\\'+ args.name + '_fourier'+ str(args.smooth)+ '.jpg')
    # plt.show()
    # y_f = y_f[1:len(y_f)]
    # time = time[1:len(time)]
    # brightness = brightness[1:len(brightness)]
    # for i in range(1):
    #
    #     n = np.argmax(y_f)#[int(len(y_f)/20):len(y_f)])
    #     print(y_f[n], n, np.max(y_f))
    #
    #     y_f[n]=0
    #
    # k=0
    # yff = scipy.fftpack.irfft(y_f)
    # brightness = np.asarray(brightness)
    # brightness -= yff
    # for i in range(1, len(brightness)-1):
    #     if brightness[i-1] < brightness[i] and brightness[i]>brightness[i+1]:
    #         k+=1
    # print('number of oscillations:', k)
    #
    # plt.plot(time, brightness)
    # plt.grid()
    # plt.xlabel('Time, sec')
    # plt.ylabel('Mean pixel fourier reverse')
    # plt.savefig('.\plots\ '+ args.name + 'fourier_reverse'+ str(args.smooth)+ '.jpg')
    # plt.show()
    plt.plot(time, brightness)
    plt.grid()
    plt.xlabel('Time, sec')
    plt.ylabel('Mean pixel')
    plt.show()
    # plt.savefig('.\plots\ '+ args.name+'.jpg')
    cap.release()
    cv2.destroyAllWindows()
