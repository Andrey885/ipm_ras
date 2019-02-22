import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import argparse
import torch
import scipy

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
points = []
# mouse callback function
def draw(event,x,y,flags,param):
    global ix,iy,drawing,mode
    global x1, y1, x2,y2
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_prev = frame

        drawing = True
        ix,iy = x,y
        cv2.rectangle(frame,(ix,iy),(x,y),(0,255,0))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        #if mode == True:
        cv2.rectangle(frame,(ix,iy),(x,y),(0,255,0))
        x1 = max(x, ix)
        y1 = max(y, iy)
        x2 = min(x, ix)
        y2 = min(y, iy)
        points.append((x1,y1,x2,y2))
        print(x1, y1,x2, y2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smooth', default=11, type=int, help='odd number, output graph smoothing')
    parser.add_argument('--name', default='./special/R1272_Si growth at 500C.avi', type=str, help='video file path')
    parser.add_argument('--load', default= True, type=str, help='reload brightness array')

    args = parser.parse_args()
    cap = cv2.VideoCapture('.\RHEED\\'+ args.name)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    brightness = []
    time = []
    t = 0
    reload = args.load
    while(cap.isOpened()):
        if reload == True:
            brightness = np.load('arr.npy')
            time = np.linspace(0,len(brightness), num = len(brightness))/fps
            break
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            time.append(float(t)/fps)
            if t ==0:
                cv2.namedWindow('image')
                cv2.setMouseCallback('image',draw)
                while(1):

                    cv2.imshow('image',frame)
                    if cv2.waitKey(20) & 0xFF == 13:
                        break
                cv2.destroyAllWindows()
                cv2.waitKey(2000)
            t+=1
            N = 0

            for i in range(len(points)):
                img = frame[points[i][3]:points[i][1], points[i][2]:points[i][0]]
                N += np.mean(img, axis = (0,1))
            brightness.append(N)
        else:
            break
    time = np.asarray(time)
    np.save('arr.npy', np.asarray(brightness))
    yhat = savgol_filter(brightness, args.smooth, 3)#3 for polynom order

    plt.plot(time, yhat)
    plt.grid()
    plt.xlabel('Time, seconds')
    plt.ylabel('Mean pixel')
    plt.savefig('.\plots\S39_sSi+QDs_growth_plot'+ str(args.smooth)+ '.jpg')
    plt.show()

    ### fourier

    N = len(brightness)
    T = 1.0/fps
    t = np.linspace(0, N*T, N)
    yf = scipy.fftpack.fft(brightness)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)/2/np.pi
    k = 6 #throw out too big values in first elements
    plt.plot(xf[k:len(xf)],  2.0/N * np.abs(yf[k:N//2]))
    plt.xlabel(r'Frequency, $\frac{1}{seconds}$')
    plt.ylabel('Fourier amplitude')
    plt.grid()
    plt.show()
    # y = scipy.fftpack.ifft(yf)
    # plt.plot(t, y)
    # plt.grid()
    # plt.show()
    cap.release()
    cv2.destroyAllWindows()
