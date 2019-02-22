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

def plots(brightness):
    brightness = np.asarray(brightness)
    time = np.linspace(1, len(brightness),len(brightness))
    np.save('edges.npy', np.asarray(brightness))
    plt.plot(time, brightness)
    plt.grid()
    plt.xlabel('Time, seconds')
    plt.ylabel('Mean pixel')
    plt.savefig('.\plots\S39_sSi+QDs_growth_plot'+ '.jpg')
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
    plt.savefig('.\plots\S39_sSi+QDs_growth_plot'+ 'fourier.jpg')

    plt.grid()
    plt.show()

def draw(event,x,y,flags,param):
    global ix,iy,drawing,mode
    global x1, y1, x2,y2
    # sframe2 = frame
    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(frame,(ix,iy),(x,y),(0,255,0))
            x1 = max(x, ix)
            y1 = max(y, iy)
            x2 = min(x, ix)
            y2 = min(y, iy)
            points.append((x1,y1,x2,y2))
            print(x1, y1,x2, y2)
def count(img):
    #print(img.shape)
    cv2.imshow('img1',img)
    # cv2.resize(img, (900,900))
    # cv2.waitKey(500)
    img = cv2.Canny(img, 10, 40)
    #print(img.shape)
    cv2.imshow('img2',img)

    #cv2.resize(img, (900,900))
    cv2.waitKey(10)
    ret,thresh = cv2.threshold(img,127,255,0)
    contours = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    #area = cv2.contourArea(cnt)
    #print(M)
    #exit()
    return M['m10']

cap = cv2.VideoCapture('.\RHEED\\'+'./special/R1272_Ge QDs growth at 600C.avi')
fps    = cap.get(cv2.CAP_PROP_FPS)
brightness = []


t= 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #time.append(float(t)/fps)
        if t ==0:
            #points = [[548, 368, 92, 125]]
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',draw)
            while(1):
                cv2.imshow('image',frame)
                if cv2.waitKey(20) & 0xFF == 13:
                    break
            cv2.destroyAllWindows()
            cv2.waitKey(200)
            cv2.namedWindow('img1')
            cv2.namedWindow('img2')
        t+=1
        N = 0
        if t % 100 ==0:
            print(t)

        for i in range(len(points)):
            img = frame[points[i][3]:points[i][1], points[i][2]:points[i][0]]
            N = count(img)

            #N += np.mean(img, axis = (0,1))
        brightness.append(N)


    else:
        break
plots(brightness)
