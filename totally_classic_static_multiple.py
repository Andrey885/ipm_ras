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

        drawing = True
        ix,iy = x,y
        #cv2.rectangle(frame,(ix,iy),(x,y),(0,255,0))
    # elif event == cv2.EVENT_MOUSEMOVE:
    #     if drawing == True:
    #         if mode == True:
    #             cv2.rectangle(frame,(ix,iy),(x,y),(0,255,0),-1)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smooth', default=11, type=int, help='odd number, output graph smoothing')
    parser.add_argument('--name', default='./special/R1272_Si growth at 500C.avi', type=str, help='video file path')
    parser.add_argument('--mode', default='time_distribution',options = 'time_distribution', 'horizontal_distribution', 'vertical_distribution', type=str, help='moving or static difraction picture')
    parser.add_argument('--load', default= False, type=str, help='reload brightness array')

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
            time = np.linspace(0,len(brightness), num = len(brightness))
            break
        ret, frame = cap.read()
        if ret==True:
            time.append(float(t)/fps)
            if t ==0:
                cv2.namedWindow('image')
                cv2.setMouseCallback('image',draw)
                while(1):
                    cv2.imshow('image',frame)
                    if cv2.waitKey(20) & 0xFF == 13:
                        break
                cv2.destroyAllWindows()
                #img = np.zeros((x1-ix,y1-iy,3))
                    # cv2.imshow('im', part_img)
                    # cv2.waitKey(200)
                #print(frame.shape)
                #print(img.shape)
                #cv2.imshow('im',img)
                cv2.waitKey(2000)
            t+=1
            N = 0
            if args.mode == 'static':
                for i in range(len(points)):
                    img = frame[points[i][3]:points[i][1], points[i][2]:points[i][0]]
                    N += np.mean(img, axis = (0,1,2))
                brightness.append(N)
            if args.mode == 'moving':
                continue###
            if t %100 ==0:
                print(t)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0))
                cv2.imshow('image', frame)
                cv2.waitKey(10)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame',gray)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    np.save('arr.npy', np.asarray(brightness))
    yhat = savgol_filter(brightness, args.smooth, 3)#3 for polynom order
    plt.plot(time, yhat)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Mean pixel')
    plt.savefig('.\plots\S39_sSi+QDs_growth_plot'+ str(args.smooth)+ '.jpg')
    plt.show()
    ### fourier
    #time = np.asarray(time[1:len(time)])
    yf = scipy.fftpack.fft(brightness)

    plt.plot(1/time, np.abs(yf))
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Mean pixel fourier')
    plt.savefig('.\plots\S39_sSi+QDs_growth_plot'+ str(args.smooth)+ '.jpg')
    plt.show()
    cap.release()
    cv2.destroyAllWindows()
