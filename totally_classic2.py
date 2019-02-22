import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy
import argparse
import torch
import  torch.nn.functional as F
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
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
            x1 = max(0,x)
            y1 = max(0,y)
            x2 = max(0,ix)
            y2 = max(0,iy)
            print(x1, y1,x2, y2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smooth', default=11, type=int, help='odd number, output graph smoothing')
    parser.add_argument('--name', default='.\special\R1272_Si growth at 500C.avi', type=str, help='video file path')
    parser.add_argument('--mode', default='static', type=str, help='moving or static difraction picture')

    args = parser.parse_args()
    cap = cv2.VideoCapture('.\RHEED\\'+ args.name)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    brightness = []
    time = []
    t = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        #frame = np.mean(frame, axis = 2)
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
                img = frame[y2:y1, x2:x1]
                #print(frame.shape)
                #print(img.shape)
                cv2.imshow('im',img)
                cv2.waitKey(2000)
            t+=1
            #img = frame[103:200, 154:300]
            if args.mode == 'static':
                img = frame[y2:y1, x2:x1]
                N = np.mean(img, axis = (0,1,2))
                brightness.append(N)
            if args.mode == 'moving':
                img = torch.from_numpy(img/255.).unsqueeze(0).unsqueeze(0)
                frame = torch.from_numpy(frame/255.).unsqueeze(0).unsqueeze(0)
                conv = F.conv2d(frame, img, stride = 15)
                original_conv = F.conv2d(img, img)
                conv -=original_conv
                conv = conv.abs()
                # coords = torch.argmax(conv, dim = 0)[0]
                # coords = (coords[0][0], coords[0][0][0])
                print(coords)
            if t %100 ==0:
                print(t)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame',gray)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    yhat = savgol_filter(brightness, args.smooth, 3)#3 for polynom order

    plt.plot(time, yhat)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Mean pixel')
    plt.savefig('.\plots\S39_sSi+QDs_growth_plot'+ str(args.smooth)+ '.jpg')
    plt.show()
    cap.release()
    cv2.destroyAllWindows()
