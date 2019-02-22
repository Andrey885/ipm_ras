import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import argparse
# import torch
import scipy
import os

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
    parser.add_argument('--name', default='./special/R1272_Ge QDs growth at 600C.avi', type=str, help='video file path')
    parser.add_argument('--mode', default='horizontal',type = str, help='moving or static difraction picture' )
    parser.add_argument('--load', default= False, type=str, help='reload brightness array')

    args = parser.parse_args()
    cap = cv2.VideoCapture('.\RHEED\\'+ args.name)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    brightness = []
    time = []
    t = 0
    reload = args.load
    # image_folder = 'images'
    video_name = 'video.avi'
    #images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    height, width, layers = 480, 640, 3
    video = cv2.VideoWriter(video_name, -1, fps, (width,height))


    while(cap.isOpened()):
        if reload == True:
            brightness = np.load('arr.npy')
            time = np.linspace(0,len(brightness), num = len(brightness))
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
                b_hor = np.ones((-points[0][2]+points[0][0]))
                b_vert = np.ones((-points[0][3]+points[0][1]))
                cv2.destroyAllWindows()
                cv2.waitKey(2)
            t+=1
            img = frame[points[0][3]:points[0][1], points[0][2]:points[0][0]]
            if args.mode == 'horizontal':
                N = np.dot(img,b_hor)/img.shape[1]
                #N1 = np.mean(img, axis = 1)#not multithreaded
            if args.mode == 'vertical':
                #N = np.mean(img, axis = (0,2))#not multithreaded
                N = np.dot(img,b_vert)/img.shape[0]
            fig = plt.figure()
            plt.plot( np.linspace(0, len(N),len(N)), N)
            plt.grid()
            #print(fig)
            plt.savefig('fig.png')
            plt.close()
            frame2 = cv2.imread('fig.png')
            # if t%10 == 0:
            #     print(t)
            video.write(frame2)
            if t %100 ==0:
                print(t, 'frames out of', cap.get(cv2.CAP_PROP_FRAME_COUNT), 'processed')
                #break
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0))
                cv2.imshow('image', frame)
                cv2.waitKey(1)

        else:
            break
    print('videofile saved at ./', video_name)
    cv2.destroyAllWindows()
    video.release()
