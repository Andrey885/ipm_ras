import numpy as np
import cv2
from tkinter import *

from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
from select_video import *#select_video
from fourier_array import fourier_array

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
points = []

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

#if __name__ == '__main__':
def select_video():
    global panelA, panelB
    path = filedialog.askopenfilename()
    if len(path) > 0:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        brightness = []
        time = []
        t = 0
        while(cap.isOpened()):
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
                    # image = Image.fromarray(frame.astype('uint8'))
                    # image = ImageTk.PhotoImage(image)
                    # panelA = Label(image=image)
                    # panelA.image = image
                    # panelA.pack(side="left", padx=10, pady=10)
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


root = Tk()
root.config(height=500, width=500)
# can = Canvas(root, bg = 'red', height=100, width=100)
# can.place(relx=0.5, rely=0.5, anchor=CENTER)
panelA = None
panelB = None


btn = Button(root, text="Select the videofile", command=select_video)
btn2 = Button(root, text="Select the computed numpy array", command=fourier_array)

btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn2.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()
