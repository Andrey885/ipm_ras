import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from tkinter import filedialog
from PIL import Image

def fourier_array():
    #global panelA, panelB
    path = filedialog.askopenfilename()
    brightness = np.load(path)
    time = np.linspace(0,len(brightness), num = len(brightness))/fps
    return 0

def fig2data (fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w,h,4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll (buf, 3, axis = 2)
    return buf

def fig2img (fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data (fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w,h), buf.tostring())

def count(img,ratio):
    #cv2.imshow('img1',img)
    #img2 = img
    # = cv2.Canny(img, 10, 40)
    #img2 = np.concatenate((img2, img), axis = 1)
    #cv2.resize(img2, (img2.shape[0]*3, img2.shape[1]*3))
    #cv2.imshow('img1',img2)
    #cv2.waitKey(1)
    ret,thresh = cv2.threshold(img,127,255,0)
    contours = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    print(img)
    return M['m00'],M['m10'],M['m01'],M['m20'],M['m02'],M['m11']


def plots(brightness, fps, path):
    time = np.linspace(1, len(brightness),len(brightness))
    name = get_last_dir(path)

    figure = plt.figure()
    plot   = figure.add_subplot(111)
    plot.plot (time, brightness)
    plt.grid()
    plt.xlabel('Time, seconds')
    plt.ylabel('Mean pixel')
    #im = fig2img (figure)
    #im.show()
    np.save('./plots/ '+ name +'.npy', np.asarray(brightness))

    ### fourier

    N = len(brightness)
    T = 1.0/fps
    t = np.linspace(0, N*T, N)
    yf = scipy.fft(brightness)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)*2*np.pi
    k = 6 #throw out too big values in first elements
    figure_f = plt.figure()
    plot = figure_f.add_subplot(111)
    plot.plot(xf[k:len(xf)],  2.0/N * np.abs(yf[k:N//2]))
    plt.grid()
    plt.xlabel(r'Frequency, $\frac{1}{seconds}$')
    plt.ylabel('Fourier amplitude')
    plt.savefig('./plots/' + name + '_fourier.jpg')
    #im_f = fig2img (figure)

    return figure, figure_f

def get_last_dir(path):
    path = path[:(len(path)-4)]
    i = 1
    name = ''
    while(i<len(path)-1):
        if path[-i] != '/':
            name = path[-i]+name
        else:
            return name
        i+=1
