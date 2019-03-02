from tkinter import *
from tkinter import filedialog
import scipy
from utils.utils import *
import cv2
import PIL.Image, PIL.ImageTk
import time
import matplotlib.pyplot as plt
import time

class App:
    def __init__(self, root, window_title):
        self.points = []
        self.x = 0
        self.y = 0
        self.x1 = 0
        self.y1 = 0
        self.t = 0
        self.drawing = False
        self.num_rectangles = 0
        self.root = root
        self.f_left = LabelFrame(root, text = 'Choose action')
        self.f_center = LabelFrame(root, text = 'Display')
        #self.f_right = LabelFrame(root, text = 'Plots')
        self.btn3 = None
        self.btn = Button(self.f_left, text="Select the videofile", command=self.select_file)
        self.btn2 = Button(self.f_left, text="Select the computed numpy array", command=self.select_file)
        self.canvas = Canvas(self.f_center, width = 720, height = 576, bg = 'white')
        #self.canvas_graph = Canvas(self.f_right, width = 400, height = 576, bg = 'white')
        self.canvas.pack()
        #self.canvas_graph.pack()
        self.f_left.pack(side = 'left')
        #self.f_right.pack(side = 'right')
        self.f_center.pack(side = 'right')
        self.btn.pack( expand=False, padx="10", pady="10")
        self.btn2.pack(expand=False, padx="10", pady="10")

        self.root.mainloop()

    def select_file(self):
        self.canvas.delete("all")
        self.video_source = filedialog.askopenfilename()
        if self.video_source[-4:]=='.avi':
            self.vid = VideoCap(self.video_source)
            self.delay = 0#0?
            if self.btn3:
                self.btn3.destroy()
                self.btn4.destroy()
                self.btn5.destroy()
                self.btn6.destroy()
            self.first_frame()
        elif self.video_source[-4:]=='.npy':
            self.brightness = np.load(self.video_source)
            self.im, self.im_f = plots(self.brightness, 25,self.video_source)#assume 25 fps
            if self.btn3:
                self.btn3.destroy()
                self.btn4.destroy()
                self.btn5.destroy()
                self.btn6.destroy()
            self.results()

    def results(self):
            im = fig2img (self.im)
            im_f = fig2img (self.im_f)
            w, h = im.size
            res = Image.new("RGBA", (w, 2*h))
            res.paste(im)
            res.paste(im_f, (0, h))
            res = res.resize((int(w*576/2/h),int(576)))
            self.graphic = PIL.ImageTk.PhotoImage(res)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image = self.graphic, anchor = 'nw')
            self.btn3 = Button(self.f_center, text="Full view", command=self.show)
            self.btn3.pack( expand=False, padx="10", pady="10", side = LEFT)
            self.btn4 = Button(self.f_center, text="Save plot as", command=self.save_plot)
            self.btn5 = Button(self.f_center, text="Save fourier plot as", command=self.save_fourier_plot)
            self.btn6 = Button(self.f_center, text="Save array as", command=self.save_array)
            self.btn3.pack( expand=False, padx="10", pady="10", side = LEFT)
            self.btn4.pack( expand=False, padx="10", pady="10", side = LEFT)
            self.btn5.pack( expand=False, padx="10", pady="10", side = LEFT)
            self.btn6.pack( expand=False, padx="10", pady="10", side = LEFT)
            self.video_source = 0

    def save_plot(self):
        f = filedialog.asksaveasfilename(initialdir = "./",title = "Select file",
        filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        fig = self.im
        plt.savefig(f)

    def save_fourier_plot(self):
        f = filedialog.asksaveasfilename(initialdir = "./",title = "Select file",
        filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        fig = self.im_f
        plt.savefig(f)

    def save_array(self):
        f = filedialog.asksaveasfilename(initialdir = "./",title = "Select file",
        filetypes = (("numpy arrays","*.npy"),("all files","*.*")))
        np.save(f, np.asarray(self.brightness))

    def show(self):
        self.im.show()
        self.im_f.show()

    def first_frame(self):
        self.brightness = []
        self.t = 0
        self.num_rectangles=0
        print('t zeroed')
        self.points = []
        ret, frame = self.vid.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        def mouse_move(event):
            if self.drawing == True:
                self.x = event.x
                self.y = event.y
                self.canvas.bind('<ButtonRelease>', mouse_stop)

        def mouse_on(event):
            self.x1 = event.x
            self.y1 = event.y
            self.drawing = True
            self.canvas.bind('<Motion>', mouse_move)

        def mouse_stop(event):
            self.drawing = False
            x1 = max(self.x1, event.x)
            y1 = max(self.y1, event.y)
            x2 = min(self.x1, event.x)
            y2 = min(self.y1, event.y)
            self.points.append((x1,y1,x2,y2))
            self.canvas.create_rectangle(self.points[self.num_rectangles][0],self.points[self.num_rectangles][1],
                self.points[self.num_rectangles][2],self.points[self.num_rectangles][3])
            self.num_rectangles +=1
        def enter(event):
            self.t += 1
            self.update()
        if ret:
            if self.t ==0:
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = NW)
                self.canvas.bind('<Button-1>', mouse_on)
                self.root.bind('<Return>', enter)

    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.t != 0:
                N = 0
                if self.t % 100 ==0:
                    print(self.t)
                for i in range(self.num_rectangles):
                    img = frame[self.points[i][3]:self.points[i][1], self.points[i][2]:self.points[i][0]]
                    img2 = cv2.Canny(img, 10, 40)
                    N += count(img2)
                    #N += np.mean(img, axis = (0,1))
                self.brightness.append(N)
                if self.t%10 == 0:
                    img = np.concatenate((img2, img), axis = 1)
                    self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image = self.photo, anchor = NW)
                    self.canvas.create_rectangle(self.points[self.num_rectangles-1][0],self.points[self.num_rectangles-1][1],
                        self.points[self.num_rectangles-1][2],self.points[self.num_rectangles-1][3])
                    self.canvas.update()
            self.t+=1
            self.root.after(self.delay, self.update)
        else:
            self.brightness = np.asarray(self.brightness)
            self.im, self.im_f = plots(self.brightness, self.vid.fps,self.video_source)
            self.results()
            return 0



class VideoCap:
    def __init__(self, video_source):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

            # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

            # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    # Create a window and pass it to the Application object
App(Tk(), "Tkinter and OpenCV")
