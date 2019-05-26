from tkinter import *
from tkinter import filedialog
#import scipy
from utils.utils import *
import cv2
import PIL.Image, PIL.ImageTk
import time
import matplotlib.pyplot as plt
from tkinter.ttk import Progressbar

class App:
    def __init__(self, root, window_title):
        self.points = []
        #self.trash_arr = []
        self.x = 0
        self.y = 0
        self.x1 = 0
        self.y1 = 0
        self.t = 0
        self.dist1 = []
        self.dist2 = []
        self.space_mode = None
        self.drawing = False
        self.num_rectangles = 0
        self.root = root
        self.f_left = LabelFrame(root, text = 'Choose action')
        self.f_center = LabelFrame(root, text = 'Display')
        #self.f_right = LabelFrame(root, text = 'Plots')
        self.btn3 = None
        self.btn = Button(self.f_left, text="Select the videofile", command=self.select_file_difraction)
        self.btn2 = Button(self.f_left, text="Select the computed numpy array", command=self.select_file_np)
        self.btn2_1 = Button(self.f_left, text="Select the videofile for horizontal projection",
                            command=self.select_file_horizontal)
        self.btn2_2 = Button(self.f_left, text="Select the videofile for vertical projection",
                            command=self.select_file_vertical)
        self.canvas = Canvas(self.f_center, width = 720, height = 576, bg = 'white')
        #self.canvas_graph = Canvas(self.f_right, width = 400, height = 576, bg = 'white')
        self.canvas.pack()
        #self.canvas_graph.pack()
        self.f_left.pack(side = 'left')
        #self.f_right.pack(side = 'right')
        self.f_center.pack(side = 'right')
        self.btn.pack( expand=False, padx="10", pady="10")
        self.btn2.pack(expand=False, padx="10", pady="10")
        self.btn2_1.pack(expand=False, padx="10", pady="10")
        self.btn2_2.pack(expand=False, padx="10", pady="10")

        self.root.mainloop()

    def select_file_difraction(self):
        self.space_mode = 'zero'
        self.canvas.delete("all")
        self.video_source = filedialog.askopenfilename()
        self.vid = VideoCap(self.video_source)
        self.delay = 0#0?
        if self.btn3:
            self.btn3.destroy()
            self.btn4.destroy()
            self.btn5.destroy()
            self.btn6.destroy()
        self.first_frame()

    def select_file_np(self):
        self.space_mode = 'zero'
        self.canvas.delete("all")
        self.video_source = filedialog.askopenfilename()
        self.brightness = np.load(self.video_source)
        self.im, self.im_f = plots(self.brightness, 25,self.video_source)#assume 25 fps
        if self.btn3:
            self.btn3.destroy()
            self.btn4.destroy()
            self.btn5.destroy()
            self.btn6.destroy()
        self.results()

    def select_file_horizontal(self):
        self.canvas.delete("all")
        self.video_source = filedialog.askopenfilename()
        self.vid = VideoCap(self.video_source)
        self.delay = 0#0?
        self.space_mode = 'horizontal'
        if self.btn3:
            self.btn3.destroy()
            self.btn4.destroy()
            self.btn5.destroy()
            self.btn6.destroy()
        self.first_frame()

    def select_file_vertical(self):
        self.canvas.delete("all")
        self.video_source = filedialog.askopenfilename()
        self.vid = VideoCap(self.video_source)
        self.delay = 0#0?
        self.space_mode = 'vertical'
        if self.btn3:
            self.btn3.destroy()
            self.btn4.destroy()
            self.btn5.destroy()
            self.btn6.destroy()
        self.first_frame()
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
            if self.video_source[-4:]=='.avi':
                self.progress.destroy()
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
            self.ratio = (x1-x2)/(y1-y2)
            #print(self.ratio)
            self.canvas.create_rectangle(self.points[self.num_rectangles][0],self.points[self.num_rectangles][1],
                self.points[self.num_rectangles][2],self.points[self.num_rectangles][3])
            self.num_rectangles +=1
        def escape(event):
            self.num_rectangles = 0
            self.points = []
            self.canvas.create_image(0, 0, image = self.photo, anchor = 'nw')

        def enter(event):
            self.points = np.asarray(self.points)
            self.progress=Progressbar(self.f_center,orient=HORIZONTAL,length=100,mode='determinate')
            self.progress.pack(side = BOTTOM, fill = 'x')
            #self.t += 1
            if self.space_mode == 'zero':
                self.update()
            if self.space_mode == 'horizontal' or self.space_mode == 'vertical':
                self.video_name = 'video.avi'
                height, width, layers = 480, 640, 3
                self.output_video = cv2.VideoWriter(self.video_name, -1, self.vid.fps, (width,height))
                self.update_space()

        if ret:
            if self.t ==0:

                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = NW)
                self.canvas.bind('<Button-1>', mouse_on)
                self.root.bind('<Return>', enter)
                self.root.bind('<Escape>', escape)


    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #if self.t ==0:
                # self.stack_len = 1
                # self.stack = np.zeros((self.stack_len,-self.points[0][3]+self.points[0][1],-self.points[0][2]+self.points[0][0]))
                # for i in range(self.stack_len):
                #     self.stack[i] = frame[self.points[0][3]:self.points[0][1], self.points[0][2]:self.points[0][0]]
                #self.img1 = frame[self.points[0][3]:self.points[0][1], self.points[0][2]:self.points[0][0]]

            if self.t != 0:
                N = 0
                if self.t % 100 ==0:
                    print(self.t)
                for i in range(self.num_rectangles):
                    # for i in range(self.stack_len-1):
                    #     self.stack[i] = self.stack[i+1]
                    #print(frame[self.points[0][3]:self.points[0][1], self.points[0][2]:self.points[0][0]].shape)
                    len_x = -self.points[i][2]+self.points[i][0]
                    len_y = -self.points[i][2]+self.points[i][0]
                    img = frame[self.points[i][3]:self.points[i][1], self.points[i][2]:self.points[i][0]]
                    # ret,thresh = cv2.threshold(img,127,255,0)
                    _, thresh = cv2.threshold(img,117,255,0)
                    # thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    #             cv2.THRESH_BINARY,11,2)
                    # thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                    #             cv2.THRESH_BINARY,57,0)
                    #cv2.imshow('thr', thresh)
                    #cv2.waitKey(300)
                    contours = cv2.findContours(thresh, 1, 2)
                    cnt = contours[0]
                    M = cv2.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])#+10

                    self.points[i][0]+=int(cX-len_x//2)
                    self.points[i][2]+=int(cX-len_x//2)
                    self.points[i][1]+=int(cY-len_y//2)
                    self.points[i][3]+=int(cY-len_y//2)
                    img = frame[self.points[i][3]:self.points[i][1], self.points[i][2]:self.points[i][0]]
                    # img = img[(cX-int(img.shape[0]/3)):(cX+int(img.shape[0]/3)),
                    # (cY-int(img.shape[1]/3)):(cY+int(img.shape[1]/3))]
                    mean = np.mean(img)
                    std = np.std(img)
                    for i in range(img.shape[0]):
                        for j in range(img.shape[1]):
                            if img[i][j] < mean or img[i][j]>252:# - std/2:
                                img[i][j]=0
                    N += count4(img)
                    cv2.imshow('frame', img)

                    #N += np.sum(img, axis = (0,1))
                    #print(N)
                if self.num_rectangles==3:
                    self.dist1.append(abs((self.points[0][2]+self.points[0][0])/2 -
                                        ((self.points[1][2]+self.points[1][0])/2)))
                    self.dist2.append(abs((self.points[0][2]+self.points[0][0])/2 -
                                        ((self.points[2][2]+self.points[2][0])/2)))
                self.brightness.append(N)
                if self.t%10 == 0:
                    #img = np.concatenate((img2, img), axis = 1)
                    self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image = self.photo, anchor = NW)
                    for i in range(self.num_rectangles):
                        self.canvas.create_rectangle(self.points[i][0],self.points[i][1],
                            self.points[i][2],self.points[i][3])
                    self.canvas.update()
                    print(self.t,'/', self.vid.t_max)
                    self.progress['value'] = int(self.t/self.vid.t_max*100)
                    self.f_center.update_idletasks()
            self.t+=1
            #self.trash_arr.append(self.t)
            self.root.after(self.delay, self.update)
        else:
            self.brightness = np.asarray(self.brightness)
            #self.trash_arr = np.asarray(self.trash_arr)
            #self.brightness = np.sin(self.trash_arr)
            self.im, self.im_f = plots(self.brightness, self.vid.fps,self.video_source)
            self.results()
            # plt.plot(self.dist1)
            # plt.grid()
            # plt.show()
            # plt.plot(self.dist2)
            # plt.grid()
            # plt.show()
            return 0

    def update_space(self):
        ret, frame = self.vid.get_frame()
        self.b_vert = np.ones((-self.points[0][2]+self.points[0][0]))
        self.b_hor = np.ones((-self.points[0][3]+self.points[0][1]))
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.t != 0:
                N = 0
                if self.t % 100 ==0:
                    print(self.t)
                for i in range(self.num_rectangles):
                    img = frame[self.points[i][3]:self.points[i][1], self.points[i][2]:self.points[i][0]]
                    if self.space_mode == 'horizontal':
                        N = np.dot(self.b_hor,img)/img.shape[1]
                        #N1 = np.mean(img, axis = 1)#not multithreaded
                    if self.space_mode == 'vertical':
                        #N = np.mean(img, axis = (0,2))#not multithreaded
                        N = np.dot(img,self.b_vert)/img.shape[0]
                fig = plt.figure()
                plt.plot( np.linspace(0, len(N),len(N)), N)
                plt.grid()
                plt.savefig('fig.png')
                plt.close()
                frame2 = cv2.imread('fig.png')
                self.output_video.write(frame2)
                if self.t%10 == 0:
                    #img = np.concatenate((img2, img), axis = 1)
                    self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image = self.photo, anchor = NW)
                    for i in range(self.num_rectangles):
                        self.canvas.create_rectangle(self.points[i][0],self.points[i][1],
                            self.points[i][2],self.points[i][3])
                    self.canvas.update()
                    print(self.t,'/', self.vid.t_max)
                    self.progress['value'] = int(self.t/self.vid.t_max*100)
                    self.f_center.update_idletasks()

            self.t+=1
            #self.trash_arr.append(self.t)
            self.root.after(self.delay, self.update_space)
        else:
            self.output_video.release()
            print('videofile saved at ./', self.video_name)

            #self.results()
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
        self.t_max = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)

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
