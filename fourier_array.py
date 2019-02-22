import numpy as np
from tkinter import filedialog

def fourier_array():
    global panelA, panelB
    path = filedialog.askopenfilename()
    brightness = np.load(path)
    time = np.linspace(0,len(brightness), num = len(brightness))/fps
    return 0
