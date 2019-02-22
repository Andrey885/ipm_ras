import numpy as np
import PIL
from PIL import Image

arr = 255*np.ones((720,300))/20
img = Image.fromarray(arr.astype('uint8'))
