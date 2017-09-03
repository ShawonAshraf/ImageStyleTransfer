# import modules

import numpy as np
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras import backend
from keras.models import Model

import time

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

"""
    Loads and resizes images from the image path,
    converts to numeric arrays for math ops
"""


class ImageProcessor:
    def __init__(self, image_path, height, width):
        self.image_path = image_path
        self.height = height
        self.width = width
        self.image = None


    def load_image_resize(self):
        try:
            self.image = Image.open(self.image_path)
            self.image = self.image.resize(self.height, self.width)
            return self.image
        except Exception as e:
            print(e)


    def convert_image_to_array(self):
        img_array = np.asarray(self.image, dtype="float32")
        return img_array


class RGBFilter:
    AVG_RGB_FILTER = np.array([])