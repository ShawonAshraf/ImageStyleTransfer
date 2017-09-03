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
"""


class RGBFilter:
    AVG_RGB_FILTER = np.array([103.939, 116.779, 123.68])


"""
    Setting default image size as 512 X 512
"""


class ImageRes:
    HEIGHT = 512
    WIDTH = 512


class ImageLoader:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None

    def load_image_resize(self):
        try:
            self.image = Image.open(self.image_path)
            self.image = self.image.resize(ImageRes.HEIGHT, ImageRes.WIDTH)
            return self.image
        except Exception as e:
            print(e)


"""
    Converts Image to numeric arrays for math ops
"""


class ImageConverter:
    def __init__(self, image):
        self.image = image
        self.img_array = None

    def convert_image_to_array(self):
        self.img_array = np.asarray(self.image, dtype="float32")
        return self.img_array

    """
        converts rgb to bgr and bgr top rgb
    """

    def rev_color_map(self):
        return self.img_array[:, :, :, ::-1]

    def sub_avg_filter(self):
        arr = self.rev_color_map()
        return arr - RGBFilter.AVG_RGB_FILTER


"""
    CNN model using VGG16
"""


class StyleTransferCNN:
    class Weights:
        CONTENT_WEIGHT = 0.025
        STYLE_WEIGHT = 5.0
        TOTAL_VARIATION_WEIGHT = 1.0

    def __init__(self, style_img_array, content_img_array):
        self.style_img_array = style_img_array
        self.content_img_array = content_img_array
        self.combination_img = None
        self.content_loss = None
        self.input_tensor = None
        self.model = None
        self.layers = None

        self.style = None
        self.content = None

    """
        Methods are to be called sequentially
    """

    def create_backend_vars(self):
        self.style = backend.variable(self.style_img_array, dtype="float32")
        self.content = backend.variable(self.content_img_array, dtype="float32")
        self.combination_img = backend.placeholder((1, ImageRes.HEIGHT, ImageRes.WIDTH, 3))

    def prepare_model(self):
        self.model = VGG16(input_tensor=self.input_tensor, weights="imagenet", include_top=False)

    def load_layers(self):
        self.layers = dict([(layer.name, layer.output) for layer in self.model.layers])
