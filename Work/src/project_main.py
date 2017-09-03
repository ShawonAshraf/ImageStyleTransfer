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
    def load_image_resize(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((ImageRes.HEIGHT, ImageRes.WIDTH))
            return image
        except Exception as e:
            print(e)


"""
    Converts Image to numeric arrays for math ops
"""


class ImageConverter:
    def convert_image_to_array(self, image):
        img_array = np.asarray(image, dtype="float32")
        return img_array

    """
        converts rgb to bgr and bgr top rgb
    """

    def rev_color_map(self, img_array):
        return img_array[:, :, :, ::-1]

    def sub_avg_filter(self, arr):
        return arr - RGBFilter.AVG_RGB_FILTER

    def get_converted_image(self, image):
        arr = self.convert_image_to_array(image)
        return self.sub_avg_filter(arr)

"""
    class done
"""

content_image_path = "../notebook/images/content/shakib.jpg"
style_image_path = "../notebook/images/style/StarryNight_VincentVanGogh.jpg"


print("Loading images : content - {}\tstyle - {}".format(
    content_image_path.split("/")[-1],
    style_image_path.split("/")[-1]
))
loader = ImageLoader()
content_image = loader.load_image_resize(image_path=content_image_path)
style_image = loader.load_image_resize(image_path=style_image_path)


print("\n\nConverting images : content - {}\tstyle - {}".format(
    content_image_path.split("/")[-1],
    style_image_path.split("/")[-1]
))
converter = ImageConverter()
content_image = converter.get_converted_image(content_image)
style_image = converter.get_converted_image(style_image)

style_img_array = np.expand_dims(style_image, axis=0)
content_img_array = np.expand_dims(content_image, axis=0)

# CNN
print("\nCreating CNN....")







