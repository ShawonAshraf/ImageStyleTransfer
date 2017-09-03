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
            image = image.resize(ImageRes.HEIGHT, ImageRes.WIDTH)
            return image
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

        self.loss = None
        self.style_loss = None

        self.output_img = None

    """
        Methods
    """

    def create_backend_vars(self):
        self.style = backend.variable(self.style_img_array, dtype="float32")
        self.content = backend.variable(self.content_img_array, dtype="float32")
        self.combination_img = backend.placeholder((1, ImageRes.HEIGHT, ImageRes.WIDTH, 3))
        self.content_loss = backend.variable(0.)

    def create_input_tensor(self):
        self.input_tensor = backend.concatenate([self.content, self.style, self.combination_img],
                                                axis=0)

    def prepare_model(self):
        self.model = VGG16(input_tensor=self.input_tensor, weights="imagenet", include_top=False)

    def load_layers(self):
        self.layers = dict([(layer.name, layer.output) for layer in self.model.layers])

    def get_content_loss(self, content, combo):
        self.content_loss = backend.sum(backend.square(content - combo))

    def measure_content_loss(self):
        layer_features = self.layers['block2_conv2']
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]

        self.loss += self.Weights.CONTENT_WEIGHT * \
                     self.content_loss

    def __gram_matrix(self, x):
        features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
        gram = backend.dot(features, backend.transpose(features))
        return gram

    def style_loss(self):
        S = self.__gram_matrix(self.style)
        C = self.__gram_matrix(self.combination_img)
        channels = 3
        size = ImageRes.HEIGHT * ImageRes.WIDTH
        self.style_loss = backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    def use_feature_layers(self):
        feature_layers = ['block1_conv2', 'block2_conv2',
                          'block3_conv3', 'block4_conv3',
                          'block5_conv3']

        for layer_name in feature_layers:
            layer_features = self.layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = self.style_loss(style_features, combination_features)
            self.loss += (self.Weights.STYLE_WEIGHT / len(feature_layers)) * sl

    def total_variation_loss(x):
        h = ImageRes.HEIGHT
        w = ImageRes.WIDTH

        a = backend.square(x[:, :h - 1, :w - 1, :] - x[:, 1:, :w - 1, :])
        b = backend.square(x[:, :h - 1, :w - 1, :] - x[:, :h - 1, 1:, :])
        return backend.sum(backend.pow(a + b, 1.25))

    def evaluation(self):
        self.loss += self.Weights.TOTAL_VARIATION_WEIGHT * \
                     self.total_variation_loss(self.combination_img)

        grad = backend.gradients(self.loss, self.combination_img)

        outputs = [self.loss]
        outputs += grad
        f_outputs = backend.function([self.combination_img], outputs)

        def eval_loss_and_grads(x):
            x = x.reshape((1, ImageRes.HEIGHT, ImageRes.WIDTH, 3))
            outs = f_outputs([x])
            loss_value = outs[0]
            grad_values = outs[1].flatten().astype('float64')
            return loss_value, grad_values

        class Evaluator(object):
            def __init__(self):
                self.loss_value = None
                self.grads_values = None

            def loss(self, x):
                assert self.loss_value is None
                loss_value, grad_values = eval_loss_and_grads(x)
                self.loss_value = loss_value
                self.grad_values = grad_values
                return self.loss_value

            def grads(self, x):
                assert self.loss_value is not None
                grad_values = np.copy(self.grad_values)
                self.loss_value = None
                self.grad_values = None
                return grad_values

        evaluator = Evaluator()

        x = np.random.uniform(0, 255, (1, h, w, 3)) - 128.

        iterations = 10

        for i in range(iterations):
            print('Start of iteration', i)
            start_time = time.time()
            x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                             fprime=evaluator.grads, maxfun=20)
            print('Current loss value:', min_val)
            end_time = time.time()
            print('Iteration %d completed in %ds' % (i, end_time - start_time))

        x = x.reshape((ImageRes.HEIGHT, ImageRes.WIDTH, 3))
        x = x[:, :, ::-1]
        x = x + RGBFilter.AVG_RGB_FILTER
        x = np.clip(x, 0, 255).astype("uint8")

        self.output_img = Image.fromarray(x)

    def output_img(self):
        imsave("out.jpg", self.output_img)


"""
    class done
"""

content_image_path = "../notebook/images/content/shakib.jpg"
style_image_path = "../notebook/images/content/StarryNight_VincentVanGogh.jpg"

