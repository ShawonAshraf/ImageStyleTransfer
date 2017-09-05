"""
usage : from command line run as :
python style_transfer.py style_image_path content_image_path
"""


import sys

import numpy as np
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras import backend
from keras.models import Model

import time

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


# ### Loading Images : Content Image and Style Image

def load_image(image_path):
    image = Image.open(image_path)
    return image


# We will also need to resize the image to reduce computational overhead
def resize_image(image, h, w):
    resized_image = image.resize((h, w))
    return resized_image


# We need to specifiy a default dimension for our images, we're setting it to 512 X 512

h = 512
w = 512

# Now we need to load the images and resize them. First the style image and then the content image.


style_img_path = sys.argv[1]

style_img = load_image(style_img_path)
style_img_resized = resize_image(style_img, h, w)

# Now the content image.
content_img_path = sys.argv[2]
content_img = load_image(content_img_path)
content_img_resized = resize_image(content_img, h, w)

# ### Converting images to arrays for numerical analysis


style_img_array = np.asarray(style_img_resized, dtype="float32")
content_img_array = np.asarray(content_img_resized, dtype="float32")

# We need to add an additional dimension to the arrays for bias.

style_img_array = np.expand_dims(style_img_array, axis=0)
content_img_array = np.expand_dims(content_img_array, axis=0)

print(content_img_array.shape)
print(style_img_array.shape)

# Now we need the average RGB value of an image.

AVG_RGB = np.array([103.939, 116.779, 123.68])

# Now we need to subtract the avg rgb from image and format it in BGR format (as stated in VGG16 paper)

style_array = style_img_array - AVG_RGB
content_array = content_img_array - AVG_RGB


def convert_to_bgr(image_array):
    return image_array[:, :, :, ::-1]


conv_style = convert_to_bgr(style_array)
conv_content = convert_to_bgr(content_array)

# Feeding images as variable into Keras


style_var = backend.variable(conv_style)
content_var = backend.variable(conv_content)

combination_img = backend.placeholder((1, h, w, 3))

# Since using Tensorflow backend and it needs tensors, we convert the image data into one concatened tensor.

input_tensor = backend.concatenate([content_var, style_var, combination_img], axis=0)

# Getting the model ready

model = VGG16(input_tensor=input_tensor, weights="imagenet", include_top=False)

# Let's load the layers


layers = dict([(layer.name, layer.output) for layer in model.layers])
layers

# arbitrary values

content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0

loss = backend.variable(0.)


# Measuring content loss


def get_content_loss(content, combo):
    return backend.sum(backend.square(content - combo))


layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight * get_content_loss(content_image_features, combination_features)


# Measuring style loss
# The best way to get style loss is to measure the gram matrix. #
# The terms of gram matrix are proportional to specific features in the style image.
#  So makes it easy for extraction.


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = h * w
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


# We need to define which layers in convonet we should be using.

feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']

for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl


# Total variation loss

def total_variation_loss(x):
    a = backend.square(x[:, :h - 1, :w - 1, :] - x[:, 1:, :w - 1, :])
    b = backend.square(x[:, :h - 1, :w - 1, :] - x[:, :h - 1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


loss += total_variation_weight * total_variation_loss(combination_img)

# Gradient for Optimisation

grad = backend.gradients(loss, combination_img)

# Evaluation

outputs = [loss]
outputs += grad
f_outputs = backend.function([combination_img], outputs)


def eval_loss_and_grads(x):
    x = x.reshape((1, h, w, 3))
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

# Optimisation

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

x = x.reshape((h, w, 3))

# Now we need to go back to RGB from BGR. Same procedure. Also we need to add average rgb value.
x = x[:, :, ::-1]
x = x + AVG_RGB

x = np.clip(x, 0, 255).astype("uint8")

# save image

imsave("out/{}_with_{}.jpg".format(content_img_path.split("/")[-1].split(".")[0],
                                   style_img_path.split("/")[-1].split(".")[0]), x)
