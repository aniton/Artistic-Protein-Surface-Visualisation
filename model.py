import numpy as np
import tensorflow.compat.v1 as tf
import numpy as np

from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from scipy.misc import imread, imsave, imresize

layers = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',

    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1'
)

def AdaIN(content_features, style_features, alpha=1, epsilon = 1e-5):

    content_mean, content_variance = tf.nn.moments(content_features, [1, 2], keepdims=True)
    style_mean, style_variance = tf.nn.moments(style_features, [1, 2], keepdims=True)
    normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                            content_variance, style_mean,
                                                            tf.sqrt(style_variance), epsilon)
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    
    
    return normalized_content_features

class Encoder(object):

    def __init__(self, weights_path):
        # load weights (kernel and bias) from npz file
        weights = np.load(weights_path)

        idx = 0
        self.weight_vars = []

        # create the TensorFlow variables
        with tf.compat.v1.variable_scope('encoder'):
            for layer in layers:
                kind = layer[:4]

                if kind == 'conv':
                    kernel = weights['arr_%d' % idx].transpose([2, 3, 1, 0])
                    bias   = weights['arr_%d' % (idx + 1)]
                    kernel = kernel.astype(np.float32)
                    bias   = bias.astype(np.float32)
                    idx += 2

                    with tf.compat.v1.variable_scope(layer):
                        W = tf.Variable(kernel, trainable=False, name='kernel')
                        b = tf.Variable(bias,   trainable=False, name='bias')

                    self.weight_vars.append((W, b))

    def encode(self, image):
        idx = 0
        net = {}
        current = image

        for layer in layers:

            kind = layer[:4]

            if kind == 'conv':
                kernel, bias = self.weight_vars[idx]
                idx += 1
                current = conv2d(current, kernel, bias)

            elif kind == 'relu':
                current = tf.nn.relu(current)

            elif kind == 'pool':
                current = pool2d(current)

            net[layer] = current

        assert(len(net) == len(layers))
        enc = net[layers[-1]]

        return enc,net

    def preprocess(self, image, mode='BGR'):
        if mode == 'BGR':
            return image - np.array([103.939, 116.779, 123.68])
        else:
            return image - np.array([123.68, 116.779, 103.939])

    def deprocess(self, image, mode='BGR'):
        if mode == 'BGR':
            return image + np.array([103.939, 116.779, 123.68])
        else:
            return image + np.array([123.68, 116.779, 103.939])
def conv2d(x, kernel, bias):
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    return out

def pool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

import tensorflow as tf
import numpy as np


WEIGHT_INIT_STDDEV = 0.1
DECODER_LAYERS = (
    'conv4_1', 'relu4_1', 'upsample', 'conv3_4', 'relu3_4',

    'conv3_3', 'relu3_3', 'conv3_2',  'relu3_2', 'conv3_1',

    'relu3_1', 'upsample', 'conv2_2', 'relu2_2', 'conv2_1',

    'relu2_1', 'upsample', 'conv1_2',  'relu1_2', 'conv1_1',
)

class Decoder(object):

    def __init__(self,mode='train',weights_path=None):

        self.weight_vars = []

        if(mode=='train'):

            with tf.compat.v1.variable_scope('decoder'):
                self.weight_vars.append(self._create_variables(512, 256, 3, scope='conv4_1'))

                self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_4'))
                self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_3'))
                self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_2'))
                self.weight_vars.append(self._create_variables(256, 128, 3, scope='conv3_1'))

                self.weight_vars.append(self._create_variables(128, 128, 3, scope='conv2_2'))
                self.weight_vars.append(self._create_variables(128, 64, 3, scope='conv2_1'))

                self.weight_vars.append(self._create_variables(64, 64, 3, scope='conv1_2'))
                self.weight_vars.append(self._create_variables(64, 3, 3, scope='conv1_1'))
        else:
            weights = np.load(weights_path,allow_pickle=True)[()]
            with tf.compat.v1.variable_scope('decoder'):
                for layer in DECODER_LAYERS:

                    kind = layer[:4]
                    if kind == 'conv':
                        kernel = weights[layer+'/kernel']
                        bias   = weights[layer+'/bias']
                        kernel = kernel.astype(np.float32)
                        bias   = bias.astype(np.float32)

                        with tf.compat.v1.variable_scope(layer):
                            W = tf.Variable(kernel, trainable=False, name='kernel')
                            b = tf.Variable(bias,   trainable=False, name='bias')

                        self.weight_vars.append((W, b))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.compat.v1.variable_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.Variable(tf.compat.v1.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
            bias = tf.Variable(tf.zeros([output_filters]), name='bias')
            return (kernel, bias)


    def decode(self, image):
        upsample_indices = (0, 4, 6)
        final_layer_idx = len(self.weight_vars) - 1
        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            if i == final_layer_idx:
                out = conv2dd(out, kernel, bias, use_relu=False)
            else:
                out = conv2dd(out, kernel, bias)

            if i in upsample_indices:
                out = upsample(out)

        return out

def conv2dd(x, kernel, bias, use_relu=True):
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    if use_relu:
        out = tf.nn.relu(out)

    return out

def upsample(x, scale=2):
    height = tf.shape(x)[1] * scale
    width = tf.shape(x)[2] * scale
    output = tf.compat.v1.image.resize_images(x, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return output


def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    return images


def get_train_images(paths, resize_len=512, crop_height=256, crop_width=256):
    images = []
    for path in paths:
        image = imread(path, mode='RGB')
        height, width, _ = image.shape

        if height < width:
            new_height = resize_len
            new_width = int(width * new_height / height)
        else:
            new_width = resize_len
            new_height = int(height * new_width / width)

        image = imresize(image, [new_height, new_width], interp='nearest')

        # crop the image
        start_h = np.random.choice(new_height - crop_height + 1)
        start_w = np.random.choice(new_width - crop_width + 1)
        image = image[start_h:(start_h + crop_height), start_w:(start_w + crop_width), :]

        images.append(image)

    images = np.stack(images, axis=0)

    return images
