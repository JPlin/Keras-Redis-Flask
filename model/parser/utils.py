"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.color
import skimage.io
import skimage.transform

import random
import itertools
import colorsys

import tensorflow as tf
from tensorflow.python.ops import array_ops
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM

# from visualize import random_colors


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


BatchNorm = KL.BatchNormalization

MEAN_PIXEL = np.array([123.7, 116.8, 103.9], np.float32)

MEAN_MOLDED_LANDMARK68_PTS = \
    np.array([[0.31150118, 0.43065098],
              [0.3112171, 0.479555],
              [0.31499308, 0.5285248],
              [0.32577008, 0.5764169],
              [0.34579322, 0.620173],
              [0.375563, 0.65832347],
              [0.41178364, 0.6903133],
              [0.45138708, 0.71581167],
              [0.49748972, 0.7247057],
              [0.54118365, 0.71401376],
              [0.5761339, 0.68649787],
              [0.6081181, 0.65362126],
              [0.6346203, 0.61538506],
              [0.6529905, 0.57259125],
              [0.6636283, 0.5265129],
              [0.66718155, 0.47954908],
              [0.6667386, 0.43270048],
              [0.3602174, 0.38455132],
              [0.38331425, 0.36887804],
              [0.41246626, 0.36357492],
              [0.44258773, 0.36741775],
              [0.470027, 0.37900513],
              [0.5330277, 0.37964728],
              [0.558197, 0.36836636],
              [0.5856716, 0.36435285],
              [0.611964, 0.3686599],
              [0.63203347, 0.3834175],
              [0.5020143, 0.42084876],
              [0.5028395, 0.4511575],
              [0.5039485, 0.4808996],
              [0.50500464, 0.51151866],
              [0.46575123, 0.5366393],
              [0.48354352, 0.5427595],
              [0.5026651, 0.54711986],
              [0.52118057, 0.5429736],
              [0.5377653, 0.536508],
              [0.39204696, 0.42528442],
              [0.41199672, 0.41340917],
              [0.43583882, 0.41324082],
              [0.4543986, 0.42725304],
              [0.43397772, 0.43393713],
              [0.4113788, 0.43371892],
              [0.544003, 0.42719197],
              [0.5623568, 0.41352543],
              [0.58538425, 0.4136302],
              [0.6036947, 0.42569503],
              [0.5857447, 0.4339628],
              [0.5639519, 0.43397212],
              [0.42937383, 0.5978748],
              [0.45566574, 0.5848662],
              [0.48228574, 0.5772575],
              [0.50101954, 0.5808408],
              [0.51930374, 0.57733876],
              [0.5428822, 0.58488935],
              [0.5642669, 0.59790665],
              [0.5427994, 0.62303317],
              [0.5209273, 0.63482875],
              [0.49990356, 0.6377742],
              [0.47819215, 0.63504726],
              [0.4538526, 0.6232618],
              [0.44238397, 0.5995862],
              [0.48043576, 0.59368575],
              [0.5006087, 0.59484065],
              [0.5201079, 0.5936368],
              [0.5519401, 0.59959716],
              [0.5202506, 0.61122304],
              [0.500065, 0.6136451],
              [0.47921613, 0.61147493]], dtype=np.float32)


def molded_box_to_quant(box):
    # b x n x 4
    y1, x1, y2, x2 = tf.split(box, 4, axis=-1)
    cy = (y1 + y2) / 2.0
    cx = (x1 + x2) / 2.0
    logh = tf.log(tf.maximum(tf.abs(y2 - y1), 1e-6))
    logw = tf.log(tf.maximum(tf.abs(x2 - x1), 1e-6))
    qbox = tf.concat([cy, cx, logh, logw], axis=-1)
    return qbox


def quant_box_to_mold(box):
    # b x n x 4
    cy, cx, logh, logw = tf.split(box, 4, axis=-1)
    hh = tf.exp(logh) / 2.0
    ww = tf.exp(logw) / 2.0
    y1 = cy - hh
    y2 = cy + hh
    x1 = cx - ww
    x2 = cx + ww
    mbox = tf.concat([y1, x1, y2, x2], axis=-1)
    return mbox


def compute_box_deform(box_from, box_to):
    y1_from, x1_from, y2_from, x2_from = tf.split(box_from, 4, axis=-1)
    cy_from = (y1_from + y2_from) / 2.0
    cx_from = (x1_from + x2_from) / 2.0
    h_from = tf.maximum(tf.abs(y2_from - y1_from), 1e-6)
    w_from = tf.maximum(tf.abs(x2_from - x1_from), 1e-6)
    logh_from = tf.log(h_from)
    logw_from = tf.log(w_from)

    y1_to, x1_to, y2_to, x2_to = tf.split(box_to, 4, axis=-1)
    cy_to = (y1_to + y2_to) / 2.0
    cx_to = (x1_to + x2_to) / 2.0
    h_to = tf.maximum(tf.abs(y2_to - y1_to), 1e-6)
    w_to = tf.maximum(tf.abs(x2_to - x1_to), 1e-6)
    logh_to = tf.log(h_to)
    logw_to = tf.log(w_to)

    d_cy = (cy_to - cy_from) / h_from
    d_cx = (cx_to - cx_from) / w_from
    d_h = logh_to - logh_from
    d_w = logw_to - logw_from
    return tf.concat([d_cy, d_cx, d_h, d_w], axis=-1)


def apply_box_deform(box_from, deform):
    d_cy, d_cx, d_h, d_w = tf.split(deform, 4, axis=-1)

    y1_from, x1_from, y2_from, x2_from = tf.split(box_from, 4, axis=-1)
    cy_from = (y1_from + y2_from) / 2.0
    cx_from = (x1_from + x2_from) / 2.0
    h_from = tf.maximum(tf.abs(y2_from - y1_from), 1e-6)
    w_from = tf.maximum(tf.abs(x2_from - x1_from), 1e-6)
    logh_from = tf.log(h_from)
    logw_from = tf.log(w_from)

    cy_to = cy_from + h_from * d_cy
    cx_to = cx_from + w_from * d_cx
    h_to = tf.exp(logh_from + d_h)
    w_to = tf.exp(logw_from + d_w)

    y1_to = cy_to - h_to / 2.0
    y2_to = cy_to + h_to / 2.0
    x1_to = cx_to - w_to / 2.0
    x2_to = cx_to + w_to / 2.0
    return tf.concat([y1_to, x1_to, y2_to, x2_to], axis=-1)


def deconv_hist(x, *args, **kwargs):
    conv = KL.Conv2DTranspose(*args, **kwargs)
    x = conv(x)
    tf.summary.histogram(f'{conv.name}_weights', conv.weights[0])
    return x


def fast_hist(a, b, n):
    ''' 
    fast histogram calculation
    ---
    * a, b: label ids, a.shape == b.shape
    * n: number of classes to measure
    '''
    k = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    return np.bincount(
        n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def boxfilter(img, r):
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)

    imCum = np.cumsum(img, 0)
    imDst[0: r + 1, :] = imCum[r: 2 * r + 1, :]
    imDst[r + 1: rows - r, :] = imCum[2 * r + 1: rows, :] - \
        imCum[0: rows - 2 * r - 1, :]
    imDst[rows - r: rows, :] = np.tile(imCum[rows - 1, :],
                                       [r, 1]) - imCum[rows - 2 * r - 1: rows - r - 1, :]

    imCum = np.cumsum(imDst, 1)
    imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
    imDst[:, r + 1: cols - r] = imCum[:, 2 * r +
                                      1: cols] - imCum[:, 0: cols - 2 * r - 1]
    imDst[:, cols - r: cols] = np.tile(imCum[:, cols - 1],
                                       [r, 1]).T - imCum[:, cols - 2 * r - 1: cols - r - 1]

    return imDst


def guidedfilter(I, p, r, eps):
    (rows, cols) = I.shape
    N = boxfilter(np.ones([rows, cols]), r)

    meanI = boxfilter(I, r) / N
    meanP = boxfilter(p, r) / N
    meanIp = boxfilter(I * p, r) / N
    covIp = meanIp - meanI * meanP

    meanII = boxfilter(I * I, r) / N
    varI = meanII - meanI * meanI

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = boxfilter(a, r) / N
    meanB = boxfilter(b, r) / N

    q = meanA * I + meanB
    return q


def trimap_from_alpha(masks, k_size=5):
    # masks: [batch, num_masks, h, w]
    original_shape = tf.shape(masks)
    masks = tf.reshape(
        masks, [-1, original_shape[2], original_shape[3], 1])
    dilated = tf.nn.dilation2d(
        masks, tf.ones([k_size, k_size, 1], tf.float32), strides=[
            1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    erosed = tf.nn.erosion2d(
        masks, tf.ones([k_size, k_size, 1], tf.float32), strides=[
            1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    trimap = tf.reshape((dilated + erosed) / 2.0, original_shape)
    return trimap


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(
        nb_filter1, (1, 1), name=conv_name_base + '2a',
        use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(
        nb_filter2, (kernel_size, kernel_size),
        padding='same',
        name=conv_name_base + '2b',
        use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(
        nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               use_bias=True,
               prefix=''):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = prefix + 'res' + str(stage) + block + '_branch'
    bn_name_base = prefix + 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(
        nb_filter1, (1, 1),
        strides=strides,
        name=conv_name_base + '2a',
        use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(
        nb_filter2, (kernel_size, kernel_size),
        padding='same',
        name=conv_name_base + '2b',
        use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(
        nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(
        nb_filter3, (1, 1),
        strides=strides,
        name=conv_name_base + '1',
        use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(axis=3, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


def relu6(x):
    return K.relu(x, max_value=6)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), layer_name='conv1'):
    """Adds an initial convolution layer (with batch normalization and relu6).

    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = KL.Conv2D(filters, kernel,
                  padding='same',
                  use_bias=False,
                  strides=strides,
                  name=layer_name)(inputs)
    x = KL.BatchNormalization(axis=channel_axis, name=layer_name + '_bn')(x)
    return KL.Activation(relu6, name=layer_name + '_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating the block number.

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = keras.applications.mobilenet.DepthwiseConv2D((3, 3),
                                                     padding='same',
                                                     depth_multiplier=depth_multiplier,
                                                     strides=strides,
                                                     use_bias=False,
                                                     name='conv_dw_%d' % block_id)(inputs)
    x = KL.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = KL.Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = KL.Conv2D(pointwise_conv_filters, (1, 1),
                  padding='same',
                  use_bias=False,
                  strides=(1, 1),
                  name='conv_pw_%d' % block_id)(x)
    x = KL.BatchNormalization(
        axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return KL.Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


class DepthwiseConvBlock(object):
    def __init__(self,
                 pointwise_conv_filters, alpha,
                 depth_multiplier=1, strides=(1, 1),
                 name=None,
                 last_act=relu6):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)
        self.conv_dw = keras.applications.mobilenet.DepthwiseConv2D(
            (3, 3),
            padding='same',
            depth_multiplier=depth_multiplier,
            strides=strides,
            use_bias=False,
            name=f'{name}_conv_dw')
        self.conv_dw_bn = KL.BatchNormalization(
            axis=channel_axis, name=f'{name}_conv_dw_bn')
        self.conv_dw_relu = KL.Activation(relu6, name=f'{name}_conv_dw_relu')
        self.conv_pw = KL.Conv2D(pointwise_conv_filters, (1, 1),
                                 padding='same',
                                 use_bias=False,
                                 strides=(1, 1),
                                 name=f'{name}_conv_pw')
        self.conv_pw_bn = KL.BatchNormalization(
            axis=channel_axis, name=f'{name}_conv_pw_bn')

        self.conv_pw_relu = KL.Activation(
            last_act, name=f'{name}_conv_pw_relu')

    def __call__(self, x):
        x = self.conv_dw(x)
        x = self.conv_dw_bn(x)
        x = self.conv_dw_relu(x)
        x = self.conv_pw(x)
        x = self.conv_pw_bn(x)
        x = self.conv_pw_relu(x)
        return x


def HairMatte_graph(input_image,
                    alpha=1.0,
                    depth_multiplier=1):
    x = _conv_block(input_image, 32, alpha, strides=(2, 2), layer_name='conv1')
    p1 = x = _depthwise_conv_block(
        x, 64, alpha, depth_multiplier, strides=(1, 1), block_id=1)
    x = _depthwise_conv_block(
        x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    p2 = x = _depthwise_conv_block(
        x, 128, alpha, depth_multiplier, strides=(1, 1), block_id=3)
    x = _depthwise_conv_block(
        x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    p3 = x = _depthwise_conv_block(
        x, 256, alpha, depth_multiplier, strides=(1, 1), block_id=5)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(1, 1), block_id=7)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(1, 1), block_id=8)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(1, 1), block_id=9)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(1, 1), block_id=10)
    p4 = x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(1, 1), block_id=11)
    x = _depthwise_conv_block(
        x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(
        x, 1024, alpha, depth_multiplier, strides=(1, 1), block_id=13)
    x = BilinearUpSampling2D((2, 2))(x)
    p4 = _conv_block(p4, 1024, alpha, kernel=(
        1, 1), strides=(1, 1), layer_name='p_conv1')
    x = KL.Add()([x, p4])
    x = keras.applications.mobilenet.DepthwiseConv2D(
        (3, 3), padding='same', depth_multiplier=depth_multiplier, strides=(1, 1), use_bias=False, name='1_conv_dw')(x)
    x = _conv_block(x, 64, alpha, kernel=(1, 1),
                    strides=(1, 1), layer_name='p_conv2')
    x = KL.Activation('relu')(x)
    x = BilinearUpSampling2D((2, 2))(x)
    p3 = _conv_block(p3, 64, alpha, kernel=(
        1, 1), strides=(1, 1), layer_name='p_conv3')
    x = KL.Add()([x, p3])
    x = keras.applications.mobilenet.DepthwiseConv2D(
        (3, 3), padding='same', depth_multiplier=depth_multiplier, strides=(1, 1), use_bias=False, name='2_conv_dw')(x)
    x = _conv_block(x, 64, alpha, kernel=(1, 1),
                    strides=(1, 1), layer_name='p_conv4')
    x = KL.Activation('relu')(x)
    x = BilinearUpSampling2D((2, 2))(x)
    p2 = _conv_block(p2, 64, alpha, kernel=(
        1, 1), strides=(1, 1), layer_name='p_conv5')
    x = KL.Add()([x, p2])
    x = keras.applications.mobilenet.DepthwiseConv2D(
        (3, 3), padding='same', depth_multiplier=depth_multiplier, strides=(1, 1), use_bias=False, name='3_conv_dw')(x)
    x = _conv_block(x, 64, alpha, kernel=(1, 1),
                    strides=(1, 1), layer_name='p_conv6')
    x = KL.Activation('relu')(x)
    x = BilinearUpSampling2D((2, 2))(x)
    p1 = _conv_block(p1, 64, alpha, kernel=(
        1, 1), strides=(1, 1), layer_name='p_conv7')
    x = KL.Add()([x, p1])
    x = keras.applications.mobilenet.DepthwiseConv2D(
        (3, 3), padding='same', depth_multiplier=depth_multiplier, strides=(1, 1), use_bias=False, name='4_conv_dw')(x)
    x = _conv_block(x, 64, alpha, kernel=(1, 1),
                    strides=(1, 1), layer_name='p_conv8')
    x = KL.Activation('relu')(x)
    x = BilinearUpSampling2D((2, 2))(x)
    x = keras.applications.mobilenet.DepthwiseConv2D(
        (3, 3), padding='same', depth_multiplier=depth_multiplier, strides=(1, 1), use_bias=False, name='5_conv_dw')(x)
    x = _conv_block(x, 64, alpha, kernel=(1, 1),
                    strides=(1, 1), layer_name='p_conv9')
    x = KL.Activation('relu')(x)
    x = _conv_block(x, 2, alpha, kernel=(1, 1),
                    strides=(1, 1), layer_name='p_conv10')

    return x


def mobilenet_graph(input_image,
                    alpha=1.0,
                    depth_multiplier=1,
                    dropout=1e-3):
    x = _conv_block(input_image, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    return x


def SubpixelConv2D(input_shape, scale=4):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return KL.Lambda(subpixel, output_shape=subpixel_shape, name='subpixel')


############################################################
#  VGG Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py


def resize_images_bilinear(X,
                           height_factor=1,
                           width_factor=1,
                           target_height=None,
                           target_width=None,
                           data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(
                np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(
                np.array([height_factor, width_factor]).astype('int32'))
        X = K.permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = K.permute_dimensions(X, [0, 3, 1, 2])
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor,
                         original_shape[3] * width_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(
                np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(
                np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor,
                         original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)


class BilinearUpSampling2D(KL.Layer):
    def __init__(self,
                 size=(1, 1),
                 target_size=None,
                 data_format='default',
                 **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'
                               }, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [KL.InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2]
                        if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3]
                         if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0], input_shape[1], width, height)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1]
                        if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2]
                         if input_shape[2] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0], width, height, input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(
                x,
                target_height=self.target_size[0],
                target_width=self.target_size[1],
                data_format=self.data_format)
        else:
            return resize_images_bilinear(
                x,
                height_factor=self.size[0],
                width_factor=self.size[1],
                data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def vgg16_graph(input_image):
    x = input_image
    # Block 1
    x = KL.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = KL.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    pool1 = C1 = x
    print('C1:{}'.format(C1._keras_shape))

    # Block 2
    x = KL.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = KL.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    pool2 = C2 = x
    print('C2:{}'.format(C2._keras_shape))

    # Block 3
    x = KL.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = KL.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = KL.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    pool3 = C3 = x
    print('C3:{}'.format(C3._keras_shape))

    # Block 4
    x = KL.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = KL.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = KL.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    pool4 = C4 = x
    print('C4:{}'.format(C4._keras_shape))

    # Block 5
    x = KL.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = KL.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = KL.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)  # TODO
    pool5 = C5 = x
    print('C5:{}'.format(C5._keras_shape))

    return [C1, C2, C3, C4, C5]


class PyramidROIAlignAll(KE.Layer):
    """Implements ROI Pooling on all levels of the feature pyramid.

    Params:
    - pool_shape: [height, width] of the output pooled regions.

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels * num_feature_maps].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlignAll, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]
        batch = tf.shape(boxes)[0]
        num_boxes = tf.shape(boxes)[1]

        # [batch * num_boxes, 4]
        boxes = tf.reshape(boxes, [-1, 4])
        # [batch, num_boxes]
        box_inds = tf.tile(
            tf.expand_dims(tf.range(batch), axis=-1), multiples=[1, num_boxes])
        # batch * num_boxes
        # box_inds = tf.Print(box_inds, [box_inds[0], box_inds[1]])
        box_inds = tf.reshape(box_inds, [-1])

        pooled = []
        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        for feature_map in inputs[1:]:
            # print('feature_map.shape: {}'.format(feature_map._keras_shape))
            # Crop and Resize
            # get: [batch * num_boxes, pool_shape[0], pool_shape[1], channels]
            pooled.append(
                tf.image.crop_and_resize(
                    feature_map,
                    boxes,
                    box_inds,
                    self.pool_shape,
                    method="bilinear"))
            # print(pooled[-1].shape)

        # Pack pooled features into one tensor
        # [batch * num_boxes, p0, p1, channels]
        # print(pooled.shape)
        pooled = tf.concat(pooled, axis=-1)
        # print('pooled.shape: {}'.format(pooled.shape))
        output_shape = [batch, num_boxes] + pooled.shape.as_list()[1:]
        pooled = tf.reshape(pooled, output_shape)
        return pooled

    def compute_output_shape(self, input_shape):
        # return input_shape[0][:2] + self.pool_shape + (input_shape[1][-1] * 4,
        #                                                )  # concat 4 features
        num_channels = 0
        for s in input_shape[1:]:
            num_channels += s[-1]
        return input_shape[0][:2] + self.pool_shape + (num_channels, )


# class CRFIterate(KE.Layer):
#     def __init__(self, num_iters, num_filters, **kwargs):
#         super(CRFIterate, self).__init__(**kwargs)
#         self.num_iters = num_iters
#         self.num_filters = num_filters
#         self.conv = KL.Conv2D(num_filters, (1, 1), padding='same',
#                               name='CRFIterate_Conv')

#     def call(self, inputs):
#         # (batch, h, w, d)
#         # (batch, h, w, 4)
#         psi, gates = inputs

#         batch = tf.shape(psi)[0]
#         h = tf.shape(psi)[1]
#         w = tf.shape(psi)[2]

#         b, y, x = tf.meshgrid(tf.range(batch), tf.range(h),
#                               tf.range(w), indexing='ij')
#         y1 = tf.maximum(y - 1, 0)
#         y2 = tf.minimum(y + 1, h - 1)
#         x1 = tf.maximum(x - 1, 0)
#         x2 = tf.minimum(x + 1, w - 1)

#         Q = psi

#         for _ in range(self.num_iters):
#             Q1 = tf.gather_nd(Q, tf.stack([b, y1, x1], axis=-1))
#             Q2 = tf.gather_nd(Q, tf.stack([b, y1, x2], axis=-1))
#             Q3 = tf.gather_nd(Q, tf.stack([b, y2, x1], axis=-1))
#             Q4 = tf.gather_nd(Q, tf.stack([b, y2, x2], axis=-1))
#             Q = Q1 * tf.expand_dims(gates[:, :, :, 0], axis=-1) + \
#                 Q2 * tf.expand_dims(gates[:, :, :, 1], axis=-1) + \
#                 Q3 * tf.expand_dims(gates[:, :, :, 2], axis=-1) + \
#                 Q4 * tf.expand_dims(gates[:, :, :, 3], axis=-1)

#             Q = self.conv(Q)
#             Q = - psi - Q
#             Q = tf.nn.softmax(Q, dim=-1)

#         return Q

#     def compute_output_shape(self, input_shapes):
#         assert input_shapes[0][:-1] == input_shapes[1][:-1]
#         return input_shapes[0][:-1] + (self.num_filters,)


class CRFIterateCorrect(KL.Layer):
    def __init__(self, num_iters, num_filters, **kwargs):
        super(CRFIterateCorrect, self).__init__(**kwargs)
        self.num_iters = num_iters
        self.num_filters = num_filters
        self.kernel_initializer = keras.initializers.get('glorot_uniform')
        self.bias_initializer = keras.initializers.get('zeros')

    def build(self, input_shape):
        kernel_shape = (1, 1) + (self.num_filters, self.num_filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=None,
                                      constraint=None)
        self.bias = self.add_weight(
            shape=(self.num_filters,),
            initializer=self.bias_initializer,
            name='bias',
            regularizer=None,
            constraint=None)
        self.built = True

    def call(self, inputs):
        # (batch, h, w, d)
        # (batch, h, w, 4)
        psi, gates = inputs

        batch = tf.shape(psi)[0]
        h = tf.shape(psi)[1]
        w = tf.shape(psi)[2]

        b, y, x = tf.meshgrid(tf.range(batch), tf.range(h),
                              tf.range(w), indexing='ij')
        y1 = tf.maximum(y - 1, 0)
        y2 = tf.minimum(y + 1, h - 1)
        x1 = tf.maximum(x - 1, 0)
        x2 = tf.minimum(x + 1, w - 1)

        Q = psi

        for _ in range(self.num_iters):
            Q1 = tf.gather_nd(Q, tf.stack([b, y1, x1], axis=-1))
            Q2 = tf.gather_nd(Q, tf.stack([b, y1, x2], axis=-1))
            Q3 = tf.gather_nd(Q, tf.stack([b, y2, x1], axis=-1))
            Q4 = tf.gather_nd(Q, tf.stack([b, y2, x2], axis=-1))
            Q = Q1 * tf.expand_dims(gates[:, :, :, 0], axis=-1) + \
                Q2 * tf.expand_dims(gates[:, :, :, 1], axis=-1) + \
                Q3 * tf.expand_dims(gates[:, :, :, 2], axis=-1) + \
                Q4 * tf.expand_dims(gates[:, :, :, 3], axis=-1)

            Q = K.conv2d(Q, self.kernel, strides=(1, 1), padding='same')
            Q = K.bias_add(Q, self.bias)

            Q = - psi - Q
            Q = tf.nn.softmax(Q, dim=-1)

        return Q

    def compute_output_shape(self, input_shapes):
        psi_shape = input_shapes[0]
        return psi_shape


class GaussionDistribute(KE.Layer):
    def __init__(self, h, w, sigma, **kwargs):
        super(GaussionDistribute, self).__init__(**kwargs)
        self.h = h
        self.w = w
        self.sigma = sigma
        self.Y, self.X = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
        self.X = tf.cast(self.X / (self.w - 1), tf.float32)
        self.Y = tf.cast(self.Y / (self.h - 1), tf.float32)

    def call(self, input):
        batch_size = tf.shape(input)[0]
        npts = tf.shape(input)[1]

        # [batch_size, h, w, npts]
        Xs = tf.tile(
            tf.expand_dims(tf.expand_dims(self.X, axis=-1), axis=0),
            multiples=[batch_size, 1, 1, npts])
        Ys = tf.tile(
            tf.expand_dims(tf.expand_dims(self.Y, axis=-1), axis=0),
            multiples=[batch_size, 1, 1, npts])

        # [batch_size, h, w, npts]
        pts_xs = tf.tile(
            tf.expand_dims(tf.expand_dims(input[:, :, 0], 1), 1),
            multiples=[1, self.h, self.w, 1])
        pts_ys = tf.tile(
            tf.expand_dims(tf.expand_dims(input[:, :, 1], 1), 1),
            multiples=[1, self.h, self.w, 1])

        squared_distances = tf.square(Ys - pts_ys) + tf.square(Xs - pts_xs)
        return tf.exp(-squared_distances / 2.0 / (self.sigma**2))

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        npts = input_shape[1]
        return (batch_size, self.h, self.w, npts)


class HeatMapToCoord(KE.Layer):
    ''' [batch, h, w, npts] -> [batch, npts, 2]
    '''

    def __init__(self, **kwargs):
        super(HeatMapToCoord, self).__init__(**kwargs)

    def call(self, input):
        # [batch, h, w, npts]
        batch_size = tf.shape(input)[0]
        h = tf.shape(input)[1]
        w = tf.shape(input)[2]
        npts = tf.shape(input)[3]

        Y, X = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
        X = tf.cast(X, tf.float32)
        Y = tf.cast(Y, tf.float32)
        h = tf.cast(h, tf.float32)
        w = tf.cast(w, tf.float32)

        X = X / (w - 1.0)
        Y = Y / (h - 1.0)

        # [batch_size, h, w, npts]
        Xs = tf.tile(
            tf.expand_dims(tf.expand_dims(X, axis=-1), axis=0),
            multiples=[batch_size, 1, 1, npts])
        Ys = tf.tile(
            tf.expand_dims(tf.expand_dims(Y, axis=-1), axis=0),
            multiples=[batch_size, 1, 1, npts])

        # [batch, npts]
        denorm = tf.reduce_mean(tf.reduce_mean(input, axis=1), axis=1)

        # [batch, npts]
        pts_xs = tf.reduce_mean(tf.reduce_mean(Xs * input, axis=1), axis=1)
        pts_ys = tf.reduce_mean(tf.reduce_mean(Ys * input, axis=1), axis=1)

        pts_xs = pts_xs / denorm
        pts_ys = pts_ys / denorm

        # [batch, npts, (x, y)]
        pts = tf.stack([pts_xs, pts_ys], axis=-1)
        return pts

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        npts = input_shape[3]
        return (batch_size, npts, 2)


class Bottleneck(KE.Layer):
    expansion = 2

    def __init__(self, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # self.bn1 = nn.BatchNorm2d(inplanes)
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(
        #     planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        # self.bn3 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.downsample = downsample
        # self.stride = stride
        self.bn1 = KL.BatchNormalization(axis=-1)
        self.conv1 = KL.Conv2D(planes, kernel_size=1, use_bias=True)
        self.bn2 = KL.BatchNormalization(axis=-1)
        self.conv2 = KL.Conv2D(
            planes,
            kernel_size=3,
            strides=stride,
            padding='same',
            use_bias=True)
        self.bn3 = KL.BatchNormalization(axis=-1)
        self.conv3 = KL.Conv2D(planes * 2, kernel_size=1, use_bias=True)
        self.relu = KL.Activation('relu')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(KE.Layer):
    def __init__(self, block, num_blocks, planes, depth, **kwargs):
        super(Hourglass, self).__init__(**kwargs)
        self.depth = depth
        self.block = block
        self.upsample = KL.UpSampling2D((2, 2))
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return KM.Sequential(layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(res)
        return hg

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = KL.MaxPool2D((2, 2), strides=2, padding='valid')(x)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = self.upsample(low3)
        out = KL.Add()([up1, up2])
        return out

    def call(self, input):
        return self._hour_glass_forward(self.depth, input)


class HourglassNet(KE.Layer):
    '''Hourglass model from Newell et al ECCV 2016'''

    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = KL.Conv2D(
            self.inplanes,
            kernel_size=7,
            strides=2,
            padding='same',
            use_bias=True)
        self.bn1 = KL.BatchNormalization(axis=-1)
        self.relu = KL.Activation('relu')
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = KL.MaxPool2D(2, 2)

        # build hourglass modules
        # ch = self.num_feats * block.expansion
        # hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        # for i in range(num_stacks):
        #     hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
        #     res.append(self._make_residual(block, self.num_feats, num_blocks))
        #     fc.append(self._make_fc(ch, ch))
        #     score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
        #     if i < num_stacks - 1:
        #         fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
        #         score_.append(
        #             nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))

        ch = self.num_feats * block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(KL.Conv2D(num_classes, kernel_size=1, use_bias=True))
            if i < num_stacks - 1:
                fc_.append(KL.Conv2D(ch, kernel_size=1, use_bias=True))
                score_.append(KL.Conv2D(ch, kernel_size=1, use_bias=True))

        self.hg = hg
        self.res = res
        self.fc = fc
        self.score = score
        self.fc_ = fc_
        self.score_ = score_

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = KM.Sequential([
                KL.Conv2D(
                    planes * block.expansion,
                    kernel_size=1,
                    strides=stride,
                    use_bias=True)
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return KM.Sequential(layers)

    def _make_fc(self, inplanes, outplanes):
        bn = KL.BatchNormalization(axis=-1)
        conv = KL.Conv2D(outplanes, kernel_size=1, use_bias=True)
        return KM.Sequential([conv, bn, self.relu])

    def call(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = KL.Add()([x, fc_, score_])

        return out


def build_fpn(feature_maps):
    C1, C2, C3, C4, C5 = feature_maps
    # Top-down Layers
    P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)
    ])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)
    ])
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)
    ])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    return P2, P3, P4, P5, P6


def build_fpn_dw_conv(feature_maps):
    C1, C2, C3, C4, C5 = feature_maps
    # Top-down Layers
    P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)
    ])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)
    ])
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)
    ])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = DepthwiseConvBlock(256, 1, 1, (3, 3), name="fpn_p2")(P2)
    P3 = DepthwiseConvBlock(256, 1, 1, (3, 3), name="fpn_p3")(P3)
    P4 = DepthwiseConvBlock(256, 1, 1, (3, 3), name="fpn_p4")(P4)
    P5 = DepthwiseConvBlock(256, 1, 1, (3, 3), name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    return P2, P3, P4, P5, P6

# def build_dw_fpn(feature_maps):
#     C1, C2, C3, C4, C5 = feature_maps
#     # Top-down Layers
#     P5 = DepthwiseConvBlock(256, 1, 1, strides=(1, 1), name='fpn_c5p5')(C5)
#     P4 = KL.Add(name="fpn_p4add")([
#         KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
#         DepthwiseConvBlock(256, 1, 1, strides=(1, 1), name='fpn_c4p4')(C4)
#     ])
#     P3 = KL.Add(name="fpn_p3add")([
#         KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
#         DepthwiseConvBlock(256, 1, 1, strides=(1, 1), name='fpn_c3p3')(C3)
#     ])
#     P2 = KL.Add(name="fpn_p2add")([
#         KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
#         DepthwiseConvBlock(256, 1, 1, strides=(1, 1), name='fpn_c2p2')(C2)
#     ])
#     # Attach 3x3 conv to all P layers to get the final feature maps.
#     P2 = DepthwiseConvBlock(256, 1, 1, strides=(
#         3, 3), padding="SAME", name="fpn_p2")(P2)
#     P3 = DepthwiseConvBlock(256, 1, 1, strides=(
#         3, 3), padding="SAME", name="fpn_p3")(P3)
#     P4 = DepthwiseConvBlock(256, 1, 1, strides=(
#         3, 3), padding="SAME", name="fpn_p4")(P4)
#     P5 = DepthwiseConvBlock(256, 1, 1, strides=(
#         3, 3), padding="SAME", name="fpn_p5")(P5)
#     # P6 is used for the 5th anchor scale in RPN. Generated by
#     # subsampling from P5 with stride of 2.
#     P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
#     return P2, P3, P4, P5, P6


def build_mask_heads(rois,
                     feature_maps,
                     pool_size,
                     deconv_num=1,
                     head_class_nums=None,
                     bn_axis=2,
                     conv_kernel_size=3,
                     deconv_kernel_size=2):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps.
    pool_size: The width of the square feature map generated from ROI Pooling.

    Returns:
    fg_masks: [batch, num_fg_masks, height, width]
    bg_masks: [batch, num_bg_masks, height, width]
    """

    # ROI Pooling
    aligned = PyramidROIAlignAll(
        [pool_size, pool_size], name="roi_align_mask")([rois] + feature_maps)
    # print(aligned._keras_shape)

    num_rois = rois.shape[1].value
    fg_masks = [None] * num_rois
    bg_masks = [None] * num_rois

    if head_class_nums is None:
        head_class_nums = [1] * num_rois
    assert len(head_class_nums) == num_rois

    def _slice_lambda(index):
        return lambda x: x[:, index, :, :, :]

    for i in range(num_rois):
        # with tf.device(get_gpu_name_fn(i + 1)):
        x = KL.Lambda(_slice_lambda(i))(aligned)
        # print(x._keras_shape)

        x = KL.Conv2D(
            256, (conv_kernel_size, conv_kernel_size), padding="same", name="mrcnn_mask_conv1_%d" % i)(x)
        x = BatchNorm(axis=bn_axis, name='mrcnn_mask_bn1_%d' % i)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(
            256, (conv_kernel_size, conv_kernel_size), padding="same", name="mrcnn_mask_conv2_%d" % i)(x)
        x = BatchNorm(axis=bn_axis, name='mrcnn_mask_bn2_%d' % i)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(
            256, (conv_kernel_size, conv_kernel_size), padding="same", name="mrcnn_mask_conv3_%d" % i)(x)
        x = BatchNorm(axis=bn_axis, name='mrcnn_mask_bn3_%d' % i)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(
            256, (conv_kernel_size, conv_kernel_size), padding="same", name="mrcnn_mask_conv4_%d" % i)(x)
        x = BatchNorm(axis=bn_axis, name='mrcnn_mask_bn4_%d' % i)(x)
        x = KL.Activation('relu')(x)

        if deconv_num == 1:  # to be compatible with previous trained models
            x = KL.Conv2DTranspose(
                256, (deconv_kernel_size, deconv_kernel_size),
                strides=2,
                activation="relu",
                name="mrcnn_mask_deconv_%d" % i)(x)
        else:
            for k in range(deconv_num):
                x = KL.Conv2DTranspose(
                    256, (deconv_kernel_size, deconv_kernel_size),
                    strides=2,
                    activation="relu",
                    name="mrcnn_mask_deconv%d_%d" % (k + 1, i))(x)

        num_classes_this_head = head_class_nums[i]
        assert num_classes_this_head > 0

        x = KL.Conv2D(
            1 + num_classes_this_head, (1, 1), strides=1,
            activation='linear')(x)
        x = KL.Lambda(
            lambda xx: tf.nn.softmax(xx, dim=-1),
            name="mrcnn_fullmask_%d" % i)(x)

        # [batch, height, width, num_classes]
        # [batch, num_classes, height, width]
        fg_masks[i] = KL.Lambda(
            lambda xx: tf.transpose(xx[:, :, :, 1:], [0, 3, 1, 2]),
            name='mrcnn_fg_mask_%d' % i)(x)

        # [batch, height, width]
        bg_masks[i] = KL.Lambda(
            lambda xx: xx[:, :, :, 0], name='mrcnn_bg_mask_%d' % i)(x)

        print(fg_masks[i]._keras_shape, fg_masks[i].shape,
              bg_masks[i]._keras_shape, bg_masks[i].shape)

    fg_masks = KL.Lambda(
        lambda xx: tf.concat(xx, axis=1), name='mrcnn_fg_masks')(fg_masks)
    bg_masks = KL.Lambda(
        lambda xx: tf.stack(xx, axis=1), name='mrcnn_bg_masks')(bg_masks)
    return fg_masks, bg_masks


def build_mask_heads_with_transfer_refine(rois,
                                          feature_maps,
                                          pool_size,
                                          deconv_num=1,
                                          head_class_nums=None,
                                          bn_axis=2,
                                          transfer_refine=None):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps.
    pool_size: The width of the square feature map generated from ROI Pooling.

    Returns:
    fg_masks: [batch, num_fg_masks, height, width]
    bg_masks: [batch, num_bg_masks, height, width]
    """

    # ROI Pooling
    aligned = PyramidROIAlignAll(
        [pool_size, pool_size], name="roi_align_mask")([rois] + feature_maps)
    # print(aligned._keras_shape)

    num_rois = rois.shape[1].value
    fg_masks = [None] * num_rois
    bg_masks = [None] * num_rois

    if head_class_nums is None:
        head_class_nums = [1] * num_rois
    assert len(head_class_nums) == num_rois

    def _slice_lambda(index):
        return lambda x: x[:, index, :, :, :]

    head_mask_features = [None] * num_rois
    for i in range(num_rois):
        # with tf.device(get_gpu_name_fn(i + 1)):
        x = KL.Lambda(_slice_lambda(i))(aligned)
        # print(x._keras_shape)

        x = KL.Conv2D(
            256, (3, 3), padding="same", name="mrcnn_mask_conv1_%d" % i)(x)
        x = BatchNorm(axis=bn_axis, name='mrcnn_mask_bn1_%d' % i)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(
            256, (3, 3), padding="same", name="mrcnn_mask_conv2_%d" % i)(x)
        x = BatchNorm(axis=bn_axis, name='mrcnn_mask_bn2_%d' % i)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(
            256, (3, 3), padding="same", name="mrcnn_mask_conv3_%d" % i)(x)
        x = BatchNorm(axis=bn_axis, name='mrcnn_mask_bn3_%d' % i)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(
            256, (3, 3), padding="same", name="mrcnn_mask_conv4_%d" % i)(x)
        x = BatchNorm(axis=bn_axis, name='mrcnn_mask_bn4_%d' % i)(x)
        x = KL.Activation('relu')(x)

        if deconv_num == 1:  # to be compatible with previous trained models
            x = KL.Conv2DTranspose(
                256, (2, 2),
                strides=2,
                activation="relu",
                name="mrcnn_mask_deconv_%d" % i)(x)
        else:
            for k in range(deconv_num):
                x = KL.Conv2DTranspose(
                    256, (2, 2),
                    strides=2,
                    activation="relu",
                    name="mrcnn_mask_deconv%d_%d" % (k + 1, i))(x)
        # [batch, h, w, 256]
        head_mask_features[i] = x

    mask_feature_size = pool_size * 2**deconv_num

    def _transfer_box_crop_lambda(from_ind, to_ind):
        # xx: [rois, from_features]
        return lambda xx: transfer_box_crop(
            xx[1], xx[0][:, from_ind, :], xx[0][:, to_ind, :],
            [mask_feature_size, mask_feature_size])

    if transfer_refine is not None:
        head_mask_features_after_transfer = [None] * num_rois
        for to_ind, from_inds in transfer_refine.items():
            to_features = [head_mask_features[to_ind]]
            for from_ind in from_inds:
                transfered_feature = KL.Lambda(
                    _transfer_box_crop_lambda(from_ind, to_ind))(
                        [rois, head_mask_features[from_ind]])
            to_features.append(transfered_feature)
            head_mask_features_after_transfer[to_ind] = KL.Concatenate(
                axis=-1, name='mrcnn_mask_feature_after_transfer_%d' %
                to_ind)(to_features)
    else:
        head_mask_features_after_transfer = head_mask_features

    for i in range(num_rois):
        x = head_mask_features[i]
        num_classes_this_head = head_class_nums[i]
        assert num_classes_this_head > 0

        x = KL.Conv2D(
            1 + num_classes_this_head, (1, 1), strides=1,
            activation='linear')(x)
        x = KL.Lambda(
            lambda xx: tf.nn.softmax(xx, dim=-1),
            name="mrcnn_fullmask_%d" % i)(x)

        # [batch, height, width, num_classes]
        # [batch, num_classes, height, width]
        fg_masks[i] = KL.Lambda(
            lambda xx: tf.transpose(xx[:, :, :, 1:], [0, 3, 1, 2]),
            name='mrcnn_fg_mask_%d' % i)(x)

        # [batch, height, width]
        bg_masks[i] = KL.Lambda(
            lambda xx: xx[:, :, :, 0], name='mrcnn_bg_mask_%d' % i)(x)

        print(fg_masks[i]._keras_shape, fg_masks[i].shape,
              bg_masks[i]._keras_shape, bg_masks[i].shape)

    if len(fg_masks) > 1:
        fg_masks = KL.Lambda(
            lambda xx: tf.concat(xx, axis=1), name='mrcnn_fg_masks')(fg_masks)
    else:
        fg_masks = KL.Lambda(lambda xx: xx, name='mrcnn_fg_masks')(fg_masks[0])

    if len(bg_masks) > 1:
        bg_masks = KL.Lambda(
            lambda xx: tf.stack(xx, axis=1), name='mrcnn_bg_masks')(bg_masks)
    else:
        bg_masks = KL.Lambda(
            lambda xx: tf.expand_dims(xx, axis=1),
            name='mrcnn_bg_masks')(bg_masks[0])
    return fg_masks, bg_masks


def build_mask_heads_shared(rois,
                            feature_maps,
                            pool_size,
                            deconv_num=1,
                            head_class_nums=None,
                            bn_axis=2):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps.
    pool_size: The width of the square feature map generated from ROI Pooling.

    Returns:
    fg_masks: [batch, num_fg_masks, height, width]
    bg_masks: [batch, num_bg_masks, height, width]
    """

    # ROI Pooling
    aligned = PyramidROIAlignAll(
        [pool_size, pool_size], name="roi_align_mask")([rois] + feature_maps)
    # print(aligned._keras_shape)

    num_rois = rois.shape[1].value
    fg_masks = [None] * num_rois
    bg_masks = [None] * num_rois

    if head_class_nums is None:
        head_class_nums = [1] * num_rois
    assert len(head_class_nums) == num_rois

    def _slice_lambda(index):
        return lambda x: x[:, index, :, :, :]

    neck = KM.Sequential(
        [
            KL.Conv2D(
                256, (3, 3),
                padding="same",
                name="mrcnn_mask_conv1",
                input_shape=(pool_size, pool_size, aligned.shape[-1].value)),
            BatchNorm(axis=bn_axis, name='mrcnn_mask_bn1'),
            KL.Activation('relu'),
            KL.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv2"),
            BatchNorm(axis=bn_axis, name='mrcnn_mask_bn2'),
            KL.Activation('relu'),
            KL.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv3"),
            BatchNorm(axis=bn_axis, name='mrcnn_mask_bn3'),
            KL.Activation('relu'),
            KL.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv4"),
            BatchNorm(axis=bn_axis, name='mrcnn_mask_bn4'),
            KL.Activation('relu')
        ],
        name='mrcnn_sequential_convs')

    for i in range(num_rois):
        # with tf.device(get_gpu_name_fn(i + 1)):
        x = KL.Lambda(_slice_lambda(i))(aligned)
        # print(x._keras_shape)

        x = neck(x)

        if deconv_num == 1:  # to be compatible with previous trained models
            x = KL.Conv2DTranspose(
                256, (2, 2),
                strides=2,
                activation="relu",
                name="mrcnn_mask_deconv_%d" % i)(x)
        else:
            for k in range(deconv_num):
                x = KL.Conv2DTranspose(
                    256, (2, 2),
                    strides=2,
                    activation="relu",
                    name="mrcnn_mask_deconv%d_%d" % (k + 1, i))(x)

        num_classes_this_head = head_class_nums[i]
        assert num_classes_this_head > 0

        x = KL.Conv2D(
            1 + num_classes_this_head, (1, 1), strides=1,
            activation='linear')(x)
        x = KL.Lambda(
            lambda xx: tf.nn.softmax(xx, dim=-1),
            name="mrcnn_fullmask_%d" % i)(x)

        # [batch, height, width, num_classes]
        # [batch, num_classes, height, width]
        fg_masks[i] = KL.Lambda(
            lambda xx: tf.transpose(xx[:, :, :, 1:], [0, 3, 1, 2]),
            name='mrcnn_fg_mask_%d' % i)(x)

        # [batch, height, width]
        bg_masks[i] = KL.Lambda(
            lambda xx: xx[:, :, :, 0], name='mrcnn_bg_mask_%d' % i)(x)

        print(fg_masks[i]._keras_shape, fg_masks[i].shape,
              bg_masks[i]._keras_shape, bg_masks[i].shape)

    fg_masks = KL.Lambda(
        lambda xx: tf.concat(xx, axis=1), name='mrcnn_fg_masks')(fg_masks)
    bg_masks = KL.Lambda(
        lambda xx: tf.stack(xx, axis=1), name='mrcnn_bg_masks')(bg_masks)
    return fg_masks, bg_masks


def box_crop(img, boxes, outshape):
    """
    Input:
    ---
    - img: (batch, h, w)
    - boxes: (batch, (y1, x1, y2, x2)), normalized coordinates
    """
    ratio_y, ratio_x = gen_box_cropping_grid(boxes, outshape)
    return bilinear_sample(img, ratio_y, ratio_x)


def inverse_box_crop(img, boxes, outshape):
    """
    Input:
    ---
    - img: (batch, h, w)
    - boxes: (batch, (y1, x1, y2, x2)), normalized coordinates
    """
    ratio_y, ratio_x = gen_inverse_box_cropping_grid(boxes, outshape)
    # print('ratio_y.shape={}'.format(ratio_y.shape))
    return bilinear_sample(img, ratio_y, ratio_x)


def transfer_box_crop(img, boxes_from, boxes_to, outshape):
    ratio_y, ratio_x = gen_transfer_box_cropping_grid(boxes_from, boxes_to,
                                                      outshape)
    print('img.shape={}, ratio_y.shape={}, ratio_x.shape={}'.format(
        img.shape, ratio_y.shape, ratio_x.shape))
    return bilinear_sample(img, ratio_y, ratio_x)


def gen_box_cropping_grid(boxes, outshape):
    """
    Input:
    ---
    - boxes: (batch, (y1, x1, y2, x2)), normalized coordinates

    Returns:
    ---
    - map_y, map_x: (batch, outshape[0], outshape[1]), coordinates in [0, 1]
    """
    h, w = outshape[:2]
    batch_size = tf.shape(boxes)[0]

    # each of (batch, 1)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)

    # now each of (batch, h, w)
    y1 = tf.tile(tf.reshape(y1, shape=(-1, 1, 1)), [1, h, w])
    x1 = tf.tile(tf.reshape(x1, shape=(-1, 1, 1)), [1, h, w])
    y2 = tf.tile(tf.reshape(y2, shape=(-1, 1, 1)), [1, h, w])
    x2 = tf.tile(tf.reshape(x1, shape=(-1, 1, 1)), [1, h, w])

    # (h, w)
    map_y, map_x = tf.meshgrid(
        tf.range(h, dtype=tf.float32),
        tf.range(w, dtype=tf.float32),
        indexing='ij')

    # now each of (batch, h, w)
    map_y = tf.tile(tf.expand_dims(map_y, 0), [batch_size, 1, 1])
    map_x = tf.tile(tf.expand_dims(map_x, 0), [batch_size, 1, 1])

    # to normalized coordinates [0, 1],
    # may contain invalid coordinates
    return map_y / h * (y2 - y1) + y1, map_x / w * (x2 - x1) + x1


def gen_inverse_box_cropping_grid(boxes, outshape):
    """
    Input:
    ---
    - boxes: (batch, (y1 x1 y2 x2)), normalized coordinates

    Returns:
    ---
    - map_y, map_x: (batch, outshape[0], outshape[1]), coordinates in [0, 1]
    """
    h, w = outshape[:2]
    # print('h=%d, w=%d' % (h, w))
    batch_size = tf.shape(boxes)[0]
    # print('boxes.shape={}'.format(boxes.shape))

    # each of (batch, 1)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # print('y1.shape={}, x1.shape={}'.format(y1.shape, x1.shape))

    # now each of (batch, h, w)
    y1 = tf.tile(tf.reshape(y1, shape=(-1, 1, 1)), [1, h, w])
    x1 = tf.tile(tf.reshape(x1, shape=(-1, 1, 1)), [1, h, w])
    y2 = tf.tile(tf.reshape(y2, shape=(-1, 1, 1)), [1, h, w])
    x2 = tf.tile(tf.reshape(x2, shape=(-1, 1, 1)), [1, h, w])
    # print('y1.shape={}, x1.shape={}'.format(y1.shape, x1.shape))
    # print('y2.shape={}, x2.shape={}'.format(y2.shape, x2.shape))

    # (h, w)
    map_y, map_x = tf.meshgrid(
        tf.range(h, dtype=tf.float32),
        tf.range(w, dtype=tf.float32),
        indexing='ij')
    # print('map_y.shape={}, map_x.shape={}'.format(map_y.shape, map_x.shape))

    # now each of (batch, h, w)
    map_y = tf.tile(tf.expand_dims(map_y, axis=0), [batch_size, 1, 1])
    map_x = tf.tile(tf.expand_dims(map_x, axis=0), [batch_size, 1, 1])
    # print('map_y.shape={}, map_x.shape={}'.format(map_y.shape, map_x.shape))

    # to normalized coordinates [0, 1],
    # may contain invalid coordinates
    # print('map_y / h shape={}'.format((map_y / h).shape))
    return (map_y / tf.cast(h, tf.float32) - y1) / (y2 - y1), \
        (map_x / tf.cast(w, tf.float32) - x1) / (x2 - x1)


def gen_transfer_box_cropping_grid(boxes_from, boxes_to, outshape):
    """
    Input:
    ---
    - boxes_from/to: (batch, (y1 x1 y2 x2)), normalized coordinates

    Returns:
    ---
    - map_y, map_x: (batch, outshape[0], outshape[1]), coordinates in [0, 1]
    """
    h, w = outshape[:2]
    # print('h=%d, w=%d' % (h, w))
    batch_size = tf.shape(boxes_from)[0]

    print('boxes_from.shape={}'.format(boxes_from.shape))
    print('boxes_to.shape={}'.format(boxes_to.shape))

    def _get_box_coords(boxes):
        # each of (batch, 1)
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
        # print('y1.shape={}, x1.shape={}'.format(y1.shape, x1.shape))

        # now each of (batch, h, w)
        y1 = tf.tile(tf.reshape(y1, shape=(-1, 1, 1)), [1, h, w])
        x1 = tf.tile(tf.reshape(x1, shape=(-1, 1, 1)), [1, h, w])
        y2 = tf.tile(tf.reshape(y2, shape=(-1, 1, 1)), [1, h, w])
        x2 = tf.tile(tf.reshape(x2, shape=(-1, 1, 1)), [1, h, w])
        # print('y1.shape={}, x1.shape={}'.format(y1.shape, x1.shape))
        # print('y2.shape={}, x2.shape={}'.format(y2.shape, x2.shape))
        return y1, x1, y2, x2

    y1_from, x1_from, y2_from, x2_from = _get_box_coords(boxes_from)
    y1_to, x1_to, y2_to, x2_to = _get_box_coords(boxes_to)

    # (h, w)
    map_y, map_x = tf.meshgrid(
        tf.range(h, dtype=tf.float32),
        tf.range(w, dtype=tf.float32),
        indexing='ij')
    # print('map_y.shape={}, map_x.shape={}'.format(map_y.shape, map_x.shape))

    # now each of (batch, h, w)
    map_y = tf.tile(tf.expand_dims(map_y, axis=0), [batch_size, 1, 1])
    map_x = tf.tile(tf.expand_dims(map_x, axis=0), [batch_size, 1, 1])
    # print('map_y.shape={}, map_x.shape={}'.format(map_y.shape, map_x.shape))

    # to normalized coordinates [0, 1],
    # may contain invalid coordinates
    # print('map_y / h shape={}'.format((map_y / h).shape))
    return (map_y / h * (y2_to - y1_to) + y1_to - y1_from) / (y2_from - y1_from), \
        (map_x / w * (x2_to - x1_to) + x1_to - x1_from) / (x2_from - x1_from)
    # return (map_y / tf.cast(h, tf.float32) - y1_to) / (y2_to - y1_to), \
    #     (map_x / tf.cast(w, tf.float32) - x1) / (x2 - x1)


def get_pixel_value(img, y, x):
    """
    Input
    -----
    - img: (batch, h, w, ...)
    - y, x: (batch, oh, ow), ints

    Returns
    -------
    - output: (batch, oh, ow, ...)
    """
    img_shape = tf.shape(img)
    batch_size, h, w = img_shape[0], img_shape[1], img_shape[2]
    feature_shape_part = tf.slice(img_shape, [3], [-1])
    oh, ow = tf.shape(y)[1], tf.shape(y)[2]

    y = tf.cast(y, tf.int32)
    x = tf.cast(x, tf.int32)

    # (batch, oh, ow)
    valid_plain = tf.logical_and(
        tf.logical_and(y >= 0, y < h), tf.logical_and(x >= 0, x < w))
    valid = tf.reshape(valid_plain,
                       tf.concat(
                           [[batch_size, oh, ow],
                            tf.ones_like(feature_shape_part)],
                           axis=0))
    valid = tf.tile(valid, tf.concat([[1, 1, 1], feature_shape_part], axis=0))

    # print('valid.s')
    y = tf.where(valid_plain, y, tf.zeros_like(y))
    x = tf.where(valid_plain, x, tf.zeros_like(x))

    batch_idx = tf.range(batch_size, dtype=tf.int32)
    batch_idx = tf.reshape(batch_idx, (-1, 1, 1))
    batch_idx = tf.tile(batch_idx, (1, oh, ow))

    # (batch, oh, ow, 3)
    indices = tf.stack([batch_idx, y, x], axis=-1)

    return tf.where(valid, tf.gather_nd(img, indices),
                    tf.zeros(tf.shape(valid), img.dtype))


def bilinear_sample(img, ratio_y, ratio_x):
    """
    Input
    ---
    - img: (batch, h, w, ...)
    - ratio_y, ratio_x: (batch, oh, ow)

    Returns
    ---
    - sampled: (batch, oh, ow, ...)
    """
    img_shape = tf.shape(img)
    batch_size, h, w = img_shape[0], img_shape[1], img_shape[2]
    feature_shape_part = tf.slice(img_shape, [3], [-1])
    oh, ow = tf.shape(ratio_y)[1], tf.shape(ratio_y)[2]

    y = ratio_y * tf.cast(h, tf.float32)
    x = ratio_x * tf.cast(w, tf.float32)

    y0 = tf.floor(y)
    x0 = tf.floor(x)
    y1 = y0 + 1
    x1 = x0 + 1

    # get pixel value at corner coords
    # (batch, oh, ow, ...)
    Ia = get_pixel_value(img, y0, x0)
    Ib = get_pixel_value(img, y1, x0)
    Ic = get_pixel_value(img, y0, x1)
    Id = get_pixel_value(img, y1, x1)

    # compute deltas
    # (batch, oh, ow)
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    broadcast_shape = tf.concat(
        [[batch_size, oh, ow],
         tf.ones_like(feature_shape_part)], axis=0)
    wa = tf.reshape(wa, broadcast_shape)
    wb = tf.reshape(wb, broadcast_shape)
    wc = tf.reshape(wc, broadcast_shape)
    wd = tf.reshape(wd, broadcast_shape)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    return out


def align(image, align_matrix, aligned_shape, is_bhwc=True):
    single_data = False
    if len(image.shape) == 3:
        single_data = True
        image = np.expand_dims(image, axis=0)

    if not is_bhwc:
        image = np.transpose(image, [0, 2, 3, 1])

    batch = image.shape[0]
    mat = np.transpose(align_matrix)
    trans = skimage.transform.ProjectiveTransform(mat)

    aligned = [None] * batch
    for i in range(batch):
        aligned = skimage.transform.warp(image[i], trans.inverse,
                                         output_shape=aligned_shape)
    aligned = np.stack(aligned, axis=0)

    if not is_bhwc:
        aligned = np.transpose(aligned, [0, 3, 1, 2])
    if single_data:
        aligned = aligned[0]
    return aligned


def reverse_align(image, align_matrix, original_shape, is_bhwc=True):
    single_data = False
    if len(image.shape) == 3:
        single_data = True
        image = np.expand_dims(image, axis=0)

    if not is_bhwc:
        image = np.transpose(image, [0, 2, 3, 1])

    batch = image.shape[0]
    mat = np.transpose(align_matrix)
    # mat = np.linalg.inv(mat)
    trans = skimage.transform.ProjectiveTransform(mat)

    aligned = [None] * batch
    for i in range(batch):
        aligned[i] = skimage.transform.warp(image[i], trans,
                                            output_shape=original_shape)
    aligned = np.stack(aligned, axis=0)

    if not is_bhwc:
        aligned = np.transpose(aligned, [0, 3, 1, 2])

    if single_data:
        aligned = aligned[0]
    return aligned


def focal_loss(
        prediction_tensor,
        target_tensor,  # weights=None,
        alpha=0.25,
        gamma=2,
        prediction_is_logits=True):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(
        prediction_tensor) if prediction_is_logits else prediction_tensor
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = array_ops.where(target_tensor >= sigmoid_p,
                                target_tensor - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
        - (1 - alpha) * (neg_p_sub ** gamma) * \
        tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return per_entry_cross_ent


############################################################
#  Bounding Boxes
############################################################


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


SINGLE_LANDMARK68_INDICES = {
    'face': list(range(0, 17)),
    'lb': list(range(17, 22)),
    'rb': list(range(22, 27)),
    'nose': list(range(27, 36)),
    'le': list(range(36, 42)),
    're': list(range(42, 48)),
    'mouth': list(range(48, 68)),
    'ulip': list(range(48, 55)),
    'llip': list(range(54, 60)) + [48],
    'imouth': list(range(60, 68)),
    'hair': list(range(0, 68)),
    'lr': list(range(0, 68)),
    'rr': list(range(0, 68)),
    'body': list(range(0, 68))
}


def get_single_landmark68_indices(face_label_name):
    return SINGLE_LANDMARK68_INDICES[face_label_name]


def get_single_landmark68_box_padding(face_label_name):
    # y1 x1 y2 x2
    if face_label_name == 'face':
        return [-130, -30, 30, 30]
    elif face_label_name in ['lb', 'rb']:
        return [-15, -15, 20, 15]
    elif face_label_name == 'nose':
        return [-15, -20, 15, 20]
    elif face_label_name == 'mouth':
        return [-15, -15, 20, 15]
    else:
        return [-15, -15, 15, 15]


def extract_landmark68_boxes(pts,
                             face_label_names,
                             padding_dict,
                             scale_ratio=1.0):
    """Compute bounding boxes from landmarks.
    pts: [68, 2]
    face_label_names:[['face'], ['eyes', 'le', 're']...]
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    num_boxes = len(face_label_names)
    boxes = np.zeros([num_boxes, 4], dtype=np.float32)
    for i in range(num_boxes):
        x1y1 = None
        x2y2 = None
        for name in face_label_names[i]:
            pts_here = pts[get_single_landmark68_indices(name), :]
            if pts_here.size == 0:
                continue
            x1y1_here, x2y2_here = np.min(
                pts_here, axis=0), np.max(
                    pts_here, axis=0)
            if padding_dict is not None:
                padding = np.array(padding_dict[name], dtype=np.float32)
            else:
                # padding = np.array(
                    # get_single_landmark68_box_padding(name), dtype=np.float32)
                padding = np.zeros([4])
            x1y1_here += padding[[1, 0]]
            x2y2_here += padding[[3, 2]]
            if x1y1 is None:
                x1y1 = x1y1_here
            else:
                x1y1 = np.minimum(x1y1, x1y1_here)
            if x2y2 is None:
                x2y2 = x2y2_here
            else:
                x2y2 = np.maximum(x2y2, x2y2_here)

        if x1y1 is None or x2y2 is None:
            continue
        c = (x1y1 + x2y2) / 2.0
        s = (x2y2 - x1y1) * scale_ratio / 2.0
        x1y1 = c - s
        x2y2 = c + s
        boxes[i, 0] = x1y1[1]
        boxes[i, 1] = x1y1[0]
        boxes[i, 2] = x2y2[1]
        boxes[i, 3] = x2y2[0]
    return boxes


def extract_landmark68_boxes_graph(pts,
                                   face_label_names,
                                   molded_padding_dict,
                                   head_box_assigned=None):
    """Compute bounding boxes from landmarks.
    pts: [batch_size, 68, 2]
    face_label_names:[['face'], ['eyes', 'le', 're']...]
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    num_boxes = len(face_label_names)
    boxes = [None] * num_boxes
    batch = tf.shape(pts)[0]
    for i in range(num_boxes):
        if head_box_assigned and i in head_box_assigned:
            boxes[i] = tf.tile(
                tf.expand_dims(
                    tf.constant(head_box_assigned[i], dtype=tf.float32), 0),
                [batch, 1])
        else:
            y1, x1, y2, x2 = [], [], [], []
            for name in face_label_names[i]:
                inds = get_single_landmark68_indices(name)
                if inds:
                    pts_here = tf.stack([pts[:, i, :] for i in inds], axis=1)
                    x1y1 = tf.reduce_min(pts_here, axis=1)  # batch x 2
                    x2y2 = tf.reduce_max(pts_here, axis=1)  # batch x 2
                    molded_padding = tf.constant(
                        molded_padding_dict.get(
                            name, np.array([-0.01, -0.01, +0.01, +0.01], np.float32)),
                        dtype=np.float32)
                    y1.append(x1y1[:, 1] + molded_padding[0])
                    x1.append(x1y1[:, 0] + molded_padding[1])
                    y2.append(x2y2[:, 1] + molded_padding[2])
                    x2.append(x2y2[:, 0] + molded_padding[3])
            assert len(y1) > 0
            y1 = tf.reduce_min(tf.stack(y1, axis=1), axis=1)
            x1 = tf.reduce_min(tf.stack(x1, axis=1), axis=1)
            y2 = tf.reduce_max(tf.stack(y2, axis=1), axis=1)
            x2 = tf.reduce_max(tf.stack(x2, axis=1), axis=1)
            boxes[i] = tf.stack([y1, x1, y2, x2], axis=1)  # batch x 4
    boxes = tf.stack(boxes, axis=1)  # batch x num_boxes x 4
    return boxes


# RANDOM_COLORS = [
#     np.array((1.0, 0.0, 0.0), np.float32),
#     np.array((0.0, 1.0, 0.0), np.float32),
#     np.array((0.0, 0.0, 1.0), np.float32),
#     np.array((1.0, 1.0, 0.0), np.float32),
#     np.array((0.0, 1.0, 1.0), np.float32),
#     np.array((1.0, 0.0, 1.0), np.float32),
#     np.array((0.5, 1.0, 0.0), np.float32),
#     np.array((0.0, 0.5, 1.0), np.float32),
#     np.array((0.5, 0.0, 1.0), np.float32),
#     np.array((1.0, 0.5, 0.0), np.float32),
#     np.array((0.0, 1.0, 0.5), np.float32),
#     np.array((1.0, 0.0, 0.5), np.float32),
# ] + random_colors(20)

# class_names:
#   - [face, lb, rb, le, re, nose, ulip, llip, imouth]
#   - [lb]
#   - [rb]
#   - [le]
#   - [re]
#   - [nose]
#   - [ulip]
#   - [llip]
#   - [imouth]

# heads:
#   - [0]
#   - [1]
#   - [2]
#   - [3]
#   - [4]
#   - [5]
#   - [6, 7, 8]

# RANDOM_COLORS = [
#     np.array((1.0, 1.0, 1.0), np.float32),
#     np.array((1.0, 0.0, 0.0), np.float32), # face
#     np.array([255, 144, 30], np.float32) / 255.0, # lb
#     np.array([255, 165, 0], np.float32) / 255.0, # rb
#     np.array([30, 144, 255], np.float32) / 255.0, # le
#     np.array([65, 105, 255], np.float32) / 255.0, # re
#     np.array([0, 255, 255], np.float32) / 255.0, # nose
#     np.array([0, 255, 0], np.float32) / 255.0, #ulip
#     np.array([0, 0, 255], np.float32) / 255.0, # llip
#     np.array([255, 255, 255], np.float32) / 255.0, # imouth
#     np.array((1.0, 0.5, 0.0), np.float32),
#     np.array((0.0, 1.0, 0.5), np.float32),
#     np.array((1.0, 0.0, 0.5), np.float32),
# ] + random_colors(20)

RANDOM_COLORS = [
    np.array((1.0, 1.0, 1.0), np.float32),
    np.array((20, 20, 255), np.float32) / 255.0,  # hair?
    np.array((255, 250, 79), np.float32) / 255.0,  # face
    np.array([255, 125, 138], np.float32) / 255.0,  # lb
    np.array([213, 32, 29], np.float32) / 255.0,  # rb
    np.array([0, 144, 187], np.float32) / 255.0,  # le
    np.array([0, 196, 253], np.float32) / 255.0,  # re
    np.array([255, 129, 54], np.float32) / 255.0,  # nose
    np.array([88, 233, 135], np.float32) / 255.0,  # ulip
    np.array([0, 117, 27], np.float32) / 255.0,  # llip
    np.array([255, 76, 249], np.float32) / 255.0,  # imouth
    np.array((1.0, 0.5, 0.0), np.float32),
    np.array((0.0, 1.0, 0.5), np.float32),
    np.array((1.0, 0.0, 0.5), np.float32),
] + random_colors(20)

# RANDOM_COLORS = [
#     np.array((1.0, 1.0, 1.0), np.float32),
#     np.array((1.0, 0.0, 1.0), np.float32), # body
#     np.array((0.0, 1.0, 0.0), np.float32), # hair
#     np.array((1.0, 0.0, 0.0), np.float32), # face
#     np.array((0.5, 1.0, 0.7), np.float32), #lr
#     np.array((0.7, 1.0, 0.5), np.float32), #rr
#     np.array([255, 144, 30], np.float32) / 255.0, # lb
#     np.array([255, 165, 0], np.float32) / 255.0, # rb
#     np.array([30, 144, 255], np.float32) / 255.0, # le
#     np.array([65, 105, 255], np.float32) / 255.0, # re
#     np.array([0, 255, 255], np.float32) / 255.0, # nose
#     np.array([0, 255, 0], np.float32) / 255.0, #ulip
#     np.array([0, 0, 255], np.float32) / 255.0, # llip
#     np.array([255, 255, 255], np.float32) / 255.0, # imouth
#     np.array((1.0, 0.5, 0.0), np.float32),
#     np.array((0.0, 1.0, 0.5), np.float32),
#     np.array((1.0, 0.0, 0.5), np.float32),
# ] + random_colors(20)


def blend_labels(image, labels):
    assert len(labels.shape) == 2
    colors = RANDOM_COLORS
    # colors = np.array(
    #     [(0.0, 1.0, 0.7272727272727271), (0.0, 0.7272727272727275, 1.0),
    #      (0.3636363636363633, 0.0, 1.0), (0.36363636363636376, 1.0, 0.0),
    #      (0.0, 0.18181818181818166, 1.0), (1.0, 0.5454545454545454, 0.0),
    #      (0.9090909090909092, 1.0, 0.0), (0.9090909090909092, 0.0, 1.0),
    #      (0.0, 1.0, 0.18181818181818166), (1.0, 0.0, 0.5454545454545459),
    #      (1.0, 0.0, 0.0)], np.float32)
    if image is None:
        image = np.zeros([labels.shape[0], labels.shape[1], 3], np.float32)
        alpha = 1.0
    else:
        image = image / np.max(image) * 0.4
        alpha = 0.6
    for i in range(1, np.max(labels) + 1):
        image += alpha * \
            np.tile(
                np.expand_dims(
                    (labels == i).astype(np.float32), -1),
                [1, 1, 3]) * colors[(i) % len(colors)]
    image[np.where(image > 1.0)] = 1.0
    image[np.where(image < 0)] = 0.0
    return image


def blend_alphas(image, alphas):
    colors = RANDOM_COLORS
    image = image / np.max(image) / 2.0
    for i in range(alphas.shape[0]):
        image += 0.5 * \
            np.tile(
                np.expand_dims(
                    alphas[i, :, :], -1),
                [1, 1, 3]) * colors[(i) % len(colors)]
    # image[np.where(image > 1.0)] = 1.0
    # image[np.where(image < 0)] = 0.0
    return image


def flatten_masks(masks, class_ids=None):
    if class_ids is None:
        class_ids = np.arange(1, masks.shape[0] + 1)
    assert masks.shape[0] == len(class_ids)
    for class_id in class_ids:
        assert 0 < class_id
    mask = np.zeros(masks.shape[1:], dtype=np.int32)
    for i, class_id in enumerate(class_ids):
        mask[np.where(masks[i, :, :])] = class_id
    return mask


def order_masks(mask, class_ids, max_gt_instances):
    class_exists = np.zeros([max_gt_instances], dtype=np.uint8)
    # order masks
    # first allocate masks as [height, width, MAX_GT_INSTANCES]
    masks = np.zeros(
        [mask.shape[0], mask.shape[1], max_gt_instances], dtype=np.uint8)
    for i, class_id in enumerate(class_ids):
        masks[:, :, class_id - 1] = np.logical_or(mask[:, :, i],
                                                  masks[:, :, class_id - 1])
        class_exists[class_id - 1] = True
    return masks, class_exists


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(image,
                                    (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1], ), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1], ), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
        mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    # print(bbox)
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    # print(full_mask.shape)
    # print(mask.shape)
    if y2 <= 0 or x2 <= 0:
        return full_mask
    if y1 < 0:
        mask = mask[-y1:, :]
        y1 = 0
    if x1 < 0:
        mask = mask[:, -x1:]
        x1 = 0

    if y1 > full_mask.shape[0] or x1 > full_mask.shape[1]:
        return full_mask
    if y2 > full_mask.shape[0]:
        mask = mask[:full_mask.shape[0] - y2, :]
        y2 = full_mask.shape[0]
    if x2 > full_mask.shape[1]:
        mask = mask[:, :full_mask.shape[1] - x2]
        x2 = full_mask.shape[1]

    full_mask[y1:y2, x1:x2] = mask
    return full_mask


def unmold_mask_float(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    y1, x1, y2, x2 = bbox
    # print(bbox)
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.float32)
    # print(full_mask.shape)
    # print(mask.shape)
    if y2 <= 0 or x2 <= 0:
        return full_mask
    if y1 < 0:
        mask = mask[-y1:, :]
        y1 = 0
    if x1 < 0:
        mask = mask[:, -x1:]
        x1 = 0

    if y1 > full_mask.shape[0] or x1 > full_mask.shape[1]:
        return full_mask
    if y2 > full_mask.shape[0]:
        mask = mask[:full_mask.shape[0] - y2, :]
        y2 = full_mask.shape[0]
    if x2 > full_mask.shape[1]:
        mask = mask[:, :full_mask.shape[1] - x2]
        x2 = full_mask.shape[1]

    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
#  Anchors
############################################################


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate(
        [box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(
            generate_anchors(scales[i], ratios, feature_shapes[i],
                             feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_ap(gt_boxes,
               gt_class_ids,
               pred_boxes,
               pred_class_ids,
               pred_scores,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding and sort predictions by score from high to low
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    # Loop through ground truth boxes and find matching predictions
    match_count = 0
    pred_match = np.zeros([pred_boxes.shape[0]])
    gt_match = np.zeros([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] == 1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                break

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum(
        (recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def main():
    plt.plot(*zip(*MEAN_MOLDED_LANDMARK68_PTS))


if __name__ == '__main__':
    main()
