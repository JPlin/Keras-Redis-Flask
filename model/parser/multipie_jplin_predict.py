import time
import numpy as np
import colorsys
import random
import matplotlib.pyplot as plt

'''Face Parsing'''
import os
import sys
import math
import warnings
import functools
import operator
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from pprint import pprint
import shutil
import importlib
from skimage import io, transform
import yaml
from dataset import get_dataset
from data_generator import data_generator

import paths
import dataset
import sys
import os


color_dict = {
    'bg': [0, 0, 0],
    'face': [255, 250, 79],
    'body': [0, 0, 255],
    'hair': [0, 255, 0],
    'lb': [255, 125, 138],
    'rb': [213, 32, 29],
    'le': [0, 144, 187],
    're': [0, 196, 253],
    'nose': [255, 129, 54],
    'mouth': [255, 0, 255],
    #'ul': [88, 233, 135],
    #'im': [255, 76, 249],
    #'dl': [0, 117, 27],
    'lear': [0, 0, 139],
    'rear': [65, 105, 225]
}

classes = ['bg', 'body', 'hair', 'face', 'lear', 'rear',
           'lb', 'rb', 'le', 're', 'nose', 'mouth']


def blend_labels(image, labels):
    assert len(labels.shape) == 2
    if image is None:
        image = np.zeros([labels.shape[0], labels.shape[1], 3], np.float32)
        alpha = 1.0
    else:
        image = image * 0.7
        alpha = 0.3

    for i in range(1, np.max(labels)):
        image += alpha * np.tile(
            np.expand_dims(
                (labels == i).astype(np.float32), -1),
            [1, 1, 3]) * color_dict[classes[(i) % len(classes)]]

    #image[np.where(image > 1.0)] = 1.0
    #image[np.where(image < 0)] = 0.0
    return image


def _flatten_pred_masks(pred_masks):
    single_data = False
    if len(pred_masks.shape) == 3:
        single_data = True
        pred_masks = np.expand_dims(pred_masks, 0)

    batch = pred_masks.shape[0]
    flatten_pred_labels = np.zeros((batch,) + pred_masks.shape[2:], np.int32)

    heads = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    for class_ids in heads:
        bg_mask_this_head = 1.0 - np.sum(
            pred_masks[:, class_ids, :, :], axis=1, keepdims=True)
        fg_bg_masks_this_head = np.concatenate(
            [pred_masks[:, class_ids, :, :], bg_mask_this_head], axis=1
        )
        max_ids_this_head = np.argmax(fg_bg_masks_this_head, axis=1)
        for j, class_id in enumerate(class_ids):
            flatten_pred_labels[np.where(
                max_ids_this_head == j
            )] = class_id + 1

    if single_data:
        flatten_pred_labels = flatten_pred_labels[0]
    return flatten_pred_labels


def predict_masks(model, inputs):
    images = inputs[1]
    image_len = images.shape[0]
    start_time = time.time()
    preds = model.predict(inputs, verbose=0)
    print('time cost %f sec ' % (time.time() - start_time))

    if not isinstance(preds, list):
        preds = [preds]

    pred_masks = np.clip(preds[0], 0.0, 1.0)
    if len(preds) >= 2:
        pred_molded_head_boxes = preds[1]
        pred_head_boxes = pred_molded_head_boxes * 512

    MEAN_PIXEL = np.array([123.7, 116.8, 103.9], np.float32)
    images = images + MEAN_PIXEL
    pred_labels = [None] * image_len
    for k in range(image_len):
        pred_labels = _flatten_pred_masks(pred_masks)

        image = images[k]
        blended_labels = blend_labels(image, pred_labels[k])

        plt.imshow(image.astype(np.uint8))
        plt.show()
        plt.imshow(pred_labels[k])
        plt.show()
        plt.imshow(blended_labels.astype(np.uint8))
        plt.show()


if __name__ == '__main__':
    FLAGS = {}
    FLAGS['weights'] = '中文/parsing.h5'
    FLAGS['architecture'] = 'arc_eccv2018'

    # get option file path from weight path
    options = yaml.load(open('multipie_jinpli.yaml', 'r').read())
    options['name'] = 'multipie_jinpli'

    weight_folder = os.path.dirname(
        os.path.abspath(FLAGS['weights']))

    arc = importlib.import_module(FLAGS['architecture'])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    tf_config.gpu_options.visible_device_list = ','.join(
        [str(i) for i in [4, 5, 6, 7]])
    sess = tf.Session(config=tf_config)
    set_session(sess)

    model = arc.Architecture(
        mode='inference',
        options=options,
        model_dir=weight_folder,
        gpu_count=1,
        tfdbg=0
    )

    print("Loading weights ", FLAGS['weights'])
    model.load_weights(FLAGS['weights'], by_name=True)
    print('model.log_dir = %s' % model.log_dir)

    dataset_val = get_dataset('multipie_test', options)
    generator = data_generator(
        dataset_val,
        model.required_data_names('inference'),
        shuffle=True,
        batch_size=1
    )
    for inputs, _ in generator:
        print('inputs[0]', inputs[0])
        print('inputs[1].shape {}'.format(inputs[1].shape))
        predict_masks(model.keras_model, inputs)
