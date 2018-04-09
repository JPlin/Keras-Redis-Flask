# USAGE
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.applications import imagenet_utils
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import img_to_array
from threading import Thread
from PIL import Image
import numpy as np
import base64
import flask
import redis
import uuid
import time
import json
import sys
import warnings
import functools
import operator
import tensorflow as tf
import shutil
import importlib
import yaml
import os

from model.parser.dataset import get_dataset
from model.parser.data_generator import data_generator
from model.parser import arc_eccv2018 as arc

from lib import tool
import settings

class Parser():
    def __init__(self , db):
        self.db = db
        self.color_dict = {
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

        self.classes = ['bg', 'body', 'hair', 'face', 'lear', 'rear',
               'lb', 'rb', 'le', 're', 'nose', 'mouth']
        self.model = None
        self.build()
    
    def build(self):
        weight_path = settings.P_WEIGHT_PATH
        options = yaml.load(open(settings.P_YAML_PATH , 'r').read())
        options['name'] = 'arc_eccv2018'

        weight_folder = os.path.dirname(os.path.abspath(weight_path))

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True
        tf_config.gpu_options.visible_device_list = '0'
        sess = tf.Session(config=tf_config)
        set_session(sess)

        model = arc.Architecture(
            mode = 'inference',
            options = options,
            model_dir = weight_folder,
            gpu_count = 1,
            tfdbg = 0
        )

        print('Loading weights ' , weight_path)
        model.load_weights(weight_path , by_name = True)
        self.model = model.keras_model

    def _blend_labels(self,image, labels):
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
                [1, 1, 3]) * self.color_dict[self.classes[(i) % len(self.classes)]]

        return image


    def _flatten_pred_masks(self,pred_masks):
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

    @staticmethod
    def prepare_image(image , target):
        MEAN_PIXEL = np.array([123.7, 116.8, 103.9], np.float32)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target)
        image = img_to_array(image)
        image = np.expand_dims(image , axis = 0)
        image = image - MEAN_PIXEL
        return image

    def batch_image_parsing(self):
        MEAN_PIXEL = np.array([123.7, 116.8, 103.9], np.float32)
        while True:
            queue = self.db.lrange(settings.P_IMAGE_QUEUE , 0 , settings.BATCH_SIZE - 1)

            imageIDs = []
            batch = None
            for q in queue:
                q = json.loads(q.decode("utf-8"))
                image = tool.base64_decode_image(q["image"] , settings.P_IMAGE_DTYPE , (1 , settings.P_IMAGE_HEIGHT , 
                settings.P_IMAGE_WIDTH , settings.P_IMAGE_CHANS))

                if batch is None:
                    batch = image
                else:
                    batch = np.vstack([batch, image])
                imageIDs.append(q['id'])

            if len(imageIDs) > 0:
                print("* Batch size: {}".format(batch.shape))
                start_time = time.time()
                batch_size = batch.shape[0]
                exists = np.array([1] * batch_size , np.uint8)
                inputs = [exists , batch]
                preds = self.model.predict(inputs , verbose = 0)
                print('time cost %f sec '% (time.time() - start_time))
                
                if not isinstance(preds , list):
                    preds = [preds]
                
                pred_masks = np.clip(preds[0] , 0.0 , 1.0)
                if len(preds) >= 2:
                    pred_molded_head_boxes = preds[1]
                    pred_head_boxes = pred_molded_head_boxes * 512
                
                batch_size = batch.shape[0]
                blended_labels = [None] * batch_size
                pred_labels = self._flatten_pred_masks(pred_masks)
                for k in range(batch_size):
                    image = batch[k]
                    blended_labels[k] = self._blend_labels(image + MEAN_PIXEL, pred_labels[k])


                for (imageID , pred) in zip(imageIDs , blended_labels):
                    pred = pred.astype(np.uint8)
                    output = tool.base64_encode_image(pred)
                    self.db.set(imageID ,json.dumps({'image':output}))
                
                self.db.ltrim(settings.P_IMAGE_QUEUE , len(imageIDs) , -1)
            
            time.sleep(settings.SERVER_SLEEP)


# if this is the main thread of execution start the model server process
if __name__ == '__main__':
    print("* Starting parser model service...")
    #parser = Parser(settings.db)
    parser = Parser(redis.from_url(settings.REDIS_URL))
    parser.batch_image_parsing()

