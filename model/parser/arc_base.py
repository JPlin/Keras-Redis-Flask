"""
ArcBase
"""

import os
import sys
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import glob
import random
import math
import datetime
import itertools
import json
import re
import logging
from sys import platform
# from skimage import
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
from skimage import transform, io, draw
import scipy.io as scio
from scipy.misc import imresize
import yaml
import time

import utils
from data_generator import data_generator, multi_data_generator, CachedDataSequence

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


def print_tensor(inp, data, name=None):
    return KL.Lambda(lambda x: tf.Print(x, data), name=name)(inp)


class SimpleTensorBoard(keras.callbacks.Callback):
    def __init__(self,
                 log_dir='./logs',
                 freq=0,
                 write_graph=False,
                 batch_size=32):
        super(SimpleTensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.freq = freq
        self.merged = None
        self.write_graph = write_graph
        self.batch_size = batch_size
        self.sess = None
        self.writer = None

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        self.merged = tf.summary.merge_all()
        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.validation_data and self.freq and self.merged is not None:
            if epoch % self.freq == 0:
                val_data = self.validation_data
                tensors = self.model.inputs

                if self.model.uses_learning_phase and K.learning_phase(
                ) not in tensors:
                    tensors = tensors + [K.learning_phase()]
                # for input in tensors:
                # print('{}, shape={}'.format(input.name, input.shape))

                # print('self.model.uses_learning_phase=%d' % self.model.uses_learning_phase)
                # print('len(val_data)=%d, len(tensors)=%d' % (len(val_data), len(tensors)))
                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()


############################################################
#  Utility Functions
############################################################


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


############################################################
#  ArcBase Class
############################################################
def _is_valid_mode(mode):
    return mode.startswith('training') or mode == 'inference'


class ArcBase():
    """Basic Architecture.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, options, model_dir, gpu_count, tfdbg):
        """
        mode: "training..." or "inference"
        options: from yaml
        model_dir: Directory to save training logs and trained weights
        """
        assert _is_valid_mode(mode)
        self.mode = mode
        self.options = options
        self.model_dir = model_dir
        self.gpu_count = gpu_count
        self.set_log_dir()
        # self.input_dict = {}
        self.keras_model = self.build(mode=mode, options=options)
        self.tfdbg = tfdbg

    # def get_input(self,
    #               shape=None,
    #               batch_shape=None,
    #               name=None,
    #               dtype=None,
    #               sparse=False,
    #               tensor=None):
    #     if name in self.input_dict:
    #         i = self.input_dict[name]
    #     else:
    #         i = KL.Input(
    #             shape=shape,
    #             batch_shape=batch_shape,
    #             name=name,
    #             dtype=dtype,
    #             sparse=sparse,
    #             tensor=tensor)
    #         self.input_dict[name] = i
    #     return i

    def define_graph(self, mode, options):
        ''' returns [inputs, outputs]
        '''
        raise NotImplementedError()

    def required_data_names(self, mode=None):
        ''' returns [input_data_names, output_data_names]
        '''
        raise NotImplementedError()

    def build(self, mode, options):
        """Build the keras model.
            input_shape: The shape of the input image.
            mode: "training..." or "inference".
        """
        assert _is_valid_mode(mode)

        if self.gpu_count == 1:
            print('single gpu mode')
            inputs, outputs = self.define_graph(mode, options)
            model = KM.Model(inputs, outputs, name=options['name'])
        else:
            print('multi gpu mode')
            with tf.device('/cpu:0'):
                inputs, outputs = self.define_graph(mode, options)
                model = KM.Model(inputs, outputs, name=options['name'])
            # model = multi_gpu_model(model, gpus=self.gpu_count)
            from parallel_model import ParallelModel
            model = ParallelModel(model, self.gpu_count)
        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.options['name'].lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"),
                             checkpoints)  # FIXME: prefix may not be mask_rcnn
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    @staticmethod
    def _transfer_weights(to_model, from_model):
        to_layers = to_model.inner_model.layers \
            if hasattr(to_model, "inner_model") \
            else to_model.layers
        from_layers = from_model.inner_model.layers \
            if hasattr(from_model, "inner_model") \
            else from_model.layers

        from_index = {}
        for from_layer in from_layers:
            if from_layer.name:
                from_index.setdefault(from_layer.name, []).append(from_layer)

        # We batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for to_layer in to_layers:
            name = to_layer.name
            for from_layer in from_index.get(name, []):
                to_weights = to_layer.weights
                from_weights = from_layer.get_weights()
                if len(to_weights) != len(from_weights):
                    raise ValueError(
                        'Target layer %s expects %d weight(s), '
                        'but the source layer have %d element(s)' %
                        (name, len(to_weights), len(from_weights)))
                print('%d Weights in Layer %s is transfered' %
                      (len(to_weights), name))
                for to_w, from_w in zip(to_weights, from_weights):
                    weight_value_tuples.append((to_w, from_w))
        K.batch_set_value(weight_value_tuples)

    def set_mode_and_transfer_weights(self, mode):
        if self.mode == mode:
            return
        print('mode change from %s to %s' % (self.mode, mode))
        self.mode = mode
        new_keras_model = self.build(mode, self.options)
        ArcBase._transfer_weights(new_keras_model, self.keras_model)
        self.keras_model = new_keras_model

    def load_weights(self, filepath, by_name=True,
                     exclude=None, try_continue_training=True):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(
                f, layers, skip_mismatch=True)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        if try_continue_training:
            self.set_log_dir(filepath)
        else:
            self.set_log_dir(None)

    def compile(self, learning_rate, momentum, decay=0.):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        print('compile is called with learning_rate=%f, momentum=%f' %
              (learning_rate, momentum))

        # def _print_layer_weights(prefix, layer):

        # for layer in self.keras_model.layers[18].layers:
        #     weights = layer.trainable_weights
        #     wsize = sum([np.prod(K.get_value(w).shape) for w in weights])
        #     print('size(%s.weights)=%d' % (layer.name, wsize))

        # Optimizer object
        optimizer_type = self.options.get('optimizer', 'sgd')
        print(f'using {optimizer_type} optimizer')
        if optimizer_type == 'sgd':
            optimizer = keras.optimizers.SGD(
                lr=learning_rate, momentum=momentum,
                decay=decay, clipnorm=5.0)
        elif optimizer_type == 'adam':
            optimizer = keras.optimizers.Adam(
                lr=learning_rate, beta_1=0.9, beta_2=0.99,
                decay=decay, clipnorm=5.0)
        elif optimizer_type == 'adadelta':
            optimizer = keras.optimizers.Adadelta(
                lr=learning_rate)

        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        # self.keras_model.output_names

        # loss_names = ["l%d" % i for i in range(self.config.MAX_GT_INSTANCES)]
        loss_names = [
            name for name in self.keras_model.output_names
            if name.endswith('_loss') or name.endswith('_l')
        ]
        print('loss_names: {}'.format(loss_names))

        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(
                tf.reduce_mean(layer.output, keep_dims=True))

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.options['weight_decay'])(w) / tf.cast(
                tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name
        ]
        if reg_losses:
            self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(
                tf.reduce_mean(layer.output, keep_dims=True))

    def set_trainable(self, layer_regex, keras_model=None, indent=0,
                      verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(
                    int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.options['name'].lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir,
                                            "mask_rcnn_{}_*epoch*.h5".format(
                                                self.options['name'].lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_datasets, val_datasets, stage_id):
        """Train the model.
        train_datasets, val_datasets: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode.startswith(
            "training"), "Create model in training mode."

        learning_rate_base = self.options['learning_rate_base']
        stage = self.options['stages'][stage_id]

        learning_rate = stage['lr_ratio'] * learning_rate_base
        epochs = stage['epoch']
        layers = stage['layers']

        learning_rate_decay_factor = \
            self.options.get('learning_rate_decay_factor', 0.9)
        learning_rate_decay_factor =\
            stage.get('learning_rate_decay_factor', learning_rate_decay_factor)
        min_learning_rate = self.options.get('min_learning_rate', 1e-5)
        min_learning_rate = stage.get('min_learning_rate', min_learning_rate)

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads":
            r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(head\_.*)",
            # From a specific Resnet stage and up
            "3+":
            r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+":
            r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+":
            r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all":
            ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        batch_size = self.options['images_per_gpu'] * self.gpu_count

        # Data generators
        train_generator = multi_data_generator(
            train_datasets,
            required_data_names=self.required_data_names(self.mode),
            shuffle=True,
            batch_size=batch_size)
        val_generator = multi_data_generator(
            val_datasets,
            required_data_names=self.required_data_names(self.mode),
            shuffle=False,
            batch_size=batch_size)

        if 'learning_rate_warmup' in self.options:
            lr_wu, lr_wu_epochs = self.options['learning_rate_warmup']
        else:
            lr_wu = learning_rate
            lr_wu_epochs = 0

        lr_log_file = open(os.path.join(self.log_dir, 'lr_log.txt'), 'a')
        lr_input_file = os.path.join(self.log_dir, 'lr_input.txt')

        def _get_learning_rate(e):
            # print('e = %d' % e)
            lr = 0
            if os.path.exists(lr_input_file):
                print('reading learning rate from lr_input.txt')
                lr_input = open(lr_input_file, 'r').read().strip()
                try:
                    lr = float(lr_input)
                except:
                    print('the value in lr_input.txt is not a number, ignored')
                    lr = 0
            if lr <= 0:
                if e < lr_wu_epochs:
                    lr = lr_wu
                    print('warming up')
                else:
                    lr = learning_rate * (
                        learning_rate_decay_factor **
                        (e - self.epoch - lr_wu_epochs))
                    lr = max(lr, min_learning_rate)
            print('learning rate now is %f' % lr)
            lr_log_file.write(f'epoch {e}: learning rate = {lr}\n')
            lr_log_file.flush()
            return lr

        # Callbacks
        callbacks = [
            SimpleTensorBoard(log_dir=self.log_dir, freq=1),
            # keras.callbacks.TerminateOnNaN(),
            keras.callbacks.LearningRateScheduler(_get_learning_rate)
        ]

        if not ('nologs' in self.options and self.options['nologs']):
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    self.checkpoint_path, verbose=0, save_weights_only=True))
        else:
            print('there shall be no logs since `nologs` is set to `on`')

        # if self.tfdbg == 1:
        #     prev_sess = K.get_session()
        #     sess = tf_debug.LocalCLIDebugWrapperSession(prev_sess)
        #     K.set_session(sess)

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch,
                                                     learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.options['learning_momentum'])

        # keras.utils.Sequence
        if platform.startswith('linux'):
            print('enable multiprocessing')
            workers = max(batch_size // 2, 2)
            use_multiprocessing = True
        else:
            print('disable multiprocessing')
            workers = 1
            use_multiprocessing = False

        steps_per_epoch = self.options.get(
            'steps_per_epoch',
            (train_datasets[0].num_images + batch_size - 1) // batch_size)

        self.keras_model.fit_generator(
            # CachedDataSequence(train_generator, 30),
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            # batch_size=batch_size,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.options['validation_steps'],
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
        self.epoch = max(self.epoch, epochs)

        # if self.tfdbg == 1:
        #     K.set_session(prev_sess)

    def _flatten_gt_masks(self, gt_masks):
        single_data = False
        if len(gt_masks.shape) == 3:  # m x h x w
            single_data = True
            gt_masks = np.expand_dims(gt_masks, 0)

        batch = gt_masks.shape[0]
        num_masks = gt_masks.shape[1]
        flatten_gt_labels = np.zeros((batch, ) + gt_masks.shape[2:], np.int32)
        for i in range(num_masks):
            flatten_gt_labels[np.where(gt_masks[:, i, :, :])] = i + 1

        if single_data:
            assert flatten_gt_labels.shape[0] == 1
            flatten_gt_labels = flatten_gt_labels[0]
        return flatten_gt_labels

    def _flatten_predicted_masks(self, pred_masks):
        single_data = False
        if len(pred_masks.shape) == 3:
            single_data = True
            pred_masks = np.expand_dims(pred_masks, 0)

        batch = pred_masks.shape[0]
        flatten_pred_labels = np.zeros(
            (batch,) + pred_masks.shape[2:], np.int32)
        # recover segmentation from masks
        # the latter labels covers the former labels
        for class_ids in self.options['heads']:
            # there is no order within the same head, choose by max
            bg_mask_this_head = 1.0 - np.sum(
                pred_masks[:, class_ids, :, :], axis=1, keepdims=True)
            fg_bg_masks_this_head = np.concatenate(
                [pred_masks[:, class_ids, :, :], bg_mask_this_head], axis=1)
            max_ids_this_head = np.argmax(fg_bg_masks_this_head, axis=1)
            for j, class_id in enumerate(class_ids):
                flatten_pred_labels[np.where(
                    max_ids_this_head == j)] = class_id + 1

        if single_data:
            assert flatten_pred_labels.shape[0] == 1
            flatten_pred_labels = flatten_pred_labels[0]
        return flatten_pred_labels

    def freeze_to_pb(self, pb_file_path, output_node_names=None):
        assert self.mode == 'inference'
        if output_node_names is None:
            output_node_names = [n.name.split(':')[0]
                                 for n in self.keras_model.outputs]
        print(f'output_node_names={output_node_names}')
        graph = tf.graph_util.remove_training_nodes(K.get_session().graph_def)
        constant_graph = tf.graph_util.convert_variables_to_constants(
            K.get_session(), graph, output_node_names)
        with tf.gfile.FastGFile(pb_file_path, 'wb') as f:
            f.write(constant_graph.SerializeToString())

    def eval_head_boxes(self, dataset_val, eval_dir=None):
        if eval_dir is not None:
            eval_dir = eval_dir.replace('\\', '/')
            if not os.path.exists(eval_dir):
                os.mkdir(eval_dir)
        assert self.mode == 'inference'
        num_images = len(dataset_val.image_ids)
        batch_size = self.options['images_per_gpu'] * self.gpu_count
        generator = data_generator(
            dataset_val,
            self.required_data_names(self.mode),
            shuffle=False,
            batch_size=batch_size)

        start_id = 0
        ious = []
        for inputs, _ in generator:
            stop_id = start_id + batch_size
            print('processing %d-%d' % (start_id, stop_id))

            preds = self.keras_model.predict(inputs, verbose=0)
            if not isinstance(preds, list):
                preds = [preds]

            pred_molded_head_boxes = preds[1]
            for k in range(batch_size):
                ind = (start_id + k) % num_images
                box = pred_molded_head_boxes[k]  # 7 x 4
                y1, x1, y2, x2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
                areas = (x2 - x1) * (y2 - y1)

                gt_box, _ = dataset_val.load_molded_head_boxes(ind, None)
                gt_y1, gt_x1, gt_y2, gt_x2 = gt_box[:,
                                                    0], gt_box[:, 1], gt_box[:, 2], gt_box[:, 3]
                gt_areas = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

                xx1 = np.maximum(x1, gt_x1)
                yy1 = np.maximum(y1, gt_y1)
                xx2 = np.minimum(x2, gt_x2)
                yy2 = np.minimum(y2, gt_y2)

                h = np.maximum(0, yy2 - yy1)
                w = np.maximum(0, xx2 - xx1)

                intersection = w * h
                iou = intersection / (areas + gt_areas - intersection)
                ious.append(iou)
                # print(iou)

            start_id += batch_size
            if start_id >= num_images:
                break

        ious = np.stack(ious, axis=0)
        print(np.mean(ious, axis=0))

    def eval_masks(self,
                   dataset_val,
                   eval_dir,
                   save_images,
                   save_stats,
                   gui,
                   sort_class_name,
                   verbose=True):
        eval_dir = eval_dir.replace('\\', '/')
        assert self.mode == 'inference'
        num_images = len(dataset_val.image_ids)

        def _vprint(*args):
            if verbose:
                print(*args)
        _vprint('num_images: %d' % num_images)

        eval_class_ids = []
        eval_names = []
        for k, inds in self.options['eval_classes'].items():
            eval_names.append(k)
            eval_class_ids.append(inds)
            if k == sort_class_name:
                sort_class_ids = inds
        assert len(eval_class_ids) == len(eval_names)

        # evaluate
        def _collect_wrong_pixel_num_for_sort(hist_vals):
            intersected = 0
            label_ids = sort_class_ids
            for label_id1 in label_ids:
                for label_id2 in label_ids:
                    intersected += hist_vals[label_id1, label_id2]
            A = hist_vals[label_ids, :].sum()
            B = hist_vals[:, label_ids].sum()
            return A + B - 2 * intersected

        def _collect_f1(hist_vals):
            f1s = dict()
            for i, label_ids in enumerate(eval_class_ids):
                name = eval_names[i]
                _vprint('ids of {} is {}'.format(name, label_ids))
                intersected = 0
                for label_id1 in label_ids:
                    for label_id2 in label_ids:
                        intersected += hist_vals[label_id1, label_id2]
                A = hist_vals[label_ids, :].sum()
                B = hist_vals[:, label_ids].sum()
                f1 = 2 * intersected / (A + B)
                f1s[name] = f1
            return f1s

        # including background
        batch_size = self.options['images_per_gpu'] * self.gpu_count
        generator = data_generator(
            dataset_val,
            self.required_data_names(self.mode),
            shuffle=False,
            batch_size=batch_size)
        gt_generator = data_generator(
            dataset_val, ['image', 'masks'],
            shuffle=False,
            batch_size=batch_size)

        all_hists = [None] * num_images
        all_gt_labels = [None] * num_images
        all_pred_labels = [None] * num_images
        all_pred_masks = [None] * num_images
        all_pred_head_boxes = [None] * num_images

        all_pred_labels_in_original = [None] * num_images
        all_pred_masks_in_original = [None] * num_images

        start_id = 0
        for (inputs, _), ((_, gt_masks_exist, images, gt_masks), _) in zip(
                generator, gt_generator):
            stop_id = start_id + batch_size
            _vprint('processing %d-%d' % (start_id, stop_id))

            start_time = time.time()
            preds = self.keras_model.predict(inputs, verbose=0)
            elapsed_time = time.time() - start_time
            print('time cost %f sec' % elapsed_time)

            if not isinstance(preds, list):
                preds = [preds]

            pred_masks = np.clip(preds[0], 0.0, 1.0)
            if len(preds) >= 2:
                pred_molded_head_boxes = preds[1]
                pred_head_boxes = pred_molded_head_boxes * \
                    self.options['image_size']

            assert (pred_masks.shape[0], pred_masks.shape[1]) == \
                (gt_masks.shape[0], gt_masks.shape[1])

            assert pred_masks.shape == gt_masks.shape

            for k in range(batch_size):
                ind = (start_id + k) % dataset_val.num_images

                # first collect labels
                all_pred_masks[ind] = pred_masks[k]
                all_gt_labels[ind] = self._flatten_gt_masks(gt_masks[k])

                all_pred_labels[ind] = self._flatten_predicted_masks(
                    pred_masks[k])

                align_matrix, align_matrix_exists = dataset_val.load_align_matrix(
                    ind)
                original_image, original_image_exists = dataset_val.load_original_image(
                    ind)
                has_original_data = align_matrix_exists and original_image_exists
                if has_original_data:
                    all_pred_masks_in_original[ind] = utils.reverse_align(
                        pred_masks[k], align_matrix,
                        original_image.shape, is_bhwc=False)

                    all_pred_labels_in_original[ind] = self._flatten_predicted_masks(
                        all_pred_masks_in_original[ind])

                # second compute hists
                num_masks = len(self.options['class_names'])
                assert pred_masks.shape[1] == num_masks
                assert gt_masks.shape[1] == num_masks

                all_hists[ind] = utils.fast_hist(
                    all_gt_labels[ind], all_pred_labels[ind], num_masks + 1)

                if 'pred_head_boxes' in locals():
                    all_pred_head_boxes[ind] = pred_head_boxes[k]

                # calc individual wrong pixel nums
                wpn_here = _collect_wrong_pixel_num_for_sort(all_hists[ind])
                _vprint('# of wrong classified %s pixels = %d' %
                        (sort_class_name, wpn_here))

                # write to files
                image = images[k]

                blended_labels = utils.blend_labels(
                    image, all_pred_labels[ind])
                blended_alphas = utils.blend_alphas(image, all_pred_masks[ind])
                blended_alphas = (blended_alphas - np.min(blended_alphas)) / \
                    (np.max(blended_alphas) - np.min(blended_alphas))
                blended_labels_gt = utils.blend_labels(
                    image, all_gt_labels[ind])

                if has_original_data:
                    blended_labels_in_original = utils.blend_labels(
                        original_image, all_pred_labels_in_original[ind])
                    blended_alphas_in_original = utils.blend_alphas(
                        original_image, all_pred_masks_in_original[ind])
                    blended_alphas_in_original = (blended_alphas_in_original - np.min(blended_alphas_in_original)) / \
                        (np.max(blended_alphas_in_original) -
                         np.min(blended_alphas_in_original))

                if gui:
                    #plt.imshow(blended_alphas)
                    #plt.show()
                    pass

                if save_images:
                    im_fname = os.path.basename(
                        dataset_val.source_image_link(ind))
                    folder = os.path.join(eval_dir, '%.4f_%05d_%s' % (
                        wpn_here / (self.options['image_size']**2), ind, im_fname))
                    # folder = os.path.join(eval_dir, os.path.basename(dataset_val.source_image_link(ind)))
                    folder = folder.replace('\\', '/')
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    assert os.path.exists(folder)
                    io.imsave(os.path.join(
                        folder, 'input.jpg').replace('\\', '/'), image)

                    if 'pred_head_boxes' in locals():
                        boxes = pred_head_boxes[k]  # k x (y1 x1 y2 x2)
                        boxes = np.clip(
                            boxes, 0, self.options['image_size'] - 1)
                        labels_with_boxes = utils.blend_labels(
                            None, all_pred_labels[ind])
                        for kk in range(boxes.shape[0]):
                            y1, x1, y2, x2 = boxes[kk].astype(int)
                            # draw boxes
                            labels_with_boxes[draw.line(y1, x1, y1, x2)] = 1.0
                            labels_with_boxes[draw.line(y1, x2, y2, x2)] = 1.0
                            labels_with_boxes[draw.line(y2, x2, y2, x1)] = 1.0
                            labels_with_boxes[draw.line(y2, x1, y1, x1)] = 1.0
                        io.imsave(os.path.join(
                            folder, 'labels_with_boxes.jpg').replace('\\', '/'), labels_with_boxes)

                    io.imsave(os.path.join(
                        folder, 'blended_labels.png').replace('\\', '/'), blended_labels)
                    io.imsave(os.path.join(
                        folder, 'blended_labels_gt.png').replace('\\', '/'), blended_labels_gt)
                    io.imsave(os.path.join(
                        folder, 'blended_alphas.png').replace('\\', '/'), blended_alphas)

                    # io.imsave(os.path.join(
                    #     folder, 'blended_labels128.jpg'), blended_labels128)
                    # io.imsave(os.path.join(
                    #     folder, 'blended_labels_gt128.jpg'), blended_labels_gt128)
                    # io.imsave(os.path.join(
                    #     folder, 'blended_alphas128.jpg'), blended_alphas128)

                    io.imsave(os.path.join(
                        folder, 'blended_labels_in_original.jpg').replace('\\', '/'), blended_labels_in_original)
                    # io.imsave(os.path.join(
                    #     folder, 'blended_labels_gt_in_original.jpg').replace('\\', '/'), blended_labels_gt_in_original)
                    io.imsave(os.path.join(
                        folder, 'blended_alphas_in_original.jpg').replace('\\', '/'), blended_alphas_in_original)

            start_id += batch_size
            if stop_id >= num_images:
                break

        all_hists = np.stack(all_hists, axis=0)[:num_images]
        all_gt_labels = np.stack(
            all_gt_labels, axis=0)[:num_images]
        all_pred_labels = np.stack(
            all_pred_labels, axis=0)[:num_images]
        # all_pred_masks = np.stack(all_pred_masks, axis=0)[:num_images]
        # all_pred_landmark68_pts = np.stack(
        #     all_pred_landmark68_pts, axis=0)[:num_images]

        # all_hists128 = np.stack(all_hists128, axis=0)[:num_images]

        # all_hists_in_original = np.stack(
        #     all_hists_in_original, axis=0)[:num_images]
        # all_gt_labels_in_original = np.stack(
        #     all_gt_labels_in_original, axis=0)[:num_images]
        # all_pred_labels_in_original = np.stack(
        #     all_pred_labels_in_original, axis=0)[:num_images]
        # all_pred_masks_in_original = np.stack(
        #     all_pred_masks_in_original, axis=0)[:num_images]
        # all_pred_landmark68_pts_in_original = np.stack(
        #     all_pred_landmark68_pts_in_original, axis=0)[:num_images]

        hists = np.sum(all_hists, axis=0)
        # hists128 = np.sum(all_hists128, axis=0)
        # hists_in_original = np.sum(all_hists_in_original, axis=0)

        f1s = _collect_f1(hists)
        for name, f1 in f1s.items():
            print('#f1.aligned of %s\t\t=%f' % (name, f1))
        # f1s128 = _collect_f1(hists128)
        # for name, f1 in f1s128.items():
        #     print('#f1.aligned128 of %s\t\t=%f' % (name, f1))
        # f1s_in_original = _collect_f1(hists_in_original)
        # for name, f1 in f1s_in_original.items():
            # print('#f1.original of %s\t\t=%f' % (name, f1))

        if save_stats:
            with open(os.path.join(eval_dir, 'f1s.csv'), 'w') as csv_file:
                for name in f1s.keys():
                    csv_file.write(',%s' % name)
                csv_file.write('\n')
                csv_file.write('f1s aligned')
                for name in f1s.keys():
                    csv_file.write(',%f' % f1s[name])
                csv_file.write('\n')
                # csv_file.write('f1s 128 aligned')
                # for name in f1s.keys():
                #     csv_file.write(',%f' % f1s128[name])
                # csv_file.write('\n')
                # csv_file.write('f1s original')
                # for name in f1s.keys():
                #     csv_file.write(',%f' % f1s_in_original[name])

            scio.savemat(
                os.path.join(eval_dir, 'data.mat'),
                {
                    'all_hists': all_hists,
                    # 'all_hists128': all_hists128,
                    # 'all_hists_in_original': all_hists_in_original,
                    'f1s': f1s,
                    # 'f1s128': f1s128,
                    # 'f1s_in_original': f1s_in_original,
                    'all_gt_labels': all_gt_labels,
                    'all_pred_labels': all_pred_labels
                },
                do_compression=True)
        return f1s[sort_class_name]

    def predict(self, dataset, verbose=True):
        '''
        for ind, preds in model.predict(dataset):
            ...
        '''
        assert self.mode == 'inference'
        # including background
        batch_size = self.options['images_per_gpu'] * self.gpu_count
        generator = data_generator(
            dataset,
            self.required_data_names(self.mode),
            shuffle=False,
            batch_size=batch_size)

        start_id = 0
        for inputs, _ in generator:
            stop_id = start_id + batch_size
            if verbose:
                print(f'processing {start_id}-{stop_id-1}')
            start_time = time.time()
            preds = self.keras_model.predict(inputs, verbose=0)
            elapsed_time_each_batch = time.time() - start_time
            if verbose:
                print(f'elapsed_time_each_batch={elapsed_time_each_batch}')

            if not isinstance(preds, list):
                preds = [preds]

            for k in range(batch_size):
                ind = (start_id + k) % dataset.num_images
                if ind >= dataset.num_images:
                    break
                each_pred = [pred[k] for pred in preds]
                yield ind, each_pred

    # def predict_masks(self,
    #                   images,
    #                   landmark68_pts):
    #     assert self.mode == 'inference'
    #     im_size = self.options['image_size']
    #     assert images.shape[1] == im_size \
    #         and images.shape[2] == im_size \
    #         and images.shape[3] == 3
    #     assert landmark68_pts.shape[1] == 68 \
    #         and landmark68_pts.shape[2] == 2
    #     batch_size = self.options['images_per_gpu'] * self.gpu_count
    #     assert images.shape[0] == batch_size and \
    #         landmark68_pts.shape[0] == batch_size

    #     molded_images = images - utils.MEAN_PIXEL
    #     molded_landmark68_pts = (landmark68_pts / im_size).astype(np.float32)

    #     inputs = [
    #         np.ones([batch_size], np.uint8),
    #         np.ones([batch_size], np.uint8), molded_images,
    #         molded_landmark68_pts
    #     ]
    #     pred_masks = self.keras_model.predict(inputs, verbose=0)[0]
    #     assert pred_masks.shape[0] == batch_size

    #     # if Tbacks is not None:
    #     #     assert original_shape is not None
    #     #     assert Tbacks.shape[0] == batch_size
    #     #     t_pred_masks = np.transpose(pred_masks, [0, 2, 3, 1])
    #     #     for i in range(batch_size):
    #     #         trans = transform.ProjectiveTransform(matrix=Tbacks[i])
    #     #         t_pred_masks[i] = transform.warp(
    #     #             t_pred_masks[i],
    #     #             trans.inverse,
    #     #             output_shape=original_shape)
    #     #     pred_masks = np.transpose(t_pred_masks, [0, 3, 1, 2])

    #     flatten_pred_labels = np.zeros(
    #         [batch_size, pred_masks.shape[2], pred_masks.shape[3]], np.int32)
    #     # recover segmentation from masks
    #     # the latter labels covers the former labels
    #     for class_ids in self.options['heads']:
    #         # there is no order within the same head, choose by max
    #         bg_mask_this_head = 1.0 - np.sum(
    #             pred_masks[:, class_ids, :, :], axis=1, keepdims=True)
    #         fg_bg_masks_this_head = np.concatenate(
    #             [pred_masks[:, class_ids, :, :], bg_mask_this_head], axis=1)
    #         max_ids_this_head = np.argmax(fg_bg_masks_this_head, axis=1)
    #         for j, class_id in enumerate(class_ids):
    #             flatten_pred_labels[np.where(
    #                 max_ids_this_head == j)] = class_id + 1

    #     return pred_masks, flatten_pred_labels

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers
