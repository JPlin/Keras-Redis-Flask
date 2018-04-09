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
from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
from pprint import pprint

import utils
from arc_base import ArcBase
from utils import PyramidROIAlignAll, resnet_graph, vgg16_graph, \
    build_fpn, build_mask_heads, build_mask_heads_with_transfer_refine, \
    build_mask_heads_shared, \
    inverse_box_crop, BatchNorm
# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Architecture(ArcBase):
    def define_graph(self, mode, options):
        assert mode in ['training', 'inference']

        box_pred_method = options['box_pred_method']
        print(f'box_pred_method: {box_pred_method}')
        assert box_pred_method in [
            'lbf_guided', 'regress_landmark', 'regress_segbox', 'gt_segbox']

        batch_size = options['images_per_gpu']

        heads = options['heads']
        num_heads = len(heads)
        print(f'num_heads={num_heads}')
        num_masks = 0
        for class_ids in heads:
            num_masks += len(class_ids)
        assert num_masks == len(options['class_names'])
        print(f'num_masks={num_masks}')

        head_label_names = []
        for class_ids in heads:
            names_this_head = []
            for class_id in class_ids:
                names_this_head += options['class_names'][class_id]
            head_label_names.append(names_this_head)
        assert len(head_label_names) == len(heads)

        h = w = options['image_size']

        # assert h > 0 and w > 0 and h % 2**6 == 0 and w % 2**6 == 0
        if 'landmark_box_paddings448' in options:
            delta = options.get('landmark_box_padding_additional_ratio', 0.0)
            molded_padding_dict = {
                name:
                np.array(padding, np.float32) / 448.0 +
                    np.array([-delta, -delta, +delta, +delta], np.float32)
                for name, padding in options['landmark_box_paddings448'].items()
            }
        else:
            raise RuntimeError('padding information required')

        pprint(molded_padding_dict)

        # mean landmark68 pts
        mean_molded_landmark68_pts = tf.stack(
            [utils.MEAN_MOLDED_LANDMARK68_PTS],
            name='mean_molded_landmark68_pts')
        # mean head boxes
        mean_molded_head_boxes = utils.extract_landmark68_boxes_graph(
            mean_molded_landmark68_pts,
            head_label_names,
            molded_padding_dict)

        dropout_rate = options.get('dropout_rate', 0.0)
        print(f'dropout_rate={dropout_rate}')

        # Inputs
        input_molded_image = KL.Input(
            shape=[h, w, 3], name="input_molded_image")  # molded
        input_molded_image_exist = KL.Input(
            shape=[1], name='input_molded_image_exist', dtype=tf.uint8)
        print('input: %s' % input_molded_image.name)
        print('input_molded_image_exist.shape: {}, {}'.format(
            input_molded_image_exist.shape,
            input_molded_image_exist._keras_shape))

        if mode == 'training':
            input_gt_masks = KL.Input(
                shape=[num_masks, h, w], name="input_gt_masks")
            input_gt_masks_exist = KL.Input(
                shape=[1], name='input_gt_masks_exist', dtype=tf.uint8)
            print('input_gt_masks_exist.shape: {}, {}'.format(
                input_gt_masks_exist.shape, input_gt_masks_exist._keras_shape))
            molded_gt_masks = KL.Lambda(lambda xx: tf.cast(xx, tf.float32))(
                input_gt_masks)

        if box_pred_method == 'lbf_guided':
            input_molded_lbf_landmark68_pts = KL.Input(
                shape=[68, 2],
                dtype=tf.float32,
                name="input_molded_lbf_landmark68_pts")
            input_molded_lbf_landmark68_pts_exist = KL.Input(
                shape=[1],
                name='input_molded_lbf_landmark68_pts_exist',
                dtype=tf.uint8)
            print('input_molded_lbf_landmark68_pts_exist.shape: {}, {}'.format(
                input_molded_lbf_landmark68_pts_exist.shape,
                input_molded_lbf_landmark68_pts_exist._keras_shape))

        elif box_pred_method == 'regress_landmark':
            if mode == 'training':
                input_gt_molded_landmark68_pts = KL.Input(
                    shape=[68, 2],
                    dtype=tf.float32,
                    name='input_gt_molded_landmark68_pts')
                input_gt_molded_landmark68_pts_exist = KL.Input(
                    shape=[1],
                    name='input_gt_molded_landmark68_pts_exist', dtype=tf.uint8)

        elif box_pred_method == 'regress_segbox':
            def _box_to_std_deform(box):
                return utils.compute_box_deform(mean_molded_head_boxes, box)

            def _std_deform_to_box(deform):
                return utils.apply_box_deform(mean_molded_head_boxes, deform)

            if mode == 'training':
                input_gt_molded_head_boxes = KL.Input(
                    shape=[num_heads, 4],
                    dtype=tf.float32,
                    name='input_gt_molded_head_boxes')
                input_gt_molded_head_boxes_exist = KL.Input(
                    shape=[1],
                    name='input_gt_molded_head_boxes_exist', dtype=tf.uint8)

                # get box deforms
                input_gt_head_box_deforms = KL.Lambda(
                    _box_to_std_deform,
                    name='input_gt_head_box_deforms')(
                        input_gt_molded_head_boxes)

        elif box_pred_method == 'gt_segbox':
            input_gt_molded_head_boxes = KL.Input(
                shape=[num_heads, 4],
                dtype=tf.float32,
                name='input_gt_molded_head_boxes')
            input_gt_molded_head_boxes_exist = KL.Input(
                shape=[1],
                name='input_gt_molded_head_boxes_exist', dtype=tf.uint8)

        # Construct Backbone Network
        box_from = options.get('box_from', 'P2')

        def _expand_boxes_by_ratio(boxes, rel_ratio):
            y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)
            cy = (y1 + y2) / 2.0
            cx = (x1 + x2) / 2.0
            h2 = (y2 - y1) / 2.0
            w2 = (x2 - x1) / 2.0
            yy1 = cy - h2 * (1 + rel_ratio)
            xx1 = cx - w2 * (1 + rel_ratio)
            yy2 = cy + h2 * (1 + rel_ratio)
            xx2 = cx + w2 * (1 + rel_ratio)
            return tf.concat([yy1, xx1, yy2, xx2], axis=-1)

        if options['backbone'] == 'vgg16':
            print('making vgg16 backbone')
            C1, C2, C3, C4, C5 = vgg16_graph(input_molded_image)
            assert box_from == 'C5'
            mrcnn_feature_maps = [C5]
        elif options['backbone'] == 'vgg16fpn':
            print('making vgg16fpn backbone')
            C1, C2, C3, C4, C5 = vgg16_graph(input_molded_image)

            P2, P3, P4, P5, _ = build_fpn([C1, C2, C3, C4, C5])
            if box_from == 'P2':
                box_feature = P2
            elif box_from == 'C5':
                box_feature = C5
            elif box_from == 'C4':
                box_feature = C4
            mrcnn_feature_maps = [P2, P3, P4, P5]
        elif options['backbone'] == 'vgg16fpnP2':
            print('making vgg16fpnP2 backbone')
            C1, C2, C3, C4, C5 = vgg16_graph(input_molded_image)

            P2, P3, P4, P5, _ = build_fpn([C1, C2, C3, C4, C5])
            if box_from == 'P2':
                box_feature = P2
            elif box_from == 'C5':
                box_feature = C5
            elif box_from == 'C4':
                box_feature = C4
            mrcnn_feature_maps = [P2]
        elif options['backbone'] == 'resnet50':
            C1, C2, C3, C4, _ = resnet_graph(
                input_molded_image, 'resnet50', False)
            assert box_from == 'C4'
            box_feature = C4
            mrcnn_feature_maps = [C4]
        elif options['backbone'] == 'resnet50fpn':
            print('making resnet50fpn backbone')
            C1, C2, C3, C4, C5 = resnet_graph(
                input_molded_image, 'resnet50', True)

            P2, P3, P4, P5, _ = build_fpn([C1, C2, C3, C4, C5])
            if box_from == 'P2':
                box_feature = P2
            elif box_from == 'C5':
                box_feature = C5
            elif box_from == 'C4':
                box_feature = C4
            mrcnn_feature_maps = [P2, P3, P4, P5]
        elif options['backbone'] == 'resnet50fpnP2':
            print('making resnet50fpnP2 backbone')
            C1, C2, C3, C4, C5 = resnet_graph(
                input_molded_image, 'resnet50', True)

            P2, P3, P4, P5, _ = build_fpn([C1, C2, C3, C4, C5])
            if box_from == 'P2':
                box_feature = P2
            elif box_from == 'C5':
                box_feature = C5
            elif box_from == 'C4':
                box_feature = C4
            mrcnn_feature_maps = [P2]
        elif options['backbone'] == 'resnet50fpnC4':
            C1, C2, C3, C4, C5 = resnet_graph(
                input_molded_image, 'resnet50', True)

            P2, P3, P4, P5, _ = build_fpn([C1, C2, C3, C4, C5])
            if box_from == 'P2':
                box_feature = P2
            elif box_from == 'C5':
                box_feature = C5
            elif box_from == 'C4':
                box_feature = C4
            mrcnn_feature_maps = [C4]
        else:
            raise NotImplementedError()

        if box_pred_method in ['regress_landmark', 'regress_segbox']:
            # get box and optionally landmarks
            with tf.name_scope('box_neck'):
                x = box_feature
                box_neck_conv_num = options['box_neck_conv_num']
                for k in range(box_neck_conv_num):
                    x = KL.Conv2D(320, (3, 3), strides=(1, 1),
                                  padding='same', name=f'box_conv{k}')(x)
                    x = KL.BatchNormalization(name=f'box_convbn{k}')(x)
                    x = KL.Activation('relu')(x)

                x = KL.Conv2D(1280, (1, 1), name='box_conv_last')(x)
                x = KL.BatchNormalization(name=f'box_convbn_last')(x)

                x = KL.GlobalAveragePooling2D()(x)
                x = KL.Dropout(dropout_rate)(x)
            box_feature = x
            print(f'box_feature.shape={box_feature.shape}')

        if box_pred_method == 'lbf_guided':
            molded_head_boxes = KL.Lambda(
                lambda xx: utils.extract_landmark68_boxes_graph(
                    xx, head_label_names, molded_padding_dict),
                name='molded_head_boxes')(input_molded_lbf_landmark68_pts)

        elif box_pred_method == 'regress_landmark':
            x = box_feature
            x = KL.Dense(68 * 2, name='box_landmark_fc')(x)
            x = KL.Reshape((68, 2))(x)  # landmark68 offsets

            pred_molded_landmark68_pts = KL.Lambda(
                lambda xx: xx + mean_molded_landmark68_pts,
                name='pred_molded_landmark68_pts')(x)
            molded_head_boxes = KL.Lambda(
                lambda xx: utils.extract_landmark68_boxes_graph(
                    xx, head_label_names, molded_padding_dict),
                name='molded_head_boxes')(pred_molded_landmark68_pts)

            # compute landmark loss
            if mode == 'training':
                # Point loss
                def _l2_loss(pts1, pts2):
                    # (batch, 68, 2)
                    return tf.reduce_mean(
                        tf.norm(pts1 - pts2, axis=-1), axis=-1)
                landmark68_loss = KL.Lambda(lambda xx: _l2_loss(xx[0], xx[1]))(
                    [pred_molded_landmark68_pts, input_gt_molded_landmark68_pts])
                landmark68_loss = KL.Lambda(
                    lambda xx: tf.where(
                        tf.reshape(xx[0] > 0, tf.shape(xx[1])),
                        xx[1], tf.zeros_like(xx[1])),
                    name='landmark68_loss')([
                        input_gt_molded_landmark68_pts_exist, landmark68_loss])
                print('landmark68_loss.shape={}, {}'.format(
                    landmark68_loss.shape, landmark68_loss._keras_shape))

        elif box_pred_method == 'regress_segbox':
            x = box_feature
            x = KL.Dense(num_heads * 4, name='box_fc')(x)

            use_rpn_box_loss = options.get('use_rpn_box_loss', True)
            print(f'use_rpn_box_loss={use_rpn_box_loss}')

            if use_rpn_box_loss:
                pred_head_box_deforms = KL.Reshape(
                    (num_heads, 4))(x)  # box deforms

                pred_molded_head_boxes = KL.Lambda(
                    _std_deform_to_box, name='pred_molded_head_boxes')(
                    pred_head_box_deforms)
                head_box_padding_ratio = options['head_box_padding_ratio']
                molded_head_boxes = KL.Lambda(lambda xx: tf.stop_gradient(
                    xx + tf.constant([
                        - head_box_padding_ratio,
                        - head_box_padding_ratio,
                        head_box_padding_ratio,
                        head_box_padding_ratio
                    ], tf.float32)), name='molded_head_boxes')(pred_molded_head_boxes)

                # compute segbox loss
                if mode == 'training':
                    # Box loss
                    use_soft_l1_loss = options.get('use_soft_l1_loss', True)

                    def _l1_loss(box_deform1, box_deform2):
                        # (batch, num_heads, 4)
                        if use_soft_l1_loss:
                            return tf.reduce_mean(
                                tf.sqrt(tf.square(box_deform1 -
                                                  box_deform2) + K.epsilon()),
                                axis=[1, 2])
                        else:
                            return tf.reduce_mean(tf.abs(box_deform1 - box_deform2), axis=[1, 2])
                    box_loss = KL.Lambda(lambda xx: _l1_loss(xx[0], xx[1]))(
                        [input_gt_head_box_deforms, pred_head_box_deforms])
                    box_loss = KL.Lambda(
                        lambda xx: tf.where(tf.reshape(
                            xx[0] > 0, tf.shape(xx[1])), xx[1], tf.zeros_like(xx[1])),
                        name='box_loss')([
                            input_gt_molded_head_boxes_exist,
                            box_loss])

                    print('box_loss.shape={}, {}'.format(
                        box_loss.shape, box_loss._keras_shape))
            else:
                pred_molded_head_boxes = KL.Reshape((num_heads, 4))(x)

                head_box_padding_ratio = options['head_box_padding_ratio']
                molded_head_boxes = KL.Lambda(lambda xx: tf.stop_gradient(
                    xx + tf.constant([
                        - head_box_padding_ratio,
                        - head_box_padding_ratio,
                        head_box_padding_ratio,
                        head_box_padding_ratio
                    ], tf.float32)), name='molded_head_boxes')(pred_molded_head_boxes)

                # compute segbox loss
                if mode == 'training':
                    # Box loss
                    use_soft_l1_loss = options.get('use_soft_l1_loss', True)

                    def _l1_loss(box_deform1, box_deform2):
                        # (batch, num_heads, 4)
                        if use_soft_l1_loss:
                            return tf.reduce_mean(
                                tf.sqrt(tf.square(box_deform1 -
                                                  box_deform2) + K.epsilon()),
                                axis=[1, 2])
                        else:
                            return tf.reduce_mean(tf.abs(box_deform1 - box_deform2), axis=[1, 2])
                    box_loss = KL.Lambda(lambda xx: _l1_loss(xx[0], xx[1]))(
                        [input_gt_molded_head_boxes, pred_molded_head_boxes])
                    box_loss = KL.Lambda(
                        lambda xx: tf.where(tf.reshape(
                            xx[0] > 0, tf.shape(xx[1])), xx[1], tf.zeros_like(xx[1])),
                        name='box_loss')([
                            input_gt_molded_head_boxes_exist,
                            box_loss])

                    print('box_loss.shape={}, {}'.format(
                        box_loss.shape, box_loss._keras_shape))

        elif box_pred_method == 'gt_segbox':
            head_box_padding_ratio = options['head_box_padding_ratio']
            molded_head_boxes = KL.Lambda(lambda xx: tf.stop_gradient(
                xx + tf.constant([
                    - head_box_padding_ratio,
                    - head_box_padding_ratio,
                    head_box_padding_ratio,
                    head_box_padding_ratio
                ], tf.float32)), name='molded_head_boxes')(input_gt_molded_head_boxes)

        if 'fixed_head_box' in options:
            # replace certain molded_head_boxes with assigned ones
            fixed_head_box = options['fixed_head_box']
            fixed_head_box_flags = np.zeros((num_heads), np.uint8)
            fixed_head_box_values = np.zeros((num_heads, 4), np.float32)
            for head_id, box in fixed_head_box.items():
                fixed_head_box_flags[head_id] = 1
                fixed_head_box_values[head_id, :] = np.array(box, np.float32)
            print(f'fixed_head_box_flags={fixed_head_box_flags}')
            print(f'fixed_head_box_values={fixed_head_box_values}')

            fixed_head_box_flags = tf.tile(
                tf.expand_dims(tf.expand_dims(
                    tf.constant(fixed_head_box_flags), 0), -1),
                [tf.shape(molded_head_boxes)[0], 1, 4])
            fixed_head_box_values = tf.tile(
                tf.expand_dims(tf.constant(fixed_head_box_values), 0),
                [tf.shape(molded_head_boxes)[0], 1, 1])
            molded_head_boxes = KL.Lambda(lambda xx: tf.where(
                fixed_head_box_flags, fixed_head_box_values, xx))(molded_head_boxes)

        # visualize pts and boxes
        # with tf.name_scope('boxes_pts'):

        #     def _show_boxes_pts(im, boxes, pts=None):
        #         return visualize.tf_display_boxes_pts(
        #             im, boxes, pts, utils.MEAN_PIXEL)

        #     show_num = min(batch_size, 3)
        #     if box_pred_method == 'regress_landmark':
        #         label_pts = [('pred_molded_landmark68_pts',
        #                       pred_molded_landmark68_pts)]
        #         if mode == 'training':
        #             label_pts.append(
        #                 ('input_gt_molded_landmark68_pts', input_gt_molded_landmark68_pts))
        #         for label, pts in label_pts:
        #             plot_ims = []
        #             for k in range(show_num):
        #                 im = tfplot.ops.plot(_show_boxes_pts, [
        #                     input_molded_image[k, :, :, :],
        #                     molded_head_boxes[k, :, :],
        #                     pts[k, :, :]])
        #                 plot_ims.append(im)
        #             plot_ims = tf.stack(plot_ims, axis=0)
        #             tf.summary.image(
        #                 name=label, tensor=plot_ims)
        #     else:
        #         plot_ims = []
        #         for k in range(show_num):
        #             im = tfplot.ops.plot(_show_boxes_pts, [
        #                 input_molded_image[k, :, :, :],
        #                 molded_head_boxes[k, :, :]])
        #             plot_ims.append(im)
        #         plot_ims = tf.stack(plot_ims, axis=0)
        #         tf.summary.image(
        #             name='molded_head_boxes', tensor=plot_ims)

        # Construct Head Networks
        head_class_nums = [len(class_ids) for class_ids in heads]

        # ROI Pooling
        pool_size = options.get('pool_size', 56)
        deconv_num = options.get('deconv_num', 2)
        conv_num = options.get('conv_num', 1)

        molded_head_boxes = KL.Lambda(tf.stop_gradient)(molded_head_boxes)

        aligned = PyramidROIAlignAll(
            [pool_size, pool_size], name="roi_align_mask")(
                [molded_head_boxes] + mrcnn_feature_maps)
        # print(aligned._keras_shape)

        fg_masks = [None] * num_heads
        bg_masks = [None] * num_heads

        def _slice_lambda(index):
            return lambda xx: xx[:, index, :, :, :]

        head_mask_features = [None] * num_heads
        for i in range(num_heads):
            x = KL.Lambda(_slice_lambda(i))(aligned)

            for k in range(conv_num):
                x = KL.Conv2D(
                    256, (3, 3),
                    padding="same",
                    name=f"mrcnn_mask_conv{k+1}_{i}")(x)
                x = BatchNorm(axis=-1, name=f'mrcnn_mask_bn{k+1}_{i}')(x)
                x = KL.Activation('relu')(x)
                if dropout_rate > 0:
                    x = KL.Dropout(dropout_rate)(x)

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

        for i in range(num_heads):
            x = head_mask_features[i]
            num_classes_this_head = head_class_nums[i]
            assert num_classes_this_head > 0

            x = KL.Conv2D(
                1 + num_classes_this_head, (1, 1), strides=1,
                name='mrcnn_mask_conv_last_%d' % i,
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
            mrcnn_fg_masks = KL.Lambda(
                lambda xx: tf.concat(xx, axis=1), name='mrcnn_fg_masks')(fg_masks)
        else:
            mrcnn_fg_masks = KL.Lambda(
                lambda xx: xx, name='mrcnn_fg_masks')(fg_masks[0])

        if len(bg_masks) > 1:
            mrcnn_bg_masks = KL.Lambda(
                lambda xx: tf.stack(xx, axis=1), name='mrcnn_bg_masks')(bg_masks)
        else:
            mrcnn_bg_masks = KL.Lambda(
                lambda xx: tf.expand_dims(xx, axis=1),
                name='mrcnn_bg_masks')(bg_masks[0])

        # [batch, num_masks+num_heads, height, width]
        mrcnn_masks = KL.Concatenate(
            axis=1, name='mrcnn_masks')([mrcnn_fg_masks, mrcnn_bg_masks])
        print('mrcnn_masks.shape={}, {}'.format(mrcnn_masks.shape,
                                                mrcnn_masks._keras_shape))

        def _tile_by_head_classes(data):
            tiled = [None] * num_masks
            for i, class_ids in enumerate(heads):
                for class_id in class_ids:
                    tiled[class_id] = data[:, i]
            assert None not in tiled
            return tf.stack(tiled, axis=1)

        # Unmold masks back to image view
        def _unmold_mask(masks, boxes):
            # masks: (batch, num_masks, h, w)
            # boxes: (batch, num_heads, 4)
            mask_h, mask_w = tf.shape(masks)[2], tf.shape(masks)[3]

            # (batch, num_masks, 4)
            boxes = _tile_by_head_classes(boxes)

            masks = tf.reshape(masks, (-1, mask_h, mask_w))
            boxes = tf.reshape(boxes, (-1, 4))

            unmolded_masks = inverse_box_crop(masks, boxes, [h, w])
            unmolded_masks = tf.reshape(unmolded_masks, (-1, num_masks, h, w))
            return unmolded_masks

        output_masks = KL.Lambda(
            lambda xx: _unmold_mask(xx[0], xx[1]),
            name='output_masks')([mrcnn_fg_masks, molded_head_boxes])
        print('output_masks.shape={}, {}'.format(
            output_masks.shape, output_masks._keras_shape))

        # if options.get('full_view_mask_loss', False):

        if mode == "training":
            head_mask_shape = [mask_feature_size, mask_feature_size]
            print('head_mask_shape={}'.format(head_mask_shape))

            # mask loss
            # extract target gt fg masks
            def _extract_gt_fg_batched(gt_masks, boxes):
                # gt_masks: [batch, num_masks, h, w]
                # boxes: [batch, num_heads, 4]

                # [batch * num_masks, h, w, 1]
                gt_masks = tf.reshape(gt_masks, [-1, h, w, 1])

                # [batch, num_masks, 4]
                boxes = _tile_by_head_classes(boxes)
                # [batch * num_masks, 4]
                boxes = tf.reshape(boxes, [-1, 4])

                # [batch * num_masks, mask_h, mask_w]
                target_masks = tf.image.crop_and_resize(
                    gt_masks, boxes, tf.range(tf.shape(gt_masks)[0]),
                    head_mask_shape)
                target_masks = tf.reshape(target_masks,
                                          [-1, num_masks] + head_mask_shape)
                return target_masks

            target_gt_fg_masks = KL.Lambda(
                lambda xx: _extract_gt_fg_batched(xx[0], xx[1]))(
                    [molded_gt_masks, molded_head_boxes])

            # extract target gt bg masks
            def _extract_gt_bg_batched(gt_masks, boxes):
                # gt_masks: [batch, num_masks, h, w]
                # boxes: [batch, num_heads, 4]

                gt_bg_masks = [None] * num_heads
                for i, class_ids in enumerate(heads):
                    gt_masks_this_head = [None] * len(class_ids)
                    for j, class_id in enumerate(class_ids):
                        # each of [batch, h, w]
                        gt_masks_this_head[j] = gt_masks[:, class_id, :, :]
                    # [batch, len(class_ids), h, w]
                    gt_masks_this_head = tf.stack(gt_masks_this_head, axis=1)
                    # [batch, h, w]
                    gt_bg_masks[i] = 1.0 - tf.reduce_max(
                        gt_masks_this_head, axis=1)

                # [batch, num_heads, h, w]
                gt_bg_masks = tf.stack(gt_bg_masks, axis=1)
                # [batch * num_heads, h, w, 1]
                gt_bg_masks = tf.reshape(gt_bg_masks, [-1, h, w, 1])

                # [batch * num_heads, 4]
                boxes = tf.reshape(boxes, [-1, 4])

                # [batch * num_heads, mask_h, mask_w]
                target_masks = tf.image.crop_and_resize(
                    gt_bg_masks, boxes, tf.range(tf.shape(gt_bg_masks)[0]),
                    head_mask_shape, extrapolation_value=1)  # !!!
                target_masks = tf.reshape(target_masks,
                                          [-1, num_heads] + head_mask_shape)
                return target_masks

            target_gt_bg_masks = KL.Lambda(
                lambda xx: _extract_gt_bg_batched(xx[0], xx[1]))(
                    [molded_gt_masks, molded_head_boxes])

            target_gt_masks = KL.Concatenate(
                axis=1, name='target_gt_masks')(
                    [target_gt_fg_masks, target_gt_bg_masks])
            print('target_gt_masks.shape={}, {}'.format(
                target_gt_masks.shape, target_gt_masks._keras_shape))

            mask_loss_im = KL.Lambda(
                lambda xx: K.binary_crossentropy(target=xx[0], output=xx[1]),
                name="mask_ls_im")([target_gt_masks, mrcnn_masks])
            print('mask_loss_im.shape: {} {}'.format(mask_loss_im._keras_shape,
                                                     mask_loss_im.shape))

            mask_loss_im_reduced = KL.Lambda(
                lambda xx: tf.reduce_mean(xx, axis=[2, 3]),
                name='mask_loss_im_reduced')(mask_loss_im)

            def _get_individual_losses(loss_im, name, index):
                return KL.Lambda(
                    lambda xx: tf.reduce_mean(xx[:, index], axis=[1, 2]),
                    name=name)(loss_im)

            # visualization
            with tf.name_scope('original_masks'):
                for i, class_ids in enumerate(heads):
                    for j, class_id in enumerate(class_ids):
                        name = head_label_names[i][j]
                        fg_target_pred_original_view = tf.expand_dims(tf.concat([
                            tf.cast(
                                input_gt_masks[:, class_id, :, :], tf.float32),
                            output_masks[:, class_id, :, :]], axis=-1), axis=-1)
                        tf.summary.image(
                            f'fg_target_pred_original_view_{i}_{name}',
                            fg_target_pred_original_view)

            with tf.name_scope('cropped_masks'):
                for i, class_ids in enumerate(heads):
                    for j, class_id in enumerate(class_ids):
                        name = head_label_names[i][j]
                        fg_target_pred_loss = tf.expand_dims(tf.concat([
                            target_gt_fg_masks[:, class_id, :, :],
                            mrcnn_fg_masks[:, class_id, :, :],
                            mask_loss_im[:, class_id]], axis=-1), axis=-1)
                        tf.summary.image(
                            f'fg_target_pred_loss_{name}', fg_target_pred_loss)
                    bg_target_pred_loss = tf.expand_dims(tf.concat([
                        target_gt_bg_masks[:, i, :, :],
                        mrcnn_bg_masks[:, i, :, :],
                        mask_loss_im[:, i + num_masks]], axis=-1), axis=-1)
                    tf.summary.image(
                        f'bg_target_pred_loss_{i}', bg_target_pred_loss)

            mask_loss = KL.Lambda(
                lambda xx: tf.reduce_mean(xx, axis=[1]))(mask_loss_im_reduced)
            mask_loss = KL.Lambda(
                lambda xx: tf.where(tf.reshape(
                    xx[0] > 0, tf.shape(xx[1])), xx[1], tf.zeros_like(xx[1])),
                name='mask_loss')([input_gt_masks_exist, mask_loss])
            print('mask_loss.shape={}, {}'.format(mask_loss.shape,
                                                  mask_loss._keras_shape))

            if box_pred_method == 'lbf_guided':
                inputs = [
                    input_molded_image_exist,
                    input_gt_masks_exist,
                    input_molded_lbf_landmark68_pts_exist,
                    input_molded_image,
                    input_gt_masks,
                    input_molded_lbf_landmark68_pts
                ]
                outputs = [mask_loss]
            elif box_pred_method == 'regress_landmark':
                inputs = [
                    input_molded_image_exist,
                    input_gt_masks_exist,
                    input_gt_molded_landmark68_pts_exist,
                    input_molded_image,
                    input_gt_masks,
                    input_gt_molded_landmark68_pts
                ]
                outputs = [mask_loss, landmark68_loss]
            elif box_pred_method == 'regress_segbox':
                inputs = [
                    input_molded_image_exist,
                    input_gt_masks_exist,
                    input_gt_molded_head_boxes_exist,
                    input_molded_image,
                    input_gt_masks,
                    input_gt_molded_head_boxes,
                ]
                outputs = [mask_loss, box_loss]
            elif box_pred_method == 'gt_segbox':
                inputs = [
                    input_molded_image_exist,
                    input_gt_masks_exist,
                    input_gt_molded_head_boxes_exist,
                    input_molded_image,
                    input_gt_masks,
                    input_gt_molded_head_boxes
                ]
                outputs = [mask_loss]
        else:
            if box_pred_method == 'lbf_guided':
                inputs = [
                    input_molded_image_exist,
                    input_molded_lbf_landmark68_pts_exist,
                    input_molded_image,
                    input_molded_lbf_landmark68_pts
                ]
                outputs = [
                    output_masks,
                    molded_head_boxes
                ]
            elif box_pred_method == 'regress_landmark':
                inputs = [
                    input_molded_image_exist,
                    input_molded_image
                ]
                outputs = [
                    output_masks,
                    molded_head_boxes,
                    pred_molded_landmark68_pts
                ]
            elif box_pred_method == 'regress_segbox':
                inputs = [
                    input_molded_image_exist,
                    input_molded_image
                ]
                outputs = [
                    output_masks,
                    molded_head_boxes
                ]
            elif box_pred_method == 'gt_segbox':
                inputs = [
                    input_molded_image_exist,
                    input_gt_molded_head_boxes_exist,
                    input_molded_image,
                    input_gt_molded_head_boxes
                ]
                outputs = [
                    output_masks,
                    molded_head_boxes
                ]
        return [inputs, outputs]

    def required_data_names(self, mode=None):
        if mode is None:
            mode = self.mode
        assert mode in ['training', 'inference']
        box_pred_method = self.options['box_pred_method']
        if mode == 'training':
            if box_pred_method == 'lbf_guided':
                return ['molded_image', 'masks', 'molded_lbf_landmark68_pts']
            elif box_pred_method == 'regress_landmark':
                return ['molded_image', 'masks', 'molded_landmark68_pts']
            elif box_pred_method == 'regress_segbox':
                return ['molded_image', 'masks', 'molded_head_boxes']
            elif box_pred_method == 'gt_segbox':
                return ['molded_image', 'masks', 'molded_head_boxes']
        else:
            if box_pred_method == 'lbf_guided':
                return ['molded_image', 'molded_lbf_landmark68_pts']
            elif box_pred_method == 'gt_segbox':
                return ['molded_image', 'molded_head_boxes']
            else:
                return ['molded_image']
