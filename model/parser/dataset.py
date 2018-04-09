import os
import sys
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import yaml
import random
import math
import socket
import numpy as np
from scipy.io import loadmat, savemat
from skimage import io, color, draw, exposure, transform

import utils
import paths

DATASET_HANDLERS = {}


class Dataset(object):
    ''' The base class of all datasets
    '''

    def __init__(self, options, pretend_not_exist_data_names=None):
        ''' pretend_not_exist_data_names only works in load_data_as_list(...)
        '''
        self.options = options
        self._image_ids = []
        self.image_info = []

        self.im_size = self.options['image_size']
        # assert self.im_size in [128, 256, 448, 512]

        self.heads = self.options['heads']
        self.num_heads = len(self.heads)
        self.num_classes = len(self.options['class_names'])

        self.head_label_names = []
        for class_ids in self.heads:
            names_this_head = []
            for class_id in class_ids:
                names_this_head += self.options['class_names'][class_id]
            self.head_label_names.append(names_this_head)

        # (image_id, pre) -> (data, exists)
        self.load_data_handlers = {
            'image': self.load_image,
            'molded_image': self.load_molded_image,
            'masks': self.load_masks,

            'landmark68_pts': self.load_landmark68_pts,
            'molded_landmark68_pts': self.load_molded_landmark68_pts,
            'lbf_landmark68_pts': self.load_lbf_landmark68_pts,
            'molded_lbf_landmark68_pts': self.load_molded_lbf_landmark68_pts,

            'mask_boxes': self.load_mask_boxes,
            'molded_mask_boxes': self.load_molded_mask_boxes,
            'head_boxes': self.load_head_boxes,
            'molded_head_boxes': self.load_molded_head_boxes,
            'molded_pad_head_boxes': self.load_molded_pad_head_boxes,

            'lbf_landmark68_boxes': self.load_lbf_landmark68_boxes,
            'molded_lbf_landmark68_boxes': self.load_molded_lbf_landmark68_boxes,

            'original_image_shape': self.load_original_image_shape,
            'original_image': self.load_original_image,
            'original_masks': self.load_original_masks,

            'align_matrix': self.load_align_matrix
        }
        self.pretend_not_exist_data_names = \
            pretend_not_exist_data_names \
            if pretend_not_exist_data_names is not None else set()

    def load_data_as_list(self, image_id, data_names):
        pre = self.prepare_loading(image_id)
        results = [
            self.load_data_handlers[data_name](image_id, pre)
            for data_name in data_names
        ]
        for i, data_name in enumerate(data_names):
            if data_name in self.pretend_not_exist_data_names:
                results[i] = (results[i][0], False)
        return results

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def make_image_ids(self):
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

    def prepare_loading(self, image_id):
        return None

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id, pre):
        return np.zeros([self.im_size, self.im_size, 3], np.uint8), False

    def load_molded_image(self, image_id, pre):
        im, exists = self.load_image(image_id, pre)
        if exists:
            return im - utils.MEAN_PIXEL, True
        else:
            return im, False

    def load_masks(self, image_id, pre):
        return np.zeros([self.num_classes, self.im_size, self.im_size],
                        np.uint8), False

    def load_landmark68_pts(self, image_id, pre):
        return np.zeros([68, 2], np.float32), False

    def load_molded_landmark68_pts(self, image_id, pre):
        pts, exists = self.load_landmark68_pts(image_id, pre)
        if exists:
            return (pts / self.im_size).astype(np.float32), True  # attention!
        else:
            return pts, False

    def load_lbf_landmark68_pts(self, image_id, pre):
        return np.zeros([68, 2], np.float32), False

    def load_molded_lbf_landmark68_pts(self, image_id, pre):
        pts, exists = self.load_lbf_landmark68_pts(image_id, pre)
        if exists:
            return (pts / self.im_size).astype(np.float32), True
        else:
            return pts, False

    def load_mask_boxes(self, image_id, pre):
        masks, exists = self.load_masks(image_id, pre)
        if exists:
            # get bounding boxes from masks
            num_masks, h, w = masks.shape
            assert h == self.im_size and w == self.im_size
            assert num_masks == self.num_classes

            boxes = np.zeros([num_masks, 4], np.float32)
            for i in range(num_masks):
                m = masks[i, :, :]
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
            return boxes, True
        else:
            return np.zeros([masks.shape[0], 4], np.float32), False

    def load_head_boxes(self, image_id, pre=None):
        mask_boxes, exists = self.load_mask_boxes(image_id, pre)
        if exists:
            valid_box_inds = np.where(np.any(mask_boxes != 0, axis=-1))
            boxes = np.zeros([self.num_heads, 4], np.float32)
            for i, class_ids in enumerate(self.heads):
                valid_class_ids = np.intersect1d(valid_box_inds, class_ids)
                if len(valid_class_ids) == 0:
                    continue
                y1 = np.min(mask_boxes[valid_class_ids, 0], axis=0)
                x1 = np.min(mask_boxes[valid_class_ids, 1], axis=0)
                y2 = np.max(mask_boxes[valid_class_ids, 2], axis=0)
                x2 = np.max(mask_boxes[valid_class_ids, 3], axis=0)
                boxes[i] = np.array([y1, x1, y2, x2])
            return boxes, True
        else:
            return np.zeros([self.num_heads, 4], np.float32), False

    def load_molded_mask_boxes(self, image_id, pre):
        boxes, exists = self.load_mask_boxes(image_id, pre)
        if exists:
            return (boxes / self.im_size).astype(np.float32), True
        else:
            return boxes, False

    def load_molded_head_boxes(self, image_id, pre):
        boxes, exists = self.load_head_boxes(image_id, pre)
        if exists:
            return (boxes / self.im_size).astype(np.float32), True
        else:
            return boxes, False

    def load_molded_pad_head_boxes(self, image_id, pre):
        mboxes, exists = self.load_molded_head_boxes(image_id, pre)
        p = self.options['head_box_padding_ratio']
        head_box_padding_ratios = np.array([-p, -p, +p, +p], np.float32)
        # y1 x1 y2 x2
        if exists:
            return mboxes + head_box_padding_ratios, True
        else:
            return mboxes, False

    def load_lbf_landmark68_boxes(self, image_id, pre):
        pts, exists = self.load_lbf_landmark68_pts(image_id, pre)
        if exists:
            boxes = utils.extract_landmark68_boxes(
                pts,
                self.head_label_names, None)
            return boxes, True
        else:
            return np.zeros([self.num_heads, 4], np.float32), False

    def load_molded_lbf_landmark68_boxes(self, image_id, pre):
        boxes, exists = self.load_lbf_landmark68_boxes(image_id, pre)
        if exists:
            return (boxes / self.im_size).astype(np.float32), True
        else:
            return boxes, False

    def load_original_image_shape(self, image_id, pre):
        return None, False

    def load_original_image(self, image_id, pre):
        return None, False

    def load_original_masks(self, image_id, pre):
        return None, False

    def load_align_matrix(self, image_id, pre):
        return np.zeros([3, 3], np.float32), False


def _flip_landmark68_pts(pts, im_size):
    # print('flipped!')
    # flip landmarks pts
    pts = np.array(pts)
    pts[:, 0] = im_size - 1 - pts[:, 0]
    # flip indices
    new_pts = np.array(pts)
    # face
    new_pts[0:8:1, :] = pts[16:8:-1, :]
    new_pts[16:8:-1, :] = pts[0:8:1, :]
    # brows
    new_pts[17:22:1, :] = pts[26:21:-1, :]
    new_pts[26:21:-1, :] = pts[17:22:1, :]
    # eyes
    new_pts[[36, 37, 38, 39, 40, 41], :] = pts[
        [45, 44, 43, 42, 47, 46], :]
    new_pts[[45, 44, 43, 42, 47, 46], :] = pts[
        [36, 37, 38, 39, 40, 41], :]
    # nose
    new_pts[[31, 32], :] = pts[[35, 34], :]
    new_pts[[35, 34], :] = pts[[31, 32], :]
    # mouth
    new_pts[48:55:1, :] = pts[54:47:-1, :]
    new_pts[60:65:1, :] = pts[64:59:-1, :]
    new_pts[65:68:1, :] = pts[67:64:-1, :]
    new_pts[59:54:-1, :] = pts[55:60:1, :]
    return new_pts


class PureImagesDataset(Dataset):
    def __init__(self, options, images):
        super(PureImagesDataset, self).__init__(options)
        self.images = images
        if self.images.dtype == np.float32:
            self.images = (images * 255).astype(np.uint8)
        for i in range(images.shape[0]):
            self.add_image(
                'Face', image_id=i)
        self.make_image_ids()

    def prepare_loading(self, image_id):
        return None

    def load_image(self, image_id, pre=None):
        im = self.images[image_id]
        assert im.shape[0] == im.shape[1] == self.im_size
        assert im.shape[2] == 3
        return im, True


class PureImageFolderDataset(Dataset):
    '''
        root/
            Aligned512/
            Original/
    '''

    def __init__(self, options, folder):
        super(PureImageFolderDataset, self).__init__(options)

        aligned_root = os.path.join(folder, 'Aligned%d' % self.im_size)
        original_root = os.path.join(folder, 'Original')
        names = os.listdir(aligned_root)
        print(f'there are {len(names)} images in {aligned_root}')
        for i, name in enumerate(names):
            self.add_image(
                'Face', image_id=i,
                path=os.path.join(aligned_root, name),
                original_path=os.path.join(original_root, name),
                align_mat_path=os.path.join(aligned_root, name[:-3] + 'mat'))
        self.make_image_ids()

    def prepare_loading(self, image_id):
        return None

    def load_image(self, image_id, pre=None):
        path = self.image_info[image_id]['path']
        im = io.imread(path)
        assert im.shape[0] == im.shape[1] == self.im_size
        assert im.shape[2] == 3
        return im, True

    def load_original_image(self, image_id, pre=None):
        path = self.image_info[image_id]['original_path']
        im = io.imread(path)
        return im, True

    def load_align_matrix(self, image_id, pre=None):
        assert pre is None
        assert self.im_size == 448
        info = self.image_info[image_id]
        align_matrix_path = info['align_mat_path']
        T = loadmat(align_matrix_path)['T']
        assert T.shape[0] == T.shape[1] == 3
        return T, True


class PublicLandmark68Dataset(Dataset):
    def __init__(self,
                 options,
                 from_to_ratio=None,
                 mode='train',
                 aug_flip=True,
                 aug_transform=True,
                 aug_gamma=True):
        super(PublicLandmark68Dataset, self).__init__(options)
        assert self.im_size == 512

        self.head_label_names = []
        for class_ids in self.options['heads']:
            names_this_head = []
            for class_id in class_ids:
                names_this_head += self.options['class_names'][class_id]
            self.head_label_names.append(names_this_head)

        assert len(self.head_label_names) == len(self.options['heads'])

        self.aug_flip = aug_flip
        self.aug_transform = aug_transform
        self.aug_gamma = aug_gamma

        data_root = paths.public_landmark68_root_path()
        if mode == 'train':
            list_path = os.path.join(data_root, '300w_train_list.txt')
            sub_data_root = 'LandmarkTrainData'
        elif mode == 'test':
            list_path = os.path.join(data_root, '300w_test_list.txt')
            sub_data_root = 'LandmarkTestData'

        with open(list_path) as list_file:
            lines = [line.strip() for line in list_file]
            if from_to_ratio is not None:
                fr = int(from_to_ratio[0] * len(lines))
                to = int(from_to_ratio[1] * len(lines))
                lines = lines[fr:to]
            img_paths = [
                os.path.join(data_root,
                             sub_data_root,
                             line.strip()).replace('\\', '/')
                for line in lines
            ]
            # print(img_paths[0:10])
            img_paths = [p for p in img_paths if os.path.exists(p)]
            print('final img path num: %d' % len(img_paths))

        for i, img_path in enumerate(img_paths):
            self.add_image(
                "Face",
                image_id=i,
                path=img_path,
                landmark68_path=img_path[:-3] + 'txt'
            )
        self.make_image_ids()

    def prepare_loading(self, image_id):
        if self.aug_flip:
            fliped = random.uniform(0, 1) < 0.5
        else:
            fliped = False

        if self.aug_transform:
            im_size = self.im_size
            trans_scale = transform.SimilarityTransform(
                scale=random.normalvariate(1.0, 0.1))
            trans_rot = transform.SimilarityTransform(
                rotation=random.normalvariate(0, math.pi / 8))
            center = np.array([im_size / 2, im_size / 2])
            trans_shift = transform.SimilarityTransform(translation=-center)
            trans_shift_inv = transform.SimilarityTransform(
                translation=center + np.random.normal(0, 5, (2)))
            T = trans_scale + (trans_shift + (trans_rot + trans_shift_inv))
        else:
            T = None

        if self.aug_gamma:
            gamma = math.exp(max(-1.6, min(1.6, random.normalvariate(0, 0.8))))
        else:
            gamma = None

        return fliped, T, gamma

    def load_image(self, image_id, pre=None):
        im_path = self.image_info[image_id]['path']
        im = io.imread(im_path)
        im_size = self.im_size
        assert im.shape[:2] == (im_size, im_size)
        if len(im.shape) == 2:  # grey image
            im = np.tile(np.expand_dims(im, axis=-1), [1, 1, 3])
        assert im.dtype == np.uint8
        assert im.shape[2] == 3
        if pre is not None:
            flipped, T, gamma = pre
            im = im / 255.0
            if flipped:
                im = im[:, ::-1, :]
            if T is not None:
                im = transform.warp(im, T.inverse)
            if gamma is not None:
                im = exposure.adjust_gamma(im, gamma)
            im = (im * 255).astype(np.uint8)

        return im, True

    def load_landmark68_pts(self, image_id, pre=None):
        landmark68_path = self.image_info[image_id]['landmark68_path']
        im_size = self.im_size
        assert im_size == 512
        with open(landmark68_path, 'r') as f:
            pts = [float(v) for v in f.readline().strip().split()]
            pts = np.reshape(np.array(pts), [68, 2])
            if pre is not None:
                flipped, T, gamma = pre
                if flipped:
                    pts = _flip_landmark68_pts(pts, im_size)
                if T is not None:
                    pts = T(pts)

        return pts, True


DATASET_HANDLERS['300w_train'] = lambda options: PublicLandmark68Dataset(
    options, from_to_ratio=[0, 1], mode='train')
DATASET_HANDLERS['300w_test'] = lambda options: PublicLandmark68Dataset(
    options, from_to_ratio=[0, 1], mode='test',
    aug_flip=False, aug_transform=False, aug_gamma=False)


class MSRALandmark68Dataset(Dataset):
    ''' paths.face_landmark68_root_path()
       - /badAlign
       - /wellAlign
    '''

    def __init__(self,
                 options,
                 from_to_ratio=None,
                 aug_flip=True,
                 aug_transform=True,
                 aug_gamma=True):
        super(MSRALandmark68Dataset, self).__init__(options)
        assert self.im_size == 448

        self.head_label_names = []
        for class_ids in self.options['heads']:
            names_this_head = []
            for class_id in class_ids:
                names_this_head += self.options['class_names'][class_id]
            self.head_label_names.append(names_this_head)
        assert len(self.head_label_names) == len(self.options['heads'])

        self.aug_flip = aug_flip
        self.aug_transform = aug_transform
        self.aug_gamma = aug_gamma

        data_root = paths.face_landmark68_root_path()
        list_path = os.path.join(data_root, 'imagelist_all.txt')
        with open(list_path) as list_file:
            lines = [line.strip() for line in list_file]
            if from_to_ratio is not None:
                fr = int(from_to_ratio[0] * len(lines))
                to = int(from_to_ratio[1] * len(lines))
                lines = lines[fr:to]
            img_paths = [
                os.path.join(data_root,
                             line.split()[1][12:]).replace('\\', '/')
                for line in lines
            ]
            img_paths = [p for p in img_paths if os.path.exists(p)]
            print('final img path num: %d' % len(img_paths))

        for i, img_path in enumerate(img_paths):
            self.add_image(
                "Face",
                image_id=i,
                path=img_path,
                image224_path=img_path.replace('Align', 'AlignImage224'),
                landmark68_path=img_path[:-3] + 'txt')
        self.make_image_ids()

    def prepare_loading(self, image_id):
        if self.aug_flip:
            flipped = random.uniform(0, 1) < 0.5
        else:
            flipped = False

        if self.aug_transform:
            im_size = self.im_size
            trans_scale = transform.SimilarityTransform(
                scale=random.normalvariate(1.0, 0.1))
            trans_rot = transform.SimilarityTransform(
                rotation=random.normalvariate(0, math.pi / 8))
            center = np.array([im_size / 2, im_size / 2])
            trans_shift = transform.SimilarityTransform(translation=-center)
            trans_shift_inv = transform.SimilarityTransform(
                translation=center + np.random.normal(0, 5, (2)))
            T = trans_scale + (trans_shift + (trans_rot + trans_shift_inv))
        else:
            T = None

        if self.aug_gamma:
            gamma = math.exp(max(-1.6, min(1.6, random.normalvariate(0, 0.8))))
        else:
            gamma = None
        return flipped, T, gamma

    def load_image(self, image_id, pre=None):
        im_path = self.image_info[image_id]['path']
        im = io.imread(im_path)
        im_size = self.im_size
        assert im.shape[:2] == (im_size, im_size)
        if pre is not None:
            flipped, T, gamma = pre
            im = im / 255.0
            if flipped:
                im = im[:, ::-1, :]
            if T is not None:
                im = transform.warp(im, T.inverse)
            if gamma is not None:
                im = exposure.adjust_gamma(im, gamma)
            im = (im * 255).astype(np.uint8)
        return im, True

    def load_landmark68_pts(self, image_id, pre=None):
        landmark68_path = self.image_info[image_id]['landmark68_path']
        im_size = self.im_size
        assert im_size == 448
        with open(landmark68_path, 'r') as f:
            pts = [float(v) for v in f.readline().strip().split()]
            pts = np.reshape(np.array(pts), [68, 2])
            if pre is not None:
                flipped, T, gamma = pre
                if flipped:
                    pts = _flip_landmark68_pts(pts, im_size)
                    # transform landmarks
                if T is not None:
                    pts = T(pts)
        return pts, True


DATASET_HANDLERS['fa68_train'] = lambda options: MSRALandmark68Dataset(
    options, from_to_ratio=[0, 0.9])
DATASET_HANDLERS['fa68_train_toy'] = lambda options: MSRALandmark68Dataset(
    options, from_to_ratio=[0, 0.1])
DATASET_HANDLERS['fa68_val'] = lambda options: MSRALandmark68Dataset(
    options, from_to_ratio=[0.9, 1],
    aug_flip=False, aug_transform=False, aug_gamma=False)
DATASET_HANDLERS['fa68_all'] = lambda options: MSRALandmark68Dataset(
    options, from_to_ratio=[0, 1],
    aug_flip=False, aug_transform=False, aug_gamma=False)


USE_FIRST_LANDMARK = True


def _query_masks(label_img, provided_label_names, query_label_names):
    '''
    provided_label_names:
        provided_label_names[label_in_label_img] returns the name like
        ['bg', 'body', 'face', 'hair']
    query_label_names:
        an array of array like
        [['eyes', 'le', 're'], ['face'], ['brows', 'lb', 'rb']]

    returns:
    masks: [#query, IMAGE_SIZE, IMAGE_SIZE]
    '''
    masks = []
    for label_names in query_label_names:
        mask = np.zeros(
            [label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        for label_name in label_names:
            if label_name not in provided_label_names:
                continue
            index = provided_label_names.index(label_name)
            single_mask = (label_img == index)
            mask = np.logical_or(mask, single_mask)
        masks.append(mask)
    return np.stack(masks, 0).astype(np.uint8)


def _query_landmark68_masks(pts, query_label_names, mask_size):
    masks = []
    for label_names in query_label_names:
        mask = np.zeros([mask_size, mask_size], dtype=np.uint8)

        if set(label_names).issuperset(
                set(['face', 'lb', 'rb', 'le', 're', 'nose', 'mouth'])):
            inds = list(range(0, 17)) + list(range(26, 21, -1)) + \
                list(range(21, 16, -1))
            mask[draw.polygon(pts[inds, 1], pts[inds, 0])] = 1
        elif label_names == ['nose']:
            inds = [27] + list(range(31, 36)) + [27]
            mask[draw.polygon(pts[inds, 1], pts[inds, 0])] = 1
        elif label_names == ['mouth']:
            inds = list(range(48, 60))
            mask[draw.polygon(pts[inds, 1], pts[inds, 0])] = 1
        else:
            for label_name in label_names:
                inds = utils.get_single_landmark68_indices(label_name)
                if not inds:
                    continue
                inds = inds + [inds[0]]
                mask[draw.polygon(pts[inds, 1], pts[inds, 0])] = 1

        masks.append(mask)
    return np.stack(masks, axis=0)


class SegmentationDatasetBase(Dataset):
    def __init__(self, provided_label_names, options):
        super(SegmentationDatasetBase, self).__init__(options)

        self.provided_label_names = provided_label_names

        self.head_label_names = []
        for class_ids in self.options['heads']:
            names_this_head = []
            for class_id in class_ids:
                names_this_head += self.options['class_names'][class_id]
            self.head_label_names.append(names_this_head)
        assert len(self.head_label_names) == len(self.options['heads'])

    def load_label_image(self, image_id, pre=None):
        raise NotImplementedError()

    def load_masks(self, image_id, pre=None):
        ''' returns [query_label_num, IMAGE_SIZE, IMAGE_SIZE]
        '''
        result = _query_masks(
            self.load_label_image(
                image_id, pre), self.provided_label_names,
            self.options['class_names'])
        assert result.shape[0] == len(self.options['class_names'])
        # if np.max(result) == 0:
        #     raise RuntimeError(
        #         'no mask loaded for %s' % self.image_info[image_id]['path'])
        return result, True

    def load_original_label_image(self, image_id, pre=None):
        raise NotImplementedError()

    def load_original_masks(self, image_id, pre=None):
        ''' returns [query_label_num, IMAGE_SIZE, IMAGE_SIZE]
        '''
        result = _query_masks(
            self.load_original_label_image(image_id, pre),
            self.provided_label_names, self.options['class_names'])
        assert result.shape[0] == len(self.options['class_names'])
        # if np.max(result) == 0:
        #     raise RuntimeError(
        #         'no mask loaded for %s' % self.image_info[image_id]['path'])
        return result, True


def _get_face_parsing_data_label_path(img_path):
    return img_path[:-7] + 'label.png'


def _get_face_parsing_data_landmark5_path(img_path):
    return img_path[:-7] + 'pts5.txt'


def _get_face_parsing_data_landmark68_path(img_path):
    path_parts = img_path.replace('\\', '/').split('/')
    assert path_parts[-1].endswith('png')
    path_parts[-1] = path_parts[-1][:-3] + 'txt'
    path_parts[-2] = 'landmark' + path_parts[-2][-1]
    landmark_path = '/'.join(path_parts)
    # print(landmark_path)
    return landmark_path


def _get_face_parsing_data_landmark68_label_path(img_path):
    path_parts = img_path.replace('\\', '/').split('/')
    assert path_parts[-1].endswith('png')
    path_parts[-2] = 'preparsed' + path_parts[-2][-1]
    landmark_label_path = '/'.join(path_parts)
    return landmark_label_path


class MultiPieDataset(SegmentationDatasetBase):
    def __init__(self, options, list_path=None, from_to_ratio=None,
                 grey_ratio=0.1, adjust_gamma=True,
                 no_augmented_samples=False,
                 aug_flip=False,
                 aug_transform=False):
        super(MultiPieDataset, self).__init__([
            'background', 'body', 'face', 'hair', 'lb', 'rb', 'le', 're',
            'nose', 'mouth', 'lr', 'rr', 'mouth'
        ], options)

        assert self.im_size == 512

        self.grey_ratio = grey_ratio
        self.adjust_gamma = adjust_gamma
        self.aug_flip = aug_flip
        self.aug_transform = aug_transform
        self.aug_larger = True

        # data_root = '\\\\MSRA-FACEDNN11\\haya\\FaceData\\Parsing\\MULTIPIE'
        data_root = os.path.join(paths.dataset_root(), 'Parsing', 'MULTIPIE')

        augmented_root = os.path.join(data_root, 'Augmented')
        # aligned_root = os.path.join(data_root, 'Aligned')
        # imagelabel_root = os.path.join(aligned_root, 'ImageLabel512')
        # landmark_root = os.path.join(aligned_root, 'Landmark512')
        aligned_root = '//5FTGDB2/Final_Data2'
        imagelabel_root = os.path.join(aligned_root, 'dataset1_512_loose')
        landmark_root = os.path.join(aligned_root, 'landmark1_512_loose')

        if list_path is not None:
            with open(list_path, 'r') as f:
                names = [line.strip() for line in f]
        else:
            names = [name[:-4]
                     for name in os.listdir(landmark_root)]

        self.no_augmented_samples = no_augmented_samples
        if no_augmented_samples:
            names = [name for name in names if name.endswith('00_img')]

        if from_to_ratio is not None:
            fr = int(from_to_ratio[0] * len(names))
            to = int(from_to_ratio[1] * len(names))
            names = names[fr:to]

        print('final img path num: %d' % len(names))

        for name in names:
            assert name.endswith('_img')
            assert os.path.exists(os.path.join(
                imagelabel_root, name + '.png'))
            assert os.path.exists(os.path.join(
                imagelabel_root, name[:-3] + 'label.png'))
            assert os.path.exists(os.path.join(imagelabel_root, name + '.mat'))

        # load jinpeng's data list
        with open(os.path.join(data_root, 'Source', 'dataset1.txt')) as f:
            source_paths = [line.strip() for line in f]

        for i, name in enumerate(names):
            ind = int(name[:5])

            # C:\Users\v-jinpli\Desktop\Face1008\Face1008\root\1-50\001\1\001_02_02_041_00.jpg
            # root\1-50\001\1\001_02_02_041_00.jpg
            source_path = source_paths[ind].split(',')[0][44:]
            source_path = os.path.join(data_root, 'Source', source_path)
            original_face_alpha_path = source_path[:-7] + '_Face.jpg'
            original_fg_alpha_path = source_path[:-7] + '_FG.jpg'
            original_hair_alpha_path = source_path[:-7] + '_Hair.jpg'
            # source_folder = os.path.dirname(source_path)

            self.add_image(
                "Face",
                image_id=i,
                name=name,
                path=os.path.join(imagelabel_root, name + '.png'),
                label_path=os.path.join(
                    imagelabel_root, name[:-3] + 'label.png'),
                bfh_acc_alphas_path=os.path.join(
                    aligned_root, 'BodyFaceHairAlphas', name[:-3] + 'bfh_alphas.mat'),
                align_mat_path=os.path.join(imagelabel_root, name + '.mat'),
                lbf_landmark68_path=os.path.join(landmark_root, name + '.txt'),
                original_img_path=os.path.join(augmented_root, name + '.png'),
                original_label_path=os.path.join(
                    augmented_root, name[:-3] + 'label.png'),
                original_face_alpha_path=original_face_alpha_path,
                original_fg_alpha_path=original_fg_alpha_path,
                original_hair_alpha_path=original_hair_alpha_path
            )
        self.make_image_ids()

    def prepare_loading(self, image_id):
        if self.adjust_gamma:
            gamma = math.exp(max(-1.6, min(1.6, random.normalvariate(0, 0.8))))
        else:
            gamma = None
        if random.uniform(0, 1) < self.grey_ratio:
            greyfy = True
        else:
            greyfy = False
        if self.aug_flip and random.uniform(0, 1) < 0.5:
            flipped = True
        else:
            flipped = False
        if self.aug_transform:
            if self.aug_larger:
                scale_v = 0.1
                rotation_v = math.pi / 10
                shift_v = 0.1
            else:
                scale_v = 0.1
                rotation_v = math.pi / 16
                shift_v = 0.1
            scale = random.normalvariate(1.0, scale_v)
            rotation = random.normalvariate(0, rotation_v)
            shift = np.random.normal(0, shift_v, (2))
            Tparams = (scale, rotation, shift)
        else:
            Tparams = None
        return gamma, greyfy, flipped, Tparams

    def _compose_T(self, im_size, Tparams):
        scale, rotation, shift = Tparams
        trans_scale = transform.SimilarityTransform(scale=scale)
        trans_rot = transform.SimilarityTransform(rotation=rotation)
        center = np.array([im_size / 2, im_size / 2])
        trans_shift = transform.SimilarityTransform(translation=-center)
        trans_shift_inv = transform.SimilarityTransform(
            translation=center + im_size * shift)
        T = trans_scale + (trans_shift + (trans_rot + trans_shift_inv))
        return T

    def _apply_Tparams(self, image, Tparams, order=1, preserve_range=False):
        T = self._compose_T(image.shape[0], Tparams)
        return transform.warp(image, T.inverse, order=order, preserve_range=preserve_range)

    def load_image(self, image_id, pre=None):
        im = io.imread(self.image_info[image_id]['path']).astype(np.uint8)
        assert im.shape[0] == self.im_size
        if pre is not None:
            gamma, greyfy, flipped, Tparams = pre
            im = im / 255.0
            if gamma is not None:
                im = exposure.adjust_gamma(im, gamma)
            if greyfy:
                im = color.rgb2gray(im)
                im = np.tile(np.expand_dims(im, -1), [1, 1, 3])
            if flipped:
                im = np.flip(im, axis=1)
            if Tparams is not None:
                im = self._apply_Tparams(im, Tparams)
            im = (im * 255).astype(np.uint8)
        assert len(im.shape) == 3 and im.shape[-1] == 3
        return im, True

    def _flip_label_image(self, label_img):
        label_img = np.flip(label_img, axis=1)
        # flip the label ids
        lb_id = self.provided_label_names.index('lb')
        rb_id = self.provided_label_names.index('rb')
        le_id = self.provided_label_names.index('le')
        re_id = self.provided_label_names.index('re')
        label_img_flipped = np.copy(label_img)
        label_img_flipped[label_img == lb_id] = rb_id
        label_img_flipped[label_img == rb_id] = lb_id
        label_img_flipped[label_img == le_id] = re_id
        label_img_flipped[label_img == re_id] = le_id
        label_img = label_img_flipped
        return label_img

    def load_label_image(self, image_id, pre=None):
        label_img = io.imread(
            self.image_info[image_id]['label_path'], as_grey=True).astype(np.int32)
        if pre is not None:
            _, _, flipped, Tparams = pre
            if flipped:
                label_img = self._flip_label_image(label_img)
            if Tparams is not None:
                prev_t = label_img.dtype
                label_img = self._apply_Tparams(
                    label_img, Tparams, 0, True).astype(prev_t)
        assert label_img.shape[0] == self.im_size
        return label_img

    def load_lbf_landmark68_pts(self, image_id, pre=None):
        landmark_path = self.image_info[image_id]['lbf_landmark68_path']
        assert self.im_size == 448
        with open(landmark_path, 'r') as f:
            f.readline()
            if not USE_FIRST_LANDMARK:
                f.readline()
            pts = [float(v) for v in f.readline().strip().split()]
            pts = np.reshape(np.array(pts), [68, 2])
        if pre is not None:
            _, _, flipped, Tparams = pre
            if flipped:
                pts = _flip_landmark68_pts(pts, self.im_size)
            if Tparams is not None:
                T = self._compose_T(self.im_size, Tparams)
                pts = T(pts)
        return pts, True

    def load_original_image(self, image_id, pre=None):
        im = io.imread(self.image_info[image_id]
                       ['original_img_path']).astype(np.uint8)
        assert pre is None
        assert len(im.shape) == 3 and im.shape[-1] == 3
        return im, True

    def load_original_image_shape(self, image_id, pre=None):
        assert pre is None
        return [480, 640], True

    def load_align_matrix(self, image_id, pre=None):
        assert pre is None
        assert self.im_size == 448
        info = self.image_info[image_id]
        align_matrix_path = info['align_mat_path']
        T = loadmat(align_matrix_path)['T']
        assert T.shape[0] == T.shape[1] == 3
        return T, True


def _filter_out_masks(dataset):
    dataset.pretend_not_exist_data_names.add('masks')
    dataset.pretend_not_exist_data_names.add('alphas')
    return dataset


def _filter_out_head_boxes(dataset):
    for name in {
        'mask_boxes',
        'molded_mask_boxes',
        'head_boxes',
        'molded_head_boxes',
            'molded_pad_head_boxes'}:
        dataset.pretend_not_exist_data_names.add(name)
    return dataset


DATASET_HANDLERS['multipie_train'] = lambda options: MultiPieDataset(
    options, from_to_ratio=[0, 0.9], no_augmented_samples=False,
    grey_ratio=0.1, adjust_gamma=True,
    aug_flip=False, aug_transform=True)
DATASET_HANDLERS['multipie_train_toy'] = lambda options: MultiPieDataset(
    options, from_to_ratio=[0, 0.1], no_augmented_samples=False,
    grey_ratio=0.1, adjust_gamma=True,
    aug_flip=False, aug_transform=True)
DATASET_HANDLERS['multipie_test'] = lambda options: MultiPieDataset(
    options, from_to_ratio=[0.9, 1],
    grey_ratio=0, adjust_gamma=False, no_augmented_samples=True,
    aug_flip=False, aug_transform=False)
DATASET_HANDLERS['multipie_test_toy'] = lambda options: MultiPieDataset(
    options, from_to_ratio=[0.9, 0.91],
    grey_ratio=0, adjust_gamma=False, no_augmented_samples=True,
    aug_flip=False, aug_transform=False)

DATASET_HANDLERS['multipie_train_no_mask'] = \
    lambda options: _filter_out_masks(MultiPieDataset(
        options, from_to_ratio=[0, 0.9], no_augmented_samples=False,
        grey_ratio=0.1, adjust_gamma=True,
        aug_flip=False, aug_transform=True))
DATASET_HANDLERS['multipie_train_no_headbox'] = \
    lambda options: _filter_out_head_boxes(MultiPieDataset(
        options, from_to_ratio=[0, 0.9], no_augmented_samples=False,
        grey_ratio=0.1, adjust_gamma=True,
        aug_flip=False, aug_transform=True))
DATASET_HANDLERS['multipie_train_toy_no_headbox'] = \
    lambda options: _filter_out_head_boxes(MultiPieDataset(
        options, from_to_ratio=[0, 0.1], no_augmented_samples=False,
        grey_ratio=0.1, adjust_gamma=True,
        aug_flip=False, aug_transform=True))


class HelenSmithDataset(SegmentationDatasetBase):
    def __init__(self, options, list_path=None,
                 from_to_ratio=None, grey_ratio=0.1, adjust_gamma=True,
                 no_augmented_samples=False,
                 aug_flip=False,
                 aug_transform=False,
                 no_occ=False,
                 aug_larger=False):
        super(HelenSmithDataset, self).__init__([
            'background', 'body', 'face', 'hair', 'lb', 'rb', 'le', 're',
            'nose', 'mouth', 'lr', 'rr', 'ulip', 'imouth', 'llip'
        ], options)

        assert self.im_size in [128, 256, 448, 512]

        self.grey_ratio = grey_ratio
        self.adjust_gamma = adjust_gamma
        self.aug_flip = aug_flip
        self.aug_transform = aug_transform
        self.aug_larger = aug_larger

        # data_root = '\\\\MSRA-FACEDNN11\\haya\\FaceData\\Parsing\\Helen'
        data_root = os.path.join(paths.dataset_root(), 'Parsing', 'Helen')

        if no_occ:
            aligned_root = os.path.join(data_root, 'AlignedNoOcc')
            augmented_root = os.path.join(data_root, 'AugmentedNoOcc')
        else:
            aligned_root = os.path.join(data_root, 'Aligned')
            augmented_root = os.path.join(data_root, 'Augmented')

        imagelabel_root = os.path.join(aligned_root, 'ImageLabel')
        imagelabel128_root = os.path.join(aligned_root, 'ImageLabel128')
        imagelabel224_root = os.path.join(aligned_root, 'ImageLabel224')
        imagelabel256_root = os.path.join(aligned_root, 'ImageLabel256')
        imagelabel512_root = os.path.join(aligned_root, 'ImageLabel512')

        landmark_root = os.path.join(aligned_root, 'Landmark')
        landmark128_root = os.path.join(aligned_root, 'Landmark128')
        landmark224_root = os.path.join(aligned_root, 'Landmark224')
        landmark256_root = os.path.join(aligned_root, 'Landmark256')
        landmark512_root = os.path.join(aligned_root, 'Landmark512')

        if self.im_size == 128:
            imagelabel_root = imagelabel128_root
            landmark_root = landmark128_root
        elif self.im_size == 224:
            imagelabel_root = imagelabel224_root
            landmark_root = landmark224_root
        elif self.im_size == 256:
            imagelabel_root = imagelabel256_root
            landmark_root = landmark256_root
        elif self.im_size == 512:
            imagelabel_root = imagelabel512_root
            landmark_root = landmark512_root
        else:
            assert self.im_size == 448

        if list_path is not None:
            with open(list_path, 'r') as f:
                names = [line.strip() for line in f]
        else:
            names = [name[:-4]
                     for name in os.listdir(
                landmark_root)]
        assert len(names) > 0

        if no_augmented_samples:
            names = [name for name in names if name.endswith('00_img')]
        assert len(names) > 0

        names = [name for name in names if os.path.exists(os.path.join(
            landmark_root, name + '.txt'))]
        assert len(names) > 0

        for name in names:
            assert name.endswith('_img')
            assert os.path.exists(os.path.join(
                imagelabel_root, name + '.png').replace('\\', '/'))
            assert os.path.exists(os.path.join(
                imagelabel_root, name[:-3] + 'label.png').replace('\\', '/'))
            assert os.path.exists(os.path.join(imagelabel_root, name + '.mat'))

        if from_to_ratio is not None:
            fr = int(from_to_ratio[0] * len(names))
            to = int(from_to_ratio[1] * len(names))
            names = names[fr:to]

        print('final img path num: %d' % len(names))

        for i, name in enumerate(names):
            self.add_image(
                "Face",
                image_id=i,
                name=name,
                path=os.path.join(imagelabel_root, name + '.png'),
                label_path=os.path.join(
                    imagelabel_root, name[:-3] + 'label.png'),
                align_mat_path=os.path.join(imagelabel_root, name + '.mat'),
                lbf_landmark68_path=os.path.join(landmark_root, name + '.txt'),
                original_img_path=os.path.join(augmented_root, name + '.png'),
                original_label_path=os.path.join(
                    augmented_root, name[:-3] + 'label.png'))
        self.make_image_ids()

    def prepare_loading(self, image_id):
        if self.adjust_gamma:
            gamma = math.exp(max(-1.6, min(1.6, random.normalvariate(0, 0.8))))
        else:
            gamma = None
        if random.uniform(0, 1) < self.grey_ratio:
            greyfy = True
        else:
            greyfy = False
        if self.aug_flip and random.uniform(0, 1) < 0.5:
            flipped = True
        else:
            flipped = False
        if self.aug_transform:
            if self.aug_larger:
                scale_v = 0.1
                rotation_v = math.pi / 10
                shift_v = 0.1
            else:
                scale_v = 0.1
                rotation_v = math.pi / 16
                shift_v = 0.1
            scale = random.normalvariate(1.0, scale_v)
            rotation = random.normalvariate(0, rotation_v)
            shift = np.random.normal(0, shift_v, (2))
            Tparams = (scale, rotation, shift)
        else:
            Tparams = None
        return gamma, greyfy, flipped, Tparams

    def _compose_T(self, im_size, Tparams):
        scale, rotation, shift = Tparams
        trans_scale = transform.SimilarityTransform(scale=scale)
        trans_rot = transform.SimilarityTransform(rotation=rotation)
        center = np.array([im_size / 2, im_size / 2])
        trans_shift = transform.SimilarityTransform(translation=-center)
        trans_shift_inv = transform.SimilarityTransform(
            translation=center + im_size * shift)
        T = trans_scale + (trans_shift + (trans_rot + trans_shift_inv))
        return T

    def _apply_Tparams(self, image, Tparams, order=1, preserve_range=False):
        T = self._compose_T(image.shape[0], Tparams)
        return transform.warp(image, T.inverse, order=order, preserve_range=preserve_range)

    def load_image(self, image_id, pre=None):
        im = io.imread(self.image_info[image_id]['path']).astype(np.uint8)
        assert im.shape[0] == self.im_size
        if pre is not None:
            gamma, greyfy, flipped, Tparams = pre
            im = im / 255.0
            if gamma is not None:
                im = exposure.adjust_gamma(im, gamma)
            if greyfy:
                im = color.rgb2gray(im)
                im = np.tile(np.expand_dims(im, -1), [1, 1, 3])
            if flipped:
                im = np.flip(im, axis=1)
            if Tparams is not None:
                im = self._apply_Tparams(im, Tparams)
            im = (im * 255).astype(np.uint8)
        assert len(im.shape) == 3 and im.shape[-1] == 3
        return im, True

    def _flip_label_image(self, label_img):
        label_img = np.flip(label_img, axis=1)
        # flip the label ids
        lb_id = self.provided_label_names.index('lb')
        rb_id = self.provided_label_names.index('rb')
        le_id = self.provided_label_names.index('le')
        re_id = self.provided_label_names.index('re')
        label_img_flipped = np.copy(label_img)
        label_img_flipped[label_img == lb_id] = rb_id
        label_img_flipped[label_img == rb_id] = lb_id
        label_img_flipped[label_img == le_id] = re_id
        label_img_flipped[label_img == re_id] = le_id
        label_img = label_img_flipped
        return label_img

    def load_label_image(self, image_id, pre=None):
        label_img = io.imread(
            self.image_info[image_id]['label_path'], as_grey=True).astype(np.int32)
        if pre is not None:
            _, _, flipped, Tparams = pre
            if flipped:
                label_img = self._flip_label_image(label_img)
            if Tparams is not None:
                prev_t = label_img.dtype
                label_img = self._apply_Tparams(
                    label_img, Tparams, 0, True).astype(prev_t)
        assert label_img.shape[0] == self.im_size
        return label_img

    def load_lbf_landmark68_pts(self, image_id, pre=None):
        landmark_path = self.image_info[image_id]['lbf_landmark68_path']
        with open(landmark_path, 'r') as f:
            f.readline()
            if not USE_FIRST_LANDMARK:
                f.readline()
            pts = [float(v) for v in f.readline().strip().split()]
            pts = np.reshape(np.array(pts), [68, 2])
        if pre is not None:
            _, _, flipped, Tparams = pre
            if flipped:
                pts = _flip_landmark68_pts(pts, self.im_size)
            if Tparams is not None:
                T = self._compose_T(self.im_size, Tparams)
                pts = T(pts)
        return pts, True

    def load_original_image(self, image_id, pre=None):
        im = io.imread(self.image_info[image_id]
                       ['original_img_path']).astype(np.uint8)
        assert pre is None
        assert len(im.shape) == 3 and im.shape[-1] == 3
        return im, True

    def load_original_image_shape(self, image_id, pre=None):
        assert pre is None
        return io.imread(self.image_info[image_id]
                         ['original_img_path']).shape, True

    def load_original_label_image(self, image_id, pre=None):
        assert pre is None
        info = self.image_info[image_id]
        labels = io.imread(info['original_label_path'],
                           as_grey=True).astype(np.int32)
        return labels

    def load_align_matrix(self, image_id, pre=None):
        assert pre is None
        info = self.image_info[image_id]
        align_matrix_path = info['align_mat_path']
        T = loadmat(align_matrix_path)['T']
        assert T.shape[0] == T.shape[1] == 3
        return T, True


DATASET_HANDLERS['helen_smith_test'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=True,
    adjust_gamma=False, grey_ratio=0.0,
    list_path=os.path.join(
        paths.dataset_root(), 'Parsing', 'Helen', 'test_names.txt'), no_occ=True)
DATASET_HANDLERS['helen_smith_tune'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=True,
    adjust_gamma=False, grey_ratio=0.0,
    list_path=os.path.join(
        paths.dataset_root(), 'Parsing', 'Helen', 'tune_names.txt'), no_occ=True)
DATASET_HANDLERS['helen_smith_train'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=False,
    adjust_gamma=True, grey_ratio=0.1,
    list_path=os.path.join(
        paths.dataset_root(),
        'Parsing', 'Helen', 'example_aug_names.txt'))
DATASET_HANDLERS['helen_smith_train_simple'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=True,
    adjust_gamma=True, grey_ratio=0.1,
    list_path=os.path.join(
        paths.dataset_root(),
        'Parsing', 'Helen', 'example_aug_names.txt'))
DATASET_HANDLERS['helen_smith_train_pure'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=True,
    adjust_gamma=False, grey_ratio=0.0,
    list_path=os.path.join(
        paths.dataset_root(),
        'Parsing', 'Helen', 'example_aug_names.txt'), no_occ=True)
DATASET_HANDLERS['helen_smith_train_toy'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=False, from_to_ratio=[0, 0.1],
    adjust_gamma=True, grey_ratio=0.1,
    list_path=os.path.join(
        paths.dataset_root(),
        'Parsing', 'Helen', 'example_aug_names.txt'))
DATASET_HANDLERS['helen_smith_train_and_tune'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=False,
    adjust_gamma=True, grey_ratio=0.1,
    list_path=os.path.join(
        paths.dataset_root(),
        'Parsing', 'Helen', 'example_and_tune_aug_names.txt'))
DATASET_HANDLERS['helen_smith_train_no_mask'] = \
    lambda options: _filter_out_masks(HelenSmithDataset(
        options, no_augmented_samples=False,
        adjust_gamma=True, grey_ratio=0.1,
        list_path=os.path.join(
            paths.dataset_root(),
            'Parsing', 'Helen', 'example_aug_names.txt')))
DATASET_HANDLERS['helen_smith_train_no_headbox'] = \
    lambda options: _filter_out_head_boxes(HelenSmithDataset(
        options, no_augmented_samples=False,
        adjust_gamma=True, grey_ratio=0.1,
        list_path=os.path.join(
            paths.dataset_root(),
            'Parsing', 'Helen', 'example_aug_names.txt')))

DATASET_HANDLERS['helen_all'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=False,
    adjust_gamma=False, grey_ratio=0,
    list_path=None,
    aug_flip=False, aug_transform=False, no_occ=True, aug_larger=False)

DATASET_HANDLERS['helen_smith_train_x'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=False,
    adjust_gamma=True, grey_ratio=0.1,
    list_path=os.path.join(
        paths.dataset_root(),
        'Parsing', 'Helen', 'example_aug_names.txt'),
    aug_flip=True, aug_transform=True, no_occ=True)
DATASET_HANDLERS['helen_smith_train_y'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=False,
    adjust_gamma=True, grey_ratio=0.1,
    list_path=os.path.join(
        paths.dataset_root(),
        'Parsing', 'Helen', 'example_aug_names.txt'),
    aug_flip=True, aug_transform=True, no_occ=True, aug_larger=True)
DATASET_HANDLERS['helen_smith_train_y_no_mask'] = \
    lambda options: _filter_out_masks(HelenSmithDataset(
        options, no_augmented_samples=False,
        adjust_gamma=True, grey_ratio=0.1,
        list_path=os.path.join(
            paths.dataset_root(),
            'Parsing', 'Helen', 'example_aug_names.txt'),
        aug_flip=True, aug_transform=True, no_occ=True, aug_larger=True))
DATASET_HANDLERS['helen_smith_train_y_no_headbox'] = \
    lambda options: _filter_out_head_boxes(HelenSmithDataset(
        options, no_augmented_samples=False,
        adjust_gamma=True, grey_ratio=0.1,
        list_path=os.path.join(
            paths.dataset_root(),
            'Parsing', 'Helen', 'example_aug_names.txt'),
        aug_flip=True, aug_transform=True, no_occ=True, aug_larger=True))
DATASET_HANDLERS['helen_smith_train_and_tune_y'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=False,
    adjust_gamma=True, grey_ratio=0.1,
    list_path=os.path.join(
        paths.dataset_root(),
        'Parsing', 'Helen', 'example_and_tune_aug_names.txt'),
    aug_flip=True, aug_transform=True, no_occ=True, aug_larger=True)


DATASET_HANDLERS['helen_original_test'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=True,
    adjust_gamma=False, grey_ratio=0.0,
    list_path=os.path.join(
        paths.dataset_root(), 'Parsing', 'Helen', 'original_test_names.txt'))
DATASET_HANDLERS['helen_original_train'] = lambda options: HelenSmithDataset(
    options, no_augmented_samples=False,
    adjust_gamma=True, grey_ratio=0.1,
    list_path=os.path.join(
        paths.dataset_root(),
        'Parsing', 'Helen', 'original_train_names.txt'))


class LFWDataset(SegmentationDatasetBase):
    def __init__(self, options, train=True):
        super(LFWDataset, self).__init__([
            'background', 'face', 'hair'
        ], options)

        # assert self.im_size  == 250

        lfw_root = '\\\\MININT-37Q0T4O\\Datasets\\LFW-PL'

        self.train = train

        if self.train:
            self.grey_ratio = 0.1
            self.adjust_gamma = True
            self.aug_flip = True
            self.aug_transform = True
            self.list_file = lfw_root + '\\parts_train.txt'
        else:
            self.grey_ratio = 0
            self.adjust_gamma = False
            self.aug_flip = False
            self.aug_transform = False
            self.list_file = lfw_root + '\\parts_test.txt'

        image_paths = []
        label_paths = []
        for line in open(self.list_file):
            name, ind = line.strip().split(' ')
            ind = int(ind)
            ind = '%04d' % ind
            image_path = lfw_root + '\\images\\' + name + '\\' + name + '_' + ind + '.jpg'
            label_path = lfw_root + '\\parts\\parts_lfw_funneled_gt_images_png\\' + \
                name + '_' + ind + '.png'
            image_paths.append(image_path)
            label_paths.append(label_path)

        print('final img path num: %d' % len(image_paths))

        for i, p in enumerate(image_paths):
            self.add_image(
                "Face",
                image_id=i,
                name=p,
                path=p,
                label_path=label_paths[i])
        self.make_image_ids()

    def prepare_loading(self, image_id):
        if self.adjust_gamma:
            gamma = math.exp(max(-1.6, min(1.6, random.normalvariate(0, 0.8))))
        else:
            gamma = None
        if random.uniform(0, 1) < self.grey_ratio:
            greyfy = True
        else:
            greyfy = False
        if self.aug_flip and random.uniform(0, 1) < 0.5:
            flipped = True
        else:
            flipped = False
        if self.aug_transform:
            scale_v = 0.1
            rotation_v = math.pi / 10
            shift_v = 0.1
            scale = random.normalvariate(1.0, scale_v)
            rotation = random.normalvariate(0, rotation_v)
            shift = np.random.normal(0, shift_v, (2))
            Tparams = (scale, rotation, shift)
        else:
            Tparams = None
        return gamma, greyfy, flipped, Tparams

    def _compose_T(self, im_size, Tparams):
        scale, rotation, shift = Tparams
        trans_scale = transform.SimilarityTransform(scale=scale)
        trans_rot = transform.SimilarityTransform(rotation=rotation)
        center = np.array([im_size / 2, im_size / 2])
        trans_shift = transform.SimilarityTransform(translation=-center)
        trans_shift_inv = transform.SimilarityTransform(
            translation=center + im_size * shift)
        T = trans_scale + (trans_shift + (trans_rot + trans_shift_inv))
        return T

    def _apply_Tparams(self, image, Tparams, order=1, preserve_range=False):
        T = self._compose_T(image.shape[0], Tparams)
        return transform.warp(image, T.inverse, order=order, preserve_range=preserve_range)

    def load_image(self, image_id, pre=None):
        im = io.imread(self.image_info[image_id]['path']).astype(np.uint8)
        # print(im.shape)
        # print(self.image_info[image_id]['path'])

        if pre is not None:
            gamma, greyfy, flipped, Tparams = pre
            im = im / 255.0
            if gamma is not None:
                im = exposure.adjust_gamma(im, gamma)
            if greyfy:
                im = color.rgb2gray(im)
                im = np.tile(np.expand_dims(im, -1), [1, 1, 3])
            if flipped:
                im = np.flip(im, axis=1)
            if Tparams is not None:
                im = self._apply_Tparams(im, Tparams)
            im = (im * 255).astype(np.uint8)
        assert len(im.shape) == 3 and im.shape[-1] == 3

        if im.shape[0] != self.im_size:
            im = transform.resize(im, (self.im_size, self.im_size))
            im = (im * 255).astype(np.uint8)
        return im, True

    def _flip_label_image(self, label_img):
        label_img = np.flip(label_img, axis=1)
        # flip the label ids
        # lb_id = self.provided_label_names.index('lb')
        # rb_id = self.provided_label_names.index('rb')
        # le_id = self.provided_label_names.index('le')
        # re_id = self.provided_label_names.index('re')
        # label_img_flipped = np.copy(label_img)
        # label_img_flipped[label_img == lb_id] = rb_id
        # label_img_flipped[label_img == rb_id] = lb_id
        # label_img_flipped[label_img == le_id] = re_id
        # label_img_flipped[label_img == re_id] = le_id
        # label_img = label_img_flipped
        return label_img

    def load_label_image(self, image_id, pre=None):
        label_img = io.imread(
            self.image_info[image_id]['label_path'], as_grey=True).astype(np.int32)
        if pre is not None:
            _, _, flipped, Tparams = pre
            if flipped:
                label_img = self._flip_label_image(label_img)
            if Tparams is not None:
                prev_t = label_img.dtype
                label_img = self._apply_Tparams(
                    label_img, Tparams, 0, True).astype(prev_t)
        if label_img.shape[0] != self.im_size:
            label_img = transform.resize(
                label_img, (self.im_size, self.im_size), order=0)
            label_img = (label_img * 255).astype(int)
        assert label_img.shape[0] == self.im_size
        return label_img


DATASET_HANDLERS['lfw_train'] = lambda options: LFWDataset(
    options, train=True)
DATASET_HANDLERS['lfw_test'] = lambda options: LFWDataset(
    options, train=False)

DATASET_HANDLERS['lfw_train_no_mask'] = lambda options: _filter_out_masks(
    LFWDataset(
        options, train=True))


def get_dataset(name, options):
    return DATASET_HANDLERS[name](options)


def _get_min_helen_paddings():
    options = yaml.load(
        open(os.path.join(ROOT_DIR, 'options', 'helen_v2.yaml')))
    dataset = HelenSmithDataset(
        options, grey_ratio=0.0, adjust_gamma=False, no_augmented_samples=True)

    max_padding = np.tile(np.array([[448, 448, 0, 0]], np.float32), [7, 1])
    paddings = np.zeros([dataset.num_images, 7, 4])

    for i in dataset.image_ids:
        im, _ = dataset.load_image(i)
        head_boxes, _ = dataset.load_head_boxes(i)
        landmark68_pts, _ = dataset.load_lbf_landmark68_pts(i)

        head_label_names = []
        for class_ids in options['heads']:
            names_this_head = []
            for class_id in class_ids:
                names_this_head += options['class_names'][class_id]
            head_label_names.append(names_this_head)
        boxes = utils.extract_landmark68_boxes(
            landmark68_pts, head_label_names, padding_dict=None)

        padding = head_boxes - boxes
        max_padding[:, 0] = np.minimum(max_padding[:, 0], padding[:, 0])
        max_padding[:, 1] = np.minimum(max_padding[:, 1], padding[:, 1])
        max_padding[:, 2] = np.maximum(max_padding[:, 2], padding[:, 2])
        max_padding[:, 3] = np.maximum(max_padding[:, 3], padding[:, 3])

        # delts =
        paddings[i] = padding

        # fig = plt.figure(figsize=(8, 8))
        # visualize.tf_display_boxes_pts(
        #     im, head_boxes / 448, None, np.zeros([3]), fig)
        # # plt.show()
        # visualize.tf_display_boxes_pts(
        #     im, boxes / 448, None, np.zeros([3]), fig)
        # plt.show()

    savemat('helen_lbf_lm68_paddings.mat', {
            'paddings': paddings}, do_compression=True)
    print(max_padding)


def _multipie_alphas():
    pass

# def _compute_mean_head_boxes():
#     options = yaml.load(
#         open(os.path.join(ROOT_DIR, 'options', 'fa68_helen_googlenet.yaml')))
#     ds = get_dataset('helen_smith_train', options)
#     for i in ds.image_inds:
#         (pts, _)


# if __name__ == '__main__':

    # '100040721_1_00_img.png'

    # options = yaml.load(
    #     open(os.path.join(ROOT_DIR, 'options', 'multipie_hairmatte.yaml')))
    # options['image_size'] = 512
    # # ds = get_dataset('helen_smith_train_y', options)

    # # ds = HelenSmithDataset(
    # #     options, no_augmented_samples=True,
    # #     adjust_gamma=False, grey_ratio=0,
    # #     list_path=os.path.join(
    # #         paths.dataset_root(),
    # #         'Parsing', 'Helen', 'example_aug_names.txt'),
    # #     aug_flip=False, aug_transform=False, no_occ=True, aug_larger=False)

    # ds = get_dataset('multipie_train', options)
    # print('what')
    # inds = list(range(ds.num_images))
    # # random.shuffle(inds)
    # for i in inds:
    #     # if not '1000' in ds.source_image_link(i):
    #         # continue
    #     print(ds.source_image_link(i))
    #     (im, _), (molded_image, _), (masks, _) = ds.load_data_as_list(
    #         i, ['image', 'molded_image', 'masks'])
    #     print(np.unique(im), np.unique(molded_image), np.unique(masks))
    #     # print(np.max(im), np.max(im), np.max(masks))
    #     blended = utils.blend_labels(im, utils.flatten_masks(masks))
    #     blended = np.concatenate([im / 255.0, blended], axis=1)

        # # (im, _), (pts, _) = ds.load_data_as_list(
        # #     i, ['image', 'molded_landmark68_pts'])
        # # blended = im / 255.0

        # fig, ax = plt.subplots(1)
        # ax.axis('off')
        # # for i in range(pts.shape[0]):
        # #     # pts
        # #     x, y = pts[i] * np.array([im.shape[0], im.shape[1]], np.float32)
        # #     p = patches.Circle((x, y), radius=1.0, edgecolor='red')
        # #     ax.add_patch(p)
        # #     ax.text(x, y, str(i), fontsize=7, color='red')
        # '''
        # for i in range(boxes.shape[0]):
        #     # boxes
        #     y1, x1, y2, x2 = boxes[i] * ds.im_size
        #     x1 += ds.im_size
        #     x2 += ds.im_size
        #     p = patches.Rectangle(
        #         (x1, y1),
        #         x2 - x1,
        #         y2 - y1,
        #         linewidth=2,
        #         alpha=0.7,
        #         linestyle="dashed",
        #         edgecolor='blue',
        #         facecolor='none')
        #     ax.add_patch(p)
        # '''
        # ax.imshow(blended)
        # plt.show()
