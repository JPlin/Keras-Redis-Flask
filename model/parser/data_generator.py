import numpy as np
import keras
import tensorflow as tf
import h5py
from pprint import pprint


############################################################
#  Data Generator
############################################################


def data_generator(dataset,
                   required_data_names,
                   shuffle=True,
                   batch_size=1,
                   no_repeat=False):
    """A generator
    required_data_names: a list of data names, supported names include:
        'image', 'gt_masks', 'head_boxes', 'landmark68_pts'
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            image_id = image_ids[image_index]

            # Load single data
            single_data_list = dataset.load_data_as_list(
                image_id, required_data_names)

            # Init batch arrays
            if b == 0:
                batched_exists = [
                    np.zeros([batch_size], dtype=np.uint8)
                    for _ in single_data_list
                ]
                batched_data = [
                    np.zeros((batch_size, ) + d.shape, dtype=d.dtype)
                    for d, _ in single_data_list
                ]

            # Add to batch
            for i, (d, e) in enumerate(single_data_list):
                batched_exists[i][b] = e
                batched_data[i][b] = d

            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = batched_exists + batched_data
                # assert not np.any(batched_exists[1])
                # print([(input.shape[0], input.dtype) for input in inputs])
                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0

            if no_repeat and image_index + 1 == len(image_ids):
                break

        except (GeneratorExit, KeyboardInterrupt):
            raise
        # except:
        #     # Log it and skip the image
        #     print("Error processing image {}".format(
        #         dataset.source_image_link(image_id)))
        #     error_count += 1
        #     if error_count > 1:
        #         raise


def multi_data_generator(datasets,
                         required_data_names,
                         shuffle=True,
                         batch_size=1):
    """A generator
    required_data_names: a list of data names, supported names include:
        'image', 'gt_masks', 'head_boxes', 'landmark68_pts'
    """
    assert isinstance(datasets, list)

    images_inds_table = [np.arange(ds.num_images) for ds in datasets]

    b = 0  # batch item index
    num_datasets = len(datasets)
    image_index = [-1] * num_datasets
    error_count = 0
    error_images = []

    # which dataset in which batch slice
    dataset_id = -1

    # Keras requires a generator to run indefinately.
    while True:
        try:
            dataset_id = (dataset_id + 1) % num_datasets

            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index[dataset_id] = (image_index[dataset_id] + 1) % len(
                images_inds_table[dataset_id])
            if shuffle and image_index[dataset_id] == 0:
                np.random.shuffle(images_inds_table[dataset_id])

            image_id = images_inds_table[dataset_id][image_index[dataset_id]]

            # Load single data
            dataset = datasets[dataset_id]
            single_data_list = dataset.load_data_as_list(
                image_id, required_data_names)
            # print('dataset_id: %d, image_id: %d' % (dataset_id, image_id))

            # Init batch arrays
            if b == 0:
                batched_exists = [
                    np.zeros([batch_size], dtype=np.uint8)
                    for _ in single_data_list
                ]
                batched_data = [
                    np.zeros((batch_size, ) + d.shape, dtype=d.dtype)
                    for d, _ in single_data_list
                ]

            # Add to batch
            for i, (d, e) in enumerate(single_data_list):
                batched_exists[i][b] = e
                batched_data[i][b] = d

            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = batched_exists + batched_data
                # assert not np.any(batched_exists[1])
                # print([(input.shape, input.dtype) for input in inputs])
                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0

        except (GeneratorExit, KeyboardInterrupt):
            raise
        # except:
        #     # Log it and skip the image
        #     print("Error processing image {}".format(
        #         dataset.source_image_link(image_id)))
        #     error_images.append(dataset.source_image_link(image_id))
        #     error_count += 1
        #     pprint(error_images)
        #     # if error_count > 1:
        #     #     raise


class CachedDataSequence(keras.utils.Sequence):
    def __init__(self, generator, num_steps):
        self.num_steps = num_steps
        # preload
        print('preloading %d batched data from data_generator' % num_steps)
        self.batched_data = [None] * num_steps
        for i in tqdm(range(num_steps)):
            self.batched_data[i] = next(generator)
        print('done preloading')

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        return self.batched_data[idx]
