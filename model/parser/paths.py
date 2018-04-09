import os
import socket


def dataset_root():
    if socket.gethostname() == 'msravcg-w01':
        return '/mnt/data2/remote/msra-facednn11/FaceData'
    else:
        return '//MSRA-FACEDNN11/haya/FaceData'


def face_landmark68_root_path():
    return os.path.join(dataset_root(), 'Landmark', '68p')


def public_landmark68_root_path():
    return os.path.join(dataset_root(), 'Landmark', '300W')


def resnet50_imagenet_model_path():
    return os.path.join(dataset_root(), 'Models', 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


def vgg16_imagenet_model_path():
    return os.path.join(dataset_root(), 'Models', 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
