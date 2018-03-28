import os
import PIL
import simplejson
import traceback
from PIL import Image
import numpy as np
import base64
import uuid
import time
import json
import sys
import io

''' about image process
'''


def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions


def gen_file_name(filename, UPLOAD_FOLDER):
    """
    If file was exist already, rename it and return a new name
    """

    i = 1
    while os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
        name, extension = os.path.splitext(filename)
        filename = '%s_%s%s' % (name, str(i), extension)
        i += 1

    return filename


def create_thumbnail(image, UPLOAD_FOLDER, THUMBNAIL_FOLDER):
    try:
        base_width = 80
        img = Image.open(os.path.join(UPLOAD_FOLDER, image))
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img = img.resize((base_width, h_size), PIL.Image.ANTIALIAS)
        img.save(os.path.join(THUMBNAIL_FOLDER, image))
        return True

    except:
        print(traceback.format_exc())
        return False


def base64_encode_image(a):
        # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")


def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)

    # return the decoded image
    return a
