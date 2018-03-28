#!flask/bin/python

# Author: Ngo Duy Khanh
# Email: ngokhanhit@gmail.com
# Git repository: https://github.com/ngoduykhanh/flask-file-uploader
# This work based on jQuery-File-Upload which can be found at https://github.com/blueimp/jQuery-File-Upload/
import PIL
import simplejson
import traceback
import redis
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
import io
import os

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from flask_bootstrap import Bootstrap
from werkzeug import secure_filename

app = Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
app.config['SECRET_KEY'] = '123456'
app.config['UPLOAD_FOLDER'] = 'data/'
app.config['THUMBNAIL_FOLDER'] = 'data/thumbnail/'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'gif', 'png', 'jpg',
                          'jpeg', 'bmp', 'rar', 'zip', '7zip', 'doc', 'docx'])
IGNORED_FILES = set(['.gitignore'])

bootstrap = Bootstrap(app)


def classify_process():
        # load the pre-trained Keras model (here we are using a model
        # pre-trained on ImageNet and provided by Keras, but you can
        # substitute in your own networks just as easily)
    print("* Loading model...")
    print("* Model loaded")

    # continually pool for new images to classify
    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        print('??')

        # sleep for a small amount
        time.sleep(1)


if __name__ == '__main__':
    # ignore the warning
    import warnings
    warnings.simplefilter("ignore")
    # load the function used to classify input images in a *separate*
    # thread than the one used for main classification
    print("* Starting model service...")
    t = Thread(target=classify_process(), args=())
    t.daemon = True
    t.start()

    # start the web server
    print("* Starting web service...")
    app.run(debug=True, port=9191, threaded=True)
