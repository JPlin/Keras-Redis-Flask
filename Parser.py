# USAGE
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from threading import Thread
from PIL import Image
from . import tool
import numpy as np
import base64
import flask
import redis
import uuid
import time
import json
import sys
import io
