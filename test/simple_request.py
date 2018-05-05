# cmd line
# python simple_request.py

# import the necessary packages
import requests
import numpy as np
import base64
import matplotlib.pyplot as plt

# initialize the Keras REST API endpoint URL along with the input
# image path
# KERAS_REST_API_URL = "https://paringweb.herokuapp.com/parsing"
# KERAS_REST_API_URL = "http://paringweb.herokuapp.com/classify"
KERAS_REST_API_URL = "http://localhost:9191/parsing"
IMAGE_PATH = "static/img/test.png"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was sucessful
if r['success']:
    q = r['predictions']
    a = np.frombuffer(base64.decodestring(
        q['image'].encode('utf-8')), dtype=np.uint8)
    a = a.reshape((512, 512, 3))
    plt.imshow(a)
    plt.show()
    input('Press any key')
else:
    print("Request failed")

# if r["success"]:
#     # loop over the predictions and display them
#     for (i, result) in enumerate(r["predictions"]):
#         print("{}. {}: {:.4f}".format(i + 1, result["label"],
#                                       result["probability"]))

# # otherwise, the request failed
# else:
#     print("Request failed")
