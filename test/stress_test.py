# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread
import requests
import time

# initialize the Keras REST API endpoint URL along with the input
# image path
#KERAS_REST_API_URL = "http://localhost:9191/classify"
#KERAS_REST_API_URL = "http://localhost:9191/parsing"
KERAS_REST_API_URL = "http://localhost:9191/parsing"

IMAGE_PATH = "static/img/test.png"

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 64
SLEEP_COUNT = 0.05

start_time = time.time()
total_time = 0
def call_predict_endpoint(n):
    global total_time
    # load the input image and construct the payload for the request
    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}

    # submit the request
    r = requests.post(KERAS_REST_API_URL, files=payload).json()

    time_consume = time.time() - start_time
    # ensure the request was sucessful
    if r["success"]:
        print("[INFO] thread {} OK , time {}".format(
            n, time_consume))

    # otherwise, the request failed
    else:
        print("[INFO] thread {} FAILED , time {}".format(
            n, time_consume))
    total_time += time_consume

thread_list = []
# loop over the number of threads
for i in range(0, NUM_REQUESTS):
    # start a new thread to call the API
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    t.start()
    thread_list.append(t)
    time.sleep(SLEEP_COUNT)

for i in thread_list:
    i.join()
print('average time consume :' , total_time / 64)
# insert a long sleep so we can wait until the server is finished
# processing the images
time.sleep(300)
