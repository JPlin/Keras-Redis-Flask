# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import tensorflow
import numpy as np
import redis
import time
import json
import os
from lib import tool
import settings

db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)
#db = redis.from_url(os.environ.get("REDIS_URL"))


class Classifier:
    def __init__(self, db):
        self.db = db
        # load the pre-trained Keras model (here we are using a model
        # pre-trained on ImageNet and provided by Keras, but you can
        # substitute in your own networks just as easily)
        print("* Loading model...")
        self.model = ResNet50(weights=settings.C_WEIGHT_PATH)
        self.graph = tensorflow.get_default_graph()
        print("* Model loaded")

    @staticmethod
    def prepare_image(image, target):
        # if the image mode is not RGB, convert it
        if image.mode != "RGB":
            image = image.convert("RGB")

        # resize the input image and preprocess it
        image = image.resize(target)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # return the processed image
        return image
    
    def one_image_classify(self , image):
        image = image.reshape((1, settings.C_IMAGE_HEIGHT, settings.C_IMAGE_WIDTH, settings.C_IMAGE_CHANS))
        print("image.shape " , image.shape)
        with self.graph.as_default():
            preds = self.model.predict(image)
        results = imagenet_utils.decode_predictions(preds)
        outputs = []
        for ( _ , label , prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            outputs.append(r)
        return outputs

    def batch_image_classify(self):
        # continually pool for new images to classify
        while True:
            # attempt to grab a batch of images from the database, then
            # initialize the image IDs and batch of images themselves
            queue = self.db.lrange(
                settings.C_IMAGE_QUEUE, 0, settings.BATCH_SIZE - 1)
            imageIDs = []
            batch = None
            # loop over the queue
            for q in queue:
                # deserialize the object and obtain the input image
                q = json.loads(q.decode("utf-8"))
                image = tool.base64_decode_image(q["image"], settings.C_IMAGE_DTYPE,
                                                 (1, settings.C_IMAGE_HEIGHT, settings.C_IMAGE_WIDTH, settings.C_IMAGE_CHANS))

                # check to see if the batch list is None
                if batch is None:
                    batch = image

                # otherwise, stack the data
                else:
                    batch = np.vstack([batch, image])

                # update the list of image IDs
                imageIDs.append(q["id"])

            # check to see if we need to process the batch
            if len(imageIDs) > 0:
                # classify the batch
                print("* Batch size: {}".format(batch.shape))
                preds = self.model.predict(batch)
                results = imagenet_utils.decode_predictions(preds)

                # loop over the image IDs and their corresponding set of
                # results from our model
                for (imageID, resultSet) in zip(imageIDs, results):
                    # initialize the list of output predictions
                    output = []

                    # loop over the results and add them to the list of
                    # output predictions
                    for (imagenetID, label, prob) in resultSet:
                        r = {"label": label, "probability": float(prob)}
                        output.append(r)

                    # store the output predictions in the database, using
                    # the image ID as the key so we can fetch the results
                    self.db.set(imageID, json.dumps(output))

                # remove the set of images from our queue
                self.db.ltrim(settings.C_IMAGE_QUEUE, len(imageIDs), -1)

            # sleep for a small amount
            time.sleep(settings.SERVER_SLEEP)


# if this is the main thread of execution start the model server process
if __name__ == '__main__':
    print("* Starting model service...")
    classfier = Classifier(db)
    classfier.batch_image_classify()
