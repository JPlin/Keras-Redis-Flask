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

from lib.upload_file import uploadfile
from lib import tool
from Classifier import Classifier
import settings

app = Flask(__name__)
app.config['SECRET_KEY'] = settings.SECRET_KEY
app.config['UPLOAD_FOLDER'] = settings.UPLOAD_FOLDER
app.config['THUMBNAIL_FOLDER'] = settings.THUMBNAIL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = settings.MAX_CONTENT_LENGTH
bootstrap = Bootstrap(app)
# db = redis.StrictRedis(host=settings.REDIS_HOST,
#                       port=settings.REDIS_PORT, db=settings.REDIS_DB)
db = redis.from_url(os.environ.get("REDIS_URL"))


@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files = request.files['file']

        if files:
            filename = secure_filename(files.filename)
            filename = tool.gen_file_name(
                filename, app.config['UPLOAD_FOLDER'])
            mime_type = files.content_type

            if not tool.allowed_file(files.filename, settings.ALLOWED_EXTENSIONS):
                result = uploadfile(
                    name=filename, type=mime_type, size=0, not_allowed_msg="File type not allowed")

            else:
                # save file to disk
                uploaded_file_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], filename)
                files.save(uploaded_file_path)

                # create thumbnail after saving
                if mime_type.startswith('image'):
                    tool.create_thumbnail(
                        filename, app.config['UPLOAD_FOLDER'], app.config['THUMBNAIL_FOLDER']
                    )

                # get file size after saving
                size = os.path.getsize(uploaded_file_path)

                # return json for js call back
                result = uploadfile(name=filename, type=mime_type, size=size)

            return simplejson.dumps({"files": [result.get_file()]})

    if request.method == 'GET':
        # get all file in ./data directory
        files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(
            os.path.join(app.config['UPLOAD_FOLDER'], f)) and f not in settings.IGNORED_FILES]

        file_display = []

        for f in files:
            size = os.path.getsize(os.path.join(
                app.config['UPLOAD_FOLDER'], f))
            file_saved = uploadfile(name=f, size=size)
            file_display.append(file_saved.get_file())

        return simplejson.dumps({"files": file_display})

    return redirect(url_for('index'))


@app.route("/delete/<string:filename>", methods=['DELETE'])
def delete(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_thumb_path = os.path.join(app.config['THUMBNAIL_FOLDER'], filename)

    if os.path.exists(file_path):
        try:
            os.remove(file_path)

            if os.path.exists(file_thumb_path):
                os.remove(file_thumb_path)

            return simplejson.dumps({filename: 'True'})
        except:
            return simplejson.dumps({filename: 'False'})


# serve static files
@app.route("/thumbnail/<string:filename>", methods=['GET'])
def get_thumbnail(filename):
    return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename=filename)


@app.route("/data/<string:filename>", methods=['GET'])
def get_file(filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER']), filename=filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route("/classify", methods=["POST"])
def classifier_predict():
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            data = {"success": False}
            # read the image in PIL format and prepare it for
            # classification
            image = flask.request.files["image"].read()
            # initialize the data dictionary that will be returned from the view
            image = Image.open(io.BytesIO(image))
            image = Classifier.prepare_image(
                image, (settings.C_IMAGE_WIDTH, settings.C_IMAGE_HEIGHT))

            # ensure our NumPy array is C-contiguous as well,
            # otherwise we won't be able to serialize it
            image = image.copy(order="C")

            # generate an ID for the classification then add the
            # classification ID + image to the queue
            k = str(uuid.uuid4())
            d = {"id": k, "image": tool.base64_encode_image(image)}
            db.rpush(settings.C_IMAGE_QUEUE, json.dumps(d))
            print('push image : ', k)

            # keep looping until our model server returns the output
            # predictions
            while True:
                # attempt to grab the output predictions
                output = db.get(k)

                # check to see if our model has classified the input
                # image
                if output is not None:
                    # add the output predictions to our data
                    # dictionary so we can return it to the client
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    print('get result :', k)
                    # delete the result from the database and break
                    # from the polling loop
                    db.delete(k)
                    break

                # sleep for a small amount to give the model a chance
                # to classify the input image
                time.sleep(settings.CLIENT_SLEEP)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


@app.route("/parsing", methods=["POST"])
def parser_predict():
    pass


if __name__ == '__main__':
    # ignore the warning
    import warnings
    warnings.simplefilter("ignore")
    # load the function used to classify input images in a *separate*
    # thread than the one used for main classification
    '''
    t = Thread(target=classfier.classify_process, args=())
    t.daemon = True
    t.start()
    '''
    # start the web server
    print("* Starting web service...")
    app.run(debug=True, port=9191, threaded=True)
