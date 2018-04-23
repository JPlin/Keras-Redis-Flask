import os
import redis

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
USE_CLASSIFIER = False
USE_PARSER = True
# initialize --- app config ---
SECRET_KEY = '123456'
TEMP_FOLDER = os.path.join('tmp/')
UPLOAD_FOLDER = os.path.join(PROJECT_PATH, '/tmp/')
THUMBNAIL_FOLDER = os.path.join(PROJECT_PATH, '/tmp/')
MAX_CONTENT_LENGTH = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['txt', 'gif', 'png', 'jpg',
                          'jpeg', 'bmp', 'rar', 'zip', '7zip', 'doc', 'docx'])
IGNORED_FILES = set(['.gitignore', 'score.json'])

# initialize --- AWS settings ---
IMAGE_BUCKET = 'parsing-img'
THUMBNAIL_BUCKET = 'parsing-thumbnail'
RESULT_BUCKET = 'parsing-result'
OTHER_BUCKET = 'parsing-other'

# initialize --- Redis connection settings ---

REDIS_URL = 'redis://h:p90abd37d0c5ea5a94cf2f364d7b099b4386d10975a5b2e6f0033cea3b9085317@ec2-34-201-226-230.compute-1.amazonaws.com:12709'
REDIS_HOST = "localhost"
REDIS_PORT = "6379"
REDIS_DB = 0
# db = redis.StrictRedis(host=REDIS_HOST,
#                       port=REDIS_PORT, db=REDIS_DB)
#db = redis.from_url(os.environ.get("REDIS_URL"))
db = redis.from_url(REDIS_URL)

# initialize constants used to --- control classifier image spatial dimensions and data type ---
C_IMAGE_WIDTH = 224
C_IMAGE_HEIGHT = 224
C_IMAGE_CHANS = 3
C_IMAGE_DTYPE = "float32"
INDEX_PATH = os.path.join(
    PROJECT_PATH, 'model/classifier/imagenet_class_index.json')
C_WEIGHT_PATH = os.path.join(
    PROJECT_PATH, 'model/classifier/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

# initialize constants used to --- control parser image spatial dimensions and data type ---
P_IMAGE_WIDTH = 512
P_IMAGE_HEIGHT = 512
P_IMAGE_CHANS = 3
P_IMAGE_DTYPE = "float32"
P_WEIGHT_PATH = os.path.join(PROJECT_PATH, 'model/parser/parsing.h5')
P_YAML_PATH = os.path.join(PROJECT_PATH, 'model/parser/multipie_jinpli.yaml')

# initialize constants used for -- server queueing ---
C_IMAGE_QUEUE = "classifier_image_queue"
P_IMAGE_QUEUE = "parsing_image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.05
CLIENT_SLEEP = 0.05
