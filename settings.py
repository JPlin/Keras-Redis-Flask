import os
import redis

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
# initialize app config
SECRET_KEY = '123456'
TEMP_FOLDER = os.path.join('tmp/')
UPLOAD_FOLDER = os.path.join(PROJECT_PATH , '/tmp/')
THUMBNAIL_FOLDER = os.path.join(PROJECT_PATH , '/tmp/')
MAX_CONTENT_LENGTH = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['txt', 'gif', 'png', 'jpg',
                          'jpeg', 'bmp', 'rar', 'zip', '7zip', 'doc', 'docx'])
IGNORED_FILES = set(['.gitignore' , 'score.json'])

# initialize AWS settings
IMAGE_BUCKET = 'parsing-img'
THUMBNAIL_BUCKET = 'parsing-thumbnail'
OTHER_BUCKET = 'parsing-other'

# initialize Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = "6379"
REDIS_DB = 0
#db = redis.StrictRedis(host=REDIS_HOST,
                       port=REDIS_PORT, db=REDIS_DB)
db = redis.from_url(os.environ.get("REDIS_URL"))

# initialize constants used to control classifier image spatial dimensions and data type
C_IMAGE_WIDTH = 224
C_IMAGE_HEIGHT = 224
C_IMAGE_CHANS = 3
C_IMAGE_DTYPE = "float32"
INDEX_PATH = os.path.join(PROJECT_PATH , 'model/classifier/imagenet_class_index.json')
C_WEIGHT_PATH = os.path.join(PROJECT_PATH , 'model/classifier/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

# initialize constants used to control parser image spatial dimensions and data type
P_IMAGE_WIDTH = 512
P_IMAGE_HEIGHT = 512
P_IMAGE_CHANS = 3
P_IMAGE_DTYPE = "float32"

# initialize constants used for server queueing
C_IMAGE_QUEUE = "classifier_image_queue"
P_IMAGE_QUEUE = "parsing_image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.5
CLIENT_SLEEP = 0.5
