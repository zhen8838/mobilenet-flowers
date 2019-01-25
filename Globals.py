import os


DATA_DIR = '/media/zqh/Datas/DataSet/flower_photos'
CLASS_NUM = 5
SEED = 3
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
LABEL_PATH = os.path.join(DATA_DIR, 'Label.csv')
TRAIN_NUM = 3303
BATCH_SIZE = 32

# LOG_DIR = 'flower_graph'
TRAIN_LOG_DIR = 'log/train'
TEST_LOG_DIR = 'log/test'
PRE_0_5_CKPT = 'mobilenet_v1_0.5_224/mobilenet_v1_0.5_224.ckpt'
PRE_0_75_CKPT = 'mobilenet_v1_0.75_224/mobilenet_v1_0.75_224.ckpt'
PRE_1_0_CKPT = 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt'
NEW_CKPT_NAME = 'flower.ckpt'
