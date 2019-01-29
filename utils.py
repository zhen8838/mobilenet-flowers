from Globals import TRAIN_PATH, TEST_PATH, LABEL_PATH, BATCH_SIZE, CLASS_NUM, SEED
import csv
import tensorflow as tf
from preprocessing import inception_preprocessing
import matplotlib.pyplot as plt
import numpy as np


def parser(filename, label, is_training):
    # with tf.gfile.GFile(filename, 'rb') as f:
    img = tf.read_file(filename)  # f.read()
    img = tf.image.decode_jpeg(img, channels=3)
    # NOTE the inception_preprocessing will convert image scale to [-1,1]

    img_resized = inception_preprocessing.preprocess_image(img, 240, 320, is_training=is_training,
                                                           add_image_summaries=False)

    one_hot_label = tf.one_hot(label, CLASS_NUM, 1, 0)
    # one_hot_label = [5]
    one_hot_label = one_hot_label[tf.newaxis, tf.newaxis, :]
    net_label = tf.tile(one_hot_label, [4, 6, 1])
    # NOTE 匹配网络输出,只有(0,0)有效

    return img_resized, net_label


def get_filelist(fliepath):
    name_list = []
    label_list = []
    train_reader = csv.reader(open(fliepath, 'r'))
    for pa, la in train_reader:
        name_list.append(pa)
        label_list.append(int(la))
    return name_list, label_list


def create_dataset(namelist, labelist, batchsize, is_training=True):
    # create the dataset from the list
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(namelist), tf.constant(labelist)))
    # parser the data set
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=lambda filename, label:
        parser(filename, label, is_training),
        batch_size=batchsize,
        # add drop_remainder avoid output shape less than batchsize
        drop_remainder=True))
    # shuffle and repeat
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(50, seed=SEED))
    # clac step for per epoch
    step_for_epoch = int(len(labelist)/batchsize)
    return dataset, step_for_epoch


def create_iter(dataset):
    data_it = dataset.make_one_shot_iterator()
    # 定义个获取下一组数据的操作(operator)
    return data_it.get_next()


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in (vars(args)).items():
            f.write('%s: %s\n' % (key, str(value)))
