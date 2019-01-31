import csv
import tensorflow as tf
from preprocessing import inception_preprocessing
import matplotlib.pyplot as plt
import numpy as np


def parser(filename, label, class_num, height, witdh, is_training):
    # with tf.gfile.GFile(filename, 'rb') as f:
    img = tf.read_file(filename)  # f.read()
    img = tf.image.decode_jpeg(img, channels=3)

    img_resized = inception_preprocessing.preprocess_image(
        img, height, witdh, is_training=is_training,
        add_image_summaries=False)
    # NOTE the inception_preprocessing will convert image scale to [-1,1]

    one_hot_label = tf.one_hot(label, class_num, 1, 0)
    one_hot_label = one_hot_label[tf.newaxis, tf.newaxis, :]
    # NOTE 匹配网络输出,只有(0,0)有效
    return img_resized, one_hot_label


def get_filelist(fliepath):
    name_list = []
    label_list = []
    train_reader = csv.reader(open(fliepath, 'r'))
    for pa, la in train_reader:
        name_list.append(pa)
        label_list.append(int(la))
    return name_list, label_list


def create_dataset(namelist: list, labelist: list, batchsize: int, class_num: int,
                   height: int, witdh: int, seed: int, is_training=True):
    # create the dataset from the list
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(namelist), tf.constant(labelist)))
    # parser the data set
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=lambda filename, label:
        parser(filename, label, class_num,
               height, witdh, is_training),
        batch_size=batchsize,
        # add drop_remainder avoid output shape less than batchsize
        drop_remainder=True))
    # shuffle and repeat
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(50, None, seed=seed))
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
