import os
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import inception_preprocessing
from Globals import *
import tensorflow as tf
from tensorflow.contrib import slim
from nets.mobilenet_v1 import *
from load_data import *
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score
import skimage


if __name__ == "__main__":
    tf.reset_default_graph()
    # =========== define the ckpt path===================
    # NOTE modfiy to your path
    RESTORE_CKPT_PATH = 'log/train/0.50_23:05:58'
    TEST_IMG_NUM = 100
    # ===================== end =============================
    # generate the data
    namelist, labellist = get_filelist(TRAIN_PATH)  # TEST_PATH)
    dataset, epochstep = create_dataset(namelist, labellist, BATCH_SIZE, parser)
    next_img, next_label = create_iter(dataset)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(RESTORE_CKPT_PATH, 'final.ckpt') + '.meta')   # 载入图结构，保存在.meta文件中
        # restore the graph and weight
        saver.restore(sess, os.path.join(RESTORE_CKPT_PATH, 'final.ckpt'))
        # todo 增加数据输入的操作!!!!
        # define loss and acc
        accuracy, accuracy_op = tf.metrics.accuracy(tf.argmax(next_label, axis=-1), tf.argmax(predict, axis=-1))
        loss = tf.losses.softmax_cross_entropy(next_label, predict)
        sess.run(tf.local_variables_initializer())  # NOTE the accuracy must init local variable
        # 使用进度条库
        with tqdm(total=TEST_IMG_NUM, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}{postfix}]', unit=' batch', dynamic_ncols=True) as t:
            for j in range(TEST_IMG_NUM):
                acc, los, pred = sess.run([accuracy, loss, predict], {X: next_img.eval()})
                t.set_postfix(loss='{:<5.3f}'.format(los), acc='{:5.2f}%'.format(acc*100))
                t.update()
