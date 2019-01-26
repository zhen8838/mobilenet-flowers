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
    RESTORE_CKPT_PATH = 'log/train/save_18:40:14'
    TEST_IMG_NUM = 100
    # ===================== end =============================
    namelist, labellist = get_filelist(TEST_PATH)
    dataset, epochstep = create_dataset(namelist, labellist, BATCH_SIZE, is_training=False)
    next_img, next_label = create_iter(dataset)

    with tf.Session() as sess:
        # load the model~
        g = tf.get_default_graph()
        tf.saved_model.loader.load(sess, ["serve"], RESTORE_CKPT_PATH)
        # get output tensor
        predict = g.get_tensor_by_name('Output_label:0')
        batch_label = tf.placeholder(tf.float32, shape=[None, 1, 1, 5])
        # clac the loss and acc
        accuracy, accuracy_op = tf.metrics.accuracy(tf.argmax(batch_label, axis=-1), tf.argmax(predict, axis=-1))
        tf.losses.softmax_cross_entropy(batch_label, predict)  # NOTE predict can't be softmax
        total_loss = tf.losses.get_total_loss()  # NOTE add this can use in test

        sess.run(tf.local_variables_initializer())
        with tqdm(total=TEST_IMG_NUM, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}{postfix}]', unit=' batch', dynamic_ncols=True) as t:
            for j in range(TEST_IMG_NUM):
                img, label = sess.run([next_img, next_label])
                pred, acc, los, _ = sess.run([predict, accuracy, total_loss, accuracy_op], feed_dict={'Input_image:0': img, batch_label: label})  # type:np.ndarray
                t.set_postfix(loss='{:<5.3f}'.format(los), acc='{:5.2f}%'.format(acc*100))
                t.update()
