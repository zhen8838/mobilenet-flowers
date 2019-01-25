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


def restore_ckpt(sess: tf.Session, ckptpath: str):
    loader = tf.train.Saver()
    loader.restore(sess, ckptpath)


if __name__ == "__main__":
    # =========== define the ckpt path===================
    # NOTE modfiy to your path
    RESTORE_CKPT_PATH = 'log/train/0.50_20:11:11'
    # ===================== end =============================
    # generate the data
    namelist, labellist = get_filelist(TEST_PATH)
    dataset, epochstep = create_dataset(namelist, labellist, 8, parser)
    next_img, next_label = create_iter(dataset)
    # define the model
    predict, endpoints = new_mobilenet(next_img, CLASS_NUM, depth_multiplier, is_training=True)
    # define loss
    loss = tf.losses.softmax_cross_entropy(next_label, predict)

    # calc the accuracy
    accuracy, accuracy_op = tf.metrics.accuracy(tf.argmax(next_label, axis=-1), tf.argmax(predict, axis=-1))

    with tf.Session() as sess:
        restore_ckpt(sess, pre_ckpt)
        # define the log and saver
        nowtime = datetime.now()
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(os.path.join(TRAIN_LOG_DIR, '{:2.2f}_{:%H:%M:%S}'.format(depth_multiplier, nowtime)), graph=sess.graph)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('leraning rate', current_learning_rate)
        merged = tf.summary.merge_all()
        # 使用进度条库
        for i in range(epoch):
            with tqdm(total=epochstep, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}{postfix}]', unit=' batch', dynamic_ncols=True) as t:
                for j in range(epochstep):
                    summary, _, losses, acc, _, lrate = sess.run([merged, train_op, loss, accuracy, accuracy_op, current_learning_rate])
                    step_cnt = i*epochstep+j
                    writer.add_summary(summary, i*epochstep+j)
                    t.set_postfix(loss='{:<5.3f}'.format(losses), acc='{:5.2f}%'.format(acc*100), leraning_rate='{:7f}'.format(lrate))  # 修改此处添加后缀
                    t.update()
        saver.save(sess, os.path.join(TRAIN_LOG_DIR, '{:2.2f}_{:%H:%M:%S}/final.ckpt'.format(depth_multiplier, nowtime)))
