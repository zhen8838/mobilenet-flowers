import os
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import inception_preprocessing
import sys
import argparse
from Globals import *
import tensorflow as tf
from tensorflow.contrib import slim
from nets.mobilenet_v1 import *
from utils import *
from tqdm import tqdm
from datetime import datetime
from skimage.io import imshow, show


PRE_0_5_CKPT = 'mobilenet_v1_0.5_224/mobilenet_v1_0.5_224.ckpt'
PRE_0_75_CKPT = 'mobilenet_v1_0.75_224/mobilenet_v1_0.75_224.ckpt'
PRE_1_0_CKPT = 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt'


def new_mobilenet(images: tf.Tensor, num_classes: int, depth_multiplier: float, is_training: bool):
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
        nets, endpoints = mobilenet_v1_base(images, depth_multiplier=depth_multiplier)

    # add the new layer
    with tf.variable_scope('Flowers'):
        with slim.arg_scope([slim.conv2d],  padding='VALID', activation_fn=None, weights_initializer=slim.initializers.xavier_initializer_conv2d()):
            with slim.arg_scope([slim.batch_norm], activation_fn=tf.nn.relu6,  center=True, scale=True, decay=0.9997, epsilon=0.001):
                if depth_multiplier == 1.0 or depth_multiplier == 0.75:
                    # nets= [?,8,10,1024]
                    nets = slim.conv2d(nets, 512, (3, 3), padding='SAME')
                    nets = slim.batch_norm(nets)
                else:
                    pass
                # nets=(?, 8, 10, 512)
                nets = slim.conv2d(nets, 256, (3, 3))
                nets = slim.batch_norm(nets)
                # nets = (?, 6, 8, 256)
                nets = slim.conv2d(nets, 128, (3, 3))
                nets = slim.batch_norm(nets)
                # nets = (?, 4, 6, 128)
                nets = slim.conv2d(nets, 5, (1, 1), activation_fn=None)
                # nets = (?, 4, 6, 5)
                # tf.contrib.layers.softmax(nets)
    return nets, endpoints


def restore_ckpt(sess: tf.Session, ckptpath: str):
    variables_to_restore = slim.get_model_variables()
    loader = tf.train.Saver([var for var in variables_to_restore if 'MobilenetV1' in var.name])
    loader.restore(sess, ckptpath)


def main(args):
    tf.random.set_random_seed(args.seed)
    # generate the data
    namelist, labellist = get_filelist(args.data_path)
    dataset, epochstep = create_dataset(namelist, labellist, args.batch_size)
    next_img, next_label = create_iter(dataset)
    # define the model
    # NOTE add placeholder_with_default node for test
    batch_image = tf.placeholder_with_default(next_img, shape=[None, 240, 320, 3], name='Input_image')
    batch_label = tf.placeholder_with_default(next_label, shape=[None, 4, 6, 5], name='Input_label')
    true_label = batch_label[:, 0, 0, :]
    nets, endpoints = new_mobilenet(batch_image, args.class_num, args.depth_multiplier, is_training=True)
    logits = nets[:, 0, 0, :]
    predict = tf.identity(logits, name='Output_label')
    # define loss
    tf.losses.softmax_cross_entropy(true_label, predict)  # NOTE predict can't be softmax
    total_loss = tf.losses.get_total_loss(name='total_loss')  # NOTE add this can use in test
    # =========== define the hyperparamter===================
    # todo 增加学习率递减
    # step_cnt = tf.Variable(0, trainable=False)
    global_steps = tf.train.create_global_step()
    # ===================== end =============================

    # define train optimizer
    current_learning_rate = tf.train.exponential_decay(args.init_learning_rate, global_steps, args.learning_rate_decay_epochs*args.max_nrof_epochs,
                                                       args.learning_rate_decay_factor, staircase=True)

    train_op = tf.train.AdamOptimizer(learning_rate=current_learning_rate).minimize(total_loss, global_step=global_steps)
    # calc the accuracy
    accuracy, accuracy_op = tf.metrics.accuracy(tf.argmax(true_label, axis=-1), tf.argmax(predict, axis=-1), name='clac_acc')

    with tf.Session() as sess:
        # init the model and restore the pre-train weight
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # NOTE the accuracy must init local variable
        if args.depth_multiplier == 0.5:
            pre_ckpt = PRE_0_5_CKPT
        elif args.depth_multiplier == 0.75:
            pre_ckpt = PRE_0_75_CKPT
        elif args.depth_multiplier == 1.0:
            pre_ckpt = PRE_1_0_CKPT
        restore_ckpt(sess, pre_ckpt)
        # define the log and saver
        subdir = os.path.join(args.log_dir, datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(subdir, graph=sess.graph)
        write_arguments_to_file(args, os.path.join(subdir, 'arguments.txt'))
        tf.summary.scalar('loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('leraning rate', current_learning_rate)
        merged = tf.summary.merge_all()
        # 使用进度条库
        for i in range(args.max_nrof_epochs):
            with tqdm(total=epochstep, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}{postfix}]', unit=' batch', dynamic_ncols=True) as t:
                for j in range(epochstep):
                    summary, _, losses, acc, _, lrate, step_cnt = sess.run([merged, train_op, total_loss, accuracy, accuracy_op,
                                                                            current_learning_rate, global_steps])
                    writer.add_summary(summary, step_cnt)
                    t.set_postfix(loss='{:<5.3f}'.format(losses), acc='{:5.2f}%'.format(acc*100), leraning_rate='{:7f}'.format(lrate))
                    t.update()
        saver.save(sess, save_path=os.path.join(subdir, 'model.ckpt'), global_step=global_steps)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=5)

    parser.add_argument('--data_path', type=str,
                        help='Path to the data directory containing csv file.',
                        default='/media/zqh/Datas/DataSet/flower_photos/train.csv')

    parser.add_argument('--log_dir', type=str,
                        help='Path to the log directory ckpt file.',
                        default='log/train')

    parser.add_argument('--depth_multiplier', type=float,
                        help='the mobilenet depth_multiplier can use {0.5,0.75,1.0}', default=0.5)

    parser.add_argument('--class_num', type=int,
                        help='Dimensionality of the embedding.', default=5)

    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=32)

    parser.add_argument('--init_learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.0006)

    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=10)

    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=0.9)

    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
