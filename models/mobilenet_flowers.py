import tensorflow as tf
from tensorflow.contrib import slim
from models.mobilenet_v1 import *


def mobilenet_separabe(images: tf.Tensor, num_classes: int, depth_multiplier: float, is_training: bool):
    flower_point = ['Conv2d_0_depthwise', 'Conv2d_0_pointwise', 'Conv2d_1_depthwise', 'Conv2d_1_pointwise', 'Final']
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
        nets, endpoints = mobilenet_v1_base(images, depth_multiplier=depth_multiplier)

    # add the new layer
    with tf.variable_scope('Flowers'):
        with slim.arg_scope([slim.batch_norm], is_training=is_training,  center=True, scale=True, decay=0.9997, epsilon=0.001):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d], normalizer_fn=slim.batch_norm, activation_fn=None,):
                if depth_multiplier == 1.0 or depth_multiplier == 0.75:
                    # nets= [?,8,10,1024]
                    nets = slim.conv2d(nets, 512, (3, 3), padding='SAME')
                    nets = slim.batch_norm(nets)
                else:
                    pass
                # nets=(?, 8, 10, 512)
                nets = slim.separable_conv2d(nets, None, (3, 3), stride=2, scope=flower_point[0])
                nets = tf.nn.relu6(nets, name=flower_point[0]+'/relu6')
                endpoints[flower_point[0]] = nets
                # nets = (?, 4, 5, 512)
                nets = slim.conv2d(nets, 256, (1, 1), scope=flower_point[1])
                nets = tf.nn.relu6(nets, name=flower_point[1]+'/relu6')
                endpoints[flower_point[1]] = nets
                # nets = (?, 4, 5, 256)
                nets = slim.separable_conv2d(nets, None, (3, 3), scope=flower_point[2])
                nets = tf.nn.relu6(nets, name=flower_point[2]+'/relu6')
                endpoints[flower_point[2]] = nets
                # nets = (?, 4, 5, 256)
                nets = slim.conv2d(nets, 128, (1, 1), scope=flower_point[3])
                nets = tf.nn.relu6(nets, name=flower_point[3]+'/relu6')
                endpoints[flower_point[3]] = nets
                # nets = (?, 4, 5, 128)
                nets = slim.conv2d(nets, 5, (3, 3), normalizer_fn=None, activation_fn=None, scope=flower_point[4])
                endpoints[flower_point[4]] = nets
                # nets = (?, 4, 5, 5)
                # tf.contrib.layers.softmax(nets)
    return nets, endpoints


def mobilenet_conv(images: tf.Tensor, num_classes: int, depth_multiplier: float, is_training: bool):
    flower_point = ['Conv2d_0_depthwise', 'Conv2d_0_pointwise', 'Conv2d_1', 'Final']
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
        nets, endpoints = mobilenet_v1_base(images, depth_multiplier=depth_multiplier)

    # add the new layer
    with tf.variable_scope('Flowers'):
        with slim.arg_scope([slim.batch_norm], is_training=is_training,  center=True, scale=True, decay=0.9997, epsilon=0.001):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME', normalizer_fn=slim.batch_norm, activation_fn=None,):
                if depth_multiplier == 1.0 or depth_multiplier == 0.75:
                    # nets= [?,8,10,1024]
                    nets = slim.conv2d(nets, 512, (3, 3), padding='SAME')
                    nets = slim.batch_norm(nets)
                else:
                    pass
                # (?, 8, 10, 512) ===> (?, 4, 5, 512)
                nets = slim.separable_conv2d(nets, None, (3, 3), stride=(2, 2), scope=flower_point[0])
                nets = tf.nn.relu6(nets, name=flower_point[0]+'/relu6')
                endpoints[flower_point[0]] = nets
                # ! (?, 4, 5, 512)===>(?, 4, 5, 128) 难道是稀疏卷积之后不能加(3,3)卷积?
                nets = slim.conv2d(nets, 256, (1, 1), scope=flower_point[1])
                # nets = slim.conv2d(nets, 256, (3, 3), scope=flower_point[1])
                
                nets = tf.nn.relu6(nets, name=flower_point[1]+'/relu6')
                endpoints[flower_point[1]] = nets
                # nets = (?, 4, 5, 128)
                nets = slim.conv2d(nets, 128, (3, 3),  scope=flower_point[2])
                nets = tf.nn.relu6(nets, name=flower_point[2]+'/relu6')
                endpoints[flower_point[2]] = nets
                # nets = (?, 4, 5, 64)
                nets = slim.conv2d(nets, 5, (3, 3), normalizer_fn=None, activation_fn=None, scope=flower_point[3])
                endpoints[flower_point[3]] = nets
                # nets = (?, 4, 5, 5)
                # tf.contrib.layers.softmax(nets)
    return nets, endpoints
