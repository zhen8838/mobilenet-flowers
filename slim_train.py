import tensorflow as tf
import shutil
import os
from nets.mobilenet_v1 import *
from utils import *
from tensorflow.contrib import slim
from datasets import flowers
from preprocessing import inception_preprocessing

LOG_DIR = 'flower_graph'
CKPT_PATH = 'mobilenet_v1_0.5_224/mobilenet_v1_0.5_224.ckpt'
NEW_CKPT_NAME = 'flower.ckpt'
DATA_DIR = '/media/zqh/Datas/DataSet/flowers'


def load_batch(dataset, batch_size=32, height=224, width=224, is_training=False):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])

    # Preprocess image for usage by Inception. 压缩图像到对应大小预处理
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)

    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
        [image, image_raw, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)

    return images, images_raw, labels


def new_mobilenet(images: tf.Tensor, num_classes: int, is_training: bool):
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
        nets, endpoints = mobilenet_v1_base(images, depth_multiplier=0.5)
    # the nets shape is [?,7,7,512]

    # add the new layer
    with tf.variable_scope('Flowers'):
        nets = slim.conv2d(nets, 125, (1, 1), activation_fn=tf.nn.relu6, weights_initializer=slim.initializers.xavier_initializer_conv2d())
        nets = slim.flatten(nets)
        nets = slim.fully_connected(nets, num_classes, activation_fn=None, weights_initializer=slim.initializers.xavier_initializer())
        logits = tf.contrib.layers.softmax(nets)
    return logits


if __name__ == "__main__":
    """ This file will be define the new mobilenet+yolo network! """
    tf.reset_default_graph()
    # ! data load 不需要占位符了
    dataset = flowers.get_split('train', DATA_DIR)
    images, _, labels = load_batch(dataset)

    # ! define the network
    probabilities = new_mobilenet(images, flowers._NUM_CLASSES, is_training=True)

    # ! define loss
    slim.losses.sigmoid_cross_entropy(probabilities, slim.one_hot_encoding(labels, flowers._NUM_CLASSES))
    total_loss = tf.losses.get_total_loss()

    clean_dir(LOG_DIR)

    # Create some summaries to visualize the training process:
    tf.summary.scalar('losses/Total Loss', total_loss)

    # Specify the optimizer and create the train op:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # Run the training:
    final_loss = slim.learning.train(
        train_op,
        logdir=LOG_DIR,
        number_of_steps=500,  # For speed, we just do 1 epoch
        save_summaries_secs=30)

    print('Finished training. Final batch loss %d' % final_loss)
