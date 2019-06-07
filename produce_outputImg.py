#!data/anaconda510/bin/python
import numpy as np
import cv2
import tensorflow as tf
import os
import os

import tensorflow as tf

import GetData

DATA_NAME = 'Data'
TRAIN_SOURCE = "Train"
TEST_SOURCE = 'Test'
RUN_NAME = "SELU_Run03"
OUTPUT_NAME = 'Output'
CHECKPOINT_FN = 'model.ckpt'

WORKING_DIR = os.getcwd()

TRAIN_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TRAIN_SOURCE)
TEST_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TEST_SOURCE)

ROOT_LOG_DIR = os.path.join(WORKING_DIR, OUTPUT_NAME)
LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)
tf.reset_default_graph()

v1 = tf.Variable(tf.constant(0.1, shape = [2]), name="v1")
v2 = tf.Variable(tf.constant(0.2, shape = [2]), name="v2")

test_data = tfmodel.GetData(TEST_DATA_DIR)
images_batch, labels_batch = test_data.next_batch(1)
feed_dict = {images: images_batch, labels: labels_batch}
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, CHECKPOINT_FL)
    print("Model restored.")
    result = sess.run( [softmax_logits], feed_dict=feed_dict)
    result = 
