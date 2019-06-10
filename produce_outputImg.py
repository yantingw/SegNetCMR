#!data/anaconda510/bin/python
import numpy as np
import cv2
import tensorflow as tf
import os
import os

import tensorflow as tf

import tfmodel.GetData

DATA_NAME = 'Data'
TRAIN_SOURCE = "Train"
TEST_SOURCE = 'Test'
RUN_NAME = "SELU_Run03"
OUTPUT_NAME = 'Output'
#CHECKPOINT_FN = 'model.ckpt'
output_img_data = 'Img'


WORKING_DIR = os.getcwd()

TRAIN_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TRAIN_SOURCE)
TEST_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TEST_SOURCE)

ROOT_LOG_DIR = os.path.join(WORKING_DIR, OUTPUT_NAME)
LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
image_dir = os.path.join(LOG_DIR, output_img_data)
print(image_dir)
#CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

tf.reset_default_graph()

v1 = tf.Variable(tf.constant(0.1, shape = [2]), name="v1")
v2 = tf.Variable(tf.constant(0.2, shape = [2]), name="v2")

test_data = tfmodel.GetData(TEST_DATA_DIR)
label_data = tfmodel.GetData(TEST_DATA_DIR)

saver = tf.train.Saver()

with tf.Session() as sess:
    model_file=tf.train.latest_checkpoint('ckpt/')
    saver.restore(sess,model_file)
    print("Model restored.")
    num = 0
    while　True :
        images_batch, labels_batch = test_data.next_batch(1)
        feed_dict = {images: images_batch, labels: labels_batch}
        result_soft,result_logits = sess.run( [softmax_logits,lologits] , feed_dict=feed_dict)
        result_soft = np.array(result_soft)
        result_logits = np.array(result_logits)
        np.save(result_logit,"img")
        num+=1
        print(f"get pic {num}")
        if(num==279):
            break;
                      
