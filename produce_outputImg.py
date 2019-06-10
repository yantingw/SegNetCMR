#!data/anaconda510/bin/python
import numpy as np
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
model_path = os.path.join(LOG_DIR,'model.ckpt-2500.meta')

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(model_path)
    model_file= tf.train.latest_checkpoint('model.ckpt')
    new_saver.restore(sess,model_file)
    print("Model restored.")
    num = 0
    while True:
        images_batch, labels_batch = test_data.next_batch(1)
        feed_dict = {images: images_batch, labels: labels_batch}
        result_soft,result_logits = sess.run( [softmax_logits,lologits] , feed_dict=feed_dict)
        result_soft = np.array(result_soft)
        result_logits = np.array(result_logits)
        predict_img = result_logits
        for idx in range(result_soft.shape[0]):
            for col in range(result_soft.shape[1]):
                for row in range(result_soft.shape[2]):
                   if result_soft[idx,col,row,0]>result_soft[idx,col,row,1] :
                       predict_img [idx,col,row] = 0
                   else:
                        predict_img [idx,col,row] = 1

        num+=1
        np.save(predict_img,f"img{num}")
        print(f"get pic {num}")
        if(num==279):
            break
                      
