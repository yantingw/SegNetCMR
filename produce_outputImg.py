#!/data/anaconda510/bin/python
import numpy as np
import tensorflow as tf
import os
import os

import tensorflow as tf

import tfmodel

# Dice Coefficient to work outside Tensorflow


def dice_coef_2(y_true, y_pred):

    smooth = 1
    side = len(y_true[0])

    y_true_f = y_true.reshape(side*side)

    y_pred_f = y_pred.reshape(side*side)

    intersection = sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)





DATA_NAME = 'Data'
TRAIN_SOURCE = "Train"
TEST_SOURCE = 'Test'
RUN_NAME = "SELU_Run03"
OUTPUT_NAME = 'Output'
#CHECKPOINT_FN = 'model.ckpt'
output_img_data = 'Img'


WORKING_DIR = os.getcwd()

TEST_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TEST_SOURCE)

ROOT_LOG_DIR = os.path.join(WORKING_DIR, OUTPUT_NAME)
LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
image_dir = os.path.join(LOG_DIR, output_img_data)
print(image_dir)
#CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

tf.reset_default_graph()

test_data = tfmodel.GetData(TEST_DATA_DIR)
#label_data = tfmodel.GetData(TEST_DATA_DIR)
model_path = os.path.join(LOG_DIR,'model.ckpt-2500.meta')

with tf.Session() as sess:
    """
    saver = tf.train.Saver()
    module_file = tf.train.latest_checkpoint(LOG_DIR)    
    saver.restore(sess, module_file)
    """
    new_saver = tf.train.import_meta_graph(model_path)
    new_saver.restore(sess,os.path.join(LOG_DIR, 'model.ckpt-2500'))
    #new_saver.restore(sess,tf.train.latest_checkpoint(LOG_DIR))
    print("Model restored.")
    num = 0
    dic_record = list
    graph = tf.get_default_graph()
    print([tensor.name for tensor in tf.get_default_graph().as_graph_def().node])

    softmax_logits = graph.get_tensor_by_name("output:0")
    """   
    softmax_logits = graph.get_tensor_by_name("softmax_logits:0")
    logits= graph.get_tensor_by_name("logits:0")   
    images =graph.get_tensor_by_name("images:0")
    labels =graph.get_tensor_by_name("labels:0")
    """


    
    while True:
        images_batch, labels_batch = test_data.no_shuffle_next_batch(1)
        feed_dict = {images: images_batch, labels: labels_batch}
        result_soft,result_logits = sess.run( [softmax_logits, logits] , feed_dict=feed_dict)
        
        
        result_soft = np.array(result_soft)
        result_logits = np.array(result_logits)
        predict_img = result_logits
        dic_record.append(dice_coef_2(labels_batch,predict_img))


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
    np.save(dic_record,'dic_table')                  
