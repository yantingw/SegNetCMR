import os

import tensorflow as tf
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import tfmodel

DATA_NAME = 'Data'
TRAIN_SOURCE = "Train"
TEST_SOURCE = 'Test'
RUN_NAME = "SELU_Run_step16500_batch6"
OUTPUT_NAME = 'Output'
CHECKPOINT_FN = 'model.ckpt'

WORKING_DIR = os.getcwd()

TRAIN_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TRAIN_SOURCE)
TEST_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TEST_SOURCE)

ROOT_LOG_DIR = os.path.join(WORKING_DIR, OUTPUT_NAME)
LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

TRAIN_WRITER_DIR = os.path.join(LOG_DIR, TRAIN_SOURCE)
TEST_WRITER_DIR = os.path.join(LOG_DIR, TEST_SOURCE)

NUM_EPOCHS = 10
MAX_STEP = 16500
BATCH_SIZE = 1

LEARNING_RATE = 1e-04

SAVE_RESULTS_INTERVAL = 5
SAVE_CHECKPOINT_INTERVAL = 100


def main():

    test_data = tfmodel.GetData(TEST_DATA_DIR)
    #print("test_data:", test_data)

    g = tf.Graph()

    with g.as_default():

        images, labels = tfmodel.placeholder_inputs(batch_size=BATCH_SIZE)

        logits, softmax_logits = tfmodel.inference(images, class_inc_bg=2)

        tfmodel.add_output_images(images=images, logits=logits, labels=labels)

        loss = tfmodel.loss_calc(logits=logits, labels=labels)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_op = tfmodel.training(loss=loss, learning_rate=1e-04, global_step=global_step)

        accuracy = tfmodel.evaluation(logits=logits, labels=labels)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.global_variables())

    sm = tf.train.SessionManager(graph=g)

    with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR) as sess:

        sess.run(tf.local_variables_initializer())

        try:

            count = 0
            while True:

                if (count > 279):
                    break

                else:
                    #images_batch, labels_batch = test_data.next_batch(BATCH_SIZE)

                    images_batch, labels_batch, fname = test_data.get_one_image()
                    feed_dict = {images: images_batch[None, ...], labels: labels_batch[None, ...]}
                    print('image_data1:', images_batch[None, ...].shape)

                    image_data = sess.run(logits, feed_dict=feed_dict)
                    print('image_data2:', image_data.shape)

                    image = np.argmax(image_data, axis=-1)[0, ...]
                    #image = np.reshape(image)
                    print('image:', image.shape)

                    plt.imsave(os.path.join((r'C:\Users\Yun\Documents\GitHub\SegNetCMR\result_images\SELU_Run_step16500_batch6'), fname), image, cmap='gray')
                    count = count + 1


        except Exception as e:
            print('Exception')
            print(e)

        print("Stopping")


if __name__ == '__main__':
    main()
