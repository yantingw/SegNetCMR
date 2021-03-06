import numpy as np

import tensorflow as tf

from tensorflow.python.framework import ops

from .layers import unpool_with_argmax

def selu(x, name='selu'):
    with ops.name_scope(name) as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def c2rb(net, filters, kernel_size, activation=True, scope=None):

    with tf.variable_scope(scope):
       # print(f"before :{scope}")
        #print("the net shape is " )
        #print(net.shape.as_list())

        kernal_units = kernel_size[0] * kernel_size[1] * net.shape.as_list()[-1]
        #print(kernal_units)
        net = tf.layers.conv2d(net, filters, kernel_size,
                               padding='same',
                               activation=None,
                               use_bias=True,
                               bias_initializer=tf.zeros_initializer(),
                               kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=np.sqrt(1/kernal_units)),
                               name='conv')

        if activation:
        #    print("it is activation")
            net = selu(net, name='selu')
            
       # print("after the net shape is " )
       # print(net.shape.as_list())
       # print("\n")
        return net


def inference(images, class_inc_bg = None):

    images = (2.0/255.0) * images - 1.0
#[6, 256, 256,1]
    with tf.variable_scope('pool1'):
        net = c2rb(images, 64, [3, 3], scope='conv1_1')
       # [6, 256, 256, 64]
        net = c2rb(net, 64, [3, 3],  scope='conv1_2')
        net, arg1 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')
#[6, 128, 128, 64]
    with tf.variable_scope('pool2'):
        net = c2rb(net, 128, [3, 3], scope='conv2_1')
        net = c2rb(net, 128, [3, 3], scope='conv2_2')
        net, arg2 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool2')
#[6, 128, 128, 128]
    with tf.variable_scope('pool3'):
        net = c2rb(net, 256, [3, 3], scope='conv3_1')
        net = c2rb(net, 256, [3, 3], scope='conv3_2')
        net = c2rb(net, 256, [3, 3], scope='conv3_3')
        net, arg3 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool3')
#[6,64,64, 256]
    with tf.variable_scope('pool4'):
        net = c2rb(net, 512, [3, 3], scope='conv4_1')
        net = c2rb(net, 512, [3, 3], scope='conv4_2')
        net = c2rb(net, 512, [3, 3], scope='conv4_3')
        net, arg4 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')
#[6, 16, 16, 512]
    with tf.variable_scope('pool5'):
        net = c2rb(net, 512, [3, 3], scope='conv5_1')
        net = c2rb(net, 512, [3, 3], scope='conv5_2')
        net = c2rb(net, 512, [3, 3], scope='conv5_3')
        net, arg5 = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool5')

    with tf.variable_scope('unpool5'):
        net = unpool_with_argmax(net, arg5, name='maxunpool5')
        net = c2rb(net, 512, [3, 3], scope='uconv5_3')
        net = c2rb(net, 512, [3, 3], scope='uconv5_2')
        net = c2rb(net, 512, [3, 3], scope='uconv5_1')

        net = unpool_with_argmax(net, arg4, name='maxunpool4')
        net = c2rb(net, 512, [3, 3], scope='uconv4_3')
        net = c2rb(net, 512, [3, 3], scope='uconv4_2')
        net = c2rb(net, 256, [3, 3], scope='uconv4_1')
#[6, 32, 32, 256]
    with tf.variable_scope('unpool3'):
        net = unpool_with_argmax(net, arg3, name='maxunpool3')
        net = c2rb(net, 256, [3, 3], scope='uconv3_3')
        net = c2rb(net, 256, [3, 3], scope='uconv3_2')
        net = c2rb(net, 128, [3, 3], scope='uconv3_1')
#6, 64, 64, 256
    with tf.variable_scope('unpool2'):
        net = unpool_with_argmax(net, arg2, name='maxunpool2')
        net = c2rb(net, 128, [3, 3], scope='uconv2_2')
        net = c2rb(net, 64, [3, 3], scope='uconv2_1')
#6, 128, 128, 64
    with tf.variable_scope('unpool1'):
        net = unpool_with_argmax(net, arg1, name='maxunpool1')
        net = c2rb(net, 64, [3, 3], scope='uconv1_2')
#6, 256, 256, 64
    with tf.variable_scope('output'):
        #class_inc_bg is "2" (0 & 1)
        logits = c2rb(net, class_inc_bg, [3, 3], activation=False, scope='logits')
        softmax_logits = tf.nn.softmax(logits=logits, dim=3, name='softmax_logits') #at dimension3 doing softmax 
        predict_img =np.zeros((softmax_logits.shape.as_list()[0],softmax_logits.shape.as_list()[1],softmax_logits.shape.as_list()[2])) #
        """
        for idx in range(softmax_logits.shape.as_list()[0]):
            for col in range(softmax_logits.shape.as_list()[1]):
                for row in range(softmax_logits.shape.as_list()[2]):
                   if softmax_logits[idx,col,row,0]>softmax_logits[idx,col,row,1] :
                       predict_img [idx,col,row] = 0
                   else:
                        predict_img [idx,col,row] = 1
        """
#6, 256, 256, 2
        #print(f"the soend :{predict_img}")
    return logits, softmax_logits