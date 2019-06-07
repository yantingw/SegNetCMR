import tensorflow as tf

def placeholder_inputs(batch_size):

    images = tf.placeholder(tf.float32, [batch_size, 256, 256, 3])
    labels = tf.placeholder(tf.int64, [batch_size, 256, 256, 3])

    return images, labels