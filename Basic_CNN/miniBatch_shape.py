'''
Date: Apr 5 ,2017
Author:Hong San Wong

The following code is to understand the arguement shape
during mini batch training

'''

from __future__ import print_function

import tensorflow as tf


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

print(x.shape)
print(y.shape)
'''


batch_x, batch_y = mnist.train.next_batch(50)
print(batch_x.shape)
print(batch_y.shape)