'''
Author: San Wong

Construct a Forward AlexNet to handle DVS images
Taking a working code: DirtyAlex_Net -> myalexnet_forward.py for reference.

'''

from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


import tensorflow as tf


from caffe_classes_DVS import class_names

# Load up Normalized Data set
norm_train_data = np.load('norm_train_data_labeled.npy')
norm_test_data = np.load('norm_test_data_labeled.npy')

# According to this:
# https://github.com/hpssjellis/easy-tensorflow-on-cloud9/blob/master/aymericdamien-Examples/examples/alexnet.py
# Need to sperate train_data_x and train_data_y
train_data_x = norm_train_data[:,]



#Shuffle
np.random.shuffle(norm_train_data)
np.random.shuffle(norm_test_data)

# tf.zeros: (shape, dtype=tf.float32,name=None)
# tf.zeros([3, 4], tf.int32) ==> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


# Each training image is 100 by 100 pixels
numOfTrainImg = norm_train_data[0]
numOfTestImg = norm_test_data[0]
img_size = 100
img_channel = 1
numOfClass = 1002


# According to myalexnet_forward
train_x = zeros((numOfTrainImg,img_size,img_size,img_channel)).astype(float32)
train_y = zeros((numOfTrainImg, numOfClass)) # There were 1000 class, but we have added 2 more classes
xdim = train_x.shape[1:] # Return the dimension of each input image (100,100,1): 100 by 100 by 1 (greyscale)
ydim = train_y.shape[1]

test_x = zeros((numOfTestImg,img_size,img_size,img_channel)).astype(float32)
test_y = zeros((numOfTestImg, numOfClass))

#Pre-train matrix weight
net_data = load("bvlc_alexnet.npy").item()


# define a convolution layer
def conv(input,kernel,k_h,k_w,c_o,s_h,s_w,padding="VALID",group=1):
	c_i = input.get_shape()[-1] # Get the last argument of the size/dimension
	# Check condition (Number of channel). Trigger an error if the condition is false
	assert c_i%group == 0 # Divides left hand operand by Right hand operand and returns remainder
	assert c_o%group == 0

	# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None,data_format=None,name=None)
	# input tensor shape: [batch,in_height,in_width,in_channels] filter/kernel tensor of shape: [filter_height, filter_width, in_channels, out_channels]
	# conv2d does the following work
	# 1. Flattens the filter to a 2D matrix with shape: 
	# [filter_height*filter_width*in_channels, out_channels]
	# 2. Extracts image patches from the input tensor to form a Virtual tensor of shape:
	# [batch, out_height, out_width, filter_height*filter_width*in_channels]
	convolve = lambda i, k: tf.nn.conv2d(i,k,[1,s_h,s_w,1],padding=padding)

	if group == 1:
		conv = convolve(input, kernal)
	else:
		"""
		For tensorflow < 0.12
		x = tf.split(0,n_steps,x) which is tf.splite(axis, num_or_size_splits, value)
		For tensorflow > 0.12
		x = tf.split(x, n_steps, 0) which is tf.splite(value, num_or_size_splits, axis)

		tf.concat: concat(values, axis, name = 'concat')

		"""


		input_groups = tf.split(input, group, 3)
		kernal_groups = tf.split(kernal, group, 3)

		output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
		conv = tf.concat(output_groups,3)
	return tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


# tf graph input
x = tf.placeholder(tf.float32, (None,) + xdim) # <tf.Tensor 'Placeholder:0' shape=(?, 100, 100, 1) dtype=float32>
y = tf.placeholder(tf.float32, [None, numOfClass]) # I define this

'''
# According to this:
# https://github.com/hpssjellis/easy-tensorflow-on-cloud9/blob/master/aymericdamien-Examples/examples/alexnet.py
x1 = tf.placeholder(tf.float32, [None, img_size*img_size]) # <tf.Tensor 'Placeholder_1:0' shape=(?, 10000) dtype=float32>
y1 = tf.placeholder(tf.float32, [None, numOfClass])
# keep_prob = tf.placeholder(tf.types.float32) # We havne't been using DROPOUT yet
'''

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20


'''
# =============================== NETWORK ===================================

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(net_data["fc8"][0])
fc8b = tf.Variable(net_data["fc8"][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)

'''

# According to : https://github.com/hpssjellis/easy-tensorflow-on-cloud9/blob/master/aymericdamien-Examples/examples/alexnet.py
# Create an Alex_net function
def alex_net(x,net_data):
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #fc6
    #fc(4096, name='fc6')
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    #fc7
    #fc(4096, name='fc7')
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    #fc8
    #fc(1000, relu=False, name='fc8')
    fc8W = tf.Variable(net_data["fc8"][0])
    fc8b = tf.Variable(net_data["fc8"][1])
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

    #prob
    #softmax(name='prob'))
    prob = tf.nn.softmax(fc8)

    # Shoud I return the Prob (i.e: Value after fc8 pass through Softmax layer)
    # Or should I return the raw data (before fc8 is passed into Softmax layer)
    return prob


# I write this next_batch function 
def next_batch(batch_size, x, y, curr_index):
	batch_xs = x[curr_index:curr_index+batch_size]
	batch_ys = y[curr_index:curr_index+batch_size]
	return batch_xs,batch_ys




# Store Layers Weight and Bias
# We have a set of pre-trained weight. So I guess I am not suppose to define random_normal weight and basis




'''
# According to "myalexnet_forward.py" That's how to initialize the session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
'''


# According to : https://github.com/hpssjellis/easy-tensorflow-on-cloud9/blob/master/aymericdamien-Examples/examples/alexnet.py
# Construct model (Pass in x:placeholder and net_data: weight and basis)
pred = alex_net(x,net_data)

# Define Loss and Optimizer
# tf.nn.softmax_cross_entropy_with_logits: Measures the probability error in discrete classification tasks in which the classes are mutually exclusive (each entry is in exactly one class)
# logit: function that maps probability ([0,1]) to R([-inf, inf])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.types.float32))

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
# This is where you define all the input (i.e: x,y)
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
    	step_mod = step-1
    	curr_index = step_mod * batch_size
        batch_xs, batch_ys = next_batch(batch_size,x,y,curr_index)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"
    # Calculate accuracy for 256 mnist test images
    #print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256]})






