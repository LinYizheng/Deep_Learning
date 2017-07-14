'''
Remake Basic CNN with batch norm
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import math


# //////////////////////////////  UTILITIES //////////////////////

# Create data object =============================================
class data_object:

    # Constructor
    def __init__(self,img,label):
        self.img = img
        self.label = label # 1 for Hand and 0 for Others

# End of data_object class =======================================



# /////////////////////////// DATA ////////////////////////////////

'''
Function call flow:
Import data => data_object_arr => shuffer => convertion => next_batch
'''



# Import data
    #from tensorflow.examples.tutorials.mnist import input_data
    #mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
testImg_arr = np.load('testImg_arr.npy')
trainImg_arr = np.load('trainImg_arr.npy')



# Shuffle loaded data arr =======================================
#np.random.shuffle(testImg_arr)
#np.random.shuffle(trainImg_arr)

def shuffle_data(data_array):
    np.random.shuffle(data_array)
    return data_array

# ===============================================================

# ====================== Data Convertion ========================
def convert_data(data_array):
    arr_shape = data_array.shape
    arr_W = arr_shape[0] # which in our case, we are expecting 1554
    img_arr_W = data_array[0].img.shape[0] # which in our case, we are epxecting 10000
    # Define img matrix
    img_matrix = np.zeros((arr_W,img_arr_W),dtype=np.int)
    # Define label matrix
    label_matrix = np.zeros((arr_W,n_classes),dtype=np.int)
    # put img_arr into the matrix
    for index in range(arr_W):
        img_matrix[index,:] = data_array[index].img.transpose()
        curr_label = data_array[index].label
        if curr_label == 1:
            label_matrix[index,1] = 1
        else:
            label_matrix[index,0] = 1

    return img_matrix, label_matrix
# ================================================================

# Extract Next Bactch ============================================
def next_batch(data_array,batch_size,start_index):
    temp_arr=[]
    end_index = start_index+batch_size
    temp_arr = data_array[start_index:end_index]
    return temp_arr
# End of Next Batch ==============================================


# ===================== Parameters and Placeholer =================
learning_rate = 0.001
training_iters = len(trainImg_arr)
batch_size = 50
display_step = 1
n_input = 10000
n_classes = 2
dropout = 0.75

'''
# Placeholder to hold the ground truth
x = tf.placeholder(tf.float32, [None, n_input])
y_ = tf.placeholder(tf.float32, [None, n_classes])
'''


keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


weights = {

    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# ==================================================================




# ================== BUILDING BLOCKS FOR GRAPH =======================

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # Test ZERO PADDING
    # x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID') padding='SAME' => [2,4] i.e: 4 padding on X and Y dimension
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    
    x = tf.nn.bias_add(x, b)

    # Update: April 11, 17
    # Try to add batch nor. so we don't perfer RELU here "tf.nn.relu(x)"
    return x

# Define a maxpool layer
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Batch norm wrapper
def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

# ======================================================================


# ================================ GRAPH ===============================
# y is the ground truth while y_ is the predicted
# Update: Just wanna try build_graph(weights, biases, dropout, is_training) instead of build_graph(x, weights, biases, dropout, is_training):
# Define placeholder within the build_graph instead of outside the function because for that we can avoid passing through 

def build_graph(weights, biases, dropout, is_training):

	# Trail Update (might take it off)
	# Move x, y_ placeholder here.
	x = tf.placeholder(tf.float32, [None, n_input])
	y_ = tf.placeholder(tf.float32, [None, n_classes])



	x = tf.reshape(x, shape=[-1,100,100,1])

	# Layer 1 (Conv layer)
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	bn1 = batch_norm_wrapper(conv1, is_training)
	l1 = tf.nn.relu(bn1)
	pool1_out = maxpool2d(l1, k=2)



	# Layer 2
	conv2 = conv2d(pool1_out, weights['wc2'], biases['bc2'])
	bn2 = batch_norm_wrapper(conv2, is_training)
	l2 = tf.nn.relu(bn2)
	pool2_out = maxpool2d(l2, k=2)

	# Fully Connected Layer
	fc1 = tf.reshape(pool2_out, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	# Apply Dropout
	fc1 = tf.nn.dropout(fc1, dropout)
	y = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

	# Loss, Optimizer and Prediction
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	return (x,y_), train_step, accuracy, y, tf.train.Saver()



# ======================== TRAIN THE NETWORK =============================

# Shuffle Training Data
shuffle_data(trainImg_arr)
train_img_matrix, train_label_matrix = convert_data(trainImg_arr)

# Shuffle Testing Data
shuffle_data(testImg_arr)
test_img_matrix, test_label_matrix = convert_data(testImg_arr)

# Reset the graph
#tf.reset_default_graph()
#(x, y_), train_step, accuracy, _, saver = build_graph(weights, biases, dropout, is_training=True)

init = tf.global_variables_initializer()


# Store the acc value
acc = []

with tf.Session() as sess:
	sess.run(init)
	step = 0

	while step*batch_size < training_iters:
		print(step)
		start_index = step*batch_size

		# Ground Truth Training sample
		batch_x = next_batch(train_img_matrix,batch_size,start_index)
        batch_y = next_batch(train_label_matrix,batch_size,start_index)
        train_step.run(feed_dict={x:batch_x, y_:batch_y})

        if step % display_step == 0:
        	res = sess.run([accuracy], feed_dict={x:test_img_matrix, y_:test_label_matrix})
        	acc.append(res[0])
      	

      	step += 1


     #saved_model = saver.save(sess, './temp-bn-save')



print("Final Accuracy:" ,acc[-1])



