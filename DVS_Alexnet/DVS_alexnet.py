'''
Author: San Wong

Construct an AlexNet to handle DVS images
'''

import tensorflow as tf
import numpy as np


class AlexNet(object):

	def __init__(self,x,keep_prob,num_classes,skip_layer,weights_path='DEFAULT'):

		# Parse input arguement into class variables
		self.X = x
		self.NUM_CLASSES = num_classes
		self.KEEP_PROB = keep_prob
		self.SKIP_LAYER = skip_layer

		if weights_path == 'DEFAULT':
			self.WEIGHTS_PATH = 'bvlc.alexnet.npy'
		else:
			self.WEIGHTS_PATH = weights_path

		# Call the function and create a computational graph of AlexNet
		self.create()


	def create(self):
		# 1st layer [ Conv(ReLU) -> Pool -> Lrn ]
		conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
		pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
		norm1 = lrn(pool1, 2, 2e-05, 0,75, name='norm1')


		# 2nd layer [ Repeat of 1st layer]
		conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
		pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
		norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

		# 3rd layer: Conv(ReLU)
		conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

		# 4th layer: Conv(ReLU). Split into 2 groups
		conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')

		# 5th layer: Conv(ReLU) -> Pool. Split into 2 groups
		conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
		pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

		# 6th layer: Flatten -> Fully Connected (ReLU) -> Dropout
		flattened = tf.reshape(pool5,[-1, 6*6*256]) # reshape([A,B]): A is num of col, where B is num of roll
		# If reshape([-1]) => means flatten. Therefore, [-1, 6*6*256] means flatten the array into each array length of 6*6*256
		fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
		dropout6 = dropout(fc6, self.KEEP_PROB)

		# 7th layer: Fully Connected (ReLU) -> Dropout
		fc7 = fc(dropout6, 4096, 4096, name='fc7')
		dropout7 = dropout(fc7, self.KEEP_PROB)

		# 8th layer
		self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')


	def load_initial_weights(self,session):
		'''
		According to the tutorial. Weight can be downloaded from CS toronto site. Weights comes as a dict of lists.
		eg. weights['conv1'] is a list
		'''

		# Load weights
		weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

		# Loop over all layer names stored in the weights dict
		for op_name in weights_dict:
			# Chech if the layer need to be re-initialized
			if op_name not in self.SKIP_LAYER:
				with tf.variable_scope(op_name, reuse = True):
					#Loop over list of weights and assign them to their corresponding
					# tf variable
					for data in weights_dict[op_name]:
						#Biases
						if len(data.shape) == 1:
							var = tf.get_variable('biases',trainable = False)
							session.run(var.assign(data))

						#Weights
					    else:
					    	var = tf.get_variable('weights',trainable = False)
					    	session.run(var.assign(data))



'''
Define Layers for the network

	# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None,data_format=None,name=None)
	# input tensor shape: [batch,in_height,in_width,in_channels] filter/kernel tensor of shape: [filter_height, filter_width, in_channels, out_channels]
	# conv2d does the following work
	# 1. Flattens the filter to a 2D matrix with shape: 
	# [filter_height*filter_width*in_channels, out_channels]
	# 2. Extracts image patches from the input tensor to form a Virtual tensor of shape:
	# [batch, out_height, out_width, filter_height*filter_width*in_channels]


	strides: a list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input


'''
def conv(x,filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
	# Get number of input channel
	input_channels = int(x.get_shape()[-1])

	convolve = lambda i, k: tf.nn.conv2d(i,k,strides=[1,stride_y,stride_x,1],padding=padding)

	with tf.variable_scope(name) as scope:
		# Create tf variables for weights and biases of the conv layer
		weights = tf.get_variable('weights',shape=[filter_height,filter_width,input_channels/groups,num_filters])
		biases = tf.get_variable('biases', shape=[num_filters])

		if groups == 1:
			conv = convolve(x,weights)

		else:
			# Split input and weights
			# tf.split(value,num_or_size_splits,axis=0,num=None,name='split')
			input_groups = tf.split(input,num_or_size_splits=groups,axis=3)
			weights_groups = tf.split(weights,num_or_size_splits=groups, axis=3)
			output_groups = [convolve(i,k) for i,k in zip(input_groups,weights_groups)]

			# Concat the convolved output together tf.concat(value,axis,name)
			conv = tf.concat(values = output_groups, axis = 3)

		#Add biase
		bias = tf.reshape(tf.nn.bias_add(conv,biases),conv.get_shape().as_list())

		# Apply relu
		relu = tf.nn.relu(bias,name=scope.name)

		return relu



def fc(x, num_in, num_out, name, relu=True):
	#Variable scope allows to create new variables and to share already created ones while providing checks to not create or share by accident.
	with tf.variable_scope(name) as scope:

		#Create tf variables for weights and biases
		weights = tf.get_variable('weights',shape=[num_in,num_out],trainable=True)
		biases = tf.get_variable('biases',[num_out],trainable=True)

		# Matrix multiply weight and input and add bias
		# x is usually a 2D tensor. Dimension typically: batch, in_units
		# weights is usually a 2D tensor. Dimension typically: in_units, out_units
		# biases: 1D tensor. Dimension: out_units
		act = tf.nn.xw_plus_b(x,weights,biases,name=scope.name)

		if relu ==True:
			relu = tf.nn.relu(act)
			return relu
		else:
			return act

'''
max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
)


value: A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
padding: A string, either 'VALID' or 'SAME'. The padding algorithm. See the comment here
data_format: A string. 'NHWC' and 'NCHW' are supported.
name: Optional name for the operation.

Return: A Tensor with type tf.float32. The max pooled output tensor.
'''


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
	return tf.nn.maxpool(x, ksize=[1,filter_height,filter_width,1],strides=[1,stride_y,stride_x,1],padding=padding,name=name)


'''
local_response_normalization(
    input,
    depth_radius=None,
    bias=None,
    alpha=None,
    beta=None,
    name=None
)
'''

def lrn(x, radius, alpha, beta, name, bias=1.0):
	return tf.nn.local_response_normalization(x,depth_radius=radius,alpha=alpha,beta=beta,name=name)


def dropout(x, keep_prob):
	return tf.nn.dropout(x,keep_prob)




