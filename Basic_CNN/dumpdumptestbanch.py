from __future__ import print_function

import tensorflow as tf
import numpy as np
import math



n_classes = 2


# Create data object =============================================
class data_object:

    # Constructor
    def __init__(self,img,label):
        self.img = img
        self.label = label # 1 for Hand and 0 for Others

# End of data_object class =======================================


trainImg_arr = np.load('trainImg_arr.npy')

def shuffle_data(data_array):
    np.random.shuffle(data_array)
    return data_array



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



def next_batch(data_array,batch_size,start_index):
    temp_arr=[]
    end_index = start_index+batch_size
    temp_arr = data_array[start_index:end_index]
    return temp_arr



# Batch norm wrapper
def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    inputs = tf.convert_to_tensor(inputs)

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               np.int(pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              np.int(pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)









# ================ TEST BANCH =====================

shuffle_data(trainImg_arr)
train_img_matrix, train_label_matrix = convert_data(trainImg_arr)

start_index = 0
batch_size = 50
is_training = True

batch_x = next_batch(train_img_matrix,batch_size,start_index)
batch_y = next_batch(train_label_matrix,batch_size,start_index)


batch_x_first = batch_x[0]
print(len(batch_x_first))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    test_input = batch_x_first

    res = []

    res = batch_norm_wrapper(test_input, is_training = True, decay = 0.999)

    print(len(res))



