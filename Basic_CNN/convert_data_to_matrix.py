'''
Author: Hong San Wong
Date: Apr 5, 2017


As DATA_OBJECT format may not be the best choice of data storage
The following code is to test converting DATA_OBJECT to a matrix format

Here's the blueprint
Currently, our trainImg_arr is in shape of (1554,)
which means there are 1554 data object in there. Each object contain a 1000 by 1 array (which is the pixel array of an imgae)
And also a label

TO DO:
We need to extract these and form a matrix to fit the use of Minibatch
We need to decide if we are perform the data convertion before we call next_batch function
OR after we call the next_batch function

Will shuffer affect this? Maybe not. We can first shuffer the whole data_object array before convertion

So the function call flow can be the following: 
data_object_arr => shuffer => convertion => next_batch

GOAL: 



Given batch size as 50, 1000 pixel value and n_class=2
Create batch_x: in shape of (50, 1000)
Create batch_y: in shape of (50,2)
'''

import tensorflow as tf
import numpy as np


# Create data object =============================================
class data_object:

    # Constructor
    def __init__(self,img,label):
        self.img = img
        self.label = label # 1 for Hand and 0 for Others

# End of data_object class =======================================



# Load up data
trainHandImg_array = np.load('trainHandImg_array.npy')
print('trainHandImg_array shape:')
print(trainHandImg_array.shape)

trainOthersImg_array = np.load('trainOthersImg_array.npy')
print('trainOthersImg_array shape:')
print(trainOthersImg_array.shape)

trainImg_arr = np.load('trainImg_arr.npy')
print('trainImg_arr size')
print(trainImg_arr.shape)

print('trainImg_arr[0].img.shape')
print(trainImg_arr[0].img.shape) # Expect to have shape: (10000,)


# The next_batch function used during training
def next_batch(data_array,batch_size,start_index):
    temp_arr=[]
    end_index = start_index+batch_size
    temp_arr = data_array[start_index:end_index]
    return temp_arr

batch_size = 50
start_index = 0
n_class = 2
test_batch = []
test_batch = next_batch(trainImg_arr,batch_size,start_index)
print('test_batch shape')
print(test_batch.shape)


def convert_data(data_array):
	arr_shape = data_array.shape
	arr_W = arr_shape[0] # which in our case, we are expecting 1554
	img_arr_W = data_array[0].img.shape[0] # which in our case, we are epxecting 10000
	# Define img matrix
	img_matrix = np.zeros((arr_W,img_arr_W),dtype=np.int)
	# Define label matrix
	label_matrix = np.zeros((arr_W,n_class),dtype=np.int)
	# put img_arr into the matrix
	for index in range(arr_W):
		img_matrix[index,:] = data_array[index].img.transpose()
		curr_label = data_array[index].label
		if curr_label == 1:
			label_matrix[index,1] = 1
		else:
			label_matrix[index,0] = 1

	return img_matrix, label_matrix


# Test on 'convert_data' function
my_img_matrix, my_label_matrix = convert_data(trainImg_arr)
print('my_img_matrix shape')
print(my_img_matrix.shape)
print('my_label_matrix shape')
print(my_label_matrix.shape)






