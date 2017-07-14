
'''
Mar 30, 2017
Author: Hong San Wong
Email: hswong1@uci.edu
This file is created to test if npy file can be loaded

'''
import tensorflow as tf
import numpy as np
import cv2

# Create data object =============================================
class data_object:

    # Constructor
    def __init__(self,img,label):
        self.img = img
        self.label = label # 1 for Hand and 0 for Others

# End of data_object class =======================================


# Convert an image to pixel array (i.e: 1x10000)



testImg_arr = np.load('testImg_arr.npy')
trainImg_arr = np.load('trainImg_arr.npy')

print('testImg_arr shape:')
print(testImg_arr.shape)

# Load Success
# Now, try to display an image from the loaded array
test_img = testImg_arr[0].img
	#cv2.imshow('1st Test img',test_img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
print('test_img size')
print(test_img.shape)	

# Convert image to Grayscale
#test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
#print(test_img_gray.shape)

# Flatten the image into a 1D array
#test_img_gray_flatten = test_img_gray.flatten()
#print(test_img_gray_flatten.shape)

# Image Display works
# Now, try to print out label

test_img_label = testImg_arr[0].label
print('test_img_label:')
print(test_img_label)


# That works too. Good to go


'''
Date: Apr 5, 2017
The following code test the minibatch data shape
'''

def next_batch(data_array,batch_size,start_index):
    temp_arr=[]
    end_index = start_index+batch_size
    temp_arr = data_array[start_index:end_index]
    return temp_arr



trainImg_next_batch = []
batch_x = []
batch_y = []

batch_size = 50
start_index = 0

# Load data (batch) into trainImg_next_batch
trainImg_next_batch = next_batch(trainImg_arr,batch_size,start_index)
print('trainImg_next_batch size:')
print(trainImg_next_batch.shape)

# Fill up batch_x and batch_y
for index in range(len(trainImg_next_batch)):
    curr_data_object = trainImg_next_batch[index]
    batch_x.append(curr_data_object.img)
    batch_y.append(curr_data_object.label)

print('batch_x size')
print(batch_x.shape)
print('batch_y size')
print(batch_y.shape)
