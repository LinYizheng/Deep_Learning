'''
Author: Hong San Wong
Date: 31, Mar, 2017



This code is to see if I can extract part of the array from npy file

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



def next_batch(data_array,batch_size,start_index):
    temp_arr=[]
    end_index = start_index+batch_size
    temp_arr = data_array[start_index:end_index]
    return temp_arr

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
print(trainImg_arr[0].img.shape)


print('trainImg_arr[0].label')
print(trainImg_arr[0].label)

# Can I just access to the first element
print('access to first arguemnt of shape output')
print(trainImg_arr[0].img.shape[0])
print('Type Checking: type(trainImg_arr[0].img.shape[0])')
print(type(trainImg_arr[0].img.shape[0]))


batch_size = 50

# Try to copy one element to an empty array
# trail_arr = []
# trail_arr = testImg_arr[0:3]

curr_batch = []
curr_batch = next_batch(trainImg_arr,batch_size,0)

print(curr_batch.shape)



