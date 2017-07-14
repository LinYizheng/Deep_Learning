from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import math


def shuffle_data(data_array):
    np.random.shuffle(data_array)
    return data_array




test_arr = [1,2,3,4,5]

def print_arr(data_array):
	for index in range(len(data_array)):
		print(data_array[index])

	return



print('before shuffle')
print_arr(test_arr)




print('after shuffle')

shuffle_data(test_arr)
print_arr(test_arr)