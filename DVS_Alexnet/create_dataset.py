'''
Author: San Wong


Create h5py data set for DVS images
'''
import h5py
import os, os.path
import numpy as np
from numpy import *
import cv2
from sklearn import preprocessing

'''
# Setp1: Define Data path (Local Host)
test_hand_data_path = "/Users/san/Documents/GitHub/Deep_Learning/DVS_Alexnet/data/Testing/Hand"
test_others_data_path = "/Users/san/Documents/GitHub/Deep_Learning/DVS_Alexnet/data/Testing/Others"
train_hand_data_path = "/Users/san/Documents/GitHub/Deep_Learning/DVS_Alexnet/data/Training/Hand"
train_others_data_path = "/Users/san/Documents/GitHub/Deep_Learning/DVS_Alexnet/data/Training/Others"
'''


# Setp1: Define Data path (Server)
test_hand_data_path = "/home/user/san/Deep_Learning/DVS_Alexnet/data/Testing/Hand"
test_others_data_path = "/home/user/san/Deep_Learning/DVS_Alexnet/data/Testing/Others"
train_hand_data_path = "/home/user/san/Deep_Learning/DVS_Alexnet/data/Training/Hand"
train_others_data_path = "/home/user/san/Deep_Learning/DVS_Alexnet/data/Training/Others"


# Number of files in each folder
test_hand_data_len = len([name for name in os.listdir(test_hand_data_path) if os.path.isfile(os.path.join(test_hand_data_path,name))])
test_others_data_len = len([name for name in os.listdir(test_others_data_path) if os.path.isfile(os.path.join(test_others_data_path,name))])
train_hand_data_len = len([name for name in os.listdir(train_hand_data_path) if os.path.isfile(os.path.join(train_hand_data_path,name))])
train_others_data_len = len([name for name in os.listdir(train_others_data_path) if os.path.isfile(os.path.join(train_others_data_path,name))])

print(test_hand_data_len)
print(test_others_data_len)
print(train_hand_data_len)
print(train_others_data_len)

img_size = 10000
arr_size = 10001 # img_size plus 1 to include label: 1 for hand and 0 for others

# Test - Hand
testHandImg_array = np.zeros(shape=(test_hand_data_len, arr_size))

#init counter
i=0

for filename in os.listdir(test_hand_data_path):
    print('Going through Test-Hand Images')
    print(filename)
    if filename.endswith(".jpg"): 
        curr_path = os.path.join(test_hand_data_path, filename)
        curr_img = cv2.imread(curr_path)
        # Convert img to grayscale
        curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        # Flatten the image into a 1D array
        curr_img_gray_flatten = curr_img_gray.flatten()
        curr_img_gray_flatten = np.append(curr_img_gray_flatten,1)
        print(i)
        testHandImg_array[i] = curr_img_gray_flatten
        i+=1
        continue
    else:
        continue

# Save Labeled data and No-label data
np.save('testHandImg_array_labeled',testHandImg_array)
testHandImg_array_no_label = testHandImg_array[:,0:testHandImg_array.shape[1]-1]
np.save('testHandImg_array_no_label',testHandImg_array_no_label)

# Save Normalized Labeled data and Normalized No-label data
# Normalize data (No-Label)
norm_testHandImg_array_no_label = preprocessing.normalize(testHandImg_array_no_label)
np.save('norm_testHandImg_array_no_label',norm_testHandImg_array_no_label)
# Normalize data (Labeled)
label_col = ones((test_hand_data_len,1))
norm_testHandImg_array_labeled = np.hstack((norm_testHandImg_array_no_label,label_col))
np.save('norm_testHandImg_array_labeled',norm_testHandImg_array_labeled)






# Test - Others
testOthersImg_array = np.zeros(shape=(test_others_data_len, arr_size))

#init counter
i=0

for filename in os.listdir(test_others_data_path):
    print('Going through Test-Others Images')
    print(filename)
    if filename.endswith(".jpg"): 
        curr_path = os.path.join(test_others_data_path, filename)
        curr_img = cv2.imread(curr_path)
        # Convert img to grayscale
        curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        # Flatten the image into a 1D array
        curr_img_gray_flatten = curr_img_gray.flatten()
        curr_img_gray_flatten = np.append(curr_img_gray_flatten,0)
        print(i)
        testOthersImg_array[i] = curr_img_gray_flatten
        i+=1
        continue
    else:
        continue

# Save Labeled data and No-label data
np.save('testOthersImg_array_labeled',testOthersImg_array)
testOthersImg_array_no_label = testOthersImg_array[:,0:testOthersImg_array.shape[1]-1]
np.save('testHandImg_array_no_label',testOthersImg_array_no_label)

# Normalize data (No-Label)
norm_testOthersImg_array_no_label = preprocessing.normalize(testOthersImg_array_no_label)
np.save('norm_testOthersImg_array_no_label',norm_testOthersImg_array_no_label)
# Normalize data (Labeled)
label_col = zeros((test_others_data_len,1))
norm_testOthersImg_array_labeled = np.hstack((norm_testOthersImg_array_no_label,label_col))
np.save('norm_testOthersImg_array_labeled',norm_testOthersImg_array_labeled)










# Train - Hand
trainHandImg_array = np.zeros(shape=(train_hand_data_len, arr_size))

#init counter
i=0

for filename in os.listdir(train_hand_data_path):
    print('Going through Train-Hand Images')
    print(filename)
    if filename.endswith(".jpg"): 
        curr_path = os.path.join(train_hand_data_path, filename)
        curr_img = cv2.imread(curr_path)
        # Convert img to grayscale
        curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        # Flatten the image into a 1D array
        curr_img_gray_flatten = curr_img_gray.flatten()
        curr_img_gray_flatten = np.append(curr_img_gray_flatten,1)
        print(i)
        trainHandImg_array[i] = curr_img_gray_flatten
        i+=1
        continue
    else:
        continue

# Save Labeled data and No-label data
np.save('trainHandImg_array_labeled',trainHandImg_array)
trainHandImg_array_no_label = trainHandImg_array[:,0:trainHandImg_array.shape[1]-1]
np.save('trainHandImg_array_no_label',trainHandImg_array_no_label)

# Save Normalized Labeled data and Normalized No-label data
# Normalize data (No-Label)
norm_trainHandImg_array_no_label = preprocessing.normalize(trainHandImg_array_no_label)
np.save('norm_trainHandImg_array_no_label',norm_trainHandImg_array_no_label)
# Normalize data (Labeled)
label_col = ones((train_hand_data_len,1))
norm_trainHandImg_array_labeled = np.hstack((norm_trainHandImg_array_no_label,label_col))
np.save('norm_trainHandImg_array_labeled',norm_trainHandImg_array_labeled)










# Train - Others
trainOthersImg_array = np.zeros(shape=(train_others_data_len, arr_size))

#init counter
i=0

for filename in os.listdir(train_others_data_path):
    print('Going through Train-Others Images')
    print(filename)
    if filename.endswith(".jpg"): 
        curr_path = os.path.join(train_others_data_path, filename)
        curr_img = cv2.imread(curr_path)
        # Convert img to grayscale
        curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        # Flatten the image into a 1D array
        curr_img_gray_flatten = curr_img_gray.flatten()
        curr_img_gray_flatten = np.append(curr_img_gray_flatten,0)
        print(i)
        trainOthersImg_array[i] = curr_img_gray_flatten
        i+=1
        continue
    else:
        continue


# Save Labeled data and No-label data
np.save('trainOthersImg_array_labeled',trainOthersImg_array)
trainOthersImg_array_no_label = trainOthersImg_array[:,0:trainOthersImg_array.shape[1]-1]
np.save('trainOthersImg_array_no_label',trainOthersImg_array_no_label)

# Save Normalized Labeled data and Normalized No-label data
# Normalize data (No-Label)
norm_trainOthersImg_array_no_label = preprocessing.normalize(trainOthersImg_array_no_label)
np.save('norm_trainOthersImg_array_no_label',norm_trainOthersImg_array_no_label)
# Normalize data (Labeled)
label_col = zeros((train_others_data_len,1))
norm_trainOthersImg_array_labeled = np.hstack((norm_trainOthersImg_array_no_label,label_col))
np.save('norm_trainOthersImg_array_labeled',norm_trainOthersImg_array_labeled)




# Finally. We combine all the data into TRAIN-DATA group and TEST_DATA group
# After concatenate, we normalize them
train_data_labeled = np.concatenate((trainHandImg_array,trainOthersImg_array),axis=0)
test_data_labeled = np.concatenate((testHandImg_array,testOthersImg_array),axis=0)
np.save('train_data_labeled',train_data_labeled)
np.save('test_data_labeled',test_data_labeled)

# Normalized (Train)
label_col = train_data_labeled[:,-1]
label_col_size = label_col.shape[0]
label_col_reshape = np.reshape(label_col,(label_col_size,1))

norm_train_data = preprocessing.normalize(train_data_labeled[:,0:train_data_labeled.shape[1]-1])
norm_train_data_labeled = np.hstack((norm_train_data,label_col_reshape))
np.save('norm_train_data_labeled',norm_train_data_labeled)


# Normalized (Test)
label_col = test_data_labeled[:,-1]
label_col_size = label_col.shape[0]
label_col_reshape = np.reshape(label_col,(label_col_size,1))

norm_test_data = preprocessing.normalize(test_data_labeled[:,0:test_data_labeled.shape[1]-1])
norm_test_data_labeled = np.hstack((norm_test_data,label_col_reshape))
np.save('norm_test_data_labeled',norm_test_data_labeled)




print('DONE')

