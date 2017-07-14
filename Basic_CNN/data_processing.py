'''
Mar 30, 2017
Author: Hong San Wong
Email: hswong1@uci.edu
This file describe a image processing before import to the CNN for training
It creates an array of Data_object, which contains two 

'''


from __future__ import print_function

import glob
from PIL import Image
import numpy as np
import cv2
import os

# Create data object =============================================
class data_object:

    # Constructor
    def __init__(self,img,label):
        self.img = img
        self.label = label # 1 for Hand and 0 for Others

# End of data_object class =======================================


# Read Input and store its label
# This function will be called to load data both in Training and Testing
def read(dataset):
    if dataset is "training":
        print(dataset)
        
        imageFolderPath='data/Training'
        print(imageFolderPath)


        #trainHandImgPath=glob.glob(imageFolderPath+"/Hand/*.JPG")
        trainHandImgPath=imageFolderPath+"/Hand"
        print(trainHandImgPath) # Looks good here

        # Create train_handImage_label array
        trainHandImg_array = []
        # print(trainHandImg_array)
        
 
        
        # Mod Mar 30, 2017
        for filename in os.listdir(trainHandImgPath):
            print('Inside listdir loop')
            print(filename)
            if filename.endswith(".jpg"): 
                    # print(os.path.join(trainHandImgPath, filename))
                curr_path = os.path.join(trainHandImgPath, filename)
                print(curr_path)
                curr_img = cv2.imread(curr_path)
                    # Display Image
                    #cv2.imshow('curr_img',curr_img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()

                # Convert img to grayscale
                curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
                # Flatten the image into a 1D array
                curr_img_gray_flatten = curr_img_gray.flatten()    
                # Create Data_object
                curr_data_object = data_object(curr_img_gray_flatten,1)
                trainHandImg_array.append(curr_data_object)

                continue
            else:
                continue
        


        # Save array
        np.save('trainHandImg_array',trainHandImg_array)






        
        # Create train_othersImg_label array
        trainOthersImg_array = []
        trainOthersImgPath=imageFolderPath+"/Others"
        print(trainOthersImgPath) # Looks good here

        for filename in os.listdir(trainOthersImgPath):
            print('Inside listdir loop')
            print(filename)
            if filename.endswith(".jpg"): 
                curr_path = os.path.join(trainOthersImgPath, filename)
                print(curr_path)
                curr_img = cv2.imread(curr_path)
                    # Display Image
                    #cv2.imshow('curr_img',curr_img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()

                # Convert img to grayscale
                curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
                # Flatten the image into a 1D array
                curr_img_gray_flatten = curr_img_gray.flatten() 



                # Create Data_object
                curr_data_object = data_object(curr_img_gray_flatten,0)
                trainOthersImg_array.append(curr_data_object)

                continue
            else:
                continue

        # Save array
        np.save('trainOthersImg_array',trainOthersImg_array)

        # Combine both array into ONE Training data array
        trainImg_arr = []
        trainImg_arr = np.concatenate((trainHandImg_array,trainOthersImg_array),axis=0)
        np.save('trainImg_arr',trainImg_arr)


    elif dataset is "testing":

        print(dataset)
        
        imageFolderPath='data/Testing'
        print(imageFolderPath)



        testHandImgPath=imageFolderPath+"/Hand"
        print(testHandImgPath) # Looks good here

        # Create train_handImage_label array
        testHandImg_array = []
        # print(testHandImg_array)
        
 
        
        # Mod Mar 30, 2017
        for filename in os.listdir(testHandImgPath):
            print('Inside listdir loop')
            print(filename)
            if filename.endswith(".jpg"): 
                    # print(os.path.join(testHandImgPath, filename))
                curr_path = os.path.join(testHandImgPath, filename)
                print(curr_path)
                curr_img = cv2.imread(curr_path)
                    # Display Image
                    #cv2.imshow('curr_img',curr_img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()


                # Convert img to grayscale
                curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
                # Flatten the image into a 1D array
                curr_img_gray_flatten = curr_img_gray.flatten()
                     
                # Create Data_object
                curr_data_object = data_object(curr_img_gray_flatten,1)
                testHandImg_array.append(curr_data_object)

                continue
            else:
                continue
        


        # Save array
        np.save('testHandImg_array',testHandImg_array)






        
        # Create train_othersImg_label array
        testOthersImg_array = []
        testOthersImgPath=imageFolderPath+"/Others"
        print(testOthersImgPath) # Looks good here

        for filename in os.listdir(testOthersImgPath):
            print('Inside listdir loop')
            print(filename)
            if filename.endswith(".jpg"): 
                curr_path = os.path.join(testOthersImgPath, filename)
                print(curr_path)
                curr_img = cv2.imread(curr_path)
                    # Display Image
                    #cv2.imshow('curr_img',curr_img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()

                # Convert img to grayscale
                curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
                # Flatten the image into a 1D array
                curr_img_gray_flatten = curr_img_gray.flatten() 

                # Create Data_object
                curr_data_object = data_object(curr_img_gray_flatten,0)
                testOthersImg_array.append(curr_data_object)

                continue
            else:
                continue

        # Save array
        np.save('testOthersImg_array',testOthersImg_array)

        # Combine both array into ONE Training data array
        testImg_arr = []
        testImg_arr = np.concatenate((testHandImg_array,testOthersImg_array),axis=0)
        np.save('testImg_arr',testImg_arr)

    else:
        raise ValueError("dataset must be 'testing' or 'training'")


def main():
    read("training")
    read("testing")



if __name__ == "__main__": main()