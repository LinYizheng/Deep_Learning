'''
Author: San Wong

The following code convert DVS image from Gray scale to RGB to further fit into the AlexNet

It will return a LMDB file

'''

from __future__ import print_function

import glob
from PIL import Image
import numpy as np
import cv2
import os, os.path


# For LMDB
import lmdb
import caffe



# Read input
def read(dataset):
	if dataset is "training":
		imageFolderPath='data/Training'
		trainHandImgPath=imageFolderPath+"/Hand"


		# Show how many files are there in Training folder
		num_files = len([f for f in os.listdir(trainHandImgPath) if os.path.isfile(os.path.join(trainHandImgPath,f))])

