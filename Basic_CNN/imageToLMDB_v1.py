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


imageFolderPath='data/Training'
trainHandImgPath=imageFolderPath+"/Hand"

def write_images_to_lmdb(img_dir, db_name):
    for root, dirs, files in os.walk(img_dir, topdown = False):
        if root != img_dir:
            continue
        map_size = 64*64*3*2*len(files)
        env = lmdb.Environment(db_name, map_size=map_size)
        txn = env.begin(write=True,buffers=True)
        for idx, name in enumerate(files):
            X = mp.imread(os.path.join(root, name))
            y = 1
            datum = array_to_datum(X,y)
            str_id = '{:08}'.format(idx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())   
    txn.commit()
    env.close()
    print " ".join(["Writing to", db_name, "done!"])