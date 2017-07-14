import sys

caffe_root = "/home/pi/caffe/"
sys.path.append(caffe_root + "python")

import matplotlib as mpl
mpl.use('Qt4Agg')

import time
import caffe
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from caffe import layers as L
from caffe.proto.caffe_pb2 import SolverParameter

# set print defaults
np.set_printoptions(suppress=True, formatter={'float': '{: 0.4f}'.format})

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# initialize network
netprotofile = caffe_root + "models/bvlc_googlenet/deploy.prototxt"
modelfile = caffe_root + "models/bvlc_googlenet/bvlc_googlenet.caffemodel"

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(netprotofile, modelfile, caffe.TEST)

time.sleep(1)


# visualize first layer filters
def vis_square(data, padwidth):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, padwidth), (0, padwidth))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)

filters = net.params['conv1/7x7_s2'][0].data
plt.figure(1)
vis_square(filters.transpose(0, 2, 3, 1), 1)

# image preprocessing
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1, 3, 224, 224)

image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
plt.figure(2)
plt.imshow(image)

transformed_image = transformer.preprocess('data', image)

# predict
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
print 'predicted class is:', output_prob.argmax()

labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:'
for i in top_inds:
    print output_prob[i], '\t', labels[i]

feat = net.blobs['conv1/7x7_s2'].data[0]
plt.figure(3)
vis_square(feat, 2)
plt.show()