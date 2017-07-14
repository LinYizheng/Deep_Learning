#############################
# default parameters
#############################
gps_root = "/home/pi/GPS_Project/GPS_main"
caffe_root = "/home/pi/caffe/"
exp_name = "ja_mode.experiment"
mean_value = [104, 117, 123]

# for pre-train net
train_batch_size = 50
test_batch_size = 200

# for GPS vision net
batch_size = 50
dim_input = 6
dim_output = 6
#############################
# end of default parameters
#############################

import sys
sys.path.append(caffe_root + "python")
sys.path.append(gps_root + "python")

import os
import json
import caffe
import caffe.draw

from caffe import layers as L, params as P
from gps.algorithm.policy_opt import __file__ as policy_opt_path

sys.path.append('/'.join(str.split(policy_opt_path, '/')[:-1]))

net_folder = gps_root + "experiments/" + exp_name + "/vision_net/"
data_folder = gps_root + "experiments/" + exp_name + "/pre_train_data/"
if not os.path.exists(net_folder):
    os.mkdir(net_folder)

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

data_layer_info_train = json.dumps({
    'shape': [{'dim': (batch_size, dim_input)},
              {'dim': (batch_size, dim_output)},
              {'dim': (batch_size, dim_output, dim_output)}]
})


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler, bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)


def fc_relu(bottom, nout, param=learned_param, weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0.0)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param, weight_filler=weight_filler, bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)


def spatial_softmax(bottom, nout, weight_filler):
    reshaped = L.Reshape(bottom, shape={'dim': [0, 0, -1]})
    softmax = L.Softmax(reshaped, axis=2)
    fpoints = L.InnerProduct(softmax, num_output=nout, weight_filler=weight_filler, param=frozen_param)
    return reshaped, softmax, fpoints


def caffenet(netname, batchsize, mean_val, folder, pre_train=False, data_layer_info=None, freeze_cnn=False):
    n = caffe.NetSpec()
    if pre_train:
        img_lmdb = folder + "image_lmdb"
        n.data = L.Data(source=img_lmdb, batch_size=batchsize, backend=P.Data.LMDB,
                        transform_param=dict(mean_value=mean_val))
    else:
        n.data = L.Input(shape={'dim': [batch_size, 3, 240, 240]})

    param_cnn = learned_param if freeze_cnn is False else frozen_param
    conv1, n.relu1 = conv_relu(n.data, 7, 64, stride=2, param=param_cnn)
    n.__setattr__('conv1/7x7_s2', conv1)  # set the name of the first conv to the name of the first conv of googlenet
    n.conv2, n.relu2 = conv_relu(n.relu1, 5, 32, stride=1, param=param_cnn)
    n.conv3, n.relu3 = conv_relu(n.relu2, 5, 32, stride=1, param=param_cnn)

    # add spatial softmax
    n.reshape4, n.softmax4, n.fpoints4 = spatial_softmax(n.relu3, 64, weight_filler=
        dict(type='weight', image_size=109, channel=32, kernel_size=[7, 5, 5], stride=[2, 1, 1]))

    if pre_train:
        lbl_lmdb = folder + "label_lmdb"
        n.label = L.Data(source=lbl_lmdb, batch_size=batchsize, backend=P.Data.LMDB)
        n.out = L.InnerProduct(n.fpoints4, num_output=27, param=learned_param,
                               weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0.0))
        n.loss = L.EuclideanLoss(bottom=['out', 'label'])
    else:
        n.obs, n.action, n.precision = L.Python(ntop=3, python_param=dict(module='policy_layers',
                                                param_str=data_layer_info, layer='PolicyDataLayer'))
        n.concat4 = L.Concat(bottom=['fpoints4', 'obs'], axis=1)

        n.fc5, n.relu5 = fc_relu(n.concat4, 40)
        n.fc6, n.relu6 = fc_relu(n.relu5, 40)
        n.output = L.InnerProduct(n.relu6, num_output=6, param=learned_param,
                               weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0.0))
        n.loss = L.Python(n.output, n.action, n.precision, loss_weight=1.0,
                       python_param=dict(module='policy_layers', layer='WeightedEuclideanLoss'))

    proto = n.to_proto()
    with open(netname + '.prototxt', 'w') as f:
        f.write(str(proto))

    caffe.draw.draw_net_to_file(proto, netname + '.png', 'TB')


caffenet(net_folder + 'pre_train/pre_train_net', train_batch_size, mean_value, data_folder + "training_data/", True)
caffenet(net_folder + 'pre_train/pre_train_test_net', test_batch_size, mean_value, data_folder + "testing_data/", True)
caffenet(net_folder + 'vision_net', batch_size, mean_value, None, False, data_layer_info_train)
caffenet(net_folder + 'vision_net_cnn_frozen', batch_size, mean_value, None, False, data_layer_info_train, freeze_cnn=True)
