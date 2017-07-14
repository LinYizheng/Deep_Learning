import matplotlib as mpl
mpl.use('Qt4Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('./python')
import numpy as np
import os

try:
   import cPickle as pickle
except:
   import pickle

from gps.sample.sample_list import SampleList
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import ACTION, RR_COORD6D, XPLUS, JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE
from gps.utility.coord_transform import *

np.set_printoptions(suppress=True, formatter={'float': '{: 3.4f}'.format})

root_dir = "./experiments/"
# root_dir = "/home/pi/gps_rr/experiments/archive/"
exp_name = "ja_mode_01032017_085345.experiment"
# exp_name = "TrajOpt.experiment"
file_dir = root_dir + exp_name + "/data_files/"
logfile_dir = root_dir + exp_name + "/log/"
# file_dir = root_dir + exp_name + "/data/"
# npzFileName = root_dir + exp_name + "/targets_test.npz"
npzFileName = root_dir + exp_name + "/targets.npz"

# ccond = [9, 14, 19, 24]
ccond = []
ncond = 36  # number of conditions
T = 100  # total number of time steps
nitr = 12  # number of iterations
nsample = 3  # number of samples
dim = 1  # the dimension that will be plot in figure 1


def read_from_npz(filename, key):
    if os.path.exists(filename):
        tmp = dict(np.load(filename))
        if key in tmp:
            return tmp[key]

    return None


tgt = np.zeros((ncond, 12))
brick = np.zeros((ncond, 6))
for i in range(ncond):
    ja = read_from_npz(npzFileName, "target" + str(i))[:6]
    p = read_from_npz(npzFileName, "obs_tgt" + str(i))
    tgt[i] = np.concatenate((ja, p))

# print tgt

# tgt_train = np.zeros((36, 6))
# for i in range(36):
#     tgt_train[i] = read_from_npz(root_dir + exp_name + "/targets.npz", "obs_tgt" + str(i))


pol_samples = pickle.load(open(file_dir + ('pol_sample_itr_%02d.pkl' % (nitr - 1))))
# pol_samples = pickle.load(open(file_dir + ('traj_sample_itr_%02d.pkl' % (nitr - 1))))

sensor_dim = 6
samples = np.zeros((0, nsample, sensor_dim))
for c in range(ncond):
    samples_cond = np.array([sample.get(RR_COORD6D, T - 1) for sample in pol_samples[c].get_samples()])
    samples = np.concatenate((samples, samples_cond.reshape(1, nsample, sensor_dim)))
obs_tgt = tgt[:, 6:]

samples_mean = np.mean(samples, axis=1)
error_x = np.abs(samples_mean[:, 0] - obs_tgt[:, 0])
error_y = np.abs(samples_mean[:, 1] - obs_tgt[:, 1])
error_z = np.abs(samples_mean[:, 2] - obs_tgt[:, 2])
error_xyz = np.linalg.norm(samples_mean[:, :3] - obs_tgt[:, :3], axis=1)
error_uvw = np.zeros(ncond)
axis = 0
for i in range(ncond):
    cur_rot = rot_mat(samples_mean[i, 3:] / 180.0 * np.pi)
    tgt_rot = rot_mat(obs_tgt[i, 3:] / 180.0 * np.pi)
    cur_rot_axis = cur_rot[:, axis].reshape(3, 1)
    tgt_rot_axis = tgt_rot[:, axis].reshape(3, 1)
    _error = cur_rot_axis.T.dot(tgt_rot_axis)
    _error = np.arccos(_error) / np.pi * 180.0
    error_uvw[i] = _error

pickle.dump(dict(x=error_x, y=error_y, z=error_z, xyz=error_xyz, uvw=error_uvw),
            open(root_dir + exp_name + '/error_analysis.pkl', 'wb'))
np.savetxt(root_dir + exp_name + '/error_analysis.txt',
           np.c_[error_x, error_y, error_z, error_xyz, error_uvw],
           fmt='%8.2f', delimiter='\t')

plt.figure(0)
# plt.scatter(tgt_train[:, 0], tgt_train[:, 1], c='grey')
# smp = samples.reshape(-1, sensor_dim)
smp = samples_mean
marker = ['v', '<', 'o', '>', '^']
color = ['r', 'g', 'r', 'g', 'r']
for i in range(5):
    index = range(i*6, (i+1)*6)
    plt.scatter(obs_tgt[index, 0], obs_tgt[index, 1], s=40, marker=marker[i])
    # index = range(i * 48, (i + 1) * 48)
    plt.scatter(np.fmax(np.fmin(smp[index, 0], 570), 525),
                np.fmax(np.fmin(smp[index, 1], 80), -80),
                c=color[i], marker=marker[i], s=40)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.savefig(root_dir + exp_name + '/error_analysis.png')

# plot trajectory of chosen dimension
if len(ccond) == 0:
    num_cond = min(ncond, 4)
    conditions = range(num_cond)
else:
    num_cond = min(len(ccond), 4)
    conditions = ccond[:4]
n_subplots = int(np.ceil(np.sqrt(nitr)))
x = np.arange(T)
plt.figure(1)
color = ['r', 'g', 'b', 'y']
_tgt = np.tile(tgt[:, dim].reshape(ncond, 1), (1, T))
for itr in range(nitr):
    sampleData = pickle.load(open(file_dir + ("sample_itr%02d.pkl" % itr), 'rb'))

    plt.subplot(n_subplots, n_subplots, itr+1)
    plt.title(("Iteration %02d" % itr))

    c = 0
    for n in conditions:
        plt.plot(x, _tgt[n], color[c] + '--')

        if itr > 0:
            pred_traj = pickle.load(open(file_dir + ("pred_traj_itr%02d.pkl" % (itr-1)), 'rb'))
            plt.plot(x, pred_traj[n, :, dim], color[c] + '-.')

        for i in range(nsample):
            plt.plot(x, sampleData[n, i, :, dim], color[c])

        c += 1

n_subplots = int(np.ceil(np.sqrt(ncond)))
x = np.arange(T)
plt.figure(2)
_tgt = np.tile(tgt[:, dim].reshape(ncond, 1), (1, T))
pol_samples = pickle.load(open(file_dir + ('pol_sample_itr_%02d.pkl' % (nitr - 1))))
for n in range(ncond):
    samples_cond = np.array([sample.get(JOINT_ANGLES) for sample in pol_samples[n].get_samples()])

    ax = plt.subplot(n_subplots, n_subplots, n + 1)
    # ax.set_ylim((-20, 20))
    plt.title(("Condition %02d" % n))

    plt.plot(x, _tgt[n], '--')

    for i in range(nsample):
        plt.plot(x, samples_cond[i, :, dim])

# file_path = logfile_dir + "traj.ps"
# plt.savefig(file_path, dpi=300)

# plot 3d trajectory
# plt.figure(2)
# for itr in range(nitr):
#     sampleData = pickle.load(open(file_dir + ("sample_itr%02d.pkl" % itr), 'rb'))
#
#     ax = plt.subplot(3, 5, itr+1, projection='3d')
#     plt.title(("Iteration %02d" % itr))
#
#     _tgt = tgt[:ncond, 6:9]
#     # for cond, color in [(0, 'r'), (1, 'b'), (2, 'g')]:
#     for cond, color in [(0, 'r'), (1, 'b')]:
#         ax.scatter(_tgt[cond, 0], _tgt[cond, 1], _tgt[cond, 2], c=color, s=50)
#
#         for i in range(nsample):
#             data = sampleData[cond, i, :, 6:9]
#             ax.plot(data[:, 0], data[:, 1], data[:, 2], color)
#
plt.show()
#
# plot cost function
# plt.figure(3)
# for cond in range(ncond):
#     for itr in range(nitr):
#         costData = pickle.load(open(file_dir + ("cost_itr%02d.pkl" % itr), 'rb'))
#
#         ax1 = plt.subplot(4, 5, itr+1)
#         plt.title(("Iteration %02d" % itr))
#         ax2 = ax1.twinx()
#
#         for i in range(nsample):
#             # yl2 = costData[cond][i]['l2']
#             # ylog = costData[cond][i]['log']
#             # ylu = costData[cond][i]['lu']
#             ytotal = costData[cond][i]['total']
#             ycc = costData[cond][i]['cc']
#
#             # line = ax1.plot(x, yl2, 'r', x, ylu, 'b', x, ytotal, 'g', x, ylog, 'y', )
#             ax1.plot(x, ytotal, 'g')
#             ax2.plot(x, ycc, 'g--')
#             # plt.legend(line, ['l2', 'lu', 'total', 'log', 'cc'], loc=0)
# #
# #     # color = ['red', 'blue', 'green']
# #     # file_path = logfile_dir + 'cost_' + color[cond] + '.ps'
# #     # plt.savefig(file_path, dpi=300)
#     plt.show()

# dim_action = 12 + dim
#
# plt.figure(3)
#
# sampleData = pickle.load(open(file_dir + ("sample_itr%02d.pkl" % (nitr-1)), 'rb'))
#
# for cond in range(ncond):
#     condd = sampleData[cond, :, :, dim_action].reshape((nsample * T, ))
#
#     plt.subplot(1, 3, cond)
#     plt.title(("Condition %d" % cond))
#     plt.hist(condd, 20)
#
# plt.show()