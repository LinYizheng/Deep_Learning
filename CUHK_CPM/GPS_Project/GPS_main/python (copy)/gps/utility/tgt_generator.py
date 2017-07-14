import numpy as np
#############################
# default parameters
#############################
gps_root = "/home/pi/GPS_Project/GPS_main/"
caffe_root = "/home/pi/caffe/"
exp_name = "ja_mode_12282016_160102.experiment"
N = 80
# brick location region
brick_lim = dict(
    xlim=[ 0.60,  0.70],
    ylim=[-0.05,  0.05],
    zlim=[ 0.775, 0.775]
)
brick2peg_offset = np.array([0, 0, 130])
ee2peg_offset = np.array([0, 0, -105])
ip = np.array([-90, 0, 0, 0, 0, 0])
#############################
# end of default parameters
#############################

import sys
sys.path.append(gps_root + "python")

import matplotlib.pyplot as plt
import time
import os
from coord_transform import *
from gps.agent.pi.pi_robot_API import Communication

np.set_printoptions(suppress=True, formatter={'float': '{: 3.4f}'.format})

npzFileNameTrain = gps_root + "experiments/" + exp_name + "/targets.npz"
# npzFileNameTrain = gps_root + "experiments/" + exp_name + "/targets_test.npz"

com = Communication()

# brick_poss = np.array([[0.63, -0.02, 0.775],
#                       [0.63,  0.02, 0.775],
#                       [0.67, -0.02, 0.775],
#                       [0.67,  0.02, 0.775]])

# x = np.arange(0.64, 0.6601, 0.004)
# y = np.arange(-0.01, 0.0101, 0.004)
# x = np.arange(0.642, 0.65801, 0.004)
# y = np.arange(-0.008, 0.0081, 0.004)
x = np.arange(0.644, 0.6561, 0.004)
y = np.arange(-0.006, 0.00601, 0.004)
u = np.arange(-30.0, 30.1, 15.0) / 180.0 * np.pi
mesh = np.meshgrid(x, y, u)
mesh_reshape = np.array(mesh).T.reshape(-1, 3)
# brick_poss = np.concatenate((mesh_reshape, np.ones((N, 1)) * 0.775), axis=1)
brick_poss = np.concatenate((mesh_reshape[:, :2], np.ones((N, 1)) * 0.775, np.zeros((N, 2)),
                             mesh_reshape[:, 2].reshape(N, 1)), axis=1)

def save_to_npz(filename, key, value):
    tmp = {}
    if os.path.exists(filename):
        with np.load(filename) as f:
            tmp = dict(f)
    tmp[key] = value
    np.savez(filename, **tmp)

# randomly generate brick locations
for n in range(N):
    com.Pub_Position(np.array([418, 0, 630, 0, -90, 180]))
    time.sleep(2)

    # brick_pos_dict = {}
    # for key, value in brick_lim.iteritems():
    #     brick_pos_dict[key[0]] = np.round((np.random.rand() * (value[1] - value[0]) + value[0]) * 1000) / 1000
    # brick_pos = np.array([brick_pos_dict['x'], brick_pos_dict['y'], brick_pos_dict['z']])
    brick_pos = brick_poss[n]

    # move brick to location
    print "Brick location:", brick_pos
    com.Set_Object_Pos('hole', brick_pos)
    time.sleep(2)

    # get brick feature points
    brick_cd_world = get_brick_coord_in_world_m(brick_pos)
    brick_coord = world2robot_mm(brick_cd_world)
    # print brick_coord[0]
    peg_center = brick_coord[0] + brick2peg_offset
    ee_rot = rot_mat(np.array([brick_pos[5], -np.pi/2, -np.pi]))
    print ee_rot.dot(ee2peg_offset.reshape(3, 1)).T[0]
    ee_pos_tgt = np.r_[ee_rot.dot(ee2peg_offset.reshape(3, 1)).T[0] + peg_center,
                 np.array([brick_pos[5] / np.pi * 180.0, -90, -180])] + np.random.randn(6) * 0.001
    # ee_point_tgt = get_peg_coord_in_robot_mm(ee_pos_tgt[:3], ee_pos_tgt[3:] / 180.0 * np.pi)[5:].reshape((-1,))
    print "End-effector pose:", ee_pos_tgt
    # print ee_point_tgt
    com.Pub_Position(ee_pos_tgt)
    time.sleep(2)
    raw_input()
    ja_tgt = com.Get_angle()

    ja_ini = ip + np.r_[np.random.randn(3), np.array([0, 0, 0])]
    com.Pub_Angle(ja_ini)
    time.sleep(2)
    ee_pos_ini = com.Get_position()
    ee_point_ini = get_peg_coord_in_robot_mm(ee_pos_ini[:3], ee_pos_ini[3:] / 180.0 * np.pi)[5:].reshape((-1,))
    print ee_point_ini

    save_to_npz(npzFileNameTrain, "initial" + str(n), ja_ini)
    save_to_npz(npzFileNameTrain, "target" + str(n), ja_tgt)
    save_to_npz(npzFileNameTrain, "obs_tgt" + str(n), ee_pos_tgt)
    save_to_npz(npzFileNameTrain, "brick" + str(n), np.r_[brick_pos, np.array([0, 0, 0])])


# plot = True
# sleeptime = 3
# radius = 20.0
# nppc = 4
# nstep = 3
# cp = np.array([400.00, 300.00, 500.00, 0.00, -80.00, 180.00])
# ip = np.array([-80.0, -14.0, -47.0, 105.0, 40.0, -85.0])
#
# size = nppc * nstep
# points = np.tile(cp, [size, 1])
# step = radius/nstep
# theta = np.arange(-np.pi, np.pi, 0.01)
#
# outx = radius * np.sin(theta) + cp[0]
# outy = radius * np.cos(theta) + cp[1]
#
# dtheta = np.pi / nppc
# theta = np.arange(-np.pi, np.pi, np.pi/nppc*2)
# x = np.zeros(nppc * nstep)
# y = np.zeros(nppc * nstep)
#
# for i in range(1, nstep+1):
#     r = i * step
#     theta += dtheta
#     x[(i-1)*nppc:i*nppc] = r * np.sin(theta)
#     y[(i-1)*nppc:i*nppc] = r * np.cos(theta)
#
# points[:, 0] += x
# points[:, 1] += y
#
# points = np.random.permutation(points)
# points[:, 2:6] += np.random.randn(size, 4) * 0.1
#
# train_pt = points[:nppc*(nstep-1)]
# test_pt = points[nppc*(nstep-1):size]
#
# if plot:
#     plt.figure(1)
#     plt.plot(outx, outy)
#     plt.plot(train_pt[:, 0], train_pt[:, 1], 'ro')
#     plt.plot(test_pt[:, 0], test_pt[:, 1], 'go')
#     plt.show()






#
# print "Saving testing points..."
# i = 0
# for p in test_pt:
#     com.Pub_Position(p)
#     print "Moving to point: ", p
#
#     time.sleep(sleeptime)
#
#     ja = com.Get_angle()
#     initial = ip + np.random.randn(6)
#     save_to_npz(npzFileNameTest, "initial" + str(i), initial)
#     save_to_npz(npzFileNameTest, "target" + str(i), ja)
#     save_to_npz(npzFileNameTest, "obs_tgt" + str(i), p)
#     print "Saved point: ", initial, ja, p
#     i += 1



