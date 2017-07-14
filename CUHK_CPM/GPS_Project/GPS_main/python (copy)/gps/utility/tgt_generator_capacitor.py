import os
import sys
import numpy as np

_file_path = os.path.abspath(os.path.dirname(__file__))
_root_path = os.path.dirname(os.path.dirname(_file_path))
sys.path.append(_root_path)

from gps.GazeboInterface import RobotControl, PneumaticGripper, Model
#############################
# default parameters
#############################
gps_root = "/home/pi/GPS_Project/GPS_main/"
caffe_root = "/home/pi/caffe/"
exp_name = "ja_mode_02222017_231624.experiment"
N = 5
# capacitor location region
# for six axis z=775, for four axis z=625
location_region = dict(
    xlim=[ 250,  350],
    ylim=[-50,  50],
    zlim=[ 625, 625],
    ulim=[0, 0],
    vlim=[0, 0],
    wlim=[0, 0]
)
offset_z = 100
ip = np.array([90, -90, -60, 0, 0, 0])  # for four axis
# ip = np.array([-90, 0, 0, 0, 0, 0])  # for six axis
#############################
# end of default parameters
#############################

import sys
sys.path.append(gps_root + "python")

import matplotlib.pyplot as plt
import time
import os
# from coord_transform import *
from gps.agent.pi.pi_robot_API import Communication

np.set_printoptions(suppress=True, formatter={'float': '{: 3.4f}'.format})

npzFileNameTrain = gps_root + "experiments/" + exp_name + "/targets.npz"
# npzFileNameTrain = gps_root + "experiments/" + exp_name + "/targets_test.npz"

robot = RobotControl()
model = Model()


def save_to_npz(filename, key, value):
    tmp = {}
    if os.path.exists(filename):
        with np.load(filename) as f:
            tmp = dict(f)
    tmp[key] = value
    np.savez(filename, **tmp)

obj_poses = np.array([[320, -20, 630, 0, 0, 90],
                      [320,  20, 630, 0, 0, 90],
                      [340,   0, 630, 0, 0, 90],
                      [360, -20, 630, 0, 0, 90],
                      [360,  20, 630, 0, 0, 90]])

# randomly generate capacitor locations
for n in range(N):
    robot.pubPosition('Go', np.array([200, -200, 100, 0, 0, 0]))
    time.sleep(2)

    # obj_pose_dict = {}
    # for key, value in location_region.iteritems():
    #     obj_pose_dict[key[0]] = np.round((np.random.rand() * (value[1] - value[0]) + value[0]) * 1000) / 1000
    # obj_pose = np.array([obj_pose_dict['x'], obj_pose_dict['y'], obj_pose_dict['z'],
    #                      obj_pose_dict['u'], obj_pose_dict['v'], obj_pose_dict['w']])
    obj_pose = obj_poses[n]

    # move to location
    capacitor_name = 'capacitor%d' % n
    print capacitor_name, obj_pose
    if not model.addGazeboModel(capacitor_name, 'capacity', init_pos=obj_pose):
        model.setLinkPos(capacitor_name + '::base_link', obj_pose)
    time.sleep(2)
    ee_pose = obj_pose[:3].copy()
    ee_pose[2] = offset_z
    # ee_pose = np.r_[ee_pose, np.array([0, 0, 180])] + np.random.randn(6) * 0.001  # for six axis
    ee_pose = np.r_[ee_pose, np.array([0, 0, 0])] + np.random.randn(6) * 0.001  # for four axis

    print "End-effector pose:", ee_pose
    robot.pubPosition('GO', ee_pose)
    time.sleep(2)
    raw_input()
    ja_tgt = robot.getAngle()[:]

    ja_ini = ip + np.r_[np.random.randn(3), np.array([0, 0, 0])]
    robot.pubAngle(ja_ini)
    time.sleep(2)

    save_to_npz(npzFileNameTrain, "initial" + str(n), ja_ini)
    save_to_npz(npzFileNameTrain, "target" + str(n), ja_tgt)
    save_to_npz(npzFileNameTrain, "obs_tgt" + str(n), ee_pose)
    save_to_npz(npzFileNameTrain, "obj_pose" + str(n), obj_pose)
