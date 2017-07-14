import os
import sys
import time
import numpy as np
np.set_printoptions(suppress=True, formatter={'float': '{: 3.4f}'.format})

_file_path = os.path.abspath(os.path.dirname(__file__))
_root_path = os.path.dirname(os.path.dirname(_file_path))
sys.path.append(_root_path)

from gps.GazeboInterface import RobotControl, Model
#############################
# default parameters
#############################
gps_root = "/home/pi/GPS_Project/GPS_main"
exp_name = "ja_mode_02242017_090718.experiment"
# N = 9
# capacitor location region, for randomly generate capacitor locations
# for six axis z=775, for four axis z=625
location_region = dict(
    xlim=[ 250,  350],
    ylim=[-50,  50],
    zlim=[ 625, 625],
    ulim=[0, 0],
    vlim=[0, 0],
    wlim=[0, 0]
)
offset_z = 100 # default ee z axis
ip = np.array([90, 0, -60, 0, 0, 0])  # initial state, for four axis
# ip = np.array([-90, 0, 0, 0, 0, 0])  # for six axis

# 12cm * 12cm area (total 49 capacitors)
N = 49
x = np.arange(240, 360.1, 20)
y = np.arange(-60, 60.1, 20)
z = np.ones((N, 1)) * 630
u = np.random.rand(N, 1) * 180
v = np.zeros((N, 1))
w = np.ones((N, 1)) * 90
mesh = np.array(np.meshgrid(x, y)).T.reshape(N, 2)
obj_poses = np.c_[mesh, z, u, v, w]
#############################
# end of default parameters
#############################

npzFileNameTrain = os.path.join(gps_root, 'experiments', exp_name, 'targets.npz')

robot = RobotControl()
model = Model()


def save_to_npz(filename, key, value):
    tmp = {}
    if os.path.exists(filename):
        with np.load(filename) as f:
            tmp = dict(f)
    tmp[key] = value
    np.savez(filename, **tmp)


for n in range(N):
    # randomly generate capacitor locations
    # obj_pose_dict = {}
    # for key, value in location_region.iteritems():
    #     obj_pose_dict[key[0]] = np.round((np.random.rand() * (value[1] - value[0]) + value[0]) * 1000) / 1000
    # obj_pose = np.array([obj_pose_dict['x'], obj_pose_dict['y'], obj_pose_dict['z'],
    #                      obj_pose_dict['u'], obj_pose_dict['v'], obj_pose_dict['w']])

    # create capacitor
    obj_pose = obj_poses[n]
    capacitor_name = 'capacitor%d' % n
    print capacitor_name, obj_pose
    if not model.addGazeboModel(capacitor_name, 'capacity', init_pos=obj_pose):
        model.setLinkPos(capacitor_name + '::base_link', obj_pose)
    time.sleep(2)

    # calculate target ee pose
    ee_pose = obj_pose[:3].copy()
    ee_pose[2] = offset_z
    # ee_pose = np.r_[ee_pose, np.array([0, 0, 180])] + np.random.randn(6) * 0.001  # for six axis
    ee_pose = np.r_[ee_pose, np.array([0, 0, 0])] + np.random.randn(6) * 0.001  # for four axis
    print "End-effector pose:", ee_pose

    # move ee to position
    robot.pubPosition('GO', ee_pose)
    time.sleep(2)

    # get target joint angle
    ja_tgt = robot.getAngle()[:]

    # randomly generate initial ee pose
    ja_ini = ip + np.r_[np.random.randn(3), np.array([0, 0, 0])]

    save_to_npz(npzFileNameTrain, "initial" + str(n), ja_ini[:4])
    save_to_npz(npzFileNameTrain, "target" + str(n), ja_tgt[:4])
    save_to_npz(npzFileNameTrain, "obs_tgt" + str(n), ee_pose[:4])
    save_to_npz(npzFileNameTrain, "obj_pose" + str(n), obj_pose)
