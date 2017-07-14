gps_root = "/home/pi/GPS_Project/GPS_main/"
import sys
sys.path.append(gps_root + "python")
from gps.agent.pi.pi_robot_API import Communication

import matplotlib as mpl
mpl.use('Qt4Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import time
import numpy as np

try:
   import cPickle as pickle
except:
   import pickle


np.set_printoptions(suppress=True, formatter={'float': '{: 3.2f}'.format})

root_dir = gps_root + 'experiments/'
exp_name = "ja_mode_12022016_082523.experiment"
file_dir = root_dir + exp_name + "/data_files/"
logfile_dir = root_dir + exp_name + "/log/"
npzFileName = root_dir + exp_name + "/targets.npz"

ncond = 4
T = 100
itr_start = 0
itr_end = 15
nsample = 2
sleeptime = 1.0/8

com = Communication()

itritems = [(cond, sample) for cond in range(ncond) for sample in range(nsample)]
itritems = (1, 1)

def read_from_npz(filename, key):
    if os.path.exists(filename):
        tmp = dict(np.load(filename))
        if key in tmp:
            return tmp[key]

    return None

tgt = np.zeros((ncond, 12))
brick = np.zeros((ncond, 6))
for i in range(ncond):
    ja = read_from_npz(npzFileName, "target" + str(i))
    p = read_from_npz(npzFileName, "obs_tgt" + str(i))
    brick[i] = read_from_npz(npzFileName, "brick" + str(i))
    tgt[i] = np.concatenate((ja, p))

for itr in range(itr_start, itr_end):
    sampleData = pickle.load(open(file_dir + ("sample_itr%02d.pkl" % itr), 'rb'))

    # for cond, sample in itritems:
    cond, sample = itritems
    # reset to starting point
    com.Pub_Angle(sampleData[cond, sample, 0, :6])
    com.Set_Object_Pos('hole', brick[cond])
    time.sleep(2)

    for step in range(T):
        print "Iteration %d, condition %d, sample %d, step %d" % (itr, cond, sample, step)

        ja = sampleData[cond, sample, step, :6]
        print "Joint angles:", ja
        print tgt
        pos = sampleData[cond, sample, step, 6:12]
        print "EE coordinates:", pos

        com.Pub_Angle(ja)

        time.sleep(sleeptime)

    plt.figure(0)
    ax = plt.subplot(1, 1, 1, projection='3d')

    _tgt = tgt[cond, 6:]
    ax.scatter(_tgt[0], _tgt[1], _tgt[2], s=50)

    data = sampleData[cond, 0, :, 6:]
    ax.plot(data[:, 0], data[:, 1], data[:, 2])

    plt.show()