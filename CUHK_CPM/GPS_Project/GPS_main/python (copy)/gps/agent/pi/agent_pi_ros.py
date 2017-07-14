""" This file defines an agent for the PI ROS environment. """

import os
import cv2
import time
import caffe
import logging
import numpy as np

from pi_robot_API import Communication

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.sample.sample import Sample

from gps.proto.gps_pb2 import ACTION, RR_COORD6D, XPLUS, JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE
from gps.algorithm.policy.caffe_policy import CaffePolicy
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
import matplotlib.pyplot as plt

baseLogger = logging.getLogger('base')
sampleLogger = logging.getLogger('sample')

class AgentPIROS(Agent):

    def __init__(self, hyperparams=None):

        config = hyperparams
        Agent.__init__(self, config)

        conditions = self._hyperparams['conditions']
        self.frequency = self._hyperparams['sampling_freq']
        self.sleepTime4Reset = self._hyperparams['sleep_time']

        for field in ('x0', 'ee_points_tgt', 'ee_obs_tgt'):
            self._hyperparams[field] = setup(self._hyperparams[field], conditions)
        self.sensor_dims = self._hyperparams['sensor_dims']

        # self.x0 & self.tgt are lists of initial states & target states for all conditions, aka, x0s & ee_tgts
        self.x0 = self._hyperparams['x0']
        self.tgt = self._hyperparams['ee_points_tgt']
        self.obs_tgt = self._hyperparams['ee_obs_tgt']
        # self.brick = self._hyperparams['brick_loc']
        self.obj_pose = self._hyperparams['obj_pose']

        if self._hyperparams['vision_on']:
            self.image_folder = self._hyperparams['raw_images']
            if not os.path.exists(self.image_folder):
                os.mkdir(self.image_folder)

            img_sz = self.img_sz = self._hyperparams['image_size']
            self.transformer = caffe.io.Transformer({'data': [self.T, 3, img_sz, img_sz]})
            self.transformer.set_transpose('data', (2, 0, 1))
            self.transformer.set_mean('data', self._hyperparams['mean_value'])
            # TODO comment this line
            # self.transformer.set_channel_swap('data', (2, 1, 0))

        self.com = Communication()

    def sample(self, pol, itr, cond, nsmp, data=None):
        print 'zx, enter sample'

        if isinstance(pol, LinearGaussianPolicy):
            pol_string = "Linear Gaussian Policy"
        elif isinstance(pol, CaffePolicy):
            pol_string = "Caffe Policy"

        sampleLogger.info(pol_string)
        print pol_string

        baseLogger.info("Initial state:")
        baseLogger.info(self.x0[cond])
        baseLogger.info("Target state:")
        baseLogger.info(self.tgt[cond])
        baseLogger.info(self.obs_tgt[cond])
        sampleLogger.info("Initial state:")
        sampleLogger.info(self.x0[cond])
        sampleLogger.info("Target state:")
        sampleLogger.info(self.tgt[cond])
        sampleLogger.info(self.obs_tgt[cond])

        # reset the robot to initial state
        baseLogger.info("Reset to initial state...")
        self.com.Pub_Angle(self.x0[cond])
        # self.com.Set_Object_Pos('hole', self.brick[cond])
        self.com.Set_Object_Pos('capacitor0%d' % cond, self.obj_pose[cond])
        time.sleep(self.sleepTime4Reset)

        ee_coord = self.com.Get_position()  # read_coordinate
        ja = self.x0[cond] # joint angle of initial

        if self._hyperparams['vision_on']:
            image = self.com.Get_image_RGB()
            image_meta = np.zeros((self.T, 3, self.img_sz, self.img_sz), dtype=np.float32)
        else:
            image = image_meta = None

        # generate noise vectors
        noise = generate_noise(self.T, self.dU, self._hyperparams)

        sample = Sample(self)
        dja = self.sensor_dims[JOINT_ANGLES]
        dact = self.sensor_dims[ACTION]
        dcod = self.sensor_dims[RR_COORD6D]
        djv = self.sensor_dims[JOINT_VELOCITIES]

        joint_angles = np.zeros((self.T, dja))
        actions = np.zeros((self.T, dact))
        ee_coords = np.zeros((self.T, dcod))
        joint_velocities = np.zeros((self.T, djv))
        XP = np.zeros((self.T, dja+dcod))

        v_cur = np.zeros(djv)
        # errors = np.zeros((self.T, 2))
        # TODO: add ee_points
        for t in range(self.T):
            sampleLogger.info("Condition: %d", cond)
            sampleLogger.info("Time step %d, joint angles:", t)
            sampleLogger.info(ja)
            sampleLogger.info("Time step %d, end-effector coordinates:", t)
            sampleLogger.info(ee_coord)

            if data is not None:
                data[t, :dja] = ja
                data[t, dja:dja+dcod] = ee_coord
            print "Condition:", cond
            print "Time step %d, joint angles and end-effector coordinates:" % t
            print "cur: ", ja, ee_coord
            print "tgt: ", self.tgt[cond], self.obs_tgt[cond]

            if self._hyperparams['vision_on']:
                meta = self.image_folder + "%d_%d_%d_%d.png" % (itr, cond, nsmp, t)
                cv2.imwrite(meta, image)
                transformed_image = self.transformer.preprocess('data', image)
                image_meta[t] = transformed_image
            else:
                transformed_image = None

            # tgt_noise = np.r_[np.random.randn(3), np.random.randn(3) * 0.1]
            tgt_noise = np.zeros(6)
            # xp_cur = np.concatenate((ja, self.obs_tgt[cond] + tgt_noise))
            xp_cur = np.concatenate((ja, self.obs_tgt[cond] + tgt_noise))
            u_cur = pol.act(ja, xp_cur, transformed_image, t, noise[t])
            # errors[t] = tgt_noise[:2]

            joint_angles[t] = ja.copy()
            actions[t] = u_cur.copy()
            joint_velocities[t] = v_cur.copy()
            ee_coords[t] = ee_coord.copy()
            XP[t] = xp_cur.copy()

            self.com.Pub_Angle(ja+u_cur)

            sampleLogger.info("Time step %d, X+U", t)
            sampleLogger.info(u_cur + ja)
            if data is not None:
                data[t, dja+dcod:] = u_cur
            print "Time step %d, X+U" % t
            print u_cur+ja

            time.sleep(1.0 / self.frequency)
            ee_coord = self.com.Get_position()  # read_coordnate
            ja = self.com.Get_angle()  # read_joint
            v_cur = self.com.Get_velocity()  # read_velocity
            if self._hyperparams['vision_on']:
                image = self.com.Get_image_RGB()

        # plt.figure(0)
        # ax = plt.subplot(1, 1, 1, projection='3d')
        # ax.scatter(self.obs_tgt[cond][0], self.obs_tgt[cond][1], self.obs_tgt[cond][2], s=50)
        # ax.plot(ee_coords[:, 0], ee_coords[:, 1], ee_coords[:, 2])
        # plt.show()
        # print 'Mean error:', np.mean(np.abs(errors), 0)
        # print 'Max error:', np.amax(np.abs(errors), 0)
        # print 'Std error:', np.std(np.abs(errors), 0)

        if self._hyperparams['test']:
            insert_pose = ee_coord + np.array([0, 0, -self._hyperparams['insert_offset'], 0, 0, 0])
            self.com.Pub_Position(insert_pose)
            time.sleep(self.sleepTime4Reset)

        sample.set(ACTION, actions)
        sample.set(RR_COORD6D, ee_coords)
        sample.set(JOINT_ANGLES, joint_angles)
        sample.set(JOINT_VELOCITIES, joint_velocities)
        sample.set(XPLUS, XP)
        sample.set(RGB_IMAGE, image_meta)

        self._samples[cond].append(sample)
        # raw_input()
        return sample
