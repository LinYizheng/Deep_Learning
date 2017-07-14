""" This file defines an agent for the PR2 ROS environment. """

import time
import numpy as np
import logging

from gps.agent.agent import Agent
import gps.gui.shm_module as shm
from gps.agent.agent_utils import generate_noise, setup
from gps.sample.sample import Sample

from gps.proto.gps_pb2 import ACTION, RR_COORD6D, XPLUS, JOINT_ANGLES, JOINT_VELOCITIES
from gps.utility.general_utils import check_shape, save_hyperparams, print_hyperparams
from gps.algorithm.policy.caffe_policy import CaffePolicy
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy

baseLogger = logging.getLogger('base')
sampleLogger = logging.getLogger('sample')

class AgentSHM(Agent):

    def __init__(self, hyperparams=None):
        """
        Initialize agent.
        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node: Whether or not to initialize a new ROS node.
        """
        print '\ninstantiate AgentSHM'
        config = hyperparams
        Agent.__init__(self, config)

        conditions = self._hyperparams['conditions']
        self.frequency = 16
        self.x0 = []
        for field in ('x0', 'ee_points_tgt', 'ee_obs_tgt'):
            self._hyperparams[field] = setup(self._hyperparams[field],
                                             conditions)

        # initialized in agent.py
        # self.x_data_types = self._hyperparams['state_include']
        # self.obs_data_types = self._hyperparams['obs_include']

        # self.x0 & self.tgt are lists of initial states & target states for all conditions, aka, x0s & ee_tgts
        self.x0 = self._hyperparams['x0']
        self.tgt = self._hyperparams['ee_points_tgt']
        self.obs_tgt = self._hyperparams['ee_obs_tgt']

        self.use_tf = False
        self.observations_stale = True

        # shm_size & shm_buf is for communication with Gazebo via shared memory
        self.shm_size = 768
        self.shm_buf = shm.shm_link("/home/pi/gps_rr/src/pi_six_asix/pi_control/src/shm.bin", self.shm_size)
        print '\ninstantiate AgentSHM done!'

        np.set_printoptions(suppress=True, formatter={'float': '{: 0.2f}'.format})

    def sample(self, pol, cond, data=None, verbose=False, save=True):

        x_cur = self.x0[cond]
        check_shape(x_cur, (self.dX,))

        if isinstance(pol, LinearGaussianPolicy):
            sampleLogger.info("Linear Gaussian Policy")
            print "Linear Gaussian Policy"
        elif isinstance(pol, CaffePolicy):
            sampleLogger.info("Caffe Policy")
            print "Caffe Policy"

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

        baseLogger.info("Reset to initial state.")
        # reset the robot
        # shm.shm_write(self.shm_buf, 1, self.shm_size, 1, 1, 1)  # go_flag
        # shm.shm_write(self.shm_buf, 1, self.shm_size, 0, 1, 1)  # coord_mode
        # shm.shm_write(self.shm_buf, shm.float2uchar(x_cur, 24), self.shm_size, 27, 24, 1)  # coord data
        shm.shm_write(self.shm_buf, 0, self.shm_size, 1, 1, 1)  # ja_mode
        shm.shm_write(self.shm_buf, shm.float2uchar(x_cur, 24), self.shm_size, 2, 24, 1)  # joint angle data
        time.sleep(self.sleepTime4Reset)
        eepos = shm.uchar2float(shm.shm_read(self.shm_buf, self.shm_size, 77, 24, 0), 24)  # read_coordnate
        eepos = np.array(eepos[:], dtype=np.float).reshape(self.dX, )

        noise = generate_noise(self.T, self.dU, self._hyperparams)

        sample = Sample(self)
        X = np.zeros((self.T, self.dX))
        U = np.zeros((self.T, self.dU))
        XP = np.zeros((self.T, self.dX+6))
        # eepos = np.zeros((self.T, self.dX))
        V = np.zeros((self.T, self.dU))
        v_cur = np.zeros(self.dU)
        for t in range(self.T):
            sampleLogger.info("Time step %d, X", t)
            sampleLogger.info(x_cur)
            sampleLogger.info(eepos)
            if data is not None:
                data[t, :self.dX] = x_cur
                data[t, self.dX:self.dX*2] = eepos
            print "Time step %d, X" % t
            print "cur: ", x_cur, eepos
            print "tgt: ", self.tgt[cond], self.obs_tgt[cond]

            xp_cur = np.concatenate((x_cur, self.obs_tgt[cond]))
            u_cur = pol.act(x_cur, xp_cur.copy(), t, noise[t])

            X[t] = x_cur.reshape(self.dX,)
            U[t] = u_cur.reshape(self.dU,)
            V[t] = v_cur.reshape(self.dU, )
            XP[t] = xp_cur.reshape((self.dX+6,))
            # every time send coordnate need to write to go_flag, coordnate_flag, and coordnatedata.
            # shm.shm_write(self.shm_buf, 1, self.shm_size, 1, 1, 1)  # go_flag
            # shm.shm_write(self.shm_buf, 1, self.shm_size, 0, 1, 1)  # coord_mode
            # shm.shm_write(self.shm_buf, shm.float2uchar(x_cur, 24), self.shm_size, 27, 24, 1)  # coord data
            shm.shm_write(self.shm_buf, 0, self.shm_size, 1, 1, 1)  # ja_mode
            shm.shm_write(self.shm_buf, shm.float2uchar(x_cur+u_cur, 24), self.shm_size, 2, 24, 1)  # joint angle data
            sampleLogger.info("Time step %d, X+U", t)
            sampleLogger.info(u_cur + x_cur)
            if data is not None:
                data[t, self.dX*2:] = u_cur
            print "Time step %d, X+U" % t
            print u_cur+x_cur

            time.sleep(1.0 / self.frequency)
            eepos = shm.uchar2float(shm.shm_read(self.shm_buf, self.shm_size, 77, 24, 0), 24)  # read_coordnate
            eepos = np.array(eepos[:], dtype=np.float).reshape(self.dX, )
            x_cur = shm.uchar2float(shm.shm_read(self.shm_buf, self.shm_size, 52, 24, 0), 24)  # read_joint
            x_cur = np.array(x_cur[:], dtype=np.float).reshape(self.dX,)
            v_cur = shm.uchar2float(shm.shm_read(self.shm_buf, self.shm_size, 103, 24, 0), 24)  # read_velocity
            v_cur = np.array(v_cur[:], dtype=np.float).reshape(self.dU, )

        sample.set(ACTION, U)
        sample.set(RR_COORD6D, X)
        sample.set(JOINT_VELOCITIES, V)
        sample.set(XPLUS, XP)

        self._samples[cond].append(sample)