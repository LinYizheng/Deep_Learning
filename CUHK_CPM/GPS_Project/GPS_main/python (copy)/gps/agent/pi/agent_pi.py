import time
import logging
import numpy as np

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.sample.sample import Sample
from gps.GazeboInterface import RobotControl, PneumaticGripper, Model

from gps.proto.gps_pb2 import ACTION, RR_COORD6D, XPLUS, JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE
from gps.algorithm.policy.caffe_policy import CaffePolicy
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy

baseLogger = logging.getLogger('base')
sampleLogger = logging.getLogger('sample')


class AgentPI(Agent):

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
        # zx, these four data read from 'targets.npz' file.
        # zx, x0 == initial, tgt == target, obs_tgt == obs_tgt, obj_pose == obj_pose
        self.x0 = self._hyperparams['x0']
        self.tgt = self._hyperparams['ee_points_tgt']
        self.obs_tgt = self._hyperparams['ee_obs_tgt']
        self.obj_pose = self._hyperparams['obj_pose']

        self.robot = RobotControl()
        self.gripper = PneumaticGripper("mybot")
        self.model = Model()

    def sample(self, policy, iteration, condition, sample, data=None):
        if isinstance(policy, LinearGaussianPolicy):
            pol_string = "Linear Gaussian Policy"
        elif isinstance(policy, CaffePolicy):
            pol_string = "Caffe Policy"

        sampleLogger.info(pol_string)
        print pol_string

        baseLogger.info("Initial state:")
        baseLogger.info(self.x0[condition])
        baseLogger.info("Target state:")
        baseLogger.info(self.tgt[condition])
        baseLogger.info(self.obs_tgt[condition])
        sampleLogger.info("Initial state:")
        sampleLogger.info(self.x0[condition])
        sampleLogger.info("Target state:")
        sampleLogger.info(self.tgt[condition])
        sampleLogger.info(self.obs_tgt[condition])

        # zx, put the capacitor object to the specific location of the desktop,
        # according to different obj_pose coordinates.
        self._set_obj(condition)
        self._reset_robot(condition)

        ee_coord = self.robot.getPosition()[:]  # read_coordinate
        ja = self.x0[condition]  # joint angle of initial

        # generate noise vectors
        noise = generate_noise(self.T, self.dU, self._hyperparams)

        sample = Sample(self)
        dim_ja = self.sensor_dims[JOINT_ANGLES]
        dim_u = self.sensor_dims[ACTION]
        dim_pos = self.sensor_dims[RR_COORD6D]

        joint_angles = np.zeros((self.T, dim_ja))
        actions = np.zeros((self.T, dim_u))
        ee_coords = np.zeros((self.T, dim_pos))
        XP = np.zeros((self.T, dim_ja + dim_pos))

        for t in range(self.T):
            sampleLogger.info("Condition: %d", condition)
            sampleLogger.info("Time step %d, joint angles:", t)
            sampleLogger.info(map(lambda x: '%.3f' % x, ja))
            sampleLogger.info("Time step %d, end-effector coordinates:", t)
            sampleLogger.info(map(lambda x: '%.3f' % x, ee_coord))

            if data is not None:
                data[t, :dim_ja] = ja
                data[t, dim_ja:dim_ja+dim_pos] = ee_coord
            print "Condition:", condition
            print "Time step %d, joint angles and end-effector coordinates:" % t
            print "cur: ", map(lambda x: '%.3f' % x, ja), map(lambda x: '%.3f' % x, ee_coord)
            print "tgt: ", self.tgt[condition], self.obs_tgt[condition]

            # tgt_noise = np.r_[np.random.randn(3), np.random.randn(3) * 0.1]
            tgt_noise = np.zeros(6)
            xp_cur = np.concatenate((ja, self.obs_tgt[condition] + tgt_noise))
            u_cur = policy.act(ja, xp_cur, None, t, noise[t])

            joint_angles[t] = ja[:]
            actions[t] = u_cur.copy()
            ee_coords[t] = ee_coord[:]
            XP[t] = xp_cur.copy()

            self.robot.pubAngle(ja+u_cur)

            sampleLogger.info("Time step %d, X+U", t)
            sampleLogger.info(u_cur + ja)
            if data is not None:
                data[t, dim_ja+dim_pos:] = u_cur
            print "Time step %d, X+U" % t
            print u_cur+ja

            time.sleep(1.0 / self.frequency)
            ee_coord = self.robot.getPosition()[:]  # read_coordnate
            ja = self.robot.getAngle()[:]  # read_joint

        sample.set(ACTION, actions)
        sample.set(RR_COORD6D, ee_coords)
        sample.set(JOINT_ANGLES, joint_angles)
        sample.set(XPLUS, XP)

        self._samples[condition].append(sample)
        return sample

    def _set_obj(self, condition):
        # if brick
        # if capacitor
        capacitor_name = 'capacitor%d' % condition
        if not self.model.addGazeboModel(capacitor_name, 'capacity', init_pos=self.obj_pose[condition]):
            self.model.setLinkPos(capacitor_name + '::base_link', self.obj_pose[condition])
        self.model.ControlModelStatic(capacitor_name, True)

    def _reset_robot(self, condition):
        baseLogger.info("Reset to initial state...")
        self.robot.pubAngle(self.x0[condition])
        time.sleep(self.sleepTime4Reset)

