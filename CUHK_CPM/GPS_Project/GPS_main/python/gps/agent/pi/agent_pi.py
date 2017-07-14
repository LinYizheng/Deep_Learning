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

        self.conditions = self._hyperparams['conditions']
        self.frequency = self._hyperparams['sampling_freq']
        self.sleepTime4Reset = self._hyperparams['sleep_time']

        for field in ('x0', 'ee_points_tgt', 'ee_obs_tgt'):
            self._hyperparams[field] = setup(self._hyperparams[field], self.conditions)
        self.sensor_dims = self._hyperparams['sensor_dims']

        # self.x0 & self.tgt are lists of initial states & target states for all conditions, aka, x0s & ee_tgts
        self.x0 = self._hyperparams['x0']
        self.tgt = self._hyperparams['ee_points_tgt']
        self.obs_tgt = self._hyperparams['ee_obs_tgt']
        self.obj_pose = self._hyperparams['obj_pose']

        self.robot = RobotControl()
        self.gripper = PneumaticGripper("mybot")
        self.model = Model()

        self.capacitor_names = ['capacitor%d' % condition for condition in range(self.conditions)]
        self._deploy_objs()
        self._test = False

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

        zx_time_1_s = time.time()
        self._reset_obj(condition)
        self._reset_robot(condition)

        ee_coord = self.robot.getPosition()[:4]  # read_coordinate
        ja = self.x0[condition]  # joint angle of initial

        # generate noise vectors
        noise = generate_noise(self.T, self.dU, self._hyperparams)
        print 'zx, noise.shape = {}'.format(noise.shape)
        sample = Sample(self)
        dim_ja = self.sensor_dims[JOINT_ANGLES]
        dim_u = self.sensor_dims[ACTION]
        dim_pos = self.sensor_dims[RR_COORD6D]

        joint_angles = np.zeros((self.T, dim_ja))
        actions = np.zeros((self.T, dim_u))
        ee_coords = np.zeros((self.T, dim_pos))
        XP = np.zeros((self.T, dim_ja + dim_pos))
        zx_time_1_e = time.time()
        print 'zx, time_1 = {}'.format(zx_time_1_e - zx_time_1_s)
        # zx, dim_ja == 4
        # zx, dim_u == 4
        # zx, dim_pos == 4
        # zx, self.T = 100

        # zx, joint_angles.shape = (100, 4)
        # zx, actions.shape = (100, 4)
        # zx, ee_coords.shape = (100, 4)
        # zx, XP.shape = (100, 8)

        for t in range(self.T):
            zx_one_step_s = time.time()
            sampleLogger.info("Condition: %d", condition)
            sampleLogger.info("Time step %d, joint angles:", t)
            sampleLogger.info(map(lambda x: '%.3f' % x, ja))
            sampleLogger.info("Time step %d, end-effector coordinates:", t)
            sampleLogger.info(map(lambda x: '%.3f' % x, ee_coord))

            if data is not None:
                # zx, sampleData.shape == (40, 3, 100, 12)
                # zx, data == sampleData[cond, i]
                # zx, data.shape == (100, 12)
                data[t, :dim_ja] = ja                       # zx, data[t, 1:4]<-init joint angles
                data[t, dim_ja:dim_ja+dim_pos] = ee_coord   # zx, data[t, 5:8]<-init ee pose coord
            print "Condition:", condition
            print "Time step %d, joint angles and end-effector coordinates:" % t
            print "cur: ", map(lambda x: '%.3f' % x, ja), map(lambda x: '%.3f' % x, ee_coord)
            print "tgt: ", self.tgt[condition], self.obs_tgt[condition]

            # when training, add noise to target for better generalization performance
            # tgt_noise = np.r_[np.random.randn(2) * 8, np.random.randn(2) * 0.1]
            # do not add this noise when testing
            zx_time_2_s = time.time()
            zx_time_2_1_s = time.time()
            tgt_noise = np.zeros(4)
            xp_cur = np.concatenate((ja, self.obs_tgt[condition] + tgt_noise))
            zx_time_2_1_e = time.time()
            print 'zx, time_2_1 = {}'.format(zx_time_2_1_e - zx_time_2_1_s)
            print 'zx, tgt_noise.shape = {}'.format(tgt_noise.shape)
            print 'zx, xp_cur.shape = {}'.format(xp_cur.shape)
            zx_time_2_2_s = time.time()
            u_cur = policy.act(ja, xp_cur, None, t, noise[t])
            zx_time_2_2_e = time.time()
            print 'zx, time_2_2 = {}'.format(zx_time_2_2_e - zx_time_2_2_s)
            print 'zx, u_cur.shape = {}'.format(u_cur.shape)
            zx_time_2_3_s = time.time()
            joint_angles[t] = ja[:]
            actions[t] = u_cur.copy()
            ee_coords[t] = ee_coord[:]
            XP[t] = xp_cur.copy()
            zx_time_2_3_e = time.time()
            print 'zx, time_2_3 = {}'.format(zx_time_2_3_e - zx_time_2_3_s)
            zx_time_2_e = time.time()
            print 'zx, time_2 = {}'.format(zx_time_2_e - zx_time_2_s)

            zx_time_3_s = time.time()
            self.robot.pubAngle(np.r_[ja+u_cur, [0, 0]])
            zx_time_3_e = time.time()
            print 'zx, time_3 = {}'.format(zx_time_3_e - zx_time_3_s)

            sampleLogger.info("Time step %d, X+U", t)
            sampleLogger.info(u_cur + ja)
            zx_time_4_1_s = time.time()
            if data is not None:
                data[t, dim_ja+dim_pos:] = u_cur # zx, data[t, 9:12]<-action
            print "Time step %d, X+U" % t
            print u_cur+ja

            time.sleep(1.0 / self.frequency)
            print 'zx, sleep {}'.format(1.0/self.frequency)
            zx_time_4_1_e = time.time()
            print 'zx, time_4_1 = {}'.format(zx_time_4_1_e - zx_time_4_1_s)
            zx_time_4_2_s = time.time()
            ee_coord = self.robot.getPosition()[:4]  # read_coordnate
            ja = self.robot.getAngle()[:4]  # read_joint
            zx_time_4_2_e = time.time()
            print 'zx, time_4_2 = {}'.format(zx_time_4_2_e - zx_time_4_2_s)
            zx_one_step_e = time.time()
            print 'zx, one_step = {}'.format(zx_one_step_e - zx_one_step_s)

        if self.test:
            insert_pose = ee_coord + np.array([0, 0, -self._hyperparams['insert_offset'], 0])
            self.gripper.catch()
            self.robot.pubPosition('Go', np.r_[insert_pose, [0, 0]])
            if self._wait_for_catch(20):
                self.robot.pubPosition('Go', np.r_[ee_coord, [0, 0]], wait=True)
                self.robot.pubPosition('Go', np.array([200, 200, 100, 0, 0, 0]), wait=True)
                self.model.ControlModelStatic('capacitor%d' % condition, False)
                self.gripper.release()
                time.sleep(0.5)
                self.model.ControlModelStatic('capacitor%d' % condition, True)
            else:
                self.gripper.release()
                self.robot.pubPosition('Go', np.r_[ee_coord, [0, 0]], wait=True)

        sample.set(ACTION, actions)
        sample.set(RR_COORD6D, ee_coords)
        sample.set(JOINT_ANGLES, joint_angles)
        sample.set(XPLUS, XP)

        self._samples[condition].append(sample)
        return sample

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, value):
        self._test = value

    def _wait_for_catch(self, N):
        """
        :param N: total wait time = N * 0.1 seconds
        :return: True if catch else (timeout) False
        """
        t = 0
        while t < N:
            if self.gripper.getGripperState():
                return True
            time.sleep(0.1)
            t += 1
        return False

    def _deploy_objs(self):
        for m in range(self.conditions):
            if not self.model.addGazeboModel(self.capacitor_names[m], 'capacity', init_pos=self.obj_pose[m]):
                self._reset_obj(m)
                continue
            self.model.ControlModelStatic(self.capacitor_names[m], True)

    def _reset_obj(self, m):
        self.model.setLinkPos(self.capacitor_names[m] + '::base_link', self.obj_pose[m])

    def _reset_robot(self, condition):
        baseLogger.info("Reset to initial state...")
        self.robot.pubAngle(np.r_[self.x0[condition], [0, 0]])
        time.sleep(self.sleepTime4Reset)

