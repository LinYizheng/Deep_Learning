""" This file defines the main object that runs experiments. """

import os
import sys
import imp
import copy
import time
import cPickle
import logging
import argparse
import threading
import numpy as np

# import matplotlib as mpl
# mpl.use('Qt4Agg')
# import matplotlib.pyplot as plt

# Add gps/python to path so that imports work.
_file_path = os.path.abspath(os.path.dirname(__file__))
_root_path = os.path.dirname(_file_path)
sys.path.append(_root_path)

from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
# from gps.sample.sample import Sample
# from gps.proto.gps_pb2 import ACTION, RR_COORD6D, XPLUS, JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE

np.set_printoptions(suppress=True, formatter={'float': '{: 0.2f}'.format})

# initialize logger
baseLogger = logging.getLogger('base')
baseLogger.setLevel(logging.INFO)
sampleLogger = logging.getLogger('sample')
sampleLogger.setLevel(logging.DEBUG)
dynamicsLogger = logging.getLogger('dynamics')
dynamicsLogger.setLevel(logging.DEBUG)
lgpolicyLogger = logging.getLogger('lgpolicy')
lgpolicyLogger.setLevel(logging.DEBUG)
costLogger = logging.getLogger('cost')
costLogger.setLevel(logging.DEBUG)
debugger = logging.getLogger('debugger')
debugger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
basefmt = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
ch.setFormatter(basefmt)
baseLogger.addHandler(ch)


class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config):
        # config includes almost everything in hyperparams.py
        self._hyperparams = config

        # number of conditions
        self._conditions = config['common']['conditions']
        # a list [0,1,2,...,#conditions-1]
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams=config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']

        # initialize an agent object
        # AgentPIROS(config['agent'])
        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger()
        self.gui = GPSTrainingGUI(config['common']) if config['gui_on'] else None

        # replace the content in algorithm['agent'] with the agent object
        config['algorithm']['agent'] = self.agent
        # self.algorithm = AlgorithmBADMM(config['algorithm'])
        self.algorithm = config['algorithm']['type'](config['algorithm'])

    def run(self, itr_load=None, testing=False):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        itr_start = self._initialize(itr_load)

        if testing:
            print '='*5, ' Testing ', '=' * 5
            # call reset to make sure robot doesn't crash
            self.agent.reset(self._train_idx[0])

            while True:

                from gps.utility.general_utils import get_ee_points

                self.gui.process_test_mode()
                EE_POINTS = np.array([[0.02, -0.025, 0.05], [0.02, -0.025, 0.05],
                                      [0.02, 0.05, 0.0]])

                ######## setting Testing input: ee_pos and ee_rot
                '''
                # 05192016_140230.experiment
                #>>> a['trial_arm_1_target_ee_pos']
                ee_pos = np.array([[ 0.93796071,  0.0743069 , -0.1501604 ]])
                #>>> a['trial_arm_1_target_ee_rot']
                ee_rot = np.array([[[ 0.79615081, -0.53677305, -0.27931807],
                                    [-0.60414234, -0.67920041, -0.41677192],
                                    [ 0.03399899,  0.50056117, -0.86503328]]])
                '''
                '''
                #>>> a['trial_arm_1_target_ee_pos']
                ee_pos = np.array([[ 0.83093201, -0.05959223, -0.21891968]])
                #>>> a['trial_arm_1_target_ee_rot']
                ee_rot = np.array([[[ 0.64001542, -0.2471554 ,  0.72752626],
                                    [-0.6068464 ,  0.41818058,  0.67591601],
                                    [-0.47129365, -0.87409336,  0.11765665]]])
                '''
                # 05162016_145948.experiment
                ee_pos = np.array([[   0.811,  0.058, -0.147]])
                ee_rot = np.array([[[  0.438, -0.881,  -0.18],
                                    [  -0.84, -0.472,  0.268],
                                    [ -0.321,  0.034, -0.947]]])



                tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos, ee_rot).T)
                print 'Target points:', tgt
                ee_pts = [
                    [0.02, -0.025, 0.05, 0.02, -0.025, 0.05, 0.02, 0.05, 0.0],
                    tgt,
                ]
                sample = self.agent.sample(
                    self.algorithm.policy_opt.policy, 1, ee_pts=ee_pts,
                    verbose=True, save=False)
                print 'open and close gripper'
                self.gc.open("l")
                self.gc.close("l")

                if self.gui:
                    self.gui.set_status_text('Updating testing trajectory...')
                    self.gui.update_test_traj(sample)
            self._end()

            print '='*5, ' Done testing ', '=' * 5
            return

        # default #iterations == 10
        # zx, time_total.shape == (12,)
        # zx, time_sample.shape == (12,)
        # zx, time_train.shape == (12,)
        # zx, itr_start == 0, self._hyperparams[iteration] = 12
        time_total = np.zeros(self._hyperparams['iterations'])
        time_sample = np.zeros(self._hyperparams['iterations'])
        time_train = np.zeros(self._hyperparams['iterations'])
        for itr in range(itr_start, self._hyperparams['iterations']):
            time_total_start = time.time()

            # zx, sampleData.shape = (5, 3, 100, 18)
            # zx, pred_traj.shape = (5, 100, 12)
            sampleData = np.zeros(
                (self._conditions, self._hyperparams['num_samples'], self._hyperparams['agent']['T'], 18)) # joint angles: 6, ee pose: 6, actions: 6
            pred_traj = np.zeros(
                (self._conditions, self._hyperparams['agent']['T'], 12)) # joint angles: 6, actions: 6
            time_sample_start = time.time()

            # zx, zx, self._train_idx = [0, 1, 2, 3, 4]
            for cond in self._train_idx:
                # deal with each condition
                # default num_samples = 5
                # zx, self._hyperparams['num_samples'] = 3
                for i in range(self._hyperparams['num_samples']):
                    # take sample for cond and i
                    self._take_sample(itr, cond, i, sampleData[cond, i])
            time_sample_end = time.time()
            time_sample[itr] = time_sample_end - time_sample_start

            traj_sample_lists = [
                self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                for cond in self._train_idx
            ]

            for cond in self._train_idx:
                for t in range(self.algorithm.T):
                    lgpolicyLogger.info("Iteration: %d, condition: %d, time step: %d", itr, cond, t)
                    lgpolicyLogger.info("K:")
                    lgpolicyLogger.info(self.algorithm.cur[cond].traj_distr.K[t])
                    lgpolicyLogger.info("k:")
                    lgpolicyLogger.info(self.algorithm.cur[cond].traj_distr.k[t])
                    lgpolicyLogger.info("Cov:")
                    lgpolicyLogger.info(self.algorithm.cur[cond].traj_distr.pol_covar[t])
            lgpolicyLogger.info("Iteration %d finished.\n", itr)

            debugger.info("Iteration: %d", itr)
            costLogger.info("Iteration: %d", itr)
            costData = []
            time_train[itr] = self._take_iteration(itr, traj_sample_lists, costData, pred_traj)

            # pol_sample_lists = self._take_policy_samples()

            print 'logging data...'
            pol_sample_lists = None
            self._log_data(itr, traj_sample_lists, pol_sample_lists)

            self.data_logger.pickle(self._data_files_dir + ("sample_itr%02d.pkl" % itr), copy.copy(sampleData))
            self.data_logger.pickle(self._data_files_dir + ("cost_itr%02d.pkl" % itr), copy.copy(costData))
            self.data_logger.pickle(self._data_files_dir + ("pred_traj_itr%02d.pkl" % itr), copy.copy(pred_traj))

            time_total_end = time.time()
            time_total[itr] = time_total_end - time_total_start
            cPickle.dump(dict(total=time_total, sample=time_sample, train=time_train),
                         open(self._data_files_dir + "average_time.pkl", 'wb'))
            # cPickle.dump(self.algorithm.policy_opt.loss, open(self._data_files_dir + "loss.pkl", 'wb'),
            #              protocol=cPickle.HIGHEST_PROTOCOL)
            np.savetxt(self._data_files_dir + "average_time.txt", np.c_[time_total, time_sample, time_train],
                       fmt='%8.2f', delimiter='\t')

        self._end()

    def test_policy(self, itr, N):
        """
        Take N policy samples of the algorithm state at iteration itr,
        for testing the policy to see how it is behaving.
        (Called directly from the command line --policy flag).
        Args:
            itr: the iteration from which to take policy samples
            N: the number of policy samples to take
        Returns: None
        """
        algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.algorithm = self.data_logger.unpickle(algorithm_file)
        if self.algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1) # called instead of sys.exit(), since t
        traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            ('traj_sample_itr_%02d.pkl' % itr))

        pol_sample_lists = self._take_policy_samples(N)
        self.data_logger.pickle(
            self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
            copy.copy(pol_sample_lists)
        )

        if self.gui:
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.set_status_text(('Took %d policy sample(s) from ' +
                'algorithm state at iteration %d.\n' +
                'Saved to: data_files/pol_sample_itr_%02d.pkl.\n') % (N, itr, itr))

    def _initialize(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None:
            if self.gui:
                self.gui.set_status_text('Press \'collect\' to begin.')
            return 0
        else:
            algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load
            self.algorithm = self.data_logger.unpickle(algorithm_file)
            if self.algorithm is None:
                print("Error: cannot find '%s.'" % algorithm_file)
                os._exit(1) # called instead of sys.exit(), since this is in a thread

            if self.gui:
                traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('traj_sample_itr_%02d.pkl' % itr_load))
                pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('pol_sample_itr_%02d.pkl' % itr_load))
                self.gui.update(itr_load, self.algorithm, self.agent,
                    traj_sample_lists, pol_sample_lists)
                self.gui.set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' +
                    'Press \'collect\' to begin.') % itr_load)
            return itr_load + 1

    def _take_sample(self, itr, cond, i, data):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """

        # traj_distr is an instance of LinGaussPolicy
        pol = self.algorithm.cur[cond].traj_distr
        baseLogger.info("Iteration %d condition %d sample %d", itr, cond, i)
        sampleLogger.info("Iteration %d condition %d sample %d", itr, cond, i)

        # zx, self.gui = None
        if self.gui:
            self.gui.set_image_overlays(cond)   # Must call for each new cond.
            redo = True
            while redo:
                while self.gui.mode in ('wait', 'request', 'process'):
                    if self.gui.mode in ('wait', 'process'):
                        time.sleep(0.01)
                        continue
                    # 'request' mode.
                    if self.gui.request == 'reset':
                        try:
                            self.agent.reset(cond)
                        except NotImplementedError:
                            self.gui.err_msg = 'Agent reset unimplemented.'
                    elif self.gui.request == 'fail':
                        self.gui.err_msg = 'Cannot fail before sampling.'
                    self.gui.process_mode()  # Complete request.

                self.gui.set_status_text(
                    'Sampling: iteration %d, condition %d, sample %d.' %
                    (itr, cond, i)
                )
                self.agent.sample(
                    pol, cond,
                    verbose=(i < self._hyperparams['verbose_trials'])
                )

                if self.gui.mode == 'request' and self.gui.request == 'fail':
                    redo = True
                    self.gui.process_mode()
                    self.agent.delete_last_sample(cond)
                else:
                    redo = False
        else:
            # zx, enter branche
            print 'zx, before'
            self.agent.sample(pol, itr, cond, i, data)

        sampleLogger.info("Sample finished at: Iteration %d condition %d sample %d\n", itr, cond, i)


    def _take_iteration(self, itr, sample_lists, data, pred_traj):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Calculating.')
            self.gui.start_display_calculating()
        return self.algorithm.iteration(sample_lists, data, pred_traj)
        # if self.gui:
        #     self.gui.stop_display_calculating()

    def _take_policy_samples(self, N=None):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """
        if 'verbose_policy_trials' not in self._hyperparams:
            return None
        if not N:
            N = self._hyperparams['verbose_policy_trials']
        if self.gui:
            self.gui.set_status_text('Taking policy samples.')
        pol_samples = [[None for _ in range(N)] for _ in range(self._conditions)]
        for cond in range(len(self._test_idx)):
        # for cond in [0, 4, 20, 24, 12]:
            for i in range(N):
                # change policy here for testing
                # NN policy
                pol_samples[cond][i] = self.agent.sample(
                    self.algorithm.policy_opt.policy, 99, cond, i
                )
                # LG policy
                # pol_samples[cond][i] = self.agent.sample(
                #     self.algorithm.cur[cond].traj_distr, 199, cond, i
                # )

            # samples = np.zeros((0, N, self.agent.sensor_dims[RR_COORD6D]))
            # for c in range(cond + 1):
            #     samples_cond = np.array([sample.get(RR_COORD6D, self.agent.T - 1) for sample in pol_samples[c]])
            #     samples = np.concatenate((samples, samples_cond.reshape(1, N, self.agent.sensor_dims[RR_COORD6D])))
            # print samples
            # print samples.shape
            # obs_tgt = np.array(self.agent.obs_tgt)
            # plt.figure(0)
            # plt.scatter(obs_tgt[:, 0], obs_tgt[:, 1])
            # plt.scatter(obs_tgt[cond, 0], obs_tgt[cond, 1], c='g', s=40)
            # plt.scatter(samples.reshape(-1, self.agent.sensor_dims[RR_COORD6D])[:, 0],
            #             samples.reshape(-1, self.agent.sensor_dims[RR_COORD6D])[:, 1], c='grey', marker='x', s=40)
            # print samples[cond]
            # print samples[cond, :, 0]
            # print samples[cond, :, 1]
            # plt.scatter(samples[cond, :, 0], samples[cond, :, 1], c='r', marker='x', s=40)
            # plt.xlabel('x-axis')
            # plt.ylabel('y-axis')
            # plt.show()
        return [SampleList(samples) for samples in pol_samples]

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Logging data and updating GUI.')
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.save_figure(
                self._data_files_dir + ('figure_itr_%02d.png' % itr)
            )
        if 'no_sample_logging' in self._hyperparams['common']:
            return
        self.data_logger.pickle(
            self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
            copy.copy(self.algorithm)
        )
        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )
        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

    def run_test(self, model_name, target_pos):
        """
        Test a model.
        Args:
            model_name: linear/nonlinear model name.
            target_pos: x, y, z
        Returns: Trajectory
        """
        # trajectory output

        # plot and save



    def _end(self):
        """ Finish running and exit. """
        if self.gui:
            self.gui.set_status_text('Training complete.')
            self.gui.end_mode()


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Train/Test the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM only)')
    parser.add_argument('-s', '--test', type=int,
                        help='testing the model')

    args = parser.parse_args()
    # experiment name
    exp_name = args.experiment
    # from which iteration to resume
    resume_training_itr = args.resume
    # test the non-linear policy, sample number is N
    # user define which iteration to test in branch 'elif test_policy_N' in this function
    test_policy_N = args.policy
    testing_model_name = args.test

    exp_dir = 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'
    testing_itr = args.test

    if args.new: #-n
        from shutil import copy

        # make experiment dir
        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir))
        os.makedirs(exp_dir)
        os.makedirs(exp_dir + 'log/')
        os.makedirs(exp_dir + 'data_files/')

        # retrieve hyparameters and targets from previous experiment
        # previous experiment name is recorded in the hidden file .previous_experiment
        prev_exp_file = '.previous_experiment'
        prev_exp_dir = None
        try:
            with open(prev_exp_file, 'r') as f:
                prev_exp_dir = f.readline()  ##read previous
            copy(prev_exp_dir + 'hyperparams.py', exp_dir)
            if os.path.exists(prev_exp_dir + 'targets.npz'):
                copy(prev_exp_dir + 'targets.npz', exp_dir)
        except IOError as e:
            print e
            with open(hyperparams_file, 'w') as f:
                f.write('# To get started, copy over hyperparams from another experiment.\n' +
                        '# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.')
        # record the current experiment name in .previous_experiment
        with open(prev_exp_file, 'w') as f:
            f.write(exp_dir)

        exit_msg = ("Experiment '%s' created.\nhyperparams file: '%s'" %
                    (exp_name, hyperparams_file))
        if prev_exp_dir and os.path.exists(prev_exp_dir):
            exit_msg += "\ncopied from     : '%shyperparams.py'" % prev_exp_dir
        sys.exit(exit_msg)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))

    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    # initialize logger
    basefh = logging.FileHandler(exp_dir + 'log/base.log')
    samplefh = logging.FileHandler(exp_dir + 'log/sample.log')
    dynamicsfh = logging.FileHandler(exp_dir + 'log/dynamics.log')
    lgpolicyfh = logging.FileHandler(exp_dir + 'log/lgpolicy.log')
    costfh = logging.FileHandler(exp_dir + 'log/cost.log')
    debuggerfh = logging.FileHandler(exp_dir + 'log/debugger.log')

    basefh.setFormatter(basefmt)
    baseLogger.addHandler(basefh)

    simplefmt = logging.Formatter('%(message)s')
    samplefh.setFormatter(simplefmt)
    dynamicsfh.setFormatter(simplefmt)
    lgpolicyfh.setFormatter(simplefmt)
    costfh.setFormatter(simplefmt)
    debuggerfh.setFormatter(simplefmt)

    sampleLogger.addHandler(samplefh)
    dynamicsLogger.addHandler(dynamicsfh)
    lgpolicyLogger.addHandler(lgpolicyfh)
    costLogger.addHandler(costfh)
    debugger.addHandler(debuggerfh)

    baseLogger.info("Experiment: %s", exp_name)
    baseLogger.info("Hyper parameters path: %s", hyperparams_file)

    if args.targetsetup:   ##-t
        import wx
        from gps.gui.pi_six_axis_gui import ControlPanel
        baseLogger.info("Setting up initial and target...")
        npzfile = exp_dir + "targets.npz"
        ex = wx.App()
        ControlPanel(None, npzfile)
        ex.MainLoop()

    elif test_policy_N:  ##-p
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        random.seed(0)
        np.random.seed(0)
        data_files_dir = exp_dir + 'data_files/'
        data_filenames = os.listdir(data_files_dir)
        algorithm_prefix = 'algorithm_itr_'
        algorithm_filenames = [f for f in data_filenames if f.startswith(algorithm_prefix)]
        current_algorithm = sorted(algorithm_filenames, reverse=True)[0]

        # by default it will get the newest iteration result
        # you can specify yourself
        current_itr = int(current_algorithm[len(algorithm_prefix):len(algorithm_prefix)+2])
        # current_itr = 5

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            test_policy = threading.Thread(
                target=lambda: gps.test_policy(itr=current_itr, N=test_policy_N)
            )
            test_policy.daemon = True
            test_policy.start()

            plt.ioff()
            plt.show()
        else:
            gps.test_policy(itr=current_itr, N=test_policy_N)

    elif args.test:  ##-s
        print 'Testing iteration = ', testing_itr
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        random.seed(0)
        np.random.seed(0)
        print ("-s\n")
        # instruct GUI to do the right thing
        hyperparams.config['common']['test_mode'] = True

        # start grippers for testing purpose
        assert start_controllers(['l_gripper_controller','r_gripper_controller']) == True, 'fail to start grippers'

        gps = GPSMain(hyperparams.config)

        # initialize gripper controller
        gps.gc = GripperController()



        if hyperparams.config['gui_on']:
            run_gps = threading.Thread(
                target=lambda: gps.run(itr_load=testing_itr, testing=True)
            )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            gps.run(itr_load=testing_itr, testing=True)

    else:  ## -r
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        random.seed(0)
        np.random.seed(0)
        baseLogger.info("Starting training from iteration 0 ...")
        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            run_gps = threading.Thread(
                target=lambda: gps.run(itr_load=resume_training_itr)
            )
            run_gps.daemon = True
            run_gps.start()


            plt.ioff()
            plt.show()
        else:
            gps.run(itr_load=resume_training_itr)


if __name__ == "__main__":
    main()
