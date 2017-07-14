""" Hyperparameters for PR2 policy optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.pi.agent_pi_ros import AgentPIROS
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_coord6d import CostRR6D
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_lqr_rr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.gui.util import load_from_npz
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE, RR_COORD6D, XPLUS, RGB_IMAGE
from gps.gui.config import generate_experiment_info


EE_POINTS = np.array([[0.02, -0.025, 0.05], [0.02, -0.025, 0.05],
                      [0.02, 0.05, 0.0]])

SENSOR_DIMS = {
    RR_COORD6D: 6,
    JOINT_ANGLES: 6,
    JOINT_VELOCITIES: 6,
    ACTION: 6,
    XPLUS: 12,
    RGB_IMAGE: 1
}

# PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])
PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/TrajOpt.experiment/'

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 2,
}

test = False
if test:
    common['target_filename'] = EXP_DIR + 'test.npz'

x0s = []
ee_tgts = []
obs_tgts = []
brick_locs = []
reset_conditions = []

# Set up each condition.
print common['target_filename']
for i in xrange(common['conditions']):
    rr_coord6D_init = load_from_npz(common['target_filename'], 'initial'+str(i))
    rr_coord6D_tgt = load_from_npz(common['target_filename'], 'target'+str(i))
    rr_obs_tgt = load_from_npz(common['target_filename'], 'obs_tgt'+str(i))
    brick_loc = load_from_npz(common['target_filename'], 'brick'+str(i))

    x0s.append(rr_coord6D_init)
    ee_tgts.append(rr_coord6D_tgt)
    obs_tgts.append(rr_obs_tgt)
    brick_locs.append(brick_loc)

# for x0 in x0s:
#     x0 += np.random.randn(SENSOR_DIMS[JOINT_VELOCITIES]) * 0.5

data_loaded_check = False
if data_loaded_check:
    import os
    print 'data loaded check'
    print 'x0s'
    print x0s
    print 'ee_tgts'
    print ee_tgts


if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentPIROS,
    'dt': 0.02,
    'conditions': common['conditions'],
    'T': 50,
    'x0': x0s,
    'ee_points_tgt': ee_tgts,
    'ee_obs_tgt': obs_tgts,
    'brick_loc': brick_locs,
    'reset_conditions': reset_conditions,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES],
    'end_effector_points': EE_POINTS,
    'obs_include': [JOINT_ANGLES],
    'meta_include': [RGB_IMAGE]
}

algorithm = {
    # 'type': AlgorithmBADMM,
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 20,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-1]),
    'policy_dual_rate': 0.1,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 50.0,
    'init_pol_wt': 0.0001,
    'min_step_mult': 0.01,
    'max_step_mult': 5.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'exp_step_increase': 2.0,
    'exp_step_decrease': 0.5,
    'exp_step_upper': 0.5,
    'exp_step_lower': 1.0,
    'max_policy_samples': 6,
    'policy_sample_mode': 'add',
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    # 'type': init_lqr_rr,
    #'init_gains':  1.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 0.5,
    'stiffness_vel': 0.25,
    # 'final_weight': 50.0,
    'final_weight': 1,
    'dt': agent['dt'],
    'T': agent['T'],
    'tgt': ee_tgts
}

coord6d_cost = {
    'type': CostRR6D
}

torque_cost = {
    'type': CostAction,
#    'wu': 5e-3 / PR2_GAINS,
    'wu': np.ones(SENSOR_DIMS[ACTION]),
}

state_cost = {
    'type': CostState,
    'ramp_option': RAMP_QUADRATIC,
    'l1': 1.0,
    'l2': 0.3,
    'alpha': 1e-5,
    'wp_final_multiplier': 1.0
}

fk_cost1 = {
    'type': CostFK,
    # Target end effector is subtracted out of EE_POINTS in ROS so goal
    # is 0.
    'target_end_effector': np.zeros(3 * EE_POINTS.shape[0]),
    # 'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 0.1,
    'l2': 0.0001,
    'ramp_option': RAMP_LINEAR,
}

fk_cost2 = {
    'type': CostFK,
    'target_end_effector': np.zeros(3 * EE_POINTS.shape[0]),
    # 'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 1.0,
    'l2': 0.0,
    'wp_final_multiplier': 10.0,  # Weight multiplier on final timestep.
    'ramp_option': RAMP_FINAL_ONLY,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [coord6d_cost],
    'weights': [1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptCaffe,
    'weights_file_prefix': EXP_DIR + 'policy',
    'iterations': 100,
    'batch_size': agent['T']
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 10,
}

config = {
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': 0,
    'verbose_policy_trials': 1,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'num_samples': 2,
}

common['info'] = generate_experiment_info(config)
