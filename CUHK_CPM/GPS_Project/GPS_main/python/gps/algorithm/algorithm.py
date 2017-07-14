""" This file defines the base algorithm class. """

import abc
import copy
import logging

import numpy as np

from gps.algorithm.config import ALG
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo
from gps.utility.general_utils import extract_condition

baseLogger = logging.getLogger('base')
dynamicsLogger = logging.getLogger('dynamics')
costLogger  = logging.getLogger('cost')
debugger = logging.getLogger('debugger')


class Algorithm(object):
    """ Algorithm superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        # only the algorithm dict in hyperparams.py is passed to this class
        # fetch parameters from config.py file in this folder
        config = copy.deepcopy(ALG)
        # update parameters with hyperparams
        config.update(hyperparams)
        self._hyperparams = config

        # get condition index and number of conditions
        if 'train_conditions' in hyperparams:
            # not in current hyperparams.py file
            # this may be used to select certain conditions from all the available conditions
            self._cond_idx = hyperparams['train_conditions']
            self.M = len(self._cond_idx)
        else:
            self.M = hyperparams['conditions']
            self._cond_idx = range(self.M)

        # set iteration count to 0
        self.iteration_count = 0

        # Grab a few values from the agent.
        # the agent object is initialized in gps_main
        agent = self._hyperparams['agent']
        self.T = self._hyperparams['T'] = agent.T
        self.dU = self._hyperparams['dU'] = agent.dU
        self.dX = self._hyperparams['dX'] = agent.dX
        self.dO = self._hyperparams['dO'] = agent.dO

        # fetch init_traj_distr dict from algorithm dict
        # and add some items
        # NOTE: init_traj_distr is a pointer to config['init_traj_distr']
        # self._hyperparams is a pointer to config
        # changing init_traj_distr is equivalent to changing self._hyperparams['init_traj_distr']
        init_traj_distr = config['init_traj_distr']
        init_traj_distr['x0'] = agent.x0
        init_traj_distr['dX'] = agent.dX
        init_traj_distr['dU'] = agent.dU

        # initialize empty IterationData objects for each condition.
        # refer to algorithm_utils.py
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        # initialize cost object for each condition
        # algorithm['cost']
        self.cost = [
            hyperparams['cost']['type'](hyperparams['cost'])
            for _ in range(self.M)
            ]

        self.tgt = []
        # M : number of conditions
        for m in range(self.M):
            # get tgt from agent
            self.tgt.append(agent.tgt[m].copy())
            # initialize TrajectoryInfo object, it is part of IterationData
            self.cur[m].traj_info = TrajectoryInfo()

            # get dynamics dict from algorithm dict
            dynamics = self._hyperparams['dynamics']
            # dynamics = DynamicsLRPrior
            # each policy has its own dynamics prior
            # it considers all time steps and even info from previous few iteration

            # initialize DynamicsLRPrior object as part of TrajectoryInfo
            self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)

            # init_traj_distr['x0'] contains only initial state for ONE concerning condition
            init_traj_distr = extract_condition(
                self._hyperparams['init_traj_distr'], self._cond_idx[m]
            )

            # self.cur[m].traj_distr = init_lqr(init_traj_distr)
            # initialize lg policy for each condition to hold the initial state
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)
            # get target states for each condition and save it into CostSum
            self.cost[m]._costs[0].tgt = agent.tgt[m].copy()

        del self._hyperparams['agent']  # Don't want to pickle this.

        # TrajOptLQRPython
        # supposed DDP/iLQG in this
        self.traj_opt = hyperparams['traj_opt']['type'](
            hyperparams['traj_opt']
        )

        self.base_kl_step = self._hyperparams['kl_step']


    @abc.abstractmethod
    def iteration(self, sample_list):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")

    def _update_dynamics(self):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to
        current samples.
        """
        # self.prev & self.cur is a list of IterationData elements
        # IterationData has attributes:
        # traj_info : DynamicsLRPrior
        # traj_dist : linear Gaussian Policy
        # pol_info : policy_prior
        # cost : CostSum
        # M : number of conditions

        baseLogger.info("Updating dynamics...")
        for cond in range(self.M):
            if self.iteration_count >= 1:
                self.prev[cond].traj_info.dynamics = \
                        self.cur[cond].traj_info.dynamics.copy()
            cur_data = self.cur[cond].sample_list
            # each condition has its own dynamics/prior/GMM
            # p(xt, ut, xt+1) ~ sigma Gaussian(mu, var)
            # update_prior update parameters in sigma Gaussian(mu, var), aka GMM

            self.cur[cond].traj_info.dynamics.update_prior(cur_data)
            # based on GMM, theoretically get p(xt+1 | xt, ut), actually get xt+1 = fx * x +fu * u +fc
            self.cur[cond].traj_info.dynamics.fit(cur_data)

            for t in range(self.T):
                dynamicsLogger.info("")
                dynamicsLogger.info("Iteration: %d, Condition: %d, Time step %d",
                                    self.iteration_count, cond, t)
                dynamicsLogger.info("Fm:")
                # self.cur[cond].traj_info.dynamics.Fm[t] = np.hstack((np.eye(self.dX), np.eye(self.dU)))
                dynamicsLogger.info(self.cur[cond].traj_info.dynamics.Fm[t])
                dynamicsLogger.info("fv:")
                # self.cur[cond].traj_info.dynamics.fv[t] = np.zeros(self.dX)
                dynamicsLogger.info(self.cur[cond].traj_info.dynamics.fv[t])
                dynamicsLogger.info("Cov:")
                dynamicsLogger.info(self.cur[cond].traj_info.dynamics.dyn_covar[t])


            # first : means all samples in current condition
            # last : means dimension of state
            # 0 means t=0, aka, initial state
            init_X = cur_data.get_X()[:, 0, :]
            x0mu = np.mean(init_X, axis=0)
            self.cur[cond].traj_info.x0mu = x0mu
            # by default initial_state_var : 1e-06
            # Note: for 1st iteration, all x0 for one condition are the same
            self.cur[cond].traj_info.x0sigma = np.diag(
                np.maximum(np.var(init_X, axis=0),
                           self._hyperparams['initial_state_var'])
            )

            # paper Page 29
            # Note: if all x0 for one condition are the same
            # x0mu-mu0 = 0
            prior = self.cur[cond].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[cond].traj_info.x0sigma += \
                        Phi + (N*priorm) / (N+priorm) * \
                        np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_trajectories(self, pred_traj):
        """
        Compute new linear Gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
        for cond in range(self.M):
            self.new_traj_distr[cond], self.cur[cond].eta, pred_traj[cond] = \
                    self.traj_opt.update(cond, self)

    def _eval_cost(self, cond):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        # Constants.

        costLogger.info("Condition: %d", cond)
        T, dX, dU = self.T, self.dX, self.dU
        # N is the nubmer of samples for current condition
        N = len(self.cur[cond].sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        costData = []
        for n in range(N):
            # for each sample in current condition
            sample = self.cur[cond].sample_list[n]
            # Get costs.
            # first item of four to update traj_distr
            l, lx, lu, lxx, luu, lux, data = self.cost[cond].eval(sample)

            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            # X.shape = (T, dX)
            # U.shape = (T, dU)
            # rdiff_expand.shape = (T, dX+dU, 1)
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            # cv_update.shape = (T, dX+dU)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * \
                    np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update
            # data = dict(total=l, cc=cc[n, :])
            data['cc'] = cc[n, :]
            costData.append(data)

        # Fill in cost estimate.
        # cc is average loss for all samples for concerning condition for current iteration
        # cv,Cm is average loss's 1st/2nd dev for all samples for concerning condition for current iteration
        # loss is loss of state-action pair, not of traj
        # cs.shape=cc.shape = (T,), cv.shape = (T, dX+dU)
        self.cur[cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[cond].cs = cs  # True value of cost.
        return costData

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration
        counter.
        """
        self.iteration_count += 1
        self.prev = self.cur
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = self.prev[m].traj_info.dynamics
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
        delattr(self, 'new_traj_distr')

    def _set_new_mult(self, predicted_impr, actual_impr, m):
        """
        Adjust step size multiplier according to the predicted versus
        actual improvement.
        """
        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4,
                                               predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(
            min(new_mult * self.cur[m].step_mult,
                self._hyperparams['max_step_mult']),
            self._hyperparams['min_step_mult']
        )
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            baseLogger.info('Increasing step size multiplier to %f', new_step)
        else:
            baseLogger.info('Decreasing step size multiplier to %f', new_step)

    def _measure_ent(self, m):
        """ Measure the entropy of the current trajectory. """
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(
                np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t, :, :]))
            )
        return ent
