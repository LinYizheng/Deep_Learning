import copy

import numpy as np
import logging

from gps.algorithm.cost.cost import Cost

costLogger = logging.getLogger('cost')

import os
class CostRR6D(Cost):
    """ Computes torque penalties. """
    def __init__(self, hyperparams):
        Cost.__init__(self, hyperparams)
        self.tgt = None

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """
        print 'here our rr cost'
        T = sample.T
        dU = sample.dU
        dX = sample.dX

        sample_u = sample.get_U()
        sample_x = sample.get_X()

        wl2 = 0.3       # weight of state l2 cost
        wl1 = 0.0       # weight of state l1 cost
        wlog = 0.0      # weight of state log cost
        wu = 1.0        # weight of action cost
        wt = (np.arange(T)/float(T)) ** 2     # time dependent weight
        # final_weight_multiplier = 9
        # wt[-1] *= final_weight_multiplier
        alpha = 1e-5    # offset for log cost

        tgtM = np.tile(self.tgt, (T, 1))
        d = sample_x - tgtM
        dnorm = np.sum(d ** 2, axis=1)
        denominator = alpha + wt ** 2 * dnorm

        # TODO: for log, wt --> wt^2
        ll2 = 0.5 * wl2 * wt * dnorm
        ll1 = wl1 * wt * np.sqrt(dnorm)
        llog = 0.5 * wlog * np.log(denominator)
        lu = 0.5 * wu * np.sum(sample_u ** 2, axis=1)
        l = ll2 + ll1 + llog + lu
        print l

        data = {}
        costLogger.info("l2 cost")
        costLogger.info(ll2)
        data['l2'] = ll2
        costLogger.info("l1 cost")
        costLogger.info(ll1)
        data['l1'] = ll1
        costLogger.info("log cost")
        costLogger.info(llog)
        data['log'] = llog
        costLogger.info("lu cost")
        costLogger.info(lu)
        data['lu'] = lu
        costLogger.info("total cost")
        costLogger.info(l)
        data['total'] = l

        lx = wl2 * wt.reshape((T, 1)) * d + \
             wl1 * (wt / (np.sqrt(dnorm))).reshape((T, 1)) * d + \
             wlog * (wt ** 2 / denominator).reshape((T, 1)) * d

        costLogger.info("lx")
        costLogger.info(lx)

        lu = wu * sample_u
        lxx = np.zeros((T, dX, dX))
        for t in range(T):
            lxx[t] = wl1 * np.diag(wt[t]*(1/np.sqrt(dnorm[t])-dnorm[t]**(-1.5)*d[t]**2)) + \
                     wlog * np.diag(wt[t]**2/denominator[t]-2*wt[t]**4*d[t]**2/denominator[t]**2)
        lxx += wl2 * np.tile(np.eye(dX), [T, 1, 1]) * wt.reshape((T, 1, 1))
        luu = wu * np.tile(np.eye(dU), [T, 1, 1])
        lux = np.zeros((T, dU, dX))

        return l, lx, lu, lxx, luu, lux, data
