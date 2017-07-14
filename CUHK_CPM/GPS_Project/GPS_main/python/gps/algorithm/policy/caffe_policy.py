""" This file defines a neural network policy implemented in Caffe. """
import tempfile

import numpy as np

from gps.algorithm.policy.policy import Policy

class CaffePolicy(Policy):
    """
    A neural network policy implemented in Caffe. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        test_net: Initialized caffe network that can run forward.
        var: Du-dimensional noise variance vector.
    """
    def __init__(self, test_net, deploy_net, var):
        Policy.__init__(self)
        self.net = test_net
        self.deploy_net = deploy_net
        self.chol_pol_covar = np.diag(np.sqrt(var))

    def act(self, x, obs, meta, t, noise=None):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        # Normalize obs.
        obs = obs.dot(self.scale) + self.bias

        if meta is None:
            self.net.blobs[self.net.blobs.keys()[0]].data[:] = obs
            action_mean = self.net.forward().values()[0][0]
        else:
            self.net.blobs['data'].data[0] = meta
            self.net.blobs['obs'].data[0] = obs
            self.net.forward()
            action_mean = self.net.blobs['output'].data[0]

        if noise is not None:
            action_mean += self.chol_pol_covar.T.dot(noise)
        return action_mean

    def get_weights_string(self):
        """ Return the weights of the neural network as a string. """
        raise 'NotImplemented - weights string prob in net_param'

    def get_net_param(self):
        """ Return the weights of the neural network as a string. """
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.deploy_net.share_with(self.net)
        self.deploy_net.save(f.name)
        f.close()
        with open(f.name, 'rb') as temp_file:
            weights_string = temp_file.read()
        return weights_string
