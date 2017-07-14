""" This file defines policy optimization for a Caffe policy. """
import copy
import logging
import tempfile
import time
import numpy as np
import cPickle

import caffe
from caffe.proto.caffe_pb2 import SolverParameter, TRAIN, TEST
from google.protobuf.text_format import MessageToString

from gps.algorithm.policy.caffe_policy import CaffePolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.config import POLICY_OPT_CAFFE


LOGGER = logging.getLogger(__name__)

class PolicyOptCaffeVision(PolicyOpt):
    """ Policy optimization using Caffe neural network library. """
    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_CAFFE)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        self.batch_size = self._hyperparams['batch_size']
        # self.batch_size = 1

        if self._hyperparams['use_gpu']:
            caffe.set_device(self._hyperparams['gpu_id'])
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.snapshot_prefix = self._hyperparams['weights_file_prefix']
        # self.dummy_solver = self.init_dummy_solver()
        self.vision_net_solver = self.init_vision_solver()
        self.frozen_vision_net_solver = self.init_vision_solver(freeze_cnn=True)
        self.frozen_vision_net_solver.net.copy_from(self._hyperparams['exp_dir'] + self._hyperparams['pre_trained_model'])
        self.solver = self.frozen_vision_net_solver
        self.blob_names = ['obs', 'action', 'precision', 'loss']

        self.caffe_iter = 0
        self.var = self._hyperparams['init_var'] * np.ones(dU)

        self.policy = CaffePolicy(self.vision_net_solver.net, self.vision_net_solver.test_nets[0], np.zeros(dU))

    def switch_solver(self):
        snapshot_name = self.snapshot_prefix + "_iter_" + str(self.solver.iter)
        self.vision_net_solver.restore(snapshot_name + '.solverstate')
        self.vision_net_solver.net.copy_from(snapshot_name + '.caffemodel')
        self.solver = self.vision_net_solver
        print "Switched to live vision net."

    def init_vision_solver(self, freeze_cnn=False):
        prototxt = "vision_net.prototxt" if freeze_cnn is False else "vision_net_cnn_frozen.prototxt"
        solver_param = SolverParameter()
        solver_param.display = 16
        solver_param.base_lr = 0.01
        solver_param.lr_policy = 'fixed'
        solver_param.max_iter = 10000
        solver_param.snapshot_prefix = self.snapshot_prefix
        solver_param.train_net = self._hyperparams['exp_dir'] + prototxt
        solver_param.test_net.append(self._hyperparams['exp_dir'] + prototxt)
        solver_param.test_iter.append(1)
        solver_param.test_interval = 1000000
        solver_param.type = 'Adam'
        solver_param.momentum = 0.9
        solver_param.momentum2 = 0.995
        solver_param.weight_decay = 1e-3
        solver_param.solver_mode = SolverParameter.GPU

        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.write(MessageToString(solver_param))
        f.close()

        solver = caffe.get_solver(f.name)
        return solver

    def init_dummy_solver(self):
        """ Helper method to initialize the solver. """
        solver_param = SolverParameter()
        solver_param.snapshot_prefix = self._hyperparams['weights_file_prefix']
        solver_param.display = 0  # Don't display anything.
        solver_param.base_lr = self._hyperparams['lr']
        solver_param.lr_policy = self._hyperparams['lr_policy']
        solver_param.momentum = self._hyperparams['momentum']
        solver_param.weight_decay = self._hyperparams['weight_decay']
        solver_param.type = self._hyperparams['solver_type']
        solver_param.random_seed = self._hyperparams['random_seed']

        # Pass in net parameter either by filename or protostring.
        if isinstance(self._hyperparams['network_model'], basestring):
            return caffe.get_solver(self._hyperparams['network_model'])
        else:
            network_arch_params = self._hyperparams['network_arch_params']
            network_arch_params['dim_input'] = self._dO
            network_arch_params['dim_output'] = self._dU

            network_arch_params['batch_size'] = self.batch_size
            network_arch_params['phase'] = TRAIN
            solver_param.train_net_param.CopyFrom(
                self._hyperparams['network_model'](**network_arch_params)
            )

            # For running forward in python.
            network_arch_params['batch_size'] = self.batch_size
            network_arch_params['phase'] = TEST
            solver_param.test_net_param.add().CopyFrom(
                self._hyperparams['network_model'](**network_arch_params)
            )

            # For running forward on the robot.
            network_arch_params['batch_size'] = 1
            network_arch_params['phase'] = 'deploy'
            solver_param.test_net_param.add().CopyFrom(
                self._hyperparams['network_model'](**network_arch_params)
            )

            # These are required by Caffe to be set, but not used.
            solver_param.test_iter.append(1)
            solver_param.test_iter.append(1)
            solver_param.test_interval = 1000000

            f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            f.write(MessageToString(solver_param))
            f.close()

            return caffe.get_solver(f.name)

    # TODO - This assumes that the obs is a vector being passed into the
    #        network in the same place.
    #        (won't work with images or multimodal networks)
    def update(self, meta, obs, tgt_mu, tgt_prc, tgt_wt, itr, inner_itr):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A CaffePolicy object with updated weights.
        """
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N*T, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N * T) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            for t in range(T):
                tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        img_sz = self._hyperparams['image_size']
        obs = np.reshape(obs, (N*T, dO))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))
        meta = np.reshape(meta, (N*T, 3, img_sz, img_sz))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        #TODO: Find entries with very low weights?

        if itr == self._hyperparams['freeze_period']:
            self.switch_solver()

        # Normalize obs, but only compute normalzation at the beginning.
        if itr == 0 and inner_itr == 1:
            self.policy.scale = np.diag(1.0 / np.std(obs, axis=0))
            self.policy.bias = -np.mean(obs.dot(self.policy.scale), axis=0)
        obs = obs.dot(self.policy.scale) + self.policy.bias

        # blob_names = self.solver.net.blobs.keys()
        blob_names = self.blob_names
        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.floor(N*T / self.batch_size)
        idx = range(N*T)
        average_loss = 0
        np.random.shuffle(idx)

        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]

            self.solver.net.blobs['data'].data[...] = meta[idx_i]
            self.solver.net.blobs[blob_names[0]].data[:] = obs[idx_i]
            self.solver.net.blobs[blob_names[1]].data[:] = tgt_mu[idx_i]
            self.solver.net.blobs[blob_names[2]].data[:] = tgt_prc[idx_i]
            self.solver.step(1)

            # To get the training loss:
            # train_loss = self.solver.net.blobs[blob_names[-1]].data
            # average_loss += train_loss
            # if i % 10 == 0 and i != 0:
            #     LOGGER.info('Caffe iteration %d, average loss %f', i, average_loss / 10)
            #     print 'Caffe iteration %d, average loss %f' % (i, average_loss / 10)
            #     average_loss = 0

            # To run a test:
            # if i % test_interval:
            #     print 'Iteration', i, 'testing...'
            #     solver.test_nets[0].forward()
        # print 'conv3', np.sum(np.sum(self.solver.net.params['conv3'][0].data))
        # print 'fc5', np.sum(np.sum(self.solver.net.params['fc5'][0].data))
        # Keep track of Caffe iterations for loading solver states.
        self.caffe_iter = self.solver.iter

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * N * T * \
                self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.var = 1 / np.diag(A)

        self.policy.net.share_with(self.solver.net)
        # self.policy.chol_pol_covar = np.diag(np.sqrt(self.var))
        return self.policy

    def prob(self, obs, meta):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        dU = self._dU
        N, T = obs.shape[:2]

        # Normalize obs.
        try:
            for n in range(N):
                obs[n, :, :] = obs[n, :, :].dot(self.policy.scale) + \
                        self.policy.bias
        except AttributeError:
            pass  #TODO: Should prob be called before update?

        output = np.zeros((N, T, dU))
        # blob_names = self.solver.test_nets[0].blobs.keys()
        blob_names = self.blob_names

        self.solver.test_nets[0].share_with(self.solver.net)

        time_start = time.time()
        for i in range(N):
            for t in range(T/self.batch_size):
                idx_t = range(t*self.batch_size, (t+1)*self.batch_size)
                # Feed in data.
                self.solver.test_nets[0].blobs['data'].data[...] = meta[i, idx_t]
                self.solver.test_nets[0].blobs[blob_names[0]].data[:] = obs[i, idx_t]

                # Assume that the first output blob is what we want.
                self.solver.test_nets[0].forward()
                output[i, idx_t] = self.solver.test_nets[0].blobs['output'].data[:]
        time_end = time.time()
        print 'PolicyOptCaffe.prob() elapsed time:', time_end - time_start

        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])
        # output is actions, aka, mean of the policy
        # NN only outputs the action, not the distribution of action
        # thus we cannot calculate variation of the non-linear policy
        # through NN itself, it is set by default = self.var
        return output, pol_sigma, pol_prec, pol_det_sigma

    def set_ent_reg(self, ent_reg):
        """ Set the entropy regularization. """
        self._hyperparams['ent_reg'] = ent_reg

    # For pickling.
    def __getstate__(self):
        self.solver.snapshot()
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': self.policy.scale,
            'bias': self.policy.bias,
            'caffe_iter': self.caffe_iter,
        }

    # For unpickling.
    def __setstate__(self, state):
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        self.policy.scale = state['scale']
        self.policy.bias = state['bias']
        self.caffe_iter = state['caffe_iter']
        self.solver.restore(
            self._hyperparams['weights_file_prefix'] + '_iter_' +
            str(self.caffe_iter) + '.solverstate'
        )
        self.policy.net.copy_from(
            self._hyperparams['weights_file_prefix'] + '_iter_' +
            str(self.caffe_iter) + '.caffemodel'
        )


class PolicyOptCaffe(PolicyOpt):
    """ Policy optimization using Caffe neural network library. """
    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_CAFFE)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        self.batch_size = self._hyperparams['batch_size']

        if self._hyperparams['use_gpu']:
            caffe.set_device(self._hyperparams['gpu_id'])
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.init_solver()
        self.caffe_iter = 0
        self.var = self._hyperparams['init_var'] * np.ones(dU)

        self.policy = CaffePolicy(self.solver.test_nets[0],
                                  self.solver.test_nets[1],
                                  np.zeros(dU))
        self.loss = []

    def init_solver(self):
        """ Helper method to initialize the solver. """
        solver_param = SolverParameter()
        solver_param.snapshot_prefix = self._hyperparams['weights_file_prefix']
        solver_param.display = 0  # Don't display anything.
        solver_param.base_lr = self._hyperparams['lr']
        solver_param.lr_policy = self._hyperparams['lr_policy']
        solver_param.momentum = self._hyperparams['momentum']
        solver_param.weight_decay = self._hyperparams['weight_decay']
        solver_param.type = self._hyperparams['solver_type']
        solver_param.random_seed = self._hyperparams['random_seed']

        # Pass in net parameter either by filename or protostring.
        if isinstance(self._hyperparams['network_model'], basestring):
            self.solver = caffe.get_solver(self._hyperparams['network_model'])
        else:
            network_arch_params = self._hyperparams['network_arch_params']
            network_arch_params['dim_input'] = self._dO
            network_arch_params['dim_output'] = self._dU

            network_arch_params['batch_size'] = self.batch_size
            network_arch_params['phase'] = TRAIN
            solver_param.train_net_param.CopyFrom(
                self._hyperparams['network_model'](**network_arch_params)
            )

            # For running forward in python.
            network_arch_params['batch_size'] = 1
            network_arch_params['phase'] = TEST
            solver_param.test_net_param.add().CopyFrom(
                self._hyperparams['network_model'](**network_arch_params)
            )

            # For running forward on the robot.
            network_arch_params['batch_size'] = 1
            network_arch_params['phase'] = 'deploy'
            solver_param.test_net_param.add().CopyFrom(
                self._hyperparams['network_model'](**network_arch_params)
            )

            # These are required by Caffe to be set, but not used.
            solver_param.test_iter.append(1)
            solver_param.test_iter.append(1)
            solver_param.test_interval = 1000000

            f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            f.write(MessageToString(solver_param))
            f.close()

            with open(self._hyperparams['exp_dir'] + 'simple_net.prototxt', 'w') as _file:
                _file.write(str(self._hyperparams['network_model'](**network_arch_params)))

            with open(self._hyperparams['exp_dir'] + 'simple_solver.prototxt', 'w') as _file:
                _file.write(MessageToString(solver_param))

            self.solver = caffe.get_solver(f.name)

    # TODO - This assumes that the obs is a vector being passed into the
    #        network in the same place.
    #        (won't work with images or multimodal networks)
    def update(self, meta, obs, tgt_mu, tgt_prc, tgt_wt, itr, inner_itr):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A CaffePolicy object with updated weights.
        """
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N*T, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N * T) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            for t in range(T):
                tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N*T, dO))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        #TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalzation at the beginning.
        if itr == 0 and inner_itr == 1:
            self.policy.scale = np.diag(1.0 / np.std(obs, axis=0))
            self.policy.bias = -np.mean(obs.dot(self.policy.scale), axis=0)
            cPickle.dump(dict(scale=self.policy.scale, bias=self.policy.bias),
                         open(self._hyperparams['exp_dir'] + 'scale_bias.pkl', 'wb'))
        obs = obs.dot(self.policy.scale) + self.policy.bias

        blob_names = self.solver.net.blobs.keys()

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.floor(N*T / self.batch_size)
        idx = range(N*T)
        average_loss = 0
        np.random.shuffle(idx)
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            self.solver.net.blobs[blob_names[0]].data[:] = obs[idx_i]
            self.solver.net.blobs[blob_names[1]].data[:] = tgt_mu[idx_i]
            self.solver.net.blobs[blob_names[2]].data[:] = tgt_prc[idx_i]

            self.solver.step(1)

            # To get the training loss:
            train_loss = self.solver.net.blobs[blob_names[-1]].data
            self.loss.append(train_loss[0])
            average_loss += train_loss
            if i % 500 == 0 and i != 0:

                LOGGER.debug('Caffe iteration %d, average loss %f',
                             i, average_loss / 500)
                average_loss = 0

            # To run a test:
            # if i % test_interval:
            #     print 'Iteration', i, 'testing...'
            #     solver.test_nets[0].forward()

        # Keep track of Caffe iterations for loading solver states.
        self.caffe_iter += self._hyperparams['iterations']

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * N * T * \
                self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.var = 1 / np.diag(A)

        self.policy.net.share_with(self.solver.net)
        return self.policy

    def prob(self, obs, meta):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        dU = self._dU
        N, T = obs.shape[:2]

        # Normalize obs.
        try:
            for n in range(N):
                obs[n, :, :] = obs[n, :, :].dot(self.policy.scale) + \
                        self.policy.bias
        except AttributeError:
            pass  #TODO: Should prob be called before update?

        output = np.zeros((N, T, dU))
        blob_names = self.solver.test_nets[0].blobs.keys()

        self.solver.test_nets[0].share_with(self.solver.net)

        for i in range(N):
            for t in range(T):
                # Feed in data.
                self.solver.test_nets[0].blobs[blob_names[0]].data[:] = \
                        obs[i, t]

                # Assume that the first output blob is what we want.
                output[i, t, :] = \
                        self.solver.test_nets[0].forward().values()[0][0]

        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])
        # output is actions, aka, mean of the policy
        # NN only outputs the action, not the distribution of action
        # thus we cannot calculate variation of the non-linear policy
        # through NN itself, it is set by default = self.var
        return output, pol_sigma, pol_prec, pol_det_sigma

    def set_ent_reg(self, ent_reg):
        """ Set the entropy regularization. """
        self._hyperparams['ent_reg'] = ent_reg

    # For pickling.
    def __getstate__(self):
        self.solver.snapshot()
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': self.policy.scale,
            'bias': self.policy.bias,
            'caffe_iter': self.caffe_iter,
        }

    # For unpickling.
    def __setstate__(self, state):
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        self.policy.scale = state['scale']
        self.policy.bias = state['bias']
        self.caffe_iter = state['caffe_iter']
        self.solver.restore(
            self._hyperparams['weights_file_prefix'] + '_iter_' +
            str(self.caffe_iter) + '.solverstate'
        )
        self.policy.net.copy_from(
            self._hyperparams['weights_file_prefix'] + '_iter_' +
            str(self.caffe_iter) + '.caffemodel'
        )
