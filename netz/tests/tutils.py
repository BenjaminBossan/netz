# -*- coding: utf-8 -*-
from __future__ import division
import inspect
import itertools as it
from copy import deepcopy

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import theano
from theano import function
from theano import tensor as T

from ..utils import to_32

np.random.seed(17411)


def relative_error(v0, v1, epsilon):
    numerator = np.abs([e0 - e1 for e0, e1 in zip(v0, v1)])
    denominator = np.abs(v0) + np.abs(v1) + epsilon
    return numerator / denominator


def verify_grad(net, x, y, abs_tol=None):
    def fun(x, y):
        cost = net.cost_function(y, net.feed_forward(x, deterministic=True))
        return cost
    # one-hot encode y
    encoder = OneHotEncoder(sparse=False, dtype=np.float32)
    y_ = encoder.fit_transform(y.reshape(-1, 1))
    T.verify_grad(fun, [x, y_], rng=np.random.RandomState(42), abs_tol=abs_tol)


def check_weights_shrink(v0, v1):
    relative_diff = v1 / v0
    return (0 < relative_diff).all() & (relative_diff < 1).all()


def check_relative_diff_similar(v0, v1, atol=0.2):
    relative_diff = v1 / v0
    mean_diff = np.mean(relative_diff)
    return np.allclose(mean_diff, relative_diff, atol=atol)


class GradChecker(object):
    def __init__(self, net, num_check=10, epsilon=1e-6):
        self.net = net
        self.num_check = num_check
        self.epsilon = to_32(epsilon)

    @staticmethod
    def _get_n_indices(shape, n):
        if len(shape) > 1:
            indices = list(it.product(*map(range, shape)))
        else:
            indices = list(range(shape[0]))
        np.random.shuffle(indices)
        return indices[:n]

    def _init(self, x, y):
        ys = T.matrix('y').astype(theano.config.floatX)
        if x.ndim == 2:
            xs = T.matrix('x').astype(theano.config.floatX)
        elif x.ndim == 4:
            xs = T.tensor4('x').astype(theano.config.floatX)
        self.xs_, self.ys_ = xs, ys

        encoder = OneHotEncoder(sparse=False, dtype=np.float32)
        y_ = encoder.fit_transform(y.reshape(-1, 1))
        self.y_ = y_

        self.is_init_ = True

    def _get_cost_epsilon(self, param, param_copy, x, y, epsilon, i):
        param_copy[i] += epsilon
        param.set_value(param_copy)
        cost = self.net.test_(x, y)
        param_copy[i] -= epsilon
        return cost

    def _get_n_numerical_grads(self, param, x, y, indices):
        epsilon = self.epsilon
        param_copy = deepcopy(param.get_value())
        numerical_grads = np.zeros_like(param_copy)
        for i in indices:
            cost_plus_eps = self._get_cost_epsilon(
                param, param_copy, x, y, epsilon, i)
            cost_minus_eps = self._get_cost_epsilon(
                param, param_copy, x, y, -epsilon, i)
            numerical_grads[i] = (cost_plus_eps - cost_minus_eps) / epsilon / 2
        # restore parameter
        param.set_value(param_copy)
        return numerical_grads

    def get_grads(self, x, y):
        if not hasattr(self, 'is_init_'):
            self._init(x, y)
        xs, ys, y_ = self.xs_, self.ys_, self.y_
        net = self.net
        cost = net._get_cost_function(xs, ys, True)

        theano_grads_all = []
        numerical_grads_all = []
        for layer in net.layers:
            if not layer.updater:
                continue
            get_grads = function([xs, ys], layer.get_grads(cost))
            theano_grads = get_grads(x, y_)
            for theano_grad, param in zip(theano_grads, layer.get_params()):
                indices = self._get_n_indices(param.get_value().shape,
                                              self.num_check)
                numerical_grad = self._get_n_numerical_grads(
                    param, x, y_, indices
                )
                for idx in indices:
                    theano_grads_all.append(theano_grad[idx])
                    numerical_grads_all.append(numerical_grad[idx])
        return theano_grads_all, numerical_grads_all


def get_default_arg(func, argname):
    """Get the default value of a function or method

    see: http://stackoverflow.com/questions/12627118/
         get-a-function-arguments-default-value

    """
    argument_info = inspect.getargspec(func)
    default_argument_dict = dict(
        zip(argument_info.args[-len(argument_info.defaults):],
            argument_info.defaults)
    )
    return default_argument_dict[argname]
