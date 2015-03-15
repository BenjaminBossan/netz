# -*- coding: utf-8 -*-
from __future__ import division
import itertools as it
from copy import deepcopy

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import theano
from theano import function
from theano import shared
from theano import tensor as T

from ..utils import flatten

np.random.seed(17411)


def verify_grad(net, x, y, abs_tol=None):
    def fun(x, y):
        cost = net.cost_function(y, net.feed_forward(x, deterministic=True))
        return cost
    # one-hot encode y
    encoder = OneHotEncoder(sparse=False)
    y_ = encoder.fit_transform(y.reshape(-1, 1))
    T.verify_grad(fun, [x, y_], rng=np.random.RandomState(42), abs_tol=abs_tol)


class GradChecker(object):
    def __init__(self, net, num_check=10, epsilon=1e-6):
        self.net = net
        self.num_check = num_check
        self.epsilon = epsilon

    def _get_theano_grad(self, param, X, y):
        # symbolic variables
        ys = T.dmatrix('y')
        if X.ndim == 2:
            Xs = T.matrix('X', dtype=theano.config.floatX)
        elif X.ndim == 4:
            Xs = T.tensor4('X', dtype=theano.config.floatX)
        import pdb; pdb.set_trace()
        grad = function(
            [Xs, ys], theano.grad(self.net.cost_deterministic_, param)
        )
        return grad(X, y)

    def _get_cost(self, x, y):
        net = self.net
        ys = T.matrix('y', dtype=theano.config.floatX)
        if x.ndim == 2:
            xs = T.matrix('x', dtype=theano.config.floatX)
        elif x.ndim == 4:
            xs = T.tensor4('x', dtype=theano.config.floatX)
        return self.net.cost_deterministic_(x, y)

    def _get_n_numerical_grads(self, param, x, y):
        epsilon = self.epsilon
        param_copy = deepcopy(param.get_value())
        num_grads = np.zeros_like(param_copy)
        if param_copy.ndim > 1:
            indices = list(it.product(*map(range, param_copy.shape)))
        else:
            indices = list(range(param_copy.shape[0]))
        # only check n random parameters
        np.random.shuffle(indices)
        indices = indices[:self.num_check]
        for i in indices:
            param_pe = deepcopy(param_copy)
            param_pe[i] += epsilon
            param.set_value(param_pe)
            cost_pe = self._get_cost(x, y)
            param_me = deepcopy(param_copy)
            param_me[i] -= epsilon
            param.set_value(param_me)
            cost_me = self._get_cost(x, y)
            num_grads[i] = (cost_pe - cost_me) / epsilon / 2
        # restore parameter
        param.set_value(param_copy)

        return num_grads, indices

    def spit_grads(self, x, y):
        encoder = OneHotEncoder(sparse=False)
        y_ = encoder.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        diffs = []
        params = flatten(self.net.get_layer_params())
        for param in params:
            if not param:
                continue
            theano_grad = self._get_theano_grad(param, x, y_)
            numerical_grad, indices = self._get_n_numerical_grads(
                param, x, y_
            )
            if isinstance(indices[0], tuple):
                indices = zip(*indices)
            yield theano_grad[indices], numerical_grad[indices]
