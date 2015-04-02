# -*- coding: utf-8 -*-
from __future__ import division
import operator as op
import warnings

import numpy as np
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.tensor.extra_ops import repeat
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal.downsample import max_pool_2d

from ..nonlinearities import sigmoid
from ..nonlinearities import softmax
from ..utils import shared_random_normal
from ..utils import shared_random_orthogonal
from ..utils import shared_random_uniform
from ..utils import shared_ones
from ..utils import shared_zeros
from ..utils import to_32
from .base import BaseLayer

srng = RandomStreams(seed=17411)

__all__ = ['DenseLayer']


class DenseLayer(BaseLayer):
    def __init__(self, num_units, num_features=None, *args, **kwargs):
        super(DenseLayer, self).__init__(*args, **kwargs)
        self.num_units = num_units
        self.num_features = num_features

    def initialize(self, X, y):
        super(DenseLayer, self).initialize(X, y)
        self.num_features = (self.num_features if self.num_features
                             else reduce(op.mul, self.input_shape[1:]))
        self.W = self.create_param(
            scheme=self.init_scheme,
            shape=(self.num_features, self.num_units),
            name='W_{}'.format(self.name),
        )
        self.b = self.create_param(
            scheme=self.init_scheme,
            shape=(1, self.num_units),
            name='b_{}'.format(self.name),
            broadcastable=(True, False),
        )
        self.params = [self.W, self.b]

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)
        if input.ndim > 2:
            input = input.flatten(2)
        activation = T.dot(input, self.W) + self.b
        activation.name = 'activation'
        return self.nonlinearity(activation)

    def get_output_shape(self):
        return self.input_shape[0], self.num_units

    def get_l2_cost(self):
        return 0.5 * self.lambda2 * T.sum(self.W ** 2)
