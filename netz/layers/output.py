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
from .dense import DenseLayer

srng = RandomStreams(seed=17411)

__all__ = ['OutputLayer']


class OutputLayer(DenseLayer):
    def __init__(self, num_units=None, num_features=None, nonlinearity=softmax,
                 *args, **kwargs):
        super(DenseLayer, self).__init__(*args, **kwargs)
        self.nonlinearity = nonlinearity
        self.num_units = num_units
        self.num_features = num_features

    def initialize(self, X, y):
        self.num_units = (self.num_units if self.num_units
                          else len(np.unique(y)))
        super(OutputLayer, self).initialize(X, y)
