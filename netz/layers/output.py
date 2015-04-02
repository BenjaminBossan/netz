# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

from ..nonlinearities import softmax
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
