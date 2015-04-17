# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

from ..nonlinearities import softmax
from ..nonlinearities import identity
from .dense import DenseLayer

srng = RandomStreams(seed=17411)

__all__ = ['OutputLayer', 'RegressionOutputLayer']


class OutputLayer(DenseLayer):
    def __init__(self, num_units=None, num_features=None, nonlinearity=softmax,
                 *args, **kwargs):
        super(OutputLayer, self).__init__(num_units, num_features,
                                          *args, **kwargs)
        self.nonlinearity = nonlinearity

    def initialize(self, X, y):
        self.num_units = (self.num_units if self.num_units
                          else len(np.unique(y)))
        super(OutputLayer, self).initialize(X, y)


class RegressionOutputLayer(DenseLayer):
    def __init__(self, num_units=None, num_features=None,
                 nonlinearity=identity, *args, **kwargs):
        super(RegressionOutputLayer, self).__init__(num_units, num_features,
                                                    *args, **kwargs)
        self.nonlinearity = nonlinearity

    def initialize(self, X, y):
        self.num_units = self.num_units if self.num_units else 1
        super(RegressionOutputLayer, self).initialize(X, y)
