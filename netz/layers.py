# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import theano
import theano.tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams

srng = RandomStreams(seed=17411)


from nonlinearities import sigmoid
from nonlinearities import softmax


class BaseLayer(object):
    def __init__(self, prev_layer=None, next_layer=None,
                 nonlinearity=sigmoid, params=[None], name=None):
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.nonlinearity = nonlinearity
        self.params = params
        self.name = name

    def initialize(self, X=None, y=None):
        pass

    def set_prev_layer(self, layer):
        self.prev_layer = layer

    def set_next_layer(self, layer):
        self.next_layer = layer

    def set_name(self, name):
        self.name = name

    def get_params(self):
        return [param for param in self.params]

    def get_output(self, X):
        return self.nonlinearity(self.prev_layer.get_output(X))

    def get_grads(self, cost):
        return [theano.grad(cost, param) for param in self.get_params()]

    def set_params(self, updates):
        # old_params = self.get_params()
        # for param, update in zip(old_params, updates):
        #     assert param.shape == update.shape
        for param, update in zip(self.get_params(), updates):
            param.set_value(update)


class InputLayer(BaseLayer):
    def get_output(self, input):
        if isinstance(input, np.ndarray):
            input = T.constant(input, name='input')
        return input

    def initialize(self, X, y):
        self.num_units = X.shape[1]

    def get_grads(self, loss):
        return [None]

    def get_params(self):
        return [None]


class DenseLayer(BaseLayer):
    def __init__(self, num_units, num_features=None, *args, **kwargs):
        super(DenseLayer, self).__init__(*args, **kwargs)
        self.num_units = num_units
        self.num_features = num_features

    def initialize(self, X, y):
        self.num_features = (self.num_features if self.num_features
                             else self.prev_layer.num_units)
        self.W = shared(np.random.random((self.num_features, self.num_units)),
                        name='W_{}'.format(self.name))
        self.b = shared(np.random.random((1, self.num_units)),
                        broadcastable=[True, False],
                        name='b_{}'.format(self.name))
        self.params = [self.W, self.b]

    def get_output(self, X):
        input = self.prev_layer.get_output(X)
        activation = T.dot(input, self.W) + self.b
        activation.name = 'activation'
        return self.nonlinearity(activation)


class OutputLayer(DenseLayer):
    def __init__(self, num_units=None, num_features=None,
                 prev_layer=None, next_layer=None,
                 nonlinearity=softmax, params=[None], name=None):
        self.num_units = num_units
        self.num_features = num_features
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.nonlinearity = nonlinearity
        self.params = params
        self.name = name

    def initialize(self, X, y):
        self.num_units = (self.num_units if self.num_units
                          else len(np.unique(np.argmax(y, axis=1))))
        super(OutputLayer, self).initialize(X, y)
