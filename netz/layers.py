# -*- coding: utf-8 -*-
from __future__ import division
import operator as op
import warnings

import numpy as np
import theano
import theano.tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal.downsample import max_pool_2d

from nonlinearities import sigmoid
from nonlinearities import softmax
from utils import shared_random_uniform

srng = RandomStreams(seed=17411)


class BaseLayer(object):
    def __init__(
            self,
            prev_layer=None,
            next_layer=None,
            nonlinearity=sigmoid,
            params=[None],
            name=None,
            updater=None,
    ):
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.nonlinearity = nonlinearity
        self.params = params
        self.name = name
        self.updater = updater

    def initialize(self, X=None, y=None):
        input_shape = self.prev_layer.get_output_shape()
        self.input_shape = input_shape

    def set_prev_layer(self, layer):
        self.prev_layer = layer

    def set_next_layer(self, layer):
        self.next_layer = layer

    def set_updater(self, updater):
        if self.updater is not None:
            raise warnings.warn("You are overwriting the updater set"
                                "for layer {}.".format(self.name))
        self.updater = updater

    def set_name(self, name):
        self.name = name

    def get_params(self):
        return [param for param in self.params]

    def get_output(self, X, *args, **kwargs):
        return self.nonlinearity(
            self.prev_layer.get_output(X, *args, **kwargs)
        )

    def get_output_shape(self):
        raise NotImplementedError

    def get_grads(self, cost):
        return [theano.grad(cost, param) for param in self.get_params()]

    def set_params(self, updates):
        for param, update in zip(self.get_params(), updates):
            param.set_value(update)

    def create_param(self, shape, name=None, broadcastable=None):
        high = np.sqrt(6 / sum(shape))
        low = -high
        return shared_random_uniform(shape=shape, low=low, high=high,
                                     name=name, broadcastable=broadcastable)


class InputLayer(BaseLayer):
    def get_output(self, X, *args, **kwargs):
        if isinstance(X, np.ndarray):
            X = T.constant(X, name='input')
        return X

    def set_update(self, *args, **kwargs):
        pass

    def initialize(self, X, y):
        self.input_shape = list(map(int, X.shape))
        self.input_shape[0] = None
        self.updater = None

    def get_grads(self, loss):
        return [None]

    def get_params(self):
        return [None]

    def get_output_shape(self):
        return self.input_shape


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
            shape=(self.num_features, self.num_units),
            name='W_{}'.format(self.name),
        )
        self.b = self.create_param(
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


class Conv2DLayer(BaseLayer):
    def __init__(
            self,
            num_filters,
            filter_size,
            strides=(1, 1),
            *args,
            **kwargs
    ):
        super(Conv2DLayer, self).__init__(*args, **kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides

    def initialize(self, X, y):
        super(Conv2DLayer, self).initialize(X, y)
        if X.ndim != 4:
            raise ValueError("Dimension mismatch: Input has {} dimensions "
                             "but Conv2DLayer requires 4D input (bc01)."
                             "".format(X.ndim))
        filter_shape = (self.num_filters,
                        self.input_shape[1],
                        self.filter_size[0],
                        self.filter_size[1])
        self.filter_shape_ = filter_shape
        self.W = self.create_param(
            shape=filter_shape,
            name='W_{}'.format(self.name),
        )
        self.b = self.create_param(
            shape=(self.num_filters,),
            name='b_{}'.format(self.name),
        )
        self.params = [self.W, self.b]

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)
        conv = T.nnet.conv2d(input, self.W, subsample=self.strides,
                             #image_shape=input.shape,
                             filter_shape=self.filter_shape_)
        activation = conv + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(activation)

    def get_output_shape(self):
        if self.strides != (1, 1):
            raise NotImplementedError
        nrows = self.input_shape[2] - self.filter_size[0] + 1
        ncols = self.input_shape[3] - self.filter_size[1] + 1
        return self.input_shape[0], self.num_filters, nrows, ncols


class MaxPool2DLayer(BaseLayer):
    def __init__(self, ds=(2, 2), *args, **kwargs):
        super(MaxPool2DLayer, self).__init__(*args, **kwargs)
        self.ds = ds

    def initialize(self, X, y):
        super(MaxPool2DLayer, self).initialize(X, y)
        self.updater = None

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)
        return max_pool_2d(input, self.ds)

    def get_output_shape(self):
        shape = list(self.input_shape)
        shape[2] = int(np.ceil(shape[2] / self.ds[0]))
        shape[3] = int(np.ceil(shape[3] / self.ds[1]))
        return tuple(shape)

    def get_params(self):
        return [None]


class DropoutLayer(BaseLayer):
    def __init__(self, p=0.5, rescale=True, *args, **kwargs):
        super(DropoutLayer, self).__init__(*args, **kwargs)
        self.p = p
        self.rescale = rescale

    def initialize(self, X, y):
        super(DropoutLayer, self).initialize(X, y)
        self.updater = None

    def get_output(self, X, deterministic=False, *args, **kwargs):
        input = self.prev_layer.get_output(X, deterministic=deterministic,
                                           *args, **kwargs)
        if deterministic or (self.p == 0):
            return input

        q = 1 - self.p
        if self.rescale:
            input /= q

        input_shape = self.input_shape
        if any(shape is None for shape in input_shape):
            input_shape = input.shape

        return input * srng.binomial(input_shape, p=q)

    def get_output_shape(self):
        return self.input_shape
