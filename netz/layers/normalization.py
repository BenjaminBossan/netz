# -*- coding: utf-8 -*-
from __future__ import division

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .base import BaseLayer

srng = RandomStreams(seed=17411)

__all__ = ['BatchNormLayer']


class BatchNormLayer(BaseLayer):
    def __init__(self, epsilon=1e-6, *args, **kwargs):
        super(BatchNormLayer, self).__init__(*args, **kwargs)
        self.epsilon = epsilon

    def initialize(self, X, y):
        super(BatchNormLayer, self).initialize(X, y)

        shape = list(self.input_shape)
        ndim = len(shape)
        if ndim == 2:
            self.axes_ = (0,)
        elif ndim == 4:
            self.axes_ = (0, 2, 3)
        for axis in self.axes_:
            shape[axis] = 1

        gamma = self.create_param(
            scheme='ones',
            shape=shape,
            name='gamma_{}'.format(self.name),
        )
        beta = self.create_param(
            scheme='zeros',
            shape=shape,
            name='beta_{}'.format(self.name),
        )
        self.gamma = gamma
        self.beta = beta

    def get_params(self):
        return [self.gamma, self.beta]

    def get_output_shape(self):
        return self.input_shape

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)
        input_mean = input.mean(axis=self.axes_, keepdims=True)
        input_std = input.std(axis=self.axes_, keepdims=True)
        input_mean = T.addbroadcast(input_mean, *self.axes_)
        input_std = T.addbroadcast(input_std, *self.axes_)

        gamma, beta = self.gamma, self.beta
        gamma = T.addbroadcast(self.gamma, *self.axes_)
        beta = T.addbroadcast(self.beta, *self.axes_)

        input_norm = (input - input_mean) / (input_std + self.epsilon)
        output_norm = gamma * input_norm + beta
        return output_norm
