# -*- coding: utf-8 -*-
from __future__ import division

import theano
from theano.tensor.shared_randomstreams import RandomStreams

from .base import BaseLayer

srng = RandomStreams(seed=17411)

__all__ = ['DropoutLayer']


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

        mask = srng.binomial(input_shape, p=q).astype(theano.config.floatX)
        self.mask_ = mask
        return input * mask

    def get_output_shape(self):
        return self.input_shape
