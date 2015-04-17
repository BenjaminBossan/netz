# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal.downsample import max_pool_2d

from .base import BaseLayer

srng = RandomStreams(seed=17411)

__all__ = ['MaxPool2DLayer', 'FeaturePoolLayer']


class MaxPool2DLayer(BaseLayer):
    def __init__(self, ds=(2, 2), *args, **kwargs):
        super(MaxPool2DLayer, self).__init__(*args, **kwargs)
        self.ds = ds

    def initialize(self, X, y):
        super(MaxPool2DLayer, self).initialize(X, y)
        self.updater = None

    def get_params(self):
        return [None]

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)
        return max_pool_2d(input, self.ds)

    def get_output_shape(self):
        shape = list(self.input_shape)
        shape[2] = int(np.ceil(shape[2] / self.ds[0]))
        shape[3] = int(np.ceil(shape[3] / self.ds[1]))
        return tuple(shape)


class FeaturePoolLayer(BaseLayer):
    """Currently only supports pooling over dimension 1."""
    def __init__(self, ds=2, axis=1, pool_function=T.max, *args, **kwargs):
        super(FeaturePoolLayer, self).__init__(*args, **kwargs)
        self.ds = ds
        self.axis = axis
        self.pool_function = pool_function

    def initialize(self, X, y):
        super(FeaturePoolLayer, self).initialize(X, y)
        self.updater = None

    def get_params(self):
        return [None]

    @staticmethod
    def _get_pooled_shape_plus1(shape, ds, axis):
        num_feature_maps = shape[axis]
        num_feature_maps_out = num_feature_maps // ds

        pool_shape = list(shape)
        pool_shape.insert(axis, num_feature_maps_out)
        pool_shape[axis + 1] = ds
        return pool_shape

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)

        pool_shape = self._get_pooled_shape_plus1(input.shape, self.ds,
                                                  self.axis)
        input_reshaped = input.reshape(pool_shape)
        output = self.pool_function(input_reshaped, axis=self.axis + 1)

        return output

    def get_output_shape(self):
        input_shape = self.prev_layer.get_output_shape()
        output_shape = self._get_pooled_shape_plus1(input_shape, self.ds,
                                                    self.axis)
        return tuple(output_shape[:self.axis + 1] +
                     output_shape[self.axis + 2:])
