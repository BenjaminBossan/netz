# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from sklearn.utils import murmurhash3_32 as mmh
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .base import BaseLayer
from .dense import DenseLayer
from .input import InputLayer
from ..utils import to_32

srng = RandomStreams(seed=17411)


class NormConstraintLayer(BaseLayer):
    """This layer basically does nothing but pass on the input from
    the previous layer. In addition, it will normalize the input from
    the next layer to the given constraint.

    """
    def __init__(self, max_norm, *args, **kwargs):
        super(NormConstraintLayer, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def set_updater(self, *args, **kwargs):
        pass

    def initialize(self, X, y):
        self.input_shape = self.prev_layer.input_shape
        self.updater = None

    def get_grads(self, cost):
        return [None]

    def get_params(self):
        return [None]

    def get_output_shape(self):
        return self.prev_layer.get_output_shape()

    @staticmethod
    def _get_norm(param):
        if not param:
            pass
        ndim = param.ndim
        if ndim == 1:  # embedding layer
            norm = T.sqrt(T.sqr(param), keepdims=True)
            return norm
        elif ndim == 2:  # dense layer
            norm = T.sqrt(T.sum(T.sqr(param), axis=(0,), keepdims=True))
            return norm
        elif ndim > 2:  # convolution layer
            axes = tuple(range(1, ndim))
            norm = T.sqrt(T.sum(T.sqr(param), axis=axes, keepdims=True))
            return norm

    @staticmethod
    def _constrain(params, max_norm):
        max_norm = to_32(max_norm)
        for param in params:
            norm = NormConstraintLayer._get_norm(param)
            if T.lt(T.max(abs(norm)), max_norm):
                continue
            param *= max_norm / norm

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)
        next_params = self.next_layer.get_params()
        self._constrain(next_params, self.max_norm)
        return input


class FeatureHashLayer(DenseLayer):
    """Note: No true support for batches yet, each sample is worked by
    itself.

    """
    def __init__(self, num_units, size=2 ** 20, *args, **kwargs):
        super(FeatureHashLayer, self).__init__(*args, **kwargs)
        self.num_units = num_units
        self.size = size

    def initialize(self, X, y):
        if not isinstance(self.prev_layer, InputLayer):
            raise TypeError("The FeatureHashLayer does not support following "
                            "any other layer but an InputLayer.")

        super(FeatureHashLayer, self).initialize(X, y)
        self.W = self.create_param(
            shape=(self.size, self.num_units),
            name='W_{}'.format(self.name),
        )
        self.b = self.create_param(
            shape=(1, self.num_units),
            name='b_{}'.format(self.name),
        )
        self.mask_ = self.create_param(
            shape=(1, self.num_units),
            name='mask_{}'.format(self.name),
            broadcastable=(True, False),
            limits=(0., 0.),
        )
        self.params = [self.W, self.b]

    def get_updates(self, *args, **kwargs):
        updates = super(FeatureHashLayer, self).get_updates(*args, **kwargs)
        more_updates = [(self.mask_, self.mask_new_)]
        return updates + more_updates

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)
        mask_new = np.array([mmh(x) % self.size for x in input]).reshape(-1, 1)
        mask_new = T.shared(mask_new)
        self.mask_new_ = mask_new
        output = T.sum(T.multiply(self.W, mask_new) + self.b, axis=0)
        return output
