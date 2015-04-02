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

__all__ = ['InputLayer',
           'PartialInputLayer',
           'InputConcatLayer',
           'EmbeddingLayer']


class InputLayer(BaseLayer):
    def get_output(self, X, *args, **kwargs):
        if isinstance(X, np.ndarray):
            X = T.constant(X, name='input')
        return X

    def set_updater(self, *args, **kwargs):
        pass

    def initialize(self, X, y):
        self.input_shape = list(map(int, X.shape))
        self.input_shape[0] = None
        self.updater = None

    def get_grads(self, cost):
        return [None]

    def get_params(self):
        return [None]

    def get_output_shape(self):
        return self.input_shape

    def get_output(self, X, *args, **kwargs):
        return X


class PartialInputLayer(InputLayer):
    def __init__(self, idx, *args, **kwargs):
        super(PartialInputLayer, self).__init__(*args, **kwargs)
        self.idx = idx

    def initialize(self, X, y):
        if not (isinstance(X, tuple) or isinstance(X, list)):
            raise TypeError("Input of MultipleInputLayer must be a list or "
                            "tuple, instead got {}.".format(type(X)))
        if self.idx > len(X):
            raise ValueError("You are asking for the {}'th input but there "
                             "are only {} inputs.".format(self.idx, len(X)))
        super(PartialInputLayer, self).initialize(X[self.idx], y)

    def get_output(self, X, *args, **kwargs):
        return X[self.idx]


class InputConcatLayer(BaseLayer):
    def initialize(self, X, y):
        input_shapes = [layer.get_output_shape() for
                        layer in self.prev_layer]
        self.input_shapes = input_shapes
        if not np.equal(*[len(input_shape) for input_shape in input_shapes]):
            raise ValueError("Input dimensions for InputConcatLayer should "
                             "all be equal.")
        for d in range(len(input_shapes[0])):
            if d == 1:
                continue
            if not np.equal(*[input_shape[d] for input_shape in input_shapes]):
                raise ValueError("All input shapes except for axis one "
                                 "must be equal for InputConcatLayer.")

    def set_updater(self, *args, **kwargs):
        pass

    def set_prev_layer(self, layer):
        if not self.prev_layer:
            self.prev_layer = [layer]
        else:
            self.prev_layer.append(layer)

    def get_grads(self, cost):
        return [None]

    def get_params(self):
        return [None]

    def get_output(self, X, *args, **kwargs):
        outputs = [layer.get_output(X, *args, **kwargs) for
                   layer in self.prev_layer]
        return T.concatenate(outputs, axis=1)

    def get_output_shape(self):
        # concatenate along the 1st axis
        shape = []
        input_shapes = self.input_shapes
        for d in range(len(self.input_shapes[0])):
            if d != 1:
                shape.append(input_shapes[0][d])
            else:
                shape_concat = sum([input_shape[1] for
                                    input_shape in input_shapes])
                shape.append(shape_concat)
        return tuple(shape)


class EmbeddingLayer(BaseLayer):
    def __init__(self, num_units, num_features, *args, **kwargs):
        super(EmbeddingLayer, self).__init__(*args, **kwargs)
        self.num_units = num_units
        self.num_features = num_features

    def initialize(self, X, y):
        self.input_shape = (None, self.num_units)

        # item or word embeddings
        self.W = self.create_param(
            scheme=self.init_scheme,
            shape=(self.num_features, self.num_units),
            name='W_{}'.format(self.name),
        )
        self.params = [self.W]

    def get_output(self, X, *args, **kwargs):
        return self.W[X]

    def get_output_shape(self):
        return self.input_shape[0], self.num_units

    def get_l2_cost(self):
        return 0.5 * self.lambda2 * T.sum(self.W ** 2)
