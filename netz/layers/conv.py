# -*- coding: utf-8 -*-
from __future__ import division

from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.tensor.shared_randomstreams import RandomStreams

from .base import BaseLayer

srng = RandomStreams(seed=17411)

__all__ = ['Conv2DLayer', 'Conv2DCCLayer']


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
            scheme=self.init_scheme,
            shape=filter_shape,
            name='W_{}'.format(self.name),
        )
        self.b = self.create_param(
            scheme=self.init_scheme,
            shape=(self.num_filters,),
            name='b_{}'.format(self.name),
        )
        self.params = [self.W, self.b]

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)
        conv = T.nnet.conv2d(input, self.W, subsample=self.strides,
                             filter_shape=self.filter_shape_)
        activation = conv + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(activation)

    def get_output_shape(self):
        if self.strides != (1, 1):
            raise NotImplementedError
        nrows = self.input_shape[2] - self.filter_size[0] + 1
        ncols = self.input_shape[3] - self.filter_size[1] + 1
        return self.input_shape[0], self.num_filters, nrows, ncols

    def get_l2_cost(self):
        return 0.5 * self.lambda2 * T.sum(self.W ** 2)


class Conv2DCCLayer(Conv2DLayer):
    def __init__(
            self,
            num_filters,
            filter_size,
            strides=(1, 1),
            pad=0,
            *args,
            **kwargs
    ):
        super(Conv2DLayer, self).__init__(*args, **kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides
        self.pad = pad

    def initialize(self, X, y):
        if self.num_filters % 16 != 0:
            raise ValueError("num_filters must be multiple of 16, "
                             "not {}".format(self.num_filters))
        if self.filter_size[0] != self.filter_size[1]:
            raise ValueError("filter_size must be square")
        if self.strides[0] != self.strides[1]:
            raise ValueError("strides must be the same in x and y")
        super(Conv2DCCLayer, self).initialize(X, y)
        self.filter_acts_op = FilterActs(
            stride=self.strides[0], pad=self.pad, partial_sum=1)

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)

        # shuffle dimensions to cuda convnet default c01b
        input = input.dimshuffle(1, 2, 3, 0)
        filters = self.W.dimshuffle(1, 2, 3, 0)
        biases = self.b.dimshuffle(0, 'x', 'x', 'x')
        # make data gpu contiguous
        input = gpu_contiguous(input)
        filters = gpu_contiguous(filters)

        conv = self.filter_acts_op(input, filters) + biases
        activation = self.nonlinearity(conv)
        return activation.dimshuffle(3, 0, 1, 2)
