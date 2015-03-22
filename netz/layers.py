# -*- coding: utf-8 -*-
from __future__ import division
import operator as op
import warnings

import numpy as np
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous
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
            lambda2=0,
    ):
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.nonlinearity = nonlinearity
        self.params = params
        self.name = name
        self.updater = updater
        self.lambda2 = lambda2

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

    def set_lambda2(self, lambda2):
        if self.lambda2 is not None:
            raise warnings.warn("You are overwriting the lambda2 parameter"
                                "for layer {}.".format(self.name))
        self.lambda2 = lambda2

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

    def create_param(self, shape, limits=(None, None),
                     name=None, broadcastable=None):
        low, high = limits
        high = high if high is not None else np.sqrt(6 / sum(shape))
        low = low if low is not None else -high
        return shared_random_uniform(shape=shape, low=low, high=high,
                                     name=name, broadcastable=broadcastable)

    def get_l2_cost(self):
        pass

    def get_updates(self, cost, layer):
        return self.updater.get_updates(cost, layer)


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

    def get_grads(self, cost):
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

    def get_l2_cost(self):
        return 0.5 * self.lambda2 * T.sum(self.W ** 2)


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

        mask = srng.binomial(input_shape, p=q)
        self.mask_ = mask
        return input * mask

    def get_output_shape(self):
        return self.input_shape


class BatchNormLayer(BaseLayer):
    def __init__(self, epsilon=1e-9, decay=0.1, *args, **kwargs):
        super(BatchNormLayer, self).__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.decay = decay

    def initialize(self, X, y):
        super(BatchNormLayer, self).initialize(X, y)

        shape = list(self.input_shape)
        ndim = len(shape)
        if ndim == 2:
            self.axes_ = (0,)
            ema_broadcastable = (True, False)
        elif ndim == 4:
            self.axes_ = (0, 2, 3)
            ema_broadcastable = (True, False, True, True)
        for axis in self.axes_:
            shape[axis] = 1

        self.mean_ema_ = self.create_param(
            shape=shape,
            limits=(0., 0.),
            name='mean_ema_'.format(self.name),
            broadcastable=ema_broadcastable,
        )
        self.std_ema_ = self.create_param(
            shape=shape,
            limits=(0., 0.),
            name='std_ema_'.format(self.name),
            broadcastable=ema_broadcastable,
        )
        self.gamma = self.create_param(
            shape=shape,
            limits=(0.95, 1.05),
            name='gamma_{}'.format(self.name),
        )
        self.beta = self.create_param(
            shape=shape,
            limits=(0., 0.),
            name='beta_{}'.format(self.name),
        )

    def get_params(self):
        return [self.gamma, self.beta]

    def get_updates(self, *args, **kwargs):
        updates = super(BatchNormLayer, self).get_updates(*args, **kwargs)
        # additionally update mean and std estimations
        more_updates = [(self.mean_ema_, self.mean_ema_new_),
                        (self.std_ema_, self.std_ema_new_)]
        return updates + more_updates

    def get_output_shape(self):
        return self.input_shape

    def _get_mean_std_ema(self, input, deterministic):
        if deterministic:
            return self.mean_ema_, self.std_ema_

        mean_batch = T.mean(input, self.axes_, keepdims=True)
        mean = (1 - self.decay) * self.mean_ema_ + self.decay * mean_batch
        mean = T.addbroadcast(mean, *self.axes_)

        std_batch = T.sqrt(T.var(input, self.axes_, keepdims=True) +
                           self.epsilon)
        std = (1 - self.decay) * self.std_ema_ + self.decay * std_batch
        std = T.addbroadcast(std, *self.axes_)
        return mean, std

    def get_output(self, X, deterministic, *args, **kwargs):
        input = self.prev_layer.get_output(X, deterministic, *args, **kwargs)

        mean, std = self._get_mean_std_ema(input, deterministic)
        gamma = T.addbroadcast(self.gamma, *self.axes_)
        beta = T.addbroadcast(self.beta, *self.axes_)

        input_norm = (input - mean) / std
        input_trans = gamma * input_norm + beta

        self.mean_ema_new_ = mean
        self.std_ema_new_ = std
        return self.nonlinearity(input_trans)


class InputConcatLayer(BaseLayer):
    def __init__(self, prev_layers, *args, **kwargs):
        super(InputConcatLayer, self).__init__(*args, **kwargs)
        self.prev_layers = prev_layers

    def initialize(self, X, y):
        input_shapes = [layer.get_output_shape() for
                        layer in self.prev_layers]
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

    def set_update(self, *args, **kwargs):
        pass

    def set_prev_layer(self, layer):
        pass

    def get_grads(self, cost):
        return [None]

    def get_params(self):
        return [None]

    def get_output(self, X, *args, **kwargs):
        outputs = [layer.get_output(X, *args, **kwargs) for
                   layer in self.prev_layers]
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
