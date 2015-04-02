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

from nonlinearities import sigmoid
from nonlinearities import softmax
from utils import shared_random_normal
from utils import shared_random_orthogonal
from utils import shared_random_uniform
from utils import shared_ones
from utils import shared_zeros
from utils import to_32

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
            init_scheme='Xavier',
    ):
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.nonlinearity = nonlinearity
        self.params = params
        self.name = name
        self.updater = updater
        self.lambda2 = lambda2
        self.init_scheme = init_scheme

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

    @staticmethod
    def create_param(shape, scheme='Xavier', name=None,
                     broadcastable=None):
        """ Currently 3 supported schemes:

        * He (default) : He et al. 2015
          ~ N[+/- sqrt(2 / num_units)]
        * Xavier : Glorot, Bengio 2010
          ~ U[+/- sqrt(6 / (fan_in + fan_out))]
          The assumptions work for dense and convolutional layers
        * zeros
          A tensor of just zeros
        * ones
          A tensor of just ones
        * orthogonal
          Orthogonal matrix initialization
          The assumptions work for dense and convolutional layers

        """
        schemes_known = ['He', 'Xavier', 'Zeros', 'Ones', 'Orthogonal']
        scheme_variants = schemes_known + [s.lower() for s in schemes_known]
        if scheme not in scheme_variants:
            raise TypeError("The proposed scheme {} is not supported, we only "
                            "support {}".format(', '.join(schemes_known)))

        if scheme.lower() == 'he':
            num_units = reduce(op.mul, shape)
            return shared_random_normal(shape, num_units, name, broadcastable)
        elif scheme.lower() == 'xavier':
            receptive_field_size = np.prod(shape[2:])
            high = np.sqrt(6 / reduce(op.add, shape[:2]) /
                           receptive_field_size)
            low = -high
            return shared_random_uniform(shape, low, high, name, broadcastable)
        elif scheme.lower() == 'zeros':
            return shared_zeros(shape, name, broadcastable)
        elif scheme.lower() == 'ones':
            return shared_ones(shape, name, broadcastable)
        elif scheme.lower() == 'orthogonal':
            return shared_random_orthogonal(shape, name, broadcastable)

    def get_l2_cost(self):
        pass

    def get_updates(self, cost):
        grads = self.get_grads(cost)
        params = self.get_params()
        return self.updater.get_updates(cost, grads, params)

    def __repr__(self):
        if self.name is not None:
            return self.name
        else:
            return super(BaseLayer, self).__repr__()


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
            scheme=self.init_scheme,
            shape=(self.num_features, self.num_units),
            name='W_{}'.format(self.name),
        )
        self.b = self.create_param(
            scheme=self.init_scheme,
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
    def __init__(self, ds=2, pool_function=T.max, *args, **kwargs):
        super(FeaturePoolLayer, self).__init__(*args, **kwargs)
        self.ds = ds
        self.pool_function = pool_function

    def initialize(self, X, y):
        super(FeaturePoolLayer, self).initialize(X, y)
        self.updater = None

    def get_params(self):
        return [None]

    @staticmethod
    def _get_pooled_shape(shape, ds):
        num_feature_maps = shape[1]
        num_feature_maps_out = num_feature_maps // ds

        pool_shape = list(shape)
        pool_shape.insert(1, num_feature_maps_out)
        pool_shape[2] = ds
        return pool_shape

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)

        pool_shape = self._get_pooled_shape(input.shape, self.ds)
        input_reshaped = input.reshape(pool_shape)
        output = self.pool_function(input_reshaped, axis=2)

        return output

    def get_output_shape(self):
        input_shape = self.prev_layer.get_output_shape()
        output_shape = self._get_pooled_shape(input_shape, self.ds)
        return tuple(output_shape[:2] + output_shape[3:])


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

    def set_updater(self, *args, **kwargs):
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


class RecurrentLayer(BaseLayer):
    def __init__(self, num_units, num_features=None, *args, **kwargs):
        super(RecurrentLayer, self).__init__(*args, **kwargs)
        self.num_units = num_units
        self.num_features = num_features

    def initialize(self, X, y):
        super(RecurrentLayer, self).initialize(X, y)

        if self.num_features is None:
            self.num_features = self.prev_layer.get_output_shape()[1]

        # weights relating to input at t
        self.Wh = self.create_param(
            scheme=self.init_scheme,
            shape=(self.num_features, self.num_units),
            name='W_{}'.format(self.name),
        )
        # weights relating to hidden state from t-1
        self.Uh = self.create_param(
            scheme=self.init_scheme,
            shape=(self.num_units, self.num_units),
            name='W_{}'.format(self.name),
        )
        # bias
        self.bh = self.create_param(
            scheme=self.init_scheme,
            shape=(1, self.num_units),
            name='b_{}'.format(self.name),
            broadcastable=(True, False),
        )
        self.params = [self.Wh, self.Uh, self.bh]

    def _step(self, xt, htm1):
        Wh, Uh, bh = self.Wh, self.Uh, self.bh
        ht = self.nonlinearity(T.dot(xt, Wh) + T.dot(htm1, Uh) + bh)
        return ht

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)
        h0 = self._step(input[:1], T.zeros_like(self.bh))
        output = theano.scan(
            self._step,
            sequences=[input[1:]],
            non_sequences=[h0],
        )[0]
        return output[-1]

    def get_output_shape(self):
        return (1, self.num_units)

    def get_l2_cost(self):
        return 0.5 * self.lambda2 * T.sum(self.Wh ** 2)


class GRULayer(BaseLayer):
    """ Gated Recurrent Layer

    see Chung et al. 2014
    """
    def __init__(
            self,
            num_units,
            num_features=None,
            nonlinearity_r=None,
            nonlinearity_z=None,
            *args, **kwargs):
        super(GRULayer, self).__init__(*args, **kwargs)
        self.num_units = num_units
        self.num_features = num_features
        self.nonlinearity_r = nonlinearity_r
        self.nonlinearity_z = nonlinearity_z

    def initialize(self, X, y):
        super(GRULayer, self).initialize(X, y)

        if self.num_features is None:
            self.num_features = self.prev_layer.get_output_shape()[1]
        if self.nonlinearity_r is None:
            self.nonlinearity_r = self.nonlinearity
        if self.nonlinearity_z is None:
            self.nonlinearity_z = self.nonlinearity

        # hidden weights relating to input at t
        Wh = self.create_param(
            scheme=self.init_scheme,
            shape=(self.num_features, self.num_units),
            name='Wh_{}'.format(self.name),
        )
        # hidden weights relating to hidden state from t-1
        Uh = self.create_param(
            scheme=self.init_scheme,
            shape=(self.num_units, self.num_units),
            name='Uh_{}'.format(self.name),
        )
        # hidden bias
        bh = self.create_param(
            scheme=self.init_scheme,
            shape=(1, self.num_units),
            name='bh_{}'.format(self.name),
            broadcastable=(True, False),
        )
        # weights of the update gate relating to input at t
        Wz = self.create_param(
            scheme=self.init_scheme,
            shape=(self.num_features, self.num_units),
            name='W_{}'.format(self.name),
        )
        # weights of the update gate relating to hidden state from t-1
        Uz = self.create_param(
            scheme=self.init_scheme,
            shape=(self.num_units, self.num_units),
            name='W_{}'.format(self.name),
        )
        # update gate bias
        bz = self.create_param(
            scheme=self.init_scheme,
            shape=(1, self.num_units),
            name='bz_{}'.format(self.name),
            broadcastable=(True, False),
        )
        # weights of the reset gate relating to input at t
        Wr = self.create_param(
            scheme=self.init_scheme,
            shape=(self.num_features, self.num_units),
            name='W_{}'.format(self.name),
        )
        # weights of the reset gate relating to hidden state from t-1
        Ur = self.create_param(
            scheme=self.init_scheme,
            shape=(self.num_units, self.num_units),
            name='W_{}'.format(self.name),
        )
        # reset gate bias
        br = self.create_param(
            scheme=self.init_scheme,
            shape=(1, self.num_units),
            name='br_{}'.format(self.name),
            broadcastable=(True, False),
        )
        self.Wh, self.Uh, self.bh = Wh, Uh, bh
        self.Wz, self.Uz, self.bz = Wz, Uz, bz
        self.Wr, self.Ur, self.br = Wr, Ur, br
        self.params = [Wh, Uh, bh, Wz, Uz, bz, Wr, Ur, br]

    def _step(self, xt, htm1):
        Wh, Uh, bh = self.Wh, self.Uh, self.bh
        Wr, Ur, br = self.Wr, self.Ur, self.br
        Wz, Uz, bz = self.Wz, self.Uz, self.bz

        # reset gate
        rt = self.nonlinearity_r(T.dot(xt, Wr) + T.dot(htm1, Ur) + br)
        # candidate activity
        h_tilda = self.nonlinearity(T.dot(xt, Wh) + T.dot(rt * htm1, Uh) + bh)
        # update gate
        zt = self.nonlinearity_z(T.dot(xt, Wz) + T.dot(htm1, Uz) + bz)
        # activatation
        ht = (1. - zt) * htm1 + zt * h_tilda
        return ht

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)
        h0 = self._step(input[:1], T.zeros_like(self.bh))
        output = theano.scan(
            self._step,
            sequences=[input[1:]],
            non_sequences=[h0],
        )[0]
        return output[-1]

    def get_output_shape(self):
        return (1, self.num_units)

    def get_l2_cost(self):
        weights_squared = T.sum(self.Wh ** 2) + T.sum(self.Wz ** 2)
        weights_squared += T.sum(self.Wr ** 2)
        return 0.5 * self.lambda2 * weights_squared
