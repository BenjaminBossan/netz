# -*- coding: utf-8 -*-
from __future__ import division
import operator as op
import warnings

import numpy as np
import theano
from theano.tensor.shared_randomstreams import RandomStreams

from ..nonlinearities import sigmoid
from ..utils import shared_random_orthogonal
from ..utils import shared_random_uniform
from ..utils import shared_ones
from ..utils import shared_random_normal
from ..utils import shared_zeros

srng = RandomStreams(seed=17411)


__all__ = ['BaseLayer']


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
        if self.prev_layer is not None:
            warnings.warn("You are overriding the previous layer "
                          "of layer {}.".format(self.name))
        self.prev_layer = layer

    def set_next_layer(self, layer):
        if self.next_layer is not None:
            warnings.warn("You are overriding the previous layer "
                          "of layer {}.".format(self.name))
        self.next_layer = layer

    def set_updater(self, updater):
        if self.updater is not None:
            warnings.warn("You are overriding the updater set "
                          "for layer {}.".format(self.name))
        self.updater = updater

    def set_lambda2(self, lambda2):
        if self.lambda2 is not None:
            warnings.warn("You are overriding the lambda2 parameter "
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
        * eye
          identity matrix (i.e. 1 on diagonal and 0 else).

        """
        schemes_known = ['He', 'Xavier', 'Zeros', 'Ones', 'Orthogonal', 'Eye']
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
        elif scheme.lower() == 'eye':
            return shared_eye(shape, name, broadcastable)

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
