# -*- coding: utf-8 -*-
from __future__ import division

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from .base import BaseLayer

srng = RandomStreams(seed=17411)

__all__ = ['RecurrentLayer', 'GRULayer']


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
        weights_squared = T.sum(self.Wh ** 2) + T.sum(self.Uh ** 2)
        return 0.5 * self.lambda2 * weights_squared


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
        weights_squared = (T.sum(self.Wh ** 2) + T.sum(self.Uh ** 2) +
                           T.sum(self.Wz ** 2) + T.sum(self.Uz ** 2) +
                           T.sum(self.Wr ** 2) + T.sum(self.Ur ** 2))
        return 0.5 * self.lambda2 * weights_squared
