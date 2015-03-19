# -*- coding: utf-8 -*-
from __future__ import division
from copy import deepcopy

import matplotlib.pyplot as plt
from mock import patch
import numpy as np
import pandas as pd
import pytest
from theano import shared

from ..costfunctions import crossentropy
from ..layers import DenseLayer
from ..layers import InputConcatLayer
from ..layers import InputLayer
from ..layers import OutputLayer
from ..neuralnet import NeuralNet
from ..updaters import Momentum
from ..updaters import Nesterov
from ..updaters import SGD

np.random.seed(17411)
# Number of numerically checked gradients per paramter (more -> slower)
NUM_CHECK = 3
EPSILON = 1e-6
df = pd.read_csv('netz/tests/mnist_short.csv')
X = df.values[:, 1:] / 255
X = (X - X.mean()) / X.std()
y = df.values[:, 0]
X2D = X.reshape(-1, 1, 28, 28)
NUM_CLASSES = len(np.unique(y))


def check_weights_shrink(v0, v1):
    relative_diff = v1 / v0
    return (0 < relative_diff).all() & (relative_diff < 1).all()


def check_relative_diff_similar(v0, v1, atol=0.2):
    relative_diff = v1 / v0
    mean_diff = np.mean(relative_diff)
    return np.allclose(mean_diff, relative_diff, atol=atol)


class TestSgdNet():
    @pytest.fixture(scope='session')
    def net(self):
        layers = [InputLayer(),
                  DenseLayer(100, updater=Momentum()),
                  DenseLayer(200, lambda2=0.01),
                  OutputLayer()]
        net = NeuralNet(
            layers, cost_function=crossentropy,
            updater=SGD(),
            eval_size=0,
        )
        net.fit(X, y, max_iter=1)
        return net

    def test_names(self, net):
        assert net.layers[0].name == 'input0'
        assert net.layers[1].name == 'dense0'
        assert net.layers[2].name == 'dense1'
        assert net.layers[3].name == 'output0'

    def test_updaters(self, net):
        assert net.layers[0].updater is None
        assert isinstance(net.layers[1].updater, Momentum)
        assert isinstance(net.layers[2].updater, SGD)
        assert isinstance(net.layers[3].updater, SGD)

    def test_lambda(self, net):
        assert net.layers[0].lambda2 == 0
        assert net.layers[1].lambda2 == 0
        assert net.layers[2].lambda2 == 0.01
        assert net.layers[3].lambda2 == 0

    def test_connections(self, net):
        for layer0, layer1 in zip(net.layers[:-1], net.layers[1:]):
            assert layer0.next_layer is layer1
            assert layer1.prev_layer is layer0

    def test_encoder(self, net):
        yt = np.argmax(net.encoder_.transform(y.reshape(-1, 1)), axis=1)
        assert (yt == y).all()


class TestConcatNet():
    @pytest.fixture(scope='session')
    def net(self):
        layers = [InputLayer(),
                  DenseLayer(100),
                  DenseLayer(200),
                  OutputLayer()]
        layers.insert(3, InputConcatLayer([layers[1], layers[2]]))
        layers[1].set_next_layer(layers[3])
        layers[2].set_prev_layer(layers[0])
        net = NeuralNet(
            layers, cost_function=crossentropy,
            updater=SGD(),
            eval_size=0
        )
        net.fit(X, y, max_iter=1)
        return net

    def test_connections(self, net):
        layers = net.layers
        assert layers[0].prev_layer is None
        assert layers[0].next_layer is layers[1]
        assert layers[1].prev_layer is layers[0]
        assert layers[1].next_layer is layers[3]
        assert layers[2].prev_layer is layers[0]
        assert layers[2].next_layer is layers[3]
        assert layers[3].prev_layers[0] is layers[1]
        assert layers[3].prev_layers[1] is layers[2]
        assert layers[3].prev_layer is None
        assert layers[3].next_layer is layers[4]
        assert layers[4].prev_layer is layers[3]
        assert layers[4].next_layer is None

    def test_output_shape(self, net):
        assert net.layers[3].get_output_shape() == (None, 300)
        assert net.layers[4].W.get_value().shape == (300, NUM_CLASSES)


class TestReguNet():
    @pytest.fixture(scope='session')
    def nets(self):
        layers0 = [InputLayer(),
                   DenseLayer(100),
                   OutputLayer()]
        layers1 = [InputLayer(),
                   DenseLayer(100, lambda2=0.1),
                   OutputLayer()]

        net0 = NeuralNet(layers0)
        net1 = NeuralNet(layers1)

        net0.fit(X, y, max_iter=10)
        net1.fit(X, y, max_iter=10)
        return net0, net1

    def test_l2_regu_weights(self, nets):
        w0 = nets[0].layers[1].W.get_value()
        w1 = nets[1].layers[1].W.get_value()
        v0 = np.sort(w0.flatten())
        v1 = np.sort(w1.flatten())
        # exclude middle because of possible change of sign
        split = 2 * len(v0) // 5
        v0 = np.concatenate((v0[:split], v0[-split:]))
        v1 = np.concatenate((v1[:split], v1[-split:]))
        assert check_weights_shrink(v0, v1)
        assert check_relative_diff_similar(v0, v1)
        assert not np.allclose(v0 - v1, 0, atol=1e-3)

    def test_l2_regu_bias(self, nets):
        b0 = np.sort(nets[0].layers[1].b.get_value())
        b1 = np.sort(nets[0].layers[1].b.get_value())
        split = 2 * len(b0) // 5
        b0 = np.concatenate((b0[:split], b0[-split:]))
        b1 = np.concatenate((b1[:split], b1[-split:]))
        assert np.allclose(b0 - b1, 0, atol=1e-3)
