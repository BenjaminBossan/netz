# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import pytest

from ..costfunctions import crossentropy
from ..layers import DenseLayer
from ..layers import InputConcatLayer
from ..layers import InputLayer
from ..layers import OutputLayer
from ..neuralnet import NeuralNet
from ..updaters import Momentum
from ..updaters import SGD

np.random.seed(17411)
# Number of numerically checked gradients per paramter (more -> slower)
NUM_CHECK = 3
EPSILON = 1e-6
# fake data
X = np.random.rand(100, 8 * 4 * 4)
X2D = X.reshape(-1, 8, 4, 4)
y = np.random.randint(low=0, high=3, size=100).astype(float)


class TestSgdNet():
    @pytest.fixture(scope='session')
    def net(self):
        dense0_updater = Momentum()
        layers = [InputLayer(),
                  DenseLayer(100, updater=dense0_updater),
                  DenseLayer(200),
                  OutputLayer()]
        net = NeuralNet(
            layers, cost_function=crossentropy,
            updater=SGD(),
            eval_size=0
        )
        net.fit(X, y, max_iter=1)
        return net

    def test_names(self, net):
        assert net.layers[0].name == 'input0'
        assert net.layers[1].name == 'dense0'
        assert net.layers[2].name == 'dense1'
        assert net.layers[3].name == 'output0'

    def test_connections(self, net):
        for layer0, layer1 in zip(net.layers[:-1], net.layers[1:]):
            assert layer0.next_layer is layer1
            assert layer1.prev_layer is layer0


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
        assert net.layers[4].W.get_value().shape == (300, 3)
