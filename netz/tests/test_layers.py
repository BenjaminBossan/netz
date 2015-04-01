# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import pandas as pd
import pytest

from ..costfunctions import crossentropy
from ..layers import Conv2DLayer
from ..layers import DenseLayer
from ..layers import DropoutLayer
from ..layers import InputConcatLayer
from ..layers import InputLayer
from ..layers import MaxPool2DLayer
from ..layers import OutputLayer
from ..neuralnet import NeuralNet
from ..updaters import Momentum
from ..updaters import Nesterov
from tutils import get_default_arg

np.random.seed(17412)
# Number of numerically checked gradients per paramter (more -> slower)
# fake data
X = 2 * np.random.rand(10, 5).astype(np.float32)
X -= X.mean()
y = np.random.randint(low=0, high=3, size=10).astype(np.float32)


class TestLayerUpdater:
    def test_layers_no_updater_net_no_updater(self):
        layers1 = [InputLayer(),
                   DenseLayer(3),
                   DenseLayer(4),
                   OutputLayer()]
        net = NeuralNet(layers1)
        net.initialize(X, y)
        default_updater = get_default_arg(NeuralNet.__init__, 'updater')

        assert net.layers[0].updater is None
        for layer in net.layers[1:]:
            assert type(layer.updater) == type(default_updater)

    def test_layers_no_updater_net_other_updater(self):
        layers1 = [InputLayer(),
                   DenseLayer(3),
                   DenseLayer(4),
                   OutputLayer()]
        net_updater = Nesterov()
        net = NeuralNet(layers1, updater=net_updater)
        net.initialize(X, y)

        assert net.layers[0].updater is None
        for layer in net.layers[1:]:
            assert type(layer.updater) == type(net_updater)
            assert layer.updater is net.layers[1].updater

    def test_layers_one_updater_net_no_updater(self):
        layers2 = [InputLayer(),
                   DenseLayer(3, updater=Momentum()),
                   DenseLayer(4),
                   OutputLayer()]
        net = NeuralNet(layers2)
        net.initialize(X, y)
        default_updater = get_default_arg(NeuralNet.__init__, 'updater')

        assert net.layers[0].updater is None
        assert type(net.layers[1].updater) == type(Momentum())
        for layer in net.layers[2:]:
            assert type(layer.updater) == type(default_updater)
            assert layer.updater is net.layers[2].updater

    def test_layers_one_updater_net_other_updater(self):
        layers2 = [InputLayer(),
                   DenseLayer(3, updater=Momentum()),
                   DenseLayer(4),
                   OutputLayer()]
        net_updater = Nesterov()
        net = NeuralNet(layers2, updater=net_updater)
        net.initialize(X, y)

        assert net.layers[0].updater is None
        assert type(net.layers[1].updater) == type(Momentum())
        for layer in net.layers[2:]:
            assert type(layer.updater) == type(net_updater)

    def test_layers_several_updaters(self):
        layers3 = [InputLayer(),
                   DenseLayer(3, updater=Momentum()),
                   DenseLayer(4, updater=Nesterov()),
                   OutputLayer(updater=Momentum(momentum=555.))]
        net = NeuralNet(layers3)
        net.initialize(X, y)
        default_momentum = get_default_arg(Momentum.__init__, 'momentum')

        assert net.layers[0].updater is None
        assert type(net.layers[1].updater) == type(Momentum())
        assert np.allclose(net.layers[1].updater.momentum,
                           default_momentum, atol=1e-6)
        assert type(net.layers[2].updater) == type(Nesterov())
        assert type(net.layers[3].updater) == type(Momentum())
        assert np.allclose(net.layers[3].updater.momentum, 555., atol=1e-6)
