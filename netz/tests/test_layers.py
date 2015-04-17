# -*- coding: utf-8 -*-
from __future__ import division

from mock import Mock
import numpy as np
import pandas as pd
import pytest
import theano

from ..costfunctions import crossentropy
from ..layers import Conv2DLayer
from ..layers import DenseLayer
from ..layers import DropoutLayer
from ..layers import FeaturePoolLayer
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

        assert net[0].updater is None
        for layer in net[1:]:
            assert type(layer.updater) == type(default_updater)

    def test_layers_no_updater_net_other_updater(self):
        layers1 = [InputLayer(),
                   DenseLayer(3),
                   DenseLayer(4),
                   OutputLayer()]
        net_updater = Nesterov()
        net = NeuralNet(layers1, updater=net_updater)
        net.initialize(X, y)

        assert net[0].updater is None
        for layer in net[1:]:
            assert type(layer.updater) == type(net_updater)
            assert layer.updater is net[1].updater

    def test_layers_one_updater_net_no_updater(self):
        layers2 = [InputLayer(),
                   DenseLayer(3, updater=Momentum()),
                   DenseLayer(4),
                   OutputLayer()]
        net = NeuralNet(layers2)
        net.initialize(X, y)
        default_updater = get_default_arg(NeuralNet.__init__, 'updater')

        assert net[0].updater is None
        assert type(net[1].updater) == type(Momentum())
        for layer in net[2:]:
            assert type(layer.updater) == type(default_updater)
            assert layer.updater is net[2].updater

    def test_layers_one_updater_net_other_updater(self):
        layers2 = [InputLayer(),
                   DenseLayer(3, updater=Momentum()),
                   DenseLayer(4),
                   OutputLayer()]
        net_updater = Nesterov()
        net = NeuralNet(layers2, updater=net_updater)
        net.initialize(X, y)

        assert net[0].updater is None
        assert type(net[1].updater) == type(Momentum())
        for layer in net[2:]:
            assert type(layer.updater) == type(net_updater)

    def test_layers_several_updaters(self):
        layers3 = [InputLayer(),
                   DenseLayer(3, updater=Momentum()),
                   DenseLayer(4, updater=Nesterov()),
                   OutputLayer(updater=Momentum(momentum=555.))]
        net = NeuralNet(layers3)
        net.initialize(X, y)
        default_momentum = get_default_arg(Momentum.__init__, 'momentum')

        assert net[0].updater is None
        assert type(net[1].updater) == type(Momentum())
        assert np.allclose(net[1].updater.momentum,
                           default_momentum, atol=1e-6)
        assert type(net[2].updater) == type(Nesterov())
        assert type(net[3].updater) == type(Momentum())
        assert np.allclose(net[3].updater.momentum, 555., atol=1e-6)


class TestFeaturePoolLayer:
    @pytest.mark.parametrize('ds, expected', [
        (1,  (0, 24,  1, 0, 0)),
        (2,  (0, 12,  2, 0, 0)),
        (4,  (0,  6,  4, 0, 0)),
        (12, (0,  2, 12, 0, 0)),
        (24, (0,  1, 24, 0, 0)),
    ])
    def test_pooled_shape_plus1_different_ds(self, ds, expected):
        shape = (0, 24, 0, 0)
        axis = 1

        layer = FeaturePoolLayer(ds=ds, axis=axis)
        result = layer._get_pooled_shape_plus1(shape, ds, axis)

        assert tuple(result) == expected

    @pytest.mark.parametrize('axis, expected', [
        (0,  (5, 2, 20, 30, 40)),
        (1, (10, 10, 2, 30, 40)),
        (2, (10, 20, 15, 2, 40)),
        (3, (10, 20, 30, 20, 2)),
    ])
    def test_pooled_shape_plus1_different_axis(self, axis, expected):
        shape = (10, 20, 30, 40)
        ds = 2

        layer = FeaturePoolLayer(ds=ds, axis=axis)
        result = layer._get_pooled_shape_plus1(shape, ds, axis)

        assert tuple(result) == expected

    def test_feature_pool_layer_calls_custom_function(self):
        prev_layer = Mock()
        prev_layer.get_output.return_value = np.zeros((1, 2, 3, 4))
        pool_func = Mock()
        layer = FeaturePoolLayer(ds=2, axis=3, pool_function=pool_func)
        layer.prev_layer = prev_layer
        layer.get_output(np.zeros((1, 2, 3, 4)))

        assert layer.pool_function.call_count == 1

    @pytest.mark.parametrize('axis, expected', [
        (0, (2, 6, 8, 10)),
        (1, (4, 3, 8, 10)),
        (2, (4, 6, 4, 10)),
        (3, (4, 6, 8,  5)),
    ])
    def test_shape_of_get_output(self, axis, expected):
        input = np.zeros((4, 6, 8, 10)).astype(theano.config.floatX)
        prev_layer = Mock()
        prev_layer.get_output.return_value = input
        layer = FeaturePoolLayer(ds=2, axis=axis)
        layer.prev_layer = prev_layer

        get_output = theano.function([], layer.get_output(input))
        result = get_output().shape

        assert result == expected
