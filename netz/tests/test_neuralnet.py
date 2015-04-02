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
from ..nonlinearities import rectify
from ..updaters import Adadelta
from ..updaters import Momentum
from ..updaters import SGD
from ..utils import occlusion_heatmap
from tutils import check_weights_shrink
from tutils import check_relative_diff_similar

np.random.seed(17411)
# Number of numerically checked gradients per paramter (more -> slower)
NUM_CHECK = 3
EPSILON = 1e-6
MAX_ITER = 20
# data
df = pd.read_csv('netz/tests/mnist_short.csv')
X = (df.values[:, 1:] / 255).astype(np.float32)
X = (X - X.mean()) / X.std()
y = df.values[:, 0].astype(np.int32)
X2D = X.reshape(-1, 1, 28, 28)
NUM_CLASSES = len(np.unique(y))


class TestVanillaNet():
    @pytest.fixture(scope='session')
    def net(self):
        layers = [InputLayer(),
                  DenseLayer(100),
                  OutputLayer()]
        net = NeuralNet(
            layers, cost_function=crossentropy,
            eval_size=0.5,
        )
        net.fit(X, y, max_iter=100)
        return net

    def test_initial_loss(self, net):
        # At initialization, the cost should be close to random:
        assert np.allclose(net.train_history_[0], -np.log(1 / 10), atol=0.5)
        assert np.allclose(net.valid_history_[0], -np.log(1 / 10), atol=0.5)


class TestOverfittingNet():
    n = 10  # number of samples

    @pytest.fixture(scope='session')
    def net(self):
        layers = [InputLayer(),
                  DenseLayer(500),
                  OutputLayer()]
        net = NeuralNet(
            layers, cost_function=crossentropy,
            updater=Adadelta(),
            eval_size=0.5,
        )
        net.fit(X[:self.n], y[:self.n], max_iter=100)
        return net

    def test_net_learns_small_sample_by_heart(self, net):
        assert np.allclose(net.train_history_[-1], 0., atol=1e-2)
        assert not np.allclose(net.valid_history_[-1], 0., atol=1e-2)


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
        net.fit(X, y, max_iter=MAX_ITER)
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
        net.fit(X, y, max_iter=MAX_ITER)
        return net

    def test_connections(self, net):
        layers = net.layers
        assert layers[0].prev_layer is None
        assert layers[0].next_layer is layers[1]
        assert layers[1].prev_layer is layers[0]
        assert layers[1].next_layer is layers[3]
        assert layers[2].prev_layer is layers[0]
        assert layers[2].next_layer is layers[3]
        assert layers[3].prev_layer[0] is layers[1]
        assert layers[3].prev_layer[1] is layers[2]
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
                   DenseLayer(100, lambda2=0.2),
                   OutputLayer()]

        net0 = NeuralNet(layers0)
        net1 = NeuralNet(layers1)

        net0.fit(X, y, max_iter=MAX_ITER)
        net1.fit(X, y, max_iter=MAX_ITER)
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


class TestConvDropoutNet():
    @pytest.fixture(scope='session')
    def net(self):
        layers = [InputLayer(),
                  Conv2DLayer(3, (3, 3), nonlinearity=rectify),
                  MaxPool2DLayer(),
                  DropoutLayer(p=0.2),
                  DenseLayer(10),
                  OutputLayer()]
        net = NeuralNet(layers, cost_function=crossentropy,
                        updater=Adadelta())
        net.fit(X2D, y, max_iter=20)
        return net

    def test_occlusion_min_not_in_image_border(self, net):
        border_size = 4
        x_size = X2D.shape[3]
        y_size = X2D.shape[2]
        for i in range(10):
            heatmap = occlusion_heatmap(net, X2D[i:i + 1], y[i])
            coord_y, coord_x = np.nonzero(heatmap == heatmap.min())
            assert border_size < coord_x < x_size - border_size
            assert border_size < coord_y < y_size - border_size
