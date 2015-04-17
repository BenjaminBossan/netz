"""No smart way right now to test visualizations, so just check that
no exception is raised.

"""
# -*- coding: utf-8 -*-
from __future__ import division

from mock import patch
import numpy as np
import pandas as pd
import pytest

from ..costfunctions import crossentropy
from ..layers import Conv2DLayer
from ..layers import DenseLayer
from ..layers import InputLayer
from ..layers import MaxPool2DLayer
from ..layers import OutputLayer
from ..neuralnet import NeuralNet
from ..nonlinearities import rectify
from ..updaters import Adadelta
from ..utils import occlusion_heatmap
from ..visualize import plot_conv_weights
from ..visualize import plot_conv_activity
from ..visualize import plot_loss
from ..visualize import plot_occlusion


np.random.seed(17411)
# Number of numerically checked gradients per paramter (more -> slower)
MAX_ITER = 1
# data
df = pd.read_csv('netz/tests/mnist_short.csv')
X = (df.values[:, 1::2] / 255).astype(np.float32)
X = (X - X.mean()) / X.std()
y = df.values[:, 0].astype(np.int32)
X2D = X.reshape(-1, 1, 14, 14)


class TestVisualizePlotLoss:
    @pytest.fixture
    def net(self):
        layers = [InputLayer(),
                  Conv2DLayer(3, (3, 3), nonlinearity=rectify),
                  Conv2DLayer(3, (3, 3), nonlinearity=rectify),
                  MaxPool2DLayer(),
                  DenseLayer(10),
                  OutputLayer()]
        net = NeuralNet(layers, cost_function=crossentropy,
                        updater=Adadelta())
        net.fit(X2D, y, max_iter=MAX_ITER)
        return net

    def test_plot_loss_no_args(self, net):
        plot_loss(net)

    def test_plot_loss_args(self, net):
        plot_loss(net, 'ko')

    def test_plot_loss_kwargs(self, net):
        plot_loss(net, lw=2)

    def test_plot_loss_args_kwargs(self, net):
        plot_loss(net, 'r--', lw=1)


class TestVisualizePlotConvWeights:
    @pytest.fixture
    def net(self):
        layers = [InputLayer(),
                  Conv2DLayer(3, (3, 3), nonlinearity=rectify),
                  Conv2DLayer(3, (3, 3), nonlinearity=rectify),
                  MaxPool2DLayer(),
                  DenseLayer(10),
                  OutputLayer()]
        net = NeuralNet(layers, cost_function=crossentropy,
                        updater=Adadelta())
        net.fit(X2D, y, max_iter=MAX_ITER)
        return net

    def test_plot_conv_weights_layer_1(self, net):
        plot_conv_weights(net.layers[1])

    def test_plot_conv_weights_layer_2(self, net):
        plot_conv_weights(net.layers[2])

    def test_plot_conv_weights_with_figsize_arg(self, net):
        plot_conv_weights(net.layers[1], figsize=(1, 11))

    def test_plot_conv_activity_layer_1(self, net):
        plot_conv_activity(net.layers[1], X2D[0:1])

    def test_plot_conv_activity_layer_2(self, net):
        plot_conv_activity(net.layers[2], X2D[10:11])

    def test_plot_conv_activity_with_figsize_arg(self, net):
        plot_conv_activity(net[1], X2D[100:101], figsize=(8, 1))


class TestVisualizePlotConvWeights:
    @pytest.fixture
    def net(self):
        layers = [InputLayer(),
                  Conv2DLayer(3, (3, 3), nonlinearity=rectify),
                  Conv2DLayer(3, (3, 3), nonlinearity=rectify),
                  MaxPool2DLayer(),
                  DenseLayer(10),
                  OutputLayer()]
        net = NeuralNet(layers, cost_function=crossentropy,
                        updater=Adadelta())
        net.fit(X2D, y, max_iter=MAX_ITER)
        return net

    def test_plot_occlusion_one_x(self, net):
        plot_occlusion(net, X2D[1:2], y[1:2])

    def test_plot_occlusion_more_X(self, net):
        plot_occlusion(net, X2D[10:13], y[10:13])

    def test_plot_occlusion_with_square_length_arg(self, net):
        plot_occlusion(net, X2D[1:2], y[1:2], square_length=2)
        plot_occlusion(net, X2D[1:3], y[1:3], square_length=2)

    def test_plot_occlusion_with_figsize_partly(self, net):
        plot_occlusion(net, X2D[1:2], y[1:2], figsize=(1, None))
        plot_occlusion(net, X2D[1:3], y[1:3], figsize=(1, None))

    def test_plot_occlusion_with_figsize_full(self, net):
        plot_occlusion(net, X2D[1:2], y[1:2], figsize=(1, 1))
        plot_occlusion(net, X2D[1:3], y[1:3], figsize=(1, 2))

    def test_plot_occlusion_raises_value_error_from_x(self, net):
        with pytest.raises(ValueError):
            plot_occlusion(net, X[1:2], y[1:2])
