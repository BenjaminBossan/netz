# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import pandas as pd
import pytest

from ..costfunctions import crossentropy
from ..iterators import MultipleInputsBatchIterator
from ..layers import Conv2DLayer
from ..layers import DenseLayer
from ..layers import DropoutLayer
from ..layers import InputConcatLayer
from ..layers import InputLayer
from ..layers import MaxPool2DLayer
from ..layers import OutputLayer
from ..layers import PartialInputLayer
from ..neuralnet import MultipleInputNet
from ..neuralnet import NeuralNet
from ..nonlinearities import rectify
from ..updaters import Adadelta
from ..updaters import Momentum
from ..updaters import SGD
from ..utils import flatten
from ..utils import occlusion_heatmap
from tutils import check_weights_shrink
from tutils import check_relative_diff_similar

np.random.seed(17411)
# Number of numerically checked gradients per paramter (more -> slower)
MAX_ITER = 20
# data
df = pd.read_csv('netz/tests/mnist_short.csv')
X = (df.values[:, 1:] / 255).astype(np.float32)
X = (X - X.mean()) / X.std()
y = df.values[:, 0].astype(np.int32)
X2D = X.reshape(-1, 1, 28, 28)
NUM_CLASSES = len(np.unique(y))


class TestVanillaNet:
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

    @pytest.fixture(scope='session')
    def untrained_net(self):
        layers = [InputLayer(),
                  DenseLayer(100),
                  OutputLayer()]
        net = NeuralNet(
            layers, cost_function=crossentropy,
            eval_size=0.5,
        )
        return net

    @pytest.fixture(scope='session')
    def net2(self):
        layers = [InputLayer(),
                  DenseLayer(100),
                  DenseLayer(33),
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

    def test_save_and_load_params_with_uninitialized(self, untrained_net):
        with pytest.raises(AttributeError):
            untrained_net.load_params('some_params')

    def test_save_and_load_params_with_initialized(self, net, untrained_net):
        filename = 'temp_params.np'
        params_old = [layer.get_params() for layer in net.layers]
        net.save_params(filename)
        score_old = net.score(X, y)

        untrained_net.initialize(X, y)
        untrained_net.load_params(filename)
        params_new = [layer.get_params() for layer in untrained_net.layers]
        score_new = untrained_net.score(X, y)

        # assert score_new == score_old
        assert np.isclose(score_old, score_new)
        for param_old, param_new in zip(params_old, params_new):
            for p_old, p_new in zip(param_old, param_new):
                if p_old:
                    assert (p_old.get_value() == p_new.get_value()).all()

    def test_load_params_matches_shapes(self, net, net2):
        filename = 'temp_params.np'
        params1 = flatten(layer.get_params() for layer in net.layers)
        params1 = [p.get_value() for p in params1 if p]
        params2 = flatten(layer.get_params() for layer in net2.layers)
        params2 = [p.get_value() for p in params2 if p]

        # before, parameters should differ
        assert not (params1[0] == params2[0]).all()
        assert not (params1[1] == params2[1]).all()
        assert not (params1[3] == params2[5]).all()

        net.save_params(filename)
        net2.load_params(filename)
        params2 = flatten(layer.get_params() for layer in net2.layers)
        params2 = [p.get_value() for p in params2 if p]

        # afterwards, these parameters should match
        assert (params1[0] == params2[0]).all()
        assert (params1[1] == params2[1]).all()
        assert (params1[3] == params2[5]).all()


class TestOverfittingNet:
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
        assert np.isclose(net.train_history_[-1], 0., atol=1e-2)
        assert not np.isclose(net.valid_history_[-1], 0., atol=1e-2)


class TestSgdNet:
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
        yt = np.argmax(net.encoder.transform(y.reshape(-1, 1)), axis=1)
        assert (yt == y).all()


class TestConcatNet:
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


class TestReguNet:
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


class TestConvDropoutNet:
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


class TestMagicMethodsNet:
    @pytest.fixture(scope='function')
    def net_and_layers(self):
        layers = [InputLayer(),
                  DenseLayer(100),
                  OutputLayer()]
        net = NeuralNet(layers)
        return net, layers

    def test_initial_layers(self, net_and_layers):
        net, layers = net_and_layers

        for layer_net, layer_layer in zip(net, layers):
            assert layer_net is layer_layer

    def test_adding_a_layer(self, net_and_layers):
        net, layers = net_and_layers
        layer_new = DenseLayer(123)
        net = net + layer_new

        assert net[-1] is layer_new

    def test_incrementally_adding_a_layer(self, net_and_layers):
        net, layers = net_and_layers
        layer_new = InputLayer(5)
        net += layer_new

        assert net[-1] is layer_new

    def test_incrementally_adding_tuple_of_layers(self, net_and_layers):
        net, layers = net_and_layers
        layers_new = (OutputLayer(), DenseLayer(9))
        net += layers_new

        assert net[-2] is layers_new[0]
        assert net[-1] is layers_new[1]

    def test_incrementally_adding_tuple_of_layers(self, net_and_layers):
        net, layers = net_and_layers
        layers_new = [InputLayer(), OutputLayer(), DenseLayer(1)]
        net += layers_new

        assert net[-3] is layers_new[0]
        assert net[-2] is layers_new[1]
        assert net[-1] is layers_new[2]

    @pytest.mark.parametrize('idx', range(-3, 3))
    def test_indexing_net(self, net_and_layers, idx):
        net = net_and_layers[0]

        assert net[idx] == net.layers[idx]

    @pytest.mark.parametrize('idx', range(-3, 3))
    def test_slicing_net(self, net_and_layers, idx):
        net = net_and_layers[0]

        assert net[idx:] == net.layers[idx:]
        assert net[:idx] == net.layers[:idx]


class TestMultipleInputNets:
    @pytest.fixture(scope='function')
    def net(self):
        layers = [PartialInputLayer(idx=0, name='partial0'),
                  PartialInputLayer(idx=1, name='partial1'),
                  DenseLayer(100, name='dense0'),
                  DenseLayer(100, name='dense1'),
                  InputConcatLayer(name='concat'),
                  OutputLayer(name='output')]
        connection_pattern = '''
        partial0->dense0
        partial1->dense1
        dense0->concat
        dense1->concat
        concat->output
        '''
        net = MultipleInputNet(
            layers,
            iterator=MultipleInputsBatchIterator(5),
            connection_pattern=connection_pattern
        )
        return net

    def test_multiple_input_net_trains(self, net):
        net.fit([X, X], y, max_iter=5)
        net.fit((X, X[::-1]), y, max_iter=5)

    def test_raises_type_error_when_not_list_or_tuple(self, net):
        with pytest.raises(TypeError):
            net.fit(X, y, max_iter=5)
