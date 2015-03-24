# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import pytest

from ..costfunctions import crossentropy
from ..layers import InputLayer
from ..layers import Conv2DLayer
from ..layers import DenseLayer
from ..layers import DropoutLayer
from ..layers import MaxPool2DLayer
from ..layers import OutputLayer
from ..neuralnet import NeuralNet
from ..nonlinearities import rectify
from ..updaters import Adadelta
from ..updaters import Adagrad
from ..updaters import Momentum
from ..updaters import Nesterov
from ..updaters import RMSProp
from ..updaters import SGD
from gradutils import GradChecker
from gradutils import verify_grad


np.random.seed(17411)
# Number of numerically checked gradients per paramter (more -> slower)
NUM_CHECK = 10
EPSILON = 1e-3
ATOL = 1e-3
# fake data
X = np.random.rand(10, 8 * 4 * 4).astype(np.float32)
X2D = X.reshape(-1, 8, 4, 4)
y = np.random.randint(low=0, high=3, size=10).astype(np.float32)


class BaseNetTest():
    def net(self):
        raise NotImplementedError

    def test_grad_theano(self, net):
        verify_grad(net, X, y)

    def test_grad_custom(self, net):
        gc = GradChecker(net, NUM_CHECK, EPSILON)
        theano_grads_all, numerical_grads_all = gc.get_grads(X, y)

        assert np.allclose(theano_grads_all, numerical_grads_all, atol=ATOL)
        # exclude error that gradients are just 0
        assert not np.allclose(theano_grads_all, 0., atol=ATOL)
        assert not np.allclose(numerical_grads_all, 0., atol=ATOL)


class BaseNetTest2D():
    def net(self):
        raise NotImplementedError

    def test_grad_theano(self, net):
        verify_grad(net, X2D, y)

    def test_grad_custom(self, net):
        gc = GradChecker(net, NUM_CHECK, EPSILON)
        theano_grads_all, numerical_grads_all = gc.get_grads(X2D, y)

        assert np.allclose(theano_grads_all, numerical_grads_all, atol=ATOL)
        # exclude error that gradients are just 0
        assert not np.allclose(theano_grads_all, 0., atol=ATOL)
        assert not np.allclose(numerical_grads_all, 0., atol=ATOL)


@pytest.mark.slow
class TestSgdNet(BaseNetTest):
    @pytest.fixture(scope='session')
    def net(self):
        layers = [InputLayer(),
                  DenseLayer(100, lambda2=0.01),
                  OutputLayer()]
        net = NeuralNet(
            layers, cost_function=crossentropy,
            updater=SGD(0.02),
            eval_size=0
        )
        net.fit(X, y, max_iter=3)
        return net


@pytest.mark.slow
class TestMomentumDifferentialUpdateNet(BaseNetTest):
    @pytest.fixture(scope='session')
    def net(self):
        updater_dense = Nesterov(momentum=0.95)
        layers = [InputLayer(),
                  DenseLayer(32),
                  DenseLayer(24, updater=updater_dense),
                  OutputLayer()]
        net = NeuralNet(layers, cost_function=crossentropy,
                        updater=Momentum(learn_rate=0.001))
        net.fit(X, y, max_iter=3)
        return net


@pytest.mark.slow
class TestAdadeltaL2RegularizationNet(BaseNetTest):
    @pytest.fixture(scope='session')
    def net(self):
        layers = [InputLayer(),
                  DenseLayer(32, lambda2=0.01),
                  DropoutLayer(),
                  DenseLayer(24, lambda2=0.01),
                  OutputLayer()]
        net = NeuralNet(layers, cost_function=crossentropy,
                        updater=Adadelta())
        net.fit(X, y, max_iter=3)
        return net


@pytest.mark.slow
class TestAdagradNet(BaseNetTest):
    @pytest.fixture(scope='session')
    def net(self):
        layers = [InputLayer(),
                  DenseLayer(32),
                  DropoutLayer(),
                  DenseLayer(24),
                  OutputLayer()]
        net = NeuralNet(layers, cost_function=crossentropy,
                        updater=Adagrad())
        net.fit(X, y, max_iter=3)
        return net


@pytest.mark.slow
class TestConvDropoutNet(BaseNetTest2D):
    @pytest.fixture(scope='session')
    def net(self):
        layers = [InputLayer(),
                  Conv2DLayer(3, (4, 4), nonlinearity=rectify),
                  MaxPool2DLayer(),
                  DropoutLayer(p=0.2),
                  DenseLayer(10),
                  OutputLayer()]
        net = NeuralNet(layers, cost_function=crossentropy,
                        updater=Nesterov())
        net.fit(X2D, y, max_iter=3)
        return net


# @pytest.mark.slow
# class TestRMSPropL2RegularizationNet(BaseNetTest):
#     @pytest.fixture(scope='session')
#     def net(self):
#         layers = [InputLayer(),
#                   DenseLayer(32, nonlinearity=rectify, lambda2=0.001),
#                   DenseLayer(24, nonlinearity=rectify, lambda2=0.001),
#                   OutputLayer()]
#         net = NeuralNet(layers, cost_function=crossentropy,
#                         updater=RMSProp())
#         net.fit(X, y, max_iter=3)
#         return net
