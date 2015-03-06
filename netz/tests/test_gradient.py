# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import pytest
import theano
import theano.tensor as T

from ..layers import InputLayer
from ..layers import DenseLayer
from ..layers import OutputLayer
from ..neuralnet import NeuralNet
from ..costfunctions import crossentropy


np.random.seed(17411)
# fake data
X = np.random.rand(10, 4)
y = np.zeros((10, 2))
for i, j in enumerate(np.random.randint(0, 2, 10)):
    y[i, j] = 1.


def verify_grad(net, x, y, abs_tol=None):
    def fun(x, y):
        cost = net.cost_function(y, net.feed_forward(x))
        return cost
    return T.verify_grad(fun, [x, y], rng=np.random.RandomState(42),
                         abs_tol=abs_tol)


class TestSgdNet():
    @pytest.fixture(scope='session')
    def net(self):
        layers = [InputLayer(),
                  DenseLayer(32),
                  DenseLayer(32),
                  OutputLayer()]
        net = NeuralNet(layers, cost_function=crossentropy,
                        update_kwargs={'learn_rate': theano.shared(0.01)})
        net.fit(X, y, max_iter=3)
        return net

    def test_grad(self, net):
        verify_grad(net, X, y)

    def test_test_works(self, net):
        with pytest.raises(theano.gradient.GradientError):
            verify_grad(net, X, y, abs_tol=1e-18)
