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

np.random.seed(17412)
# Number of numerically checked gradients per paramter (more -> slower)
# fake data
X = 2 * np.random.rand(10, 5).astype(np.float32)
X -= X.mean()
y = np.random.randint(low=0, high=3, size=10).astype(np.float32)


def test_updater_initalize_with_int_arguments():
    layers = [InputLayer(),
              DenseLayer(1, updater=Momentum(momentum=1.)),
              OutputLayer()]
    net = NeuralNet(layers)
    # this should not raise an exception:
    net.initialize(X, y)
