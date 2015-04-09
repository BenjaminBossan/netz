# -*- coding: utf-8 -*-
from __future__ import division
from copy import deepcopy

from mock import Mock
import numpy as np
import pandas as pd
import pytest
import theano

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
from ..updaters import GradientClipping

np.random.seed(17412)
# Number of numerically checked gradients per paramter (more -> slower)
# fake data
X = 2 * np.random.rand(10, 5).astype(np.float32)
X -= X.mean()
y = np.random.randint(low=0, high=3, size=10).astype(np.float32)


# def test_updater_initalize_with_int_arguments():
#     layers = [InputLayer(),
#               DenseLayer(1, updater=Momentum(momentum=1.)),
#               OutputLayer()]
#     net = NeuralNet(layers)
#     # this should not raise an exception:
#     net.initialize(X, y)


class TestGradientClipping:
    @pytest.fixture(scope='function')
    def grad_clip(self):
        updates = [('1st', np.ones((3, 3)).astype(theano.config.floatX)),
                   ('2nd', 5 * np.ones((2, 34)).astype(theano.config.floatX)),
                   ('3rd', -3 * np.ones((100, 1)).astype(theano.config.floatX))]
        updater = Momentum()
        updater.get_updates = Mock(return_value=updates)
        grad_clip = GradientClipping(2, updater)
        return grad_clip

    @pytest.fixture
    def get_norm(self, grad_clip):
        X = theano.tensor.fmatrix()
        get_norm = theano.function([X], grad_clip._get_norm(X))
        return get_norm

    @pytest.fixture
    def clip_norm(self, grad_clip, get_norm):
        X = theano.tensor.fmatrix()
        norm_max = theano.tensor.fscalar()
        norm_var = grad_clip._get_norm(X)
        clip_norm = theano.function(
            [X, norm_var, norm_max],
            grad_clip._clip_norm(X, norm_var, norm_max)
        )
        return clip_norm

    @pytest.fixture
    def clipped_grads(self, grad_clip, max_norm):
        return updates

    @pytest.mark.parametrize('arr, norm', [
        (np.zeros((11, 2)), 0.),
        (np.ones((2, 4)), np.sqrt(8.)),
        (-1 * np.ones((2, 4)), np.sqrt(8.)),
        (np.asarray([[4, 0, 3]]), 5.),
    ])
    def test_get_norm_computes_norm(self, get_norm, arr, norm):
        arr = arr.astype(theano.config.floatX)
        result = get_norm(arr)
        assert np.isclose(result, norm)

    @pytest.mark.parametrize('arr, norm_max', [
        (0.001 * np.ones((11, 2)), 1),
        (np.ones((2, 4)), 0.03),
        (-1 * np.ones((2, 4)), 2),
        (np.asarray([[4, 0, 3]]), 2.5),
    ])
    def test_clip_norm_clips_large_vars(self, get_norm, clip_norm, arr,
                                        norm_max):
        arr = arr.astype(theano.config.floatX)
        norm = get_norm(arr)
        clipped = clip_norm(arr, norm, norm_max)
        norm_clipped = get_norm(clipped)
        if norm > norm_max:
            assert np.isclose(norm_clipped, norm_max)
        else:
            assert np.isclose(norm_clipped, norm)

    def test_get_updates_does_not_touch_first_row(self, grad_clip):
        clipped_updates = grad_clip.get_updates()
        untouched_vars = zip(*clipped_updates)[0]
        assert untouched_vars == ('1st', '2nd', '3rd')

    @pytest.mark.parametrize('norm_max', [0.01, 1, 4, 10])
    def test_get_updates_clips_2nd_row(self, grad_clip, get_norm, norm_max):
        norm_max_old = grad_clip.norm_max
        grad_clip.norm_max = norm_max
        get_clipped_vars = theano.function(
            [], zip(*grad_clip.get_updates())[1])
        grad_clip.norm_max = norm_max_old

        norms = [get_norm(var) for var in
                 zip(*grad_clip.updater.get_updates())[1]]
        # updates_clipped = zip(*grad_clip.get_updates())[1]
        clipped_vars = get_clipped_vars()
        norms_clipped = [get_norm(update) for update in clipped_vars]
        for norm, norm_clipped in zip(norms, norms_clipped):
            if norm > norm_max:
                assert np.isclose(norm_clipped, norm_max)
            else:
                assert np.isclose(norm_clipped, norm)
