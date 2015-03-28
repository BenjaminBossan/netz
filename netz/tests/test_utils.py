# -*- coding: utf-8 -*-
from __future__ import division

import pytest
import numpy as np
from scipy.stats import kstest
from theano import shared

from tutils import relative_error
from ..utils import flatten
from ..utils import shared_random_normal
from ..utils import shared_random_uniform
from ..utils import shared_zeros_like


@pytest.mark.parametrize('lst, expected', [
    ([], []),
    ([[]], []),
    ([(1, 2), (3, 4)], [(1, 2), (3, 4)]),
    ([[], (1, 2), []], [(1, 2)]),
    ([[], (1, 2), [], [(3, 4)]], [(1, 2), (3, 4)]),
    ([[], (1, 2), [], [[(3, 4)]]], [(1, 2), [(3, 4)]]),
])
def test_flatten(lst, expected):
    result = flatten(lst)
    assert result == expected


class TestSharedZerosLike:
    @pytest.mark.parametrize('shape', [(4,), (5, 6, 100), (128, 1, 54, 32)])
    def test_shared_zeros_like_shape(self, shape):
        arr_old = shared(np.ones(shape))
        arr = shared_zeros_like(arr_old).get_value()
        assert arr.shape == shape

    def test_shared_zeros_like_is_zeros(self):
        arr_old = shared(np.ones((3, 4, 5)))
        arr = shared_zeros_like(arr_old).get_value()
        assert np.allclose(arr, 0., 1e-16)

    @pytest.mark.parametrize('name', [None, '', 'a_name'])
    def test_shared_zeros_like_name(self, name):
        arr_old = shared(np.ones((3, 4, 5)))
        arr = shared_zeros_like(arr_old, name=name)

    @pytest.mark.parametrize('broadcastable', [
        (False, False, False),
        (True, True, True),
        (True, False, True),
    ])
    def test_shared_zeros_like_broadcastable(self, broadcastable):
        arr_old = shared(np.ones((1, 1, 1)), broadcastable=broadcastable)
        arr = shared_zeros_like(arr_old)
        assert arr.broadcastable == broadcastable


class TestSharedRandomUniform:
    @pytest.mark.parametrize('shape', [(4,), (5, 6, 100), (128, 1, 54, 32)])
    def test_shared_random_uniform_shape(self, shape):
        arr = shared_random_uniform(shape)
        result = arr.get_value().shape
        assert result == shape

    @pytest.mark.parametrize('low, high', [
        (0, 1),
        (-100, 0),
        (-12, -11),
        (-3.5, 22.2),
        (100, 1000),
    ])
    def test_shared_random_uniform_low_high(self, low, high):
        arr = shared_random_uniform((10, 11, 12),
                                    low=low, high=high).get_value()
        minimum, maximum = arr.min(), arr.max()
        range_ = np.abs(high - low)
        assert low < minimum
        rhs = low + 0.01 * range_
        assert minimum < rhs
        assert high > maximum
        rhs = high - 0.01 * range_
        assert maximum > rhs

    def test_shared_random_uniform_distribution(self):
        arr = shared_random_uniform((13, 12, 11, 10), low=0, high=1)
        arr = arr.get_value().flatten()
        p_val = kstest(arr, 'uniform')[1]
        assert p_val > 0.05

    @pytest.mark.parametrize('broadcastable', [
        (False, False, False),
        (True, True, True),
        (True, False, True),
    ])
    def test_shared_random_uniform_broadcastable(self, broadcastable):
        arr = shared_random_uniform((1, 1, 1), broadcastable=broadcastable)
        assert arr.broadcastable == broadcastable


class TestSharedRandomNormal:
    @pytest.mark.parametrize('shape', [(4,), (5, 6, 100), (128, 1, 54, 32)])
    def test_shared_random_normal_shape(self, shape):
        arr = shared_random_normal(shape)
        result = arr.get_value().shape
        assert result == shape

    @pytest.mark.parametrize('broadcastable', [
        (False, False, False),
        (True, True, True),
        (True, False, True),
    ])
    def test_shared_random_normal_broadcastable(self, broadcastable):
        arr = shared_random_normal((1, 1, 1), broadcastable=broadcastable)
        assert arr.broadcastable == broadcastable

    @pytest.mark.parametrize('std_dev', [0, 1e-3, 1, 1e3])
    def test_shared_random_normal_std_dev(self, std_dev):
        n = 1e3
        arr = shared_random_normal((n, n), factor=std_dev).get_value()
        relative_difference = relative_error([arr.std()], [std_dev], 1e-3)
        assert np.allclose(relative_difference, 0., atol=10 / n)

    def test_shared_random_normal_distribution(self):
        arr = shared_random_normal((13, 12, 11, 10))
        arr = arr.get_value().flatten()
        p_val = kstest(arr, 'norm')[1]
        assert p_val > 0.05
