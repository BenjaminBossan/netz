# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import pytest
import theano
import theano.tensor as T

from ..nonlinearities import identity
from ..nonlinearities import rectify
from ..nonlinearities import sigmoid


def test_identity():
    x = np.array([-100, 1.5, 0, -3.01, 2])
    xt = identity(x)
    assert (xt == x).all()


def test_rectify():
    x = np.array([-100, 1.5, 0, -3.01, 2])
    xt = rectify(x)
    expected = np.array([0, 1.5, 0, 0, 2])
    assert (xt == expected).all()


@pytest.mark.parametrize('x, y', [
    (20.3, 1.),
    (-123, 0.),
    (0., 0.5),
])
def test_sigmoid(x, y):
    X = T.matrix('X')
    sigm = theano.function([X], sigmoid(X))
    x = np.array([[x]]).astype(theano.config.floatX)
    xt = sigm(x)
    assert np.allclose(xt, y, atol=1e-3)
    xs = 1 - sigm(-x)
    assert np.allclose(xs, y, atol=1e-3)
    