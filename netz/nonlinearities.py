# -*- coding: utf-8 -*-
from __future__ import division

from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import softmax


__all__ = ["sigmoid", "identity", "softmax"]


def identity(self, x):
    return x


def rectify(x):
    return (x + abs(x)) / 2.
