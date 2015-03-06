# -*- coding: utf-8 -*-
from __future__ import division

import theano.tensor as T


def crossentropy(y_true, y_pred):
    return T.mean(T.nnet.categorical_crossentropy(y_pred, y_true))


def mse(y_true, y_pred):
    return T.mean((y_true - y_pred) ** 2)
