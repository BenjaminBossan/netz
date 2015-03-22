# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import theano
from theano import shared


def to_32(x):
    if type(x) == int:
        return np.array(x).astype(np.int32)
    if type(x) == float:
        return np.array(x).astype(np.float32)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    return x
    return theano.tensor.cast(x, 'float32')


def shared_zeros_like(arr, name=None):
    arr_new = shared(np.zeros_like(arr.get_value()),
                     broadcastable=arr.broadcastable)
    arr_new.name = name
    return arr_new
    # if isinstance(arr, np.ndarray):
    #     shape = arr.get_value().shape
    #     new_var = shared(
    #         np.zeros(shape).astype(theano.config.floatX),
    #         broadcastable=arr.broadcastable,
    #     )
    # else:
    #     new_var = theano.tensor.zeros_like(arr)
    # if name is not None:
    #     new_var.name = name
    # return new_var


def shared_random_uniform(shape, low=-1, high=1, name=None,
                          broadcastable=None):
    arr = np.random.uniform(low=low, high=high,
                            size=shape).astype(theano.config.floatX)
    new_var = shared(arr, broadcastable=broadcastable)
    if name is not None:
        new_var.name = name
    return new_var


def flatten(lst):
    """For each element in the list, if the element is itself a list,
    unnest the element once and remove empty elements. Is not
    recursive.

    """
    flat_lst = []
    for sublst in lst:
        if not sublst:
            continue
        if not isinstance(sublst, list):
            flat_lst.append(sublst)
            continue
        for elem in sublst:
            flat_lst.append(elem)
    return flat_lst
