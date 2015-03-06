# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from theano import shared


def shared_zeros_like(arr, name=None):
    new_var = shared(
        np.zeros(arr.get_value().shape),
        broadcastable=arr.broadcastable,
    )
    if name is not None:
        new_var.name = name
    return new_var


# def flatten(lst):
#     return [elem for sublst in lst for elem in sublst if elem]


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


# def flatten(lst):
#     flat_lst = sum(lst, [])
#     return [elem for elem in lst if elem]
