# -*- coding: utf-8 -*-
from __future__ import division
import itertools as it

import joblib
import numpy as np
import theano
from theano import shared
floatX = theano.config.floatX


def to_32(x):
    if type(x) == int:
        return np.array(x).astype(np.float32)
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


def shared_random_uniform(shape, low=-1, high=1, name=None,
                          broadcastable=None):
    arr = np.random.uniform(low=low, high=high,
                            size=shape).astype(floatX)
    new_var = shared(arr, broadcastable=broadcastable)
    if name is not None:
        new_var.name = name
    return new_var


def shared_zeros(shape, name=None, broadcastable=None):
    arr = np.zeros(shape).astype(floatX)
    new_var = shared(arr, broadcastable=broadcastable)
    if name is not None:
        new_var.name = name
    return new_var


def shared_ones(shape, name=None, broadcastable=None):
    arr = np.ones(shape).astype(floatX)
    new_var = shared(arr, broadcastable=broadcastable)
    if name is not None:
        new_var.name = name
    return new_var


def shared_random_normal(shape, factor=1., name=None, broadcastable=None):
    arr = factor * np.random.randn(*shape).astype(floatX)
    new_var = shared(arr, broadcastable=broadcastable)
    if name is not None:
        new_var.name = name
    return new_var


def shared_random_orthogonal(shape, name=None, broadcastable=None):
    shape_2d = (shape[0], np.prod(shape[1:]))
    arr = np.random.normal(0., 1., shape_2d).astype(floatX)
    u, __, v = np.linalg.svd(arr, full_matrices=False)
    q = u if u.shape == shape_2d else v
    q = q.reshape(shape)
    new_var = shared(q, broadcastable=broadcastable)
    if name is not None:
        new_var.nsame = name
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


def np_hash(arr):
    """Hash a numpy array.
    """
    return int(joblib.hash(arr), base=16)


def build_layer_dict(layers):
    layer_dict = {layer.name: layer for layer in layers if layer.name}
    if len(layer_dict) != len(layers):
        raise ValueError("Please specify a unique name for all layers.")
    return layer_dict


def connect_layers(layers, pattern):
    layer_dict = build_layer_dict(layers)
    pattern = pattern.replace(' ', '')
    patterns = [tuple(line.split('->')) for line in
                pattern.split('\n') if line]
    for layer_name0, layer_name1 in patterns:
        layer0 = layer_dict[layer_name0]
        layer1 = layer_dict[layer_name1]
        layer0.set_next_layer(layer1)
        layer1.set_prev_layer(layer0)


def occlusion_heatmap(net, x, y, square_length=7):
    """An occlusion test that checks an image for its critical parts.

    In this test, a square part of the image is occluded (i.e. set to
    0) and then the net is tested for its propensity to predict the
    correct label. One should expect that this propensity shrinks of
    critical parts of the image are occluded. If not, this indicates
    overfitting.

    Currently, all color channels are occluded at the same time.

    See paper: Zeiler, Fergus 2013

    Parameters
    ----------
    net : NeuralNet instance
      The neural net to test.

    x : np.array
      The input data, should be of shape (1, c, x, y). Only makes
      sense with image data.

    y : np.array
      The true value of the image.

    square_length : int (default=7)
      The length of the side of the square that occludes the image.

    Results
    -------
    heat_array : np.array (with same size as image)
      An 2D np.array that at each point (i, j) contains the predicted
      probability of the correct class if the image is occluded by a
      square with center (i, j).

    """
    if (x.ndim != 4) or x.shape[0] != 1:
        raise ValueError("This function requires the input data to be of "
                         "shape (1, c, x, y), instead got {}".format(x.shape))
    img = x[0].copy()
    shape = x.shape
    heat_array = np.zeros(shape[2:])
    pad = square_length // 2
    for i, j in it.product(*map(range, shape[2:])):
        x_padded = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
        x_padded[:, i:i + square_length, j:j + square_length] = 0.
        x_occluded = x_padded[:, pad:-pad, pad:-pad]
        prob = net.predict_proba(x_occluded.reshape(1, 1, shape[2], shape[3]))
        heat_array[i, j] = prob[0, y]
    return heat_array
