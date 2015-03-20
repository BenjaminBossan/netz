# -*- coding: utf-8 -*-
from __future__ import division
import itertools as it

import matplotlib.pyplot as plt
import numpy as np
from theano import function
from theano import tensor as T

CMAPS = ['gray', 'afmhot', 'autumn', 'bone', 'cool', 'copper',
         'gist_heat', 'hot', 'pink', 'spring', 'summer', 'winter']
CMAPS = it.cycle(CMAPS)


def plot_loss(net, *args, **kwargs):
    plt.plot(net.train_history_, label='train', *args, **kwargs)
    plt.plot(net.valid_history_, label='valid', *args, **kwargs)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()


def plot_conv_weights(layer, figsize=(6, 6), *args, **kwargs):
    """Plot the weights of a specific layer. Only really makes sense
    with convolutional layers.

    Parameters
    ----------
    layer : netz.layers.layer
    """
    W = layer.W.get_value()
    shape = W.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows
    for color, cmap in zip(range(shape[1]), CMAPS):
        figs, axes = plt.subplots(nrows, ncols, figsize=figsize)
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
        for i, (r, c) in enumerate(it.product(range(nrows), range(ncols))):
            if i >= shape[0]:
                break
            axes[r, c].imshow(W[i, color], cmap=cmap,
                              interpolation='nearest', *args, **kwargs)


def plot_conv_activity(layer, x, figsize=(6, 8), *args, **kwargs):
    """Plot the acitivities of a specific layer. Only really makes
    sense with layers that work 2D data (2D convolutional layers, 2D
    pooling layers ...)

    Parameters
    ----------
    layer : netz.layers.layer

    x : numpy.ndarray
      Only takes one sample at a time, i.e. x.shape[0] == 1.

    """
    if x.shape[0] != 1:
        raise ValueError("Only one sample can be plotted at a time.")
    xs = T.tensor4('xs')
    get_activity = function([xs], layer.get_output(xs))
    activity = get_activity(x)
    shape = activity.shape
    nrows = np.ceil(np.sqrt(shape[1])).astype(int)
    ncols = nrows

    figs, axes = plt.subplots(nrows + 1, ncols, figsize=figsize)
    axes[0, ncols // 2].imshow(1 - x[0][0], cmap='gray',
                               interpolation='nearest', *args, **kwargs)
    axes[0, ncols // 2].set_title('original')
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    for i, (r, c) in enumerate(it.product(range(nrows), range(ncols))):
        if i >= shape[1]:
            break
        ndim = activity[0][i].ndim
        if ndim != 2:
            raise ValueError("Wrong number of dimensions, image data should "
                             "have 2, instead got {}".format(ndim))
        axes[r + 1, c].imshow(-activity[0][i], cmap='gray',
                              interpolation='nearest', *args, **kwargs)
