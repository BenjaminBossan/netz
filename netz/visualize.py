# -*- coding: utf-8 -*-
from __future__ import division
import itertools as it

import matplotlib.pyplot as plt
import numpy as np

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
            axes[c, r].imshow(W[i, color], cmap=cmap,
                              interpolation='nearest', *args, **kwargs)
