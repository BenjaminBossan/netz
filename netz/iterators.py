# -*- coding: utf-8 -*-
from __future__ import division

from nolearn.lasagne import BatchIterator
import numpy as np


__all__ = ['BatchIterator', 'MultipleInputsBatchIterator', 'PadBatchIterator']


class MultipleInputsBatchIterator(BatchIterator):
    """The MultipleInputsBatchIterator works with a list of numpy arrays

    Instead of a single numpy array, this takes a list of numpy arrays
    and returns a list of slices of the individual numpy arrays.

    """
    def __call__(self, many_X, y=None):
        self.many_X, self.y = many_X, y
        return self

    def __iter__(self):
        n_samples = self.y.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = [X[sl] for X in self.many_X]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)


class PadBatchIterator(BatchIterator):
    """Pads the rows of a numpy array to all have same length.

    This should by applied in particular to heterogeneous numpy arrays
    where rows don't all have the same length. Heterogeneous arrays
    may only be sliced in the 0'th dimension -- after padding, it can
    also be sliced in the 1st dimension.  You may also use a 2d array
    and choose pad_all=True to append an end token to all lines.

    Parameters
    ----------
    batch_size : int
      The size of a batch

    pad_token : anything (default=0)
      The token to pad with.

    pad_all : bool (default=True)
      Whether to pad even the longest line by a token. Makes all rows
      1 longer.

    yields
    ------
    Xb : numpy array
      A padded batch of X.

    yb : numpy array
      An untransformed batch of y.

    """
    def __init__(self, batch_size, pad_token=0, pad_all=True):
        self.batch_size = batch_size
        self.pad_token = pad_token
        self.pad_all = pad_all

    def transform(self, X, y):
        max_len = max(map(len, X))
        Xt = []
        for x in X:
            num_pad = max_len - len(x) + int(self.pad_all)
            padded_x = x + num_pad * [self.pad_token]
            Xt.append(np.array(padded_x))
        return np.asarray(Xt, dtype=X.dtype), y
