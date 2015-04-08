# -*- coding: utf-8 -*-
from __future__ import division

from nolearn.lasagne import BatchIterator


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

    def transform(self, Xb, yb):
        return Xb, yb
