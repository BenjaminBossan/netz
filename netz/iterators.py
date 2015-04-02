# -*- coding: utf-8 -*-
from __future__ import division


class MultipleInputsBatchIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

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
