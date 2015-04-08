# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import pytest

from ..iterators import MultipleInputsBatchIterator
from ..iterators import PadBatchIterator


class TestMultipleInputsBatchIterator:
    @pytest.fixture(scope='session')
    def data(self):
        n = 100
        A = 3 * np.ones((n, 3))
        B = 4 * np.ones((n, 4))
        C = 5 * np.ones((n, 5))
        y = np.zeros(n)
        return A, B, C, y

    @pytest.mark.parametrize('batch_size', [1, 5, 25])
    def test_correct_batch_size(self, data, batch_size):
        iterator = MultipleInputsBatchIterator(batch_size)
        A, B, C, y = data
        many_Xb, yb = next(iter(iterator([A, B, C], y)))
        assert len(yb) == batch_size
        for Xb in many_Xb:
            assert len(Xb) == batch_size

    @pytest.mark.parametrize('batch_size', [1, 7, 17, 50, 89])
    def test_last_batch_is_rest(self, data, batch_size):
        n = data[-1].shape[0]
        rest = n % batch_size
        last_batch_size = rest if rest != 0 else batch_size
        iterator = MultipleInputsBatchIterator(batch_size)
        A, B, C, y = data
        X_batches, y_batches = [], []
        for many_Xb, yb in iterator([A, B, C], y):
            X_batches.append(many_Xb)
            y_batches.append(yb)
        assert X_batches[-1][0].shape[0] == last_batch_size
        assert X_batches[-1][1].shape[0] == last_batch_size
        assert X_batches[-1][2].shape[0] == last_batch_size
        assert y_batches[-1].shape[0] == last_batch_size

    def test_list_one_entry(self, data):
        iterator = MultipleInputsBatchIterator(10)
        A, __, __, y = data
        many_Xb, yb = next(iter(iterator([A], y)))
        assert len(many_Xb) == 1
        assert (many_Xb[0] == A[:10]).all()
        assert (yb == y[:10]).all()

    def test_list_three_list_entries(self, data):
        iterator = MultipleInputsBatchIterator(10)
        A, B, C, y = data
        many_Xb, yb = next(iter(iterator([A, B, C], y)))
        assert len(many_Xb) == 3
        assert (many_Xb[0] == A[:10]).all()
        assert (many_Xb[1] == B[:10]).all()
        assert (many_Xb[2] == C[:10]).all()
        assert (yb == y[:10]).all()


class TestPadBatchIterator:
    @pytest.fixture
    def data(self):
        X = np.asarray([range(5),
                        range(3),
                        range(1),
                        range(2)])
        y = np.zeros(4)
        return X, y

    @pytest.mark.parametrize('token', [0, '1', 1.23])
    def test_padding_pad_all_correct_token_batch_size_1(self, data, token):
        iterator = PadBatchIterator(1, token)
        X, y = data
        X_batches = []
        for Xb, __ in iterator(X, y):
            X_batches.append(Xb)
        for x0, x1 in zip(X, X_batches):
            assert (x1 == np.asarray(x0 + [token])).all()

    @pytest.mark.parametrize('token', [0, '1', 1.23])
    def test_padding_pad_all_correct_token_batch_size_2(self, data, token):
        iterator = PadBatchIterator(2, token)
        X, y = data
        X_batches = []
        for Xb, __ in iterator(X, y):
            X_batches.append(Xb)
        for batch in X_batches:
            assert batch.ndim == 2
        assert (X_batches[0][0] == np.asarray(range(5) + [token])).all()
        assert (X_batches[0][1] == np.asarray(range(3) + 3 * [token])).all()
        assert (X_batches[1][0] == np.asarray(range(1) + 2 * [token])).all()
        assert (X_batches[1][1] == np.asarray(range(2) + [token])).all()

    @pytest.mark.parametrize('token', [0, '1', 1.23])
    def test_padding_pad_all_correct_token_batch_size_4(self, data, token):
        iterator = PadBatchIterator(4, token)
        X, y = data
        X_batches = []
        for Xb, __ in iterator(X, y):
            X_batches.append(Xb)
        for batch in X_batches:
            assert batch.ndim == 2
        assert (X_batches[0][0] == np.asarray(range(5) + [token])).all()
        assert (X_batches[0][1] == np.asarray(range(3) + 3 * [token])).all()
        assert (X_batches[0][2] == np.asarray(range(1) + 5 * [token])).all()
        assert (X_batches[0][3] == np.asarray(range(2) + 4 * [token])).all()

    def test_padding_not_pad_all_batch_size_2(self, data):
        token = 12.34
        iterator = PadBatchIterator(2, token, pad_all=False)
        X, y = data
        X_batches = []
        for Xb, __ in iterator(X, y):
            X_batches.append(Xb)
        for batch in X_batches:
            assert batch.ndim == 2
        assert (X_batches[0][0] == np.asarray(range(5))).all()
        assert (X_batches[0][1] == np.asarray(range(3) + 2 * [token])).all()
        assert (X_batches[1][0] == np.asarray(range(1) + 1 * [token])).all()
        assert (X_batches[1][1] == np.asarray(range(2))).all()
