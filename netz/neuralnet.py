# -*- coding: utf-8 -*-
from __future__ import division
from collections import defaultdict
import operator as op
import time
import warnings

from nolearn.lasagne import BatchIterator
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import theano
from theano import function
from theano import tensor as T

from costfunctions import crossentropy
from updaters import SGD
from utils import flatten
from utils import np_hash
from utils import to_32


class NeuralNet(BaseEstimator):
    def __init__(
            self,
            layers,
            updater=SGD(),
            cost_function=crossentropy,
            iterator=BatchIterator(128),
            lambda2=None,
            eval_size=0.2,
            verbose=0,
    ):
        self.layers = layers
        self.updater = updater
        self.cost_function = cost_function
        self.iterator = iterator
        self.lambda2 = lambda2
        self.eval_size = eval_size
        self.verbose = verbose

    def get_layer_params(self):
        return [layer.get_params() for layer in self.layers]

    def _get_l2_cost(self):
        l2_cost = [layer.get_l2_cost() for layer in self.layers]
        return T.sum([cost for cost in l2_cost if cost is not None])

    def _initialize_names(self):
        name_counts = defaultdict(int)
        for layer in self.layers:
            if layer.name is not None:
                continue
            raw_name = str(layer).split(' ')[0].split('.')[-1]
            raw_name = raw_name.replace('Layer', '').lower()
            name_counts[raw_name] += 1
            name = raw_name + str(name_counts[raw_name] - 1)
            layer.set_name(name)

    def _initialize_updaters(self):
        for layer in self.layers:
            if layer.updater is not None:
                continue
            if self.updater is None:
                raise TypeError("Please specify an updater for each layer"
                                "or for the neural net as a whole.")
            layer.set_updater(self.updater)

    def _initialize_lambda2(self):
        for layer in self.layers:
            if layer.lambda2 is not None:
                continue
            layer.set_lambda2(self.lambda2)

    def _initialize_connections(self):
        for layer0, layer1 in zip(self.layers[:-1], self.layers[1:]):
            if layer0.next_layer is None:
                layer0.set_next_layer(layer1)
            if layer1.prev_layer is None:
                layer1.set_prev_layer(layer0)

    def _get_cost_function(self, X, y, deterministic=True):
        y_pred = self.feed_forward(X, deterministic=deterministic)
        y_pred.name = 'y_pred'
        cost = self.cost_function(y, y_pred)
        cost += self._get_l2_cost()
        return cost

    def _initialize_functions(self, X, y):
        # symbolic variables
        ys = T.matrix('y').astype(theano.config.floatX)
        if X.ndim == 2:
            Xs = T.matrix('X').astype(theano.config.floatX)
        elif X.ndim == 4:
            Xs = T.tensor4('X').astype(theano.config.floatX)
        else:
            raise ValueError("Input must be 2D or 4D, instead got {}D."
                             "".format(X.ndim))

        # generate train function
        cost_train = self._get_cost_function(Xs, ys, False)
        updates = [layer.get_updates(cost_train) for layer in self.layers
                   if layer.updater]
        updates = flatten(updates)
        self.train_ = function([Xs, ys], cost_train, updates=updates,
                               allow_input_downcast=True)

        # generate test function
        cost_test = self._get_cost_function(Xs, ys, True)
        self.test_ = function([Xs, ys], cost_test,
                              allow_input_downcast=True)

        # generate predict function
        self._predict_proba = function(
            [Xs], self.feed_forward(Xs, deterministic=True),
            allow_input_downcast=True)

    def initialize(self, X, y):
        # set layer names
        self._initialize_names()

        # set layer updaters
        self._initialize_updaters()

        # connect layers
        self._initialize_connections()

        # initialize layers
        for layer in self.layers:
            layer.initialize(X, y)

        # initialize encoder
        self.encoder_ = OneHotEncoder(
            sparse=False, dtype=y.dtype).fit(y.reshape(-1, 1))

        # progress of cost function
        self.train_history_ = []
        self.valid_history_ = []

        # header for verbose output
        self.header_ = (
            "## Training Information\n"
            " Epoch | Train loss | Valid loss | Train/Val | Valid acc | Dur\n"
            "-------|------------|------------|-----------|-----------|------"
        )

        # generate theano functions
        self._initialize_functions(X, y)

        # print layer infos
        if self.verbose:
            shapes = [param.get_value().shape for param in
                      flatten(self.get_layer_params()) if param]
            nparams = reduce(op.add, [reduce(op.mul, shape) for
                                      shape in shapes])
            print("# ~=* Neural Network with {} learnable parameters *=~ "
                  "".format(nparams))
            print("\n")
            self._print_layer_info()
        self.is_init_ = True

    def fit(self, X, y, max_iter=5):
        if not hasattr(self, 'is_init_'):
            self.initialize(X, y)

        self._set_hash(X, y)
        X_train, X_valid, labels_train, labels_valid = (
            self.train_test_split(X, y))
        y_train = self.encoder_.transform(labels_train.reshape(-1, 1))
        if labels_valid.size > 0:
            y_valid = self.encoder_.transform(labels_valid.reshape(-1, 1))
        if self.verbose:
            print(self.header_)
            num_epochs_past = len(self.train_history_)

        for epoch in range(max_iter):
            tic = time.time()
            # ----------- train loop ---------------
            train_cost = []
            for Xb, yb in self.iterator(X_train, y_train):
                train_cost.append(self._fit(Xb, yb))
            mean_train = np.mean(train_cost)
            self.train_history_.append(mean_train)

            # ----------- valid loop ---------------
            valid_cost = []
            if self.eval_size > 0.:
                for Xb, yb in self.iterator(X_valid, y_valid):
                    valid_cost.append(self._fit(Xb, yb, mode='test'))
                # TODO: inefficient since 2 predictions are made on
                # X_valid where 1 would suffice
                accuracy_valid = self.score(X_valid, labels_valid, True)
                mean_valid = np.mean(valid_cost) if valid_cost else 0
            else:
                accuracy_valid = 0.
                mean_valid = 0.
            self.valid_history_.append(mean_valid)

            # --------- verbose feedback -----------
            if not self.verbose:
                continue
            toc = time.time()
            template = (" {:>5} | {:>10.6f} | {:>10.6f} | {:>8.3f}  |"
                        "{:>8.4f}   | {:>3.1f}s")
            print(template.format(
                num_epochs_past + epoch,
                mean_train,
                mean_valid,
                mean_train / mean_valid if mean_valid else 0.,
                accuracy_valid,
                toc - tic,
            ))
        return self

    def _fit(self, X, y, mode='train'):
        if mode == 'train':
            cost = self.train_(X, y)
        else:
            cost = self.test_(X, y)
        return cost

    def feed_forward(self, X, deterministic=False):
        return self.layers[-1].get_output(X, deterministic=deterministic)

    def predict_proba(self, X):
        return self._predict_proba(X)

    def predict(self, X):
        y_prob = self.predict_proba(X)
        return np.argmax(y_prob, axis=1)

    def score(self, X, labels, accuracy=False):
        if X.shape[0] != labels.shape[0]:
            raise ValueError(
                "Incompatible input shapes: X.shape[0] is {},"
                " y.shape[0] is {}".format(X.shape[0], labels.shape[0])
            )
        if not accuracy:
            y = self.encoder_.transform(labels.reshape(-1, 1))
            return self.test_(X, y) + 0.
        else:
            y_pred = self.predict(X)
            return np.mean(labels == y_pred)

    def train_test_split(self, X, y):
        eval_size = self.eval_size
        if eval_size:
            kf = StratifiedKFold(y, round(1. / eval_size))
            train_indices, valid_indices = next(iter(kf))
            X_train, y_train = X[train_indices], y[train_indices]
            X_valid, y_valid = X[valid_indices], y[valid_indices]
        else:
            X_train, y_train = X, y
            X_valid, y_valid = X[len(X):], y[len(y):]
        return X_train, X_valid, y_train, y_valid

    def _set_hash(self, X, y):
        self.X_hash_, self.y_hash_ = np_hash(X), np_hash(y)

    def get_train_data(self, X, y):
        X_hash, y_hash = np_hash(X), np_hash(y)
        if (X_hash != self.X_hash_) or (y_hash != self.y_hash_):
            warnings.warn("Input data has changed since last usage")
        X_train, __, y_train, __ = self.train_test_split(X, y)
        return X_train, y_train

    def get_valid_data(self, X, y):
        X_hash, y_hash = np_hash(X), np_hash(y)
        if (X_hash != self.X_hash_) or (y_hash != self.y_hash_):
            warnings.warn("Input data has changed since last usage")
        __, X_valid, __, y_valid = self.train_test_split(X, y)
        return X_valid, y_valid

    def _print_layer_info(self):
        print("## Layer information")
        print("  # | {:<18} | {:<18} | {:>12} ".format(
            "name", "output shape", "total"))
        print("----|-{}-|-{}-|-{}-".format(
            "-" * 18, "-" * 18, "-" * 12))
        for num, layer in enumerate(self.layers):
            output_shape = tuple(layer.get_output_shape())
            row = " {:>2} | {:<18} | {:<18} | {:>12}"
            print(row.format(
                num,
                layer.name,
                str(output_shape),
                str(reduce(op.mul, output_shape[1:])),
            ))
        print("\n")


class MultipleInputNet(NeuralNet):
    def _initialize_functions(self, many_X, y):
        # symbolic variables
        ys = T.matrix('y').astype(theano.config.floatX)
        many_Xs = []
        for X in many_X:
            if X.ndim == 2:
                Xs = T.matrix('X').astype(theano.config.floatX)
            elif X.ndim == 4:
                Xs = T.tensor4('X').astype(theano.config.floatX)
            else:
                raise ValueError("Input must be 2D or 4D, instead got {}D."
                                 "".format(X.ndim))
            many_Xs.append(Xs)

        # generate train function
        cost_train = self._get_cost_function(many_Xs, ys, False)
        updates = [layer.get_updates(cost_train) for layer in self.layers
                   if layer.updater]
        updates = flatten(updates)
        self.train_ = function(many_Xs + [ys], cost_train, updates=updates,
                               allow_input_downcast=True)

        # generate test function
        cost_test = self._get_cost_function(many_Xs, ys, True)
        self.test_ = function(many_Xs + [ys], cost_test,
                              allow_input_downcast=True)

        # generate predict function
        self._predict_proba = function(
            many_Xs, self.feed_forward(many_Xs, deterministic=True),
            allow_input_downcast=True)

    def initialize(self, many_X, y):
        if any([X.shape[0] != many_X[0].shape[0] for X in many_X]):
            raise ValueError("All inputs must have the same shape in the "
                             "first dimension, instead got the following "
                             "shapes: {}".format(', '.join(X.shape[0]
                                                           for X in many_X)))
        super(MultipleInputNet, self).initialize(many_X, y)

    def _fit(self, X, y, mode='train'):
        if mode == 'train':
            cost = self.train_(*(X + [y]))
        else:
            cost = self.test_(*(X + [y]))
        return cost

    def predict_proba(self, many_X):
        return self._predict_proba(*many_X)

    def score(self, many_X, labels, accuracy=False):
        if any(X.shape[0] != labels.shape[0] for X in many_X):
            raise ValueError(
                "Incompatible input shapes along 1st dim: X shapes are {}, "
                "y shape is {}".format(', '.join([X.shape[0] for X in many_X]),
                                       labels.shape[0])
            )
        if not accuracy:
            y = self.encoder_.transform(labels.reshape(-1, 1))
            return self.test_(*(X + [y])) + 0.
        else:
            y_pred = self.predict(many_X)
            return np.mean(labels == y_pred)

    def train_test_split(self, many_X, y):
        many_X_train_splits = []
        many_X_valid_splits = []
        eval_size = self.eval_size

        for X in many_X:
            if eval_size:
                kf = StratifiedKFold(y, round(1. / eval_size))
                train_indices, valid_indices = next(iter(kf))
                X_train, y_train = X[train_indices], y[train_indices]
                X_valid, y_valid = X[valid_indices], y[valid_indices]
            else:
                X_train, y_train = X, y
                X_valid, y_valid = X[len(X):], y[len(y):]
            many_X_train_splits.append(X_train)
            many_X_valid_splits.append(X_valid)
        return many_X_train_splits, many_X_valid_splits, y_train, y_valid

    def _set_hash(self, many_X, y):
        self.y_hash_ = np_hash(y)
        self.X_hash_ = np.sum(map(np_hash, many_X))

    def get_train_data(self, many_X, y):
        X_hash, y_hash = np.sum(map(np_hash, many_X)), np_hash(y)
        if (X_hash != self.X_hash_) or (y_hash != self.y_hash_):
            warnings.warn("Input data has changed since last usage")
        X_train, __, y_train, __ = self.train_test_split(many_X, y)
        return X_train, y_train

    def get_valid_data(self, many_X, y):
        X_hash, y_hash = np.sum(map(np_hash, many_X)), np_hash(y)
        if (X_hash != self.X_hash_) or (y_hash != self.y_hash_):
            warnings.warn("Input data has changed since last usage")
        __, X_valid, __, y_valid = self.train_test_split(many_X, y)
        return X_valid, y_valid


class RNN(NeuralNet):
    def __init__(
            self,
            layers,
            updater=SGD(),
            cost_function=crossentropy,
            iterator=BatchIterator(1),
            lambda2=None,
            eval_size=0.2,
            verbose=0,
    ):
        self.layers = layers
        self.updater = updater
        self.cost_function = cost_function
        self.iterator = iterator
        self.lambda2 = lambda2
        self.eval_size = eval_size
        self.verbose = verbose

    def initialize(self, X, y):
        if self.iterator.batch_size != 1:
            raise ValueError("Currently, only batch sizes of 1 are supported "
                             "(i.e. no batches) but you set a batch size of "
                             "{}.".format(self.iterator.batch_size))
        super(RNN, self).initialize(X, y)

    def _initialize_functions(self, X, y):
        # symbolic variables
        ys = T.matrix('y')
        Xs = T.ivector('X')

        # generate train function
        cost_train = self._get_cost_function(Xs, ys, False)
        updates = [layer.get_updates(cost_train) for layer in self.layers
                   if layer.updater]
        updates = flatten(updates)
        self.train_ = function([Xs, ys], cost_train, updates=updates,
                               allow_input_downcast=True)

        # generate test function
        cost_test = self._get_cost_function(Xs, ys, True)
        self.test_ = function([Xs, ys], cost_test,
                              allow_input_downcast=True)

        # generate predict function
        self._predict_proba = function(
            [Xs], self.feed_forward(Xs, deterministic=True),
            allow_input_downcast=True)

    def _fit(self, X, y, mode='train'):
        X = X[0]
        if mode == 'train':
            cost = self.train_(X, y)
        else:
            cost = self.test_(X, y)
        return cost

    def predict_proba(self, X):
        n = X.shape[0]
        pred = [self._predict_proba(x) for x in X]
        return to_32(np.asarray(pred)).reshape(n, -1)

    def score(self, X, labels, accuracy=False):
        if X.shape[0] != labels.shape[0]:
            raise ValueError(
                "Incompatible input shapes: X.shape[0] is {},"
                " y.shape[0] is {}".format(X.shape[0], labels.shape[0])
            )
        score = []
        if not accuracy:
            y = self.encoder_.transform(labels.reshape(-1, 1))
            for Xb, yb in self.iterator(X, y):
                score.append(self.test_(Xb[0], yb))
        else:
            for Xb, yb in self.iterator(X, labels):
                score.append(self.predict(Xb) == yb)
        return np.mean(score)
