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
from theano import function
from theano import shared
from theano import tensor as T

from costfunctions import crossentropy
from updaters import SGD
from utils import flatten


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

    def _initialize_functions(self, X, y):
        # symbolic variables
        ys = T.matrix('y')
        if X.ndim == 2:
            Xs = T.matrix('X')
        elif X.ndim == 4:
            Xs = T.tensor4('X')
        else:
            ValueError("Input must be 2D or 4D, instead got {}D."
                       "".format(X.ndim))

        # generate train function
        y_pred = self.feed_forward(Xs, deterministic=False)
        y_pred.name = 'y_pred'
        cost = self.cost_function(ys, y_pred)
        cost += self._get_l2_cost()
        updates = [layer.updater.get_updates(cost, layer)
                   for layer in self.layers if layer.updater]
        updates = flatten(updates)
        self.train_ = function([Xs, ys], cost, updates=updates)

        # generate test function
        y_pred = self.feed_forward(Xs, deterministic=True)
        y_pred.name = 'y_pred'
        cost = self.cost_function(ys, y_pred)
        cost += self._get_l2_cost()
        self.test_ = function([Xs, ys], cost)

        # generate predict function
        self._predict = function(
            [Xs], self.feed_forward(Xs, deterministic=True)
        )

        # generate score function
        self._score = function(
            [Xs, ys], self.cost_function(
                ys, self.feed_forward(Xs, deterministic=True)
            )
        )

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
        self.encoder_ = OneHotEncoder(sparse=False).fit(y.reshape(-1, 1))

        # progress of cost function
        self.train_history_ = []
        self.valid_history_ = []

        # header for verbose output
        self.header_ = (
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
            print(" ~=* Neural Network with {} learnable parameters *=~ "
                  "".format(nparams))
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
                epoch,
                mean_train,
                mean_valid,
                mean_train/mean_valid if mean_valid else 0.,
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
        return self._predict(X)

    def predict(self, X):
        y_prob = self.predict_proba(X)
        return np.argmax(y_prob, axis=1)

    def score(self, X, labels, accuracy=False):
        if X.shape[0] != labels.shape[0]:
            raise ValueError(
                "Incompatible input dimensions: X.shape[0] is {},"
                " y.shape[0] is {}".format(X.shape[0], labels.shape[0])
            )
        if not accuracy:
            y = self.encoder_.transform(labels.reshape(-1, 1))
            return self._score(X, y) + 0.
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

    def _get_hash(self, X, y):
        # ``iter`` seems to be necessary if X or y are just views and if
        # we want to avoid creating a copy.
        X_hash, y_hash = hash(iter(X)), hash(iter(y))
        return X_hash, y_hash

    def _set_hash(self, X, y):
        self.X_hash_, self.y_hash_ = self._get_hash(X, y)

    def get_train_data(self, X, y):
        X_hash, y_hash = self._get_hash(X, y)
        if (X_hash != self.X_hash_) or (y_hash != self.y_hash_):
            warnings.warn("Input data has changed since last usage")
        X_train, __, y_train, __ = self.train_test_split(X, y)
        return X_train, y_train

    def get_valid_data(self, X, y):
        X_hash, y_hash = self._get_hash(X, y)
        if (X_hash != self.X_hash_) or (y_hash != self.y_hash_):
            warnings.warn("Input data has changed since last usage")
        __, X_valid, __, y_valid = self.train_test_split(X, y)
        return X_valid, y_valid

    def _print_layer_info(self):
        for layer in self.layers:
            output_shape = tuple(layer.get_output_shape())
            print("  {:<18}\t{:<20}\tproduces {:>7} outputs".format(
                layer.name,
                str(output_shape),
                str(reduce(op.mul, output_shape[1:])),
            ))
