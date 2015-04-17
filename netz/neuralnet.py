# -*- coding: utf-8 -*-
from __future__ import division
from collections import defaultdict
from collections import OrderedDict
from difflib import SequenceMatcher
import operator as op
import pickle
import time
import warnings

from nolearn.lasagne import BatchIterator
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate
import theano
from theano import function
from theano import tensor as T

from costfunctions import crossentropy
from costfunctions import mse
from layers import Conv2DLayer
from updaters import SGD
from utils import connect_layers
from utils import flatten
from utils import get_conv_infos
from utils import IdleTransformer
from utils import np_hash
from utils import to_32
from utils import COLORS

RED, MAG, CYA, END = COLORS


class NeuralNet(BaseEstimator):
    def __init__(
            self,
            layers,
            updater=SGD(),
            iterator=BatchIterator(128),
            lambda2=None,
            eval_size=0.2,
            verbose=0,
            connection_pattern=None,
            cost_function=None,
            encoder=None,
            regression=False,
            recurrent=False,
    ):
        self.layers = layers
        self.updater = updater
        self.iterator = iterator
        self.lambda2 = lambda2
        self.eval_size = eval_size
        self.verbose = verbose
        self.connection_pattern = connection_pattern
        self.cost_function = cost_function
        self.encoder = encoder
        self.regression = regression
        self.recurrent = recurrent

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        if not isinstance(idx, slice) and (idx >= len(self)):
            raise IndexError
        return self.layers[idx]

    def __add__(self, layer):
        if isinstance(layer, tuple) or isinstance(layer, list):
            self.layers.extend(layer)
        else:
            self.layers.append(layer)
        return self

    def get_layer_params(self):
        return [layer.get_params() for layer in self]

    def _get_l2_cost(self):
        l2_cost = [layer.get_l2_cost() for layer in self]
        return T.sum([cost for cost in l2_cost if cost is not None])

    def _initialize_names(self):
        name_counts = defaultdict(int)
        for layer in self:
            if layer.name is not None:
                continue
            raw_name = str(layer).split(' ')[0].split('.')[-1]
            raw_name = raw_name.replace('Layer', '').lower()
            name_counts[raw_name] += 1
            name = raw_name + str(name_counts[raw_name] - 1)
            layer.set_name(name)

    def _initialize_lambda2(self):
        for layer in self:
            if layer.lambda2 is not None:
                continue
            layer.set_lambda2(self.lambda2)

    def _initialize_updaters(self):
        for layer in self:
            if layer.updater is not None:
                continue
            if self.updater is None:
                raise TypeError("Please specify an updater for each layer"
                                "or for the neural net as a whole.")
            layer.set_updater(self.updater)

    def _initialize_connections(self):
        if not self.connection_pattern:
            for layer0, layer1 in zip(self[:-1], self[1:]):
                if layer0.next_layer is None:
                    layer0.set_next_layer(layer1)
                if layer1.prev_layer is None:
                    layer1.set_prev_layer(layer0)
        else:
            connect_layers(self.layers, self.connection_pattern)

    def _get_cost_function(self):
        return mse if self.regression else crossentropy

    def _get_encoder(self, y):
        if self.regression:
            encoder = IdleTransformer()
        else:
            encoder = OneHotEncoder(sparse=False, dtype=y.dtype)
        encoder.fit(y.reshape(-1, 1))
        return encoder

    def _get_cost(self, X, y, deterministic=True):
        y_pred = self.feed_forward(X, deterministic=deterministic)
        y_pred.name = 'y_pred'
        cost = self.cost_function(y, y_pred)
        cost += self._get_l2_cost()
        return cost

    def _initialize_symbolic_variables(self, X, y):
        ys = T.matrix('y').astype(theano.config.floatX)
        ndim = X.ndim
        if self.recurrent and (ndim in (1, 2)):
            Xs = T.imatrix('X')
        elif (ndim == 2):
            Xs = T.matrix('X').astype(theano.config.floatX)
        elif ndim == 4:
            Xs = T.tensor4('X').astype(theano.config.floatX)
        else:
            if self.recurrent:
                raise ValueError("Recurrent nets take 2D or 4D input, "
                                 "instead got {}D".format(ndim))
            else:
                raise ValueError("Input must be 2D or 4D, instead got {}D."
                                 "".format(ndim))
        self.symbols_ = {'Xs': Xs, 'ys': ys, 'variables': [Xs, ys]}

    def _initialize_functions(self):
        Xs, ys = self.symbols_['Xs'], self.symbols_['ys']

        # generate train function
        cost_train = self._get_cost(Xs, ys, False)
        updates = [layer.get_updates(cost_train) for layer in self
                   if layer.updater]
        updates = flatten(updates)
        self.train_ = function([Xs, ys], cost_train, updates=updates,
                               allow_input_downcast=True)

        # generate test function
        cost_test = self._get_cost(Xs, ys, True)
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
        for layer in self:
            layer.initialize(X, y)

        # initialize cost function
        if self.cost_function is None:
            self.cost_function = self._get_cost_function()

        # initialize encoder
        if self.encoder is None:
            self.encoder = self._get_encoder(y)

        # progress of cost function
        self.train_history_ = []
        self.valid_history_ = []

        self.table_ = OrderedDict([
            ('epoch', []),
            ('train loss', []),
            ('valid loss', []),
            ('best', []),
            ('train/val', []),
            ('valid acc', []),
            ('dur', []),
        ])

        # generate symbolic theano variables
        self._initialize_symbolic_variables(X, y)

        # generate theano functions
        self._initialize_functions()

        self.is_init_ = True

        # print layer infos
        if self.verbose:
            self._print_layer_info()

    def fit(self, X, y, max_iter=5):
        if not hasattr(self, 'is_init_'):
            self.initialize(X, y)

        table = self.table_
        num_epochs_past = len(self.train_history_)
        first_iteration = True
        best_valid = np.inf

        self._set_hash(X, y)
        X_train, X_valid, labels_train, labels_valid = (
            self.train_test_split(X, y))
        y_train = self.encoder.transform(labels_train.reshape(-1, 1))
        if labels_valid.size > 0:
            y_valid = self.encoder.transform(labels_valid.reshape(-1, 1))
        if self.verbose:
            print("## Training Information\n")

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
                if self.regression:
                    accuracy_valid = 0.
                else:
                    accuracy_valid = self.score(X_valid, labels_valid, True)
                mean_valid = np.mean(valid_cost) if valid_cost else 0
            else:
                accuracy_valid = 0.
                mean_valid = 0.
            self.valid_history_.append(mean_valid)

            # -------------- logging ----------------
            best_valid = mean_valid if mean_valid < best_valid else best_valid
            toc = time.time()
            table['epoch'].append(num_epochs_past + epoch)
            table['train loss'].append(mean_train)
            table['valid loss'].append(mean_valid if mean_valid else "")
            table['best'].append(best_valid if (best_valid == mean_valid)
                                 and mean_valid else "")
            table['train/val'].append(mean_train / mean_valid if mean_valid
                                      else "")
            table['valid acc'].append(accuracy_valid if (not self.regression)
                                      and accuracy_valid else "")
            table['dur'].append(toc - tic)
            self.log_ = tabulate(table, headers='keys', tablefmt='pipe',
                                 floatfmt='.4f')
            if not self.verbose:
                continue
            if first_iteration:
                print(self.log_.split('\n', 2)[0])
                print(self.log_.split('\n', 2)[1])
                first_iteration = False
            print(self.log_.rsplit('\n', 1)[-1])

        return self

    def _fit(self, X, y, mode='train'):
        if mode == 'train':
            cost = self.train_(X, y)
        else:
            cost = self.test_(X, y)
        return cost

    def feed_forward(self, X, deterministic=False):
        return self[-1].get_output(X, deterministic=deterministic)

    def predict_proba(self, X):
        probas = []
        for Xb, yb in self.iterator(X):
            probas.append(self._predict_proba(Xb))
        return np.vstack(probas)

    def predict(self, X):
        y_prob = self.predict_proba(X)
        if self.regression:
            return y_prob
        else:
            return np.argmax(y_prob, axis=1)

    def score(self, X, labels, accuracy=False):
        if X.shape[0] != labels.shape[0]:
            raise ValueError(
                "Incompatible input shapes: X.shape[0] is {},"
                " y.shape[0] is {}".format(X.shape[0], labels.shape[0])
            )
        if not accuracy:
            y = self.encoder.transform(labels.reshape(-1, 1))
            score_batches = []
            for Xb, yb in self.iterator(X, y):
                score_batches.extend(Xb.shape[0] * [self.test_(Xb, yb) + 0.])
            return np.mean(score_batches)
        else:
            y_pred = self.predict(X)
            return np.mean(labels == y_pred)

    def train_test_split(self, X, y):
        eval_size = self.eval_size
        if eval_size:
            if self.regression:
                kf = KFold(y.shape[0], round(1. / eval_size))
            else:
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
        shapes = [param.get_value().shape for param in
                  flatten(self.get_layer_params()) if param]
        nparams = reduce(op.add, [reduce(op.mul, shape) for
                                  shape in shapes])
        print("# Neural Network with {} learnable parameters"
              "".format(nparams))
        print("\n")

        if any([isinstance(layer, Conv2DLayer) for layer in self]):
            self._print_layer_conv_info()
        else:
            self._print_layer_plain_info()

    def _print_layer_plain_info(self):
        print("## Layer information")
        nums = range(len(self.layers))
        names = [layer.name for layer in self.layers]
        output_shapes = [layer.get_output_shape() for layer in self]
        totals = [str(reduce(op.mul, shape[1:])) for shape in output_shapes]
        output_shapes = ['x'.join(map(str, shape[1:]))
                         for shape in output_shapes]
        table = OrderedDict([
            ('#', nums),
            ('name', names),
            ('output shape', output_shapes),
            ('total', totals),
        ])
        print(tabulate(table, 'keys', tablefmt='pipe'))
        print("")

    def _print_layer_conv_info(self):
        if self.verbose > 1:
            detailed = True
            tablefmt = 'simple'
        else:
            detailed = False
            tablefmt = 'pipe'
        print("## Layer information")
        print(get_conv_infos(self, detailed=detailed, tablefmt=tablefmt))
        print("\nExplanation")
        print("    X, Y:    image dimensions")
        print("    cap.:    learning capacity")
        print("    cov.:    coverage of image")
        print("    {}: capacity too low (<1/6)"
              "".format("{}{}{}".format(MAG, "magenta", END)))
        print("    {}:    image coverage too high (>100%)"
              "".format("{}{}{}".format(CYA, "cyan", END)))
        print("    {}:     capacity too low and coverage too high\n"
              "".format("{}{}{}".format(RED, "red", END)))

    def save_params(self, filename):
        params = [layer.get_params() for layer in self]
        with open(filename, 'wb') as f:
            pickle.dump(params, f, -1)

    @staticmethod
    def _param_alignment(shapes0, shapes1):
        shapes0 = map(str, shapes0)
        shapes1 = map(str, shapes1)
        matcher = SequenceMatcher(a=shapes0, b=shapes1)
        matches = []
        for block in matcher.get_matching_blocks():
            if block.size == 0:
                continue
            matches.append((list(range(block.a, block.a + block.size)),
                            list(range(block.b, block.b + block.size))))
        result = [line for match in matches for line in zip(*match)]
        return result

    def load_params(self, filename):
        if not hasattr(self, 'is_init_'):
            raise AttributeError("Before loading params, please initialize "
                                 "the network with the data, e.g. by calling "
                                 "my_net.initialize(X, y).")

        params_loaded = flatten(pickle.load(open(filename, 'rb')))
        params_current = flatten([layer.get_params() for layer in self])

        params_src = [w for w in params_loaded if w]
        params_target = [w for w in params_current if w]
        shapes_src = [param.get_value().shape for param in params_src]
        shapes_target = [param.get_value().shape for param in params_target]
        matches = self._param_alignment(shapes_src, shapes_target)
        for i, j in matches:
            # ii, jj are the indices of the layers
            ii, jj = int(0.5 * i) + 1, int(0.5 * j) + 1
            param_src = params_src[i]
            params_target[j].set_value(param_src.get_value())
            if self.verbose:
                param_name = param_src.name + ' '
                param_shape = param_src.get_value().shape
                param_shape = 'x'.join(map(str, param_shape))
                layer_name = self[jj].name
                message = ("Loaded parameter {}(shape {}) from layer {} to "
                           "layer {} ({}).")
                print(message.format(
                    param_name, param_shape, ii, jj, layer_name))


class MultipleInputNet(NeuralNet):
    def _initialize_symbolic_variables(self, many_X, y):
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
        self.symbols_ = {'Xs': many_Xs, 'ys': ys, 'variables': many_X + [ys]}

    def _initialize_functions(self):
        Xs, ys = self.symbols_['Xs'], self.symbols_['ys']

        # generate train function
        cost_train = self._get_cost(Xs, ys, False)
        updates = [layer.get_updates(cost_train) for layer in self
                   if layer.updater]
        updates = flatten(updates)
        self.train_ = function(Xs + [ys], cost_train, updates=updates,
                               allow_input_downcast=True)

        # generate test function
        cost_test = self._get_cost(Xs, ys, True)
        self.test_ = function(Xs + [ys], cost_test,
                              allow_input_downcast=True)

        # generate predict function
        self._predict_proba = function(
            Xs, self.feed_forward(Xs, deterministic=True),
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

    def feed_forward(self, X, deterministic=False):
        return self[-1].get_output(X, deterministic=deterministic)

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
            y = self.encoder.transform(labels.reshape(-1, 1))
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
                if self.regression:
                    kf = KFold(y.shape[0], round(1. / eval_size))
                else:
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
