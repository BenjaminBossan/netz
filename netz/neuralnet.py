# -*- coding: utf-8 -*-
from __future__ import division
from collections import defaultdict

from sklearn.base import BaseEstimator
from theano import function
from theano import tensor as T

from costfunctions import crossentropy
from updaters import SGD
from utils import flatten


class NeuralNet(BaseEstimator):
    def __init__(self, layers,
                 update=SGD, update_kwargs={'learn_rate': 0.01},
                 cost_function=crossentropy):
        self.layers = layers
        self.update = update
        self.update_kwargs = update_kwargs
        self.cost_function = cost_function

    def get_layer_params(self):
        return [layer.get_params() for layer in self.layers]

    def initialize(self, X, y):
        # symbolic
        Xs, ys = T.dmatrix('X'), T.dmatrix('y')

        # set layer names
        name_counts = defaultdict(int)
        for layer in self.layers:
            if layer.name is not None:
                continue
            raw_name = str(layer).split(' ')[0].split('.')[-1]
            raw_name = raw_name.replace('Layer', '').lower()
            name_counts[raw_name] += 1
            name = raw_name + str(name_counts[raw_name] - 1)
            layer.set_name(name)

        # connect layers
        for layer0, layer1 in zip(self.layers[:-1], self.layers[1:]):
            layer0.set_next_layer(layer1)
            layer1.set_prev_layer(layer0)

        # initialize layers
        for layer in self.layers:
            layer.initialize(X, y)

        self.cost_history_ = []

        # generate train function
        y_pred = self.feed_forward(Xs)
        y_pred.name = 'y_pred'
        cost = self.cost_function(ys, y_pred)
        updater = self.update(**self.update_kwargs)
        updates = updater.get_updates(cost, self.layers)
        updates = flatten(updates)  # flatten and remove empty
        self.train_ = function([Xs, ys], cost, updates=updates)

        # generate predict function
        self._predict = function([Xs], self.feed_forward(Xs))

        # generate score function
        self._score = function(
            [Xs, ys], self.cost_function(ys, self.feed_forward(Xs))
        )

        self.is_init_ = True

    def fit(self, X, y, max_iter=5):
        if not hasattr(self, 'is_init_'):
            self.initialize(X, y)
        for epoch in range(max_iter):
            self._fit(X, y)
        return self

    def feed_forward(self, X):
        return self.layers[-1].get_output(X)

    def predict(self, X):
        return self._predict(X)

    def score(self, X, y):
        return self._score(X, y)

    def _fit(self, X, y):
        cost = self.train_(X, y)
        self.cost_history_.append(cost)
