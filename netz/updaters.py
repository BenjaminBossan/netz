# -*- coding: utf-8 -*-
from __future__ import division

from theano import shared

from utils import flatten
from utils import floatX
from utils import shared_zeros_like


class BaseUpdater(object):
    def __init__(
            self, learn_rate=shared(0.01),
            lambda1=shared(0.), lambda2=shared(0.)
    ):
        self.learn_rate = floatX(learn_rate)
        self.lambda1 = floatX(lambda1)
        self.lambda2 = floatX(lambda2)

    def get_updates(self, cost, layer):
        grads = layer.get_grads(cost)
        updates = []
        for param, grad in zip(layer.get_params(), grads):
            if not param:
                continue
            updates.append(self.update_function(param, grad))
        # flatten and remove empty
        updates = flatten(updates)
        return updates

    def udpate_function(self, param, grad):
        raise NotImplementedError

    def _get_regularization(self, param):
        regularization = 0
        # don't regularize bias
        if not ((hasattr(param, 'name') and param.name.startswith('b_'))):
            regularization -= self.lambda1 * param + self.lambda2 * param ** 2
        return regularization


class SGD(BaseUpdater):
    def update_function(self, param, grad):
        update = self.learn_rate * grad
        update += self._get_regularization(param)
        return (param, param - self.learn_rate * grad)


class Momentum(BaseUpdater):
    def __init__(
            self, momentum=shared(0.9),
            *args, **kwargs
    ):
        self.momentum = floatX(momentum)
        super(Momentum, self).__init__(*args, **kwargs)

    def update_function(self, param, grad):
        update = []
        name = param.name if hasattr(param, 'name') else ''
        old_update = shared_zeros_like(param, name=name + '_old_momentum')
        new_update = self.momentum * old_update - self.learn_rate * grad
        new_update += self._get_regularization(param)
        new_update.name = name + '_new_momentum'
        update.append((old_update, new_update))
        new_param = param + new_update
        new_param.name = name + '_new_val'
        update.append((param, new_param))
        return update


class Nesterov(Momentum):
    def update_function(self, param, grad):
        update = []
        name = param.name if hasattr(param, 'name') else ''
        old_update = shared_zeros_like(param, name=name + '_old_momentum')
        new_update = self.momentum * old_update - self.learn_rate * grad
        new_update += self._get_regularization(param)
        new_update.name = name + '_new_momentum'
        update.append((old_update, new_update))
        new_param = param + self.momentum * new_update - self.learn_rate * grad
        new_param.name = name + '_new_val'
        update.append((param, new_param))
        return update
