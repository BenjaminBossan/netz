# -*- coding: utf-8 -*-
from __future__ import division

from theano import shared

from utils import flatten
from utils import shared_zeros_like


class BaseUpdater(object):
    def __init__(self, learn_rate):
        self.learn_rate = learn_rate

    def get_updates(self, cost, layers):
        layer_grads = [layer.get_grads(cost) for layer in layers]
        updates = []
        for layer, grads in zip(layers, layer_grads):
            update = []
            for param, grad in zip(layer.get_params(), grads):
                if not param:
                    continue
                update.append(self.update_function(param, grad))
            updates.append(update)
        # flatten and remove empty
        updates = flatten(updates)
        return updates

    def udpate_function(self, param, grad):
        raise NotImplementedError


class GradientChecker(BaseUpdater):
    def update_function(self, param, grad):
        return param, grad


class SGD(BaseUpdater):
    def update_function(self, param, grad):
        return (param, param - self.learn_rate * grad)


class Momentum(BaseUpdater):
    def __init__(self, learn_rate, momentum=shared(0.9)):
        self.learn_rate = learn_rate
        self.momentum = momentum

    def update_function(self, param, grad):
        update = []
        name = param.name if hasattr(param, 'name') else ''
        old_update = shared_zeros_like(param, name=name + '_old_momentum')
        new_update = self.momentum * old_update - self.learn_rate * grad
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
        new_update.name = name + '_new_momentum'
        update.append((old_update, new_update))
        new_param = param + self.momentum * new_update - self.learn_rate * grad
        new_param.name = name + '_new_val'
        update.append((param, new_param))
        return update
