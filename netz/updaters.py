# -*- coding: utf-8 -*-
from __future__ import division

from theano import tensor as T

from utils import flatten
from utils import shared_zeros_like
from utils import to_32

one = to_32(1.)


class BaseUpdater(object):
    def __init__(
            self, learn_rate=0.01,
    ):
        self.learn_rate = to_32(learn_rate)

    def get_updates(self, cost, layer):
        grads = layer.get_grads(cost)
        updates = []
        for param, grad in zip(layer.get_params(), grads):
            if not param:
                continue
            grad = to_32(grad)
            updates.append(self.update_function(param, grad))
        # flatten and remove empty
        updates = flatten(updates)
        return updates

    def udpate_function(self, param, grad):
        raise NotImplementedError


class SGD(BaseUpdater):
    def update_function(self, param, grad):
        update = []
        param_new = param - self.learn_rate * grad
        update.append((param, param_new))
        return update


class Momentum(BaseUpdater):
    def __init__(
            self, momentum=0.9,
            *args, **kwargs
    ):
        self.momentum = to_32(momentum)
        super(Momentum, self).__init__(*args, **kwargs)

    def update_function(self, param, grad):
        update = []
        update_old = shared_zeros_like(param)
        update_new = self.momentum * update_old - self.learn_rate * grad
        update.append((update_old, update_new))

        param_new = param + update_new
        update.append((param, param_new))
        return update


class Nesterov(Momentum):
    def update_function(self, param, grad):
        update = []
        update_old = shared_zeros_like(param)

        update_new = self.momentum * update_old - self.learn_rate * grad
        update.append((update_old, update_new))

        param_new = param + self.momentum * update_new - self.learn_rate * grad
        update.append((param, param_new))
        return update


class Adadelta(BaseUpdater):
    def __init__(self, learn_rate=1., rho=0.95, epsilon=1e-6,
                 *args, **kwargs):
        super(Adadelta, self).__init__(*args, **kwargs)
        self.learn_rate = to_32(learn_rate)
        self.rho = to_32(rho)
        self.epsilon = to_32(epsilon)

    def update_function(self, param, grad):
        update = []
        accu = shared_zeros_like(param)
        accu_delta = shared_zeros_like(param)

        accu_new = self.rho * accu + (one - self.rho) * grad ** 2
        update.append((accu, accu_new))

        update_new = (grad * T.sqrt(accu_delta + self.epsilon) /
                      T.sqrt(accu_new + self.epsilon))
        param_new = param - self.learn_rate * update_new
        update.append((param, param_new))

        accu_delta_new = (self.rho * accu_delta + (one - self.rho) *
                          update_new ** 2)
        update.append((accu_delta, accu_delta_new))
        return update


class Adagrad(BaseUpdater):
    def __init__(self, learn_rate=1., epsilon=1e-6,
                 *args, **kwargs):
        super(Adagrad, self).__init__(*args, **kwargs)
        self.learn_rate = to_32(learn_rate)
        self.epsilon = to_32(epsilon)

    def update_function(self, param, grad):
        update = []
        accu = shared_zeros_like(param)

        accu_new = accu + grad ** 2
        update.append((accu, accu_new))

        param_new = param - (self.learn_rate * grad /
                             T.sqrt(accu_new + self.epsilon))
        update.append((param, param_new))
        return update


class RMSProp(BaseUpdater):
    def __init__(self, learn_rate=1., rho=0.95, epsilon=1e-6,
                 *args, **kwargs):
        super(RMSProp, self).__init__(*args, **kwargs)
        self.learn_rate = to_32(learn_rate)
        self.rho = to_32(rho)
        self.epsilon = to_32(epsilon)

    def update_function(self, param, grad):
        update = []
        accu = shared_zeros_like(param)

        accu_new = self.rho * accu + (one - self.rho) * grad ** 2
        update.append((accu, accu_new))

        param_new = param - (self.learn_rate * grad /
                             T.sqrt(accu_new + self.epsilon))
        update.append((param, param_new))
        return update
