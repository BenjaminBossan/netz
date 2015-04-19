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

    def get_updates(self, cost, grads, params):
        updates = []
        for param, grad in zip(params, grads):
            if not param:
                continue
            grad = to_32(grad)
            updates.append(self._update_function(param, grad))
        # flatten and remove empty
        updates = flatten(updates)
        return updates

    def _update_function(self, param, grad):
        raise NotImplementedError


class SGD(BaseUpdater):
    def _update_function(self, param, grad):
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

    def _update_function(self, param, grad):
        update = []
        update_old = shared_zeros_like(param)
        update_new = self.momentum * update_old - self.learn_rate * grad
        update.append((update_old, update_new))

        param_new = param + update_new
        update.append((param, param_new))
        return update


class Nesterov(Momentum):
    def _update_function(self, param, grad):
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

    def _update_function(self, param, grad):
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

    def _update_function(self, param, grad):
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

    def _update_function(self, param, grad):
        update = []
        accu = shared_zeros_like(param)

        accu_new = self.rho * accu + (one - self.rho) * grad ** 2
        update.append((accu, accu_new))

        param_new = param - (self.learn_rate * grad /
                             T.sqrt(accu_new + self.epsilon))
        update.append((param, param_new))
        return update


class GradientClipping(object):
    def __init__(self, norm_max, updater):
        self.norm_max = norm_max
        self.updater = updater

    @staticmethod
    def _get_norm(var):
        return T.sqrt(T.sum(T.sqr(var)))

    @staticmethod
    def _clip_norm(var, norm_var, norm_max):
        clipped = T.switch(
            T.ge(norm_var, norm_max),
            var * norm_max / norm_var,
            var)
        return clipped

    def get_updates(self, *args, **kwargs):
        norm_max = self.norm_max
        updates = self.updater.get_updates(*args, **kwargs)
        clipped_updates = []
        for var_old, var_new in updates:
            norm_var_new = self._get_norm(var_new)
            var_new = self._clip_norm(var_new, norm_var_new, norm_max)
            clipped_updates.append((var_old, var_new))
        return clipped_updates
