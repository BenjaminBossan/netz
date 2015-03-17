# -*- coding: utf-8 -*-
from __future__ import division

from theano import shared
from theano import tensor as T

from utils import flatten
from utils import shared_zeros_like


class BaseUpdater(object):
    def __init__(
            self, learn_rate=shared(0.01),
            lambda1=shared(0.), lambda2=shared(0.)
    ):
        self.learn_rate = learn_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2

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
        """Get L1 and L2 regularization.

        Be careful to add this to the update in the
        ``update_function`` definition because this is not done
        automatically. The reason is that depending on the updating
        algorithm, this regularization is applied but to a subset of
        updates.

        """
        regularization = 0
        # don't regularize bias
        if not ((hasattr(param, 'name') and param.name.startswith('b_'))):
            regularization -= self.lambda1 * param + self.lambda2 * param ** 2
        return regularization


class SGD(BaseUpdater):
    def update_function(self, param, grad):
        update = []
        param_new = param - self.learn_rate * grad
        param_new -= self._get_regularization(param)
        update.append((param, param_new))
        return update


class Momentum(BaseUpdater):
    def __init__(
            self, momentum=shared(0.9),
            *args, **kwargs
    ):
        self.momentum = momentum
        super(Momentum, self).__init__(*args, **kwargs)

    def update_function(self, param, grad):
        update = []
        update_old = shared_zeros_like(param)

        update_new = self.momentum * update_old - self.learn_rate * grad
        update.append((update_old, update_new))

        param_new = param + update_new + self._get_regularization(param)
        update.append((param, param_new))
        return update


class Nesterov(Momentum):
    def update_function(self, param, grad):
        update = []
        update_old = shared_zeros_like(param)

        update_new = self.momentum * update_old - self.learn_rate * grad
        update.append((update_old, update_new))

        param_new = param + self.momentum * update_new - self.learn_rate * grad
        param_new += self._get_regularization(param)
        update.append((param, param_new))
        return update


class Adadelta(BaseUpdater):
    def __init__(self, learn_rate=1., rho=0.95, epsilon=1e-6,
                 *args, **kwargs):
        super(Adadelta, self).__init__(*args, **kwargs)
        self.learn_rate = learn_rate
        self.rho = rho
        self.epsilon = epsilon

    def update_function(self, param, grad):
        update = []
        accu = shared_zeros_like(param)
        accu_delta = shared_zeros_like(param)

        accu_new = self.rho * accu + (1 - self.rho) * grad ** 2
        update.append((accu, accu_new))

        update_new = (grad * T.sqrt(accu_delta + self.epsilon) /
                      T.sqrt(accu_new + self.epsilon))
        param_new = param - self.learn_rate * update_new
        param_new += self._get_regularization(param)
        update.append((param, param_new))

        accu_delta_new = (self.rho * accu_delta +
                          (1 - self.rho) * update_new ** 2)
        update.append((accu_delta, accu_delta_new))
        return update


class Adagrad(BaseUpdater):
    def __init__(self, learn_rate=1., epsilon=1e-6,
                 *args, **kwargs):
        super(Adagrad, self).__init__(*args, **kwargs)
        self.learn_rate = learn_rate
        self.epsilon = epsilon

    def update_function(self, param, grad):
        update = []
        accu = shared_zeros_like(param)

        accu_new = accu + grad ** 2
        update.append((accu, accu_new))

        param_new = param - (self.learn_rate * grad /
                             T.sqrt(accu_new + self.epsilon))
        param_new += self._get_regularization(param)
        update.append((param, param_new))
        return update


class RMSProp(BaseUpdater):
    def __init__(self, learn_rate=1., rho=0.95, epsilon=1e-6,
                 *args, **kwargs):
        super(RMSProp, self).__init__(*args, **kwargs)
        self.learn_rate = learn_rate
        self.rho = rho
        self.epsilon = epsilon

    def update_function(self, param, grad):
        update = []
        accu = shared_zeros_like(param)

        accu_new = self.rho * accu + (1 - self.rho) * grad ** 2
        update.append((accu, accu_new))

        param_new = param - (self.learn_rate * grad /
                             T.sqrt(accu_new + self.epsilon))
        param_new += self._get_regularization(param)
        update.append((param, param_new))
        return update
