import theano

from .base import BaseLayer


__all__ = ['ReshapeLayer']


class ReshapeLayer(BaseLayer):
    def __init__(self, shape, *args, **kwargs):
        super(ReshapeLayer, self).__init__(*args, **kwargs)
        self.shape = shape

    def set_updater(self, *args, **kwargs):
        pass

    def initialize(self, X, y):
        self.updater = None

    def get_grads(self, cost):
        return [None]

    def get_params(self):
        return [None]

    def get_output(self, X, *args, **kwargs):
        input = self.prev_layer.get_output(X, *args, **kwargs)
        return input.reshape(self.shape).astype(theano.config.floatX)

    def get_output_shape(self):
        input_shape = self.prev_layer.get_output_shape()
        return (input_shape[0],) + self.shape[1:]
