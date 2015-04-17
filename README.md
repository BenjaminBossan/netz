## Netz

A neural network package based on [Theano](http://deeplearning.net/software/theano/)

work in progress

Heavily inspired by:

* [Lasagne](https://github.com/benanne/Lasagne)
* [Nolearn](https://github.com/dnouri/nolearn)

### Example notebook

Look here for an example usage:

* [MNIST](http://nbviewer.ipython.org/github/BenjaminBossan/netz/blob/develop/MNIST.ipynb)
* [Convolutions](http://nbviewer.ipython.org/github/BenjaminBossan/netz/blob/develop/Convolutions3.ipynb)
* [How to go deep](http://nbviewer.ipython.org/github/BenjaminBossan/netz/blob/develop/Going_deep.ipynb)
* [Recurrent Nets](http://nbviewer.ipython.org/github/BenjaminBossan/netz/blob/develop/Recurrent_Rotten.ipynb)

### Tests

On some systems, the custom gradient checking algorithm does not seem to work for biases. However, it seems that still everything works as expected. To ignore those tests, run:

    py.test -k "not custom"
