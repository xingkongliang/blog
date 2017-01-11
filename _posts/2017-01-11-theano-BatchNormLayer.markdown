---
title: Theano BatchNormLayer
layout: post
tags: [Deep Learning, Theano]
---

# BatchNormLayer



code：
**nn.py**

```python 

import numpy as np
import theano as th
import theano.tensor as T
import lasagne
from lasagne.layers import dnn
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class BatchNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), nonlinearity=relu, **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g", regularizable=False)
        self.avg_batch_mean = self.add_param(lasagne.init.Constant(0.), (k,), name="avg_batch_mean", regularizable=False, trainable=False)
        self.avg_batch_var = self.add_param(lasagne.init.Constant(1.), (k,), name="avg_batch_var", regularizable=False, trainable=False)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            norm_features = (input-self.avg_batch_mean.dimshuffle(*self.dimshuffle_args)) / T.sqrt(1e-6 + self.avg_batch_var).dimshuffle(*self.dimshuffle_args)
        else:
            batch_mean = T.mean(input,axis=self.axes_to_sum).flatten()
            centered_input = input-batch_mean.dimshuffle(*self.dimshuffle_args)
            batch_var = T.mean(T.square(centered_input),axis=self.axes_to_sum).flatten()
            batch_stdv = T.sqrt(1e-6 + batch_var)
            norm_features = centered_input / batch_stdv.dimshuffle(*self.dimshuffle_args)

            # BN updates
            new_m = 0.9*self.avg_batch_mean + 0.1*batch_mean
            new_v = 0.9*self.avg_batch_var + T.cast((0.1*input.shape[0])/(input.shape[0]-1),th.config.floatX)*batch_var
            self.bn_updates = [(self.avg_batch_mean, new_m), (self.avg_batch_var, new_v)]

        if hasattr(self, 'g'):
            activation = norm_features*self.g.dimshuffle(*self.dimshuffle_args)
        else:
            activation = norm_features
        if hasattr(self, 'b'):
            activation += self.b.dimshuffle(*self.dimshuffle_args)

        return self.nonlinearity(activation)

def batch_norm(layer, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), **kwargs):
    """
    adapted from https://gist.github.com/f0k/f1a6bd3c8585c400c190
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    else:
        nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, b, g, nonlinearity=nonlinearity, **kwargs)

```

**main.py**

```python
import numpy as np
from numpy import linalg as LA
import theano 
from theano import function
import theano.tensor as T
import lasagne.layers as LL
import lasagne

from lasagne.init import Normal
from lasagne.layers import dnn
import nn

x = T.tensor4()

layers = [LL.InputLayer(shape=(None, 2, 6, 6))]
layers.append(nn.weight_norm())
outlayer = layers[-1]

x_input = np.random.random((1,2,6,6))
x_input = x_input.astype(theano.config.floatX)
out = LL.get_output(layers[-1], x)




```

Output:


```python


```