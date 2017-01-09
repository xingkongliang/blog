---
title: Theano MinibatchLayer
layout: post
tags: [Deep Learning, Theano]
---

# MinibatchLayer



code：
**nn.py**

```python 

import numpy as np
import theano as th
import theano.tensor as T
import lasagne
from lasagne.layers import dnn
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# minibatch discrimination layer
class MinibatchLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_kernels, dim_per_kernel=5, theta=lasagne.init.Normal(0.05),
                 log_weight_scale=lasagne.init.Constant(0.), b=lasagne.init.Constant(-1.), **kwargs):
        super(MinibatchLayer, self).__init__(incoming, **kwargs)
        self.num_kernels = num_kernels
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.theta = self.add_param(theta, (num_inputs, num_kernels, dim_per_kernel), name="theta")
        self.log_weight_scale = self.add_param(log_weight_scale, (num_kernels, dim_per_kernel), name="log_weight_scale")
        self.W = self.theta * (T.exp(self.log_weight_scale)/T.sqrt(T.sum(T.square(self.theta),axis=0))).dimshuffle('x',0,1)
        self.b = self.add_param(b, (num_kernels,), name="b")
        

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:])+self.num_kernels)

    def get_output_for(self, input, init=False, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        
        activation = T.tensordot(input, self.W, [[1], [0]])
        abs_dif = (T.sum(abs(activation.dimshuffle(0,1,2,'x') - activation.dimshuffle('x',1,2,0)),axis=2)
                    + 1e6 * T.eye(input.shape[0]).dimshuffle(0,'x',1))
        if init:
            mean_min_abs_dif = 0.5 * T.mean(T.min(abs_dif, axis=2),axis=0)
            abs_dif /= mean_min_abs_dif.dimshuffle('x',0,'x')
            self.init_updates = [(self.log_weight_scale, self.log_weight_scale-T.log(mean_min_abs_dif).dimshuffle(0,'x'))]
        
        f = T.sum(T.exp(-abs_dif),axis=2)

        if init:
            mf = T.mean(f,axis=0)
            f -= mf.dimshuffle('x',0)
            self.init_updates.append((self.b, -mf))
        else:
            f += self.b.dimshuffle('x',0)

        return T.concatenate([input, f], axis=1)
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

x = T.matrix()

layers = [LL.InputLayer(shape=(None, 10))]
layers.append(nn.MinibatchLayer(layers[-1], num_kernels=8))

outlayer = layers[-1]

x_input = np.random.random((4, 10))
x_input = x_input.astype(theano.config.floatX)
out = LL.get_output(layers[-1], x)

f = function([x], [out])
outlayer = layers[-1]

log_weight_scale = outlayer.get_params()[1]

theta = outlayer.get_params()[0]
print(theta.eval().shape)

print(log_weight_scale.eval().shape)

W = theta*(T.exp(log_weight_scale)/T.sqrt(T.sum(T.square(theta), axis=0))).dimshuffle('x',0,1)
print(W.eval().shape)
print(LA.norm(W.eval(), axis=(0)))

activation = T.tensordot(x, W, [[1],[0]])

print(activation.eval({x:x_input}).shape)

print((activation.dimshuffle(0,1,2,'x') - activation.dimshuffle('x',1,2,0)).eval({x:x_input}).shape)

abs_dif = (T.sum(abs(activation.dimshuffle(0,1,2,'x') - activation.dimshuffle('x',1,2,0)),axis=2) + 1e6 * T.eye(x.shape[0]).dimshuffle(0,'x',1))

print(abs_dif.eval({x:x_input}).shape)

f = T.sum(T.exp(-abs_dif),axis=2)

print(f.eval({x:x_input}).shape)

```

Output:


```python
In [10]: print(W_param.eval())
CudaNdarray([[[[ 0.04999165  0.02313326]
   [-0.04298718  0.09443879]]

  [[-0.0129938   0.01973312]
   [ 0.03871491  0.08463974]]]


 [[[ 0.06936961  0.01436294]
   [ 0.08334961 -0.00598382]]

  [[ 0.05869503 -0.12156428]
   [-0.04094811  0.05975082]]]


 [[[ 0.03075594 -0.03216397]
   [-0.0249342   0.0733012 ]]

  [[-0.07199171  0.000842  ]
   [ 0.03408016 -0.00191294]]]])

In [11]: print(T.sum(T.square(W_param),axis=W_axes_to_sum,keepdims=True).eval().shape)
(3, 1, 1, 1)

In [13]: print(W_new.eval())
[[[[ 0.32947862  0.15246375]
   [-0.28331444  0.62241518]]

  [[-0.08563788  0.13005453]
   [ 0.25515732  0.5578329 ]]]


 [[[ 0.36832467  0.0762614 ]
   [ 0.4425528  -0.03177166]]

  [[ 0.31164694 -0.64545727]
   [-0.21741794  0.31725276]]]


 [[[ 0.25697976 -0.2687445 ]
   [-0.20833649  0.61246467]]

  [[-0.60152328  0.00703529]
   [ 0.28475517 -0.01598351]]]]

In [14]: print(W_new.eval().shape)
(3, 2, 2, 2)

In [16]: LA.norm(W2, axis=1)
Out[16]: array([ 0.99999994,  1.        ,  1.        ], dtype=float32)


```