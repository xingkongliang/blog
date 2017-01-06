---
title: Theano WeightNormLayer
layout: post
tags: [Deep Learning, Theano]
---

# WeightNormLayer

如果输入是维度是(None, C, H, W)的4维张量（维度是（None， N），也可以），WeightNormLayer放在卷积层之后，实现了对卷积层权值W的归一化。

如果经过卷积层输出T个特征图，则W的维度是(T, C, W_H, W_W)。
计算公式为：

$$W_new = \frac{W}{\sqrt{\sum^{C,W\_N,W\_W}_{i\in 1,j\in 1,k\in 1} w_{ijk}^2}}$$

code：
**nn.py**

```python 

import numpy as np
import theano as th
import theano.tensor as T
import lasagne
from lasagne.layers import dnn
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class WeightNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
                 W=lasagne.init.Normal(0.05), train_g=False, init_stdv=1., nonlinearity=relu, **kwargs):
        super(WeightNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        self.init_stdv = init_stdv
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g", regularizable=False, trainable=train_g)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]

        # scale weights in layer below
        incoming.W_param = incoming.W
        #incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim==4:
            if isinstance(incoming, Deconv2DLayer):
                W_axes_to_sum = (0,2,3)
                W_dimshuffle_args = ['x',0,'x','x']
            else:
                W_axes_to_sum = (1,2,3)
                W_dimshuffle_args = [0,'x','x','x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x',0]
        if g is not None:
            incoming.W = incoming.W_param * (self.g/T.sqrt(1e-6 + T.sum(T.square(incoming.W_param),axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / T.sqrt(1e-6 + T.sum(T.square(incoming.W_param),axis=W_axes_to_sum,keepdims=True))

    def get_output_for(self, input, init=False, **kwargs):
        if init:
            m = T.mean(input, self.axes_to_sum)
            input -= m.dimshuffle(*self.dimshuffle_args)
            inv_stdv = self.init_stdv/T.sqrt(T.mean(T.square(input), self.axes_to_sum))
            input *= inv_stdv.dimshuffle(*self.dimshuffle_args)
            self.init_updates = [(self.b, -m*inv_stdv), (self.g, self.g*inv_stdv)]
        elif hasattr(self,'b'):
            input += self.b.dimshuffle(*self.dimshuffle_args)
            
        return self.nonlinearity(input)

def weight_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return WeightNormLayer(layer, nonlinearity=nonlinearity, **kwargs)se
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
layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(layers[-1], 3, (2,2), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
outlayer = layers[-1]

x_input = np.random.random((1,2,6,6))
x_input = x_input.astype(theano.config.floatX)
out = LL.get_output(layers[-1], x)

f = function([x], [out])
outlayer = layers[-1]

W_axes_to_sum = (1,2,3)
W_dimshuffle_args = [0, 'x','x','x']

W_param = outlayer.input_layer.get_params()[0]
print(W_param)

print(T.sum(T.square(W_param),axis=W_axes_to_sum,keepdims=True).eval().shape)

W_new = W_param/T.sqrt(T.sum(T.square(W_param),axis=W_axes_to_sum,keepdims=True))

print(W_new.eval())
print(W_new.eval().shape)

W2 = W_new.eval().reshape(3, -1)
print(LA.norm(W2, axis=1))


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