---
title: Theano GaussianNoiseLayer
layout: post
tags: [Deep Learning, Theano]
---

# GaussianNoiseLayer



code：
**nn.py**

```python 

import numpy as np
import theano as th
import theano.tensor as T
import lasagne
from lasagne.layers import dnn
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class GaussianNoiseLayer(lasagne.layers.Layer):
    def __init__(self, incoming, sigma=0.1, **kwargs):
        super(GaussianNoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, use_last_noise=False, **kwargs):
        if deterministic or self.sigma == 0:
            return input
        else:
            if not use_last_noise:
                self.noise = self._srng.normal(input.shape, avg=0.0, std=self.sigma)
            return input + self.noise
```

**main.py**

```python
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as LL
import nn

x = T.matrix()

layers = [LL.InputLayer(shape=(None, 10))]
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.3))
# 训练阶段
G_out_F = LL.get_output(layers[-1], x, deterministic=False)
f = theano.function(inputs=[x], outputs=[G_out_F])

x_input = np.zeros((2, 10), dtype=theano.config.floatX)

out1 = f(x_input)
print(out1)
# 测试阶段
G_out_T = LL.get_output(layers[-1], x, deterministic=True)
f2 = theano.function(inputs=[x], outputs=[G_out_T])

out2 = f2(x_input)
print(out2)


```

Output:


```python
[array([[-0.05688584,  0.75282449, -0.45939332, -0.13663077, -0.43985614,
        -0.57286364, -0.22335915,  0.16102344, -0.56991279,  0.50476664],
       [-0.08891879, -0.71030414, -0.20151664,  0.73555982,  0.04064834,
         0.15175492, -0.04717061,  0.0333882 ,  0.80900091,  0.27978507]], dtype=float32)]
[array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)]

```