---
title: Theano ConcatLayer
layout: post
tags: [Deep Learning, Theano]
---

# ConcatLayer

l3 = ll.ConcatLayer([l1, l2], 1)


code：

**main.py**

```python
import numpy as np
from numpy import linalg as LA
import theano 
from theano import function
import theano.tensor as T
import lasagne.layers as ll
import lasagne

x1 = T.tensor4()
x2 = T.tensor4()

l1 = ll.InputLayer(shape=(None, 3, 10, 10), input_var=x1)
l2 = ll.InputLayer(shape=(None, 2, 10, 10), input_var=x2)

l3 = ll.ConcatLayer([l1, l2], axis=1, cropping=None)


x1_input = np.random.random((2, 3, 10, 10))
x2_input = np.random.random((2, 2, 10, 10))

x1_input = x1_input.astype(theano.config.floatX)
x2_input = x2_input.astype(theano.config.floatX)

out = ll.get_output(l3)
f = theano.function(outputs=[out], inputs=[x1, x2])

output = f(x1_input, x2_input)

print(output[0].shape)


```

Output:


```python

(2, 5, 10, 10)

```