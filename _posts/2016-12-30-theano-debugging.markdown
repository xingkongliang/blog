---
title: Theano Debugging
layout: post
tags: [Others, Theano]
---

### 1. compute_test_value

> config.compute_test_value

String Value: 'off', 'ignore', 'warn', 'raise'.

Default: 'off'

将属性设置为'off'以外的其他属性会激活调试机制，Theano在构建时，即使执行这个graph on-the-fly。这允许用户在应用优化之前早期发现错误（例如尺寸不匹配）。

Theano将使用有用户提供的常量和/或者共享变量执行这个graph。通过写入它们的'tag.test_value'属性（例如 .tag.test_value = numpy.random.rand(5, 4)），可以用测试值来增强纯符号变量（例如 x = T.dmatrix()）

当不为'off'时，此选项的值指示当Op的输入不提供适当的测试值时发生的情况：

'ignore' 将静默地跳过此操作的调试机制

'warn' 将引发UserWarning并跳过此Op的调试机制

'raise' 将引发异常(Exception)

``` python
>>> from theano import config
>>> config.compute_test_value = 'raise'
>>> x = T.vector()
>>> import numpy as np
>>> x.tag.test_value = np.ones((2,))
>>> y = T.vector()
>>> y.tag.test_value = np.ones((3,))
>>> x + y
...
ValueError: Input dimension mis-match.
(input[0].shape[0] = 2, input[1].shape[0] = 3)
```

```python 
# Let's create another matrix, "B"
B = T.matrix('B')
# And, a symbolic variable which is just A (from above) dotted against B
# At this point, Theano doesn't know the shape of A or B, so there's no way for it to know whether A dot B is valid.
C = T.dot(A, B)
# Now, let's try to use it
C.eval({A: np.zeros((3, 4), dtype=theano.config.floatX),
        B: np.zeros((5, 6), dtype=theano.config.floatX)})
```
Out：

```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-14-75863a5c9f35> in <module>()
      6 # Now, let's try to use it
      7 C.eval({A: np.zeros((3, 4), dtype=theano.config.floatX),
----> 8         B: np.zeros((5, 6), dtype=theano.config.floatX)})

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/gof/graph.pyc in eval(self, inputs_to_values)
    465         args = [inputs_to_values[param] for param in inputs]
    466 
--> 467         rval = self._fn_cache[inputs](*args)
    468 
    469         return rval

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/compile/function_module.pyc in __call__(self, *args, **kwargs)
    865                     node=self.fn.nodes[self.fn.position_of_error],
    866                     thunk=thunk,
--> 867                     storage_map=getattr(self.fn, 'storage_map', None))
    868             else:
    869                 # old-style linkers raise their own exceptions

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/gof/link.pyc in raise_with_op(node, thunk, exc_info, storage_map)
    312         # extra long error message in that case.
    313         pass
--> 314     reraise(exc_type, exc_value, exc_trace)
    315 
    316 

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/compile/function_module.pyc in __call__(self, *args, **kwargs)
    853         t0_fn = time.time()
    854         try:
--> 855             outputs = self.fn()
    856         except Exception:
    857             if hasattr(self.fn, 'position_of_error'):

ValueError: Shape mismatch: x has 4 cols (and 3 rows) but y has 5 rows (and 6 cols)
Apply node that caused the error: Dot22(A, B)
Toposort index: 0
Inputs types: [TensorType(float64, matrix), TensorType(float64, matrix)]
Inputs shapes: [(3, 4), (5, 6)]
Inputs strides: [(32, 8), (48, 8)]
Inputs values: ['not shown', 'not shown']
Outputs clients: [['output']]

HINT: Re-running with most Theano optimization disabled could give you a back-trace of when this node was created. This can be done with by setting the Theano flag 'optimizer=fast_compile'. If that does not work, Theano optimizations can be disabled with 'optimizer=None'.
HINT: Use the Theano flag 'exception_verbosity=high' for a debugprint and storage map footprint of this apply node.
```

由于theano的报错的信息不是很明确。当计算的Theano表达式非常复杂的时候，像这样的错误可能特别混乱。所以我们可以使用“测试值”来解决这个问题。并不是所有Theano的方法都可以使用测试值（比如说scan）。

```python
# This tells Theano we're going to use test values, and to warn when there's an error with them.
# The setting 'warn' means "warn me when I haven't supplied a test value"
theano.config.compute_test_value = 'warn'
# Setting the tag.test_value attribute gives the variable its test value
A.tag.test_value = np.random.random((3, 4)).astype(theano.config.floatX)
B.tag.test_value = np.random.random((5, 6)).astype(theano.config.floatX)
# Now, we get an error when we compute C which points us to the correct line!
C = T.dot(A, B)
```
Out:

```

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-15-038674a75ca1> in <module>()
      6 B.tag.test_value = np.random.random((5, 6)).astype(theano.config.floatX)
      7 # Now, we get an error when we compute C which points us to the correct line!
----> 8 C = T.dot(A, B)

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/tensor/basic.pyc in dot(a, b)
   5417         return tensordot(a, b, [[a.ndim - 1], [numpy.maximum(0, b.ndim - 2)]])
   5418     else:
-> 5419         return _dot(a, b)
   5420 
   5421 

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/gof/op.pyc in __call__(self, *inputs, **kwargs)
    649                 thunk.outputs = [storage_map[v] for v in node.outputs]
    650 
--> 651                 required = thunk()
    652                 assert not required  # We provided all inputs
    653 

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/gof/op.pyc in rval(p, i, o, n)
    863             # default arguments are stored in the closure of `rval`
    864             def rval(p=p, i=node_input_storage, o=node_output_storage, n=node):
--> 865                 r = p(n, [x[0] for x in i], o)
    866                 for o in node.outputs:
    867                     compute_map[o][0] = True

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/tensor/basic.pyc in perform(self, node, inp, out)
   5235         # gives a numpy float object but we need to return a 0d
   5236         # ndarray
-> 5237         z[0] = numpy.asarray(numpy.dot(x, y))
   5238 
   5239     def grad(self, inp, grads):

ValueError: shapes (3,4) and (5,6) not aligned: 4 (dim 1) != 5 (dim 0)
```


### 2. min_informative_str
```python
>>> x = T.scalar()
>>> y = T.scalar()
>>> z = x + y
>>> z.name = 'z'
>>> a = 2. * z
>>> from theano.printing import min_informative_str
>>> print min_informative_str(a)
A. Elemwise{mul,no_inplace}
 B. TensorConstant{2.0}
 C. z
```

### 3. debugprint

```python
>>> from theano.printing import debugprint
>>> debugprint(a)
Elemwise{mul,no_inplace} [id A] ''
 |TensorConstant{2} [id B]
 |Elemwise{add,no_inplace} [id C] 'z'
   |<TensorType(float32, scalar)> [id D]
   |<TensorType(float32, scalar)> [id E]

```

### 4. Print

```python
x = theano.tensor.vector()
x = theano.printing.Print('x', attrs=['min','max'])(x)
```

### 5. Accessing a function's fgraph

```python
>>> x = T.scalar()
>>> y = x / x
>>> f = function([x], y)
>>> debugprint(f.maker.fgraph.outputs[0])
DeepCopyOp [@A] ''
|TensorConstant{1.0} [@B]
```

### 6. WrapLinkers

- 在术语中，链接器是驱动函数的执行的对象。遍历所有Apply节点，并调用输入上的Op的代码以生成输出
- 您可以编写自己的链接器，以使用额外的功能包装每个单独的调用


- 示例使用：
- 通过使用一个Wraplinker保存所有内容，并重新加载它并检查新值是否匹配，来测试该行为是确定性的。
- 如果任何值是NaN，则引发错误


```python
from theano.compile import Mode
def my_callback(i, node, fn):
    # add any preprocessing here
    fn()
    # add any postprocessing here
class MyWrapLinker(Mode):
    def __init__(self):
        wrap_linker = theano.gof.WrapLinkerMany(
            [theano.gof.OpWiseCLinker()],
            [my_callback])
    super(MyWrapLinker, self).__init__(wrap_linker,
            optimizer='fast_run')
my_mode = MyWrapLinker()
f = function(inputs, outputs, mode=my_mode)
```

### 7.DebugMode

```python

# A simple division function
num = T.scalar('num')
den = T.scalar('den')
divide = theano.function([num, den], num/den)
print(divide(10, 2))
# This will cause a NaN
print(divide(0, 0))
```
Out:

```
5.0
nan
```

使用DebugMode

```python
# To compile a function in debug mode, just set mode='DebugMode'
divide = theano.function([num, den], num/den, mode='DebugMode')
# NaNs now cause errors
print(divide(0, 0))
```
Out；

```

---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-18-fd8e17a1c37b> in <module>()
      1 # To compile a function in debug mode, just set mode='DebugMode'
----> 2 divide = theano.function([num, den], num/den, mode='DebugMode')
      3 # NaNs now cause errors
      4 print(divide(0, 0))

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/compile/function.pyc in function(inputs, outputs, mode, updates, givens, no_default_updates, accept_inplace, name, rebuild_strict, allow_input_downcast, profile, on_unused_input)
    306                    on_unused_input=on_unused_input,
    307                    profile=profile,
--> 308                    output_keys=output_keys)
    309     # We need to add the flag check_aliased inputs if we have any mutable or
    310     # borrowed used defined inputs

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/compile/pfunc.pyc in pfunc(params, outputs, mode, updates, givens, no_default_updates, accept_inplace, name, rebuild_strict, allow_input_downcast, profile, on_unused_input, output_keys)
    524                          accept_inplace=accept_inplace, name=name,
    525                          profile=profile, on_unused_input=on_unused_input,
--> 526                          output_keys=output_keys)
    527 
    528 

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/compile/function_module.pyc in orig_function(inputs, outputs, mode, accept_inplace, name, profile, on_unused_input, output_keys)
   1768                    on_unused_input=on_unused_input,
   1769                    output_keys=output_keys).create(
-> 1770             defaults)
   1771 
   1772     t2 = time.time()

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/compile/debugmode.pyc in create(self, defaults, trustme, storage_map)
   2638         # Get a function instance
   2639         _fn, _i, _o = self.linker.make_thunk(input_storage=input_storage,
-> 2640                                              storage_map=storage_map)
   2641         fn = self.function_builder(_fn, _i, _o, self.indices,
   2642                                    self.outputs, defaults, self.unpack_single,

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/gof/link.pyc in make_thunk(self, input_storage, output_storage, storage_map)
    688         return self.make_all(input_storage=input_storage,
    689                              output_storage=output_storage,
--> 690                              storage_map=storage_map)[:3]
    691 
    692     def make_all(self, input_storage, output_storage):

/usr/local/lib/python2.7/site-packages/Theano-0.7.0-py2.7.egg/theano/compile/debugmode.pyc in make_all(self, profiler, input_storage, output_storage, storage_map)
   1945         # Precompute some things for storage pre-allocation
   1946         try:
-> 1947             def_val = int(config.unittests.rseed)
   1948         except ValueError:
   1949             def_val = 666

AttributeError: 'TheanoConfigParser' object has no attribute 'unittests'
```