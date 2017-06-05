---
title: Python中functools.partial
layout: post
tags: [Python]
---


# parital

```
import functools
def add(a, b):
    return a + b

add(4, 2)
output: 6

plus3 = functools.partial(add, 3)
plus5 = functools.partial(add, 5)
plus3(4)
output: 7

plus3(7)
output: 10

plus5(10)
output: 15
```
### 函数的partial应用
典型的，函数在执行时，要带上所有必要的参数进行调用。然后，有时参数可以在函数被调用之前提前获知。这种情况下，一个函数有一个或多个参数预先就能用上，以便函数能用更少的参数进行调用。