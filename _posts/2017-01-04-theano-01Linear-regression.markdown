---
title: 01.Linear Regression Theano实现
layout: post
tags: [Machine Learning, Theano]
---

# 线性回归 

对于给定的数据集$$D={(x_1,y_1),(x_2,y_2),...,(x_n,y_n)}$$，其中$$x_i=(x_{i1};x_{i2};...;x_{id})$$，$$y_i\in \mathbb{R}$$。“线性回归”（linear regression）试图学得一个线性模型以尽可能准确的预测输出的标记值。

$$
\hat{y}(w,x)=w_0+w_1x_1+...+w_px_p
$$

在sklearn中，$$w = (w_1, ..., w_p)$$ 作为 coef_ 并且 $$w_0$$ 作为 intercept_。

我们通过最小化均方误差来求得w（即为上式$$w$$）和b（即为上式$$w_0$$）的值，即：

$$
(w^*,b^*)=\mathop{argmin}\limits_{(w,b)}\sum^m_{i=1}(f(x_i)-y_i)^2
$$

基于均方误差最小化来进行模型求解的方法称为“最小二乘法”（least square method）。在线性回归中，最小二乘法就是寻找一条直线，使得多有样本到直线上的欧氏距离之和最小。

这个函数式一个凸函数，所以只有一个最优解，因此可以通过梯度下降算法求得最优解，代码中就是运用了梯度下降算法。或者可以对函数求导，让导数为0，即可得到w和b的最优解。

code：

```python 
#!/usr/bin/env python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import theano
import theano.tensor as T
import pdb

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
diabetes_X_test = diabetes_X_test.reshape(-1, 1)
diabetes_y_test = diabetes_y_test.reshape(-1, 1)
diabetes_X_train = diabetes_X_train.reshape(-1, 1)
diabetes_y_train = diabetes_y_train.reshape(-1, 1)

x = T.dmatrix('x')
y = T.dmatrix('y')

W = theano.shared(value=np.zeros((1,1), dtype=theano.config.floatX), 
                    name='W', borrow=True)
b = theano.shared(value=np.zeros((1,), dtype=theano.config.floatX)+0.1, 
                    name='b', borrow=True)

p_y_given_x = T.dot(x,W) + b

cost = T.mean(T.sqr(p_y_given_x - y))

g_W = T.grad(cost=cost, wrt=W)
g_b = T.grad(cost=cost, wrt=b)

learning_rage = 0.1
updates = [(W, W - learning_rage * g_W),
           (b, b - learning_rage * g_b)]
# pdb.set_trace()
train_model = theano.function(inputs=[x, y], outputs=cost, updates=updates)

predict = theano.function(inputs=[x], outputs=p_y_given_x)

test_err = theano.function(inputs=[x, y], outputs=cost)

for i in range(10000):
    # training
    err = train_model(diabetes_X_train, diabetes_y_train)
    if i % 500 == 0:
        print(err)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Sklearn Coefficients: \n', regr.coef_)
print('Sklearn intercept_: \n', regr.intercept_)
print('Theano W:', W.get_value())
print('Theano b:', b.get_value())
# The mean squared error
print("Sklearn Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))

print("Theano Mean squared error: %.2f" 
      % test_err(diabetes_X_test, diabetes_y_test))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, predict(diabetes_X_test), color='blue',
         linewidth=3)
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='red',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
```

Output:

```
29437.9844071
5222.11922636
4760.51165798
4467.06878323
4280.52738095
4161.94427696
4086.56106406
4038.64008777
4008.17699554
3988.81176843
3976.5013185
3968.67560988
3963.70079372
3960.53830772
3958.52795951
3957.24994749
3956.43752353
3955.92106056
3955.59274762
3955.38405396
('Sklearn Coefficients: \n', array([[ 938.23786125]]))
('Sklearn intercept_: \n', array([ 152.91886183]))
('Theano W:', array([[ 928.12677002]], dtype=float32))
('Theano b:', array([ 152.92367554], dtype=float32))
Sklearn Mean squared error: 2548.07
Theano Mean squared error: 2559.79
Variance score: 0.47

```


![01.Linear_regression](\images\post-covers\01.Linear_regression.png)