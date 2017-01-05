---
title: Cross Entropy
layout: post
tags: [Machine Learning]
---


# 1.交叉熵（Cross entropy）


$$H(p,q)=E_p[-\mathop{log} q]=E(p)+D_{KL}(p||q)$$


这里 $$H(p)$$ 是p的熵，$$D_{KL}(p\|\|q)$$是KL散度（[Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)）。


特别的，在信息论中，D(P\|\|Q)表示当用概率分布Q来拟合真实分布P时，产生的信息损耗，其中P表示真实分布，Q表示P的拟合分布。

KL散度定义：

$$D_{KL}(P\|Q)=\sum \limits_i P(i) \mathop{log}\frac{P(i)}{Q(i)}$$

交叉熵定义：

$$H(p,q)=-\sum_{x}p(x)\,\log q(x)$$


推导：

$$H(p,q)=\operatorname {E}_{p}[l_{i}]=\operatorname {E}_{p}\left[\log {\frac  {1}{q(x_{i})}}\right]$$

$$H(p,q)=\sum_{x_i}p(x_{i})\log {\frac{1}{q(x_{i})}}$$

$$H(p,q)=-\sum_{x}p(x)\,\log q(x)$$


![KL](https://imgsa.baidu.com/baike/c0%3Dbaike150%2C5%2C5%2C150%2C50/sign=be90023ebb014a9095334eefc81e5277/562c11dfa9ec8a13c3a0ea6cf503918fa1ecc080.jpg)

# 2.二次代价函数

二次代价函数的形式如下：

$$C=\frac{1}{2n}\sum \limits_x\|y(x)-a^L(x)\|^2$$

其中，C代表代价，x代表样本，y表示实际值，a表示输出值，n表示样本的总数。为了简单说明，以一个样本为例：

$$C=\frac{(y-1)^2}{2}$$

其中$$a=\sigma(z)$$是激活函数，$$z=wx+b$$。对于w和b的梯度推导如下：

$$\frac{\partial C}{\partial w}=(a-y)\sigma'(z)x$$

$$\frac{\partial C}{\partial b}=(a-y)\sigma'(z)$$

从上式可以看出，w和b的梯度和激活函数的梯度成正比，激活函数的梯度越大，w和b的大小调整的就越快，训练收敛的就越快。而神经网络常用的激活函数为sigmoid函数，该激活函数的曲线图下图所示：
![sigmoid](http://img.blog.csdn.net/20160402165516510)

这明显有悖于我们的初衷，这使得z的值比较大或者比较小的时候，w和b的更新速度回非常的慢。

# 3.交叉熵代价函数

由于sigmoid激活函数有许多优点，所以我们打算把二次代价函数换成交叉熵代价函数：

$$C=-\frac{1}{n}\sum \limits_x [ylna+(1-y)ln(1-a)]$$

其中，x表示样本，n表示样本的总数。那么，重新计算参数w的梯度：

$$\frac{\partial C}{\partial w_j}=-\frac{1}{n}\sum \limits_x(\frac{y}{\sigma(z)}-\frac{(1-y)}{1-\sigma(z)})\frac{\partial \sigma}{\partial w_j}\\
=-\frac{1}{n}\sum \limits_x(\frac{y}{\sigma(z)}-\frac{(1-y)}{1-\sigma(z)})\sigma'(z)x_j\\
=\frac{1}{n}\sum \limits_x \frac{\sigma'(z) x_j}{\sigma(z)(1-\sigma(z))}(\sigma(z)-y)\\
=\frac{1}{n}\sum \limits_x x_j(\sigma(z)-y)$$

w的梯度公式中原来的$$\sigma'(z)$$被消掉了；另外该梯度公式中的$$\sigma(z)-y$$表示输出值与真实值之间的误差。因此，当误差越大，梯度就越大，参数w调整的越快。同理可得，b的梯度为：

$$\frac{\partial C}{\partial b}=\frac{1}{n}\sum \limits_x(\sigma(z)-y)$$

# 3.log-likelihood cost

对数似然函数也常用来作为softmax回归的代价函数。如果最后一层（也就是输出）是通过sigmoid函数，就采用交叉熵代价函数。

而再深度学习中更普遍的做法是将softmax作为最后一层，此时常用的代价函数式log-likelihood cost。

> In fact, it’s useful to think of a softmax output layer with log-likelihood cost as being quite similar to a sigmoid output layer with cross-entropy cost。

其实这两者是一致的，logistic回归用的就是sigmoid函数，softmax回归是logistic回归的多类别推广。log-likelihood代价函数在二类别时就可以化简为交叉熵代价函数的形式。具体可以参考[UFLDL教程](http://deeplearning.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92)

### 以上内容来自于以下几个网站：

[维基百科](https://en.wikipedia.org/wiki/Cross_entropy)

[博客1](http://blog.csdn.net/u012162613/article/details/44239919)

[博客2](http://blog.csdn.net/u014313009/article/details/51043064)

[知乎1](https://www.zhihu.com/question/41252833)