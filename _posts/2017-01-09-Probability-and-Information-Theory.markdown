---
title: 概率论与信息论
layout: post
tags: [Deep Learning, Machine Learning]
---

## 3.3 概率分布

### 3.3.1 离散型变量和概率分布律函数

离散型变量的概率分布可以用**概率分布律函数**（probability mass function,PMF）来描述。

### 3.3.2 连续型变量和概率密度函数

连续性随机变量的分布可以用**概率密度函数**（probability desity function,PDF）来描述。

给出一个连续型随机变量的PDF的例子，考虑实数区间上的均匀分布。我们可以通过函数$$u(x;a,b)$$来实现，其中a和b是区间的端点满足a>b。**符号“;”表示“以什么为参数”**；我们把x作为函数的自变量，a和b作为定义函数的参数。


## 3.4 边缘概率

有时候，我们知道了一组变量的联合概率分布，想要了解其中一个自己的概率分布。这种定义在子集上的概率分布被称为**边缘概率分布**（marginal probability distribution）。

例如，假设有离散型随机变量x和y，并且我们知道$$P(x,y)$$。我们可以依据下面的**求和法则**（sum rule）来计算$$P(x)$$:

$$\forall x\in \mathrm{x}, P(\mathrm{x}=x)=\sum_y P(\mathrm{x}=x,\mathrm{y}=y)$$

对于连续型变量，我们需要用积分代替去和：

$$p(x)=\int p(x,y)dy.$$


## 3.5 条件概率

很多情况下，我们感兴趣的是某个事件，在给定其他事件发生时，出现的概率。这种这种概率我们叫做条件概率。我们将给定$$\mathrm{x}=x$$的$$\mathrm{y}=y$$发生的条件概率即为$$P(\mathrm{y}=y \vert \mathrm{x}=x)$$。这个条件概率可以通过下面的公式计算：

$$P(\mathrm{y}=y|\mathrm{x}=x)=\frac{P(\mathrm{y}=y,\mathrm{x}=x)}{P(\mathrm{x}=x)}$$


## 3.6 条件概率的链式法则

任何多维随机变量的联合概率分布，都可以分解成只有一个变量的条件概率相乘的形式：

$$P(\mathrm{x}^{(1)},...,\mathrm{x}^{(n)})=P(\mathrm{x}^{(1)} \prod^2_{i=2} P(\mathrm{x}^{(i)}\vert \mathrm{x}^{(1)},...\mathrm{x}^{(i-1)})$$

这个规则被称为概率的**链式法则**（chain rule）或者**乘法法则**（product rule）。它可以直接从公式3.5条件概率定义中得到。


## 3.7 独立性和条件独立性

相互独立的（independent）：

$$\forall x \in \mathrm{x},y\in \mathrm{y},p(\mathrm{x}=x,\mathrm{y}=y)=p(\mathrm{x}=x)p(\mathrm{y}=y).$$

条件独立的（conditionally independ）：

$$\forall x \in \mathrm{x},y\in \mathrm{y}, z\in \mathrm{z},p(\mathrm{x}=x,\mathrm{y}=y\vert \mathrm{z}=z)=p(\mathrm{x}=x\vert \mathrm{z}=z)p(\mathrm{y}=y\vert \mathrm{z}=z).$$



## 3.13 信息论

信息量应该满足基本的一下几条：

- 非常可能发生的事件信息量要比较少，并且极端情况下，确保能够发生的事件应该没有信息量。
- 更不可能发生的事件要具有更高的信息量。
- 独立事件应具有增量的信息。例如，投掷的硬币两次正面朝上传递的信息量，应该是投掷一次正面朝上的信息量的两倍。

为了满足 以上三个性质，我们定义一个事件$$\mathrm{x}=x$$的**自信息**（self-information）为

$$I(x)=-\log P(x).$$

我们用log表示自然对数，底数为e。因此我们定义的$$I(x)$$单位是**奈特（nats）**。一奈特是以$$\frac{1}{e}$$的概率观测到一个事件时获得的信息量。

自信息只处理单个的输出。我们可以用**香农熵**（Shannon entropy）来对整个概率分布中的不确定性总量进行量化：

$$H(x)=\mathbb{E}_{x\sim P}[I(x)]=-\mathbb{E}_{x\sim P}[\log P(x)],$$

也记做$$H(P)$$。换言之，一个分布的香农熵是指遵循这个分布事件所产生的期望信息总量。


如果我们对于同一个随机变量x有两个单独的概率分布$$P(\mathrm{x})$$和$$Q(\mathrm{x})$$，我们可以使用**KL散度**（Kullback-Leibler （KL） divergence）来衡量这两个分布的差异：

$$D_{KL}(P\Vert Q)=\mathbb E_{x\sim P}\left[ \log \frac{P(x)}{Q(x)} \right]=\mathbb E_{x\sim P}\left[\log P(x)-\log Q(x)\right].$$

在离散变量的情况下，KL散度衡量的是，当我们使用一种被设计成能够使得代理分布Q产生的消息的长度最小的编码时，发送包含有管理分布P产生的符号消息时，所需的额外信息量（如果我们使用底数为2的对数时信息量用比特衡量，但在机器学习中，我们通常用奈特和自然对数。）

KL散度具有非负性，但是它不是对称的。

一个和KL散度密切联系的量是**交叉熵**（cross-entropy）$$H(P,Q)=H(P)+D_{KL}(P\Vert Q)$$，它和KL散度很像但是缺少左边一项：

$$H(P,Q)=-\mathbb E_{x\sim P}\log Q(x).$$

针对Q最小化互信息等价于最小化KL散度，因为Q并不参与被省略的那一项。

