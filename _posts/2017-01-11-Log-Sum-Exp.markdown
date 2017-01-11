---
title: Log Sum Exp
layout: post
tags: [Deep Learning, Machine Learning]
---

## 定义

LogSumExp（LSE）函数是maximum函数的一个平滑近似，主要被机器学习算法所使用。它被定义为参数的指数和的对数形式：

$$LSE(x_{1},\dots ,x_{n})=\log \left(\exp(x_{1})+\cdots +\exp(x_{n})\right)$$

or：

$$z=\log \sum^N_{n=1} \exp \{x_n\}.$$


$${\displaystyle \max {\{x_{1},\dots ,x_{n}\}}\leq LSE(x_{1},\dots ,x_{n})\leq \max {\{x_{1},\dots ,x_{n}\}}+\log(n)}$$

当只有一个参数是非零时满足下限，而当所有参数相等时满足上限。

## Log-Sum-Exp的一个计算技巧：

如果输出向量是[0 1 1]，我们直接得到1.55。现在我们考虑[1000 1001 1000]，我们就会得到inf。如果是[-1000 -999 -1000]，我们就会得到-inf。在我们的64-bit double类型里，exp{1000}=inf并且exp{-1000}=0。我们通过下式来解决上溢和下溢问题。

$$\log \sum^N_{n=1} \exp \{x_n\} = a + \log \sum^N_{n=1} \exp \{x_n-a\}$$

$$a=\max_n x_n$$