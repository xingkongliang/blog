---
title: 生成式对抗网络
layout: post
tags: [Deep Learning]
---


## DCGANs in TensorFlow

[carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
我们定义网络结构：

```python

def generator(self, z):
    self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4,
                                           'g_h0_lin', with_w=True)

    self.h0 = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
    h0 = tf.nn.relu(self.g_bn0(self.h0))

    self.h1, self.h1_w, self.h1_b = conv2d_transpose(h0,
        [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1', with_w=True)
    h1 = tf.nn.relu(self.g_bn1(self.h1))

    h2, self.h2_w, self.h2_b = conv2d_transpose(h1,
        [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2', with_w=True)
    h2 = tf.nn.relu(self.g_bn2(h2))

    h3, self.h3_w, self.h3_b = conv2d_transpose(h2,
        [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3', with_w=True)
    h3 = tf.nn.relu(self.g_bn3(h3))

    h4, self.h4_w, self.h4_b = conv2d_transpose(h3,
        [self.batch_size, 64, 64, 3], name='g_h4', with_w=True)

    return tf.nn.tanh(h4)

def discriminator(self, image, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
    h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
    h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
    h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
    h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h3_lin')

    return tf.nn.sigmoid(h4), h4

```
当我们初始化这个类时，我们将使用这些函数来创建模型。 我们需要两个版本的鉴别器共享（或重用）参数。 一个用于来自数据分布的图像的minibatch，另一个用于来自发生器的图像的minibatch。

```python
self.G = self.generator(self.z)
self.D, self.D_logits = self.discriminator(self.images)
self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

```
接下来我们定义损失函数。我们在D的预测值和我们理想的判别器输出值之间使用[交叉熵](https://en.wikipedia.org/wiki/Cross_entropy)，而没有只用求和，因为这样的效果更好。判别器希望对“真实”数据的预测全部是1，并且来自生成器的“假”数据的预测全部是零。生成器希望判别器对所有假样本的预测都是1。

```python
self.d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits,
                                            tf.ones_like(self.D)))
self.d_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                            tf.zeros_like(self.D_)))
self.d_loss = self.d_loss_real + self.d_loss_fake

self.g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                            tf.ones_like(self.D_)))

```
收集每个模型的变量，以便可以单独进行训练。

```python
t_vars = tf.trainable_variables()

self.d_vars = [var for var in t_vars if 'd_' in var.name]
self.g_vars = [var for var in t_vars if 'g_' in var.name]

```
现在我们准备好优化参数，我们将使用[ADAM](https://arxiv.org/abs/1412.6980)，这是一种在现代深度学习中常见的自适应非凸优化方法。ADAM通常与SGD竞争，并且（通常）不需要手动调节学习速率，动量和其他超参数。

```python
d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(self.d_loss, var_list=self.d_vars)
g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(self.g_loss, var_list=self.g_vars)
```

我们已经准备好了解我们的数据。在每个epoch中，我们在每个minibatch中采样一些图像，并且运行优化器更新网络。有趣的是，如果G仅更新一次，判别器的损失则不会为零。另外，我认为`d_loss_fake`和`d_loss_real`在最后的额外的调用回到是一点点不必要的计算，并且是冗余的，因为这些值是作为`d_optim`和`g_optim`的一部分计算的。作为TensorFlow中的练习，您可以尝试优化此部分并将RP发送到原始库。


```python
for epoch in xrange(config.epoch):
    ...
    for idx in xrange(0, batch_idxs):
        batch_images = ...
        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ self.images: batch_images, self.z: batch_z })

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })

        # Run g_optim twice to make sure that d_loss does not go to zero
        # (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })

        errD_fake = self.d_loss_fake.eval({self.z: batch_z})
        errD_real = self.d_loss_real.eval({self.images: batch_images})
        errG = self.g_loss.eval({self.z: batch_z})
```



### Generative Adversarial Networks代码整理

- [**InfoGAN-TensorFlow**](https://github.com/openai/InfoGAN):InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets

- [**iGAN-Theano**](https://github.com/junyanz/iGAN):Generative Visual Manipulation on the Natural Image Manifold

- [**SeqGAN-TensorFlow**](https://github.com/LantaoYu/SeqGAN):SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient

- [**DCGAN-Tensorflow**](https://github.com/carpedm20/DCGAN-tensorflow):Deep Convolutional Generative Adversarial Networks 

- [**dcgan_code-Theano**](https://github.com/Newmu/dcgan_code):Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

- [**improved-gan-Theano**](https://github.com/openai/improved-gan):Improved Techniques for Training GANs

- [**chainer-DCGAN**](https://github.com/mattya/chainer-DCGAN):Chainer implementation of Deep Convolutional Generative Adversarial Network

- [**keras-dcgan**](https://github.com/jacobgil/keras-dcgan)


