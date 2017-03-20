---
title: Caffe学习笔记01
layout: post
tags: [Deep Learning]
---

# 1.数据集

 CIFAR10数据集由60000幅32x32大小的彩色图像构成，一共有10个类别，其中每个类别有6000张图像。有50000张训练图像和10000张测试图像。

![CIFAR10数据集](\blog\images\post-covers\2017-03-16-caffe-notebook01-01.png)

# 2.准备工作

首先你需要从[CIFAR-10网站](http://www.cs.toronto.edu/~kriz/cifar.html)下载并且转化数据格式。为了实现这个，你可以简单地运行下面的指令：

（1）下载数据集
```
cd $CAFFE_ROOT/data/cifar10
./get_cifar10.sh
```
**CAFFE_ROOT是Caffe的根目录**

![下载数据集](\blog\images\post-covers\2017-03-16-caffe-notebook01-02.png)

（2）转换数据格式

```
cd $CAFFE_ROOT
./examples/cifar10/create_cifar10.sh
```

![转化数据格式](\blog\images\post-covers\2017-03-16-caffe-notebook01-03.png)

到此为止，我们已经下好了数据集保存在/CAFFE_ROOT/data/cifar10文件夹下，并且数据集转换格式后的文件和数据集的均值在/CAFFE_ROOT/examples/cifar10文件夹下。


# 3.模型

模型文件是存在在/CAFFE_ROOT/examples/cifar10文件夹下的prototxt文件中，我们以cifar10_quick_train_test.prototxt文件为例。

```
name: "CIFAR10_quick"
layer {
  name: "cifar"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mean_file: "examples/cifar10/mean.binaryproto"
  }
  data_param {
    source: "examples/cifar10/cifar10_train_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "cifar"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mean_file: "examples/cifar10/mean.binaryproto"
  }
  data_param {
    source: "examples/cifar10/cifar10_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 32
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.0001
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "pool1"
  top: "pool1"
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 32
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: AVE
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 64
    pad: 2
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "pool3"
  type: "Pooling"
  bottom: "conv3"
  top: "pool3"
  pooling_param {
    pool: AVE
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool3"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 64
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}

```
以上内容就是我们的网络结构，我们可以使用可视化的方式观看，通过可视化工具[**点击这里**](http://ethereon.github.io/netscope/#/editor)，我们可以观察我们定义的网络。可以看出这个简单的网络由数据输入层、卷积层、池化层、RELU层和全连接层构成。所以说如果我们想更改网络结构只需修改prototxt文件即可。

![网络可视化](\blog\images\post-covers\2017-03-16-caffe-notebook01-04.png)


# 4. 训练和测试

在我们写好网络定义protobuf文件和参数设置文件solver之后（实际上是示例代码已经给咱们写好了）。简单的运行`train_quick.sh`即可。

```
cd $CAFFE_ROOT
./examples/cifar10/train_quick.sh
```

![训练](\blog\images\post-covers\2017-03-16-caffe-notebook01-05.png)

train_quick.sh是一个简单的脚本，会把执行信息显示出来。

分别会显示

（1）我们在solver中设置的参数信息
```
I0316 03:04:56.309629 31444 solver.cpp:48] Initializing solver from parameters:
test_iter: 100
test_interval: 500
base_lr: 0.001
display: 100
max_iter: 4000
lr_policy: "fixed"
momentum: 0.9
weight_decay: 0.004
snapshot: 4000
snapshot_prefix: "examples/cifar10/cifar10_quick"
solver_mode: GPU
device_id: 0
net: "examples/cifar10/cifar10_quick_train_test.prototxt"
```
（2）从prototxt文件中定义的网络结构
```
name: "CIFAR10_quick"
state {
  phase: TRAIN
}
layer {
  name: "cifar"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mean_file: "examples/cifar10/mean.binaryproto"
  }
  data_param {
    source: "examples/cifar10/cifar10_train_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"

```
（3）创建网络的层与层之间连接顺序，及所需要的内存，和哪些层需要反传计算。
```
I0316 03:04:56.311331 31444 layer_factory.hpp:77] Creating layer cifar
I0316 03:04:56.311914 31444 net.cpp:91] Creating Layer cifar
I0316 03:04:56.311936 31444 net.cpp:399] cifar -> data
I0316 03:04:56.311974 31444 net.cpp:399] cifar -> label
I0316 03:04:56.311990 31444 data_transformer.cpp:25] Loading mean file from: examples/cifar10/mean.binaryproto
I0316 03:04:56.313457 31446 db_lmdb.cpp:35] Opened lmdb examples/cifar10/cifar10_train_lmdb
I0316 03:04:56.344117 31444 data_layer.cpp:41] output data size: 100,3,32,32
I0316 03:04:56.348085 31444 net.cpp:141] Setting up cifar
I0316 03:04:56.348131 31444 net.cpp:148] Top shape: 100 3 32 32 (307200)
I0316 03:04:56.348142 31444 net.cpp:148] Top shape: 100 (100)
I0316 03:04:56.348150 31444 net.cpp:156] Memory required for data: 1229200
I0316 03:04:56.348161 31444 layer_factory.hpp:77] Creating layer conv1
I0316 03:04:56.348196 31444 net.cpp:91] Creating Layer conv1
I0316 03:04:56.348206 31444 net.cpp:425] conv1 <- data
I0316 03:04:56.348223 31444 net.cpp:399] conv1 -> conv1
I0316 03:04:56.758509 31444 net.cpp:141] Setting up conv1
I0316 03:04:56.758564 31444 net.cpp:148] Top shape: 100 32 32 32 (3276800)
I0316 03:04:56.758575 31444 net.cpp:156] Memory required for data: 14336400
I0316 03:04:56.758605 31444 layer_factory.hpp:77] Creating layer pool1
I0316 03:04:56.758627 31444 net.cpp:91] Creating Layer pool1
...


I0316 03:04:56.773228 31444 net.cpp:156] Memory required for data: 31978800 <--请看好，这里是需要的内存
I0316 03:04:56.773244 31444 layer_factory.hpp:77] Creating layer loss
I0316 03:04:56.773257 31444 net.cpp:91] Creating Layer loss
I0316 03:04:56.773267 31444 net.cpp:425] loss <- ip2
I0316 03:04:56.773275 31444 net.cpp:425] loss <- label
I0316 03:04:56.773288 31444 net.cpp:399] loss -> loss
I0316 03:04:56.773317 31444 layer_factory.hpp:77] Creating layer loss
I0316 03:04:56.773861 31444 net.cpp:141] Setting up loss
I0316 03:04:56.773887 31444 net.cpp:148] Top shape: (1)
I0316 03:04:56.773896 31444 net.cpp:151]     with loss weight 1
I0316 03:04:56.773921 31444 net.cpp:156] Memory required for data: 31978804

<---------------请看好，这里是需要反传计算的层------------------->
I0316 03:04:56.773931 31444 net.cpp:217] loss needs backward computation.
I0316 03:04:56.773941 31444 net.cpp:217] ip2 needs backward computation.
I0316 03:04:56.773948 31444 net.cpp:217] ip1 needs backward computation.
I0316 03:04:56.773957 31444 net.cpp:217] pool3 needs backward computation.
I0316 03:04:56.773964 31444 net.cpp:217] relu3 needs backward computation.
I0316 03:04:56.773973 31444 net.cpp:217] conv3 needs backward computation.
I0316 03:04:56.773983 31444 net.cpp:217] pool2 needs backward computation.
I0316 03:04:56.773991 31444 net.cpp:217] relu2 needs backward computation.
I0316 03:04:56.773999 31444 net.cpp:217] conv2 needs backward computation.
I0316 03:04:56.774008 31444 net.cpp:217] relu1 needs backward computation.
I0316 03:04:56.774016 31444 net.cpp:217] pool1 needs backward computation.
I0316 03:04:56.774024 31444 net.cpp:217] conv1 needs backward computation.
I0316 03:04:56.774034 31444 net.cpp:219] cifar does not need backward computation.
I0316 03:04:56.774044 31444 net.cpp:261] This network produces output loss
I0316 03:04:56.774061 31444 net.cpp:274] Network initialization done.

...

```
（4）最后会输出优化过程结果

```
)
I0316 03:07:11.809412 31705 sgd_solver.cpp:106] Iteration 4700, lr = 0.0001
I0316 03:07:14.348947 31705 solver.cpp:228] Iteration 4800, loss = 0.427255
I0316 03:07:14.349012 31705 solver.cpp:244]     Train net output #0: loss = 0.427255 (* 1 = 0.427255 loss)
I0316 03:07:14.349023 31705 sgd_solver.cpp:106] Iteration 4800, lr = 0.0001
I0316 03:07:17.086114 31705 solver.cpp:228] Iteration 4900, loss = 0.469861
I0316 03:07:17.086187 31705 solver.cpp:244]     Train net output #0: loss = 0.469861 (* 1 = 0.469861 loss)
I0316 03:07:17.086199 31705 sgd_solver.cpp:106] Iteration 4900, lr = 0.0001
I0316 03:07:20.278740 31705 solver.cpp:464] Snapshotting to HDF5 file examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5
I0316 03:07:20.289602 31705 sgd_solver.cpp:283] Snapshotting solver state to HDF5 file examples/cifar10/cifar10_quick_iter_5000.solverstate.h5
I0316 03:07:20.296205 31705 solver.cpp:317] Iteration 5000, loss = 0.593591
I0316 03:07:20.296250 31705 solver.cpp:337] Iteration 5000, Testing net (#0)
I0316 03:07:21.401197 31705 solver.cpp:404]     Test net output #0: accuracy = 0.7573
I0316 03:07:21.401386 31705 solver.cpp:404]     Test net output #1: loss = 0.751382 (* 1 = 0.751382 loss)
I0316 03:07:21.401397 31705 solver.cpp:322] Optimization Done.
I0316 03:07:21.401404 31705 caffe.cpp:222] Optimization Done.

```
其中╮(╯_╰)╭每100次迭代次数显示一次训练时lr（learning rate）和loss（训练损失函数），每500次测试一次，输出accuracy（准确率）和loss（测试损失函数）

当5000次迭代之后，正确率约是0.75，该模型的参数存储在二进制protobuf格式在cifar10_quick_iter_5000

然后，这个模型就可以用来运行在新数上了。训练好的模型如下图所示。

![训练好的模型](\blog\images\post-covers\2017-03-16-caffe-notebook01-06.png)

# 5.GUP or CPU
可以通过在cifar10*solver.prototxt文件可以使用选择使用CPU或是GPU训练模型。
```
# solver mode: CPU or GPU
solver_mode: CPU
```

### 附带train_quick.sh文件和cifar10_quick_solver.prototxt文件内容。
train_quick.sh
```
#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt

# reduce learning rate by factor of 10 after 8 epochs
$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate.h5

```

cifar10_quick_solver.prototxt
```
# reduce the learning rate after 8 epochs (4000 iters) by a factor of 10

# The train/test net protocol buffer definition
net: "examples/cifar10/cifar10_quick_train_test.prototxt"
# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
test_iter: 100
# Carry out testing every 500 training iterations.
test_interval: 500
# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.001
momentum: 0.9
weight_decay: 0.004
# The learning rate policy
lr_policy: "fixed"
# Display every 100 iterations
display: 100
# The maximum number of iterations
max_iter: 4000
# snapshot intermediate results
snapshot: 4000
snapshot_format: HDF5
snapshot_prefix: "examples/cifar10/cifar10_quick"
# solver mode: CPU or GPU
solver_mode: GPU

```