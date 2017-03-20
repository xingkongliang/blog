---
title: Caffe学习笔记03-Interfaces
layout: post
tags: [Deep Learning]
---

# Interfaces

Caffe有命令行、Python和MATLAB接口。

# Command Line

命令行界面-cmdcaffe-是用于模型训练、评分和诊断的caffe工具。运行caffe没有任何参数的帮助。这个工具和其他的工具可以在`caffe/build/tools`中找到。（以下示例调用需要先完成LeNet/MNIST示例。）

**训练：**`caffe train`从头开始学习模型、从已保存的快照恢复学习、以及将模型微调到新的数据和任务：

- 所有的训练都需要通过一个solver（求解）配置，通过`-solver solver.prototxt'参数。
- 恢复需要使用`-snapshot model_iter_1000.solverstate`参数去加载solver快照。
- Fine-tuning（微调）需要模型初始化的`-weights model.caffemodel`参数。

例如，你可以运行：

```
# train LeNet
caffe train -solver examples/mnist/lenet_solver.prototxt
# train on GPU 2
caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 2
# resume training from the half-way point snapshot
caffe train -solver examples/mnist/lenet_solver.prototxt -snapshot examples/mnist/lenet_iter_5000.solverstate
```

有关微调的完整示例，请参阅 examples/finetuning_on_flickr_style，但是单独的训练调用是

```
# fine-tune CaffeNet model weights for style recognition
caffe train -solver examples/finetuning_on_flickr_style/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
```

# MATLAB

## Use MatCaffe

### 创建网络并访问其图层和Blob
创建一个网络：

```
net = caffe.Net(model, weights, 'test'); % create net and load weights
```

或者
```
net = caffe.Net(model, 'test'); % create net but not load weights
net.copy_from(weights); % load weights
```
这里创建的这个网络对象：
```
  Net with properties:

           layer_vec: [1x23 caffe.Layer]
            blob_vec: [1x15 caffe.Blob]
              inputs: {'data'}
             outputs: {'prob'}
    name2layer_index: [23x1 containers.Map]
     name2blob_index: [15x1 containers.Map]
         layer_names: {23x1 cell}
          blob_names: {15x1 cell}
```

两个contrainers.Map对象通过它的名称来查找图层或blob的索引。

您可以访问网络中的每个blob。用ones填充blob的'data'：
```
net.blobs('data').set_data(ones(net.blobs('data').shape));
```

要将blob的'data'中所有值乘以10:
```
net.blobs('data').set_data(net.blobs('data').get_data() * 10);
```

请注意，有Matlab是1索引和列主，Matlab中通常的4个blob维度是[width, height, channels, num]，width是最快的维度。还要注意，图像在BGR通道中。此外，Caffe使用单精度浮点数据（single-precision）。如果您的数据不是single，set_data会自动将其装换为single。

你也可以访问每一层，所以你可以给网络做操作。例如，要将conv1参数乘以10：
```
net.params('conv1', 1).set_data(net.params('conv1', 1).get_data() * 10); % set weights
net.params('conv1', 2).set_data(net.params('conv1', 2).get_data() * 10); % set bias
```
或者，您可以使用
```
net.layers('conv1').params(1).set_data(net.layers('conv1').params(1).get_data() * 10);
net.layers('conv1').params(2).set_data(net.layers('conv1').params(2).get_data() * 10);
```

要保存刚刚修改的网络：
```
net.save('my_net.caffemodel');
```

要获取层的类型（字符串）：
```
layer_type = net.layers('conv1').type;
```

### Forward and backward

正向传递可以使用`net.forward`或者`net.forward_prefilled`。函数`net.forward`接收包含输入blob的数据的N-D数组单元阵列，并输出包含来自输出blob的数据单元阵列。函数`net.forward_prefilled`在正向传递期间使用blob中的现有数据，不需要输入，也不产生输出。在为输入blob创建一些数据后，想`data=rand(net.blobs('data').shape);`你可以运行：

```
res = net.forward({data});
prob = res{1};
```
或着

```
net.blobs('data').set_data(data);
net.forward_prefilled();
prob = net.blobs('prob').get_data();
```

后向传播使用`net.backward`或者`net.backward_prefilled`,并将get_data和set_data替换为get_diff和set_diff。在为输出blob创建一些梯度之后，像`prob_diff=rand(net.blobs('prob').shape);`你可以运行：

```
res = net.backward({prob_diff});
data_diff = res{1};
```
或者
```
net.blobs('prob').set_diff(prob_diff);
net.backward_prefilled();
data_diff = net.blobs('data').get_diff();
```
然而，上述反向计算没有得到正确的结果，因为Caffe决定网络不需要反向计算。要获得正确的反向结果，您需要在网络原型中（network prototxt）设置"force_backward:true"。

执行前向和后向传递后，您还可以获取内部blob中的数据和差异。例如，在正向传递之后提取pool5特征:

```
pool5_feat = net.blobs('pool5').get_data();
```

### Reshape

假设您想要一次运行1幅图像，而不是10个：

```
net.blobs('data').reshape([227 227 3 1]); % reshape blob 'data'
net.reshape();
```
然后整个网络被重塑，现在net.blobs('prob').shape应该是[1000 1];

### Training

假设您已经在我们的ImageNet教程学习之后创建好了训练和验证lmdbs，在ILSVRC 2012分类数据集上创建求解器（solver）和训练：

```
solver = caffe.Solver('./models/bvlc_reference_caffenet/solver.prototxt');
```
它创建solver对象:
```
Solver with properties:

          net: [1x1 caffe.Net]
    test_nets: [1x1 caffe.Net]
```

训练：
```
solver.solver();
```
或者训练只有1000次迭代（这样你可以在训练更多的迭代之前做一些事情）：

```
solver.step(1000);
```
要获取迭代数：
```
iter = solver.iter();
```

获取其网络:
```
train_net = solver.net;
test_net = solver.test_nets(1);
```

要从快照"your_snapshot.solverstate"恢复：

```
solver.restore('your_snapshot.solverstate');
```

### Input and output

`caffe.io`类提供了基本的输入函数`load_image`和`read_mean`。例如，要读取ILSVRC 2012均值文件（假设您已经运行./data/ilsvrc12/get_ilsvrc_aux.sh下载了imagenet示例辅助文件）：

```
mean_data = caffe.io.read_mean('./data/ilsvrc12/imagenet_mean.binaryproto');    
```

要读取Caffe的示例图像并调整为[width, height]，并假设我们想要width=256;height=256;
```
im_data = caffe.io.load_image('./examples/images/cat.jpg');
im_data = imresize(im_data, [width, height]); % resize using Matlab's imresize
```

请记住，width是最快的维度，通道是BGR，这与Matlab存储图像的通常方式不同。如果你不想使用`caffe.io.load_image`并且喜欢自己加载图像，您可以这样做


```
im_data = imread('./examples/images/cat.jpg'); % read image
im_data = im_data(:, :, [3, 2, 1]); % convert from RGB to BGR
im_data = permute(im_data, [2, 1, 3]); % permute width and height
im_data = single(im_data); % convert to single precision
```

另外，您可以看一下caffe/matlab/demo/classification_demo.m，看看如何通过从一幅图像中take crops来准备输入。

我们在caffe/matlab/hdf5creation中展示如何使用Matlab读写HDF5数据。我们不提供额外的数据输出功能，因为Matlab本身在输出功能上已经很强大了。

### Clear nets and solvers

调用`caffe.reset_all()`清除所有已经创建的求解器和独立网络。




