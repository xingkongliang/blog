---
title: Caffe学习笔记02-Extracting Features
layout: post
tags: [Deep Learning]
---

在本教程中，我们将使用包含C++使用程序的预训练模型提取特征。

按照说明[安装Caffe](http://caffe.berkeleyvision.org/installation.html)并从caffe根目录运行python scripts/download_model_binary.py models/bvlc_reference_caffenet。 如果您需要有关以下工具的详细信息，请参阅其源代码，其中通常提供其他文档。

# 1.选择数据运行

我们将创建一个临时文件件存储一些文件。

```
mkdir examples/_temp
```

生成要处理的文件的列表。我们将使用caffe/examples/images下的图像。
```
find `pwd`/examples/images -type f -exec echo {} \; > examples/_temp/temp.txt
```

![temp.txt](\blog\images\post-covers\2017-03-16-caffe-notebook02-01.png)

我们使用的ImageDataLayer在每个文件名后面都要有标签，所以让我们在每行的末尾添加一个0。
```
sed "s/$/ 0/" examples/_temp/temp.txt > examples/_temp/file_list.txt
```
![file_list.txt](\blog\images\post-covers\2017-03-16-caffe-notebook02-02.png)

# 2.定义特征提取网络结构

实际上，从数据集中减去平均值图像显着提高了分类精度。 下载ILSVRC数据集的平均值。

```
./data/ilsvrc12/get_ilsvrc_aux.sh
```
我们在网络定义prototxt中将使用到`data/ilsvrc212/imagenet_mean.binaryproto`这个均值文件。


让我们复制和修改网络定义。我们将使用`ImageDataLayer`，它将为我们加载和调整图像大小。

```
cp examples/feature_extraction/imagenet_val.prototxt examples/_temp
```

![imagenet_val](\blog\images\post-covers\2017-03-16-caffe-notebook02-03.png)
至此，我们的_temp文件夹下将有以上三个文件。

# 3.特征提取

执行以下代码：
```
./build/tools/extract_features.bin models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel examples/_temp/imagenet_val.prototxt fc7 examples/_temp/features 10 leveldb
```
运行结果如下图所示：
![extract feature](\blog\images\post-covers\2017-03-16-caffe-notebook02-04.png)

我们提取的是fc7层的特征，这里fc7层表示我们参考模型的最高级别的特征。我们同样可以使用其他层，比如conv5
或pool3。

上面的最后一个参数是数据mini-batch的数量。

这些特征被存储到LevelDB examples/_temp/features中，准备通过一些代码访问。

如果遇到“Check failed: status.ok() Failed to open leveldb examples/_temp/features”这个错误，这是因为上次运行命令时创建了目录examples/_temp /features。删除它并再次运行即可。

```
rm -rf examples/_temp/features/
```
现在如果你想使用Python wrapper提取特征，请查看[filter visualization notebook](http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb)
# 4.清理
清理临时文件夹

```
rm -r examples/_temp
```

# 5.补充说明

以下是imagenet_val.prototxt文件的一部分内容，我们可以看到均值文件的路径定义和数据来源路径的定义。
```
name: "CaffeNet"
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  transform_param {
    mirror: false
    crop_size: 227
    mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
  }
  image_data_param {
    source: "examples/_temp/file_list.txt"
    batch_size: 50
    new_height: 256
    new_width: 256
  }
}
```
参考网址：[Caffe Extracting Features](http://caffe.berkeleyvision.org/gathered/examples/feature_extraction.html)