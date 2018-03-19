---
title: Caffe2 教程
layout: post
tags: [Deep Learning]
---

# Caffe2 教程

本教程来自Caffe2官网

Caffe2 官网：https://caffe2.ai/

Caffe2 github： https://github.com/caffe2/caffe2

---

翻译与整理：张天亮

邮箱：tianliangjay@gmail.com

Blog：https://xingkongliang.github.io/blog/

---

# 1. Caffe2 常用函数
```python
import numpy as np
import time

# These are the droids you are looking for.
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
```
## Workspaces
```python
print("Current blobs in the workspace: {}".format(workspace.Blobs()))
print("Workspace has blob 'X'? {}".format(workspace.HasBlob("X")))#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ : "ZhangTianliang"
# Date: 18-3-15

from __future__ import absolute_import
from __future__ import division
```

我们可以使用`FeedBlob()`把blobs送入workspace。
```python
X = np.random.randn(2, 3).astype(np.float32)
print("Generated X from numpy:\n{}".format(X))
workspace.FeedBlob("X", X)
```
使用`FetchBlobs()`获取Blob。
```python
print("Current blobs in the workspace: {}".format(workspace.Blobs()))
print("Workspace has blob 'X'? {}".format(workspace.HasBlob("X")))
print("Fetched X:\n{}".format(workspace.FetchBlob("X")))
```
打印当前workspace空间名称。当存在多个workspace时，使用`SwitchWorkspace()`转换当前空间。
```python
print("Current workspace: {}".format(workspace.CurrentWorkspace()))
print("Current blobs in the workspace: {}".format(workspace.Blobs()))

# Switch the workspace. The second argument "True" means creating 
# the workspace if it is missing.
workspace.SwitchWorkspace("gutentag", True)

# Let's print the current workspace. Note that there is nothing in the
# workspace yet.
print("Current workspace: {}".format(workspace.CurrentWorkspace()))
print("Current blobs in the workspace: {}".format(workspace.Blobs()))
```

使用`ResetWorkspace()`清理当前空间。
```python
workspace.ResetWorkspace()
```

## Operators

在Caffe2中Operator类似于函数。

创建一个Operator。
```python
# Create an operator.
op = core.CreateOperator(
    "Relu", # The type of operator that we want to run
    ["X"], # A list of input blobs by their names
    ["Y"], # A list of output blobs by their names
)
# and we are done!
```
打印当前operator。
```python
print("Type of the created op is: {}".format(type(op)))
print("Content:\n")
print(str(op))
```
运行这个operator。我们首先把X送入workspace。然后以最简单的方式运行operator，`workspace.RunOperaotrOnce(operator)`。
```python
workspace.FeedBlob("X", np.random.randn(2, 3).astype(np.float32))
workspace.RunOperatorOnce(op)
```

如果需要，Operators也可以选择参数。
```python
op = core.CreateOperator(
    "GaussianFill",
    [], # GaussianFill does not need any parameters.
    ["Z"],
    shape=[100, 100], # shape argument as a list of ints.
    mean=1.0,  # mean as a single float
    std=1.0, # std as a single float
)
print("Content of op:\n")
print(str(op))
```
运行这个operator，并且可视化结果。
```python
workspace.RunOperatorOnce(op)
temp = workspace.FetchBlob("Z")
pyplot.hist(temp.flatten(), bins=50)
pyplot.title("Distribution of Z")
```
![Alt text](\blog\images\post-covers\2018-03-19-1521102136044.png)

## Nets

网络本质上是计算图。一个网络由多个operators组成，就像一个由一些列命令组成的程序一些样。

Caffe2的`core`是一个围绕NetDef protocol buffer(协议缓冲区)的包装类。

```python
net = core.Net("my_first_net")
print("Current network proto:\n\n{}".format(net.Proto()))
```
创建一个叫做X的blob， 并且使用`GaussianFill`用一些随机的数据填充它。
```python
X = net.GaussianFill([], ["X"], mean=0.0, std=1.0, shape=[2, 3], run_once=0)
print("New network proto:\n\n{}".format(net.Proto()))
```

```python
print("Type of X is: {}".format(type(X)))
print("The blob name is: {}".format(str(X)))
output：
Type of X is: <class 'caffe2.python.core.BlobReference'>
The blob name is: X
```
继续创建W和b。
```python
W = net.GaussianFill([], ["W"], mean=0.0, std=1.0, shape=[5, 3], run_once=0)
b = net.ConstantFill([], ["b"], shape=[5,], value=1.0, run_once=0)
```

```python
Y = X.FC([W, b], ["Y"])
```

可视化这个图。
```python
from caffe2.python import net_drawer
from IPython import display
graph = net_drawer.GetPydotGraph(net, rankdir="LR")
display.Image(graph.create_png(), width=800)
```

![Alt text](\blog\images\post-covers\2018-03-19-1521103032823.png)

有两种方式从Python运行一个网络。
1. 使用`workspace.RunNetOnce()`实例化，运行并且释放这个网络。
2. 稍微有一点复杂，涉及两部操作：(a) 调用`workspace.CreateNet()`来创建空工作空间所拥有的C++网络对象，(b) 通过网络名称传递给它来使用`workspace.RunNet()`。

```python
workspace.ResetWorkspace()
print("Current blobs in the workspace: {}".format(workspace.Blobs()))
workspace.RunNetOnce(net)
print("Blobs in the workspace after execution: {}".format(workspace.Blobs()))
# Let's dump the contents of the blobs
for name in workspace.Blobs():
    print("{}:\n{}".format(name, workspace.FetchBlob(name)))
```

```python
workspace.ResetWorkspace()
print("Current blobs in the workspace: {}".format(workspace.Blobs()))
workspace.CreateNet(net)
workspace.RunNet(net.Proto().name)
print("Blobs in the workspace after execution: {}".format(workspace.Blobs()))
for name in workspace.Blobs():
    print("{}:\n{}".format(name, workspace.FetchBlob(name)))
```
RunNetOnce和RunNet之间有一些区别，但可能主要的区别是计算时间开销。 由于RunNetOnce涉及序列化在Python和C之间传递的protobuf并实例化网络，因此运行可能需要更长的时间。 让我们来看看这种情况下的开销是多少。
```python
# It seems that %timeit magic does not work well with
# C++ extensions so we'll basically do for loops
start = time.time()
for i in range(1000):
    workspace.RunNetOnce(net)
end = time.time()
print('Run time per RunNetOnce: {}'.format((end - start) / 1000))

start = time.time()
for i in range(1000):
    workspace.RunNet(net.Proto().name)
end = time.time()
print('Run time per RunNet: {}'.format((end - start) / 1000))
```
Output：
```
Run time per RunNetOnce: 0.000364284992218
Run time per RunNet: 4.42600250244e-06
```

# 2. 图像加载和预处理

```python
%matplotlib inline
import skimage
import skimage.io as io
import skimage.transform 
import sys
import numpy as np
import math
from matplotlib import pyplot
import matplotlib.image as mpimg
print("Required modules imported.")
```

## Caffe使用BGR的顺序
由于OpenCV在Caffe中的传统支持，并且它粗粝蓝绿红（BGR）顺序的图像，而不是更常用的红绿蓝（RGB）顺序，所以Cafffe2也期望BGR顺序。
```python
# You can load either local IMAGE_FILE or remote URL
# For Round 1 of this tutorial, try a local image.
IMAGE_LOCATION = 'images/cat.jpg'

img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)

# test color reading
# show the original image
pyplot.figure()
pyplot.subplot(1,2,1)
pyplot.imshow(img)
pyplot.axis('on')
pyplot.title('Original image = RGB')

# show the image in BGR - just doing RGB->BGR temporarily for display
imgBGR = img[:, :, (2, 1, 0)]
#pyplot.figure()
pyplot.subplot(1,2,2)
pyplot.imshow(imgBGR)
pyplot.axis('on')
pyplot.title('OpenCV, Caffe2 = BGR')
```
![Alt text](\blog\images\post-covers\2018-03-19-1521112881069.png)


### Caffe喜欢CHW顺序

- H：Height
- W：Width
- C：Channel（as in color）

```python
# Image came in sideways - it should be a portait image!
# How you detect this depends on the platform
# Could be a flag from the camera object
# Could be in the EXIF data
# ROTATED_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/8/87/Cell_Phone_Tower_in_Ladakh_India_with_Buddhist_Prayer_Flags.jpg"
ROTATED_IMAGE = "images/cell-tower.jpg"
imgRotated = skimage.img_as_float(skimage.io.imread(ROTATED_IMAGE)).astype(np.float32)
pyplot.figure()
pyplot.imshow(imgRotated)
pyplot.axis('on')
pyplot.title('Rotated image')

# Image came in flipped or mirrored - text is backwards!
# Again detection depends on the platform
# This one is intended to be read by drivers in their rear-view mirror
# MIRROR_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/2/27/Mirror_image_sign_to_be_read_by_drivers_who_are_backing_up_-b.JPG"
MIRROR_IMAGE = "images/mirror-image.jpg"
imgMirror = skimage.img_as_float(skimage.io.imread(MIRROR_IMAGE)).astype(np.float32)
pyplot.figure()
pyplot.imshow(imgMirror)
pyplot.axis('on')
pyplot.title('Mirror image')
```
## 图像处理操作
### 镜像
```python
# Run me to flip the image back and forth
imgMirror = np.fliplr(imgMirror)
pyplot.figure()
pyplot.imshow(imgMirror)
pyplot.axis('off')
pyplot.title('Mirror image')
```

### 旋转
```python
# Run me to rotate the image 90 degrees
imgRotated = np.rot90(imgRotated)
pyplot.figure()
pyplot.imshow(imgRotated)
pyplot.axis('off')
pyplot.title('Rotated image')
```

### 调整
```python
# Model is expecting 224 x 224, so resize/crop needed.
# Here are the steps we use to preprocess the image.
# (1) Resize the image to 256*256, and crop out the center.
input_height, input_width = 224, 224
print("Model's input shape is %dx%d") % (input_height, input_width)
#print("Original image is %dx%d") % (skimage.)
img256 = skimage.transform.resize(img, (256, 256))
pyplot.figure()
pyplot.imshow(img256)
pyplot.axis('on')
pyplot.title('Resized image to 256x256')
print("New image shape:" + str(img256.shape))
```

### 重新缩放（Rescaling）
```python
print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
print("Model's input shape is %dx%d") % (input_height, input_width)
aspect = img.shape[1]/float(img.shape[0])
print("Orginal aspect ratio: " + str(aspect))
if(aspect>1):
    # landscape orientation - wide image
    res = int(aspect * input_height)
    imgScaled = skimage.transform.resize(img, (input_height, res))
if(aspect<1):
    # portrait orientation - tall image
    res = int(input_width/aspect)
    imgScaled = skimage.transform.resize(img, (res, input_width))
if(aspect == 1):
    imgScaled = skimage.transform.resize(img, (input_height, input_width))
pyplot.figure()
pyplot.imshow(imgScaled)
pyplot.axis('on')
pyplot.title('Rescaled image')
print("New image shape:" + str(imgScaled.shape) + " in HWC")
```

Output：
```
Original image shape:(360, 480, 3) and remember it should be in H, W, C!
Model's input shape is 224x224
Orginal aspect ratio: 1.33333333333
New image shape:(224, 298, 3) in HWC
```

### 裁剪
```python
# Compare the images and cropping strategies
# Try a center crop on the original for giggles
print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]
# yes, the function above should match resize and take a tuple...

pyplot.figure()
# Original image
imgCenter = crop_center(img,224,224)
pyplot.subplot(1,3,1)
pyplot.imshow(imgCenter)
pyplot.axis('on')
pyplot.title('Original')

# Now let's see what this does on the distorted image
img256Center = crop_center(img256,224,224)
pyplot.subplot(1,3,2)
pyplot.imshow(img256Center)
pyplot.axis('on')
pyplot.title('Squeezed')

# Scaled image
imgScaledCenter = crop_center(imgScaled,224,224)
pyplot.subplot(1,3,3)
pyplot.imshow(imgScaledCenter)
pyplot.axis('on')
pyplot.title('Scaled')
```

### 上采样
```python
imgTiny = "images/Cellsx128.png"
imgTiny = skimage.img_as_float(skimage.io.imread(imgTiny)).astype(np.float32)
print "Original image shape: ", imgTiny.shape
imgTiny224 = skimage.transform.resize(imgTiny, (224, 224))
print "Upscaled image shape: ", imgTiny224.shape
```
Output:
```
Original image shape:  (128, 128, 4)
Upscaled image shape:  (224, 224, 4)
```
PNG格式图像是4个通道。
## 最终处理和批处理

我们首先将图像的数据顺序切换到BGR，然后重新编码用于GPU处理的列（HWC->CHW），然后向图像添加第四维(N)，表示图像的数量。最终的顺序是：N，C，H，W。
```python
# this next line helps with being able to rerun this section
# if you want to try the outputs of the different crop strategies above
# swap out imgScaled with img (original) or img256 (squeezed)
imgCropped = crop_center(imgScaled,224,224)
print "Image shape before HWC --> CHW conversion: ", imgCropped.shape
# (1) Since Caffe expects CHW order and the current image is HWC,
#     we will need to change the order.
imgCropped = imgCropped.swapaxes(1, 2).swapaxes(0, 1)
print "Image shape after HWC --> CHW conversion: ", imgCropped.shape

pyplot.figure()
for i in range(3):
    # For some reason, pyplot subplot follows Matlab's indexing
    # convention (starting with 1). Well, we'll just follow it...
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(imgCropped[i])
    pyplot.axis('off')
    pyplot.title('RGB channel %d' % (i+1))

# (2) Caffe uses a BGR order due to legacy OpenCV issues, so we
#     will change RGB to BGR.
imgCropped = imgCropped[(2, 1, 0), :, :]
print "Image shape after BGR conversion: ", imgCropped.shape
# for discussion later - not helpful at this point
# (3) We will subtract the mean image. Note that skimage loads
#     image in the [0, 1] range so we multiply the pixel values
#     first to get them into [0, 255].
#mean_file = os.path.join(CAFFE_ROOT, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mean = np.load(mean_file).mean(1).mean(1)
#img = img * 255 - mean[:, np.newaxis, np.newaxis]

pyplot.figure()
for i in range(3):
    # For some reason, pyplot subplot follows Matlab's indexing
    # convention (starting with 1). Well, we'll just follow it...
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(imgCropped[i])
    pyplot.axis('off')
    pyplot.title('BGR channel %d' % (i+1))
# (4) finally, since caffe2 expect the input to have a batch term
#     so we can feed in multiple images, we will simply prepend a
#     batch dimension of size 1. Also, we will make sure image is
#     of type np.float32.
imgCropped = imgCropped[np.newaxis, :, :, :].astype(np.float32)
print 'Final input shape is:', imgCropped.shape
```
## 重点
```python
# HWC to CHW
imgCropped = imgCropped.swapaxes(1, 2).swapaxes(0, 1)
# RGB to BGR
imgCropped = imgCropped[(2, 1, 0), :, :]
# prepend a batch dimension of size 1
imgCropped = imgCropped[np.newaxis, :, :, :].astype(np.float32)
```
# 3. 加载预训练的模型

## 模型下载
[Github caffe2/models](http://github.com/caffe2/models)
```
python -m caffe2.python.models.download -i squeezenet
```
`- i`标志位表示安装在`/caffe2/python/models`文件夹下。
或者下载这个模型的repo：`git clone https://github.com/caffe2/models`。

```python
%matplotlib inline
import sys
sys.path.insert(0, '/usr/local')
from caffe2.proto import caffe2_pb2
import numpy as np
import skimage.io
import skimage.transform
from matplotlib import pyplot
import os
from caffe2.python import core, workspace, models
import urllib2
print("Required modules imported.")
```

```python
# Configuration --- Change to your setup and preferences!
CAFFE_MODELS = "/usr/local/caffe2/python/models"

# sample images you can try, or use any URL to a regular image.

IMAGE_LOCATION = "images/flower.jpg"

# What model are we using? You should have already converted or downloaded one.
# format below is the model's: 
# folder, INIT_NET, predict_net, mean, input image size
# you can switch squeezenet out with 'bvlc_alexnet', 'bvlc_googlenet' or others that you have downloaded
# if you have a mean file, place it in the same dir as the model
MODEL = 'squeezenet', 'init_net.pb', 'predict_net.pb', 'ilsvrc_2012_mean.npy', 227

# codes - these help decypher the output and source from a list from AlexNet's object codes to provide an result like "tabby cat" or "lemon" depending on what's in the picture you submit to the neural network.
# The list of output codes for the AlexNet models (squeezenet)
codes =  "https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes"
print "Config set!"
```
### crop_center和rescale函数
```python
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    print("Model's input shape is %dx%d") % (input_height, input_width)
    aspect = img.shape[1]/float(img.shape[0])
    print("Orginal aspect ratio: " + str(aspect))
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    print("New image shape:" + str(imgScaled.shape) + " in HWC")
    return imgScaled
```
### 加载均值
```python
# set paths and variables from model choice and prep image
CAFFE_MODELS = os.path.expanduser(CAFFE_MODELS)
# mean can be 128 or custom based on the model
# gives better results to remove the colors found in all of the training images
MEAN_FILE = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[3])
if not os.path.exists(MEAN_FILE):
    mean = 128
else:
    mean = np.load(MEAN_FILE).mean(1).mean(1)
    mean = mean[:, np.newaxis, np.newaxis]
print "mean was set to: ", mean
```
### 设定输入图片尺寸以及检查模型是否存在
```python
# some models were trained with different image sizes, this helps you calibrate your image
INPUT_IMAGE_SIZE = MODEL[4]

# make sure all of the files are around...
#if not os.path.exists(CAFFE2_ROOT):
#    print("Houston, you may have a problem.") 
INIT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[1])
print 'INIT_NET = ', INIT_NET
PREDICT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[2])
print 'PREDICT_NET = ', PREDICT_NET
if not os.path.exists(INIT_NET):
    print(INIT_NET + " not found!")
else:
    print "Found ", INIT_NET, "...Now looking for", PREDICT_NET
    if not os.path.exists(PREDICT_NET):
        print "Caffe model file, " + PREDICT_NET + " was not found!"
    else:
        print "All needed files found! Loading the model in the next block."
  
```

### 加载图像以及转换图像格式
```python
# load and transform image
img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)

# switch to CHW
img = img.swapaxes(1, 2).swapaxes(0, 1)

# switch to BGR
img = img[(2, 1, 0), :, :]

# remove mean for better results
img = img * 255 - mean

# add batch size
img = img[np.newaxis, :, :, :].astype(np.float32)
print "NCHW: ", img.shape
```
## 打开protobufs，加载它们到workspace，并且运行网络
```python
# initialize the neural net

with open(INIT_NET) as f:
    init_net = f.read()
with open(PREDICT_NET) as f:
    predict_net = f.read()
    
p = workspace.Predictor(init_net, predict_net)

# run the net and return prediction
results = p.run([img])

# turn it into something we can play with and examine which is in a multi-dimensional array
results = np.asarray(results)
print "results shape: ", results.shape
```
Output：
```
results shape:  (1, 1, 1000, 1, 1)
```
### 读取结果
```python
# the rest of this is digging through the results 
results = np.delete(results, 1)
index = 0
highest = 0
arr = np.empty((0,2), dtype=object)
arr[:,0] = int(10)
arr[:,1:] = float(10)
for i, r in enumerate(results):
    # imagenet index begins with 1!
    i=i+1
    arr = np.append(arr, np.array([[i,r]]), axis=0)
    if (r > highest):
        highest = r
        index = i 

# top 3 results
print "Raw top 3 results:", sorted(arr, key=lambda x: x[1], reverse=True)[:3]

# now we can grab the code list
response = urllib2.urlopen(codes)

# and lookup our result from the list
for line in response:
    code, result = line.partition(":")[::2]
    if (code.strip() == str(index)):
        print MODEL[0], "infers that the image contains ", result.strip()[1:-2], "with a ", highest*100, "% probability"
```
## 模型快速加载

在此脚本工作之前，需要先下载模型并且安装它，可以执行下面的命令：
```
sudo python -m caffe2.python.models.download -i squeezenet
```

```python
# load up the caffe2 workspace
from caffe2.python import workspace
# choose your model here (use the downloader first)
from caffe2.python.models import squeezenet as mynet
# helper image processing functions
import helpers

# load the pre-trained model
init_net = mynet.init_net
predict_net = mynet.predict_net
# you must name it something
predict_net.name = "squeezenet_predict"
workspace.RunNetOnce(init_net)
workspace.CreateNet(predict_net)
p = workspace.Predictor(init_net.SerializeToString(), predict_net.SerializeToString())

# use whatever image you want (urls work too)
img = "images/flower.jpg"
# average mean to subtract from the image
mean = 128
# the size of images that the model was trained with
input_size = 227

# use the image helper to load the image and convert it to NCHW
img = helpers.loadToNCHW(img, mean, input_size)

# submit the image to net and get a tensor of results
results = p.run([img])   
response = helpers.parseResults(results)
# and lookup our result from the inference list
print response
```

# 4. Python Op 教程

在本教程中，我们介绍允许使用Python编写Caffe2 operators的Python operator，我们还讨论一些底层实现的细节。

## Forward Python Operator, Net.Python() Interface

Caffe2提供了一个高级接口，这可以帮助创建Python ops。
```python
from caffe2.python import core, workspace
import numpy as np

def f(inputs, outputs):
    outputs[0].feed(2 * inputs[0].data)

workspace.ResetWorkspace()
net = core.Net("tutorial")
net.Python(f)(["x"], ["y"])
workspace.FeedBlob("x", np.array([3.]))
workspace.RunNetOnce(net)
print(workspace.FetchBlob("y"))
```
Output：
```
[6.]
```
`net.Python()`函数返回一个可调用，可以像其他任何operator一样使用。


```python
def f_reshape(inputs, outputs):
    outputs[0].reshape(inputs[0].shape)
    outputs[0].data[...] = 2 * inputs[0].data

workspace.ResetWorkspace()
net = core.Net("tutorial")
net.Python(f_reshape)(["x"], ["z"])
workspace.FeedBlob("x", np.array([3.]))
workspace.RunNetOnce(net)
print(workspace.FetchBlob("z"))
```
此示例正常工作，因为`reshape`方法更新底层Caffe2张量，随后调用`.data`属性返回一个Numpy数组，该数组与Caffe2张量共享内存。 `f_reshape`中的最后一行将数据复制到共享内存位置。

当传递`pass_workspace=True`时，工作空间被传递给operator的Python函数：
```python
def f_workspace(inputs, outputs, workspace):
    outputs[0].feed(2 * workspace.blobs["x"].fetch())

workspace.ResetWorkspace()
net = core.Net("tutorial")
net.Python(f_workspace, pass_workspace=True)([], ["y"])
workspace.FeedBlob("x", np.array([3.]))
workspace.RunNetOnce(net)
print(workspace.FetchBlob("y"))
```

## Gradient Python Operator

另外一个重要的`net.Python`参数是`grad_f`，一个对应梯度operator的Python函数。
```python
def f(inputs, outputs):
            outputs[0].reshape(inputs[0].shape)
            outputs[0].data[...] = inputs[0].data * 2

def grad_f(inputs, outputs):
    # Ordering of inputs is [fwd inputs, outputs, grad_outputs]
    grad_output = inputs[2]

    grad_input = outputs[0]
    grad_input.reshape(grad_output.shape)
    grad_input.data[...] = grad_output.data * 2

workspace.ResetWorkspace()
net = core.Net("tutorial")
net.Python(f, grad_f)(["x"], ["y"])
workspace.FeedBlob("x", np.array([3.]))
net.AddGradientOperators(["y"])
workspace.RunNetOnce(net)
print(workspace.FetchBlob("x_grad"))
```python
# 5. A Toy Regression
这是一个简单的回归教程。

我们处理的这个问题非常简单，有两维的数据输入`x`和一维的数据输出`y`，权重向量`w=[2.0, 1.5]`和偏差`b=0.5`。这个等式产生ground truth：
$$y=wx+b$$
我们将在Caffe2 Operator中写出每一个数学运算。如果你的算法是相对标准的，比如CNN模型，这往往是一种过分注重细节的行为。在MNIST教程中，我们将演示如何使用CNN模型helper更容易地构建模型。
```python
from caffe2.python import core, cnn, net_drawer, workspace, visualize
import numpy as np
from IPython import display
from matplotlib import pyplot
```
## 声明计算图
我们声明了两个计算图：一个用于初始化我们将用于计算中的各种参数和常量，另一个用于运行随机梯度下降的主图。

首先我们初始化网络：注意这个名字并不重要，我们基本上想把初始化代码放在一个网络中，这样我们就可以调用`RunNetOnce()`去执行它。我们有一个单独的`init_net`的原因是，这些operators不需要为整个训练过程运行多次。

```python
init_net = core.Net("init")
# The ground truth parameters.
W_gt = init_net.GivenTensorFill(
    [], "W_gt", shape=[1, 2], values=[2.0, 1.5])
B_gt = init_net.GivenTensorFill([], "B_gt", shape=[1], values=[0.5])
# Constant value ONE is used in weighted sum when updating parameters.
ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
# ITER is the iterator count.
ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0, dtype=core.DataType.INT32)

# For the parameters to be learned: we randomly initialize weight
# from [-1, 1] and init bias with 0.0.
W = init_net.UniformFill([], "W", shape=[1, 2], min=-1., max=1.)
B = init_net.ConstantFill([], "B", shape=[1], value=0.0)
print('Created init net.')
```

主要的训练网络定义如下。我们将通过多个步骤展示创建的内容。
- 产生损失的正向传播
- 通过自动求导产生的反向传播
- 参数更新部分，这是一个标准的SGD


```python
train_net = core.Net("train")
# First, we generate random samples of X and create the ground truth.
X = train_net.GaussianFill([], "X", shape=[64, 2], mean=0.0, std=1.0, run_once=0)
Y_gt = X.FC([W_gt, B_gt], "Y_gt")
# We add Gaussian noise to the ground truth
noise = train_net.GaussianFill([], "noise", shape=[64, 1], mean=0.0, std=1.0, run_once=0)
Y_noise = Y_gt.Add(noise, "Y_noise")
# Note that we do not need to propagate the gradients back through Y_noise,
# so we mark StopGradient to notify the auto differentiating algorithm
# to ignore this path.
Y_noise = Y_noise.StopGradient([], "Y_noise")

# Now, for the normal linear regression prediction, this is all we need.
Y_pred = X.FC([W, B], "Y_pred")

# The loss function is computed by a squared L2 distance, and then averaged
# over all items in the minibatch.
dist = train_net.SquaredL2Distance([Y_noise, Y_pred], "dist")
loss = dist.AveragedLoss([], ["loss"])
```

## 网络可视化
现在，我们看一眼整个网络。从下图，可以看出它主要由四个部分组成：
- 为次批次随机生成X（GaussianFill生成X）
- 使用W_gt，B_gt和FC operator来生成ground truth Y_gt
- 使用当前的参数W和B进行预测
- 比较输出并计算损失

```python
graph = net_drawer.GetPydotGraph(train_net.Proto().op, "train", rankdir="LR")
display.Image(graph.create_png(), width=800)
```

![Alt text](\blog\images\post-covers\2018-03-19-1521190765127.png)

现在，类似于所有其他框架，Caffe2允许我们自动生成梯度operators。我们可视化看看。
```python
# Get gradients for all the computations above.
gradient_map = train_net.AddGradientOperators([loss])
graph = net_drawer.GetPydotGraph(train_net.Proto().op, "train", rankdir="LR")
display.Image(graph.create_png(), width=800)
```
![Alt text](\blog\images\post-covers\2018-03-19-1521190876284.png)

一旦我们获得了参数的梯度，我们将添加图的SGD部分：获取当前步的学习率，然后进行参数更新。在这个例子中，我们没有做任何事情：只是简单的SGDs。

```python
# Increment the iteration by one.
train_net.Iter(ITER, ITER)
# Compute the learning rate that corresponds to the iteration.
LR = train_net.LearningRate(ITER, "LR", base_lr=-0.1,
                            policy="step", stepsize=20, gamma=0.9)

# Weighted sum 
train_net.WeightedSum([W, ONE, gradient_map[W], LR], W)
train_net.WeightedSum([B, ONE, gradient_map[B], LR], B)

# Let's show the graph again.
graph = net_drawer.GetPydotGraph(train_net.Proto().op, "train", rankdir="LR")
display.Image(graph.create_png(), width=800)
```

![Alt text](\blog\images\post-covers\2018-03-19-1521190892301.png)

## 创建网络
现在我们已经创建了这个网络，让我们运行他们。
```python
workspace.RunNetOnce(init_net)
workspace.CreateNet(train_net)
```
在我们开始任何训练迭代之前，让我们看看参数。
```python
print("Before training, W is: {}".format(workspace.FetchBlob("W")))
print("Before training, B is: {}".format(workspace.FetchBlob("B")))
```
Output:
```
Before training, W is: [[-0.905963   -0.21433014]]
Before training, B is: [0.]
```
```python
for i in range(100):
    workspace.RunNet(train_net.Proto().name)
```
现在，让我们看一下训练之后的参数。
```python
print("After training, W is: {}".format(workspace.FetchBlob("W")))
print("After training, B is: {}".format(workspace.FetchBlob("B")))

print("Ground truth W is: {}".format(workspace.FetchBlob("W_gt")))
print("Ground truth B is: {}".format(workspace.FetchBlob("B_gt")))
```
Output:
```
After training, W is: [[2.011532  1.4848436]]
After training, B is: [0.49105117]
Ground truth W is: [[2.  1.5]]
Ground truth B is: [0.5]
```
## 运行网络权重变化可视化
看起来很简单吧？然我们仔细看看训练步骤中参数更新的进展情况。为此，让我们重新初始化参数，并且查看在迭代中参数的变化。请记住，我们可以随时从工作区中获取Blob。

```python
workspace.RunNetOnce(init_net)
w_history = []
b_history = []
for i in range(50):
    workspace.RunNet(train_net.Proto().name)
    w_history.append(workspace.FetchBlob("W"))
    b_history.append(workspace.FetchBlob("B"))
w_history = np.vstack(w_history)
b_history = np.vstack(b_history)
pyplot.plot(w_history[:, 0], w_history[:, 1], 'r')
pyplot.axis('equal')
pyplot.xlabel('w_0')
pyplot.ylabel('w_1')
pyplot.grid(True)
pyplot.figure()
pyplot.plot(b_history)
pyplot.xlabel('iter')
pyplot.ylabel('b')
pyplot.grid(True)
pyplot.show()
```
![Alt text](\blog\images\post-covers\2018-03-19-1521196437690.png)

![Alt text](\blog\images\post-covers\2018-03-19-1521196443210.png)

你可以观察到随机梯度下降的非常典型的行为：由于噪声，整个训练中参数波动很大。


# 6. MNIST

在本教程中，我们将展示如何训练实际的模型。我们使用MNIST数据集和LeNet模型，略有改动，即用ReLU代替sigmoid激活函数。

我们将使用model helper，这有助于我们处理模型初始化。

首先，我们导入必备的库。
```python
%matplotlib inline
from matplotlib import pyplot
import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe


from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
print("Necessities imported!")
```
## 数据整理
我们将在训练期间跟踪统计数据，并将这些数据存储在本地文件夹中。我们需要为数据设置一个数据文件夹，并为统计数据建立一个根文件夹。
```python
# This section preps your image and test set in a lmdb database
def DownloadResource(url, path):
    '''Downloads resources from s3 by url and unzips them to the provided path'''
    import requests, zipfile, StringIO
    print("Downloading... {} to {}".format(url, path))
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall(path)
    print("Completed download and extraction.")
    
    
current_folder = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks')
data_folder = os.path.join(current_folder, 'tutorial_data', 'mnist')
root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
db_missing = False

if not os.path.exists(data_folder):
    os.makedirs(data_folder)   
    print("Your data folder was not found!! This was generated: {}".format(data_folder))

# Look for existing database: lmdb
if os.path.exists(os.path.join(data_folder,"mnist-train-nchw-lmdb")):
    print("lmdb train db found!")
else:
    db_missing = True
    
if os.path.exists(os.path.join(data_folder,"mnist-test-nchw-lmdb")):
    print("lmdb test db found!")
else:
    db_missing = True

# attempt the download of the db if either was missing
if db_missing:
    print("one or both of the MNIST lmbd dbs not found!!")
    db_url = "http://download.caffe2.ai/databases/mnist-lmdb.zip"
    try:
        DownloadResource(db_url, data_folder)
    except Exception as ex:
        print("Failed to download dataset. Please download it manually from {}".format(db_url))
        print("Unzip it and place the two database folders here: {}".format(data_folder))
        raise ex

if os.path.exists(root_folder):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree(root_folder)
    
os.makedirs(root_folder)
workspace.ResetWorkspace(root_folder)

print("training data folder:" + data_folder)
print("workspace root folder:" + root_folder)
```

我们将使用`ModelHelper`类来表示我们的主模型，并使用`brew`和`Operators`来构建我们的模型。`brew`模块具有一组包装函数，可以自动将参数初始化和实际计算分离为两个网络。在`ModelHelper`对象底层，有两个底层网络`param_init_net`和`net`，分别记录初始化网络和主网络。

为了模块化，我们将模型分为多个不同的部分：
1. 数据输入部分（AddInput 函数）
2. 主计算部分（AddLeNetModel 函数）
3. 训练部分 - 添加梯度运算符，更新等（AddTrainingOperators 函数）
4. 簿记部分，我们只打印统计数据以供检查（AddBookkeepingOperators 函数）

`AddInput`将从DB中加载数据。我们将MNIST数据存储为像素值，因此在批处理之后，这将为我们提供具有形状（batch_size, num_channels, width, height）的数据。类型为`uint8`形状为 `[batch_size, 1, 28, 28]`的数据和类型为`int`形状为`[batch_size]`的标签。

由于我们要做浮点计算，因此我们将把数据类型转换为浮点数据类型。为了获得更好的数值稳定性，我么将它们缩放到[0, 1]，而不是在[0, 255]范围内表示数据。请注意，我们正在为此运算符进行就地计算：我们不需要预先缩放数据。现在，在计算反向回传时，我们不需要为它计算梯度。`StopGradient`的确如此：在前向中它什么都不做，而在反向传播中所做的只是告诉梯度发生器“梯度不需要经过我”。
```python
def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label
```
下面我们将看到网络将出现的预测转化为概率。输出结果将符合0到1之间，因此越接近1，说明数字越可能与预测相匹配。这个转化是通过softmax函数实现的，我们可以在下面代码中看到。下面的`AddLeNetMode`函数将输出softmax。这个函数不光有softmax，它还是具有卷基层的计算模型。

在[图像处理中对内核的解释](https://en.wikipedia.org/wiki/Kernel_%28image_processing%29)可能对于为什么在下面的卷积图层中使用`kernel=5`提供更多有用的信息。`dim_in`是输入通道的数量，`dim_out`是输出通道的数量。
## 设定LeNet模型
```python
def AddLeNetModel(model, data):
    '''
    This part is the standard LeNet model: from data to the softmax prediction.
    
    For each convolutional layer we specify dim_in - number of input channels
    and dim_out - number or output channels. Also each Conv and MaxPool layer changes the
    image size. For example, kernel of size 5 reduces each side of an image by 4.

    While when we have kernel and stride sizes equal 2 in a MaxPool layer, it divides
    each side in half.
    '''
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax
```
下面的`AddAccuracy`函数为模型添加了一个精度运算符。我们将在下一个函数中使用它来跟踪模型的准确性。
```python
def AddAccuracy(model, softmax, label):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy
```
接下来的函数`AddTrainingOperators`，将训练operators添加到模型中。

在第一步中，我们应用一个operator `LabelCrossEntropy`，计算输出和标签集之间的交叉熵。在计算模型损失之前，该operator几乎总是在获取softmax之后使用。它将采用`[softmax, label]`矩阵和“Cross Entropy”的标签`xent`。

	xent = model.LabelCrossEntropy([softmax, label], 'xent')

`AveragedLoss` 将取代交叉熵并返回交叉熵中发现的损失的平均值。

    loss = model.AveragedLoss(xent, "loss")

为了簿记目的，我们还将通过调用`AddAccuracy`函数来计算模型的准确性，如下所示：

    AddAccuracy(model, softmax, label)

下一行是训练模型的关键部分：我们将所有梯度运算符添加到模型中。这些梯度是我们在上面计算的损失梯度。

    model.AddGradientOperators([loss])
 
接下来的几行支持非常简单的随机梯度下降。Caffe2的工作人员正在努力包装这些SGD操作，他们会在准备就绪时更新此操作。现在，你可以看到我们如何显示具有基本SGD算法的operators。

`Iter` operator 是我们在训练过程中运行的迭代次数的计数器。我们使用`brew.iter` 辅助函数将其添加到模型中。

    ITER = brew.iter(model, "iter")

我们做了一个简单的学习率方法，其中 `lr = base_lr * (t ^ gamma)`。注意，我们正在做最小化，所以`base_lr`是负数，我们要进入DOWNHILL方向。

    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
        
`ONE`是用于梯度更新的常量值。我们只需要创建一次，所以它明确第放置在`param_init_net`中。

    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)

现在， 对于每个参数，我们都会进行梯度更新。我们如何获得每个参数的梯度 - `ModelHelper` 会跟踪这些变量。更新是一个简单的加权和： `param = param + param_grad * LR`

    for param in model.params:
        param_grad = model.param_to_grad[param]
        model.WeightedSum([param, ONE, param_grad, LR], param)        
        
我们需要定期检查模型的参数。这是通过`Checkpoint` operator 完成的。它还需要一个参数“every”，以便我们不会太频繁地使用检查点。这样，我们每20次迭代就让我们checkpoint。

    model.Checkpoint([ITER] + model.params, [],
                   db="mnist_lenet_checkpoint_%05d.lmdb",
                   db_type="lmdb", every=20)

```python
def AddTrainingOperators(model, softmax, label):
    """Adds training operators to the model."""
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = brew.iter(model, "iter")
    # set the learning rate schedule
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - ModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)
```

以下函数`AddBookkeepingOpeators`添加了一些bookkeeping operators，我们稍后可以检查它们。这些operators不会影响训练过程：它们只收集统计信息并将其打印到文件或日志中。
```python
def AddBookkeepingOperators(model):
    """This adds a few bookkeeping operators that we can inspect later.
    
    These operators do not affect the training procedure: they only collect
    statistics and prints them to file or to logs.
    """    
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
    # Now, if we really want to be verbose, we can summarize EVERY blob
    # that the model produces; it is probably not a good idea, because that
    # is going to take time - summarization do not come for free. For this
    # demo, we will only show how to summarize the parameters and their
    # gradients.
```
现在，让我们创建用于训练和测试的模型。我们之前创建的函数将被执行。请记住我们正在做的四个步骤：
1. 数据输入
2. 主要的计算
3. 训练
4. 簿记

在我们送入数据之前，需要定义我们的训练模型。我们基本上需要上面定义的每一个组件。在这个例子中，我们在mnist_train数据集上使用`HCHW`存储顺序。

```python
arg_scope = {"order": "NCHW"}
train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)
data, label = AddInput(
    train_model, batch_size=64,
    db=os.path.join(data_folder, 'mnist-train-nchw-lmdb'),
    db_type='lmdb')
softmax = AddLeNetModel(train_model, data)
AddTrainingOperators(train_model, softmax, label)
AddBookkeepingOperators(train_model)

# Testing model. We will set the batch size to 100, so that the testing
# pass is 100 iterations (10,000 images in total).
# For the testing model, we need the data input part, the main LeNetModel
# part, and an accuracy part. Note that init_params is set False because
# we will be using the parameters obtained from the train model.
test_model = model_helper.ModelHelper(
    name="mnist_test", arg_scope=arg_scope, init_params=False)
data, label = AddInput(
    test_model, batch_size=100,
    db=os.path.join(data_folder, 'mnist-test-nchw-lmdb'),
    db_type='lmdb')
softmax = AddLeNetModel(test_model, data)
AddAccuracy(test_model, softmax, label)

# Deployment model. We simply need the main LeNetModel part.
deploy_model = model_helper.ModelHelper(
    name="mnist_deploy", arg_scope=arg_scope, init_params=False)
AddLeNetModel(deploy_model, "data")
# You may wonder what happens with the param_init_net part of the deploy_model.
# No, we will not use them, since during deployment time we will not randomly
# initialize the parameters, but load the parameters from the db.

```
## 模型可视化
现在，让我们看看使用caffe2所具有的简单图形可视化工具，看看training和deploy模型长什么样子。如果一下命令失败，请安装`graphviz`。安装方式参考下边的命令行：
		
	sudo yum install graphviz

 
```python
from IPython import display
graph = net_drawer.GetPydotGraph(train_model.net.Proto().op, "mnist", rankdir="LR")
display.Image(graph.create_png(), width=800)
```
![Alt text](\blog\images\post-covers\2018-03-19-1521362938543.png)

现在，上图显示了训练阶段的一切操作：白色节点是blobs，绿色矩形节点是正在运行的operator。你可以看到这些像轨道一样的平行线路：这些线路是从正向传播山城的blobs到反向operators的依赖关系。

让我们以更简单的方式显示图形，只显示必要的依赖关系并仅显示operators。如果你仔细观察，可以看到图的左半部分是正向传播，图的右半部分是反向传播，右边是一组参数更新和汇总operators。
```python
graph = net_drawer.GetPydotGraphMinimal(
    train_model.net.Proto().op, "mnist", rankdir="LR", minimal_dependency=True)
display.Image(graph.create_png(), width=800)
```
![Alt text](\blog\images\post-covers\2018-03-19-1521362947582.png)

现在，当我们运行网络时，一种方法是直接从Python运行它。当我们运行网络时，我们可以定期从网络中提取blobs，让我们先来说明我们是如何做到这一点。

在此之前，我们再次重申一下，`ModelHelper`类尚未执行任何操作。它所做的只是声明网络，这基本上是创建协议缓冲区。例如，我们将显示训练模型参数初始化网络的一部分序列化协议缓冲区。
```python
print(str(train_model.param_init_net.Proto())[:400] + '\n...')
```
Output:
```
name: "mnist_train_init"
op {
  output: "dbreader_/home/tianliang/caffe2_notebooks/tutorial_data/mnist/mnist-train-nchw-lmdb"
  name: ""
  type: "CreateDB"
  arg {
    name: "db_type"
    s: "lmdb"
  }
  arg {
    name: "db"
    s: "/home/tianliang/caffe2_notebooks/tutorial_data/mnist/mnist-train-nchw-lmdb"
  }
}
op {
  output: "conv1_w"
  name: ""
  type: "XavierFill"
  arg {
    name: "shape"
  
...
```

我们还会将所有协议缓冲区（protocol buffers）转存到磁盘，以便你可以轻松检查它们。这些协议缓冲区与老版的Caffe的网络定义非常相似。
```python
with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.net.Proto()))
with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.net.Proto()))
with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
    fid.write(str(deploy_model.net.Proto()))
print("Protocol buffers files have been created in your root folder: " + root_folder)
```
Output：
```
Protocol buffers files have been created in your root folder: /home/tianliang/caffe2_notebooks/tutorial_files/tutorial_mnist
```
接下来我们将运行训练过程。我们将在Python中驱动所有计算，但是你也可以将计划写入磁盘，以便你可以使用C++完全训练这些内容。
## 运行模型并训练
我们首先必须用以下方式初始化网络：

	workspace.RunNetOnce(train_model.param_init_net)

由于我们要多次运行主网络，我们首选创建一个网络，将从protobuf生成的实际网络放入工作区。

	workspace.CreateNet(train_model.net)

我们将把我们运行网络迭代次数设置为200，并创建两个numpy阵列来记录每次迭代的准确性和损失。

	total_iters = 200
	accuracy = np.zeros(total_iters)
	loss = np.zeros(total_iters)

With the network and tracking of accuracy and loss setup we can now loop the 200 interations calling workspace.
运行网络`RunNet`并传递的网络名称`train_model.net.Proto().name`。在每次迭代中，我们用`workspace.FetchBlob('accuracy')`和`workspace.FetchBlob('loss')`来获取计算的精度和损失。

	for i in range(total_iters):
		workspace.RunNet(train_model.net.Proto().name)
		accuracy[i] = workspace.FetchBlob('accuracy')
		loss[i] = workspace.FetchBlob('loss')
最后，我们可以用pyplot绘制结果。
```python
# The parameter initialization network only needs to be run once.
workspace.RunNetOnce(train_model.param_init_net)
# creating the network
workspace.CreateNet(train_model.net, overwrite=True)
# set the number of iterations and track the accuracy & loss
total_iters = 200
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)
# Now, we will manually run the network for 200 iterations. 
for i in range(total_iters):
    workspace.RunNet(train_model.net)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
# After the execution is done, let's plot the values.
pyplot.plot(loss, 'b')
pyplot.plot(accuracy, 'r')
pyplot.legend(('Loss', 'Accuracy'), loc='upper right')
```
![Alt text](\blog\images\post-covers\2018-03-19-1521362978940.png)

现在我们可以采样一些数据并做出预测。
```python
# Let's look at some of the data.
pyplot.figure()
data = workspace.FetchBlob('data')
_ = visualize.NCHW.ShowMultiple(data)
pyplot.figure()
softmax = workspace.FetchBlob('softmax')
_ = pyplot.plot(softmax[0], 'ro')
pyplot.title('Prediction for the first image')
```
![Alt text](\blog\images\post-covers\2018-03-19-1521362985791.png)

![Alt text](\blog\images\post-covers\2018-03-19-1521362999048.png)


```python
# Convolutions for this mini-batch
pyplot.figure()
conv = workspace.FetchBlob('conv1')
shape = list(conv.shape)
shape[1] = 1
# We can look into any channel. This of it as a feature model learned
conv = conv[:,15,:,:].reshape(shape)

_ = visualize.NCHW.ShowMultiple(conv)
```
![Alt text](\blog\images\post-covers\2018-03-19-1521363007012.png)

我们将运行测试网络，并再次报告测试精度。请注意，虽然test_model将使用从train_model获取的参数，但仍然必须运行`test_model.param_init_net`以初始化输入数据。在此次运行中，我们只需要跟踪精度，而且我们只需要运行100次迭代。
```python
# run a test pass on the test net
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite=True)
test_accuracy = np.zeros(100)
for i in range(100):
    workspace.RunNet(test_model.net.Proto().name)
    test_accuracy[i] = workspace.FetchBlob('accuracy')
# After the execution is done, let's plot the values.
pyplot.plot(test_accuracy, 'r')
pyplot.title('Acuracy over test batches.')
print('test_accuracy: %f' % test_accuracy.mean())
```
![Alt text](\blog\images\post-covers\2018-03-19-1521363015306.png)
##模型权重保存
让我们将训练后的权重和偏差保存到文件中。
```python
# construct the model to be exported
# the inputs/outputs of the model are manually specified.
pe_meta = pe.PredictorExportMeta(
    predict_net=deploy_model.net.Proto(),
    parameters=[str(b) for b in deploy_model.params], 
    inputs=["data"],
    outputs=["softmax"],
)

# save the model to a file. Use minidb as the file format
pe.save_to_db("minidb", os.path.join(root_folder, "mnist_model.minidb"), pe_meta)
print("The deploy model is saved to: " + root_folder + "/mnist_model.minidb")
```
## 加载保存的模型并预测
现在，我们可以加载模型并运行prediction来验证它是否有效。
```python
# we retrieve the last input data out and use it in our prediction test before we scratch the workspace
blob = workspace.FetchBlob("data")
pyplot.figure()
_ = visualize.NCHW.ShowMultiple(blob)

# reset the workspace, to make sure the model is actually loaded
workspace.ResetWorkspace(root_folder)

# verify that all blobs are destroyed. 
print("The blobs in the workspace after reset: {}".format(workspace.Blobs()))

# load the predict net
predict_net = pe.prepare_prediction_net(os.path.join(root_folder, "mnist_model.minidb"), "minidb")

# verify that blobs are loaded back
print("The blobs in the workspace after loading the model: {}".format(workspace.Blobs()))

# feed the previously saved data to the loaded model
workspace.FeedBlob("data", blob)

# predict
workspace.RunNetOnce(predict_net)
softmax = workspace.FetchBlob("softmax")

# the first letter should be predicted correctly
pyplot.figure()
_ = pyplot.plot(softmax[0], 'ro')
pyplot.title('Prediction for the first image')
```
Output：
```
The blobs in the workspace after reset: []
The blobs in the workspace after loading the model: [u'!!META_NET_DEF', u'!!PREDICTOR_DBREADER', u'conv1', u'conv1_b', u'conv1_w', u'conv2', u'conv2_b', u'conv2_w', u'data', u'fc3', u'fc3_b', u'fc3_w', u'pool1', u'pool2', u'pred', u'pred_b', u'pred_w', u'softmax']
Text(0.5,1,u'Prediction for the first image')
```
![Alt text](\blog\images\post-covers\2018-03-19-1521363026007.png)

![Alt text](\blog\images\post-covers\2018-03-19-1521363032094.png)

本教程重点介绍了Caffe2的一些功能以及展示了创建简单的CNN是多么容易。
