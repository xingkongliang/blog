---
title: 感受野（receptive file）计算
layout: post
tags: [Deep Learning]
---



## 什么是感受野

影响最后卷积层输出的像素面积尺寸。

![](http://images2015.cnblogs.com/blog/846839/201610/846839-20161010201946696-456664256.png)

## 感受野如何计算？


下图为VGG16的感受野（receptive field）尺寸

![image](http://note.youdao.com/yws/public/resource/48c558937031c8bb74a1aae054dff471/xmlnote/C8CC31C35EF643D09622355DA7BB381D/5102)

下图为计算方法：

![image](http://note.youdao.com/yws/public/resource/48c558937031c8bb74a1aae054dff471/xmlnote/B92DCA76601C46C5BDB9B4783ED4A99E/5104)

 感受野计算时有下面的几个情况需要说明：

　　（1）第一层卷积层的输出特征图像素的感受野的大小等于滤波器的大小

　　（2）深层卷积层的感受野大小和它之前所有层的滤波器大小和步长有关系

　　（3）计算感受野大小时，忽略了图像边缘的影响，即不考虑padding的大小，关于这个疑惑大家可以阅读一下参考文章2的解答进行理解

这里的每一个卷积层还有一个strides的概念，这个strides是之前所有层stride的乘积。  

　　即strides（i） = stride(1) * stride(2) * ...* stride(i-1) 

　　关于感受野大小的计算采用top to down的方式， 即先计算最深层在前一层上的感受野，然后逐渐传递到第一层，使用的公式可以表示如下：　　　

　　     RF = 1 #待计算的feature map上的感受野大小
　　for layer in （top layer To down layer）:
　　　　RF = ((RF -1)* stride) + fsize

stride 表示卷积的步长； fsize表示卷积层滤波器的大小　　

用python实现了计算Alexnet  zf-5和VGG16网络每层输出feature map的感受野大小，实现代码：

```python
#!/usr/bin/env python

# [filter size, stride, padding]
net_struct = {'alexnet': {'net':[[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0]],
                   'name':['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5']},
       'vgg16': {'net':[[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[3,1,1],
                        [2,2,0],[3,1,1],[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[3,1,1],[2,2,0]],
                 'name':['conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2',
                         'conv3_3', 'pool3','conv4_1','conv4_2','conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5']},
       'zf-5':{'net': [[7,2,3],[3,2,1],[5,2,2],[3,2,1],[3,1,1],[3,1,1],[3,1,1]],
               'name': ['conv1','pool1','conv2','pool2','conv3','conv4','conv5']}}



def outFromIn(isz, net, layernum):
    totstride = 1
    insize = isz
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2*pad) / stride + 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride

def inFromOut(net, layernum):
    RF = 1
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        RF = ((RF -1)* stride) + fsize
    return RF

if __name__ == '__main__':
    imsize = 224

    print "layer output sizes given image = %dx%d" % (imsize, imsize)
    
    for net in net_struct.keys():
        print '************net structrue name is %s**************'% net
        for i in range(len(net_struct[net]['net'])):
            p = outFromIn(imsize,net_struct[net]['net'], i+1)
            rf = inFromOut(net_struct[net]['net'], i+1)
            print "Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (net_struct[net]['name'][i], p[0], p[1], rf)

```

Output:
```
layer output sizes given image = 224x224
************net structrue name is vgg16**************
Layer Name = conv1_1, Output size = 224, Stride =   1, RF size =   3
Layer Name = conv1_2, Output size = 224, Stride =   1, RF size =   5
Layer Name = pool1, Output size = 112, Stride =   2, RF size =   6
Layer Name = conv2_1, Output size = 112, Stride =   2, RF size =  10
Layer Name = conv2_2, Output size = 112, Stride =   2, RF size =  14
Layer Name = pool2, Output size =  56, Stride =   4, RF size =  16
Layer Name = conv3_1, Output size =  56, Stride =   4, RF size =  24
Layer Name = conv3_2, Output size =  56, Stride =   4, RF size =  32
Layer Name = conv3_3, Output size =  56, Stride =   4, RF size =  40
Layer Name = pool3, Output size =  28, Stride =   8, RF size =  44
Layer Name = conv4_1, Output size =  28, Stride =   8, RF size =  60
Layer Name = conv4_2, Output size =  28, Stride =   8, RF size =  76
Layer Name = conv4_3, Output size =  28, Stride =   8, RF size =  92
Layer Name = pool4, Output size =  14, Stride =  16, RF size = 100
Layer Name = conv5_1, Output size =  14, Stride =  16, RF size = 132
Layer Name = conv5_2, Output size =  14, Stride =  16, RF size = 164
Layer Name = conv5_3, Output size =  14, Stride =  16, RF size = 196
Layer Name = pool5, Output size =   7, Stride =  32, RF size = 212
************net structrue name is zf-5**************
Layer Name = conv1, Output size = 112, Stride =   2, RF size =   7
Layer Name = pool1, Output size =  56, Stride =   4, RF size =  11
Layer Name = conv2, Output size =  28, Stride =   8, RF size =  27
Layer Name = pool2, Output size =  14, Stride =  16, RF size =  43
Layer Name = conv3, Output size =  14, Stride =  16, RF size =  75
Layer Name = conv4, Output size =  14, Stride =  16, RF size = 107
Layer Name = conv5, Output size =  14, Stride =  16, RF size = 139
************net structrue name is alexnet**************
Layer Name = conv1, Output size =  54, Stride =   4, RF size =  11
Layer Name = pool1, Output size =  26, Stride =   8, RF size =  19
Layer Name = conv2, Output size =  26, Stride =   8, RF size =  51
Layer Name = pool2, Output size =  12, Stride =  16, RF size =  67
Layer Name = conv3, Output size =  12, Stride =  16, RF size =  99
Layer Name = conv4, Output size =  12, Stride =  16, RF size = 131
Layer Name = conv5, Output size =  12, Stride =  16, RF size = 163
Layer Name = pool5, Output size =   5, Stride =  32, RF size = 195
```

参考博客：

[卷积神经网络物体检测之感受野大小计算](http://www.cnblogs.com/objectDetect/p/5947169.html)

[How to calculate receptive field size?](https://stackoverflow.com/questions/35582521/how-to-calculate-receptive-field-size)

