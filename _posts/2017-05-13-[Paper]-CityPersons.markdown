---
title: Paper-CityPersons:A Diverse Dataset for Pedestrian Detection
layout: post
tags: [Deep Learning]
---

# 摘要

卷积网络已经在行人检测方面取得了重大的进展，但是仍然存在关于合适的结构和训练数据的开放性问题。我们重新审视CNN的涉及，并指出关键调整，使普通的Faster RCNN能够在Caltech数据集上取得最先进的结果。

为了从更多更好的数据中进一步改进，我们在Cityscapes数据集之上引入一组新的人物标注，即CityPersons。CityPersons的多样性使我们能够首次训练一个可以通过多个基准的单一CNN模型。此外，通过CityPersons进行额外的训练，我们使用FasterRCNN在Caltech上获得了最高的成绩，特别是针对困难的情况（严重的遮挡和小尺寸），并且提供了更高的定位质量。


![Figure 1](\blog\images\post-covers\2017-05-13-paper01.png)


## 该工作的主要贡献：

1. 提出CityPersons数据集，在Cityscapes数据集（训练、验证和测试集）上提供一套新的高质量的带有bounding box标注的行人检测数据集。训练和验证的标注即将被公开，并且基线将被设定。
2. 我们在Caltech和KITTI数据集上报告了新的Faster RCNN的state-of-art的结果，由于适当地调整了检测模型并且使用CityPersons进行了预训练。我们在更难的检测情况（小目标和遮挡）下现实了更好的结果并且整体都有更高的定位精度。
3. 使用CityPersons，我们获得了最佳的跨数据集的泛化行人检测结果。
4. 我们利用额外的Cityscapes标注信息展示了初步结果。使用语义标签作为额外的监督，我们在小目标行人上获得了有可能的改进。


# 2 Faster RCNN上的改进

原始的Faster RCNN在Caltech数据集上表现不佳的原因是它无法处理这个数据集上占主导地位的小尺度目标（50~70个像素）。为了更好的处理小尺度的行人目标，作者提出了5个改进。

### M1 量化RPN缩放（Quantized RPN scales）
RPN的默认缩放尺度是稀疏的[0.5, 1, 2]，并且假定目标尺度均匀分布。然而当我们观察Caltech数据集时，我们发现小尺度的人比大尺度的人要多很多。我们的直觉是让网络为小尺寸生成更多的候选区域（proposals），因此更好地处理它们。We split the full scale range in 10 quantile bins (equal amount of samples per bin), and use the resulting 11 endpoints as RPN scales to generate proposals.

### M2 输入上采样（Input up-scaling）
简单地将输入图像上采样2倍，提供了MR0的3.74百分点显著增益。我们将其归因于ImageNet预训练外观分布更好的匹配。使用较大的上采样因子没有进一步的提升。

### M3 更精细的特征步长（Finer feature stride）
在Caltech中大多数行人都有 高x宽=80x40。默认的VGG16有一个16个像素的特征步长。与目标的宽度相比，这样一个粗糙的步长会减少行人目标上具有高分的机会，并且迫使网络处理相对应的目标外观会有一个很大的位移。从VGG16一处第四个max-pooling层可以将步长减少到8个像素；这有助于检测器处理小目标。

### M4 忽视区域处理（Ignore region handling）
原始的Faster RCNN代码不能处理忽视区域。简单地将这些驱虎作为背景将会引入混了的样本，并且对检测器产生负面的影响。通过确保在训练RPN候选区域期间避免对忽视区域进行采样，我们观察到了MR 1.33百分点的改善。

### M5 求解器（Solver）
从标准的Caffe SGD求解器切换到Adam求解器，在试验中提供了一致的增益。

M1和M2是关键，性能提升结果如下表所示。
![Figure 2](\blog\images\post-covers\2017-05-13-paper02.png)


# 3 CityPersons数据集
Cityscapes数据集是为城市街景中语义分割任务而创建的。为27个城市的5000幅图像提供30个视觉的类别的精细像素级标注。精细的标注包含行人和车辆的实例标签。另外来自其他23个城市的20000幅图像都具有粗略的标签，没有实例标签。

在本文中，我们提出了基于Cityscapes数据的CityPersons数据集，为行人检测提供一个新的数据集。对5000个精细标注子集中的每帧图像，我们为行人创建了高质量的边界框标注（3.1节）。在3.2节，我们将CityPersons与以前的数据集进行对比：体积、多样性和遮挡。在第4节中，我们将介绍如何使用新的数据来改进其他数据集的结果。





