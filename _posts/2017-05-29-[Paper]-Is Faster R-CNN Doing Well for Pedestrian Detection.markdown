---
title: Is Faster R-CNN Doing Well for Pedestrian Detection?
layout: post
tags: [Deep Learning]
---

这篇论文被收录在ECCV2016，下载请点击[这里](https://arxiv.org/abs/1607.07032)

# 摘要
检测行人被认为是超过一般物体检测的一个特殊主题。虽然最近深度学习的目标检测器Fast/Faster R-CNN已经显示出了对于一般通用目标的出色的性能，但是它们在行人检测方面取得了有限的成功，并且先前的行人检测器通常是用手工设计的特征和深度卷积特征相结合的混合方法。在这篇论文中，我们研究了针对行人检测的Faster R-CNN方法。我们发现在Faster R-CNN中区域候选网络（RPN）确实可以作为一个独立的行人检测器，并且运行很好，但令人吃惊的是，后面的分类器降低了结果。我们认为两个原因导致了这个不令人满意的准确性：（1）用于处理小目标的特征图分辨率不足；（2）缺乏挖掘难反例的任何bootstrapping策略。在这些观察的推动下，我么提出了一个非常简单但有效的行人检测基线，使用RPN，然后在共享的高分辨率卷积特征图上使用boosted forests。我们在几个基准（Caltech，INRIA，ETH和KITTI），具有很好的准确性和速度。

![Figure 1](\blog\images\post-covers\2017-05-29-paper01.png)
图1 在行人检测中，Fast/Faster R-CNN的两个挑战。

# 3 方法
我们的方法包含两个部分（如图2所示）：一个RPN用于产生目标候选窗口和卷积特征图，一个是Boosted Forest使用这些卷积特征对这些窗口进行分类。
![Figure 2](\blog\images\post-covers\2017-05-29-paper02.png)
图2 我们的流程。RPN被用于计算候选窗口、分数和特征图。候选窗口被送到级联的Boosted Forest（BF）进行分类，使用由RPN计算的特征图池化获得的特征集合。

## 3.1 用于行人检测的RPN
在Faster R-CNN中的RPN在多类别目标检测场景中被设计为类别不可知的检测器（proposer）。对于单一类别检测，RPN自然地成为了唯一涉及类别的检测器。我们特别定制了用于行人检测的RPN，如下所述。

我们采用了0.41（宽度与高度）的单一纵横比的锚点（anchors）（参考窗口）。这是在参考文献[2]中提及的行人平均纵横比。这与具有多个纵横比的锚点的原始RPN[1]不同。不适当的长宽比的锚点与很少的样本相关联，因此对于检测精度来说是噪声和有害的。此外我们使用了9个不同尺度的锚点，从40像素高度开始，缩放步长为1.3x。这涉及到比[1]更广泛的尺度。多尺度锚点的使用放弃了使用特征金字塔检测多尺度对象的要求。

和[1]一样，我们采用了在ImageNet数据集上预训练的VGG-16网络作为我们的backbone网络。RPN建立在Conv5_3层之上，后边是中间3x3卷积层和两个同级1x1卷积层，用于分类和边界框回归。以这种方式，RPN会以16像素（Conv5_3）的步幅回归窗口。分类层为候选窗口提供置信度分数，这可以将其作为随后的Boosted Forest级联的初始分数。

值得注意的是，尽管我们将下一节中使用“a trous”技巧来提高分辨率和减少步幅，但是我们继续使用向相同的RPN，步幅为16像素。特征提取时只能利用a trous技巧，而不是fine-tuning。

## 3.2 特征提取
通过RPN生成候选区域之后，我们采用RoI池化来提取固定长度的特征。这些特征将用来训练BF，如下一节所介绍。与Faster R-CNN不同，需要将这些特征送入原始的全连接层，从而限制其尺度，BF分类器对特征的尺度没有约束。例如，我们可以从Conv3_3（stride=4个像素）和Conv4_3
（stride=8个像素）中提取RoIs的特征。我们将特征池化到一个7x7的固定分辨率。来自不同层的这些特征没有通过任何规范化直接连接，这主要是由于BF分类器的灵活性。相反，对于深度分类器，当特征连接的时候，需要小心地对特征进行规范化处理。

值得注意的是，由于对特征的维度没有约束，因此我们可以灵活地使用，增加分辨率的特征。具体来说，给定RPN（在Conv3上stride=4，在Conv4上stride=8，在Conv5上stride=16）fine-tuned层，我们可以使用 a trous 技巧来计算较高分辨率的卷积特征图。例如，我们可以将Pool3的步长设定为1，并将所有Conv4滤波器扩展为2倍？？，从而将Conv4的步幅从8减少到4，与以前的方法相比，微调了扩展滤波器，在我们的方法中，我们只使用它们进行特征提取，而不需要微调新的RPN。

虽然我们采用了与Faster R-CNN相同的RoI分辨率（7x7），但是这些RoIs是比Fast R-CNN（Conv5_3）在更高分辨率的特征图上（例如，Conv3_3，Conv4_3或者Conv4_3 a trous）。如果RoI的输入分辨率小鱼输出（即<7x7），则pooling bins会崩溃，特征变得平坦（flat）且不具有判别性。这个问题在我们的方法中得到了缓解，因为它不限于在我们的后端分类器中使用Conv5_3的特征。

## 3.3 Boosted Forest
RPN产生了候选区域，置信度分数和特征，这些都用于训练级联的Boosted Forest分类器。我们采用RealBoost算法，主要遵循论文中给出的参数。

## 3.4 实现细节

我们在[11,12,15]中采用单尺度训练和测试，而不使用特征金字塔。调整图像的大小，使其较短的边缘具有N个像素（在Caltech上N=720，在INRIA上N=600，在ETH上N=810，在KITTI上N=500）。对于RPN训练，如果锚点具有大于0.5的IoU（Intersection-over-Union）则被认为是一个正样本，否则为负样本。我们采用以图像为中心的训练方案，每个mini-batch包含1幅图像和120个随机采样的锚点组成，用于计算损失。在一个mini-batch中正样本与负样本的比例为1:5。RPN的其他超参数如[1]，我们采用公开的代码[1]来fine-tune RPN。我们注意到，在[1]中，fine-tuning过程中忽略了跨边界的锚点，而在我们的实现过程中，我们在fine-tuning期间保留了跨边界的负锚点，从而经验性地提高了这些数据集的准确性。

通过fine-tune RPN，我们采用阈值为0.7的非极大值抑制（NMS）来过滤目标候选区域。然后，这些候选区域通过得分进行排序。对于BF训练，我们通过选择每个图像的钱1000个候选窗口（和ground truths）来构建训练集。Caltech和KITTI数据集的树的深度设置为5，INRIA和ETH数据集的树的深度设置为2。在测试期间，我们只使用了排名前100的候窗口，并且由BF分类。

# 4 实验额分析
## 4.1 数据集
我们对4个基准进行了评估：Caltech[2]，INRIA[3]，ETH[4]和KITTI[5]。默认情况下，IoU阈值为0.5用于确定这些数据集中的True Positive。

在Caltech数据集上，训练数据集增加了10倍（42782幅图像）。在标准的测试集中的4024张图像用于评估“合理”设置下的原始注释（行人至少50像素高，至少65%可见）。评估度量是在[$$10^{-2},10^{0}$$]的log-average Miss Rate On False Positive Per Image(FPPI)（在[6]之后定义为$$MR_{-2}$$，或者in short MR）。此外，我们还在[6]提供的新的标注信息上测试了我们的模型，[6]改正了原始标注的错误信息。这个数据集定义为“Caltech-New”。
INRIA和ETH数据集经常用于验证模型的泛化能力。我们的模型在INRIA训练集中对614个正样本和1218个负样本进行了训练。

KITTI数据集由具有可用立体数据的图像组成。我们对左侧相机的7481张图像进行了训练，并对标准的7518张测试图像进行评估。KITTI在三个难度级别评估PASCAL风格的平均精度（mAP）：容易，中度和困难。

## 4.2 Ablating Experiments
#### Is RPN Good for Pedestrian Detection?


![Figure 3](\blog\images\post-covers\2017-05-29-paper03.png)
对Caltech集合的RPN和三个现有方法在目标候选区域质量（召回率和IoU）方面进行比较，评估了每个图像的平均1,4或者100个候选窗口。

重点！RPN作为一个独立的行人检测器实现了14.9%的MR，如表1所示。在Fig4中只有两种state-of-art方法超过这个性能。

**表1** 在Caltech数据集上对比不同的分类器和特征。所有的方法都基于VGG-16（包括R-CNN）。所有的条目都使用了相同的RPN候选区域。
![Figure 6](\blog\images\post-covers\2017-05-29-paper06.png)

![Figure 7](\blog\images\post-covers\2017-05-29-paper07.png)

#### How Important is Feature Resolution？
使用R-CNN（用VGG-16网络）方法实现了13.1的MR，略好于独立RPN检测器（14.9%的MR），它使用的窗口和上面提到的RPN是一样的。R-CNN从图像上剪切的目标候选区域，并且调整到224x224的尺度，因此它受小目标的影响比较小。这表明如果提取224x224精细的特征，下游的分类器可以提升精度。

然而同样在RPN提取的候选窗口上训练一个Fast R-CNN分类器，性能掉到了20.2%。尽管R-CNN在这个任务上工作很好，但是Fast R-CNN却产生了更糟糕的结果。

这个问题部分是因为低分辨率的特征。在Conv5上使用a trous trick，把stride从16减少到8个像素，这个问题得到了缓解，实现了16.2%的MR。这说明更高的分辨率是有帮助的。

**表2** 我们RPN+BF方法在Caltech数据集上不同特征的比较。所有条目均基于VGG16和同一组RPN候选窗口。
![Figure 4](\blog\images\post-covers\2017-05-29-paper04.png)

#### How Important Is Bootstrapping?
**表3** 在Caltech数据集上进行有无bootstrapping的比较
![Figure 5](\blog\images\post-covers\2017-05-29-paper05.png)

# 参考文献
[1] Faster R-CNN: towards real-time object detection with region proposal networks

[2] Pedestrian detection: an evaluation of the state of the art

[3] Histograms of oriented gradients for human detection

[4] Depth and appearance for mobile scene analysis

[5] Are we ready for autonomous driving? The kitti vision benchmark suite

[6] How far are we from solving pedestrian detection?