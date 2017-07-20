---
title: Deep Learning of Scene-specific Classifier for Pedestrian Detection
layout: post
tags: [Deep Learning]
---

这篇论文被收录在ECCV-2014

# 摘要

检测器的性能非常依靠他的训练集合，并且当这个检测器应用到一个新的场景中性能会明显下降，这是由于在训练数据集域和目标场景有较大的差异。为了弥合这种外观的差距，我们提出了一个深度模型去自动学习静态监控视频监控中的场景特征和视觉模式，而无需目标场景中的任何手工标签。它共同学习了一个特定场景分类器和目标样本的分布。这两个任务共享了具有判别性和标示性能力的多尺度特征表示。我们还在深度模型中提出一个利用特定场景视觉模式图案进行行人检测的聚类层。我们专门设计了目标函数，不仅包含目标训练样本的置信度分数，而且通过拟合目标样本的边际分布，自动加权源训练样本的重要性。与最好的域适应（domain adaptation）方法相比，该方法在MIT Traffic Dataset和CUHK Square Dataset上显著提高了检测率， 在1 FPPI提高了10%。

# 1. Introduction

在INRIA行人数据集上训练的模型的性能在MIT Traffic数据集上下降很明显，由于源域和目标域样本外观的差异，所以在一个数据集下训练的模型应用到另一个场景中很难有满意的性能。当今，大量的监控摄像机被使用，所以在所有特定场景中手工标定样本的方法是不切实际的。另一方面，像视频监控这一类应用，一个摄像头捕获的样本外观差异很小。因此，通过从一般数据集中传递知识来训练特定场景的检测器，改进特定场景的检测性能的方法是实际的。在特定场景检测器方面已经有了很多工作，它们的训练过程由通用检测器来辅助，用于自动收集来自目标场景的训练样本，而无需手动标记它们。

学习一个特定场景的检测器被认为是一个域适应（domain adaptation）问题。它涉及两种不同类型的数据：来自源数据集的$$\mathbf x_s$$和来自目标场景的$$\mathbf x_t$$，它们有着非常不同的分布$$p_s(\mathbf x_s)$$和$$p_t(\mathbf x_t)$$。源数据集包含大量的标记数据，然而目标场景没有或者有少量的具有标记的训练数据。目标是使源数据集训练的分类器适应到目标场景，例如使用函数$$y_t=f(\mathbf x_t)$$来从$$\mathbf x_t$$估计标签$$y_t$$。作为一个重要的预处理，我们可以从$$\mathbf x_t$$中提取特征，并且有$$y_t=f(\phi(\mathbf x_t))$$，这里$$\phi(\mathbf x_t)$$是提取的特征，像HOG或者SIFT。我们也预期边缘分布$$p_s(\phi(\mathbf x_s))$$和$$p_t(\phi(\mathbf x_t))$$非常不同。我们研究针对特定场景检测器的深度模型有以下三个动机。

- 首先，代替仅仅适应性地调整通用手工特征的权重这些现有的域适应方法，希望自动学习特定场景的特征以最佳捕获目标场景的判别信息。这可以通过深度学习很好地实现。
- 其次，学习$$p_t(\phi(\mathbf x_t))$$很重要，当$$\phi(\mathbf x_t)$$的维度很高是，这是具有挑战性的，而深度模型可以以分层和无监督的方式很好的学习$$p_t(\phi(\mathbf x_t))$$。1）在有标记的目标训练样本数目很小的情况下，联合学习$$p_t(\phi(\mathbf x_t))$$和$$f(\phi(\mathbf x_t))$$的特征表示是有益的，可以避免$$f(\phi(\mathbf x_t))$$的过拟合，因为规则中加入了$$p_t(\phi(\mathbf x_t))$$。2）$$p_t(\phi(\mathbf x_t))$$也有助于在学习特定场景分类器中评估源样本的重要性。一些源样本不会出现在目标场景中，并且可能导致误导的训练。它们的影响应该减少。
- 第三，目标场景具有正确和错误的特定场景视觉模式，它们会重复出现。例如，true positives 和 false negatives 有着相似的模式，因为在一个特定场景的行人有相同的视角、移动模式、姿态、北京和行人尺寸。因此，特定地学习波或这些模式是理想的。

这些观察激励着我们开发同意的深度模型，学习特定场景的视觉特性、视觉特性的分布和重复的视觉模式。我们的贡献总结如下：

- 深度模型学习了多尺度特定场景的特征。
- 深度模型完成了分类和重建的任务，判别能力和表示能力共享了特定表示。由于目标训练样本用上线文信息自动的被选择和标记，所以分类的目标函数对目标训练样本的置信度得分进行编码，从而使得学习的深度模型对于标注目标训练样本的错误是鲁棒的。同时，自动编码器重建目标训练样本，并对目标场景中的样本的分布进行建模。
- 通过我们专门设计的目标函数，训练样本对学习器的影响由其出现在目标数据中的概率加权。
- 在深度模型中提出一个新的聚类层，以捕获场景特定的模式。在这些模式上的样本的分布被作为检测的附加特征。

我们的创新来自于视觉问题的观点，我们将其纳入深度模型。于最新的域适应结果相比，我们的深度学习方法在两个公开数据集上1 FPPI(每幅图片的False Positive)的检测率显著提高了10%。

# 2. Related Work

许多通用的人检测方法使用深度模型或者part based models学习特征、聚类外观混合、变形和可见性。他们假设源样本的分布和目标域样本的分布是相似的。我们的主要贡献目的是解决域适应问题，其中两个域中的数据分布明显不同，标记的目标训练样本很少或者包含错误标记样本。

许多域适应方法[1][2]学习了源域和目标域所共享的特征。他们将手工设计的特征映射到子空间或者流形中，而不是在源数据中学习特征。在无监督和迁移学习挑战和学习分层模型的挑战中，研究了一些深层模型。已经证明使用深度模型的迁移学习在这些挑战、动物和车辆识别以及情感分析中都是有效的。我们收到这些工作的启发。然而，我们专注于在不同的域共享非监督学习的特征，并且使用和现有的一般深度模型相同的结构和目标函数。我们在这两个方面都有所创新。

一组关于特定场景检测器的工作[3-7]构建了自动标注器，用于自动从目标场景中获取可靠的样本，以又重新训练通用的检测器。Wang等人[7]演技了丰富的上线文线索，以获得可靠的目标场景的样本，预测它们的标签和置信度得分。它们对分类器的训练包含了置信度分数，并且对于标记错误是鲁棒的。我们的方法属于这个组。通过这些方法获得的可靠的样本可以用作我们学习深度模型方法的输入。另一组工作[8][9]属于协同训练（co-training）框架，其中两个不同的特征集合的两个不同的分类器同时被训练用于相同的任务。[7]的实验比较表明，训练行人检测器时，协同学练容易漂移（drift），其性能远低于[7]中提出的自适应检测器。

源数据集和目标数据集中的样本使用SVM和Boosting进行不同的重新加权。然而这些方法是启发式的，但不学习目标域数据的分布。我们的方法通过深度模型学习目标样本的分布，并将其用于重新加权样本。


# 3. The proposed deep model at the testing stage

![Figure 1](\blog\images\post-covers\2017-06-15-paper02.png)
<center>图1 深度模型的概述</center >

模型的训练阶段如图1和图2所示，它实现了分类和重建两个任务，并且任务的输入是来自于源域和目标域的训练样本。然而，在测试阶段，我们只保留部分样本用于分类，并将目标样本作为输入。提出的目标场景行人检测深度模型概述如图3所示。该模型包含三个卷积层、三个全连接层、提出的聚类层和分类标签y，标示窗口中是否包含行人。

- 
- 

![Figure 2](\blog\images\post-covers\2017-06-15-paper03.png)
<center>图2 在训练阶段的模型的结构</center >


![Figure 3](\blog\images\post-covers\2017-06-15-paper01.png)
<center>图3 在测试阶段的模型</center >

4 Training the deep model

4.1 Multi-stage learning of the deep model

深度模型学习阶段的概述如图1所示。它包括以下几个步骤：

1. **获取可信的目标训练样本。**可以使用任何现有的方法从目标场景收集可信的正训练样本和负训练样本。该论文中采用[7]的方法。它以源训练数据集（INRIA数据集）训练一个一般检测器开始，并通过附加的上下文信息（如运动、路径模型和行人尺寸）自动标记来自目标场景的训练样本。由于自动标注包含错误，所以一个分数表示与每一个训练样本相关的预测标签的置信度。目标训练样本和源训练样本都被用于重新训练特定场景检测器。
2. **特征提取。**使用目标和源训练样本，如图3中的三个CNN层用于学习行人检测任务的判别信息。
3. **分布建模。**目标场景中特征的分布是通过使用深层置信网络（deep belief net）来学习的。
4. **特定场景模式学习。**深度模型中的聚类层学习以用来捕获特定场景的视觉模式。
5. **联合学习分类和重建。**由于目标训练样本容易出错，因此要使用源样本来改善训练。目标训练样本的目标函数中的分类估计误差由其置信度得分加权，为了对标记的错误鲁棒。除了要学习特定场景分类器的判别信息之外，还包括自动编码器，以便深度模型可以重建特征时学习表示信息。使用新的目标函数，重建误差用于对训练样本加权。更好地拟合目标场景分布的样本有着更小的重建误差，并且对目标函数有着更大的影响。在这个阶段，（2）-（4）阶段预训练的参数也使用反向传播联合优化。


![Figure 4](\blog\images\post-covers\2017-06-15-paper04.png)
<center>图4 算法流程图</center >


# 参考文献
[1] Gong, B., Shi, Y., Sha, F., Grauman, K.: Geodesic flow kernel for unsupervised domain adaptation. In: CVPR (2012)

[2] Gopalan, R., Li, R., Chellappa, R.: Domain adaptation for object recognition: An unsupervised approach. In: ICCV (2011)

[3] Nair, V., Clark, J.J.: An unsupervised, online learning framework for moving object detection. In: CVPR (2004)

[4] Rosenberg, C., Hebert, M., Schneiderman, H.: Semi-supervised self-training of ob- ject detection models. In: WACV (2005)

[5] Wang, X., Hua, G., Han, T.X.: Detection by detections: Non-parametric detector adaptation for a video. In: CVPR (2012)

[6] Wang, M., Wang, X.: Automatic adaptation of a generic pedestrian detector to a specific traffic scene. In: CVPR (2011)

[7] Wang, X., Wang, M., Li, W.: Scene-specific pedestrian detection for static video surveillance. TPAMI 36, 361–374 (2014)

[8] Wu, B., Nevatia, R.: Improving part based object detection by unsupervised, online boosting. In: CVPR (2007)

[9] Levin, A., Viola, P., Freund, Y.: Unsupervised improvement of visual detectors using cotraining. In: ICCV (2003)

