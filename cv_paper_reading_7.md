# Semi-supervised semantic segmentation
## 1.Guided Collaborative Training for Pixel-wise Semi-Supervised Learning(2020 eccv)
本文提出了一种适用于逐像素任务的SSL框架GCL，一致性训练，训练过程类似于GAN。

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/21.png)

T1和T2为任务模型（task specific），具有不同的初始化，输出T1(x)和T2(x)的目标为标注y；F为“缺陷探测器”，输入为x和Tk(x)连接起来，输出为H\*W的概率图，为Tk每个像素点（预测错误）是缺陷的概率；C为一个图片处理pipeline，膨胀+模糊+正则化。

- 训练的第一阶段：固定F，训练T1,T2。对于有标注的数据，用MSE（平方）损失有监督的训练。对无标注的数据，有两个约束来学习其中的知识。
  - Dynamic Consistency Constraint，类似于伪标签方法，集成两个任务模型的知识。设置一个阈值$\xi$，将F输出的缺陷概率图中概率大于$\xi$的像素点置1。再以T1为例，若F输出的缺陷概率图中，某像素T1缺陷概率大于T2，则以T2的值为“伪标签”，用MSE计算T1损失。T2同理。
  - Flaw Correction Constraint，希望Tk的输出使F的输出趋向于0。若某像素点在T1和T2中的缺陷概率都大于$\xi$，则用MSE以0为目标计算F输出的损失（F为固定的）。
- 训练的第二阶段：固定Tk，训练F。希望检测器F的输出趋向|Tk-y|，但后者通常稀疏且sharp，较难学习。因此，我们将|Tk-y|输入C，膨胀+模糊+正则化，将输出作为F的真值，用MSE训练。

## 2. Mean teachers are better role models:Weight-averaged consistency targets improve semi-supervised deep learning results(2018 nips)
本文是半监督学习的“一致性正则化”方法，根据聚类假设（数据分布由由低密度区域分隔的均匀类样本簇组成），如果对一个未标记的数据应用实际的扰动，则预测不应发生显著变化。

首先回顾两篇这类方法的文章。

- Temporal Ensembling for Semi-Supervised Learning(2017)：提出Pi模型。对于有标签数据进行监督学习；对于无标签数据，每次进行两次前向推理，由于数据增强的随机性和dropout，这两个结果肯定不同，使用MSE损失约束。随着训练进行，无标签数据的MSE损失占的权重减小。
![Alt text](%E5%8D%8A%E7%9B%91%E7%9D%A3%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/1.png)

- Temporal Ensembling for Semi-Supervised Learning(2017)：在 Pi-Model 的基础上进一步提出了Temporal Ensembling。只进行一次前向推理，将当前预测结果与历史预测结果的平均值做均方差计算，历史预测结果由EMA(exponential moving average，指数滑动平均)计算：$y'=\alpha y'+(1-\alpha)y$
- 
![Alt text](%E5%8D%8A%E7%9B%91%E7%9D%A3%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/2.png)

Mean Teachers则是Temporal Ensembling 的改进版，Temporal Ensembling 对模型的预测值进行 EMA，需要在训练时维护所有数据的EMA预测值，而Mean Teachers 采用了对模型权重进行 EMA，每个batch更新即可。
![Alt text](%E5%8D%8A%E7%9B%91%E7%9D%A3%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/3.png)
先进行监督学习，将模型参数复制为学生和老师模型。对于无标签数据，分别加入噪声后输入学生和老师模型，计算两者的MSE损失来训练学生模型，在一个batch训练完（反向传播完），使用学生模型权重计算老师模型权重的EMA并更新。

## 3.Semi-Supervised Semantic Image Segmentation with Self-correcting Networks(2020 cvpr)
本文面对的场景为：小部分语义mask标注的样本，大部分目标bbox标注。

![Alt text](%E5%8D%8A%E7%9B%91%E7%9D%A3%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/4.png)

- Ancillary Segmentation Model:输入为图片和bbox(实际上为三维张量H\*W\*(C+1)，在bbox内的像素点，对应的类别通道置1)，输出为分割掩码图。训练开始时使用全标注样本进行训练，之后便固定参数，在初期对弱监督样本精度较高。
- Primary segmentation model:主要模型，有标注样本监督训练，无标注样本以Self-correction module的输出为真值进行训练。
- Linear Self-Correction:以最小化输出分布与Ancillary Segmentation Model和Primary segmentation model的KL散度之（加权）和为目标，该分布有解析解。训练初期，Ancillary Segmentation Model占权重高，逐渐降低。
- Convolutional Self-Correction:使用卷积网络建模学习自校正，将前两者模型的logit输出叠起来作为输入。对于有标注的样本，监督训练Primary segmentation model和Convolutional Self-Correction，对于无标注样本，用Convolutional Self-Correction的输出作为真值算交叉熵损失（此损失不传播到Convolutional Self-Correction）。初始化时，用一半有标注数据训练Ancillary Segmentation Model防止精度太高，导致Convolutional Self-Correction只采用它的，留下更多的数据让Convolutional Self-Correction学会怎么结合Ancillary和primary

## 4.Semi-Supervised Semantic Segmentation with Cross-Consistency Training(2020 cvpr)
针对语义分割任务提出的半监督方法，一致性训练。本文认为，原始输入的分布没有表现出分隔类别的低密度区域，而隐藏层表示更符合聚类假设，适合进行一致性训练。

![Alt text](%E5%8D%8A%E7%9B%91%E7%9D%A3%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/5.png)

- 首先是一个共享编码器，顶层接一个main解码器和多个辅助解码器。损失由主解码器的监督损失和无标记样本损失组成，无标注样本经过共享编码器后，隐藏表示z直接输入主解码器，增加随机扰动后分别输入辅助解码器，计算主解码器结果和各辅助解码器结果的交叉熵损失。其中，主解码器只由有标注样本训练。
- 每个迭代采样1：1的有标注/无标注样本，为了避免过拟合有标注样本，使用了类似于OHEM的方法
- 本文提出了几种不同的，对隐藏表示加入扰动的方法
- 该框架还可以应用到弱监督任务，和多个域的任务（在共享编码器后，接特定于域的主解码器和辅助解码器，对应域的样本训练对应的解码器）。

## 5.Semi-supervised semantic segmentation needs strong, varied perturbations(2020 BMVC)
本文是将一致性正则化应用到语义分割的最早的几篇文章，将监督学习中的增强方法Cutout和CutMix应用到SSL作为扰动。

- 本文的大部分篇幅在理论分析，与上一篇的思想类似，认为“原始输入的分布没有表现出分隔类别的低密度区域”，一般的一致性正则化方法不适用于分割任务。但是我没大看懂，分析了一番扰动/决策边界之类的。
- CutOut大概是，随机选输入图片里的矩形区域，置为0；CutMix是，找两张输入图片，在一张图片里取矩形掩码，和另一张图的其余部分拼在一起，作为输入，计算损失时的GT也要同样拼接。
- 本文中，使用了Mean Teacher的师生框架，将CutMix后的图片作为学生网络的输入，将原图片在老师网络的输出为真值，拼接后作为学生网络的伪标签。

## 6.ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning(2021 WACV)

依然是一致性正则化方法，本文提出了一种新的针对无标记样本的数据增强方法Classmix。

![Alt text](%E5%8D%8A%E7%9B%91%E7%9D%A3%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/6.png)

- Classmix：随机采样两张无标签输入图片A,B，对A的预测概率图取argmax后，随机取一般类别，将这些类别对应的原图像素点cut下来，粘到B上，得到增强后的图片，增强后图片的GT由A,B的预测概率图取argmax后拼接而成。
- 本文使用了Mean teacher框架，对教师网络的权重总用EMA更新。对于A,B先用教师网络推理，用增强后的图片训练并更新学生网络的参数。
- 伪标签的思想，对A,B的预测概率图取argmax，鼓励网络执行置信的预测，消减边界的不确定性，锐化，减小污染。
- 每次训练取一半有标记，损失函数由监督部分和增强部分组成，增强部分的损失占的权重由初期的很小慢慢变大。

## 7.Semi-Supervised Semantic Segmentation via Adaptive Equalization Learning (2021 NIPS)
本文针对SSL语义分割中表现不佳的特定类别，提出了adaptive equalization learning (AEL)。经典的伪标签/一致性正则化方法预测不准确，甚至可能损害这些类别的表现。

![Alt text](%E5%8D%8A%E7%9B%91%E7%9D%A3%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/7.png)

- 本文使用了Mean teacher框架，提出了两种数据增强方法，来增加表现不佳样本在训练批次中出现的频率。教师模型给出无标签样本的伪标签，并由学生模型的参数计算EMA更新；学生模型给出样本的预测，其中有标记样本的预测用来计算类别的置信度，并指导无标记样本的损失计算。
- Confidence Bank:在训练过程中，通过有标记样本的表现来维护每个类别的置信度。文中提出了多个指标，最终采用了$Conf^ {c} =  \frac {1}{N_ {l}}  \sum _ {i=1}^ {Nl} \frac {1}{N_ {i}^ {c}} \sum _ {i=1}^ {Nc}p_ {ij}^ {c} ,c \in {1,\cdots ,C}$，并且在每次训练后计算EMA更新每个类的置信度。
- Adaptive CutMix:这是针对无标注数据的数据增强方法，与原CutMix的区别在于，两张图片将按据类别置信度来采样。
- Adaptive Copy-Paste:这是针对有标注数据的数据增强方法，按照置信度计算概率，采样类别，复制源图像中属于采样类别的所有像素并将它们粘贴到目标图像上。
- Adaptive Equalization Sampling:根据置信度计算一个类别的采样率，不是使用所有像素来计算无监督损失，而是根据像素的预测对像素随机采样一个子集。
- Dynamic Re-Weighting:伪标签的质量有显著影响，本文为无标签样本的每个像素点计算权重，使置信度（此处为softmax后的最大值）高的像素点的损失具有更高的权重。


## 8.Semi-supervised semantic segmentation with cross pseudo supervision(2021 cvpr)
在论文里看是很简单的一篇文章，一致性约束方法，但能实现SOTA，也许具体训练时有些trick。

- 两个网络（可以相同或不同），随机初始化。对于有标记样本监督训练；对于无标记样本，同时输入两个网络（用相同的增强方法），将网络输出的分割置信图转化为one-hot，再作为另一张图片伪标签计算损失并反向传播。
- 文中和其他方法进行了对比讨论，我感觉大家trick都差不多，但这篇既然是SOTA，说明有其价值。
- 在初期互相伪监督训练不会越来越错吗？即使两个网络对某个像素的分类都是错误的，也极大概率是两个（错误的）类别，互相有些抵消折中，而非越训越错。实验表明，结果是逐渐提升的。

