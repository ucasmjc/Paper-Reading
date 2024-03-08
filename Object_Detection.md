
- [Object Detection](#object-detection)
  - [1.R-CNN:Rich feature hierarchies for accurate object detection and semantic segmentation(2014)](#1r-cnnrich-feature-hierarchies-for-accurate-object-detection-and-semantic-segmentation2014)
  - [2.SPP-Net:Spatial pyramid pooling in deep convolutional networks for visual recognition(2015)](#2spp-netspatial-pyramid-pooling-in-deep-convolutional-networks-for-visual-recognition2015)
  - [3.Fast R-CNN(2015)](#3fast-r-cnn2015)
  - [4.Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks(2015)](#4faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks2015)
  - [5.OHEM:Training region-based object detectors with online hard example mining(2016)](#5ohemtraining-region-based-object-detectors-with-online-hard-example-mining2016)
  - [6.Yolo v1:You only look once: Unified, real-time object detection(2016)](#6yolo-v1you-only-look-once-unified-real-time-object-detection2016)
  - [7.SSD: Single Shot MultiBox Detector(2016)](#7ssd-single-shot-multibox-detector2016)
  - [8.R-FCN: Object Detection via Region-based Fully Convolutional Networks(2016)](#8r-fcn-object-detection-via-region-based-fully-convolutional-networks2016)
  - [9.YOLO9000:Better, Faster, Stronger(2017)](#9yolo9000better-faster-stronger2017)
  - [10.FPN:Feature pyramid networks for object detection(2017)](#10fpnfeature-pyramid-networks-for-object-detection2017)
  - [11.RetinaNet:Focal loss for dense object detection(2017)](#11retinanetfocal-loss-for-dense-object-detection2017)
  - [12.Mask r-cnn(2017)](#12mask-r-cnn2017)
  - [13.Yolov3: An incremental improvement(2018)](#13yolov3-an-incremental-improvement2018)
  - [14.DERT:End-to-end object detection with transformers(2020)](#14dertend-to-end-object-detection-with-transformers2020)

  
# Object Detection
## 1.R-CNN:Rich feature hierarchies for accurate object detection and semantic segmentation(2014)

[论文链接](https://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)



本文提出R-CNN方法，将CNN方法引入目标检测领域，大大提高了目标检测效果，以此为开始引出一系列Two-Stage Object Detection工作。

- 整个训练过程可大致分为四（三）个阶段：将CNN网络在辅助大数据集上进行有监督（image-level）的预训练（或直接使用AlexNet的参数）；将原CNN网络的全连接层换为特定于目标检测任务的类别数（N+1），并输入region proposals（随机采样）进行微调（此时正例和负例的判定限制较松），使用softmax后计算损失；输入region proposals并通过CNN输出的特征（N+1维）训练SVM，输出每个类别的分数，此过程中正例和负例的判定要求严格（直接使用CNN的输出softmax误差较大）；将最后一个卷积层输出的特征训练bounding-box regression
- 测试时，对每张输入图片提取2k左右region proposals；将region proposals缩放为特定大小并输入CNN网络提取特征；将提取出的特征输入SVM进行分类打分和NMS，再将最后一个卷积层输出的特征进行bounding-box regression，对边界框进行微调
- 在本文提出的时代，目标检测的有标记数据少，过去往往通过无监督方法预训练，本文提出在其他数据集上进行有监督的预训练（使用辅助任务，如分类）再进行特定任务的微调更有利于训练大容量CNN（在特定任务数据稀少时）
- 本文使用了selective search方法提取region proposals，大致是先将图片过分割，再通过一些启发式的规则进行合并。通过该方法得到的region proposals尺寸不能直接作为CNN的输入，本文采取了先padding上下文再各向异缩放的方法。
- 本文使用CNN网络（AlexNet）进行特征提取，过去常用人工设定的特征，如SIFT,HOG
> SIFT和HOG未了解过，之后读原论文了解一下
- 根据设定好的交并比阈值区分正例和负例来对SVM训练，训练时每个batch的正例和负例的比例确定
- bbox回归希望对region proposals进行微调（中心x,y坐标和宽、高），具体细节见论文附录，主要训练了四个线性层作为平移/缩放因子，将transform后的bbox与gt做回归
> 提到了DPM，未了解过，不过似乎和bbox回归的思想差不多，且在DL时代“过时”了，便不读原论文了


## 2.SPP-Net:Spatial pyramid pooling in deep convolutional networks for visual recognition(2015)
[论文链接](https://arxiv.org/pdf/1406.4729.pdf)

本文将SPM方法(Spatial pyramid matching)应用到CNN中，提出了SPP-Net，在与R-CNN效果差不多的同时大大加速
- SPP(Spatial pyramid pooling)，将输入划分为不同尺寸的块（已设定好的数量），对每个块分别池化
- SPP的优势：将不同尺寸的输入转化为固定长度的输出，且保留空间特征；可以使用不同尺寸的图片进行训练，对目标变形更有鲁棒性；可以在多个尺度（块的尺寸）提取特征
- 每张图片只需用CNN提取一次特征，将原图的region proposals对应到特征图，即可获得其CNN特征，s通过SPP池化，将得到的不同尺寸的represention拼接起来，即可获得相同尺寸的特征向量，再送入FC分类
- SPP-Net对每张图片只提取一次特征，大大提高了效率，但训练阶段和R-CNN一样复杂，在Fast R-CNN得到解决


## 3.Fast R-CNN(2015)
[论文链接](https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)

Fast R-CNN比R-CNN更快，更准确

- 训练过程：将一张图片和RoIs（也是通过selective search）输入CNN（将最后一个池化层换为Roi池化层），在卷积层输出的特征图上找到原图Roi对应的Roi（按CNN下采样的比例缩放），将特征图的Roi池化后（统一尺寸）输入全连接层，分别得到softmax分类和bbox回归的偏移量，将二者的共同损失同时反向传播。(计算回归损失时使用了smooth L1 loss)
- 更快的原因：Fast R-CNN利用分层采样，对每张图片只需要前向传递CNN一次便可提取多个Rio对应的特征；使用多任务损失，同时更新整个网络的参数，无需像R-CNN一样分多个阶段
- Roi池化层，将特征图大小不同的Roi分块最大池化统一尺寸，文中亦讨论了其反向传播
- Fast R-CNN使用了VGG主干
- 推理时全连接层耗时很长，利用奇异值分解加速（分解成两个小的全连接层）


## 4.Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks(2015)
[论文链接](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)

基于Region的CNN在目标检测取得了很好的效果，但传统Region proposals方法（如selective search）成为推理时的时间瓶颈，本文提出RPNs(Region Proposals Networks)，在和检测网络（如Fast R-CNN）共享CNN特征的基础上，加入少量额外的卷积层实现生成regions proposals，更快，效果更好
- 将一张H\*W图片输入VGG，得到256维H/16\*W/16的特征图。每个点生成k(=9)个锚框，将特征图输入RPN，经过3\*3卷积层（输出仍为256维），分别经过两个1\*1卷积层，输出为2k维H/16\*W/16（softmax）和4k维H/16\*W/16，分别储存了该锚框包含/不包含对象的概率、锚框位置的修正量，经过一个整合层（还会丢弃许多锚框）即可输出region propsals，这即为RPN。RPN输出的proposals经过Roi池化输入分类器
- 对特征图的每个点选择生成k个锚框（3种大小和比例），且RPN相当于k个分类器和回归器。通过这种方式，网络考虑不同尺寸的bbox
- RPN和Fast R-CNN共享特征，在训练时采用了4-Step Alternating Training方式，损失函数和Fast R-CNN类似


## 5.OHEM:Training region-based object detectors with online hard example mining(2016)
[论文链接](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)

在基于region的目标检测器中，前景和背景样例比例不平衡是一个挑战，但这不是一个新的挑战，在过去20年bootstrapping方法（hard negative mining）在传统目标检测任务中（如SVM）被使用。本文提出Online Hard Example Mining，提出boottrapping方法在线学习的形式，从而应用到CNN网络中，取得了很好的效果
- bootstrapping的思想大致为：训练时有两个交替进行的阶段，使用fixed model寻找hard example（假正例，或违反当前分类边界）并加入到active训练集中（删去分类正确的example）；使用fixed训练集训练model。
- 这样的bootstrapping不适用于SGD更新（在线学习）的深度卷积网络，会大大减慢速度，OHEM便是bootstrapping的在线学习形式
- OHEM的思想为：将一张图片输入CNN提取特征，计算所有RoI的损失，按照损失NMS后，选择损失最高的样例作为mini_batch进行反向传播（其他样例的损失置0）
- 当年的深度学习tool有限制，即使损失为0依然会占用内存反向传播，故本文提出一种较复杂的实现方式，加入了一个只读网络

## 6.Yolo v1:You only look once: Unified, real-time object detection(2016)
[论文链接](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)

与R-CNN系列不同，Yolo是one-stage目标检测，不会专门预测region proposals。本文提出Yolo v1，可以实现实时检测，但精度尚达不到sota，且对小目标误差较大
- 将image分为S\*S个grid，每个grid生成B个bbox，每个bbox有五个参数（中心x，中心y，长，宽，置信度）。其中中心x和y坐标除以grid的长宽来归一化，长和宽通过除以image的长和宽来归一化。
- 置信度的计算：ground truth的置信度为P(该bbox所属的grid里有对象)*IoU。在训练时，将目标中心落在的grid看作有对象，P(该bbox所属的grid里有对象)=1，否则为0
- 训练时：包括四部分损失，中心点坐标损失（对于负责预测的bbox）+长宽损失（对于负责预测的bbox，且取平方根）+置信度误差（分别对于负责预测和无对象的grid，权重不同）+对象分类误差（对于有对象的grid）。对于每个grid，在训练时只选与ground truth IoU最大的bbox来responsible for预测，其余都算无对象
- 网络主体与GoogleNet类似，每张图片经过卷积层和全连接层后的的输出为S\*S\*(B\*5+Class)形状的张量，直接预测每个bbox的五个参数，并预测每个grid含每个类别的概率，预测时对每个类通过NMS处理
- 计算长和宽的损失时取平方根，使小尺寸的bbox对长和宽的变化更敏感
- 预训练CNN时用224\*224的图片作为输入，检测时用448\*448更高分辨率的图片

## 7.SSD: Single Shot MultiBox Detector(2016)
[论文链接](https://arxiv.org/pdf/1512.02325.pdf%22source%22)

SSD也是一个one-stage目标检测方法，在不同尺度的特征图上生成先验框进行回归和分类
- 主要思想：将主干网络VGG的FC换为多个卷积层，选取多个（6个）卷积特征图，对每个cell生成k个（4-6）尺度和比例不同的默认框（即锚框），使用相互独立的卷积核（3*3）分别对不同特征图（设为m\*n维）上的bbox进行分类（N+1类的置信度）和回归（offset），输出维度为(m\*n\*(k\*(classes+4)))，将得到的所有输出（8k+预测框）进行NMS
- SSD采用多尺度特征图进行检测，可以识别不同尺寸的目标（有些方法是把图片处理成不同的大小，然后结合不同大小图片的结果），因为随卷积网络加深，特征图的维度下降，感受野增大，更low的特征图感受野小，有利于识别小尺寸目标。
- 和Yolo的主要区别：1.利用了不同尺度的特征图；2.使用了先验框 3.利用卷积核进行预测而非FC
- 训练时：利用文中提到的匹配策略确定正例，再对负样本进行hard negative mining采样，损失函数与Faster R-CNN类似
- 随特征图所属层数加深，默认框的尺寸增大，默认框可按比例对应到原图
- SSD还做了数据增强加强鲁棒性


## 8.R-FCN: Object Detection via Region-based Fully Convolutional Networks(2016)
[论文链接](https://proceedings.neurips.cc/paper/2016/file/577ef1154f3240ad5b9b413aa7346a1e-Paper.pdf)

ResNet在图片分类任务大放异彩，本文想将FCN应用到目标检测任务，主要提出了一个既打破FCN平移不变性，又减少Rio-wise layer从而提速的方法，另外，R-FCN是应用了RPN的two-stage方法
- 背景：图像分类最新的sota网络如Resnet等使用了全卷积，因此自然地想将其应用到目标检测。但直接应用效果不好，因为卷积**translation invariance**，对平移不敏感，这有利于分类任务，而检测任务对对象的translation是敏感的。开始，作者试着将位置敏感的Roi pooling加入到卷积层之间，打破**translation invariance**，但这样会引入unshared Rio-wise layers，使计算效率变低，因此，提出了R-FCN
- R-FCN引入position-sensitive score maps使FCN对translation敏感，并且除了最后的position-sensitive Roi pooling（没有参数，很快），所有层都是shared，一张图只需前向传递一次，大大加速
- 在主干CNN提取出特征图后，一方面输入RPN生成Roi，另一方面利用3*3卷积，生成与特征图大小一致的$k^2(C+1)$个score map，即每个类有$k^2$张，每一张分别对应于一个grid（相当于将Roi分为k\*k个grid）的得分分布。对于一个Roi，在score map中找到对应位置，对于每个类的$k^2$张特征图，分别其对应grid位置的score grid，拼成一个Rio score(与Roi形状相同)，共C+1个(**原论文图示中的颜色很重要！**)。最后，通过score投票，在文中简单的将Rio score加起来得到Rio的score，共得到C+1个score，用softmax可得分类概率。
- bbox回归与分类类似，区别在于score map有$4k^2$个，与之前的类似
- 训练时使用了OHEM，预测时使用了NMS

## 9.YOLO9000:Better, Faster, Stronger(2017)
[论文链接](https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf)

yolo v2在yolo的基础上加入了一些其他工作的先进idea和新idea来提高性能和速度，yolo9000在v2的基础上利用分类数据集联合训练，使模型可以识别对象的类别（即使不在检测数据集中）
- batch normalization，省去其他正则化和dropout
- 在224\*224的分类集上预训练后，再用448\*448的图像对分类任务微调，最后用448\*448的图片训练检测任务，从而使预训练的模型更能适应高分辨率的图像
- 使用锚框并预测offset，使用k-means聚类来寻找锚框的最佳先验尺寸&比例
- 锚框预测offset的式子有所变化，将锚框中心限制在所属grid内
- 使用细粒度特征（多个特征图，如SSD）。top特征图为13\*13，加入了一个paththrough层，让26\*26的特征图拆成13\*13并和top的特征图叠起来，共同作为提取的特征
- 因为Yolo v2只使用了卷积和池化层，所以训练时使用了不同尺寸的图片以提高鲁棒性
- 目标检测的数据集远小于分类，因此本文提出了一个使用两种数据集联合训练的方法。对于带检测标签的数据正常求损失，对分类标签的数据只求分类损失，两种数据按一定比例采样。
- 通过建立WordTree统一两种数据集的标签（根据WordNet确定不同标签的从属关系），对于树中某个节点（标签）的绝对概率，通过将其自身和到root的所有父类的条件概率相乘得到。每个结点的条件概率即给定父类的条件下的概率，通过对父节点所有子节点做softmax得到。因此，在训练时，对所有同义词（属于同一父类）做multiple softmax来获得每个结点的条件概率，再对计算得到的绝对概率求loss，因此每张图片会更新GT label和其所有父类对应的梯度；在测试时，从root开始，每次选择条件概率最大的子节点，相乘直到绝对概率**小于**某个阈值。
  


## 10.FPN:Feature pyramid networks for object detection(2017)
[论文链接](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)

本文提出了FPN(Feature Pyramid Network)，利用在利用多尺度特征图的同时为高分辨率的特征图加入语义信息，获得了更快更好的结果
- 背景：为了识别不同尺寸的目标，传统方法是通过输入不同尺寸的图片，但时间和内存消耗太大。近来，如SSD使用不同尺度的特征图进行预测，实现不同分辨率的预测，但一方面，SSD为避免使用low的特征图（可能是因为包含的语义特征太少了，或者说太局部了），从第四个卷积层开始用于预测，这使得其对小目标检测不理想，因为lower层的分辨率更高，对小目标检测很重要；另一方面即使从第四层开始，相比top层，分辨率高的层的语义信息更少，不利于检测小目标，且层与层之间有语义gap。为了建立在多个尺度都具有丰富语义特征的特征图金字塔，且快速，本文提出了FPN
- Bottom-up pathway:自底向上的通路即在CNN前向传递期间生成的，尺度不断减小的特征图（选取每次下采样前的特征图），除了第一层（因为太大了占用内存）。这些特征图，越top分辨率越低，语义信息越丰富
- Top-down pathway and lateral connections：在Bottom-up pathway的top卷积层经过1\*1卷积生成特征金字塔的最top层，设置为d维，再自顶向下逐渐生成每个特征图（尺寸越来越大），生成的特征金字塔每层的形状均与Bottom-up pathway的相同（维度可能不同）。生成方式为：对toper层进行上采样（本文简单的使用最近邻），生成与lower层尺寸相同的d维特征图，再对Bottom-up pathway中相应尺寸的特征图进行1\*1卷积，生成d维特征图再与上采样后的d维特征图直接相加，最后加一个3\*3卷积生成最终的特征图
- 应用于RPN：对CNN主干应用RPN，生成特征金字塔，对每个尺度的特征图分别生成锚框（每种特征图对应一个尺度和多个比例），再接入RPN网络预测。**对于不同尺度的特征图，应用共享参数的卷积层进行预测和分类即可**，原因大抵为不同特征图共享语义
- 应用于Fast F-CNN：按照Roi的尺寸将其分给不同尺度的特征图负责预测，从而使更小的Roi分给更高分辨率；预测时，对每个特征图直接应用Roi pooling，后接两个非线性层进行预测和分类，且不同特征图预测头的参数共享

## 11.RetinaNet:Focal loss for dense object detection(2017)
[论文链接](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

提出了一种新的分类损失函数Focal Loss，使hard example的影响增大，解决one-stage方法中正负例比例严重不均的问题，提出的 RetinaNet在保持one stage模型精度的同时超过了two stage模型的精度
- 背景：one stage模型具有更快和更简单的潜力，但精度不如two stage，作者认为主要原因是one stage的前景与背景样本比例严重失衡（有许多简单的背景样本，而two stage已通过RPN选出Roi），而简单的背景样本（预测分类概率很接近1）在标准交叉熵损失下仍有不可忽略的损失，因one stage中简单的负例很多，会导致模型学习效率差，且使模型退化。本文通过修改标准交叉熵损失，解决了正负例不均衡问题，实现了比过去启发式采样、OHEM等方法更好的结果
- Focal loss:简单来说，从标准交叉熵-lnx，转为$-\alpha(1-p)^\gamma log(p)$，使p趋向于0时（hard example），损失更大，而p趋向1时，损失更小，从而降低简单负例的影响
- RetinaNet：one stage，使用了FPN作为主干网络，使用锚框回归（训练时match正负例的阈值都宽松了）
- 值得注意的一点是，在RetinaNet初始化时，通过对最后一个全连接层的偏置b设置，使训练开始时，每个锚框被模型标为前景的概率为$\pi$(0.01)，从而使背景为简单样本，避免大量背景样本的巨大不稳定影响。（默认初始化的话，前景和背景的概率差不多都是0.5）

## 12.Mask r-cnn(2017)
[论文链接](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)

mask R-CNN在实现目标检测的同时加入实例语义分割任务，并且提出Roi Asign方法，实现了更好的结果
- 网络结构的基础为以Resnet+FPN的Faster R-CNN。在提取特征图后，一方面输入RPN获得Roi，另一方面使用Roi Asign对Roi对齐到固定尺寸的m\*m特征图。在原有bbox回归和分类的基础上，加入了语义分割分支，使用FCN对m\*m特征图计算mask（对每个像素计算C个类的sigmod，未使用softmax以避免类间的竞争），后者损失为平均二元交叉熵
- 因Roi pooling对Roi的pixel-pixel对齐不好，过程中进行了两次取整操作，而Roi Asign可保留浮点数，利用双线性插值避免了取整引入的误差，大大提高了精度
- 在训练时，对于语义分割分支，只对每个mask的GT类mask值计算损失
- 预测时，先通过检测分支预测出最高分的k个Roi，再对它们进行mask分支，将得到的m\*m\*C mask取检测分支中预测的类别的得分（维度为m\*m），再resize成Roi size，以0.5阈值二值化为语义分割输出
> 对如何从m\*m维度mask resize成Roi原尺度存疑，似乎是语义分割中的dense predict方法



## 13.Yolov3: An incremental improvement(2018)
[论文链接](https://arxiv.org/pdf/1804.02767.pdf)

这不是一篇正式的论文，尽管挂在了arxiv上，而且引用量接近两万，似乎是一个技术报告。本文讲述了Yolo v3做出的更新，和一些实验结果。
- 在预测bbox，多预测了一项该bbox有目标的概率（得分）
- 在分类任务中，不再使用yolo v2中的multi softmax，而是用了独立的逻辑分类器，训练时使用了二分类交叉损失熵
- 借鉴了FPN和残差连接
- Focal loss无用，猜测是因为预测bbox的有目标的概率，起到了相同的效果，使许多简单背景样本不产生损失

## 14.DERT:End-to-end object detection with transformers(2020)
[论文链接](https://arxiv.org/pdf/2005.12872.pdf,)

本文通过二元匹配和transformer编码解码器结构实现DERT结构，大大简化了检测任务的pipeline，简化了许多人工设计（NMS,锚框），实现了集合预测（一次预测出所有对象的类别和位置）
- 结构（附录里的细节图片非常非常清晰！）：先用CNN提取特征，与固定的位置编码相加后输入transformer encoder;向decoder输入一组可训练的object query(应该是N个)，在cross-attention部分应用了encoder的输出，得到N个预测（N是一个固定值，明显大于一个图中可能的目标数量）；将N个预测输入FFN获得bbox预测和分类概率，包括"no object"类
- loss:
  - 训练时先将N个预测与GT做二元匹配（将GT用no object填充成N元组），即寻找一个排列，使预测和GT pair-wise的匹配成本加起来最低。$y_i$与$y_{pre-i}$匹配成本为：若$y_i$的类$c_i$为无对象则0，其余为$-p_{pre-i}$($c_i$)+二者bbox的损失（下述）
  - 选出排列后，按预测与GT的二元匹配，计算loss，其中分类loss用-log，bbox损失使用L1 loss和generalized IoU loss（附录中有详细说明）的加权（避免只用L1 loss对尺寸敏感）
- 辅助loss：每个decoder块后都会接上FFN进行预测并计算损失，在输入FFN前加了一个不同层之间shared层归一化
- 并行解码：没有使用原始transformer中的auto-regressive解码，而是训练decoder的输入——一组object query，只在cross-attention中使用encoder的结果，实现了并行解码

