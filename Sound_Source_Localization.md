- [1.Learning in Audio-visual Context: A Review, Analysis, and New Perspective.(2022.8)](#1learning-in-audio-visual-context-a-review-analysis-and-new-perspective20228)
- [2.Objects that Sound(2018 eccv)](#2objects-that-sound2018-eccv)
- [3.Audio-Visual Scene Analysis with Self-Supervised Multisensory Features(2018 iccv)](#3audio-visual-scene-analysis-with-self-supervised-multisensory-features2018-iccv)
- [4\*.The Sound of Pixels(2018 iccv)](#4the-sound-of-pixels2018-iccv)
- [5.Looking to Listen at the Cocktail Party:A Speaker-Independent Audio-Visual Model for Speech Separation(2018 arxiv)](#5looking-to-listen-at-the-cocktail-partya-speaker-independent-audio-visual-model-for-speech-separation2018-arxiv)
- [6.Learning to Localize Sound Source in Visual Scenes(2018 cvpr)](#6learning-to-localize-sound-source-in-visual-scenes2018-cvpr)
- [7.Self-supervised Audio-visual Co-segmentation(2019 ICASSP)](#7self-supervised-audio-visual-co-segmentation2019-icassp)
- [8.The Sound of Motions(2019 iccv)](#8the-sound-of-motions2019-iccv)
- [9.Deep Multimodal Clustering for Unsupervised Audiovisual Learning(2019 cvpr)](#9deep-multimodal-clustering-for-unsupervised-audiovisual-learning2019-cvpr)
- [10. Localizing Visual Sounds the Hard Way(2021 cvpr)](#10-localizing-visual-sounds-the-hard-way2021-cvpr)
- [11.Multiple sound sources localization from coarse to fine(2020 eccv)](#11multiple-sound-sources-localization-from-coarse-to-fine2020-eccv)
- [12.Class-aware Sounding Objects Localization via Audiovisual Correspondence(2021 TPAMI)](#12class-aware-sounding-objects-localization-via-audiovisual-correspondence2021-tpami)
- [13^.Learning to Localize Sound Sources in Visual Scenes: Analysis and Applications(2021 TPAMI)](#13learning-to-localize-sound-sources-in-visual-scenes-analysis-and-applications2021-tpami)
- [14^.Mix and Localize: Localizing Sound Sources in Mixtures(2022 cvpr)](#14mix-and-localize-localizing-sound-sources-in-mixtures2022-cvpr)
- [15^.Egocentric Deep Multi-Channel Audio-Visual Active Speaker Localization(2022 cvpr)](#15egocentric-deep-multi-channel-audio-visual-active-speaker-localization2022-cvpr)
- [16.Exploiting Transformation Invariance and Equivariance for Self-supervised Sound Localisation(2022 ACM MM)](#16exploiting-transformation-invariance-and-equivariance-for-self-supervised-sound-localisation2022-acm-mm)
- [17\*.Audio−Visual Segmentation(2022 eccv)](#17audiovisual-segmentation2022-eccv)
- [18\*.Self-Supervised Predictive Learning: A Negative-Free Method for Sound Source Localization in Visual Scenes(2022 cvpr)](#18self-supervised-predictive-learning-a-negative-free-method-for-sound-source-localization-in-visual-scenes2022-cvpr)
- [19^.Self-supervised object detection from audio-visual correspondence(2022 cvpr)](#19self-supervised-object-detection-from-audio-visual-correspondence2022-cvpr)
- [20.Visually Assisted Self-supervised Audio Speaker Localization and Tracking(2022 EUSIPCO)](#20visually-assisted-self-supervised-audio-speaker-localization-and-tracking2022-eusipco)
- [21.MarginNCE: Robust Sound Localization with a Negative Margin(2023 ICASSP)](#21marginnce-robust-sound-localization-with-a-negative-margin2023-icassp)
- [22.Localizing Visual Sounds the Easy Way(2022 eccv)](#22localizing-visual-sounds-the-easy-way2022-eccv)
- [23.A Closer Look at Weakly-Supervised Audio-Visual Source Localization(2022 nips)](#23a-closer-look-at-weakly-supervised-audio-visual-source-localization2022-nips)
- [24.Learning Sound Localization Better From Semantically Similar Samples(ICASSP 2022)](#24learning-sound-localization-better-from-semantically-similar-samplesicassp-2022)
- [25\*.Visual Sound Localization in the Wild by Cross-Modal Interference Erasing(2022 aaai)](#25visual-sound-localization-in-the-wild-by-cross-modal-interference-erasing2022-aaai)
- [26^.Sound Localization by Self-Supervised Time Delay Estimation(2022 eccv)](#26sound-localization-by-self-supervised-time-delay-estimation2022-eccv)
- [27^.Audio-Visual Cross-Attention Network for Robotic Speaker Tracking(2023 IEEE/ACM TASLP)](#27audio-visual-cross-attention-network-for-robotic-speaker-tracking2023-ieeeacm-taslp)
- [28\*.Hear The Flow: Optical Flow-Based Self-Supervised Visual Sound Source Localization(2023 WACV)](#28hear-the-flow-optical-flow-based-self-supervised-visual-sound-source-localization2023-wacv)
- [29.Exploiting Visual Context Semantics for Sound Source Localization(2023 wacv)](#29exploiting-visual-context-semantics-for-sound-source-localization2023-wacv)
- [30\*.Learning Audio-Visual Source Localization via False Negative Aware Contrastive Learning(2023 cvpr)](#30learning-audio-visual-source-localization-via-false-negative-aware-contrastive-learning2023-cvpr)
- [31.Audio-Visual Grouping Network for Sound Localization from Mixtures(2023 cvpr)](#31audio-visual-grouping-network-for-sound-localization-from-mixtures2023-cvpr)
- [32.Audio-Visual Segmentation by Exploring Cross-Modal Mutual Semantics(ACM-MM 2023)](#32audio-visual-segmentation-by-exploring-cross-modal-mutual-semanticsacm-mm-2023)
- [33.Flowgrad: Using Motion for Visual Sound Source Localization(2023 ICASSP)](#33flowgrad-using-motion-for-visual-sound-source-localization2023-icassp)
> 标^为略读，标*为有些启发的文章
# 1.Learning in Audio-visual Context: A Review, Analysis, and New Perspective.(2022.8)
audio-visual learning的特点：认知基础，视觉和听觉对人类建立认知有重要作用，并且在人的认知过程中有广泛的交流与整合；三个一致性，视觉与听觉在语义、空间和时间上密切相关；丰富的数据，视频。
- audio-visual cognitive foundation:总结了认知神经科学中视听模态的神经通路和整合，发现了三个特点Multi-modal
boosting, Cross-modal plasticity, Multi-modal collaboration,分别激发了一类audio-visual工作。
  - 大脑获取听觉信息，然后使用其中嵌入的声学线索（例如频率和音色）来确定**声源的身份**。同时，两耳之间的强度和耳间时间差为**声音位置**提供线索，这称为双耳效应。 
- audio-visual boosting:单一模态的感知只利用了部分信息，且对相应模态的噪声敏感，引入新的模态有利于增强鲁棒性和性能。
  - Audio-Visual Recognition: 语音识别、说话人识别、动作识别、情绪识别。
  - Uni-modal Enhancement:增强/降噪单模态信号。Speech Enhancement and Separation(用视觉信号辅助声音分离)、Object Sound Separation(用视觉、动作信号辅助不同对象，包括同一类别对象的声音)、Face Super-resolution and Reconstruction(语音辅助)
- cross perception:
  - cross-modal perception:单源声音生成（visual->audio）、视频生成（audio->visual）、视听深度估计
  - Audio-visual Transfer Learning:视听域之间的迁移学习。
  - Cross-modal Retrieval
- visual-audio collaboration:
  - Audio-visual Representation Learning:利用自监督等方法，训练视听模态的表征，或训练视听的预训练模型
  - Audio-visual Localization:Sound Localization in Videos，Audio-visual Saliency Detection，Audio-visual Navigation，Audio-visual Event Localization and Parsing(区分视频中的可听事件和可见事件，为每一帧的分类问题) 
  - Audio-visual Question Answering and Dialog

# 2.Objects that Sound(2018 eccv)
谷歌和牛津大学的工作，本文提出了一种vision-audio的自/无监督方法audio-visual correspondence，对于无标记的视频数据，分别提取同一秒的图片和声音数据的嵌入，让模型判断其是否对应。并且，应用AVC方法到跨模态检索和声源定位任务。
- cross-modal retrieve:文中的声音数据是“对数频谱图”（维度为Time-Frequecy），因此也使用CNN处理。AVE-Net用CNN子网络+fc分别处理图片和audio数据，得到128维嵌入，使用欧几里得距离计算嵌入之间的距离，将这个标量输入1\*2全连接层（这里充当一个阈值的作用），输出“对应” or not。在实验（输入图片/audio，检索最对应的图片/audio，2\*2排列）中，AVENet击败其他基线，表明它的嵌入具有良好的对齐性。
> 文中还探索了“多帧图片输入”的影响，尝试了mutil frame输入并使用3D卷积，和光流(OF)辅助的方法，在AVC任务上均有提升，但在检索任务上表现不佳，认为MF和OF会使嵌入质量下降。

> 文中还提到了一个网络容易过拟合的点，在采样正例时，audio总是选择以image为中心的1s，因此正例audio的中点总为0.04s的倍数（视频的帧数为25fps），而模型可能因为MPEG 编码和/或音频重采样的低级伪影识别这一点，从而简单的识别正/负例。文中对负例的采样也以0.04s的倍数为中心，明显提高了性能，还使用了一种数据增强方法（将数据随机misalign至多一秒）。
- Localizing objects that sound:设计了Audio-Visual Object Localization (AVOL-Net)，定位发声目标。与AVCNet相比，audio子网络不变，image子网络将最后两层fc换为1\*1卷积，保持了14\*14的分辨率，再用点积衡量audio与每一个patch的相似性，最终对14\*14相关性图做sigmoid后max pool，预测是否对应（AVC），并用相关性图定位发声物体。
> 我感觉在这种无监督设置下，定位问题有点病态，对于一些情形是有意义的，比如在许多吉他数据中，图片中都有相似的吉他patch，可能相互增强？文章中进行了实验，证明模型不是“显著性检测”，而是确实将音频和定位对象对应起来了，也算是“定位发出声音的物体”吧。

> 文中标注了500张图来评价定位，与直接预测图片中心（57%）相比，效果还是可以的（82%）

![Alt text](<Paper/Sound_Source_Localization/image/1.png>)

# 3.Audio-Visual Scene Analysis with Self-Supervised Multisensory Features(2018 iccv)
UCB的工作，本文也提出了一种自监督方法，通过*训练神经网络来预测视频帧和音频是否在时间上对齐*学习了一种融合视觉和音频分量的时间、多感官表示（和上一篇的方法相似但模型差距挺大的），并将这种其应用于声源定位、视听动作识别和音频源分离
- Learning a self-supervised multisensory representation:设计了一个网络来训练表征，输入为连续帧和波形，输出为是否对应。视觉子网络是3D卷积，处理时间关系；audio分支为**1D卷积**，处理时间上的相关性，并降低音频的采样率。
> 1D卷积的输入为一维数据（此处的波形图表示的是，按一定采样率选择横轴上的点，将其纵轴坐标组织成tensor代表波形图），卷积为一维（沿时间尺度加权求和，相当于平滑/滤波？）

![Alt text](<Paper/Sound_Source_Localization/image/2.png>)
- visualizing the locations of sound sources:很简单，把最后的512维特征图生成CAM。
> CAM(class activation map):根据CNN的特征图，可视化模型注意的区域。具体来说，把特征图做GAP得到权重，逐特征图加权求和，归一化后即可。
- action recognize:在动作识别的数据集上微调
- Source separation model:数据是合成的，将连续帧和混合音频输入audio-visual network ，将混合音频的频谱图输入u-net，将前者最终的特征和u-net编码器后的特征叠起来，一起输入解码器，得到分离后的频谱图（似乎是训了两个解码器，一个前景一个背景，但训练时是对称的）

# 4\*.The Sound of Pixels(2018 iccv)
MiT的工作，提出了一种自监督方法，对混合声源视频进行声音分离和声源定位，值得注意的是，在声源定位时，本文预测了逐像素的声音，与mutil-source seg有共同之处。
> 比如，可以先做语义分割，再对预测mask的像素做池化，用共同特征去预测region的sound。
- 逐像素声音预测：vision子网，向ResNet输入连续的图片帧，按时间维度最大池化，得到h\*w\*k特征图；audio子网，向U-Net输入声音的T-F频谱图，输出k个频谱图（表示音频的k种特征组分？）；合成网络，包含一个k维线性层，vision侧每个像素的k维特征加权audio端的k张图，输入线性层加权求和，得到每个像素的mask，与原频谱图相乘后做短时傅里叶逆变换得到波形。
![Alt text](<Paper/Sound_Source_Localization/image/3.png>)
- 声源定位：逐像素声音预测后，对像素的声音分量做聚类。
- 声音分离：vision端变为时空池化，得到一张图片的k维表征，audio端输入混合音频，可以得到每张图片对应的声音。
- 训练过程中数据无标注，文中有提GT mask的生成方法（若该组分在混合中幅值最大，mask的对应像素为1），作为监督信号。

# 5.Looking to Listen at the Cocktail Party:A Speaker-Independent Audio-Visual Model for Speech Separation(2018 arxiv)
google的一篇文章，引用700+，不过没发在顶会上。本文利用视觉信息做Speech Separation，并将语音分配给其对应的面孔，模型是speaker无关的（训练一次，适用于所有speaker）。不属于声源定位，算是声音分离和分配？
- 提出了一个大数据集AVSpeech，经过处理后是图片+干净的声音（单个speech），正因为这个大数据集，才训练出了泛化性强的speaker无关的模型，训练时使用的是由AVSpeech合成的数据。
- 设计了一个vision-audio语音分离模型，论文里的图示非常清楚。
- 视觉：3秒的视频有75帧（25FPS），用预训练好的人脸检测模型检测人脸（可能有多个人），再用训好的FaceNet提取人脸嵌入，得到75\*1024的输入（时间\*维度）。经过CNN后（处理的维度不是长乘宽了），在时间维度上采样到298与音频对齐。
- audio：做STFT得到频谱图，将实部和虚部叠起来作为输入，经过CNN。
- fusion：沿维度叠起来，得到298\*D，输入BLSTM（共298个时间步），将维度压缩到400，经过全连接层后输出每个speaker的掩码（实部和虚部），与噪声的频谱图相乘，再逆STFT得到每个人的声音。（预测的单人频谱图与AVSpeech的GT频谱图做L2损失）
> 文章的意思好像是，如果输入speaker的数量不同，得分别训练模型，也可以只训单个speaker的，具有一般性。A separate, dedicated model is trained for each number of visible speakers.

![Alt text](<Paper/Sound_Source_Localization/image/4.png>)

# 6.Learning to Localize Sound Source in Visual Scenes(2018 cvpr)
比较一般的一篇工作，提出了一种声源定位的无/半监督方法，双流网络+注意力融合。
- 视觉分支：VGG主干，得到特征图v，h\*w\*512
- audio分支：输入波形，网络为1维卷积（SoundNet），最终GAP成1000维特征f_s，再经过两个FC得到h
- 融合模块：用h和v逐像素计算注意力（点积），得到$\alpha$，以此进行声源定位。
- 无监督学习：用$\alpha$加权求和v，得到融合表征z，通过两个FC得到f_v，希望其与f_s在同一嵌入空间，用L2距离损失训练（正例靠近，负例远离）
- 半监督学习：本文认为无监督有许多错误，标了一批数据，用半监督做，就是把注意力图和GT图算个损失。
![Alt text](<Paper/Sound_Source_Localization/image/5.png>)

# 7.Self-supervised Audio-visual Co-segmentation(2019 ICASSP)
MIT的工作，是The Sound of Pixels的延续，但感觉有点狗尾续貂（引用量也不高）。本文没有改变The Sound of Pixels的框架和训练方法，而是为了适应实际需要（不一定总为有同步音频的视频），可以在只输入图片的情形下输出对应类别的分割图，在只输入混合音频时输出分离的声音。
> 这需求创造的有点奇怪……只有图片为什么不用（无监督）分割方法？只有音频为什么不用专门的speech分离模型？可能是把The Sound of Pixels的方法作为一种无监督方法，可以在大型无标记视频数据上训练，并实现分割和分离。
- Disentangling Internal Representations:为了在训练后独立使用图像分析网络和音频分析网络，模型需要在合成模块之前对特征进行解耦（？）具体来说，在训练后期，将子网络最后的sigmoid换为softmax，并使用温度T，让T逐渐降低，从而使图片\语音特征倾向于one-hot.
- Category to Channel Assignment:提取验证集的视觉特征，并根据其label(image-level)将类别分配给K个组分（可能有剩余）。
- 推理：选择希望分割/分离的类别对应的组分，计算分割（就用激活图）和波形。
- 实验：语义分割选择的baseline是CAM……这是不是太低了一点
![Alt text](<Paper/Sound_Source_Localization/image/6.png>)

# 8.The Sound of Motions(2019 iccv)
也是MIT的工作，也是The Sound of Pixels的延续。本文仍基于之前的框架，将图片信息换为动作信息，提出Deep Dense Trajectory (DDT), 明确建模action信息，实现声音的定位和分离，还可以解决“分离相同乐器的二重奏”的困难问题。
- motion net:输入连续的帧，先用PWC-Net预测光流，再根据密集光流迭代的预测密集轨迹(T\*H\*W\*2)，最后用CNN提取轨迹特征
- appearance net:输入连续视频帧的第一帧，用CNN提取特征
- fusion模块：图示很清楚，外表特征经sigmoid得到注意力图，乘到轨迹特征上；外表特征在时间维度膨胀后，与调制后的轨迹特征沿维度连起来，经过CNN和空间池化得到视觉特征

![Alt text](<Paper/Sound_Source_Localization/image/8.png>)
- sound seperate net:向U-net输入频谱图，在编码器后插入视觉特征。在此处用了FiLM模块，先将visual和audio在时间维度对齐，再用K_v计算仿射变换因子，分别变换audio特征，最终得到沿时间维度重新加权的特征图，输入解码器得到mask。
> 图示似乎有问题，一张图片应该只输出一个波形吧

![Alt text](<Paper/Sound_Source_Localization/image/7.png>)

# 9.Deep Multimodal Clustering for Unsupervised Audiovisual Learning(2019 cvpr)
本文提出一种无监督聚类方法Deep Multimodal Clus-
tering (DMC)，在声音定位、多源检测和视听理解等任务有很好的表现。
- 视觉:用VGG提取，得到64\*512的特征
- 音频：输入频谱图，用VGGish提取，得到124\*512特征
- 单模态聚类：引入K-Mean聚类，用点积作为度量函数，引入min函数的近似形式，使全过程可微，用EM方法迭代优化。经数学推导，得到如伪代码的过程，最终得到K个聚类中心（K为预设参数）。（W是对不同中心特定的投影，为了区别不同音视实体的表征。）
> 这里数学推导没大看明白聚类中心是怎么整理出来的
- 训练时采用最大边缘(max-margin)损失，最小化正例vision和audio对应聚类中心的距离（余弦近似度），而最大化负例的。
> 这里没大看明白，vision和audio的聚类中心之间怎么对应的，似乎是在训练初期随机设置，再由DNN自己对齐？
- 单声源定位：将audio特征平均池化得到聚类中心（1个），再与vision的每个聚类中两个，声源和其余）做余弦相似度，得到最近似的聚类中心后，利用与其相关的“软分配”（每个像素与其的距离）s_ij可视化声源位置/得到热力图。
![Alt text](<Paper/Sound_Source_Localization/image/9.png>)



# 10. Localizing Visual Sounds the Hard Way(2021 cvpr)
牛津VGG的工作，提出了一种无监督的声源定位方法，还做了一个大数据集（bbox标注声源），文章的创新点在于利用hard negative样本（比如输入图片中不是声源的部分，即背景），在无监督设置下尝试解决这个问题还是挺有意思的
- 假设有mask标注的训练过程：输入连续视频的中间帧和频谱图，分别用CNN提取特征(h\*w\*c和c维)，逐点计算两个特征的余弦相似度得到S。用对比学习的方法进行训练，以音频为query，正例为音频对应图片的mask为1部分S的均值，负例为音频对应图片的mask为0部分S的累加(hard negative)+随机采样的不对应图片的S均值，损失为（k为数据集大小）
![Alt text](<Paper/Sound_Source_Localization/image/10.png>)
- 替代mask的方法：设置阈值，对S打伪标签；显式的阈值阈值不可微，用sigmoid函数近似(温度设置越小，越像one-hot)；伪标签总不会太可信，设置了0和1之外的uncertain类，即设置了两个阈值来分别生成用于正例和负例的mask，两阈值之间的像素不会用来计算。
![Alt text](<Paper/Sound_Source_Localization/image/11.png>)

# 11.Multiple sound sources localization from coarse to fine(2020 eccv)
本文提出了一个多声源定位方法，训练为弱监督的（video-level类别），从方法上来看感觉是个很实用、效果应该不错的方法。
- 总的来说，是一个两阶段训练框架，第一个阶段以分类（用到了数据集的image-level标注）和对应（AVC）为目标训练视觉和声音的对齐，第二阶段，利用grad-CAM得到视-音每个类别的特征表示，并用L2损失对齐两个模态的特征表示。题目中的coarse to fine指的应该是从第一阶段的image-level到第二阶段的类别level
- Classification on Two Modalities:视觉分支用ResNet，听觉用CRNN，分别提取特征并进行多标签分类，损失为二元交叉熵。
> 多标签分类指的是一个样本可能属于多个标签(GT不是one-hot，而是多个1)，此处可能是视频的标签tag有多个。多标签分类的二元交叉熵，对每个类别做sigmoid（相互独立），再累加每个类别的二元交叉熵
- Audiovisual Correspondence Learning:将两个分支的特征图全局最大池化得到512维特征，叠一起经过FC得到2维预测，预测视-音是否对应
- Disentangle Features by Grad-CAM:对于每个分支，以分类预测结果和特征图为输入，Grad-CAM可以得到类别特定的热力图（表示激活的强度，C\*H\*W），以每个类的热力图（H\*W）为权重，加权全局池化特征图（H\*W\*D），得到C个D维向量（每个类别的表征）。
> Grad-CAM:根据分类结果（C维），计算每个类别预测结果相对于特征图（H\*W\*C，一般为最后一层）的梯度，对H,W维度求均值，得到C个C维向量，分别加权求和特征图，ReLU后得到每个类别对应的可视化图
- Fine-Grained Audiovisual Alignment:先分别用FC映射两个分支提取的C个表征，再用L2距离衡量表征之间的距离。在训练时，只将同一视频、同一类别的视音表征作为配对的正例，其余均为负例（负例的间距有个最大间隔），求和计算损失即可。
- 声源定位：视觉分支在Disentangle Features by Grad-CAM中不再用热力图加权求和，而直接计算语音分支提取的类表征（C\*D），逐“像素”的计算与视觉分支特征图（H\*W\*D）的距离，得到每个类的定位预测图。
- 将声源定位结果应用到声音分离，得到很好的结果。
> 这篇感觉是偏研究向的，在真实情景下类别C是未知的。（好吧，下一篇文章就改进了这个地方，不谋而合了）

![Alt text](<Paper/Sound_Source_Localization/image/12.png>)

# 12.Class-aware Sounding Objects Localization via Audiovisual Correspondence(2021 TPAMI)
是上一篇11的扩展，流程很复杂，但有迹可循，是一篇实用的工作，提出一种自监督方法，实现多声源定位，并且可以区分发声/不发声的对象。
![Alt text](<Paper/Sound_Source_Localization/image/13.png>)
- 整体是一个双阶段训练框架，将训练数据集分为单源和多源两类，第一阶段在单声源的视音对上训练，用较简单的样本构建潜在类别表示的字典，在第二阶段用复杂样本训练，用构建的词典预测图片中潜在的发声对象，并要求发声对象的视觉类别分布与混合声音的类别分布相匹配。
- Single Sounding Object Localization:用双分支网络分别提取特征，对语音特征做全局池化，逐像素计算与视觉特征的余弦相似度（H\*W），经过1\*1卷积和sigmoid后得到单声源定位图，对定位图做GMP得到标量指示“视-音是否对应”，从而计算二元交叉熵损失
- Visual Object Representation Learning:以定位图为权重，加权平均池化视觉特征图，得到每个样本C维的对象表征o_i，并以此建立潜在声音对象词典（K\*D）。为了得到K个聚类中心，利用K均值聚类，得到聚类损失 $L_ {clu}=min||o_ {i} - D^ {T} y_ {i}||_ {2}^ {2},s.t.y_ {i}\in \{0,1\}^K, \sum _ {i=1}^ {n} y_i=1$ ，其中$y_i$是该对象的伪标签（指示属于的聚类中心/潜在对象类别）。同时，提出了一种用伪标签指导分类的方法，在视觉和语音的子网后分别加个分类头，使其拟合伪标签，得到分类损失$L_{cls}=L_{ce}(y_ {i}^{\*},h_{a}(a_{i}^ {s}))+L_{ce}(y_{i}^{\*}, h_{v}(v_{i}^{s}))$。
- 第一阶段训练时使用了交替训练的方法，第一阶段用对应损失训练，第二阶段，先优化聚类，再用分类损失训练。文中提到，聚类损失不反向传播，我认为它的意思是，聚类损失仅作为目标来优化对象词典，而不改变网络参数。
- Discriminative Sounding Objects Localization:第二阶段，用对象词典中的每个表征与视觉特征图做逐像素点积，得到K张概率图，只是对应潜在对象的位置；用预测的粗声源定位图mask概率图，滤去不发声的对象；对K个概率图做GAP再做softmax，得到发声物体的类别分布，同时对全局池化后的语音特征做线性变换再softmax，也得到K维分布，计算两个分布的KL散度损失，以保证视听的一致性。
- 第二阶段训练，使用对应损失和KL散度损失，最终的预测定位图，将K张概率图逐像素做softmax得到。
 

# 13^.Learning to Localize Sound Sources in Visual Scenes: Analysis and Applications(2021 TPAMI)
就是6的那篇工作，我读到声音分支的soundnet、输入波形、1维卷积就感觉熟悉，没想到2018年的cvpr能发2021的TPAMI，文中提到标了一个声源定位的数据集。

# 14^.Mix and Localize: Localizing Sound Sources in Mixtures(2022 cvpr)
基于图模型的，但目前不太熟悉图学习方法，没细看，有需要再读。

# 15^.Egocentric Deep Multi-Channel Audio-Visual Active Speaker Localization(2022 cvpr)
这是Meta针对AR做的一片工作，作者认为过去的声源定位的环境设置大多为单通道、外心，不能适应AR，因此在360°视频和多通道麦克风阵列音频的新设置下，提出了一种声源定位方法。

一时半会做不到AR相关，先不细看了，感觉机器人感知可能和这个有些类似？不过也要考虑麦克风设置。

# 16.Exploiting Transformation Invariance and Equivariance for Self-supervised Sound Localisation(2022 ACM MM)
很简单的一篇文章，将数据增强应用到自监督的声源定位任务，是预料之中的工作。
![Alt text](<Paper/Sound_Source_Localization/image/14.png>)
- 训练方法用的是10那篇VGG的Localizing Visual Sounds the Hard Way，双分支，用语音特征逐像素去算和视觉特征的余弦相似度，加个软阈值得到定位mask。
- 数据增强：语音数据，对频谱图的T和F维度做随机mask，选择最相似的样本（提取的语音特征距离最近）进行混合（此处用了课程学习的方法）；图片数据是外观变换（色彩抖动等）和几何变换（裁剪等）
> 课程学习：由易到难的学习，在本文中，混合是两个样本的线性加权，而“最相似样本”的权重从0逐渐增加
- 孪生网络:在训练时，对同一个视音对进行两种不同的增强，分别通过“共享参数的两个网络”（也就是孪生网络，感觉可以理解为，其实就一个网络，然后两个样本并行前向传递而非串行？）
- 视音对应一致性：经过数据增强后的视音对应仍保持对应关系，计算对应损失，此处使用的是VGG的方法，估计伪mask，再计算正例和负例，最后用对比学习的InfoNCE损失；Geometric Transformation Equivariance，两个输入的视音对，其中一个只有外观变换，另一个是外观变换+几何，外观变化不改变语义，因此二者的余弦相似度map应该只有几何变换的差异，计算二者的L2损失。

# 17*.Audio−Visual Segmentation(2022 eccv)
提出一个新的任务audio-visual seg(AVS)，即对视频的声源对象进行分割，给出了一个数据集AVSBench，还提出了两个模型，分别给单声源半监督和多声源全监督任务提供了baseline。
![Alt text](<Paper/Sound_Source_Localization/image/16.png>)
- AVSBench:均为5秒的视频，分为5个1秒的clips，标注的mask为clip的最后一帧，共包括4,932个单声源视频，424个多声源视频，共23个对象类别。单声源任务较简单，仅对第一个clip标注，多声源任务五个都标，所有验证集和测试集也五个clip都标。
- AVSBench-semantic:和上面的标准方式类似，但是分了十个类别，按语义标注的，而非是二元mask
- 整体框架为编码器-解码器结构，
- 编码器：语音子网（VGGish）输入频谱图得到语音特征A，视觉子网输入T个连续帧，对于backbone每个阶段的特征图F_i，使用ASPP提取V_i，将V_i与A分别融合。
- 特征融合：文中提出了temporal pixel-wise audio-visual interac-
tion (TPAVI)融合语音和视觉信息，结构如图所示，类似于注意力方法
![Alt text](<Paper/Sound_Source_Localization/image/15.png>)
- 解码器：解码器使用 Panoptic-FPN，就是将编码器对应阶段和下一阶段（分辨率更小的方向）的特征图加入每一阶段的解码，得到预测掩码M
- 目标函数：包含两部分，首先是对mask的监督损失，其次是一个强制视-音对应的正则化项，用预测mask(下采样到对应尺寸)每个阶段的特征图后，做AVG再计算与语音特征（用线性层映射到同一维度）的KL散度， $L_ {AVM}=
KL(avg(Mi \odot Zi),A_ {i})$。后一项损失在半监督中被忽略。

# 18*.Self-Supervised Predictive Learning: A Negative-Free Method for Sound Source Localization in Visual Scenes(2022 cvpr)
感觉比较水的一篇文章，本文提出了一种“三流框架”，输入语音和两个不同增强的图片，感觉是16数据增强方法的简化版；文中还引入认知神经科学的知识(inherits the spirit of predictive coding (PC) in neuroscience)，设计了一个模块来融合对齐不同模态。
![Alt text](<Paper/Sound_Source_Localization/image/17.png>)
- 背景：背景还是很有意思的，作者认为之前声源定位的自监督方法大多使用自监督训练，特别是对比学习，而对负例的采样往往是随机的，这容易导致假负例（虽然在不同视频中，但是是同一声源对象，如不同视频中的吉他），可能会导致音频和视觉特征之间的不一致。因此，本文显式挖掘正例、而不利用负例的对比学习方法解决这个问题。
> 感觉这个方法不能很好的解决假负例的问题，效果提升是因为数据增强的正则化？也许可以从这个问题入手，比如挖掘难负例？像10那篇工作
- 整体框架：语音经STFT转换为频谱图输入VGGish，图片输入VGG，分别提取特征；应用PCM模块对齐两个模态的特征，在AM模块中预测定位图、并输入多模态表征，以进行自监督学习。
- AM(attention module):就是计算语音和视觉特征的余弦相似度图，作为定位图，再将相似度图min-max正则化，来逐像素乘视觉特征图，得到多模态表征。
- PCM：没大看懂，作用大致是细化融合特征，反馈-前向更新比较复杂，原理应该是认知神经科学的知识。
- 自监督学习：根据 SimSiam的对比学习方法，在AM后加了个MLP的预测头，希望1图片的AM输出的多模态表征在预测头映射后，尽可能与2图片的多模态表征类似，反之亦然。因此，最终的损失为， $L_ {SSPL}= \frac {1}{2}L_ {NCS} (z^ {1}, z^ {2})+\frac {1}{2}C_ {NCS}( z^ {2} , z^ {1})$,其中$L_{NCS}$为负的余弦相似度（前一项的MLP映射表征和另一项的原表征）。为了防止两个表征相互坍缩，使用了梯度停止方法，即$L_{NCS}$不更新后一项的梯度

# 19^.Self-supervised object detection from audio-visual correspondence(2022 cvpr)
本文是想利用无标签的vision-audio信息，先训练声源定位，再利用定位结果作为伪标签，训练目标检测器（不发声的目标也要检测出来）。因为声源定位方法不会区分类别，文中还在第一阶段训练了个分类网络，给每个图片分类（聚类方法）。
- 读了一下定位网络部分，就是最经典的，双分支、点积相似度、最大池化后计算对比学习损失（是否对应）。
- 分类网络使用的是Labelling unlabelled videos from scratch
with multi-modal self-supervision的方法，大致是假设K个潜在对象，再分类，没细看。
- 目标检测部分没读。

# 20.Visually Assisted Self-supervised Audio Speaker Localization and Tracking(2022 EUSIPCO)
有点离谱的一篇文章。本文引入师生框架，用视觉信息训练教师网络，生成伪标签来监督训练学生网络（只以语音为输入），学生网络可以预测声源位置，并且输出声源的分割mask，还可以实现speaker跟踪任务
> 本文将声源固定为了speaker，还使用了人脸检测器，对于定位任务不具有普适性。文章的出发点为，“如果视觉模态缺失”，只用语音模态进行声源定位，这个设置具有一定的实用意义。

![Alt text](<Paper/Sound_Source_Localization/image/18.png>)
- 教师网络：应用了一个人脸检测器，提取bbox，再选择中心偏下的一个点(0.5,0.75)作为嘴（声源）的坐标；应用预训练的PSPNet预测分割二元mask
> 我不相信一个预训练的PSPNet能在没见过的数据集上达到很好的效果……
- 学生网络：仅以声音的频谱图作为输入，经CNN提取特征。对于定位任务，将特征经过四个FC，预测坐标，计算和伪标签的MSE损失；对于分割任务，后接了三层反卷积（好像都没有横向连接），将预测的mask与伪标签做交叉熵。
> 文中认为，多任务训练（定位+分割）能提高效果。但是，我不相信，输入语音，经过这么一个简单的网络（仅仅在提取的特征后加了三层反卷积），能输出对声源的mask，这明显是病态的问题。如果效果好，只可能是过拟合。
- 文中还用音频定位网络实现speaker跟踪，没细看。
> 实验部分是以定位和跟踪为评价指标，所以语义分割只是辅助任务。

# 21.MarginNCE: Robust Sound Localization with a Negative Margin(2023 ICASSP)
本文与18的思路类似，声源定位的对比学习方法在选取正/负例时，简单的视听对应可能导致噪声，比如假正例（声源实际上在图片外，与图片无关）、假负例（虽然图片和声音不对应，但是这张图片可能与声音也在语义上相似，比如同一乐器）。
- 本文的框架就是第10篇工作Localizing Visual Sounds the Hard Way中的方法，只是将原来的InfoNCE换成了
![Alt text](<Paper/Sound_Source_Localization/image/19.png>)
- 加入了软间隔m（m为负值），即将正例的相似度总是增加m的绝对值大小。个人理解时间：当我们将符号带入log时，变为log(1+x)，最小化损失变为最小化负例与正例的比值(x)，负例的形式没有变化，正例的m有什么作用呢？当出现假正例，正例的相似度可能很小，从而导致一个大的损失，干扰模型；而加个|m|，即使出现假正例，损失也不会太大。对于真正例影响不大，让损失/梯度更加平滑了？

# 22.Localizing Visual Sounds the Easy Way(2022 eccv)
从名字上看是致敬第10篇工作Localizing Visual Sounds the Hard Way，但是感觉比较一般。本文提出了一种简单有效的方法EZ-VSL，大大提高了sota。
![Alt text](<Paper/Sound_Source_Localization/image/21.png>)
- 背景：对背景的分析比较有启发性，一个是，希望语音表征和声源的视觉表征相对应，需要声源的精确定位，但是，声源的精确定位需要语音表征和声源表征对应，是一个矛盾问题，复杂的训练可能导致次优化（文中用一种模糊化的方法处理？只相应最强的patch）；另一个是，对于图片，可以利用很多先验，比如蓝天等位置明显不会充当声源。
- Audio-visual matching by multiple-instance contrastive learning:本文的整体框架仍是双分支+对比学习，但是，只要求语音特征响应视觉特征中最强的一个patch，是 multiple-instance contrastive learning。如下图，sim为余弦相似度图，相比于过去将正例累加计算损失，该损失只取最大的部分。总损失还有一项$L_{v->a}$
![Alt text](<Paper/Sound_Source_Localization/image/20.png>)
- Object-Guided Localization:用一个在Imagenet预训练的模型，提取图片的特征。由于Imagenet预训练是分类任务，提取的特征对object敏感。将特征沿维度计算L1度量，得到H\*W定位先验图，将先验与余弦相似度图“线性叠加”，作为最终的定位图。（叠加前进行min-max归一化）

# 23.A Closer Look at Weakly-Supervised Audio-Visual Source Localization(2022 nips)
和22的作者是同一位，来自CMU，提出的现存问题很好，但是给出的方法感觉有点一般，做的实验很solid。（题目中的弱监督指的是ImageNet预训练的backbone）
![Alt text](<Paper/Sound_Source_Localization/image/23.png>)
- 背景：本文针对的问题也很有启发性，一方面，目前的工作大都容易过拟合，必须依赖验证集实现early stop，但验证集需要标注，与无监督的设置有所违背；另一方面，目前的声源定位方法没有考虑“图片中没有声源”的情况，因此当应用到真实情景时，是不可信的。
- 整体框架为Localizing Visual Sounds the Easy Way里的，提出了一个新的方法Simultaneous Localization and Audio-Visual Correspondence(SLAVC)，包括三个部分。
- Dropout:为了防止过拟合，对提取的视觉特征进行了**0.9**的dropout，很有效
- Momentum encoders:使用EMA更新编码器的权重，具体来说，有两组语-音编码器，每一组有一个模态的子网是在线的，另一个是固定的，只有在线的子网用梯度更新参数，而固定的子网由EMA更新，从而获得更稳定的自训练和增强的表示。
- Simultaneous localization and audio-visual correspondence:希望网络同时定位可能发声的对象，并强调与语音特征对应的部分。这两者在独立的子空间进行，即将语音和视觉特征线应用不同的线性映射g，再分别计算余弦相似度。定位任务希望找到最相关的空间，因此在xy维度上做softmax，对应任务希望找到最符合的音频，因此在instance维度上做softmax（后者也让图片中没有声源时，避免过度自信），分别得到相似度图S。将两个任务的相似度图相乘，得到最终的相似度图，像EZ-VSL一样计算对比损失。
![Alt text](<Paper/Sound_Source_Localization/image/22.png>)
- 总任务：语音和视觉特征都使用静态网络提取，预测的定位图为定位任务和对应任务相似度图的和。

# 24.Learning Sound Localization Better From Semantically Similar Samples(ICASSP 2022)
本文是第10篇工作Localizing Visual Sounds the Hard Way的扩展，和我之前想到的一样，除了第10篇工作强调的hard negative，还有hard positive，比如，在不同图片中但语义相似，若将hard positive当作负例进行学习，会影响效果。
- 本文没有提出新的模型/损失函数，而是提出了一种挖掘hard positive的方法。
- 利用编码器提取特征，在同一模态内，对于每个样本，计算与其余所有样本的点积A，选取前K个作为该样本的难正例P。于是，训练时的正负例选择如下图，每个样本的正例由四部分组成（视觉与语音，视觉的hp和语音，视觉和语音的hp，视觉的hp和语音的hp），代入原损失函数即可。
![Alt text](<Paper/Sound_Source_Localization/image/24.png>)

# 25*.Visual Sound Localization in the Wild by Cross-Modal Interference Erasing(2022 aaai)
和第10、11篇是一个作者，风格比较相似，复杂系统但有迹可循、特征聚类、class-specific，虽然模型的人工设计感很强，但道理讲得通
- 背景：针对真实视听声源定位的干扰问题，如属于声源但不发出声音，或者声源在屏幕之外，或者混合音频中音量大的会起主导作用。基于此，作者提出了interference eraser.
- 整体框架：是一个二阶段训练过程，第一阶段在单声源数据集上训练，第二阶段在无约束数据集上训练。
- Audio-Visual Prototype Extraction:输入图片和波形图，计算提取出的特征的余弦相似度图，GAP后计算交叉熵损失（是否对应）；对视觉特征进行深度聚类，得到K个视觉原型（聚类中心），并得到一个K维的伪标签。语音特征按伪标签分类后也聚类，可以得到K个语音原型
- Audio-Instance-Identifier:此模块为了更好的区分声音组成（消除音量影响，判别所有声音种类），扩展声音表征。在单源数据集上混合得到训练样本，对应的伪标签也可以得到。根据音频子网中间层的特征图，经过线性层得到distinguishing-steps，$\Delta$，为K\*C维，分别与语音特征相加，得到K个扩展的语音特征，分别计算扩展的语音特征和对应原型的余弦相似度，并与伪标签做交叉熵损失。直觉上来看，$\Delta$可以补偿弱声源的音量，使扩展后的语音特征更接近对应原型，提高了模型的类别感知。
> 训练使用了课程学习的方法，混合样本的比例逐步提高，混合数量也逐步提高。

![Alt text](<Paper/Sound_Source_Localization/image/25.png>)

- Class-Aware Audio-Visual Localization Maps:提取了视觉和语音特征后，计算每个视觉原型与视觉特征图的余弦相似度，作为每个类别的视觉定位图（可能包括不发声的对象）；计算扩展后的语音特征与视觉特征的余弦相似度，得到K张余弦相似度图，逐类别与视觉定位图相乘，得到K张class-specific视-音定位图(AVMaps)。整个过程滤去了无声目标，Silent Object Filter
- Acquiring Audio-Visual Distribution：将AVMaps做GAP池化后softmax，可以得到visual-guided目标分布；再计算听觉的目标分布，计算每个扩展听觉特征与对应原型的余弦相似度，为了滤去屏幕外声源的影响，借助了视觉指导（对K个定位图设置阈值，得到每个类的二元mask，再对mask做GAP得到权重），对K个余弦相似度加权正则化，再做softmax得到audio-guided 类别分布，整个过程滤去了屏幕外目标，Off-screen Noise Filter。
- Cross-Modal Distillation:为了使不同模态的知识互相指导，计算两个分布的KL散度（对称的）作为损失函数。
  
![Alt text](<Paper/Sound_Source_Localization/image/26.png>)
# 26^.Sound Localization by Self-Supervised Time Delay Estimation(2022 eccv)
任务设置和通常的声源定位差距比较大，本文针对立体声数据，通过训练了一个时间延迟预测模型，进行声源定位，没细读。

# 27^.Audio-Visual Cross-Attention Network for Robotic Speaker Tracking(2023 IEEE/ACM TASLP)
本文属于机器多模态了，根据机器人数据实现speaker DoA localization and tracking，是个比较复杂的系统，还适应传感器的数据，和一般的声源定位差距较大，以后如果做这个方向可以读一读。本文提出了一个支持深度学习的视听数据集，其中包含真实机器人捕获的信号，提供单目图像序列、多通道麦克风阵列信号和说话者 3D 位置注释。

# 28*.Hear The Flow: Optical Flow-Based Self-Supervised Visual Sound Source Localization(2023 WACV)
本文将光流信息加入一般的自监督声源定位方法中，以提供声源可能的定位，但是我认为文中提出了利用光流信息的方法不是很有道理，有些问题。
![Alt text](<Paper/Sound_Source_Localization/image/27.png>)
- 框架：完全是Localizing Visual Sounds the Hard Way的方法，只是加入了光流定位子网，对相似度图进行了修正。
- 光流定位：输入两个连续帧，光流估计网络给出光流模态的特征表示$f_f$，计算其与视觉特征$f_v$的注意力，将前者做线性映射作为Q，后者作为K和V。最终得到修正后的视觉特征，与原视觉特征逐元素相加，再计算与语音特征的余弦相似度。
> 文中的注意力计算方法很诡异，我认为两个H\*W\*C计算注意力，应该先计算A(H\*W\*H\*W)，即每个像素与其余所有像素的注意力分数，然后再加权V，得到H\*W\*C的结果，这样可以从光流中学到位置信息。但是，文中是将K和Q每个像素的C维特征算外积，计算了(H\*W\*C\*C)的注意力分数，再用V去乘对应像素的C\*C注意力map，得到H\*W\*C的结果。也就是说，文中捕捉的是视觉和光流不同通道间的注意力，然后将视觉的C维特征v，对于输出第K维，按照每个视觉维度与光流第K维的相似性加权v（从矩阵计算的角度来理解），得到输出的v的第K维。感觉有些怪，可以理解能学到光流信息，但是能学到定位信息只能说深度学习太神奇了。

# 29.Exploiting Visual Context Semantics for Sound Source Localization(2023 wacv)
本文希望更加充分地利用视觉上下文语义信息，加入视觉推理模块，并设计了两个新的loss
![Alt text](<Paper/Sound_Source_Localization/image/28.png>)
- Audio-Visual Correspondence Learning:主体框架为双分支、余弦相似度、最大池化、对比学习，计算AVC损失
- Visual Reasoning Module Structure:假设有N个潜在感兴趣区域，将视觉特征输入N个1\*1卷积提取推理图S，表征可能目标的位置；再用S(H\*W\*N)加权求和V(H\*W\*C)，得到待选区域的特征O(N\*C，N个C维特征)
> N是手工选择的，感觉可以优化一下
- Learning Objectives of Visual Reasoning：计算语音特征与O的余弦相似度，将得到的分数降序排列，前P个为正例，后Q个为负例，计算正例和负例的均值（这一部分的正负例均为图片内部的，N个待选区域），上下文损失为 $L_ {context}=\frac {1}{B}\sum _ {i=1}^ {B}\log \frac{exp(P_ {i})}{exp(P_ {i})+exp(Q_ {i})}$ 。为了使视觉上下文分支和视听对应分支互相促进，加入了一项损失保证由视觉上下文得到的推理图和原余弦相似度图分布一致，根据O与语音特征相似度排序，选择最高的“区域”对应的S中的定位图（H\*W），计算其与视听对应的余弦相似度的JS散度作为一致性损失。


# 30*.Learning Audio-Visual Source Localization via False Negative Aware Contrastive Learning(2023 cvpr)
本文针对假负例，提出了 False Negative Aware Contrastive (FNAC)方法，和我之前的想法差不多()但个人感觉，方法的设计感比较强，不太自然。
- 主要任务仍为NCE损失的对比学习，即下图中的$L_{contrast}$，这里的b是批量大小，因此b\*b\*h\*w表示每个视音对的余弦相似度图，而池化后的b\*b为sim(ai,vj)。

![Alt text](<Paper/Sound_Source_Localization/image/29.png>)
- false Negatives Suppression(FNS):首先在单一模态内计算邻接矩阵（样本间的相似度，类似于注意力分数），表示sim(ai,aj)和sim(vi,vj)。再将模态内的邻接作为模态间对比学习的软监督信号，比如，对于相邻的ai和aj，ai和vj也应该相邻，保持跨模态的一致性，因此有以下损失，其中距离用的是L1。这样可以有效抑制假负例（不在同一视频，但视觉和音频特征相似），使其距离和正例一样拉近
  
![Alt text](<Paper/Sound_Source_Localization/image/30.png>)
- True Negatives Enhancement(TNE):如果两个音频的特征区别大，那么他们指示的声源的视觉特征也应该区别大，由此来增强真负例的影响。具体来说，先得到对定位图的预测，设置阈值得到mask，mask视觉特征并池化得到声源区域的特征Z^s，不同样本的Z^s的相似性sim($Z_i^s$,$Z_j^s$)应与对应音频的相似性sim(ai,aj)一致，得到下图中的的损失L，形式与FNS类似。

![Alt text](<Paper/Sound_Source_Localization/image/31.png>)

# 31.Audio-Visual Grouping Network for Sound Localization from Mixtures(2023 cvpr)
本文针对多声源定位问题，在弱监督设置下（每个图片有个C维的声源类别标注），利用类别感知和transformer。
> 说实话，没大读懂这篇文章怎么实现的，而且网络的设计说不太通，可能实验结果比较好吧，自注意力机制强大的能力？

![Alt text](<Paper/Sound_Source_Localization/image/32.png>)
- learnable audio-visual class tokens:向transformer输入了C个可学习的类别token（数据集的声源类别为C），这C个token还输入FC+softmax，与对应的one-hot类别做交叉熵，以保证判别性，此为$L_1$
- transformer:将视觉特征+类别token共P+C个输入transformer，将听觉特征+类别token共C+1个输入另一个transformer，得到修正后的视觉特征/视觉类别token/听觉特征/听觉类别token
- Audio-Visual Grouping:在transformer后，为了得到category-aware audio-visual embeddings，以视/听觉特征作为Q和V，以对应的类别token作为K，计算注意力并与对应类别token相加。将得到的类别意识视听嵌入记为g，分别将C个视觉和听觉嵌入输入FC+sigmoid计算与y的交叉熵，y为GT标注，此处y可能为多标签，此为L_2,L_3。
- visual-audio align:从C个g中挑出N个对应GT声源类别的，计算同一类别的语音嵌入和视觉嵌入（视觉嵌入g和最最初的f_v（输入tansformer前的）的逐元素乘）的余弦相似度图，最大池化后计算对比损失。
> 关于挑出N个GT类别，文中的描述为Note
global audio and visual representations for N source embeddings are chosen from C cate-gories according to the corresponding ground-truth class
- 总损失为上面提到的四部分。推理时，没有说N个声源是怎么挑出来的，不理解，可能要看代码。
- 实验：将两个单声源的图水平拼起来，然后把声音直接加起来，感觉这样混合数据不太自然。

# 32.Audio-Visual Segmentation by Exploring Cross-Modal Mutual Semantics(ACM-MM 2023)
本文针对声源分割问题，单源/多源，设计了一个挺复杂的系统……本文指出，第17篇工作的模型，并没有很好的把音频和对象联系起来，更倾向于显著对象分割，因此提出了一种先给出潜在对象位置、再用语音显示的指导声源分割的方法。
> 这篇文章有几个地方比较模糊，和学长交流之后感觉还是有些矛盾之处，笔记便记得简洁一些。矛盾之处比如在潜在对象预测的时候使用了数据集里的类别信息，而在利用语音的时候，一直强调不知道语音类别；对于每个样本，构建了z_gt，给出了样本里所有声源的类别和二元mask，但实际多声源数据集中没有区分语义的mask，猜测是z_gt里就一个元素……

![Alt text](<Paper/Sound_Source_Localization/image/33.png>)
- Potential Sounding Object Segmentation:输入图片和N个可训练的对象嵌入到MaskForm，得到N个对象的分类（数据集的总类别+non-object）概率分布和每个对象的二元mask
- Audio Feature Extraction:用BEATs提取语音特征，经过MLP得到语音类别分布（K维）
- Audio Visual Mutual Semantics Alignment:将第一部得到的N个潜在对象，按类别和置信度滤去（滤去non-object，每个类别只留置信度最高的），用语音类别分布按类别加权剩下的潜在对象的mask
- 训练目标：对Potential Sounding Object Segmentation的损失函数设计比较抽象，此处不展开讲了；对Audio Visual Mutual Semantics Alignment就是mask的交叉熵。

# 33.Flowgrad: Using Motion for Visual Sound Source Localization(2023 ICASSP)
本文认为过去的方法大多没有考虑时间上下文，提出了三种利用光流的方法，分别是直接将光流和原定位图相乘、将视觉编码器增加一个光流通道（输入RGB+光流）、增加一个光流编码器。
> 本文提到，We only use Urbansas for evaluation because other
VSSL benchmarks have a bias towards static sound sources in
the center of the image, making the inclusion of motion infor-
mation unnecessary，其他的数据集的声源大多只是图像中心，这使运动信息没有必要。
