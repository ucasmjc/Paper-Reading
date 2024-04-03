- [1. MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model(2022 TPAMI)](#1-motiondiffuse-text-driven-human-motion-generation-with-diffusion-model2022-tpami)
- [2. MDM:Human motion diffusion model(2022 iclr)](#2-mdmhuman-motion-diffusion-model2022-iclr)
- [3. Human Motion Diffusion as a Generative Prior(2024 iclr)](#3-human-motion-diffusion-as-a-generative-prior2024-iclr)
- [4. MLD:Executing your commands via motion diffusion in latent space(2023 cvpr)](#4-mldexecuting-your-commands-via-motion-diffusion-in-latent-space2023-cvpr)
- [5. EDGE: Editable Dance Generation from Music(2023 cvpr)](#5-edge-editable-dance-generation-from-music2023-cvpr)
- [6. Mofusion: A framework for denoising-diffusion-based motion synthesis(2023 cvpr)](#6-mofusion-a-framework-for-denoising-diffusion-based-motion-synthesis2023-cvpr)
- [7. Avatars grow legs: Generating smooth human motion from sparse tracking inputs with diffusion model(2023 cvpr)](#7-avatars-grow-legs-generating-smooth-human-motion-from-sparse-tracking-inputs-with-diffusion-model2023-cvpr)
- [8. MotionGPT: Human Motion as a Foreign Language(2024 nips)](#8-motiongpt-human-motion-as-a-foreign-language2024-nips)
- [9. BeLFusion: Latent Diffusion for Behavior-Driven Human Motion Prediction(2023 iccv)](#9-belfusion-latent-diffusion-for-behavior-driven-human-motion-prediction2023-iccv)
- [10. HumanMAC: Masked Motion Completion for Human Motion Prediction(2023 iccv)](#10-humanmac-masked-motion-completion-for-human-motion-prediction2023-iccv)
- [11. Make-An-Animation: Large-Scale Text-conditional 3D Human Motion Generation(2023 iccv)](#11-make-an-animation-large-scale-text-conditional-3d-human-motion-generation2023-iccv)
- [12. Priority-Centric Human Motion Generation in Discrete Latent Space(2023 iccv)](#12-priority-centric-human-motion-generation-in-discrete-latent-space2023-iccv)
- [13. FineDance: A Fine-grained Choreography Dataset for 3D Full Body Dance Generation(2023 iccv)](#13-finedance-a-fine-grained-choreography-dataset-for-3d-full-body-dance-generation2023-iccv)
- [14. Enhanced Fine-Grained Motion Diffusion for Text-Driven Human Motion Synthesi(2024 aaai)](#14-enhanced-fine-grained-motion-diffusion-for-text-driven-human-motion-synthesi2024-aaai)

# 1. MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model(2022 TPAMI)
第一个应用扩散模型进行T2M生成的方法
- 问题设置：motion由一个动作序列$\theta_i \in R^{F\times d}$组成，F为帧数，d为pose statement的表征，不同数据集不同；训练时给定$(text_i,\theta_i)$
- Diffusion:扩散模型与文生图里的DDPM一致，固定方差，预测噪声。
- Cross-Modality Linear Transformer:文中主要设计了预测噪声的模型，由于运动生成任务中的变量长度可变，基于Unet的卷积网络不合适。如下图所示，训了一个text encoder，文本特征经交叉注意力加入生成；自注意力和交叉注意力层均使用线性注意力提高效率；为了加入时间条件（扩散的时间戳），将t编码后与文本特征相加，再在Stylization Block(蓝色块)中融入生成。
![Alt text](Paper\diffusion\image\image-26.png)
- Fine-grained Controlling:
  - Body Part-independent Controlling：为了实现对于身体部位的细粒度控制，采用噪声插值，即$\epsilon=\sum (\epsilon_i \times M_i)$，其中$\epsilon_i$是对于第i部位的text生成的噪声，$M_i$为one-hot向量。但是，文中发现直接插值效果不好，因为不同part噪声间差距太大，因此在采样时加入平滑项
  ![Alt text](Paper\diffusion\image\image-27.png)
  梯度引导不同part生成的噪声相近，文中没有指明梯度是相对谁的，理论上来说是对于$x_t$的，但感觉相对$x_t$的梯度直接加到噪声预测中不太对劲（其实也就差一个scale）。
  - Time-varied Controlling：为了实现对生成时间的可控生成，给定$\{text_{i,j},[l_{i,j},r_{i,j}]\}$，希望$[l_{i,j},r_{i,j}]$时间里生成$text_{i,j}$对应的动作。文中给出的方法是，分别生成m段动作，然后根据$[l_{i,j},r_{i,j}]$均pad到目标长度，然后如下式插值
  ![Alt text](Paper\diffusion\image\image-28.png)
- 该方法的明显缺陷是，动作生成是一个序列生成，而文中直接预测$R^{F\times d}$，循环依次生成的连续性应该更好，不过好像是T个姿态token输入transformer，也还行；并且，应该可以设置一些额外的先验损失
# 2. MDM:Human motion diffusion model(2022 iclr)
本文采用了transformer架构的DDPM模型进行动作生成，还加入了几个该领域的几何损失。
- 设置：给定条件，如audio，text，离散类别，生成姿态序列$\{x^i\in R^{J\times D}\}_{i=1}^N$，J为关节数，D为描述关节状态的表征维数。
- 扩散：文中采用了DDPM框架，follow DALLE-2直接预测x而非噪声，采用 classifier-free guidance训练
![Alt text](Paper\diffusion\image\image-30.png)
- 文中加入了运动生成领域的几种几何损失，可以增强物理特性并防止伪影，促进自然和连贯的运动。下面三个损失分别约束，位置（旋转关节时）、足接触和速度变化连续性，前两个不太显然，没有仔细研究其物理意义。
<p align = "center">  
<img src="Paper\diffusion\image\image-31.png"  width="300" />
</p>

- 模型：模型很简单，将条件编码后与时间戳编码相加作为输入token之一$z_{tk}$，与位置编码相加后和其他token（含噪声的姿态token）一起输入Transformer encoder。
![Alt text](Paper\diffusion\image\image-32.png)
- 编辑：文中声称可以在时间和空间上对某些部分编辑，采用的方法是，在采样时，保持不需要改变的部分，而只将待编辑的部分用噪声。

# 3. Human Motion Diffusion as a Generative Prior(2024 iclr)

- 动机：由于缺乏数据，MDM难以合成长的多段动作、只合成单人运动、缺乏细节控制（trajectory and end-effector tracking），本文引入基于扩散先验（预训练好的动作生成模型，文中是MDM）的三种组合形式：顺序组合、并行组合和模型组合，分别解决这三个问题。
- sequential composition:为了生成任意长的运动，文中提出DoubleTake方法，可以zero-shot将MDM生成的短动作（10s）合成为长序列，并在中间加入过渡，这样运动的每个片段都可以用不同的文本提示和不同的序列长度来控制。具体来说，采用一种双阶段生成方法，第一阶段，并行生成所有片段，但是，在去噪的每一步，相邻片段进行“握手”，每个片段的第一秒与上一片段的最后一秒取平均（1s可以改变），使得当前片段动作的前缀被迫等于前一个动作的后缀；由于语义不同的动作之间的转换中出现伪影和不一致，第二阶段，对第一阶段结果进行细化，对于握手后的$(s_i,\tau_i,s_{i+1})$，采用impainting方法（保持上下文区域不变，而只在部分区域加噪去噪），重新细化生成握手部分$\tau_i$，并且进一步采用soft mask，在$\tau_i$周围也加入了线性衰减的mask（即，也加噪去噪，但权重$\in (0,1)$），以适应$\tau_i$的细化调整。
<p align = "center">  
<img src="Paper\diffusion\image\image-33.png"  width="300" />
</p>

- parallel composition：为了生成两个人的动作，文中提出一种few-shot方法，设计了ComMDM模块（单层transformer）在冻结MDM的两个实例之间进行协调。如下图所示，对于MDM的第n层，将两个MDM的输出一起输入ComMDM，得到修正项（对称的），加入回原输出即可。除此之外，该模块还会根据输入人的初始位姿D预测新位姿（否则两个人就重合了）

<p align = "center">  
<img src="Paper\diffusion\image\image-34.png"  width="300" />
</p>

- Fine-Tuned Motion Control：
  - Single Control Fine-Tuning：文中希望在保持某些设定特征的基础上进行生成，方法比较直接，对模型进行微调，此时预设特征区域（以轨迹控制为例，trajectory）的噪声总为零而保持GT，从而模型在尝试重建其余特征时学会依赖trajectory。而在采样时，也只需采取类似措施。
  <p align = "center">  
  <img src="Paper\diffusion\image\image-35.png"  width="300" />
  </p>

  - DiffusionBlending：采用model composition。对于所有可能的控制任务微调是不现实的，对于左手+轨迹的控制，文中提出DiffusionBlending可以根据微调好的单独的左手和轨迹控制模型得到理想控制，具体来说，借鉴了classifier-free的思想，对两个MDM的结果进行了插值。
  ![Alt text](Paper\diffusion\image\image-36.png)
# 4. MLD:Executing your commands via motion diffusion in latent space(2023 cvpr)
把LDM拿过来用了，MLD=VAE+LDM
- 先训了一个motion VAE，以L个motion token和分布token为encoder的输入，得到$\mu,\sigma$预测，计算一个KL loss；再以此采样得到潜在表示z，作为decoder中交叉注意力的K和V，L个zero motion token输入decoder得到L个token，与原token做一个MSE损失。
- Motion Latent Diffusion Model：很直接的LDM，transformer架构+预测噪声
- Conditional Motion Latent Diffusion Model：条件生成也很直接，task-specifical的encoder，条件token串联进去噪模型（文中说比交叉注意力有效），classifier-free训练。

# 5. EDGE: Editable Dance Generation from Music(2023 cvpr)
舞蹈生成
- pose表征：24个关节的六自由度表示+root变换（坐标，3）+脚与地面的接触情况（脚跟+脚趾，2*2），共24\*6+3+4=151，一段舞蹈为$x\in R^{N\times 151}$
- 模型：DDPM+预测原图+MDM的辅助运动学损失+classifier-free guidance
- Editing：挺简单的，就是用了inpainting，用Binary mask保持参考动作需控制的部分，即可实现joint-wise/temporal编辑
- 长舞蹈：和Human Motion Diffusion as a Generative Prior类似，强行约束每个clip的前n秒与前一个clip一致，后n秒与后一个clip一致，如下图的颜色（batch表示batch个clip拼在一起）
  <p align = "center">  
  <img src="Paper\diffusion\image\image-38.png"  width="300" />
  </p>

- 模型：采用music预训练模型Jukebox提取音乐特征，编码后与时间戳特征一起输入交叉注意力，时间戳还进一步加到Film里（经常用在小样本里，大概就是根据输入的条件预测权重+偏置，对特征图做个线性变换）
<p align = "center">  
<img src="Paper\diffusion\image\image-37.png"  width="300" />
</p>

- 除此之外，文中还提了几个新的评价指标，没细看
# 6. Mofusion: A framework for denoising-diffusion-based motion synthesis(2023 cvpr)
将DDPM用到运动生成，然后加了几个运动学损失
- 运动学损失：单纯DDPM生成的动作不能保证在物理和解剖学上是合理的，会出现运动抖动、非法骨骼和脚滑动等伪影。引入三种运动学loss，通过重参化，根据预测的噪声得到预测的$x_0$，由于t较大时$x_0$很不准，因此引入了一个时变的损失权重，取$k=\bar \alpha_t$
  - skeleton-consistency loss：为了确保了合成运动中的骨骼长度随时间保持一致，最小化骨骼长度的时间方差，n为骨骼数（根据关节状态转换的？）
  
    ![alt text](Paper\diffusion\image\image-39.png)
  - anatomical constraint：为了惩罚骨骼长度的左/右不对称，计算对称位置骨骼长度的L1损失
  - 最后又加了一次GT约束
  
    ![alt text](Paper\diffusion\image\image-40.png)
  - 文中还提到，通过这种方法（重参化+时变权重）可以引入其他运动学损失
- 模型结构：DDPM+预测噪声，U-net+transformer插入条件，music-base用频谱图输入，text-base用CLIP
![alt text](Paper\diffusion\image\image-41.png)

# 7. Avatars grow legs: Generating smooth human motion from sparse tracking inputs with diffusion model(2023 cvpr)
设计了一种基于MLP的扩散模型，用于在给定稀疏上半身信号的条件下预测全身姿势（该情景多出现于AR/VR头戴式设备），相当于一个条件动作生成模型。
- MLP-base：没啥好说的，先提出了一个根据输入p预测输出y的MLP模型，输入的p为D维特征（仅包含上半身关节），预测的y为S维特征（包含全身关节）
<p align = "center">  
<img src="Paper\diffusion\image\image-42.png"  width="300" />
</p>

- 扩散：DDPM+预测原图，将去噪x和约束条件p分别经过FC后连接起来再经过线性层输入MLP进行去噪。文中提到，MLP结构导致模型对时间戳特征不敏感，过去直接将时间戳嵌入作为网络的附加输入，本文提出Block-wise Timestep Embedding，**在每个MLP block**都重复插入时间戳特征
![alt text](Paper\diffusion\image\image-43.png)

# 8. MotionGPT: Human Motion as a Foreign Language(2024 nips)
过去运动模型的监督是特定于任务的，很难有效地推广到集外任务或数据，因为缺乏对动作和语言之间关系的全面理解。本文构建了一个预训练的motion-language模型MotionGPT，可以推广到各种运动的下游任务，并从更多的动作语言数据中学习深入的相关知识。
![alt text](Paper\diffusion\image\image-44.png)
- Motion Tokenizer：为了将motion编码成语言，文中训练了一个VQ-VAE，作为tokenizer，此过程包含时间和特征维度的下采样
- Motion-aware Language Model:VAE将motion编码成一种“外国语言”，文中构建了词汇表 $V={V_{text},V_{motioin}}$，即语言和运动token在输入和输出中混合起来，其中有一些特殊token标记运动的开始和结束。最终，文中使用next token prediction的自回归目标对T5模型进行监督训练。整个过程分为以下两个阶段
  - Motion-language Pre-training Stage：首先是无监督预训练，使用完形填空方法；监督预训练在motion-text成对数据集上，输入text/motion让模型输出对应的motion/text，进行翻译
  - 指令微调：在motion相关任务上设计prompt并进行指令微调。

# 9. BeLFusion: Latent Diffusion for Behavior-Driven Human Motion Prediction(2023 iccv)
本文针对human motion prediction任务，根据人的历史运动预测接下来的运动，认为过去的动作生成方法为了兼顾多样性，而没能很好地和过去动作保持一致，无法直接应用到HMP。文中构建了一个将behaviour与motion/pose分离的潜在空间（我的理解是，编码的behaviour嵌入与具体的motion/pose无关），提出BelFusion模型，但我觉得这篇文章有点不靠谱，只写个思路……
- Behavioral Latent Space：为了编码behaviour潜在表示，先训了个VAE，即$p_{\theta},A_{\omega}$；然后搞了个条件VAE，$p_{\theta}$编码的是$Y_e$，这包括历史$x_m$和预测Y（可能含有噪声），得到的是behaviour潜在表示，而历史$x_m$经过g编码作为条件和z一起输入解码器$B_{\phi}$，预测$T_e$
![alt text](Paper\diffusion\image\image-45.png)
- Behavior-driven HMP：总的来说是一个LDM，训练时，将去噪后的z与GT的$p_{\theta}(Y_e)$做L1，再加了一个约束损失，保证$g_{\alpha}$的条件性
![alt text](Paper\diffusion\image\image-46.png)
采样时，先将$z_T$去噪，期间$x_m$为交叉注意力的条件；去噪后的$z_0$与条件编码后的$g_{\alpha}(x_m)$一起输入解码器得到最终预测。
![alt text](Paper\diffusion\image\image-47.png)
- 之所以这么麻烦，文中的意思是，相当于把生成连贯动作的任务交给$B_{\phi}$，它确保潜在behaviour表征被解码为任何正在进行的运动的平滑且合理的延续。
# 10. HumanMAC: Masked Motion Completion for Human Motion Prediction(2023 iccv)
也是一篇HMP的工作，和上一篇一样，关注对历史的可控约束，但感觉方法有点麻烦，不仔细研究了……
- 过去缺陷：多个损失--超参数多；双阶段--LDM的VAE；可控性不行
- 模型：先用DCT变换得到频谱图，再做DDPM，这里的动作序列是历史+预测
- masked方法：采样时，左侧分支关注历史，将历史最后一帧重复后做DCT再加入噪声，再去噪；右侧就正常去噪，最终就是两个分支结合起来，好吧，可能中间有什么数学结论是我不了解的。
<p align = "center">  
<img src="Paper\diffusion\image\image-48.png"  width="300" />
</p>

# 11. Make-An-Animation: Large-Scale Text-conditional 3D Human Motion Generation(2023 iccv)
想法很简单，text2motion数据集难采集，因此利用text2image预训练扩充数据。
- Text Pseudo-Pose (TPP) 数据集：从text2image数据集中找到关于人体pose的，再用预训练模型提取人体的关键点，得到3D pose特征，组织出一个text-pose数据集
- 训练分为两阶段，第一阶段在TTP上训练一个text2pose的扩散模型（Unet结构，如下图），第二阶段在text2motion数据集上微调，并加入1维时序卷积做attention，没有caption的motion数据进行classifier-free的训练。

<p align = "center">  
<img src="Paper\diffusion\image\image-49.png"  width="300" />
</p>

# 12. Priority-Centric Human Motion Generation in Discrete Latent Space(2023 iccv)
挺新奇的想法，但感觉有点扯，偏离了主流扩散模型……不仔细记录方法了，只记录一下思路。
- 动机：MDM和LDM主要处理连续空间内的潜在特征，但将扩散模型集成到离散空间中（如VQVAE的code）的探索仍然不够充分；先前研究统一对待所有motion token，忽略了序列中token之间固有的差异。本文据此提出priority-centric motion discrete diffusion model (M2DM)。
- 整体思路:先训了一个VQ-VAE，关于motion序列编码了一个信息丰富的codebook，再对得到的离散潜在表示做扩散，其中，不同token（相当于不同帧的表征）的加噪去噪处理有所不同（本文还提出了两种方法来评价不同code的priority）
- 离散扩散：本文中的离散是指，潜在表征是离散的(代表的是code)，因此提出了一种新的加噪和去噪方法，具体方法挺复杂，没研究它是怎么保证采样多样性的，有需要再仔细读一下。
# 13. FineDance: A Fine-grained Choreography Dataset for 3D Full Body Dance Generation(2023 iccv)
本文提出了一个大型的3D舞蹈生成（编舞）数据集，包含多个流派，并提出一个进行舞蹈生成的模型，主体结构是由MDM修改来的，具体做了针对手部和肢体的细粒度生成。
# 14. Enhanced Fine-Grained Motion Diffusion for Text-Driven Human Motion Synthesi(2024 aaai)
本文主要关注运动生成中的可控生成，给定稀疏的关键帧进行text2motion，过去的inpainting方法在此情况下效果不佳，因为过于系数的关键帧的去噪过程中会被视作噪声，而不能提供很好的条件。本文将关键帧-aware融入到训练过程，提出DiffKFC，在训练时实现KeyFrames Collaborated
![alt text](Paper\diffusion\image\image-50.png)
- 模型结构：给motion序列加噪后编码成token和时间戳、text一起输入transformer decoder，motion序列被mask后得到关键帧并编码输入关键帧encoder，提取的特征作为条件输入transformer decoder的交叉注意力层。
- 关键帧encoder：由于关键帧是稀疏的，直接提取特征效果不好，本文提出了Dilated Mask Attention Module，将稀疏关键帧特征逐渐膨胀，具体操作和mask attention比较像，就是注意力计算时只取有效帧的Value，并且范围随扩散步数逐渐扩大。
<p align = "center">  
<img src="Paper\diffusion\image\image-51.png"  width="400" />
</p>

- Transition Guidance：由于关键帧附近的动作不自然，文中引入了DCT，编码关键帧前后L帧这么一个序列，并以以下损失为guidance（主要是引导消除高频分量），修改原classifier-guidance，用梯度引导采样。

![alt text](Paper\diffusion\image\image-52.png)
- Classifier-free Guidance：文中采用了针对关键帧条件的
