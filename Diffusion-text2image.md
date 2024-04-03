- [1. Understanding Diffusion Models: A Unified Perspective](#1-understanding-diffusion-models-a-unified-perspective)
  - [ELBO](#elbo)
  - [VAE](#vae)
  - [Hierarchical VAE](#hierarchical-vae)
  - [Variational Diffusion Models](#variational-diffusion-models)
  - [Three Equivalent Interpretations](#three-equivalent-interpretations)
- [2. DDPM: Denoising Diffusion Probabilistic Models(2020.12)](#2-ddpm-denoising-diffusion-probabilistic-models202012)
- [3. DDIM: Denoising Diffusion Implicit Models(iclr 2021)](#3-ddim-denoising-diffusion-implicit-modelsiclr-2021)
- [4. Score-Based Generative Modeling through Stochastic Differential Equations(ICLR 2021)](#4-score-based-generative-modeling-through-stochastic-differential-equationsiclr-2021)
- [5. IDDPM:Improved Denoising Diffusion Probabilistic Models(2021.2)](#5-iddpmimproved-denoising-diffusion-probabilistic-models20212)
- [6. Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models(iclr 2021)](#6-analytic-dpm-an-analytic-estimate-of-the-optimal-reverse-variance-in-diffusion-probabilistic-modelsiclr-2021)
- [7. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps(2022 nips)](#7-dpm-solver-a-fast-ode-solver-for-diffusion-probabilistic-model-sampling-in-around-10-steps2022-nips)
- [8. Classifier-Guidance: Diffusion Models Beat GANs on Image Synthesis(2021 nips)](#8-classifier-guidance-diffusion-models-beat-gans-on-image-synthesis2021-nips)
- [9. Classifier-Free Diffusion Guidance(2021 nips)](#9-classifier-free-diffusion-guidance2021-nips)
- [10. CLIP: Learning Transferable Visual Models From Natural Language Supervision(2021.2)](#10-clip-learning-transferable-visual-models-from-natural-language-supervision20212)
- [11. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models(2021 openai)](#11-glide-towards-photorealistic-image-generation-and-editing-with-text-guided-diffusion-models2021-openai)
- [12. DALLE-2:Hierarchical Text-Conditional Image Generation with CLIP Latents(2022 openai)](#12-dalle-2hierarchical-text-conditional-image-generation-with-clip-latents2022-openai)
- [13. Imagen:Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding(2022 nips)](#13-imagenphotorealistic-text-to-image-diffusion-models-with-deep-language-understanding2022-nips)
- [14. LDM: High-Resolution Image Synthesis with Latent Diffusion Models](#14-ldm-high-resolution-image-synthesis-with-latent-diffusion-models)
- [15. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation(2023.5)](#15-dreambooth-fine-tuning-text-to-image-diffusion-models-for-subject-driven-generation20235)
- [16. DiT:Scalable Diffusion Models with Transformers(2023.5)](#16-ditscalable-diffusion-models-with-transformers20235)
- [17. LoRA:Low-Rank Adaption of large language model(2021.10)](#17-loralow-rank-adaption-of-large-language-model202110)
- [18. Controlnet: Adding Conditional Control to Text-to-Image Diffusion Models(2023.11)](#18-controlnet-adding-conditional-control-to-text-to-image-diffusion-models202311)

> 2-9的主要推导见手写笔记
# 1. Understanding Diffusion Models: A Unified Perspective
这是接触的第一篇diffusion的tutorial，较全面地了解了变分下界ELBO和基于似然的生成模型VAE、HVAE，最终研究了Variational Diffusion Models，首先推导了DDPM设置下的理论ELBO和具体优化方法，了解利用NN建模噪声参数$\alpha_t$的方法，最终推导了三种关于均值的参数化方法，对应VDM的三种等价解释，预测噪声/原图/得分，对扩散模型建立了一个较全面的认知。
## ELBO
对于一个给定样本x和真实分布$p^*$，生成模型希望建模p以极大化x的似然来接近$p^*$（最大化p的对数似然，等价于，最小化p与$p^*$的KL散度），并借助隐变量z建模低维表示。但是直接极大化似然不现实$p(x)=\int f(x,z)dz$，转而利用条件分布$p(x)=\frac{p(x,z)}{p(z|x)}$，其中分母的条件分布由$q_\phi$建模得到，推导可得

$$log\ p(x)=E_{q_\phi(z|x)}[log\frac{p(x,z)}{q_\phi(z|x)}]+D_{KL}[q_\phi(z|x)||p(z|x)]\geq E_{q_\phi(z|x)}[log\frac{p(x,z)}{q_\phi(z|x)}]$$

$\geq$是由于KL散度恒大于等于0，上式说明Evidence即$log\ p(x)$具有一个Lower BOund，因此任务目标由最大化似然转为最大化该ELBO，VAE，层次VAE，扩散模型的任务目标均从该ELBO出发进行优化。

## VAE
VAE将$p_\theta(x|z),q_\phi(z|x)$均建模为高斯分布，将ELBO表示为以下形式

$$E_{q_\phi(z|x)}[log\frac{p(x,z)}{q_\phi(z|x)}]=E_{q_\phi(z|x)}[log\frac{p_\theta(x|z)p(z)}{q_\phi(z|x)}]=E_{q_\phi(z|x)}[log\ p_\theta(x|z)]-D_{KL}[q_\phi(z|x)||p(z)]$$

其中，第一项为保证重建，第二项保证x到z的编码符合p(z)先验分布而防止其坍缩成狄拉克函数。

## Hierarchical VAE
Hierarchical VAE将VAE扩展到具有T层次结构的潜在变量，MHVAE(Markovian Hierarchical Variational Autoencode)的ELBO如下式
![alt text](Paper\diffusion\image\image-53.png)
这和VDM(Variational Diffusion Models)的ELBO一致

## Variational Diffusion Models
如HVAE中所说，VDM和MHVAE的ELBO形式一致，其实相当于受限条件下的MHVAE（1.z与x维度一致；2.每个时间步的潜在编码器被预先定义为高斯模型；3.潜在编码器的高斯参数随时间步变化，并且最终时间步长T的潜在分布为标准高斯分布），即（此为DDPM的设置）

$$q(x_t|x_{t-1})=N(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)$$

$$p(x_T)=N(X_T;0,I)$$

推导可得

$$q(x_t|x_0)=N(x_t;\sqrt{\bar\alpha_t}x_{0},(1-\bar\alpha_t)I)$$

VDM的ELBO推导可得以下形式
![alt text](Paper\diffusion\image\image-54.png)
而在具体处理时，重建项往往近似到去噪项中一起处理，而先验匹配项总为0不做贡献，因此我们关注去噪匹配项的优化，总的优化目标为

$$argmin_\theta E_{t\sim U\{2,T\}}[E_{q(x_t|x_0)}[D_{KL}(q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t))]]$$

具体来说，该项希望模型$p_\theta$根据$x_t$预测的$x_{t-1}$分布匹配ground truth（由前向过程计算可得）分布$q(x_{t-1}|x_t,x_0)=N(x_{t-1};\mu_q(x_t,x_0),\sum_qI)$，其中$\sum_q$预先确定，只需建模$\mu_\theta(x_t)$预测$\mu_q(x_t,x_0)$，进而最小化两个分布的KL散度可得（两个高斯分布的KL散度具有解析解）

$$argmin_\theta [||\mu_\theta-\mu_q||_2^2]$$

上式中省略了标量权重（在大部分方法中做了如此简化），而$\mu$的三种参数化推导导致了VDM的三种解释。

## Three Equivalent Interpretations
第一种解释：VDM等价于，建立一个NN，可以根据被加入任意噪声的图片$x_t$和t，来预测原图，经推导可得

$$\mu_q(x_{t-1}|x_t,x_0)=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_t+\sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{1-\bar\alpha_t}$$

进而可将去噪匹配项表示为

$$argmin_\theta [||x_\theta(x_t,t)-x_0||_2^2]$$

第二种解释，VDM等价于，建立一个NN，预测从$x_t$到$x_0$的原噪声$\epsilon$，根据

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$

可将$x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}$带入$\mu_q$得到新的参数化

$$\mu_q(x_{t-1}|x_t,x_0)=\frac{1}{\sqrt{\alpha_t}}x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}\sqrt{\alpha_t}}\epsilon$$

进而可将去噪匹配项表示为

$$argmin_\theta [||\epsilon_\theta(x_t,t)-\epsilon||_2^2]$$

第三种解释，VDM等价于，建立一个NN，预测任意噪声水平下$x_t$的score function即$\triangledown log\ p(x)$，样本在数据空间的梯度。形象地说，通过预测给定样本x的score，可以得到使似然增大的方向，从而可以从样本空间任意点$x_T\sim N(0,I)$开始迭代地根据score更新位置来产生样本。经推导可得，

$$x_0=\frac{x_t+(1-\bar\alpha_t)\triangledown_{x_t} log\ p(x_t)}{\sqrt{\bar\alpha_t}}$$

代入将$\mu_q$参数化为

$$\mu_q(x_{t-1}|x_t,x_0)=\frac{1}{\sqrt{\alpha_t}}x_t+\frac{1-\alpha_t}{\sqrt{\alpha_t}}\triangledown_{x_t} log\ p(x_t)$$

容易发现，$\triangledown_{x_t} log\ p(x_t)=-\frac{1}{\sqrt{1-\bar\alpha_t}}\epsilon$，二者仅差一个标量乘子，并且score的方向，使似然增大的方向，恰为噪声的反方向。

进而可将去噪匹配项表示为

$$argmin_\theta [||s_\theta(x_t,t)-\triangledown_{x_t} log\ p(x_t)||_2^2]$$

其中，GT得分无法直接得到，具体的得分匹配损失需要做进一步推导和处理，见 Score-Based Generative Modeling through Stochastic Differential Equations文章的手写笔记。
# 2. DDPM: Denoising Diffusion Probabilistic Models(2020.12)
主要形式已在A Unified Perspective中推理论证
- 前向过程：$\alpha_t$/$\beta_t$固定，因此无可学习参数，ELBO中的先验匹配项忽略，由于$\alpha_t$递减趋向于0，因此$q(x_T|x_0)$会趋向于$N(0,I)$
- 后向过程：方差固定（可学习设置在实验中表现不好），均值参数化为关于$\epsilon$的形式（比预测均值和原图效果好），对应的去噪匹配项损失亦关于$\epsilon$，即训练模型以预测原始噪声，根据任何噪声水平输入的图片$x_t$和t。
- training: 采样图片$x_0$，采样时间t，采样$\epsilon$从标准正态分布中；直接求得$x_t$；将$x_t$和t输入模型预测$\epsilon_{\theta}$，计算损失并反向传播
- sampling: 采样$x_T$从标准正态分布，从1到T逐渐去噪，即从$p_{\theta}(x_{t-1}|x_t)=N(x_{t-1};\mu_{\theta}(x_t,t),\sigma_t)$中采样（$\mu_{\theta}(x_t,t)$是关于$\epsilon_{\theta}$,t和$x_t$的式子）。此过程中，T大时效果好（本文为1000），因为大的 T 使逆过程接近高斯分布，因此用高斯条件分布建模的生成过程成为一个很好的近似（T小的时候，$q(x_T|x_0)$也不是标准高斯分布嘛）
- 简化的任务目标：ELBO中的重建项被简化为先验匹配项中的首项；并且，计算损失时直接忽略了关于$\epsilon$的L2损失前的系数，文中称，这种简化会使t较小情况下的损失权重相对降低，从而使模型关注t较大时的困难去噪过程。

![Alt text](Paper\diffusion\image\image.png)
- 一些胡思乱想：
  - 感觉有点绕的一个地方:训练时，目标$\epsilon$其实是一个抽象的复合的“初始噪声”，它决定了$x_0$到$x_t$，但不是简单的从$x_t$直接指向$x_0$，也不是近邻的$x_t$指向$x_{t-1}$；所以说，当推理时，这么一个预测的$\epsilon_{\theta}$是为了得到$p_{\theta}(x_{t-1}|x_t)$的分布，这个过程是按公式推出来的，同样，这个分布的均值期望是$x_{t-1}$。之所以会牵扯到这么个“原始噪声”，是因为我们把$\mu_{\theta}$中的$x_0$带入了“加噪过程中的表达式”，从而模型试图恢复加噪过程。
  - 但感觉有问题：让模型预测这么一个$\epsilon$是否不太合理？特别是$x_t$和$\epsilon$需要经过加权线性组合才得到$x_{t-1}$的期望，而这个权重是变化的且人为设置的。似乎也没问题，因为我们还向模型输入了t，这个t就与“权重”直接相关，比如，决定了当前图片$x_t$中$x_0$与方差的比值。剩下的就是让模型自己学了，emmm，靠NN的能力了。
  - 我突然明白我上边bb这两段是在哪里疑惑了。训练时，目标$\epsilon$是$x_t=a\times x_0+b\times \epsilon$，也就是说，$\epsilon$决定了$x_0$到$x_t$，但是真正推理时，预测的$\epsilon$用来计算$x_t$到$x_{t-1}$，尽管也是推导出来的，但是为什么不直接反推到$x_0$？也就是说，模型会对输入的$x_t$预测一个恢复到$x_0$的噪声，但是我推理时用这个噪声来逐步去噪。（我当然知道一步到位相当于预测原图了，效果肯定不好，相当于放弃了扩散模型的优势，这里也是采样进一步加速的入手点之一？）
  - 而之所以必须一步步去噪，是因为DDPM的假设便是如此，前向过程一步步加噪，反向过程就一步步去噪，前向过程只是根据高斯分布的性质走了捷径，推导中的$\epsilon$只是参数化的手段，而非具体的物理意义？
  - 疑惑解决了：
  ![Alt text](Paper\diffusion\image\image-1.png)

# 3. DDIM: Denoising Diffusion Implicit Models(iclr 2021)
从更高的角度，通过避开$q(x_t|x_{t-1})$，将DDPM的训练过程由马尔科夫扩散推广到非马尔科夫过程，而二者训练目标是共享的，即，并未改变DDPM的训练方法。核心是推导出了下式：
![Alt text](Paper\diffusion\image\image-2.png)
- 通过新的推到，训练过程没有变化，但生成过程中方差$\sigma_t$是自由参数，为DDPM带来了新鲜的结果。当$\sigma_t$=0时，给模型变为Denoising Diffusion Implicit Models，Implicit是指，这变成一个隐式的概率模型，当最初的$x_T$确定，之后的每一步均确定，而非DDPM那样在每一步采样。
- 加速采样：由于文章将训练过程推广到非马尔可夫过程，采样时也可将原模型视为非马尔可夫过程下训好的模型，从而根据[0,T]的一个子序列$\tau$进行非马尔科夫的采样，从而减少步数而加速采样。
- 文中发现，DDIM由于确定性采样，潜在空间的插值可以导致样本的插值，即可以通过潜在空间控制生成的样本，是DDPM做不到的。并且表现出一致性，即同一$x_T$，在不同T下生成图像的高层特征不变。



# 4. Score-Based Generative Modeling through Stochastic Differential Equations(ICLR 2021)
Song Yang老师的iclr2021 outstanding paper，提出了一个一般化的生成扩散模型理论框架，将DDPM,SDE,ODE之类的都联系了起来。我没太研究明白，但大体意思应该是懂了。
- DDPM在前向和后向过程中都是离散过程，但引入SDE可以将此过程看作时间上的**连续**变化过程，而DDPM相当于不同程度离散化（T）的结果。引入SDE可以在分析时借助连续性SDE的相关方法，而在实践时适当的离散化。
- SDE理论框架--前向过程：$dx=f_t(x)dt+g_tdw$（这个微分方程是假设，dw是布朗运动），$f_t$为drift系数，$g_t$为diffusion系数。这可以看作是离散化过程在$\delta t$->0时的极限:$x_{t+\delta t}-x_t=f_t(x_t)\delta t+g_t\sqrt{\delta t}\epsilon$，越小的$\delta t$（可以理解为$\frac1T$）表示对SDE越好的近似。
- SDE理论框架--逆向过程：$dx=[f_t(x)-g_t^2\nabla_x log p_t(x)]dt+g_tdw$（推导省去）由于$f_t,g_t$是预先设定好的，此方程中唯一不知道的是$\nabla_x log p_t(x)$，也就是score。
- SDE理论框架--得分匹配损失函数：为了使模型预测$\nabla_{x_t} log p(x_t)$，使其在任意$x_t$下预测$\nabla_{x_t} log p_t(x_t|x_0)$（加权求和后为$\nabla_{x_t} log p_t(x)$），再对$x_t$求期望得到得分匹配目标函数：$E_{x_0\sim p(x_0)}E_{x_t\sim p(x_t|x_0)}[||s_\theta (x_t,t)-\nabla_{x_t} log p_t(x_t|x_0)||]$。其中，若$f_t(x_t)$为仿射变换，则$p(x_t|x_0)$也是高斯分布。
- 解释DDPM:根据其$q(x_t|x_{t-1})$，可以推得SDE:$dx=-\frac12\beta(t)xdt+\sqrt{\beta(t)}dw$。如统一视角那篇文章中说的，DDPM预测的噪声实际就是score的反方向，差了个标量系数。
- 除此之外，这篇文章还引入了概率流ODE，如SDE解释DDPM，ODE对应DDIM，经过推导可得，上述SDE前向过程完全等价于一系列新的SDE（具有不同的方差），而当方差取0时，SDE退化为ODE，也就是“概率流ODE”，用神经网络近似其中的得分函数，因此又称为“神经ODE”。

# 5. IDDPM:Improved Denoising Diffusion Probabilistic Models(2021.2)
没大有营养的一篇，提出的一些具体方法（如采样加速）没仔细看
- 动机：尽管DDPM取得不错的效果，但是对数似然不如其他生成模型（我怀疑指的是loss的优化），作者认为这是由于DDPM中方差固定
- 可学习方差:作者让模型多预测一个维度，进行非线性变换后作为方差。具体来说，因为$L_{simple}$中不含方差，作者在原损失$L_{simple}$又加了一项$L_{vlb}$（损失权重很小），前者只优化均值，后者只优化方差
- 修改noise schedule:作者发现DDPM的线性noise在小分辨率时效果不好，提出一种新的余弦策略
- reducing gradient Noise：作者发现$L_{vlb}$优化时损失波动很厉害，难以优化，从而提出一种训练时的重要性采样方法(对t的采样)，使$L_{vlb}$的优化变得容易，从而将之前的复合loss变为单纯的$L_{vlb}$，取得最佳效果


# 6. Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models(iclr 2021)
关于离散扩散生成模型，reverse过程中的方差，DDPM分别假设数据服从两种特殊分布推出了两个可用的结果，效果差别不大；DDIM将方差设为超参数，且当方差为0时为DDIM，但此时效果不太好；IDDPM用神经网络预测方差。本文，提出reverse过程中的方差有个解析解，从而提高了采样速度和质量，该方法无需改变DDPM的训练。
![Alt text](Paper\diffusion\image\image-4.png)
- 过去$p(x_{t-1}|x_t,x_0)$中$x_0$直接参数化带入以得到$p(x_{t-1}|x_t)$，但由于$x_0$的不准确性，这个过程更应该用分布描述，即
  ![Alt text](Paper\diffusion\image\image-5.png)
从而转为预测$p(x_0|x_t)$进一步得到$p(x_{t-1}|p_t)$，本文中对$p(x_0|x_t)$的假设为$N(\mu (x_t),\sigma_t^2I)$，即方差为对角阵，且对角线元素相同。之后的Extended-Analytic-DPM将该假设扩展到允许对角线元素互不相同的情况，提出一种双阶段策略，用预训练好的DDPM模型继续训一个方差模型（以一个新提出的目标函数）。
- 该公式为论文中给出的基于得分函数的结果（笔记中的推导是基于噪声的），修正后的均值与DDIM一致，但方差与之前相比多了一个长串的修正项。式中的期望在采样时用蒙特卡洛估计，由于该期望只与时间步n有关，因此对于一个数据集和预训练模型，只需要估计一次，几乎不增加计算量。
- 本文还推出了该方差的上下界，以及一种优化策略，没细看

# 7. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps(2022 nips)
无需训练的高效采样方法，原本想仔细研究一下，但发现对ODE太不了解，性价比不高，DPM-solver已经封装的很好了，拿来主义了
- 过去SDE框架下的扩散模型在采样时耗时过多，由于SDE的随机性，求解时步长不能太大，这导致求解次数较多；而基于ODE求解的大约在50次左右，本文降到了10次
- 本文重新分析简化了ODE采样方法，推出了一个解的精确公式，避免了过去将整个神经网络参数输入ODE求解器，具体咋做的我也不太清楚。对于ODE的线性部分分析求解，对于非线性部分，通过应用变量变化，等效地简化为神经网络的指数加权积分，好吧我也不太懂。


# 8. Classifier-Guidance: Diffusion Models Beat GANs on Image Synthesis(2021 nips)
条件生成主要分为Classifier-Guidance和Classifier-Free，前者只需在预训练好的无条件扩散模型的基础上训练一个分类器，节约成本，而后者在训练时便加入了条件引导，从零训练，从而不需再额外训练classifier，且可以实现更细的控制。本文是第一篇使用Classifier-Guidance的类条件生成方法（ADM），且ADM修改了U-net结构，加入了更多的attention结构和GN等。
- 条件生成的核心为近似$p(x_{t-1}|x_t,y)$，在原$p(x_{t-1}|x_t)$的基础上加入classifier的梯度引导
- 对于DDPM，只需在原均值的基础上加一个关于条件的额外项
![Alt text](Paper\diffusion\image\image-6.png)
- 对于DDIM，将噪声替换为新的形式即可
![Alt text](Paper\diffusion\image\image-7.png)
- 梯度缩放：上述式子中的s通常大于1，从而提高结果与y的相关性
- 在接下来的工作中《More Control for Free! Image Synthesis with Semantic Diffusion Guidance》，将条件由类别推广到一般形式，推理可得以下形式，sim为相似度度量函数，一般将$x_t$和y分别用编码器提取特征后计算。
![Alt text](Paper\diffusion\image\image-8.png)
- 值得注意的是，该分类器的目标是接受含任意噪声$x_t$并尝试预测条件信息 y，因此该分类器需要和扩散模型一起临时训练，而不能直接用预训练模型。当采样时，该分类器相对y和$x_t$的梯度可以引导$x_t$越来越像y类（类似于对抗学习）

# 9. Classifier-Free Diffusion Guidance(2021 nips)
更加直接和力大砖飞的方法，在训练时直接预测条件下的噪声/得分，定义$p(x_{t-1}|x_t,y)=N(x_{t-1};\mu (x_t,y),\sigma_t^2 I)$，进一步$\mu(x_t,y)$参数化后得到目标函数$E[||\epsilon - \epsilon_{\theta}(x_t,y,t)||^2]$

为了平衡相关性和多样性，也引入$\omega = \gamma -1$，采样时有$\~\epsilon_{\theta}(x_t,y,t)=(1+\omega)\epsilon_{\theta}(x_t,y,t)-\omega \epsilon_{\theta}(x_t,t)$，但是训练时不需要分别训条件/非条件的，只需预设一个条件$\emptyset$，对应全体图片，并在训练时按一定概率分别训练。
![Alt text](Paper\diffusion\image\image-9.png)

# 10. CLIP: Learning Transferable Visual Models From Natural Language Supervision(2021.2)
CLIP(Contrastive Language-Image Pre-training)，之前只知道个大概一直没仔细读，是openai提出的一个预训练模型/方法。
![Alt text](Paper\diffusion\image\image-3.png)
- 动机：过去的视觉预训练模型大多是在imagenet上，因此在应用到下游任务时必须微调，而NLP领域的GPT3等实现了zero-shot迁移，作者认为，应该1、使用internet级别的数据量预训练；2、以文本为监督信号来扩充数据。因此，作者构建了一个包含 4 亿对(image,text)的新数据集，这些数据集是从互联网上的各种公开来源收集的。
- 方法：比较简单，一个image encoder，一个text encoder(GPT)，分别编码特征，在一个batch内计算对比损失（对称），如图(1)所示。
- zero-shot 分类：如图(2),(3)，先构建prompt，"a photo of [class]"，与图片特征相似度最高的prompt即为图片的种类
- zero-shot 检测：google之后的一篇工作，计算Roi特征与prompt的相似度

# 11. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models(2021 openai)

GLIDE(Guided Language to Image Diffusion for Generation and Editing)主要是力大砖飞，在ADM的基础上，将类guidance扩展到文本条件，分别探索了classifier-free和CLIP-guidance两种方法。
- 在训练时采取与ADM类似的架构，但是额外训练了一个transformer，提取text token，将end token作为ADM的类token，将token sequence作为注意力层的text输入，训练得到条件生成模型CADM
- 对于classifier-free，在得到CADM后，再微调以得到无条件生成模型，与之前类似
![Alt text](Paper\diffusion\image\image-11.png)
- 对于CLIP-guidance，训练仍使用了CADM（可以看到下面公式的$\mu_{\theta}(x_t|c)$）。采样方法其实就是之前提过的，将分类器梯度推广到相似度梯度，而本文使用的是CLIP来度量相似度，并且此处的CLIP是重新训练的noise-aware的模型。
![Alt text](Paper\diffusion\image\image-10.png)
- CLIP-guidance中仍使用了条件扩散模型训练，与最初那篇不同，但文中说，最初那篇说明了samples from **class-
conditional diffusion models** can often be improved
with classifier guidance，嗯……按他的说法，没区别，本来就是条件扩散模型的话相当于用CLIP梯度修正？不过文中本来就没有理论推导，也许就是实验发现，在采样时加入guidance能进一步提高质量。
- 不足：模型对于复杂的文本提示效果也不好，解决方法是支持图像再编辑以不断修正

# 12. DALLE-2:Hierarchical Text-Conditional Image Generation with CLIP Latents(2022 openai)
Openai的文生图模型，主要包括两部分，先利用CLIP根据文本y生成对应的图片潜在空间嵌入，再以该嵌入为条件生成最终图片。
![Alt text](Paper\diffusion\image\image-12.png)

- prior:将y编码为token；用CLIP提取文本嵌入；以文本嵌入为条件利用扩散模型，逐步去噪生成去噪后图片潜在空间表示。训练时，用CLIP提取GT表示后加入噪声得到$z_i^{(t)}$，直接预测GT潜在表示（等价于$x_0$）。在采样时，会生成两个$z_i$，选和$z_t$相似度更高的
![Alt text](Paper\diffusion\image\image-13.png)
- decoder：在GLIDE的基础上，将CLIP图片嵌入投影并加到现有的时间步嵌入中，以及将 CLIP 嵌入投影到四个额外的上下文标记中，并将这些标记连接到 GLIDE 文本编码器的输出序列。主要就是，将先验也作为条件之一，还发现此时文本对效果的提升不明显。并且，文中发现加入Classifier-free guidance进一步提高了效果。
- 潜在空间表示的优势：可以进行图片的插值/变换/多样性。

# 13. Imagen:Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding(2022 nips)
本文中google认为，基于纯text数据（数据规模大得多）训练的LM(BERT,T5)对于文本到图像的生成来说是非常有效的文本编码器，比过去基于text-image训练的文本编码器(CLIP)明显更有效，并且scale冻结文本编码器的大小比scale图像扩散模型的大小更能显着提高样本质量。
- 本文发现classifier-free guidance很有效，增加classifier-free guidance权重可以改善图像文本对齐，但会损害图像保真度，产生高度饱和和不自然的图像。本文认为这是由于，训练时数据均为[-1,1]，但当采样时该权重过大，会导致预测的x超出这一范围，导致训练与测试的不匹配，甚至发散。
- 因此提出静态阈值和动态阈值法，主要是将采样得到的x限制到[-1,1]的方法
- Robust cascaded diffusion models: 为了得到高分辨率的图片，级联了扩散模型以上采样，并且加入了noise conditioning augmentation，使用类似于前向过程的方法，加入噪声，进一步提高了质量
<p align = "center">  
<img src="Paper\diffusion\image\image-14.png"  width="300" />
</p>

# 14. LDM: High-Resolution Image Synthesis with Latent Diffusion Models
本文认为所有似然模型的学习都可以分为两个阶段，首先是perception compression，学会忽略高频细节而感知数据的关键特征；其次是semantic compression，学习数据的语义和概念组成。图片的大部分像素对应于难以察觉的细节，但扩散模型的梯度（训练期间）和神经网络主干（训练和推理）仍然需要在所有像素上进行评估，从而导致多余的计算和不必要的计算。因此，本文提出Latent Diffusion Models(LDM)，通过引入压缩学习阶段和生成学习阶段的明确分离来规避像素级计算,在降维后的潜在空间进行扩散，从而大大提高了效率，且可以适应更高分辨率的图片合成。

- Perceptual Image Compression:训练了一个自编码器，尝试了两种正则化方式，KL-reg为使潜在表示逼近标准正态分布，VQ-reg为采用类似于VQGAN的向量量化。并且，这种降维保持了空间结构，即仍为二维数据，并进一步在扩散模型中使用U-net结构，保留空间结构这种inductive bias，有利于重建（过去的几个工作把图片拉平了用transformer）
- Latent Diffusion Models:经过AE，我们得到一个高效、低维的潜在空间，其中高频、难以察觉的细节被过滤去。训练目标仅将x换为潜在表示z，模型使用卷积构成的time-conditioned U-net，为了充分利用inductive bias
![Alt text](Paper\diffusion\image\image-15.png)
- Conditioning Mechanisms:为了统一图片生成领域的条件模态，本文在UNet结构中加入交叉注意力机制，从而只需设计domain-specifical的prompt编码器，即可实现任意模态的条件生成，具体来说，prompt作为K和V。
![Alt text](Paper\diffusion\image\image-16.png)


# 15. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation(2023.5)
Google的工作，过去的文生图模型缺乏模仿给定subject参考集（比如确定外观的狗）并合成新的在不同背景下subject的能力。Dreambooth是一种文本到图像扩散模型“个性化”的新方法，对预训练的文生图模型进行微调，使其学会将唯一标识符与特定subject绑定，就可以使用唯一标识符来合成不同场景中subject的新颖的真实感图像，并可以保留其关键的识别特征。
- 目标：给定一个subject的几个图像，没有任何文本描述，目标是生成具有高细节保真度和**文本提示引导**的变化的subject新图像，变化包括改变subject位置、改变subject属性（例如颜色或形状）、修改subject的姿势、视点和其他语义修改。
![Alt text](Paper\diffusion\image\image-17.png)
- Designing Prompts for Few-Shot Personalization:为了将subject加入扩散模型的“词典中”，文中为few-shot set设计了以下prompt，a [identifier] [class noun]，class noun是为了利用固有的类先验知识
- Rare-token Identifiers:文中发现现存英文单词作为identifier是次优化的，因为模型需要分离其固有先验再与subject联系起来。而随机特定编码如xx55y效果也不好，因为分词器会把每个letter编码成token也有强先验。文中采用的方法是，先在词汇表中对rare token进行搜索，再其反编码回text（长度小于等于3时效果好），从而得到定义unique identifier的字符序列。
- Class-specific Prior Preservation Loss:在微调预训练扩撒模型时，为了保持先验知识并避免降低多样性，提出了一种loss
![Alt text](Paper\diffusion\image\image-19.png)
  第二项损失，先输入$x_{pr}$=a [class noun]，生成基于模型先验的图片$x_{pr}$，用它自己生成的样本来监督模型，从而在本batch的参数优化中保持该先验；第一项损失即为平常的重建损失，以few-shot set为监督。

<p align = "center">  
<img src="Paper\diffusion\image\image-18.png"  width="300" />
</p>

# 16. DiT:Scalable Diffusion Models with Transformers(2023.5)
过去扩散模型均使用卷积U-net架构，本文提出Diffusion Transformers(DiT)，基于transformer的LDM模型，尽可能忠实于标准Transformer架构以保留其scalable特性。DiT具有很好的scalable能力，当GFLOPS增加（网络宽度/深度/token长度上升），效果总在提高。
- Preliminaries:本文使用IDDPM中的训练方法，用简化训练目标训$\mu$，再用L训方差；用了Classifier-free guidance；文中采用LDM，分别用现成的卷积VAE和基于 Transformer 的 DDPM。
- Patchify，将潜在表示token化，使用ViT一致的位置编码；
- DiT Block：得到token序列后，为了处理扩散模型中的条件输入（时间t，类别c，text等），文中探索了四种加入条件信息的方法，在ViT的基础上做了微小的变化
  - In-context conditioning：将t和c作为额外的input token，类似于原来的cls token，并在最后一步移去条件token
  - Cross-attention block：在ViT自注意力后加入交叉注意力，t和c连接得到长为2的序列
  - Adaptive layer norm (adaLN) block：将标准LN层换为自适应LN层，其中的$\gamma,\beta$参数由t和c回归得到
  - adaLN-Zero block：过去工作发现将残差块初始化为恒等函数有利于优化（即主线0初始化），因此采用了类似设计，除了$\gamma,\beta$，LN还包括一个超参数$\alpha$，进行channel-wise加权，并将预测$\alpha$的MLP初始化为总预测0（残差连接在LN后边）
- Transformer decoder：经过DiT Block后的patch token经过线性解码恢复分辨率，特征维度为2D，即D维噪声和D维方差，计算可得去噪后的潜在表示。
![Alt text](Paper\diffusion\image\image-20.png)



# 17. LoRA:Low-Rank Adaption of large language model(2021.10)
本用来给LLM微调的方法，原理很简单，在扩散模型中也很有效。

<p align = "center">  
<img src="Paper\diffusion\image\image-21.png"  width="300" />
</p>

- 方法：固定大网络参数，只训练某些层参数的增量，且这些参数增量可通过矩阵分解变成更少的可训练参数。具体来说，对于参数$W_O$，在微调时训练$W=W_0+\Delta W=W_0+AB$，其中$W_0$参数冻结，只训练$A\in R^{D\times r}, B\in R^{r\times D}$，由于r可以很小（如2），要训练的参数量大大下降。在初始化时，A初始化为标准高斯分布，B初始化为0.
- 该方法主要用于某些层的线性部分，比如Transformer中的QKV的线性投影，以及FFN的线性部分
- 扩散模型：大多用LoRA微调CLIP以及Unet中交叉注意力层的线性部分。
# 18. Controlnet: Adding Conditional Control to Text-to-Image Diffusion Models(2023.11)
本文提出了一种Paper\diffusion\image\Image-to-image translation的条件控制扩散模型方法，利用附加图片（例如，边缘图、人体姿势骨架、分割图、深度、法线等）条件控制生成的图片。而在特定条件下的训练数据量，明显小于一般文本到图像训练的可用数据，不能直接训练/微调，本文提出Controlnet，增强预训练文生图扩散模型对于spatially localized, task-specifically的图像条件生成。实现了很好的效果，并对数据集大小具有很好的scalabel和鲁棒性。
- ControlNet:对于一个NN块，ControlNet将原有参数固定并复制，复制部分可训练且由零卷积残差连接到原有部分，复制块的输入为x+经过零卷积的条件c，零卷积可以保护复制块参数不被训练之初的噪声干扰
<p align = "center">  
<img src="Paper\diffusion\image\image-22.png"  width="300" />
</p>
- Controlnet for SD:对于SD的编码块和中间块使用Controlnet，并且将结果加到skip connection上。由于SD是LDM，图片条件c被resize到64\*64且经过一个tiny的CNN再输入Controlnet
<p align = "center">  
<img src="Paper\diffusion\image\image-23.png"  width="200" />
</p>

- Training:训练时，随机50%将文本$c_t$替换为空字符串，这种方法增强了 ControlNet 直接识别输入条件图像中的语义（例如边缘、姿势、深度等）的能力，以替代提示。由于零卷积不会给网络增加噪声，因此模型应该始终能够预测高质量的图像，但是对于条件控制的学习，会出现“突然收敛现象”，模型并不是逐渐学习条件控制，而是突然成功地遵循输入条件生成图像（通常优化步骤少于 10K）
![Alt text](Paper\diffusion\image\image-24.png)
- Classifier-free guidance resolution weighting：为了实现CFG，条件图片默认加入到两种噪声的生成中。对于没有文本prompt的情况，全加入会导致CFG失效，而只加入条件噪声会导致guidance过强。为此提出了CFG Resolution Weighting，在向$\epsilon_c$中加入条件图片时，Controlnet结果加入到skip-connection之前成一个权重$w_i=(\frac78)^{12-i}$，$i$为第i个block([0,12])，由浅到深权重逐渐趋向1
![Alt text](Paper\diffusion\image\image-25.png)
- Composing multiple ControlNets：可以使用多张条件图片，直接将对应Controlnet的结果加起来即可