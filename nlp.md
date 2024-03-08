nlp paper reading--于2023年春

- [1.Attention Is All You Need(2017)](#1attention-is-all-you-need2017)
- [2.GloVe: Global Vectors for Word Representation(2014)](#2glove-global-vectors-for-word-representation2014)
- [3.Enriching Word Vectors with Subword Information(2017)](#3enriching-word-vectors-with-subword-information2017)
- [4.Neural Machine Translation by Jointly Learning to Align and Translate(2014)](#4neural-machine-translation-by-jointly-learning-to-align-and-translate2014)
- [5.Improving language understanding by generative pre-training(2018)](#5improving-language-understanding-by-generative-pre-training2018)
- [6.BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](#6bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding2018)
- [7.BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](#7bart-denoising-sequence-to-sequence-pre-training-for-natural-language-generation-translation-and-comprehension)
- [8.Language Models are Few-Shot Learners(2020)](#8language-models-are-few-shot-learners2020)
- [9.Effective Approaches to Attention-based Neural Machine Translation(2015)](#9effective-approaches-to-attention-based-neural-machine-translation2015)
- [10.Neural Machine Translation of Rare Words with Subword Units(2016)](#10neural-machine-translation-of-rare-words-with-subword-units2016)
- [11.Neural Machine Translation: A Review of Methods, Resources, and Tools(2020)](#11neural-machine-translation-a-review-of-methods-resources-and-tools2020)


## 1.Attention Is All You Need(2017)

[论文链接](https://arxiv.org/pdf/1706.03762v5.pdf)

Seq2seq的经典论文，提出了Transformer结构，并应用到了encoder-decoder模型中，该模型有以下特点
- encoder-decoder结构，分别由6个独立且结构相同的层组成，且在decoder中用mask保证了auto-regressive。  
  编码器在训练和预测时都是并行的，解码器在训练时并行，预测时串行
- attention机制。在结构上包括self-attention层和encoder-decoder attention层，计算时运用了多头注意力以提取不同子空间的特征、Scaled Dot-Product Attention以避免点乘导致的方差
- Layer Normalization:与BN相比更适合序列任务
- 位置编码，以利用序列的顺序信息
- 共享embedding层和pre-softmax，以减少参数量，并使embedding得到更充分地训练。
> 因词嵌入以Xavier初始化，与位置编码相比较小，二者直接相加会使位置编码喧宾夺主，故将embedding乘一个权重
- 残差连接，使梯度更平滑，也加强记忆，防止退化，类比CV里的Resnet

与RNN,LSTM等循环模型相比，有两个显著的优点
- 计算可并行性强，利好GPU
- 更容易获得序列距离较远的特征之间的相关性

## 2.GloVe: Global Vectors for Word Representation(2014)
[论文链接](https://aclanthology.org/D14-1162.pdf)
~~~c
You shall know a word by the company it keeps.
~~~
在Glove出现之前，词向量模型主要有两种：1）对全局共现矩阵进行SVD分解，缺点是计算开销大。2）基于窗口和迭代来训练词向量，如word2vec(CBOW和skip-gram)，缺点为难以利用全局特征

Glove结合了这两种方法的优点，规避了缺陷，训练了包含全局特征且内部具有线性子结构的词向量模型。论文中令我印象较深的点主要有：
- 提出共现概率的比值比他们的原始值更有意义
- 基于贡献概率的比值，一步步推导出了损失函数
- 将skip-gram模型进行简化和等价，最终将其统一到glove模型
- 通过对共现矩阵的值进行建模，通过数学推导将模型复杂度具体化，从而与word2vec进行复杂度比较

## 3.Enriching Word Vectors with Subword Information(2017)
[论文链接](https://arxiv.org/pdf/1607.04606v2)

过去的词向量模型大部分将每个单词用不同的向量表示，这样忽略了单词的形态学特征，且对于训练数据中未出现过或出现次数较少的单词难以给出较好的表示。

该论文在skip-gram的基础上，将词向量用n-gram向量求和来表示，捕捉了单词的结构特征，且可以较好地给出训练数据中未出现的单词。基于此，同年的Bag of Tricks for Efficient Text Classification提出了fasttext模型。

## 4.Neural Machine Translation by Jointly Learning to Align and Translate(2014)
[论文链接](https://arxiv.org/pdf/1409.0473v7.pdf)


该论文提出了attention机制，在target和source sentence中建立一种软对齐，使decoder更直接地获得source sentence中的信息。论文中令我印象较深的点主要有：

- 为什么提出attention机制？
  
  在基础的encoder-decoder模型中，encoder将source sentence中的信息编码为一个定长的向量c，decoder仅从c中获取source sentence中的信息。而c中可储存的信息有限，当句子长度增长，encoder难以将足够多的信息编码到c中，模型的表现下降。
- 在decoder中，通过计算前一个时间步隐藏状态和encoder各隐藏状态的attention score来生成c_i，从而更加灵活的获取source sentence的信息
- 提出了点乘attention和additive attention，后者用以减少计算量

## 5.Improving language understanding by generative pre-training(2018)
[论文链接](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)

本文提出了第一个基于语义水平的预训练模型GPT，在无监督预训练后，通过微调可以在具体任务上取得显著的效果。论文中令我印象较深的点主要有：

- 预训练的意义？NLP领域有大量无标记的语料，无法用在具体任务的训练中，通过无监督的预训练可以在这些语料中获取信息，在具体任务中取得更好的效果
- GPT的特殊之处？之前的预训练模型大多为单词/短语/句子嵌入，GPT通过改变预训练任务，可以获得更高水平的语义信息
- 利用无标记文本预训练有两个问题。一是用什么目标任务来预训练？二是如何迁移到具体任务  
- 针对上述的两个问题，GPT采用了半监督学习，分为两个阶段。
  
  GPT在无监督预训练期间，使用LM作为目标任务，模型结构为多层Transformer解码器，单向有mask  
  迁移到具体任务时，应用监督学习进行微调，在这个过程中加入了辅助任务LM，既有利于提高监督模型的生成能力，又有利于加快收敛
- 之前的研究工作在迁移时大多会改变原模型的结构，以适应不同任务不同的输入输出。GPT针对不同的任务，将input重构为序列（有间隔符），输入到Transformer层，将其输出连接到线性层等简单结构(softmax,相加)，得到具体任务的输出。
- 接触到一个新概念Zero-shot，过几天了解一下

## 6.BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)
[论文链接](https://arxiv.org/pdf/1810.04805)

提出了一种新的预训练模型BERT，在此后几年里广泛应用于NLP的许多领域。

- 与GPT不同，BERT使用的是双向Transformer结构，预训练了一个encoder，更好的捕捉上下文信息
- 无监督预训练时用MLM和NSP作为目标任务，MLM结合上下文信息优化token表示，NSP捕捉句间关系。
- 预训练分为Feature-based方法和Fine-tuning方法，前者将预训练好的特征（如词嵌入）作为下游任务的特征之一。NSP任务便借鉴了预训练句嵌入的方法。
- 为了使BERT适应各种下游任务，其输入由“句子A+分隔符+句子B”组成，每个token包括:token embedding+A/B embedding+position embedding
- 开始标记（如[CLS]）对应的最后一层的隐藏状态常被用来分类
- 该论文做了许多Ablation study，通过控制变量来研究模型的各因素对结果的影响，值得学习
- auto-encoder:以某种方式破坏input tokens，再尝试重建，与auto-regressive的区别在于目标任务


## 7.BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
[论文链接](https://arxiv.org/pdf/1910.13461)

提出了一种新的预训练模型BART，结合了GPT和BERT，可以在更大范围的任务上微调。
- 基于Transformer的encoder-decoder模型，双向编码器和自回归解码器。
- 用autoencoder方法预训练，本文中实验了多种方法来破坏原始文本，最佳的两种为打乱句子顺序和mask任意长度(可以为0)文本，目标任务为让解码器还原完整的原始文本。
- 可以将BART用在机器翻译任务中，将用target language预训练的整个BART当作解码器，在前面加一个新的encoder，再进行微调，训练时第一阶段freeze大部分BART的参数。具体地说，将BART编码器的embedding层换成一个新的encoder，来将source sentence映射到BART可以理解的语言。

## 8.Language Models are Few-Shot Learners(2020)
[论文链接](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

之前读GPT论文时，接触到一个新概念zero-shot，便去搜了这篇论文，结果发现是GPT3的论文，和最近的chatGPT一脉相承。

- 本文认为，BERT相关的预训练模型，在应用到具体任务时，仍需要一些特定的数据集，这是他的一个限制。  
  因为：不是每个新任务都有合适的数据集；试图将一个非常大的模型微调限制到一个相对narrow的任务，预训练模型中会有许多无用/虚假的信息/相关性；人类在面对不同任务时，不需要特定的数据集微调。
- GPT3希望通过元学习，或者说in-context learning来remove上述限制，这个概念指——“在训练时模型学到一系列方法和模式识别能力，并在推理时快速自适应到新的具体任务”，进而使预训练的模型不需新的数据集更新梯度即可应用到新的任务
- in-context learning的能力随模型规模增大而提高，GPT3的模型结构与GPT2基本一样
- 研究了"data cpntamination"数据污染问题，并给出了具体衡量指标，指的是训练数据中可能包含测试集的内容。
- Few-shot指，在推理时输入K个目标任务的示例和prompt(对目标任务的自然语言描述)，在不进行梯度更新和微调的情况下得到输出。  
  one-shot即K=1时的情形，因更符合人在一些任务上的习惯而特意提出  
  zero-shot即只给prompt，这是最难的一种情况

  ## 9.Effective Approaches to Attention-based Neural Machine Translation(2015)
    [论文链接](https://arxiv.org/pdf/1508.04025)

  我认为这篇文章的新颖之处有限，引用量高的原因是第一个将attention机制很好的应用到NMT任务中。本文提出了一个应用了attention机制的NMT模型，是基于LSTM的encoder-decoder结构

  - 提出了两种cross-attention。  
  
    global attention是经典的模型，decoder每个时间步的隐藏状态都attend所有source的隐藏状态。  

    local attention先预测source与当前时间步的对齐p_t，在以此为中心的一个window里计算context vector.提出了两种计算对齐p_t的方法，local-m直接令p_t=t，local-p引入参数来预测p_t，并引入了一个以p_t为中心的高斯分布权重
  - 为了让decoder的每个时间步获得之前的对齐和隐藏状态，引入了Input-feeding方法，将前一个时间步的attentional vector(将context vector和隐藏状态连接起来后经过一个线性层和非线性层)也作为当前时间步输入的一部分。

## 10.Neural Machine Translation of Rare Words with Subword Units(2016)
[论文链接](https://arxiv.org/pdf/1508.07909.pdf)

本文提出了一种翻译rare和out-of-vocabulary单词的方法，Byte Pair Encodind，被广泛应用。
- 在开放词表的翻译任务中，过去大多是基于单词水平的模型，用back-off dictionary方法来处理OOV问题（将source中的oov词通过词典map到target）。但有两个问题，一是不同语言的word不一定是一一对应的，二是无法生成target词表中不存在的词
- 本文认为可以基于分词（subword unit）水平翻译，许多rare或oov词为名字/外来词/同源词/复合词，可以通过翻译其中的语素/音素来形译或音译，并且可以将注意力机制应用到分词水平。
- 为了处理OOV，需要分词，但又希望词表长度和输入文本的token长度不要太大，所以希望只对rare单词分词，本文通过Byte Pair Encodind(BPE)实现了。
- BPE迭代地将训练数据中频率最高的byte对合并，最终得到分词表

## 11.Neural Machine Translation: A Review of Methods, Resources, and Tools(2020)
[论文链接](https://www.sciencedirect.com/science/article/pii/S2666651020300024)

这是我读的第一篇综述，是清华大学在2020年发表的关于NMT的综述。
  
- 机器翻译最初为rule-based模型，后发展为数据驱动的统计机器翻译（SMT），现在主要为基于深度学习的神经机器翻译
- 早期利用RNN的NMT的source representation为定长向量，后引入注意力机制，实现可变长度，后出现了多层的结构。文章中对RNN,CNN,SAN(self-attention network)进行了多方面的对比。
- 过去解码器为L2R推理，后发现R2L推理有互补作用，双向推理有利于获得更好的结果。
- auturegressive的推理是串行的，为了加快推理速度，出现了非自编码NMT(NAT)，先预测目标句子的长度，再同时预测每个词
>这里提到了知识蒸馏(knowledge distillation)，一个没听过的概念
- 由于exposure bias，提出了许多可选的目标函数/loss
- 使用单语言语料提高表现。Back-translation(BT)利用单语言语料生成合成的平行语料，还讲到了许多方法，其中提到了宗老师和张老师利用target语言语料库的论文(J. Zhang, C. Zong, Exploiting source-side monolingual data in neural machine translation)
>这里提到了域适应(domain adaptation)，一个听过但不了解的概念
- 文章还介绍了无监督学习，不过我感觉不太靠谱，违反常理，就没细看
- 对于MT任务中的open vocalbulary问题，介绍了character-level和subword-level模型，提到了前两天读的BPE方法
- 整合先验知识，介绍了引入字词知识、让encoder学语法结构、给decoder学语法结构。
>这里提到了后验正则化(posterior regularization)，用这个方法来整合先验知识，一个没听过的概念
- 可解释性。先介绍了可视化参数及梯度变化的方法，又介绍了解释Transformer的一些方法
- 鲁棒性，通过对抗攻击样本提高鲁棒性，这部分不太了解。
>提到了GAN，之后有空看一下