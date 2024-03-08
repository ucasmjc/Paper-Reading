nlp paper reading--��2023�괺

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

[��������](https://arxiv.org/pdf/1706.03762v5.pdf)

Seq2seq�ľ������ģ������Transformer�ṹ����Ӧ�õ���encoder-decoderģ���У���ģ���������ص�
- encoder-decoder�ṹ���ֱ���6�������ҽṹ��ͬ�Ĳ���ɣ�����decoder����mask��֤��auto-regressive��  
  ��������ѵ����Ԥ��ʱ���ǲ��еģ���������ѵ��ʱ���У�Ԥ��ʱ����
- attention���ơ��ڽṹ�ϰ���self-attention���encoder-decoder attention�㣬����ʱ�����˶�ͷע��������ȡ��ͬ�ӿռ��������Scaled Dot-Product Attention�Ա����˵��µķ���
- Layer Normalization:��BN��ȸ��ʺ���������
- λ�ñ��룬���������е�˳����Ϣ
- ����embedding���pre-softmax���Լ��ٲ���������ʹembedding�õ�����ֵ�ѵ����
> ���Ƕ����Xavier��ʼ������λ�ñ�����Ƚ�С������ֱ����ӻ�ʹλ�ñ��������������ʽ�embedding��һ��Ȩ��
- �в����ӣ�ʹ�ݶȸ�ƽ����Ҳ��ǿ���䣬��ֹ�˻������CV���Resnet

��RNN,LSTM��ѭ��ģ����ȣ��������������ŵ�
- ����ɲ�����ǿ������GPU
- �����׻�����о����Զ������֮��������

## 2.GloVe: Global Vectors for Word Representation(2014)
[��������](https://aclanthology.org/D14-1162.pdf)
~~~c
You shall know a word by the company it keeps.
~~~
��Glove����֮ǰ��������ģ����Ҫ�����֣�1����ȫ�ֹ��־������SVD�ֽ⣬ȱ���Ǽ��㿪����2�����ڴ��ں͵�����ѵ������������word2vec(CBOW��skip-gram)��ȱ��Ϊ��������ȫ������

Glove����������ַ������ŵ㣬�����ȱ�ݣ�ѵ���˰���ȫ���������ڲ����������ӽṹ�Ĵ�����ģ�͡�����������ӡ�����ĵ���Ҫ�У�
- ������ָ��ʵı�ֵ�����ǵ�ԭʼֵ��������
- ���ڹ��׸��ʵı�ֵ��һ�����Ƶ�������ʧ����
- ��skip-gramģ�ͽ��м򻯺͵ȼۣ����ս���ͳһ��gloveģ��
- ͨ���Թ��־����ֵ���н�ģ��ͨ����ѧ�Ƶ���ģ�͸��ӶȾ��廯���Ӷ���word2vec���и��ӶȱȽ�

## 3.Enriching Word Vectors with Subword Information(2017)
[��������](https://arxiv.org/pdf/1607.04606v2)

��ȥ�Ĵ�����ģ�ʹ󲿷ֽ�ÿ�������ò�ͬ��������ʾ�����������˵��ʵ���̬ѧ�������Ҷ���ѵ��������δ���ֹ�����ִ������ٵĵ������Ը����Ϻõı�ʾ��

��������skip-gram�Ļ����ϣ�����������n-gram�����������ʾ����׽�˵��ʵĽṹ�������ҿ��ԽϺõظ���ѵ��������δ���ֵĵ��ʡ����ڴˣ�ͬ���Bag of Tricks for Efficient Text Classification�����fasttextģ�͡�

## 4.Neural Machine Translation by Jointly Learning to Align and Translate(2014)
[��������](https://arxiv.org/pdf/1409.0473v7.pdf)


�����������attention���ƣ���target��source sentence�н���һ������룬ʹdecoder��ֱ�ӵػ��source sentence�е���Ϣ������������ӡ�����ĵ���Ҫ�У�

- Ϊʲô���attention���ƣ�
  
  �ڻ�����encoder-decoderģ���У�encoder��source sentence�е���Ϣ����Ϊһ������������c��decoder����c�л�ȡsource sentence�е���Ϣ����c�пɴ������Ϣ���ޣ������ӳ���������encoder���Խ��㹻�����Ϣ���뵽c�У�ģ�͵ı����½���
- ��decoder�У�ͨ������ǰһ��ʱ�䲽����״̬��encoder������״̬��attention score������c_i���Ӷ��������Ļ�ȡsource sentence����Ϣ
- ����˵��attention��additive attention���������Լ��ټ�����

## 5.Improving language understanding by generative pre-training(2018)
[��������](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)

��������˵�һ����������ˮƽ��Ԥѵ��ģ��GPT�����޼ලԤѵ����ͨ��΢�������ھ���������ȡ��������Ч��������������ӡ�����ĵ���Ҫ�У�

- Ԥѵ�������壿NLP�����д����ޱ�ǵ����ϣ��޷����ھ��������ѵ���У�ͨ���޼ල��Ԥѵ����������Щ�����л�ȡ��Ϣ���ھ���������ȡ�ø��õ�Ч��
- GPT������֮����֮ǰ��Ԥѵ��ģ�ʹ��Ϊ����/����/����Ƕ�룬GPTͨ���ı�Ԥѵ�����񣬿��Ի�ø���ˮƽ��������Ϣ
- �����ޱ���ı�Ԥѵ�����������⡣һ����ʲôĿ��������Ԥѵ�����������Ǩ�Ƶ���������  
- ����������������⣬GPT�����˰�ලѧϰ����Ϊ�����׶Ρ�
  
  GPT���޼ලԤѵ���ڼ䣬ʹ��LM��ΪĿ������ģ�ͽṹΪ���Transformer��������������mask  
  Ǩ�Ƶ���������ʱ��Ӧ�üලѧϰ����΢��������������м����˸�������LM������������߼ලģ�͵������������������ڼӿ�����
- ֮ǰ���о�������Ǩ��ʱ����ı�ԭģ�͵Ľṹ������Ӧ��ͬ����ͬ�����������GPT��Բ�ͬ�����񣬽�input�ع�Ϊ���У��м�����������뵽Transformer�㣬����������ӵ����Բ�ȼ򵥽ṹ(softmax,���)���õ���������������
- �Ӵ���һ���¸���Zero-shot���������˽�һ��

## 6.BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)
[��������](https://arxiv.org/pdf/1810.04805)

�����һ���µ�Ԥѵ��ģ��BERT���ڴ˺�����㷺Ӧ����NLP���������

- ��GPT��ͬ��BERTʹ�õ���˫��Transformer�ṹ��Ԥѵ����һ��encoder�����õĲ�׽��������Ϣ
- �޼ලԤѵ��ʱ��MLM��NSP��ΪĿ������MLM�����������Ϣ�Ż�token��ʾ��NSP��׽����ϵ��
- Ԥѵ����ΪFeature-based������Fine-tuning������ǰ�߽�Ԥѵ���õ����������Ƕ�룩��Ϊ�������������֮һ��NSP���������Ԥѵ����Ƕ��ķ�����
- Ϊ��ʹBERT��Ӧ�������������������ɡ�����A+�ָ���+����B����ɣ�ÿ��token����:token embedding+A/B embedding+position embedding
- ��ʼ��ǣ���[CLS]����Ӧ�����һ�������״̬������������
- �������������Ablation study��ͨ�����Ʊ������о�ģ�͵ĸ����ضԽ����Ӱ�죬ֵ��ѧϰ
- auto-encoder:��ĳ�ַ�ʽ�ƻ�input tokens���ٳ����ؽ�����auto-regressive����������Ŀ������


## 7.BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
[��������](https://arxiv.org/pdf/1910.13461)

�����һ���µ�Ԥѵ��ģ��BART�������GPT��BERT�������ڸ���Χ��������΢����
- ����Transformer��encoder-decoderģ�ͣ�˫����������Իع��������
- ��autoencoder����Ԥѵ����������ʵ���˶��ַ������ƻ�ԭʼ�ı�����ѵ�����Ϊ���Ҿ���˳���mask���ⳤ��(����Ϊ0)�ı���Ŀ������Ϊ�ý�������ԭ������ԭʼ�ı���
- ���Խ�BART���ڻ������������У�����target languageԤѵ��������BART��������������ǰ���һ���µ�encoder���ٽ���΢����ѵ��ʱ��һ�׶�freeze�󲿷�BART�Ĳ����������˵����BART��������embedding�㻻��һ���µ�encoder������source sentenceӳ�䵽BART�����������ԡ�

## 8.Language Models are Few-Shot Learners(2020)
[��������](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

֮ǰ��GPT����ʱ���Ӵ���һ���¸���zero-shot����ȥ������ƪ���ģ����������GPT3�����ģ��������chatGPTһ����С�

- ������Ϊ��BERT��ص�Ԥѵ��ģ�ͣ���Ӧ�õ���������ʱ������ҪһЩ�ض������ݼ�����������һ�����ơ�  
  ��Ϊ������ÿ���������к��ʵ����ݼ�����ͼ��һ���ǳ����ģ��΢�����Ƶ�һ�����narrow������Ԥѵ��ģ���л����������/��ٵ���Ϣ/����ԣ���������Բ�ͬ����ʱ������Ҫ�ض������ݼ�΢����
- GPT3ϣ��ͨ��Ԫѧϰ������˵in-context learning��remove�������ƣ��������ָ��������ѵ��ʱģ��ѧ��һϵ�з�����ģʽʶ����������������ʱ��������Ӧ���µľ������񡱣�����ʹԤѵ����ģ�Ͳ����µ����ݼ������ݶȼ���Ӧ�õ��µ�����
- in-context learning��������ģ�͹�ģ�������ߣ�GPT3��ģ�ͽṹ��GPT2����һ��
- �о���"data cpntamination"������Ⱦ���⣬�������˾������ָ�ָ꣬����ѵ�������п��ܰ������Լ������ݡ�
- Few-shotָ��������ʱ����K��Ŀ�������ʾ����prompt(��Ŀ���������Ȼ��������)���ڲ������ݶȸ��º�΢��������µõ������  
  one-shot��K=1ʱ�����Σ������������һЩ�����ϵ�ϰ�߶��������  
  zero-shot��ֻ��prompt���������ѵ�һ�����

  ## 9.Effective Approaches to Attention-based Neural Machine Translation(2015)
    [��������](https://arxiv.org/pdf/1508.04025)

  ����Ϊ��ƪ���µ���ӱ֮�����ޣ��������ߵ�ԭ���ǵ�һ����attention���ƺܺõ�Ӧ�õ�NMT�����С����������һ��Ӧ����attention���Ƶ�NMTģ�ͣ��ǻ���LSTM��encoder-decoder�ṹ

  - ���������cross-attention��  
  
    global attention�Ǿ����ģ�ͣ�decoderÿ��ʱ�䲽������״̬��attend����source������״̬��  

    local attention��Ԥ��source�뵱ǰʱ�䲽�Ķ���p_t�����Դ�Ϊ���ĵ�һ��window�����context vector.��������ּ������p_t�ķ�����local-mֱ����p_t=t��local-p���������Ԥ��p_t����������һ����p_tΪ���ĵĸ�˹�ֲ�Ȩ��
  - Ϊ����decoder��ÿ��ʱ�䲽���֮ǰ�Ķ��������״̬��������Input-feeding��������ǰһ��ʱ�䲽��attentional vector(��context vector������״̬���������󾭹�һ�����Բ�ͷ����Բ�)Ҳ��Ϊ��ǰʱ�䲽�����һ���֡�

## 10.Neural Machine Translation of Rare Words with Subword Units(2016)
[��������](https://arxiv.org/pdf/1508.07909.pdf)

���������һ�ַ���rare��out-of-vocabulary���ʵķ�����Byte Pair Encodind�����㷺Ӧ�á�
- �ڿ��Ŵʱ�ķ��������У���ȥ����ǻ��ڵ���ˮƽ��ģ�ͣ���back-off dictionary����������OOV���⣨��source�е�oov��ͨ���ʵ�map��target���������������⣬һ�ǲ�ͬ���Ե�word��һ����һһ��Ӧ�ģ������޷�����target�ʱ��в����ڵĴ�
- ������Ϊ���Ի��ڷִʣ�subword unit��ˮƽ���룬���rare��oov��Ϊ����/������/ͬԴ��/���ϴʣ�����ͨ���������е�����/��������������룬���ҿ��Խ�ע��������Ӧ�õ��ִ�ˮƽ��
- Ϊ�˴���OOV����Ҫ�ִʣ�����ϣ���ʱ��Ⱥ������ı���token���Ȳ�Ҫ̫������ϣ��ֻ��rare���ʷִʣ�����ͨ��Byte Pair Encodind(BPE)ʵ���ˡ�
- BPE�����ؽ�ѵ��������Ƶ����ߵ�byte�Ժϲ������յõ��ִʱ�

## 11.Neural Machine Translation: A Review of Methods, Resources, and Tools(2020)
[��������](https://www.sciencedirect.com/science/article/pii/S2666651020300024)

�����Ҷ��ĵ�һƪ���������廪��ѧ��2020�귢��Ĺ���NMT��������
  
- �����������Ϊrule-basedģ�ͣ���չΪ����������ͳ�ƻ������루SMT����������ҪΪ�������ѧϰ���񾭻�������
- ��������RNN��NMT��source representationΪ����������������ע�������ƣ�ʵ�ֿɱ䳤�ȣ�������˶��Ľṹ�������ж�RNN,CNN,SAN(self-attention network)�����˶෽��ĶԱȡ�
- ��ȥ������ΪL2R��������R2L�����л������ã�˫�����������ڻ�ø��õĽ����
- auturegressive�������Ǵ��еģ�Ϊ�˼ӿ������ٶȣ������˷��Ա���NMT(NAT)����Ԥ��Ŀ����ӵĳ��ȣ���ͬʱԤ��ÿ����
>�����ᵽ��֪ʶ����(knowledge distillation)��һ��û�����ĸ���
- ����exposure bias�����������ѡ��Ŀ�꺯��/loss
- ʹ�õ�����������߱��֡�Back-translation(BT)���õ������������ɺϳɵ�ƽ�����ϣ�����������෽���������ᵽ������ʦ������ʦ����target�������Ͽ������(J. Zhang, C. Zong, Exploiting source-side monolingual data in neural machine translation)
>�����ᵽ������Ӧ(domain adaptation)��һ�����������˽�ĸ���
- ���»��������޼ලѧϰ�������Ҹо���̫���ף�Υ��������ûϸ��
- ����MT�����е�open vocalbulary���⣬������character-level��subword-levelģ�ͣ��ᵽ��ǰ�������BPE����
- ��������֪ʶ�������������ִ�֪ʶ����encoderѧ�﷨�ṹ����decoderѧ�﷨�ṹ��
>�����ᵽ�˺�������(posterior regularization)���������������������֪ʶ��һ��û�����ĸ���
- �ɽ����ԡ��Ƚ����˿��ӻ��������ݶȱ仯�ķ������ֽ����˽���Transformer��һЩ����
- ³���ԣ�ͨ���Կ������������³���ԣ��ⲿ�ֲ�̫�˽⡣
>�ᵽ��GAN��֮���пտ�һ��