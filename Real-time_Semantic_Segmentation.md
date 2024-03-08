- [Real-Time Semantic Segmentation](#real-time-semantic-segmentation)
  - [1.Enet: A deep neural network architecture for real-time semantic segmentation(2016)](#1enet-a-deep-neural-network-architecture-for-real-time-semantic-segmentation2016)
  - [2.Erfnet: Efficient residual factorized convnet for real-time semantic segmentation(2017)](#2erfnet-efficient-residual-factorized-convnet-for-real-time-semantic-segmentation2017)
  - [3.ShelfNet for Fast Semantic Segmentation(2018)](#3shelfnet-for-fast-semantic-segmentation2018)
  - [4.Contextnet: Exploring context and detail for semantic segmentation in real-time(2018)](#4contextnet-exploring-context-and-detail-for-semantic-segmentation-in-real-time2018)
  - [5.Espnet: Efficient spatial pyramid of dilated convolutions for semantic segmentation(2018)](#5espnet-efficient-spatial-pyramid-of-dilated-convolutions-for-semantic-segmentation2018)
  - [6.Espnetv2: A light-weight, power efficient, and general purpose convolutional neural network(2019)](#6espnetv2-a-light-weight-power-efficient-and-general-purpose-convolutional-neural-network2019)
  - [7.Fast-scnn: Fast semantic segmentation network(2019)](#7fast-scnn-fast-semantic-segmentation-network2019)
  - [8.swiftnetRN:In defense of pre-trained imagenet architectures for real-time semantic segmentation of road-driving images(2019)](#8swiftnetrnin-defense-of-pre-trained-imagenet-architectures-for-real-time-semantic-segmentation-of-road-driving-images2019)
  - [9.Dfanet: Deep feature aggregation for real-time semantic segmentation(2019)](#9dfanet-deep-feature-aggregation-for-real-time-semantic-segmentation2019)
  - [10.MSFNet:Real-time semantic segmentation via multiply spatial fusion network(2019)](#10msfnetreal-time-semantic-segmentation-via-multiply-spatial-fusion-network2019)
  - [11.Lednet: A lightweight encoder-decoder network for real-time semantic segmentation(2019)](#11lednet-a-lightweight-encoder-decoder-network-for-real-time-semantic-segmentation2019)
  - [12.CGNet: A Light-weight Context Guided Network for Semantic Segmentation(2020)](#12cgnet-a-light-weight-context-guided-network-for-semantic-segmentation2020)
  - [13.SFNet:Semantic flow for fast and accurate scene parsing(2020)](#13sfnetsemantic-flow-for-fast-and-accurate-scene-parsing2020)
  - [14.FANet:Real-time semantic segmentation with fast attention(2020)](#14fanetreal-time-semantic-segmentation-with-fast-attention2020)
  - [15.RegSeg:Rethink dilated convolution for real-time semantic segmentation(2021)](#15regsegrethink-dilated-convolution-for-real-time-semantic-segmentation2021)
  - [16.STDC:Rethinking BiSeNet For Real-time Semantic Segmentation(2021)](#16stdcrethinking-bisenet-for-real-time-semantic-segmentation2021)
  - [17.DDRNet:Deep dual-resolution networks for real-time and accurate semantic segmentation of road scenes(2021)](#17ddrnetdeep-dual-resolution-networks-for-real-time-and-accurate-semantic-segmentation-of-road-scenes2021)
  - [18.Pp-liteseg: A superior real-time semantic segmentation model(2022)](#18pp-liteseg-a-superior-real-time-semantic-segmentation-model2022)
  - [19.PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller(2022)](#19pidnet-a-real-time-semantic-segmentation-network-inspired-from-pid-controller2022)
  - [20.SFNet-Lite: Faster, Accurate, and Domain Agnostic Semantic Segmentation via Semantic Flow(2022)](#20sfnet-lite-faster-accurate-and-domain-agnostic-semantic-segmentation-via-semantic-flow2022)
  - [21.TopFormer: Token pyramid transformer for mobile semantic segmentation(2022)](#21topformer-token-pyramid-transformer-for-mobile-semantic-segmentation2022)
  - [22.RTFormer: Efficient Design for Real-Time Semantic Segmentation with Transformer(2022)](#22rtformer-efficient-design-for-real-time-semantic-segmentation-with-transformer2022)
  - [23.FFNet:Simple and Efficient Architectures for Semantic Segmentation(2022)](#23ffnetsimple-and-efficient-architectures-for-semantic-segmentation2022)
  - [24.DWRSeg: Dilation-wise Residual Network for Real-time Semantic Segmentation(2022)](#24dwrseg-dilation-wise-residual-network-for-real-time-semantic-segmentation2022)
  - [25.SeaFormer: Squeeze-enhanced Axial Transformer for Mobile Semantic Segmentation(2023)](#25seaformer-squeeze-enhanced-axial-transformer-for-mobile-semantic-segmentation2023)
  - [26.AFFormer:Head-Free Lightweight Semantic Segmentation with Linear Transformer(2023)](#26afformerhead-free-lightweight-semantic-segmentation-with-linear-transformer2023)
  - [27.LETNet:Lightweight Real-time Semantic Segmentation Network with Efficient Transformer and CNN(2023)](#27letnetlightweight-real-time-semantic-segmentation-network-with-efficient-transformer-and-cnn2023)
  - [28.ISANet:Interlaced Sparse Self-Attention for Semantic Segmentation(2019)](#28isanetinterlaced-sparse-self-attention-for-semantic-segmentation2019)
  - [29.Denseaspp for semantic segmentation in street scenes(2018)](#29denseaspp-for-semantic-segmentation-in-street-scenes2018)

# Real-Time Semantic Segmentation
## 1.Enet: A deep neural network architecture for real-time semantic segmentation(2016)
[��������](https://arxiv.org/pdf/1606.02147.pdf)
�����һ��ʵʱ�ָ����磬ʹ���˺ܶ�trick
![Alt text](Paper/Real-time_Semantic_Segmentation/image/1.png)
- �����²������ϲ������̵ķָ�ȶ�ʧ����segnet�ķ����ϲ���
- �ǶԳƵ�Encoder-Decoder�ṹ�����Ͳ�����
- �����ʹ��PReLU������ReLU
- ����׼����ֽ��������״������в����ӵ���һ�߲��Ǻ�����Ӷ���max pool
- ʹ�ÿն����������
- �Ľ���bottleneck��1\*1����Ϊ2�ľ������Ϊ��2\*2�ģ����ڳػ����ͳߴ��ʹ�þ�����ά��
## 2.Erfnet: Efficient residual factorized convnet for real-time semantic segmentation(2017)
[��������](http://www.robesafe.com/personal/roberto.arroyo/docs/Romera17tits.pdf)

Enet��Ȼ����С��������̫����ĸĽ��в�죬���erfnet

![Alt text](Paper/Real-time_Semantic_Segmentation/image/2.png)
- ԭresnet������ֲв�飬��������;������ơ�bottleneck����(b)����������ӣ�����ɱ���С���㱻ʹ�ã���һЩ�������ᵽ����������ӣ�(a)��׼ȷ�Ը��ߡ�
- Enet�Ľ���(b)�࣬���ĸĽ�(a)������߾��ȡ��������factorized residual layers�ֽ������Ǳ�׼����ġ����Ƚ��ơ�


## 3.ShelfNet for Fast Semantic Segmentation(2018)
[��������](https://arxiv.org/pdf/1811.11254v6.pdf)
shelfnet���ж��encoder-decoder�ԣ�����˾��ȣ���ͨ������ͨ��������С���㸺��

![Alt text](Paper/Real-time_Semantic_Segmentation/image/3.png)

- 2��4�൱��decoder��0��3�൱��encoder��1��1\*1�����CNN��ȡ������ͼ��ά4��
- ΪʲôЧ���ã�������Ϊ��shelfnet�൱�ڶ��FCN�ļ��ɣ�����ͼ�ߴ���4�ֵ�segnet�൱��4��FCN����Shelfnet�൱��29����Ӧ���������ѧ��֪ʶ�������ң�shelfnet�൱��deep��shallow·���ļ���
- S-Block�е�����3\*3�������Ȩ�أ�����˵�����RNN������BN���ǲ�ͬ�ģ��ڲ����;��ȵ�ͬʱ�����ٲ���
## 4.Contextnet: Exploring context and detail for semantic segmentation in real-time(2018)
[��������](https://arxiv.org/pdf/1805.04554)

˫��֧·�����ֱ�����ֱ��ʲ�ͬ��ͼƬ��Ӧ������ȿɷ�������
![Alt text](Paper/Real-time_Semantic_Segmentation/image/16.png)

## 5.Espnet: Efficient spatial pyramid of dilated convolutions for semantic segmentation(2018)
[��������](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sachin_Mehta_ESPNet_Efficient_Spatial_ECCV_2018_paper.pdf)

����׼����ֽ�Ϊ1\*1����ά���Ϳռ�ն��������������׽�����Ұ����������Ϊ��ά���Լ�ʹ����������Ҳ���ࡣ

![Alt text](Paper/Real-time_Semantic_Segmentation/image/17.png)

Espģ���Ƚ�ά��N/K���ٲ���ʹ��K����ͬ�����ʵĿն�������õ�K��N/Kά����������ֱ��������������αӰ�������о���grid problem��������ʹ��HFF(Hierarchical feature fusion)�������ںϡ�

## 6.Espnetv2: A light-weight, power efficient, and general purpose convolutional neural network(2019)
[��������](http://openaccess.thecvf.com/content_CVPR_2019/papers/Mehta_ESPNetv2_A_Light-Weight_Power_Efficient_and_General_Purpose_Convolutional_Neural_CVPR_2019_paper.pdf)

��Esp��Ļ����ϣ����ն������Ϊ��ȿɷ���ģ�����������1\*1���Ӷ����EESP�飬���Դ�Ϊ��Ҫ��ɲ������һ��ͨ����������Espnet v2

![Alt text](Paper/Real-time_Semantic_Segmentation/image/18.png)

## 7.Fast-scnn: Fast semantic segmentation network(2019)
[��������](https://arxiv.org/pdf/1902.04502)

���������Fast-SCNN�����encoder-decoder��˫��֧�ṹ
![Alt text](Paper/Real-time_Semantic_Segmentation/image/4.png)
- ���ü��������²���(learning to downsample)���Ƚ�ǳ���൱��˫��֧�ṹ��Ŀռ��֧
- globla feature extractor���ü�����inverted bottleneck��PPM����ȡȫ���������൱�������ķ�֧
- FFM��������֧�������ں����������������൱��������֧����ǰ����ļ���.FFM�൱��һ��skip���ӣ�encoder-decoder�ṹ��

## 8.swiftnetRN:In defense of pre-trained imagenet architectures for real-time semantic segmentation of road-driving images(2019)
[��������](http://openaccess.thecvf.com/content_CVPR_2019/papers/Orsic_In_Defense_of_Pre-Trained_ImageNet_Architectures_for_Real-Time_Semantic_Segmentation_CVPR_2019_paper.pdf)

������Ϊ��������������ͨ������ָ�ģ��+��������+�򵥵Ľ���������ʵ��Ч����Ч�ʵľ��⡣֮ǰ�����ʵʱ�ָ�ģ����ΪԤѵ��û�ã�����֤����������
![Alt text](Paper/Real-time_Semantic_Segmentation/image/5.png)

ѵ��ʱʹ��image��������encoder����Ȩ�أ�ͨ���������ںϺ�SPPʵ���������Ұ
## 9.Dfanet: Deep feature aggregation for real-time semantic segmentation(2019)
[��������](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_DFANet_Deep_Feature_Aggregation_for_Real-Time_Semantic_Segmentation_CVPR_2019_paper.pdf)

�ڴ����ٺͼ��ٲ�������ͬʱ�����־��Ȼ������䡣

![Alt text](Paper/Real-time_Semantic_Segmentation/image/19.png)
��������Xception��������Բ�ע������SE�������backbone�����룬Ϊ���ø����������ڶ��backbone��ϸ�������ռ��ḻ�Ķ����Ұ��Ϣ

## 10.MSFNet:Real-time semantic segmentation via multiply spatial fusion network(2019)
[��������](https://arxiv.org/pdf/1911.07217)
����������Ȼή�͸���Ұ������ͨ����������������ʾ�ռ䣬���ϣ�������������Ұ�����ҿ��Իָ��ռ���ʧ

![Alt text](Paper/Real-time_Semantic_Segmentation/image/6.png)

- Multi-features Fusion Module��һ��������Ч�����磬ͨ��SAP(Spatial Aware Pooling),��ÿ������Ұ�����кܺõĿռ���Ϣ�ָ������ҽ���ͬ����Ұ��Σ���ͬ�в�飩��ͬ�ֱ��ʵ������ں������������ڲ����Ӽ���ɱ�������´�����������
- ������µ�Class Boundary Supervision���񣬶�MFM�ռ��������������������ж����Ŀ����ϲ���������һ�������мල�����߽�Ԥ�⣬����loss�Ǽ�Ȩ��


## 11.Lednet: A lightweight encoder-decoder network for real-time semantic segmentation(2019)
[��������](https://arxiv.org/pdf/1905.02423)

ʹ���˷ǶԳƵ�encoder-decoder�������˲�������� split-shuffle-non-bottleneck(SS-nbt)
![Alt text](Paper/Real-time_Semantic_Segmentation/image/20.png)
- SS-nbt:�Ľ��˲в�飨��1\*1���Ǹ������о��ں���crfnet,shufflenet�����Ҷ�ͨ�����л����Լ�С��������֮���RegSegӦ�ý���������������ֽ⣬���ž��������������ͨ�����
  ![Alt text](Paper/Real-time_Semantic_Segmentation/image/21.png)
- ������ʹ����attention pyramid network (APN)������ʵ���Ǳ�׼��ע������decoder������һ������������������ϸ��encoder��������������������յõ�һ��ע����������˵Ȩ�ذɣ�����Ȩ��encoder��ԭʼ����������˸�ȫ�ֳػ����ӵ�APN�������
## 12.CGNet: A Light-weight Context Guided Network for Semantic Segmentation(2020)
[��������](https://arxiv.org/pdf/1811.08201)

����ּ�ڲ������н׶ε�������������רΪ�ָ���Ƶ����磬���GC block����������Ϊ�������GCNet

![Alt text](Paper/Real-time_Semantic_Segmentation/image/7.png)
- CG block�У�loc��ȡ�ֲ�������sur��ȡ������������joi������������ƴ������glo���se�ں�ȫ���������Ӷ�ʵ����ÿ���׶ζ���������������,��չ��non-local�ĸ��������ģ�������ڱ���׶ι������������ģ�飬��ASPP,SPP��
- CG block��ֻ������ͨ�����(depth-wise)��ʵ���������Ӹ�1\*1��Ч�����ͺܶ࣬�����Ŀ��ܵĽ���Ϊ"the local
feature and the surrounding context feature need to maintain channel independence"

## 13.SFNet:Semantic flow for fast and accurate scene parsing(2020)
[��������](https://arxiv.org/pdf/2002.10120)
������Ϊ��ͬ�������֮�����gap�������������֮֡��Ķ������죿������Ϊ��ͬ�ֱ��ʵ�����ͼ�ɿ���������������ͨ��ѧϰ����������ͳһ������Ϣ��level����С���ںϲ�ͬ�ֱ��ʵ�����ͼʱ��С��Ϣ��ʧ������� Flow Alignment Module��������Ӧ�õ�FPN�ṹ�У��õ�SFNet
![Alt text](Paper/Real-time_Semantic_Segmentation/image/8.png)
- �ں������ֱ��ʵ�����ͼh(�߷ֱ���),lʱ���Ƚ�l˫���Բ���������h�ֱ��ʵı�׼������������߷ֱ��ʣ���conca��Ϊ���룬�õ�������Ԥ�����ÿ�����ص��ƫ�ƣ���ʹ����������h�ֱ��ʵı�׼���񣨲������λ�ñ��ˣ����ٴβ���l�õ�h�ֱ��ʵ�����ͼ����ԭh��ӵõ��������������һ���в
- FAMģ����Թ㷺Ӧ�õ��ںϲ�ͬ�ֱ���������ģ����
- �����ɵ�FPN�����У�������top-down·���еĸ�����ͼͳһ�����յ�����ͼ�У�һ�����Ԥ��
- ����FAMֻ�õ�����1\*1��3\*3��������������ѧϰ�ı任����������С

## 14.FANet:Real-time semantic segmentation with fast attention(2020)
[��������](https://ieeexplore.ieee.org/ielaam/7083369/9223766/9265219-aam.pdf)

��ע�����㷺Ӧ�ã���ʱ������̫�󣬱��������fast attention����������Ǿֲ�����������Ϣ��Ϊ����Ӧ�߷ֱ������룬���������м����˶�����²�������fast attention��ȫ����Ϣ�������½����١�
![Alt text](Paper/Real-time_Semantic_Segmentation/image/22.png)
��ע���������е�softmax��ΪL2���򻯣��ڼ��ټ�������ͬʱҲ�����Ч�����Ӷ�ֻ��Ҫ����˷����ɼ���ע������������������˳������K��V�������Ӷ���n\*n\*c���͵�n\*c\*c��

## 15.RegSeg:Rethink dilated convolution for real-time semantic segmentation(2021)
[��������](https://arxiv.org/pdf/2111.09957)

��������û����Ұ����ȡ�������ã��������һ��dilate block(D block)����ͨ������D block�õ�RegSeg.
![Alt text](Paper/Real-time_Semantic_Segmentation/image/9.png)
 - D block�����˷����������٣������SEģ�飬ʹ���˲�ͬ�����ʵĿն����������������һ��group����������Ϊ1���Ӷ����������Ұ��ͬʱ�����ֲ���Ϣ
 - ���ɱ���������D block�Ķѵ�������Ϊ2��D block��һ����
## 16.STDC:Rethinking BiSeNet For Real-time Semantic Segmentation(2021)
[��������](http://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Rethinking_BiSeNet_for_Real-Time_Semantic_Segmentation_CVPR_2021_paper.pdf)

�Ż�Bisenet��˫·��ע���������࣬�������ϸ�ڷ�֧������Ӧ�õ����п���У�ֻ��ѵ��ʹ�ã�����������񣩣�STDC�Ǻ��������õ�����������
- ������STDC(Short-Term Dense Concatenate Module)��
![Alt text](Paper/Real-time_Semantic_Segmentation/image/10.png)  
��ͼΪ����Ϊ1��STDC�飬���������뼶����block������������л�ò�ͬ�߶ȵĸ���Ұ������ά�Ȳ��Ͻ��ͣ���Ϊ������Ϣ�����У������ҿ�֤���������У�block�������Բ�����Ӱ���С�����ʴ������Ϊͨ������ָ�����ݼ��ģ�block����֮��Ĳ������٣������ս���ͬblock������ͼ����������ͨ��STDC�飬���ǵõ���߶ȵ�������������ͨ���ı�block������ÿ���չ�ĸ���Ұ
- ͨ������STDC����Ϊ�������ɣ����STDC����
	![Alt text](Paper/Real-time_Semantic_Segmentation/image/11.png)  
  ÿ��stage�²���������stage 1&2��һ���������ɣ����涼��1������Ϊ2��STDC���N������Ϊ1����ɡ���stage3�������Ϊ�ռ��֧����stage4&5�����������ȫ�ֳػ������Ϊ�����ķ�֧������ARM(��SE����)�����ս�����·������������FFM(Ҳ��SE����)
- ��ѵ��ʱ���븨�������ڲ���ʱ���ã������ʵʱ�ָ��пɷ��������ռ��֧�Ľ������detail head���������ϸ��GT����Ԫ�����أ�ϸ��GT�ɲ�����ͬ��������˹�����+��ѧϰ��1\*1�������ȡ��ֵ�õ�GTϸ��ͼ��GTϸ��ͼ��Ԥ��ʹ���˶�Ԫ�����غ�Dice��������ʧ��ϸ����ռ�ı�����ϸС����ԭ������Ϊ�ǣ�������˹�������һ�ֿ�����ȡ��Ե��Ϣ�ľ���ˣ������ҵ�ͻ�䣬��ͼ�λ��Ƕȡ�
## 17.DDRNet:Deep dual-resolution networks for real-time and accurate semantic segmentation of road scenes(2021)
[��������](https://arxiv.org/pdf/2101.06085)

�����Ż���˫�߽ṹ�����һ���µ���������DDRNet�����˹���ǰ�����²������������˶��˫����Ϣ�ںϣ����������һ���µ�������ģ��DASPP����׽��߶���������Ч����Ұ��
- DASPP:ʹ�ô��ں˺ʹ󲽳��ĳػ����ڵͷֱ��ʵ�����ͼ�ϼ����ж�߶ȵĳػ������ҽ���ͬ�߶ȵ������ںϣ���ñȹ�ȥSPPģ���ǿ��������������ΪDASPP��������ͷֱ��ʵ�����ͼ���������ӵľ���㲻̫Ӱ�������ٶȡ�
![Alt text](Paper/Real-time_Semantic_Segmentation/image/12.png)
- DDRNet:���������˫����Ϣ�ں�ģ�飬�������ķ�֧���ʹ����DASPP�����ϲ����ںϽ���Ԥ�⡣ѵ��ʱ���븨��loss������Ч���ܺá�
![Alt text](Paper/Real-time_Semantic_Segmentation/image/13.png)


## 18.Pp-liteseg: A superior real-time semantic segmentation model(2022)
[��������](https://arxiv.org/pdf/2204.02681)
����˼����µ�ģ�飬һ�������ں�ģ��Unified Attention Fusion Module��һ���ۺ�ȫ����������Ϣ��Simple Pyramid Pooling Module������SPP�����õ�һ��������Pp-liteseg
![Alt text](Paper/Real-time_Semantic_Segmentation/image/14.png)
- UAFM�����ÿռ�ע��������ͨ������ֵ��max����4\*H\*W���任�õ�H\*W�ķ���$\alpha$����ͨ��ע������������ͼ�����ػ���ƽ���ػ������任�õ�Cά����$\alpha$����ǿ������ʾ����$\alpha$��Ȩ�õ�$out=\alpha up+(1-\alpha)high$����һ�������ע������һ��
![Alt text](Paper/Real-time_Semantic_Segmentation/image/15.png)
- SPPM:����SPP����С�����к������ͨ���������ٷ�֧����ɾȥshortcut(�в�����)�������ӻ�Ϊ���
- Flexible and Lightweight Decoder,��ʵ�������Ž��뽵��ά��
## 19.PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller(2022)
[��������](https://arxiv.org/pdf/2206.02066.pdf?trk=public_post_comment-text)

˫��֧�ṹ��ʵʱ����ָ�����Ӧ�ù㷺����ϸ�ڷ�֧�Ϳռ��֧�����ں�ʱ������level��ͬ������ʧ�������ô�ͳ���������PID�����ͣ��ռ��֧�൱��P��ϸ�ڷ�֧���Ͼۺ������൱��I��ֻ��PI���ܲ�����������˼���D��֧��P��΢�֣����߽��֧�������Ƴ���������˵���ñ߽��ֱ֧���ռ�������ķ�֧�������ںϡ��ɴˣ������һ������֧����PIDNet

![Alt text](Paper/Real-time_Semantic_Segmentation/image/23.png)

- ���븨�������ڿռ��֧����ָ�loss����D��֧������˵Auxiliary Derivative Branch (ADB) ������߽��Ԫ��ʧ(Dice�Լ������ƽ��)��������boundary-aware�ķָ���ʧ��ֻͳ��ADB���루���ʣ�����ĳ����ֵ�����ص�ķָ���ʧ����
- �Ľ���DDRNet��DAPPMģ�飬��Ϊ��̫���Ӷ����ܺܺõĲ��У����ҳ���������ģ�͵ı���������
![Alt text](Paper/Real-time_Semantic_Segmentation/image/24.png)

- Pag(Pixel-attention-guided fusion module):����P��֧����I��֧����ʱ������ע����������ģ���ָ���ں�
![Alt text](Paper/Real-time_Semantic_Segmentation/image/25.png)
- Bag(Boundary-attention-guided fusion module):�ñ߽�������ָ��ϸ���������������������ں�
![Alt text](Paper/Real-time_Semantic_Segmentation/image/26.png)
> ��һЩϸ����Ҫ�����루����б�Ҫ��������$\sigma$ֻ��һ�����������������صģ�2ά����Ҳ������Ϊ3ά�ġ�Pag������������ô���˷��ģ��ȶ�ӳ�䵽�趨�õ�embedά�ȣ������
## 20.SFNet-Lite: Faster, Accurate, and Domain Agnostic Semantic Segmentation via Semantic Flow(2022)
[��������](https://arxiv.org/pdf/2207.04415)

��ԭSFNet�Ļ����Ͻ�һ���Ż�������˸��õ��ںϲ�ͬ�߶�������ģ��GD-FAM
- ��������ʹ�������µ�STDC
- ��FAM�Ż�ΪGD-FAM�������ſء����������(H\*W\*4,ά�ȷ���)�󣬷ֱ�warp���ߵ�����ͼ������1\*1������ԭ����ͼȡ���ػ���ƽ���ػ�Ϊ����������ſ�ͼ��ÿ�����ص�һ��[0,1]��ֵ��������Ȩ�����ߴ�����ͼ���ںϡ�
![Alt text](Paper/Real-time_Semantic_Segmentation/image/27.png)
- ������ֻ���������ߴ磬��С�˿������������վ��Ⱥ��ٶȶ��ܴ��Ż�
![Alt text](Paper/Real-time_Semantic_Segmentation/image/28.png)
- ѵ��ʱ����OHEM����ȼල������loss����trick


## 21.TopFormer: Token pyramid transformer for mobile semantic segmentation(2022)
[��������](http://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_TopFormer_Token_Pyramid_Transformer_for_Mobile_Semantic_Segmentation_CVPR_2022_paper.pdf)

���CNN��transformer������������ָ�ܹ�TopFormer��

![Alt text](Paper/Real-time_Semantic_Segmentation/image/29.png)

- ��ʹ�ü�����mobilenet�����ɲ������ͼ�������²�����token���������²�����ͳһ�ߴ��������Ϊtransformer�����룬�Ӷ�����С����token����������С�˿�����
- CNN�Ĳ�νṹ�󣬽���ͬ�߶ȵ�������ϣ�����transformer���ں��в����ӣ���������scale-aware�����������ḻ������Ϣ��������ߴ磬��SIM( Semantics Injection Module)ע��������Ϣ����Ӧtoken�����һ������Ԥ��

## 22.RTFormer: Efficient Design for Real-Time Semantic Segmentation with Transformer(2022)
[��������](https://arxiv.org/pdf/2210.07124)

���ĸĽ���transformer�е���ע������һ������EA(�ⲿע����)���GFA(GPU�Ѻ�attention)��ʵ�����Ը��Ӷȣ���һ���棬���ڸ߷ֱ�������ͼ������ע�������ܲ�����õ���ȡ�����ϵ����Ϊÿ��λ�õĸ���Ҳ���Ƚ�С���ɴ����һ����ֱ���ע����ģ�飬ʵ����һ����Ч˫�ֱ�������RTFormer
- GFA:EA����Double Normalization����softmaxʵ�ֹ�һ����������ʵ�ֶ�ͷע������GFAʹ���˷���Double Normalization���ڵڶ�����һ��ʱ���飬���ʵ�ֶ�ͷ��Ч����һ���棬�����˶�ͷ�������˾���˷����Ӷ���GPU�Ѻã����Խ��ⲿ������ά��������M\*H����ѧϰ�����������ˣ���һ���棬����Double Normalization�����˶�ͷ������
![Alt text](Paper/Real-time_Semantic_Segmentation/image/30.png)
- �����˿�ֱ���ע����ģ�飬����ֱ���ע����ͨ�������ӵͷֱ��ʷ�֧��ѧ���ĸ߼�֪ʶ������Ч���ռ��߷ֱ��ʷ�֧��ȫ����������Ϣ��
- ���������л�������DDRNet�е�DAPPM��Ϊ������ģ��
![Alt text](Paper/Real-time_Semantic_Segmentation/image/31.png)
## 23.FFNet:Simple and Efficient Architectures for Semantic Segmentation(2022)
[��������](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Mehta_Simple_and_Efficient_Architectures_for_Semantic_Segmentation_CVPRW_2022_paper.pdf)

������ΪĿǰ��sotaģ�����̫�����ˣ��Լ��ٲ��Ѻã������һ������Ļ��ڼ�encoder-decoder��FPN����ResNet����Ϊbackbone��FFNet��������Ϊ�����򵥵�CNN�ܹ�������ָ���������������Ÿ����Ǳ��

![Alt text](Paper/Real-time_Semantic_Segmentation/image/32.png)

����Ƚ���Ȼ��stem����FPNǰ�Ķ���������һ��������㣩��FFNet��һ�������������ͨ���ԣ�FFNet�ܹ��������ɸ�������backbone�����͡���Ⱥ���ȣ������ߴ��С��head�����ͺ�head���

## 24.DWRSeg: Dilation-wise Residual Network for Real-time Semantic Segmentation(2022)
[��������](https://arxiv.org/pdf/2212.01173)

������Ϊѡ����ʵĸ���Ұ��С����ȡ������Ч�ʺ���Ҫ����ǳ����Ҫ��С�ĸ���Ұ����׽�ֲ����������߲���Ҫ����ĸ���Ұ��׽�������������ң��ڸ߲�ʹ�������ʴ�Ŀն����û�����壬��Ϊ��������������ںܴ�Ŀռ�ά���Ͻ�������֮�����ϵ����ˣ�������Ը���Ұ�������������ȱ仯��block��ǳ����SIR���򻯵ķ�תƿ���в�飩���߲���DWR
- SIR��������תƿ����ȥ��һ��1\*1���ڵͲ�ʹ��SIR��ʹ����Ұ������󣬼���ϸ����ʧ����ƽ��
- DWR:��·��block���ֱ�ʹ�ò�ͨ���������ʣ���ò�ͬ����Ұ�����������У����������·����ά������������ΪС����Ұ��������ȡ��ÿһ�㶼����Ҫ

![Alt text](Paper/Real-time_Semantic_Segmentation/image/33.png)

- ����ṹ��stem��ԭͼ�²���4����֮���Ƚ�SIR���ٽ�DWR�����ս�϶�߶�����ͼ����Ԥ��

![Alt text](Paper/Real-time_Semantic_Segmentation/image/34.png)
## 25.SeaFormer: Squeeze-enhanced Axial Transformer for Mobile Semantic Segmentation(2023)
[��������](https://arxiv.org/pdf/2301.13156)

����Ҳ�Ǽ�transformer�е���ע���������һ��ͨ��ע������Sea attention(squeeze-enhanced Axial attention)�������ӶȽ���O(HW)��������ͼ��С������
- Ӧ����Sea block�����SeaFormer����������Ϊmobile v2�飬����ǰ�����˫��֧�ṹ�������������ķ�֧�����ռ��֧�ںϣ���������Ϣ��Ȩ�ռ��֧������ͼ��
![Alt text](Paper/Real-time_Semantic_Segmentation/image/35.png)
- Sea block:��Sea��ע������FFN���
![Alt text](Paper/Real-time_Semantic_Segmentation/image/36.png)
  - squeeze-enhanced Axial������˼�壬ѹ��ά�ȣ�����ѹ�������ע��������ԭʼK(H\*W\*C)�ֱ���ˮƽ�ʹ�ֱ����ѹ��ΪH\*C��W\*C(ȡƽ��ֵ�����еĹ�ʽ���Ǳ��ȡ��ֵ����˼)��Q/Vͬ���ٷֱ�����ͷ��ע����������Ϊ����λ����Ϣ����ѹ�����kqv�ֱ����λ��Ƕ�룬����֪���Լ����ĸ�����ѹ���ģ�����
  - ��Ȼ�������͸��Ӷȣ�����ʧ�˺ܶ�ϸ����Ϣ�����������Detail enhancement kernel����KQV������������һ����ȿɷ���3\*3��1\*1
## 26.AFFormer:Head-Free Lightweight Semantic Segmentation with Linear Transformer(2023)
[��������](https://arxiv.org/pdf/2301.04648)

����������Ľ����ǿ�������һ��û�н���ͷ������������ָ��ض��ܹ�AFFormer����Ƶ�ʵĽǶȽ��ͣ����Ӷ�Ϊ O(n)

![Alt text](Paper/Real-time_Semantic_Segmentation/image/37.png)
- ����ͨ�����ཫ����ͼH\*W\*CתΪԭ��������3\*3�ֲ����ں���Ϣ��������ͼh\*w\*C����ԭ������ͼͨ��PL( prototype learning)����������ȡ������transformer�ģ���������ע�������滻��AFF�����Ӷ�Ϊ$O(hwC^2)���پ���PD(pixel descriptor)���任���ԭ�������ָ�ΪH\*W\*C
- AFF(Adaptive Frequency Filter):������Ϊ���۶�Ƶ��³���Ժ�ǿ����ģ�ͱȽ����У���Ҫ��ǿ�Զ���Ƶ����Ϣ�����á�AFF��������ģ��
![Alt text](Paper/Real-time_Semantic_Segmentation/image/38.png)
  - FSK:��KQV���飬����һ��attention���㣬Ŀ����ͨ�������������ǿ�Էָ�������Ƶ��
  - DLF:��ͨ�˲���ʹ�õ���ƽ���ػ���Bin�Ĵ�С��ͬ�����һ���ϲ���
  - DHF:��ͨ�˲���ʹ�õ��Ǿ��
- һ���棬û��decoder�����ٲ������ҽ�attention�Ż�Ϊ���ԣ��ҽ�token��HW����hw����һ���棬�����AFF��Ƶ�ʽǶ�ѧϰ����ԭ�͵ľֲ�������ʾ��������ֱ��ѧϰ��������Ƕ��������ʹ���Ⱥ��б�֤
> Ч��̫���ˣ�˵ʵ���ܶ�ط�û̫����������ȥ�����룬�治֪��Ϊʲô����ôwork��������mmsegmentationд�ģ����Ժܷ��㣡
## 27.LETNet:Lightweight Real-time Semantic Segmentation Network with Efficient Transformer and CNN(2023)
[��������](https://arxiv.org/pdf/2302.10484)

�����һ����������������LETNet
![Alt text](Paper/Real-time_Semantic_Segmentation/image/39.png)
-  Lightweight Dilated Bottleneck:ʹ�÷ֽ�������ȿɷ�������SEע���������shuffle��һ��ͨ��
-  efficient Transformer:��ʵ��������KQVǰ��һ��ά
-  ���һ��������ǿģ��FE���ֱ���ͨ���Ϳռ䷽����SEע���������Ƶģ�����������
-  ������һ��ixel Attention (PA)��Ҳ�Ǻ�SE����

## 28.ISANet:Interlaced Sparse Self-Attention for Semantic Segmentation(2019)
[��������](https://arxiv.org/pdf/1907.12273.pdf(%C3%A6%C2%AD%C2%A3%C3%A5%C5%93%C2%A8%C3%A9%CB%9C%E2%80%A6%C3%A8%C2%AF%C2%BB%C3%A7%C5%A1%E2%80%9E%C3%A4%C2%B8%E2%82%AC%C3%A7%C2%AF%E2%80%A1))

����ע�����ļ��㣬�����һ�ֽ���ϡ����ע��������ԭ���ܵ��׺;���O($N^2$)�ֽ�Ϊ����ϡ����׺;�����㣬����С���ڴ�/FLOPs/��ʱ��

������˵����H\*W�����뻮�ֳ�m\*n��h\*w�ķ���(H=mh,W=nw)���ȼ��㡰��������������ÿ��С������ͬλ�õ�Ԫ���ó���������µķ��񣨴�СΪm\*n����������ע�������ټ��㡰�̾�������������һ���õ�������ͼ�ָ�ԭ��״���ڼ���ÿ��h\*w�����ڲ�����ע�������ܵ���˵��ÿ��λ�õ�Ԫ�ض����Եõ���������λ��Ԫ�ش���������Ϣ��

![Alt text](Paper/Real-time_Semantic_Segmentation/image/17.png)

������ֱ�Ӹ�����һ�μ���pytorch���룬�Ҷ���ʵ�ַ�ʽ�Ի󣬷��ֶ�pytorch��permute��������Ϥ���������£�

>����pytorch�е�reshape��permute����
- reshape/view:�൱�ڽ�ԭ����������ֱ���������µ���״
- permute:���ڸ�ά�������ı�Ĺ�����ʲô�أ�
~~~
����ά�Ƕ�ֱ����⣬��һ������߹̶��ĳ����壬permuteֻ�ǴӲ�ͬ�Ƕ�ȥ���������³���߷����˸ı䣬��ʵ����ͬһ�������壬����view��ֱ�Ӹı䳤����ĳ���ߣ��൱��һ��û���Σ�һ��������
~~~
������6*8����Ϊ��
~~~py
tensor([[0.5855, 0.9252, 0.7436, 0.0545, 0.1243, 0.2341, 0.4057, 0.8889],
        [0.1092, 0.4451, 0.2793, 0.1091, 0.5837, 0.2935, 0.5816, 0.9718],
        [0.9975, 0.9356, 0.9426, 0.4008, 0.3347, 0.1301, 0.1406, 0.5253],
        [0.0612, 0.1610, 0.5503, 0.5757, 0.5057, 0.8157, 0.1558, 0.5449],
        [0.8162, 0.8662, 0.8467, 0.5890, 0.9397, 0.1468, 0.9264, 0.9635],
        [0.7283, 0.6237, 0.3733, 0.4426, 0.3941, 0.9812, 0.6998, 0.7632]])
~~~
Ϊ��ʵ�֡�ȡÿ��2\*2С�����ͬһλ�ã����3\*4���������ע������������reshapeתΪ3*2*4*2
~~~py
tensor([[[[0.5855, 0.9252],
          [0.7436, 0.0545],
          [0.1243, 0.2341],
          [0.4057, 0.8889]],

         [[0.1092, 0.4451],
          [0.2793, 0.1091],
          [0.5837, 0.2935],
          [0.5816, 0.9718]]],


        [[[0.9975, 0.9356],
          [0.9426, 0.4008],
          [0.3347, 0.1301],
          [0.1406, 0.5253]],

         [[0.0612, 0.1610],
          [0.5503, 0.5757],
          [0.5057, 0.8157],
          [0.1558, 0.5449]]],


        [[[0.8162, 0.8662],
          [0.8467, 0.5890],
          [0.9397, 0.1468],
          [0.9264, 0.9635]],

         [[0.7283, 0.6237],
          [0.3733, 0.4426],
          [0.3941, 0.9812],
          [0.6998, 0.7632]]]])
~~~
���Կ����ǰ������ġ�  
�ٽ���permute(1,3,0,2)��תΪ2\*2\*3\*4��״����ʱΪ
~~~py
tensor([[[[0.5855, 0.7436, 0.1243, 0.4057],
          [0.9975, 0.9426, 0.3347, 0.1406],
          [0.8162, 0.8467, 0.9397, 0.9264]],

         [[0.9252, 0.0545, 0.2341, 0.8889],
          [0.9356, 0.4008, 0.1301, 0.5253],
          [0.8662, 0.5890, 0.1468, 0.9635]]],


        [[[0.1092, 0.2793, 0.5837, 0.5816],
          [0.0612, 0.5503, 0.5057, 0.1558],
          [0.7283, 0.3733, 0.3941, 0.6998]],

         [[0.4451, 0.1091, 0.2935, 0.9718],
          [0.1610, 0.5757, 0.8157, 0.5449],
          [0.6237, 0.4426, 0.9812, 0.7632]]]])
~~~
�����м���ѭ�ġ����ʼ�Ŀռ���������ⷽ�������ǰ�ά������ȥ�����������
- ��һ��ά����2��4\*2�����еĵ�һ��2�����ǹ̶�����������Ƕ�����ԭ����Ϊ3\*4\*2�ķ��飬���磬��һ�������ķ�������
  ~~~py
  [[[0.5855, 0.9252],
    [0.7436, 0.0545],
    [0.1243, 0.2341],
    [0.4057, 0.8889]],
    [[0.9975, 0.9356],
    [0.9426, 0.4008],
    [0.3347, 0.1301],      
    [0.1406, 0.5253]],      
    [[0.8162, 0.8662],
    [0.8467, 0.5890],
    [0.9397, 0.1468],
    [0.9264, 0.9635]]]
   ~~~

- �ڶ���ά��Ϊԭ4\*2�����е�2�����ǹ̶�����������Ϊ3\*4���飬��һ�������ķ�������
  ~~~py
  [[0.5855, 0.7436, 0.1243, 0.4057],
    [0.9975, 0.9426, 0.3347, 0.1406],
    [0.8162, 0.8467, 0.9397, 0.9264]]
  ~~~
- �������͵��ĸ�ά�Ⱦ���Ȼ��

OK�����������꣬ʵ�֡�ȡÿ��2\*2С�����ͬһλ�ã����3\*4���������ע��������˼·�����ֳ����ˡ�

�Ƚ�6\*8������2\*2���黮�֣���Ϊ3\*2\*4\*2���������û�иı����ݵ�λ�ã��Ƚ�3\*2�е�2�ᵽ��ǰ���൱�ڰ���ÿ��2��ȡһ�У���Ϊ3\*4\*2���ٽ�4\*2�е�2�ᵽ��ǰ���൱�ڰ���ÿ��2�г�һ�У���Ϊ3\*4���ܵ���˵��ʵ����ÿ����������ȡһ��Ԫ�ء�

-----
���һ�����⣬��ô�����ҵķ��黻��ԭλ����3\*2\*4\*2�������Ƚ�3�ᵽ��ǰ����ʱʣ�µĵ�һ��2\*2\*4Ϊ����˳���ԭǰ���У��ٽ���һ��2�ᵽ��ǰ����ʱ�ĵ�һ��2\*4Ϊ����˳���ԭ��һ�У���2\*2�����е�11��12λ�ã��ٽ�4�ᵽ��ǰ���൱��ת�ã�ԭ����ͬһ�����������Ԫ�ص���һ�𣬵õ�4\*2��

��ˣ�ֻ��permute(2,0,3,1)���ɡ�

## 29.Denseaspp for semantic segmentation in street scenes(2018)
[��������](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
ûɶ�¶������൱��Densenet�����Щ�ܼ����ӣ����ɲ�ͬ�����ʵĿն�������������ڣ��õ��������Ұ��Χ��ͬʱ�����Ǹ��ܼ��ĳ߶ȷ�Χ��
![Alt text](Paper/Real-time_Semantic_Segmentation/image/18.png)


