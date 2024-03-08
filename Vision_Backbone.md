# Vision Backbone- [Vision Backbone](#vision-backbone)
- [Vision Backbone- Vision Backbone](#vision-backbone--vision-backbone)
  - [1.HRNet:Deep High-Resolution Representation Learning for Human Pose Estimation(2019)](#1hrnetdeep-high-resolution-representation-learning-for-human-pose-estimation2019)
  - [2.Resnest: Split-attention networks(2022)](#2resnest-split-attention-networks2022)
  - [3.Mobilenet v2: Inverted residuals and linear bottlenecks(2018)](#3mobilenet-v2-inverted-residuals-and-linear-bottlenecks2018)
  - [4.mobilenet v3:Searching for mobilenetv3(2019)](#4mobilenet-v3searching-for-mobilenetv32019)
  - [5.Beit: Bert pre-training of image transformers(2021)](#5beit-bert-pre-training-of-image-transformers2021)
  - [6.ConNext:A convnet for the 2020s(2022)](#6connexta-convnet-for-the-2020s2022)
  - [7.MAE:Masked autoencoders are scalable vision learners(2022)](#7maemasked-autoencoders-are-scalable-vision-learners2022)
  - [8.Segnext: Rethinking convolutional attention design for semantic segmentation(2022)](#8segnext-rethinking-convolutional-attention-design-for-semantic-segmentation2022)
  - [9.ViT:An image is worth 16x16 words: Transformers for image recognition at scale(2020)](#9vitan-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale2020)
  - [10.Swin Transformer: Hierarchical Vision Transformer using Shifted Windows(2021)](#10swin-transformer-hierarchical-vision-transformer-using-shifted-windows2021)
  - [11.Xception: Deep Learning with Depthwise Separable Convolutions(2017)](#11xception-deep-learning-with-depthwise-separable-convolutions2017)
  - [12.Parsenet: Looking wider to see better(2015)](#12parsenet-looking-wider-to-see-better2015)
  - [13.Shufflenet: An extremely efficient convolutional neural network for mobile devices(2018)](#13shufflenet-an-extremely-efficient-convolutional-neural-network-for-mobile-devices2018)
  - [14.Shufflenet v2: Practical guidelines for efficient cnn architecture design(20)](#14shufflenet-v2-practical-guidelines-for-efficient-cnn-architecture-design20)

## 1.HRNet:Deep High-Resolution Representation Learning for Human Pose Estimation(2019)
[��������](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.pdf)

����̬���������������һ����ȡ��߶���������������HRNet�������������б����˸߷ֱ����������������ܼ�Ԥ�⣨λ�����У�����������ɱ��϶����͡���������ֻ������߷ֱ��ʵ�����ͼ����Ԥ�⣬��Ȼ���Խ�϶�߶�һ��Ԥ�⡣

![Alt text](Paper/Vision_Backbone/image/1.png)

- ��ͼ��ʾ�������ںϲ�ͬ�ֱ��������Ĳ��֣���ͷ����Ϊexchange unit���þ�����²��������ڽ��ϲ�����1\*1�����ͳһͨ����������ӵõ���Ӧ�߶�����ͼ��
- ͼ���У���һ�β������ͷֱ�������ͼʱ�����������г߶�����ͼ�������������ֻ�������ڽ��߶ȵ�����ͼ


## 2.Resnest: Split-attention networks(2022)
[��������](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Zhang_ResNeSt_Split-Attention_Networks_CVPRW_2022_paper.pdf)

�����SENet�е�channel-wiseע������ResNext�е�group convolution����SKNet�е�split-attention�������һ�ָ�ǿ��Resnest��û��ʲô�µķ���

![Alt text](Paper/Vision_Backbone/image/4.png)



## 3.Mobilenet v2: Inverted residuals and linear bottlenecks(2018)
[��������](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)

��mobile v1����ȿɷ����������ϣ�������Linear Bottleneck��Inverted residuals

![Alt text](Paper/Vision_Backbone/image/2.png)

- Linear Bottleneck:��������Ϊ��������X��ÿ������ռ�BX�ķֲ��൱��һ����Ȥ���Σ����ο���ӳ�䵽��ά�ռ䣨�ֲ����ܼ�������ReLU����������Ե�ͬʱ���ƻ������е���Ϣ�����磬�������ռ�BXά�ȱȽϵͣ�$B^{-1}ReLU(BX)$�ָ���X�ƻ������أ���ά�򻹺á���������Ƹ�Ч����ʱ��ϣ�������ܵؽ���ά�ȣ��ֲ�ϣ��ReLU�ƻ�̫����Ϣ����˳����ڵ�ά�ȵľ��������ȥ�������Բ㣬�Ա���������Ϣ����Ϊһ��Linear bottleneck��
- Inverted residuals����ͼ��ʾ��v2�Ƚ�������ά��Ϊ����3\*3���ʱ��ȡ���ḻ�����������ٽ�����ȿɷ����������ͨ��Linear Bottleneck��ûReLU����ά����Ϊ����bloc��ͷ���м�񣬺�residual block�෴�����Ե������ܴ�̶Ȼ�����depth-wise������ЧӦ��



## 4.mobilenet v3:Searching for mobilenetv3(2019)
[��������](http://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)


��mobilenet v2�Ļ��������v3�������˺ܶ�trick���������
- ����SENet�е�ע�������ƣ�����˵����Ӧ��Ȩ
- ʹ��NAS������Ѳ���
- ������һ���µļ����

## 5.Beit: Bert pre-training of image transformers(2021)
[��������](https://arxiv.org/pdf/2106.08254.pdf)

����BERT�����������image tensformer���Ա���Ԥѵ��ģ��BEIT
- ��������ViT��࣬image patch--40% mask--embedding(��mask����ר�����)--add position--transformer encoder--predict Visual Token
- ѵ�����̣���ѵ��һ��dVAE,�ٸ���MIM(Masked Image Modeling)����ѵ��
- ѵ��dVAE��ϣ�����ÿ��patch��token(��ʵ�Ǿ���һ����������һ����ı�)������ֱ���õ�DELL-E��
- MIM����Ԥ��masked patch��Ӧ��Visual Token��softmax+������
- ����ʹ��dVAE��˼�������BEIT��ԭ����ʧ������������֤��patch�������������Ч����

## 6.ConNext:A convnet for the 2020s(2022)
[��������](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)

���Ľ��ViT,��transformer�����Խ����CNN�У����˺ܳ�ֵ�ʵ�����Ľ����磬�����һ������ɵĴ�CNN����ConvNext��������Swin transformer
- һ����ViT�����۹��������⣬CNN��induced bias������translation equivariance��������Ϊ����Ȼ��CNN�����ƣ���ΪSwin transformer���������һ���Բ�����ViT���ɴ���Ϊ��CNN�ܹ�δ�ⲻ�ܳ���transformer����Ҫ����ResNet����������������ǵ�
- �����ƣ�ÿ��block�Ĳ������������ViT��Changing stem to ��Patchify����������SWin�еĲ���������CNN��������Ϊ4����СҲΪ4���²���
- ���ResNeXt��������ȿɷ������Ͳ��о��
- ����inverted bottleneck��������˳�򣬽����˲�����
- ��ReLU����GELU���ø��ٵļ�����������Բ㣩�����ٵ����򻯲㣬��BN����LN,���²����������������ר�ŵ�һ������㣩



## 7.MAE:Masked autoencoders are scalable vision learners(2022)
[��������](http://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf)

���BERT��mask+�Ա��뷽��������vision transformer���Ӿ�����Ԥѵ��ģ��MAE

- ���Ժ��Ӿ�����Ϣ�ܶ����������Ծ��зḻ��������������ʹmask��һС����Ҳ�����и��ӵ�������⣻���Ӿ��������ڿռ��Ͼ��������ԣ�Ϊ�˿˷����������ݵ�gap���Ϳ˷������Ը��õ�ѧ�����õ���Ϣ����Ҫmask������ʣ���80%
![Alt text](Paper/Vision_Backbone/image/3.png)

- ֻ��ûmasked��patch�������������˿��Թ����ϴ�ı���������ͳһ��mask token������������룬����������������Ӧpatch����������


## 8.Segnext: Rethinking convolutional attention design for semantic segmentation(2022)
[��������](https://arxiv.org/pdf/2209.08575.pdf?trk=public_post_comment-text)

���������һ��Ϊ�˷ָ�����ļ򵥾������Segnext������˼�����Ҫ��������ǿ���encoder����߶��������ռ�ע����
- encoder��transformer���ƣ����þ��ע�����������ע����������˵����ע��������Ч�����л�����depth-wise�Ĵ�״�������׽��״����������Ұ���䣬���ٲ���
![Alt text](Paper/Vision_Backbone/image/5.png)
- decoderʹ���˶�߶�����ͼ
![Alt text](Paper/Vision_Backbone/image/6.png)
## 9.ViT:An image is worth 16x16 words: Transformers for image recognition at scale(2020)
[��������](https://arxiv.org/pdf/2010.11929)

���������Vision Transformer����transformer�ܹ�Ӧ�õ�ͼƬ�������⣬����Ԥ����ͬ���������һ�����ڷ����transformer������
- ��п����ǣ���ƪ�����ǳ���Yolo v3�Ǹ����������������˳����һ����һ���棬ViT�������ı�transformer�ṹ��Ϊ�˷����ֱ��ʹ��nlp�����Ѿ���Ӳ���ϸ�Чʵ�ֵ�transformer�ṹ������һ����attention is all you need���Ҷ��ĵ�һƪ���ģ����ĺ���ϸ����ӡ������������ǡ�
- Ԥ����Ϊ�˵õ��������룬��һ��ͼƬ�ָ�Ϊ���patch��ά��Ϊ**patch����\*(patch��\*��\*ͨ����**\)����һ��patch��������Ϊһ��token����ͨ����ѵ��������ӳ��õ�Dάpatch embedding��Ϊ�˱���λ����Ϣ��ViTҲʹ����1άposition embedding��2άЧ��ûɶ��������Ϊ��ʵ�ַ������������п�ʼ������һ����ѵ����[class]token��������״̬��Ϊ���������
- inductive bias:������Ϊ��CNN����translation equivariance��locality��inductive bias������ģ�������һ�����飩�������ŵ㵫Ҳ�����ޣ�����ģ���Լ�ѧϰ������transformer����������inductive bias���٣�ֻ��MLP��position embedding�����ռ��ϵ�����ͷ��ʼѧ������ڴ����ݼ���ѵ��ʱ����CNN�����õ����飩�� 
- ΢������΢��ʱ��removeԤѵ���ķ���ͷȻ�����³�ʼ������ѵ������ѵ����ͼ��ֱ��ʸ���Ԥѵ��ʱ��Ϊ�˱�֤Ԥѵ����position embedding��Ч���ڱ���patch-size�����ͬʱ������patch�����λ�ö�embedding���ж�ά��ֵ
- �������ᵽ�������е����ݼ���ѵ��ʱ��transformer�ı��ֲ���CNN�����������������ݼ������ʱ��ViTͨ���ڴ������ݼ���Ԥѵ������΢���õ���sota���֡�
- ���Ļ��ᵽһ�ֻ��ģ�ͣ�����CNN��ȡpatch���������ٶ���patch & position embedding��Ϊ����

## 10.Swin Transformer: Hierarchical Vision Transformer using Shifted Windows(2021)
[��������](https://arxiv.org/pdf/2103.14030)

���������һ���µ�vision transformer�ṹSwin transformer������shifted window���ͼ��㸴�Ӷȣ���ͨ��patch merge��ö�߶�����ͼ����������������FPN��U-Net��������dense prediction����
- ������������Ϊ����transformerӦ����Visionʱ��Ҫ����������֮����������һΪ�Ӿ�ʵ����в�ͬ�ĳߴ磬��Ϊ�Ӿ��������Ҫ�߷ֱ������룬��transformer������Ϊƽ�����Ӷȡ�Ϊ�˽�����������⣬Swin transformer�ֱ�ʹ���˲������ͼ�ͼ���ֲ�self-attention�ķ���
- �ṹ��Ԥ������ViT���ƣ���ͼƬ��Ϊpatch�����embedding���������������position embedding������������������Swin transformer block�����patch merge���������ڵ�patch(2*2=4��)concatenation��4d���������Բ㽵Ϊ2d���Ӷ�ʹ����ͼ�����Ϊһ�룬�൱�ڲ���Ϊ2���²��������������������������Swin transformer block���ظ��������
- Swin transformer block���������������֣����ǵĶ�ͷ��ע������(MSA)��ͬ�����Ƚ������һ��transformer block����MSAΪw-MSA����ÿ�����ص���window(ÿ��window����M\*M��patch)�ֱ������ע����������һ��block�Ľ������ڶ�������MSAΪSW-MSA��������ͼ����shifted window�������ǹ����windows���ָ����µ�window������ע����
- w-MSAʹ��ע�����ļ���תΪ���Ը��Ӷȣ�SW-MSA����w-MSA�Ĳ�ͬwindow֮��Ĺ�ϵ���ḻ��ȫ��������
- �����������һ�ָ�Чmask��������shifted window����ע���������忴����
- ����ʹ�������λ��ƫ��ڼ�����ע����ʱ���롣��Ϊ$M^2$��$M^2$��patch֮����(2M-1)\*(2M-1)�����λ�ù�ϵ��ÿ��ά��2M-1��������ѵ��һ��(2M-1)\*(2M-1)ά�ȵ�bias���󣬼���ʱ����ȡֵ����
## 11.Xception: Deep Learning with Depthwise Separable Convolutions(2017)

[��������](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf)

��Inceptionģ��������Ƶ�����ȿɷ������ĸ���ֱ��ڿռ��ͨ��ά�Ƚ��о����������ȿɷ��������������Xception���Ľ���resnet��inception��Ӧ��bottleneck��ά��depthwise


## 12.Parsenet: Looking wider to see better(2015)

[��������](https://arxiv.org/pdf/1506.04579.pdf)

�ں�ȫ��������Parsenet��L2 norm��Ϊ��ͳһ�������̫����������ϲ�����Ϊ������ȫ��������Ӱ��
![Alt text](Paper/Vision_Backbone/image/8.png)

## 13.Shufflenet: An extremely efficient convolutional neural network for mobile devices(2018)
[��������](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf)

mobilenet��bottleneck(1\*1+3\*3+1\*1)�е�3\*3����Ż�����ȿɷ�����������С�˿��������ǣ���ʱbottleneck��1\*1����ļ��㿪��ռ��90+%��shufflenet������������Ż���1\*1����Ĳ��֡���Ϊ�����ķ������ᵼ��������������������������˿�ѧϰ��channel shuffle���Ӷ�ʹ�������������
![Alt text](Paper/Vision_Backbone/image/10.png)


## 14.Shufflenet v2: Practical guidelines for efficient cnn architecture design(20)
[��������](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.pdf)

���������ʹ��FLOPs���������������һ�ּ��ָ�꣬���������ԭ�򣬲��Դ˸Ľ�shuffle v1
- ͬ��ͨ����С��С���ڴ����������������ͨ�������ʱ�ڴ��������С����˽�v1�е�botleneck��Ϊ���������ͨ�������
- ����ʹ������������MAC�����鲻�˹��࣬��v1�е������ָ����������
- ������Ƭ���ή�Ͳ��жȡ�������һ����Ƭ�������ȡ����
- ���ܺ���Ԫ�ؼ���������ADD,Relu����ȡ��v1�Ĳв����ӣ���Ϊconcatation
![Alt text](Paper/Vision_Backbone/image/9.png)
- ����ʵ��ʱ���Ƚ�channel���飬����һ����ʹ�ú��ӳ�䲢����һ���ֵ����concatation������������shuffle