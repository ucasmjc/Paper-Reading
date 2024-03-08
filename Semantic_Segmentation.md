- [Semantic Segmentation](#semantic-segmentation)
  - [1.FCN:Fully Convolutional Networks for Semantic Segmentation(2015)](#1fcnfully-convolutional-networks-for-semantic-segmentation2015)
  - [2.U-Net: Convolutional Networks for Biomedical Image Segmentation(2015)](#2u-net-convolutional-networks-for-biomedical-image-segmentation2015)
  - [3.Segnet: A deep convolutional encoder-decoder architecture for image segmentation(2016)](#3segnet-a-deep-convolutional-encoder-decoder-architecture-for-image-segmentation2016)
  - [4.PSPNet:Pyramid scene parsing network(2017)](#4pspnetpyramid-scene-parsing-network2017)
  - [5.Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs(2017)](#5deeplab-semantic-image-segmentation-with-deep-convolutional-nets-atrous-convolution-and-fully-connected-crfs2017)
  - [6.RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation(2017)](#6refinenet-multi-path-refinement-networks-for-high-resolution-semantic-segmentation2017)
  - [7.SERT:Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers(2021)](#7sertrethinking-semantic-segmentation-from-a-sequence-to-sequence-perspective-with-transformers2021)
  - [8.Deeplab v3:Rethinking atrous convolution for semantic image segmentation(2017)](#8deeplab-v3rethinking-atrous-convolution-for-semantic-image-segmentation2017)
  - [9.Bisenet: Bilateral segmentation network for real-time semantic segmentation(2018)](#9bisenet-bilateral-segmentation-network-for-real-time-semantic-segmentation2018)
  - [10.Psanet: Point-wise spatial attention network for scene parsing(2018)](#10psanet-point-wise-spatial-attention-network-for-scene-parsing2018)
  - [11.Deeplab v3+:Encoder-decoder with atrous separable convolution for semantic image segmentation(2018)](#11deeplab-v3encoder-decoder-with-atrous-separable-convolution-for-semantic-image-segmentation2018)
  - [12.Icnet:Icnet for real-time semantic segmentation on high-resolution images(2018)](#12icneticnet-for-real-time-semantic-segmentation-on-high-resolution-images2018)
  - [13.Non-local neural networks(2018)](#13non-local-neural-networks2018)
  - [14.EncNet:Context encoding for semantic segmentation(2018)](#14encnetcontext-encoding-for-semantic-segmentation2018)
  - [15.DANet:Dual attention network for scene segmentation(2019)](#15danetdual-attention-network-for-scene-segmentation2019)
  - [16.CCNet: Criss-Cross Attention for Semantic Segmentation(2019)](#16ccnet-criss-cross-attention-for-semantic-segmentation2019)
  - [17.ANN:Asymmetric non-local neural networks for semantic segmentation(2019)](#17annasymmetric-non-local-neural-networks-for-semantic-segmentation2019)
  - [18.Gcnet: Non-local networks meet squeeze-excitation networks and beyond(2019)](#18gcnet-non-local-networks-meet-squeeze-excitation-networks-and-beyond2019)
  - [19.OCRNet:Object-contextual representations for semantic segmentation(2020)](#19ocrnetobject-contextual-representations-for-semantic-segmentation2020)
  - [20.Pointrend: Image segmentation as rendering(2020)](#20pointrend-image-segmentation-as-rendering2020)
  - [21.Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation(2021)](#21bisenet-v2-bilateral-network-with-guided-aggregation-for-real-time-semantic-segmentation2021)
  - [22.DPT:Vision Transformer for Dense Prediction(2021)](#22dptvision-transformer-for-dense-prediction2021)
  - [23.Segmenter: Transformer for semantic segmentation(2021)](#23segmenter-transformer-for-semantic-segmentation2021)
  - [24.SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers(2021)](#24segformer-simple-and-efficient-design-for-semantic-segmentation-with-transformers2021)


# Semantic Segmentation
## 1.FCN:Fully Convolutional Networks for Semantic Segmentation(2015)
[��������](https://openaccess.thecvf.com/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)

����ʹ��ȫ�������ʵ����pixel-pixel,�˵��˵�����ָ�ģ�ͣ��������һ�����ö�߶������ķ���
- ��������������CNNȡ���˺ܺõ�Ч������Ǩ�Ƶ�����ָ����񡪡�ȥ������㡢��FC��Ϊconv�������ϲ���ʵ��dense predict�������Բ�ʹCNNֻ�ܽ��̶ܹ��ߴ�����룬��ÿ��FC���Ե�ЧΪһ������㣬���ʹ��FCN���Խ�������ߴ�����롣
- ���������У������������ȡ�����Ĺ����л᲻�ϵ��²�����ʹ����ͼ�ߴ粻���½�����ʹtop��������ֱ��ʽϵͲ���Ӧ��pixel-wise������ָ�������Ҫ�÷���������Ӧdense predict�����ļ�����overFeat�������shift-and-stitch������ûʹ�ã�������ʹ�����ϲ����������������/˫���Բ�ֵ�����һ���ϲ������������ʼ��Ϊ˫���Բ�ֵ����ѧϰ������pixel lossʵ����dense predict
- ��ϸ߷ֱ���ǳ��͵ͷֱ��ʸ߲������������FPNӦ���ǶԴ�����������ڶ�top�㣨����㣩�ϲ���32��ʱ��FCN-8s���������2���ϲ������뾭��1\*����ĵ��Ĳ������ӣ������2���ϲ��������뾭��1\*1����ĵ�������ӣ������8���ϲ����õ���ԭͼ�ߴ�һ�µ�������Ӷ�����˶���߶ȵ�����ͼ��������ںϸ�low�Ĳ�����ݼ���
- top����ͼ��ͨ����ΪC���������������൱������ͼ��ÿ����ΪCά������ÿ����ĵ÷֣�����Ϣ̫���ˣ������ں����߶ȵ��ϲ���������U-net�н����˸Ľ������ϲ��������Ա����˷ḻ������ͨ��

## 2.U-Net: Convolutional Networks for Biomedical Image Segmentation(2015)
[��������](https://arxiv.org/pdf/1505.04597.pdf%EF%BC%89)

����һƪ����ҽѧͼ�������ָ����ģ��������U-net��һ���㷺ȡ����������ģ��
- U-netҲ������ȫ������磬��FCN���ơ���ǰ�򴫵�һ��CNN����²�����һϵ������ͼ����top������ͼ��������3\*3�����󣬽���һϵ���ϲ���(\*2)��ÿ���ϲ����󣬽������**�²��������ж�Ӧ������ͼ**�ü���ƴ��һ��(concatenation)����������3\*3�����������һ���ϲ��������һ���ϲ�����ʹ��1\*1�������ÿ�����صķ��ࡣ�ϲ������²������̱Ƚ϶Գƣ��γ�һ��U�ͽṹ�������е�ͼƬ��������
- һ���Ƚ���Ҫ�ĵ㡣U-net�о��󲿷�ʹ�õ���3\*3����㣬û��pad������ÿ����һ�ξ���㣬����ͼ�ߴ綼��-2����Ϊ���ԭ���ϲ������²�����Ӧ������ͼ�ߴ�����������Ҫ���²���������ͼ�ü���concatenation��������Ϊ�ڱ�Եpad��ʹ��Ե���ص�������������Ӷ�Խ��Խģ��������ͼ�ߴ��½�Ҳ������overlap-tile�����й�
- Ҳ����ҽѧͼ������⣨�ֱ���̫�󣩣�Ҳ�����ǵ�ʱ���豸���ƣ��ڴ�С����Ҳ��������Ϊ������С����Ƭ��������������U-netʹ����overlap-tile���ԣ���ͼƬ��Ƭ��m\*m��patch������patch����padding����ȡpatch��Χ�����������أ���ʹpadding���patch����U-net�󣨳ߴ�ή�ͣ��ߴ�ǡΪm\*m������ͼƬ��Ե��patch��������Щ����û����������padding����ʱʹ�þ���padding����patch������Գơ�ͨ�����ַ�ʽ������ʵ�ֶ������ͼ������޷��и�����Ԥ�⣬ÿ��patchҲ�������������Ϣ��
- ��FCN���ϲ�����һ����ͬ��FCN�ϲ���ʱֱ�ӶԷ�������ϲ�������Ȼ�ܲ�׼��U-net���ϲ���ʱ�����ḻ����������������1\*1��������
> FCN�ڽ���²�������ͼʱ����1\*1�����ֱ����ӣ�U-net��concatenation�پ���3\*3����ںϣ�FPN���侭��1\*1���������پ���3\*3����ں�
- Ϊ����߶ԡ��Ӵ���Ŀ�ꡱ�����֣�����ʹ���˼�Ȩ��������ʧ��ʹ����һ����ʽ�������ģ�����ѵ��ǰ��ÿ��GTͼ����Ȩ��ͼ�����ַ�����ʹĿ����С�������нϸߵ�Ȩ��
- ҽѧͼ��ָ������һ����սΪ�б�ע���ݺ��٣�����ʹ����������ǿ�������漴�����α��Ч�����

## 3.Segnet: A deep convolutional encoder-decoder architecture for image segmentation(2016)
[��������](https://ieeexplore.ieee.org/iel7/34/4359286/07803544.pdf)

����ṹ��U-net���ƣ����²������ϲ��������࣬�����һ���µ��ϲ�����������С�ڴ档��Ȼ���ºܳ��������µ�����
- Segnet�Ķ�����ʵ�ָ�Ч�ĳ������ṹ����ע�����Ż�ʱ����ڴ����ģ�ͬʱ�ڸ���ָ���Ͼ��о�������
- SegNetӦ����encoder-decoder�ṹ���²������ϲ����׶Σ���encoderΪFCN�����+BN+ReLU+���ػ��õ��óߴ������ͼ��decoder���ϲ����ٽӾ������BN��ReLU������ʵ�����ؼ�����
- �ؼ��㣺��encoder�У�ֻ��¼����ͼmax poolingʱ���ֵ���������Ӷ�ʹ��Ҫ��¼��������Ϣ���ά���ϲ���ʱ����max pooling indices������encoder�ж�Ӧ����ͼ�ػ�ʱ�����������ʵ���ϲ�������Ӧ����ȡֵ���������㣩���ϲ����������ͼ��ϡ��ģ�����������������BN��ReLU���õ����ܵ�����ͼ������һ�׶ε��ϲ������ϲ�������ҪѧϰҲ�����Ч�ʡ�
- ʵ�������ʹ��ȫ��encoderʱ������ͼ���Եõ���õ�Ч���������ڴ�����ʱSegNet������߱���



## 4.PSPNet:Pyramid scene parsing network(2017)
[��������](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf)

���������Ӧ����Pyramid pooling module��PSPNet�����Ծۺϲ�ͬ�������������������������һ������loss��ѵ�����ResNet
- ������ȫ����Ϣ�������Ĺ�ϵ�Գ�������������ָ����Ҫ�ģ��򵥵�ʹ��ȫ�ֳػ�����ʧ�ռ��ϵ���������壬����ṩ��һ�ֽ������ػ����Ӷ�����ȫ�ֳ��������顣
- ��ͼƬ������������õ�top����ͼ�����䰴�ղ�ͬ�ߴ�ػ����ػ�����N\*N��bin(N=1,2,3,6)��N=1ʱ��Ϊ��һ���ȫ�ֳػ����������Եõ���ͬ�߶��������representation����ͬˮƽ����������Ϣ����ÿ���ػ����context representation����һ��1\*1����㽫N\*N�ߴ��ά�Ƚ�Ϊ1\N���Ӷ����ָ�ˮƽȫ������֮���Ȩ�ء�֮��ֱ�����ϲ�����˫���Բ�ֵ����ʹ�ߴ�ָ�Ϊԭ����ͼ��С���ٽ����ĸ���ԭ����ͼconcatenation�����о���Եõ����Ԥ��
- �����������磬ʹ����ResNet�����ž������ѵ��ʱ�����˶����һ�������ͼ����Ԥ�⣬��������һ��������ʧ����res4b22�в�����Ԥ�⣬��ͬ���򴫲��������磬�����Ż�ѧϰ���̡���ǰ�ߵ�Ȩ�ظ���
> �о�PSP��Ŀ�����е�SPP˼�����һ��


## 5.Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs(2017)
[��������](https://arxiv.org/pdf/1606.00915)

���������Deeplab v2����v1�Ļ����ϸĽ�����Ϊv1������û�������Զ�����Щ�ֲڣ�һЩϸ��ûŪ�����֮�����õ���ϸ�о�
- Deeplab����Ҫ�ص�Ϊ��Ӧ���˿ն����(atrous concolution)��ʹ��ASPPģ��(atrous spatial pyramid pooling)��ʹ����CRF(Conditional Random Field)����������ں����汾������
- ������Ӧ���ڷ��������CNN�����Կռ�任����һ����³���ԣ���Էָ����ⲻ�����������˷ֱ��ʡ�����ͬ�߶����塢��λ�����½�����һ���������ص�ֱ�����������ս
- �ն�������ն���������ڱ�������ͼ��Ұ��С��ͬʱ������Ұ��Ϊ���������Ұ����ȥ�����Ӳ�����ػ����ή������ͼ��Ұ��С����DeeplabӦ���˿ն��������Resnet������ػ��㼰֮��ĳػ���Ϊ����Ϊ2�Ŀն�������Ӷ���ԭ�����²���32����Ϊ�²���8����֮������˫���Բ�ֵ�ϲ���8�����ָ�ԭͼ��ߴ����Ԥ��
> �ն�������ܵ���grid problem��������Ұ���󣬵�ĳЩ���ڽ������ر����ԣ�����ͨ������ʹ�ò�ͬ�ߴ�Ŀն������ʱ����Ұ����
- ASPP����Ԥ��ʱ��Ϊ�˻�ö�߶�������������ͼ������4���߶��µĿն��������ֱ��ֽ��˾���㣬���õ���4�������һ��ȫ�ֳػ�ֵ����ȫ�ֳػ��ٲ�ֵ��ϸ�ڲ�������岿��concatenation��������������Ԥ��
> ��Ϊûʱ���v1�������ˣ����Ҳ��̫��Ҫ�ɣ�����������transformerʱ���ˣ�������һЩϸ��û�㶮���Ժ���˵

## 6.RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation(2017)
[��������](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.pdf)

����Ҳ��Ϊ�˽���²��������е��µķֱ����½����⣬���RefineNet�����²��������е�������Ϣ��ϸ������������Ϣ��top������ͼ���Ұ������ϲ����������һ����ʽ�в�ػ�
- �������ֱ����½�������ָ����񳣼�����ս��һ�ַ���Ϊ�²�����ͨ��������ȷ�ʽ�ϲ���������ʵû������ϸ������������һ�ַ���ΪDeeplab����Ŀն�������ڱ�����Ұ��С��ͬʱ�������Ұ����һ�������������ά���ռ�úܴ��ڴ棬��һ����ն����Ҳ��һ���²�����Ǳ�ڵĶ�ʧһЩ��Ϣ�����������RefineNet��ʹ���²��������еĶ�߶ȵġ��߷ֱ�������ͼ��ϸ�������ϲ���ʱ������Ϣ�ḻ���ֱ��ʵ͵�����ͼ��˼���FPN�Ƚ�����
- RefineNet�²���ʱʹ�õ�ResNet����ṹ�������˵ڶ����ػ��㿪ʼ������ͼ(1/4--1/32)����1/32������ͼ����RefineNet4(����һ��block)�����1/32��������ͼ���ٺ�1/16����ͼһ������RefineNet3�����1/16��������ͼ��������ȥ��ֱ���õ��ں���ϸ����������1/4����ͼ����softmax��˫���Բ�ֵ
- RefineNet��������ʲô���Ƚ�1/2������ͼ����ӦResnet�������ͼ����һ��RefineNet���������ֱ���������������RCU(�в�����Ԫ),ÿ��RCU��������3\*3�����ReLU�Ͳв����ӣ����г���RefineNet4�����ά��Ϊ512����Ϊ256��RCU��Ŀ���ǽ�Ԥѵ���������ڷ��������ͼ��Ӧ�ڷָ�����һ�ֽ��Ͱ��ˣ������������multi-resolution fusion���ֱ�����3\*3�������ά��ͳһΪ��͵ģ����ϲ��������ߴ�ͳһΪ���ģ�������ӣ����������Chained Residual Pooling����������м����Ĵ��в����ӵĳػ�+����飬Ҳ����ÿ����һ�γػ�+�����������ε�������������뵽��һ���ػ�+������������Եõ��ḻ�Ĳ�ͬ�߶ȵĳػ���������ͨ�����Ȩ�ؼ���������Ϊ����������Ч��׽�����������������������ͨ��һ��RCU�õ����������
- �����������У�Ӧ���˷ḻ�Ĳв�˼�룬���ж̳̣����ڣ��Ĳв����ӣ������ϲ���ʱ���²���ʱ������ͼ���ӣ����ݶȸ����׵Ĵ�����ǰ�Ĳ����У������ڶ˶Զ�ѵ��

## 7.SERT:Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers(2021)
[��������](https://arxiv.org/pdf/2012.15840.pdf)

����transformer�ṹӦ�õ�����ָ�����ʹ��encoder-decoder�ܹ������SETR�����µ㲻��
- Ԥ������ViTһ�����ȷֳ�patch����ӳ�䵽patch embedding������position embedding��Ϊ����
- encoder:24��tranformer encoder�飬��Ӧ����24������ͼ
- decoder:����ָ���ѵ����ڣ�������ͼ�ĳߴ�ָ���ԭͼ�ֱ��ʣ��������������decoder��ʽ
  - Naive:��encoder���һ������ͼreshape��3D�����þ���㽫ά��תΪ���������˫���Բ�ֵ��ԭ�ߴ�
  - PUP:��encoder���һ������ͼreshape��3D�󣬽����ϲ���\*2�;����
  - MLA(multi-Level feature Aggregation):��FPN���ƣ�ȡM��encoder������ͼ����reshape��3D���ٷֱ𾭹�������4���ϲ������ټ���һ���������ӣ��ֱ𾭹�����㣬�ٰ�ά��concatenation,��󾭹�������4���ϲ����õ�ԭ�ߴ�

## 8.Deeplab v3:Rethinking atrous convolution for semantic image segmentation(2017)
[��������](https://arxiv.org/pdf/1706.05587.pdf%EF%BC%8C%E6%8E%A8%E8%8D%90%E5%9C%A8%E7%9C%8B%E5%AE%8C%E6%9C%AC%E6%96%87%E4%B9%8B%E5%90%8E%E4%BB%94%E7%BB%86%E9%98%85%E8%AF%BB%E8%AE%BA%E6%96%87%E4%BB%A5%E5%8F%8A%E4%BB%A3%E7%A0%81%EF%BC%8C%E5%8F%AF%E4%BB%A5%E6%9B%B4%E5%A5%BD%E7%90%86%E8%A7%A3%E3%80%82)

��deeplab v2�Ļ����Ͻ����˸Ľ�������˼����Ļ�����Ӧ���˿ն������ģ�飬������v2�������ն��������backbone���ɵõ��ֱ��ʸ��������ͼ(1/8)
- ��������Resnet�ĺ󼸸�block�ĳɿն����������������ֱ��ʲ��䣬ÿ��block֮�䡢���ڲ��ľ����֮��ն����������ϵ����������һ�����ֹgrid problem����һ�����������Ұ
- ������ASPP�����Ľ������㣬������BN���ն����������ϵ��̫��Ļ�����Ч�㣨padding��������������ӣ��ﲻ���������Ұ��Ŀ�ģ���˼�����Image-level������ȫ�ֳػ��㣩�����1\*1������ϲ�������ASPP���ƴ����һ��


## 9.Bisenet: Bilateral segmentation network for real-time semantic segmentation(2018)
[��������](https://openaccess.thecvf.com/content_ECCV_2018/papers/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.pdf)

���������һ��˫�߷ָ�ģ��Bisenet��ʵ��Ч����Ч�ʵľ���

![Alt text](Paper/Semantic_Segmentation/image/1.png)

- ��Ҫ�ص��ǣ������������·����context path��spatial path��ǰ��ͨ�����ٵ��²���pretrained��������Xception���������Ұ����ýϵͷֱ��ʵĺ��ḻ��������������ͼ�����ARM(Attention refinement module)�����а���ȫ�ֳػ������߽�����������㣬�²���8������˾��ܳߴ�󵫼��������󣩣�������ԭͼ��ḻ�Ŀռ�������
- ��Ϊ����·������Ϣ��level��ͬ�������FFM����������ֵ�������BNͳһ�߶ȣ���SEģ��ѡ��������

## 10.Psanet: Point-wise spatial attention network for scene parsing(2018)
[��������](http://openaccess.thecvf.com/content_ECCV_2018/papers/Hengshuang_Zhao_PSANet_Point-wise_Spatial_ECCV_2018_paper.pdf)

����point-wiseע�������������λ�õ�ͬʱ����ȫ����Ϣ��ÿ���㶼����Ӧ��ͨ��һ����ѧϰ��ע����ӳ�����������е�����

![Alt text](Paper/Semantic_Segmentation/image/2.png)

- ������һ��˫����Ϣ����·�������ע�������������֣���һ����Ϊ������j��Ŀ���iԤ�����Ҫ�ԣ��ڶ�����ΪĿ���i��������j����Ҫ�ԣ��������ֶ�����ͼ��ÿ���㶼��H\*Wά��������������һ��2H-1\*2W-1ά������ͼ��ͨ���۽������Ĳ�ͬλ�ã����ÿ����H\*Wάע�������õ�attention map����ע����ͼ����ʽ�ɵ�ÿ�����������
- (��������ͼΪH\*W\/$C_2$) collect��H\*W\*(H\*W)ά��attention map��ÿ�����H\*Wά������ʾH\*Wÿ����Ըõ��ע������������Ӧ��Ȩ���ÿ�����$C_2$ά�������ɵøõ�����������distribution���ֵ�attetion map��ÿ�����H\*Wά������ʾ�õ��H\*W�������Ҫ�ԣ��������������ʱ��ȡȫ��ÿ�����H\*Wά�����еĵ�iά��Ϊ��ȫ�ֵ��Ŀ���i��ע������Ȩ���ۼӿɵ��������

## 11.Deeplab v3+:Encoder-decoder with atrous separable convolution for semantic image segmentation(2018)
[��������](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf)

Ϊ���ڱ�֤�ֱ��ʵ�ͬʱ����������Ϣ��deeplab v3ʹ�ÿն��������ػ����Ӷ���֤�ߴ��ͬʱ�����˸���Ұ���������ַ�������encoder-decoder�Ա߽���Ϣ��ϸ�ڡ���ˣ�deeplav v3+�����encoder-decoder�ṹ����v3��Ϊһ��ǿ���encoder��֮�����һ���򵥵�decoder����̽������ȿɷ���ն������Ӧ����ASPP��decoder��

![Alt text](Paper/Semantic_Segmentation/image/3.png)

## 12.Icnet:Icnet for real-time semantic segmentation on high-resolution images(2018)
[��������](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hengshuang_Zhao_ICNet_for_Real-Time_ECCV_2018_paper.pdf)

�����һ��ʵʱ����ָ���ICNet�����ü���ͼƬ���룬�ںϲ�ͬ�ߴ������ͼ��ʵ��coarse-to-fineԤ�⣬�ڵͷֱ�������ͼʹ���������磬�ڸ߷ֱ��ʲ���ʹ�����������磬������ǰ���㹲��Ȩ�أ��Ӷ�������С��������

![Alt text](Paper/Semantic_Segmentation/image/4.png)

- ��CFF(cascade feature fusion)ģ�飬ʹ��˫���Բ�ֵ�Ϳն����ʵ�ֲ�ͬ�ߴ�����ͼ���ں�

- ʹ�ø�����ʧ��ÿ���ߴ������ͼ���ᱻ����Ԥ�Ⲣ������ʧ��������ʧ���Ȩ


## 13.Non-local neural networks(2018)
[��������](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)

�������һ�� non-local ��������һ��ͨ�õ�non-local block����self-attentionͳһ��non-local�ķ�ʽ�У��������һЩ�������ܵ�ѡ��

![Alt text](Paper/Semantic_Segmentation/image/5.png)

## 14.EncNet:Context encoding for semantic segmentation(2018)
[��������](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Context_Encoding_for_CVPR_2018_paper.pdf)


����� Context Encoding Module,������������Ϣ��������SENet��������ͼ��ÿ��ͨ����Ȩ

![Alt text](Paper/Semantic_Segmentation/image/6.png)

- ENCģ���е�encoder layer��ͨ����ͳ�����õ�K������ʣ�����softmax��Ȩ�õ�ÿ�����ض�ÿ������ʵĲв��������ۼӵ�����ͼ��ÿ������ʵĲв�����
- ��encoder layer�����inputȫ���Ӳ㣬�õ�ÿ��ͨ����Ȩ��
- �����˸�������SE-loss��GT���Դӷָ�GT�л�ã�ÿ�����Ķ�Ԫ������


## 15.DANet:Dual attention network for scene segmentation(2019)
[��������](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf)

Ϊ�˸��õĲ�׽��������Ϣ��ȫ����Ϣ����ͨ�������ϵ�����������һ��˫ע��������DANet��ʹ������ע����ģ�����õ����õ�������ʾ

![Alt text](Paper/Semantic_Segmentation/image/7.png)

position attention module����������ͼH\*Wά�ȵ���ע�������õ�(H\*W)\*(H\*W)��ע�����������󣬼����Ȩֵ��channel attention module����������ͼͨ��ά�ȵ���ע�������õ�C\*C��ע�������������ټ����Ȩֵ����󽫶����ںϡ�

## 16.CCNet: Criss-Cross Attention for Semantic Segmentation(2019)
[��������](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_CCNet_Criss-Cross_Attention_for_Semantic_Segmentation_ICCV_2019_paper.pdf)

���������������Ϣ��ע����ģ��/non-localģ�飬����Ҫ���ɳߴ�ܴ��ע������������(H\*W)\*(H\*W)����������ռ���ڴ�󡣱������һ���µľۺ�ȫ��������ע����ģ��CCA��ÿ������ڵ�ǰ�к��м���ע������ע������������ΪH\*W\*(H+W-1)������С����ͨ��ѭ������CCA����ȡȫ����Ϣ

![Alt text](Paper/Semantic_Segmentation/image/8.png)

## 17.ANN:Asymmetric non-local neural networks for semantic segmentation(2019)
[��������](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhu_Asymmetric_Non-Local_Neural_Networks_for_Semantic_Segmentation_ICCV_2019_paper.pdf)
��CCNetһ�����������non-local�ı׶ˣ������ANN��Ӧ����Ҳ����ע������APNB(Asymmetric Pyramid Non-local Block)��AFNB(symmetric Fusion Non-local Block)��ǰ����ȡȫ�������������ں϶�߶�����

![Alt text](Paper/Semantic_Segmentation/image/9.png)

APNB��AFNB�������ڣ���ע����ģ���key��query��ά��ͨ��������C\*HW������C\*S������SԶС��HW��������ʽΪSPPģ�飬����Ҳ�ں��˶�߶�������

## 18.Gcnet: Non-local networks meet squeeze-excitation networks and beyond(2019)
[��������](http://openaccess.thecvf.com/content_ICCVW_2019/papers/NeurArch/Cao_GCNet_Non-Local_Networks_Meet_Squeeze-Excitation_Networks_and_Beyond_ICCVW_2019_paper.pdf)

��Ȼ��Ϊ�˼�non-local����������������Ƿ���non-local��׽��ע����ͼ������query����һ�����Ӷ������һ������ע����ͼH\*W�ļ�non-local�顣���н�SENet�ͼ򻯵�non-local��ͳһ��Global context modeling framework���������ߵ����ƽ�ϣ��ȼ���ȫ��ע������query����ע����ͼ��H\*W������C\*HW��ˣ��õ�Cά������ͨ������1\*1��bottleneck����C\*H\*W��ӣ��в����ӣ����ɹ㲥���ƣ���ʵ�൱��ÿ�����ؾ�ע������Ȩ�����������ȣ����������Cά����

![Alt text](Paper/Semantic_Segmentation/image/10.png)

## 19.OCRNet:Object-contextual representations for semantic segmentation(2020)
[��������](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510171.pdf)

���ľ۽��ھۺ����������������ȥ��ASPP�������non-local������ͬ�����Ľ��������������������������ḻ������

![Alt text](Paper/Semantic_Segmentation/image/11.png)

��backbone���������ͼΪH\*W\*C
- soft object region:K\*H\*W.Ϊÿ��ͼԤ��K��object region������ÿ��regionΪһ��2Dͼ��ÿ�����ص�ֵ��ʾ����������������ԡ��˹������мල�ģ���GT���ɵ���GT��������
- object region representation:��soft object region�ֱ��Ȩ����������ͣ����յõ�K��Cά������ÿ��������ʾ�ö������������
- Pixel-Region Rela:ͨ�����+softmax����ÿ�����ص���ÿ���������������ԣ��õ�H\*W\*K����ԭ��������������������Ԥ��
- ���������̵��������ʱ���и�transformation,1\*1+BN+ReLU

## 20.Pointrend: Image segmentation as rendering(2020)
[��������](http://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf)

���Ľ��ָ�����������Ⱦ�������Ե���ȵ����⣬��������һ���µ��ϲ�����ʽ������ѵ����

![Alt text](Paper/Semantic_Segmentation/image/12.png)

����CNN��ȡ����ϸ��������ͼ���Ȳ���һ�������ָ�����õ�coarse predict(7\*7)��Ӧ��˫���Բ�ֵ�����ֱ��ʷ���������Щ��������N�����ѵ㡱�����Ŷȵͣ�����˵�߽紦��������һ��MLP����Щ������Ԥ�⣨������������ͼ�ʹ�Ԥ��ͼ����������Ԥ��ͼ����˫���Բ�ֵ������ֱ��Ԥ��ͼ�ķֱ��ʴ��ڵ���ԭͼ��

## 21.Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation(2021)
[��������](https://arxiv.org/pdf/2004.02147)

������Bisenet v1����ƣ��ֱ��������֧��ϸ�ڷ�֧��׽������������ϸ�������������v1�����ľ�������������֧�����������²����Ĳ��о��stem block��Ӧ������ȿɷ������ۺ�������Gather-and-Expansion Layer���ͷ������׽�߲��������Context Embedding Block��ȫ�ֳػ���1\*1�ٲв���ӣ������Ļ���������������ۺ�ģ�飬��������֧����������ͬlevel�ֱ�ۺϣ��õ����õı�����

![Alt text](Paper/Semantic_Segmentation/image/13.png)

## 22.DPT:Vision Transformer for Dense Prediction(2021)
[��������](http://openaccess.thecvf.com/content/ICCV2021/papers/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.pdf)

��������ܼ�Ԥ���vision transformer DPT�������˸��������ԡ�

![Alt text](Paper/Semantic_Segmentation/image/14.png)

- encoder���ֺ�ViTһ����������һ�����ڷ����readout token����token��read�����б�����/��Ϊȫ�������ںϡ�
- decoder���Բ�ͬtransformer��������װ��**��ͬ�ֱ���**��������ͼ�����ʽ������ͨ��bottleneck����/�²��������н�low��transformer��ᱻ��װ�ɸ���ֱ��ʵı�ʾ����Ϊ���а�������ϸ����������֮��ʹ����������refinenet�ķ�ʽ���ں϶�߶ȵ�����ͼ

## 23.Segmenter: Transformer for semantic segmentation(2021)
[��������](https://openaccess.thecvf.com/content/ICCV2021/papers/Strudel_Segmenter_Transformer_for_Semantic_Segmentation_ICCV_2021_paper.pdf)

���������һ���µ�mask transformer decoder���Ӷ����Segmenterģ��

![Alt text](Paper/Semantic_Segmentation/image/16.png)

- encoder��ViTһ��
- ��encoder�����K��patch encoding�Ϳ�ѧϰ�������ʼ����K��classǶ��һ������mask transformer��ά�Ⱦ�ΪD���������ÿ��patchǶ�����Ƕ���������(N\*D)\*(D\*K)=N\*K���Ӷ��õ�ÿ��patch�������룬��reshape���ϲ����õ�Ԥ��ͼ


## 24.SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers(2021)
[��������](https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf)

���������һ�ּ򵥸�Ч�Ļ���transformer������ָ�ģ��Segmenter����α�������������������

![Alt text](Paper/Semantic_Segmentation/image/15.png)

- ������Swin transformer����encoder��ͨ���ϲ���������ò�ͬ�ֱ��ʵ�����ͼ���������ڣ�Segmenter�ϲ������ص��Ĳ����������ֺϲ���ľֲ�������
- ͨ����encoder��FFN�м���3\*3��Ⱦ���������ṩ�㹻��λ����Ϣ���Ӷ�ʡ����λ�ñ���
- ʹ����һ����������ALL-MLP��decoder���Ƚ���ͬ�ֱ��ʵ�����ͼͳһά�Ⱥͳߴ磬��ͨ��MLPԤ�⡣�����������Ĺؼ��ǣ�encoder�ṩ�Ķ�߶�����ͼ����Ϣ�ܷḻ������Ұ����