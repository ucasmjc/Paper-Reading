- [Vision Transformer](#vision-transformer)
  - [1.ViT:An image is worth 16x16 words: Transformers for image recognition at scale(2020)](#1vitan-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale2020)
  - [2.Swin Transformer: Hierarchical Vision Transformer using Shifted Windows(2021)](#2swin-transformer-hierarchical-vision-transformer-using-shifted-windows2021)
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
- [Semantic Segmentation](#semantic-segmentation)
  - [1.FCN:Fully Convolutional Networks for Semantic Segmentation(2015)](#1fcnfully-convolutional-networks-for-semantic-segmentation2015)
  - [2.U-Net: Convolutional Networks for Biomedical Image Segmentation(2015)](#2u-net-convolutional-networks-for-biomedical-image-segmentation2015)
  - [3.Segnet: A deep convolutional encoder-decoder architecture for image segmentation(2016)](#3segnet-a-deep-convolutional-encoder-decoder-architecture-for-image-segmentation2016)
  - [4.PSPNet:Pyramid scene parsing network(2017)](#4pspnetpyramid-scene-parsing-network2017)
  - [5.Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs(2017)](#5deeplab-semantic-image-segmentation-with-deep-convolutional-nets-atrous-convolution-and-fully-connected-crfs2017)
  - [6.RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation(2017)](#6refinenet-multi-path-refinement-networks-for-high-resolution-semantic-segmentation2017)
  - [7.SERT:Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers(2021)](#7sertrethinking-semantic-segmentation-from-a-sequence-to-sequence-perspective-with-transformers2021)
- [С��](#С��)

# Vision Transformer
## 1.ViT:An image is worth 16x16 words: Transformers for image recognition at scale(2020)
[��������](https://arxiv.org/pdf/2010.11929)

���������Vision Transformer����transformer�ܹ�Ӧ�õ�ͼƬ�������⣬����Ԥ����ͬ���������һ�����ڷ����transformer������
- ��п����ǣ���ƪ�����ǳ���Yolo v3�Ǹ����������������˳����һ����һ���棬ViT�������ı�transformer�ṹ��Ϊ�˷����ֱ��ʹ��nlp�����Ѿ���Ӳ���ϸ�Чʵ�ֵ�transformer�ṹ������һ����attention is all you need���Ҷ��ĵ�һƪ���ģ����ĺ���ϸ����ӡ������������ǡ�
- Ԥ����Ϊ�˵õ��������룬��һ��ͼƬ�ָ�Ϊ���patch��ά��Ϊ**patch����\*(patch��\*��\*ͨ����**)����һ��patch��������Ϊһ��token����ͨ����ѵ��������ӳ��õ�Dάpatch embedding��Ϊ�˱���λ����Ϣ��ViTҲʹ����1άposition embedding��2άЧ��ûɶ��������Ϊ��ʵ�ַ������������п�ʼ������һ����ѵ����[class]token��������״̬��Ϊ���������
- inductive bias:������Ϊ��CNN����translation equivariance��locality��inductive bias������ģ�������һ�����飩�������ŵ㵫Ҳ�����ޣ�����ģ���Լ�ѧϰ������transformer����������inductive bias���٣�ֻ��MLP��position embedding�����ռ��ϵ�����ͷ��ʼѧ������ڴ����ݼ���ѵ��ʱ����CNN�����õ����飩�� 
- ΢������΢��ʱ��removeԤѵ���ķ���ͷȻ�����³�ʼ������ѵ������ѵ����ͼ��ֱ��ʸ���Ԥѵ��ʱ��Ϊ�˱�֤Ԥѵ����position embedding��Ч���ڱ���patch-size�����ͬʱ������patch�����λ�ö�embedding���ж�ά��ֵ
- �������ᵽ�������е����ݼ���ѵ��ʱ��transformer�ı��ֲ���CNN�����������������ݼ������ʱ��ViTͨ���ڴ������ݼ���Ԥѵ������΢���õ���sota���֡�
- ���Ļ��ᵽһ�ֻ��ģ�ͣ�����CNN��ȡpatch���������ٶ���patch & position embedding��Ϊ����

## 2.Swin Transformer: Hierarchical Vision Transformer using Shifted Windows(2021)
[��������](https://arxiv.org/pdf/2103.14030)

���������һ���µ�vision transformer�ṹSwin transformer������shifted window���ͼ��㸴�Ӷȣ���ͨ��patch merge��ö�߶�����ͼ����������������FPN��U-Net��������dense prediction����
- ������������Ϊ����transformerӦ����Visionʱ��Ҫ����������֮����������һΪ�Ӿ�ʵ����в�ͬ�ĳߴ磬��Ϊ�Ӿ��������Ҫ�߷ֱ������룬��transformer������Ϊƽ�����Ӷȡ�Ϊ�˽�����������⣬Swin transformer�ֱ�ʹ���˲������ͼ�ͼ���ֲ�self-attention�ķ���
- �ṹ��Ԥ������ViT���ƣ���ͼƬ��Ϊpatch�����embedding���������������position embedding������������������Swin transformer block�����patch merge���������ڵ�patch(2*2=4��)concatenation��4d���������Բ㽵Ϊ2d���Ӷ�ʹ����ͼ�����Ϊһ�룬�൱�ڲ���Ϊ2���²��������������������������Swin transformer block���ظ��������
- Swin transformer block���������������֣����ǵĶ�ͷ��ע������(MSA)��ͬ�����Ƚ������һ��transformer block����MSAΪw-MSA����ÿ�����ص���window(ÿ��window����M\*M��patch)�ֱ������ע����������һ��block�Ľ������ڶ�������MSAΪSW-MSA��������ͼ����shifted window�ָ����µ�window�����window�ߴ�С��M\*M�����忴���ģ�������ע����
- w-MSAʹ��ע�����ļ���תΪ���Ը��Ӷȣ�SW-MSA����w-MSA�Ĳ�ͬwindow֮��Ĺ�ϵ���ḻ��ȫ��������
- �����������һ�ָ�Чmask��������shifted window����ע���������忴����
- ����ʹ�������λ��ƫ��ڼ�����ע����ʱ���롣��Ϊ$M^2$��$M^2$��patch֮����(2M-1)\*(2M-1)�����λ�ù�ϵ��ÿ��ά��2M-1��������ѵ��һ��(2M-1)\*(2M-1)ά�ȵ�bias���󣬼���ʱ����ȡֵ����
  
# Object Detection
## 1.R-CNN:Rich feature hierarchies for accurate object detection and semantic segmentation(2014)

[��������](https://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)



�������R-CNN��������CNN��������Ŀ�������򣬴�������Ŀ����Ч�����Դ�Ϊ��ʼ����һϵ��Two-Stage Object Detection������

- ����ѵ�����̿ɴ��·�Ϊ�ģ��������׶Σ���CNN�����ڸ��������ݼ��Ͻ����мල��image-level����Ԥѵ������ֱ��ʹ��AlexNet�Ĳ���������ԭCNN�����ȫ���Ӳ㻻Ϊ�ض���Ŀ����������������N+1����������region proposals����΢������ʱ�����͸������ж����ƽ��ɣ���ʹ��softmax��ʧ������region proposals��ͨ��CNN�����������N+1ά��ѵ��SVM�����ÿ�����ķ������˹����������͸������ж�Ҫ���ϸ�ֱ��ʹ��CNN�����softmax���ϴ󣩣������һ����������������ѵ��bounding-box regression
- ����ʱ����ÿ������ͼƬ��ȡ2k����region proposals����region proposals����Ϊ�ض���С������CNN������ȡ����������ȡ������������SVM���з����ֺ�NMS���ٽ�ʣ�µ�proposals���һ��������������������bounding-box regression���Ա߽�����΢��
- �ڱ��������ʱ����Ŀ������б�������٣���ȥ����ͨ���޼ල����Ԥѵ��������������������ݼ��Ͻ����мල��Ԥѵ����ʹ�ø�����������ࣩ�ٽ����ض������΢����������ѵ��������CNN�����ض���������ϡ��ʱ��
- ����ʹ����selective search������ȡregion proposals���������Ƚ�ͼƬ���ָ��ͨ��һЩ����ʽ�Ĺ�����кϲ���ͨ���÷����õ���region proposals�ߴ粻��ֱ����ΪCNN�����룬���Ĳ�ȡ����padding�������ٸ��������ŵķ�����
- ����ʹ��CNN���磨AlexNet������������ȡ����ȥ�����˹��趨����������SIFT,HOG
> SIFT��HOGδ�˽����֮���ԭ�����˽�һ��
- �����趨�õĽ�������ֵ���������͸�������SVMѵ����ѵ��ʱÿ��batch�������͸����ı���ȷ��
- bbox�ع�ϣ����region proposals����΢��������ϸ�ڼ����ĸ�¼����Ҫѵ�����ĸ����Բ���Ϊƽ��/��������
> �ᵽ��DPM��δ�˽���������ƺ���bbox�ع��˼���࣬����DLʱ������ʱ���ˣ��㲻��ԭ������


## 2.SPP-Net:Spatial pyramid pooling in deep convolutional networks for visual recognition(2015)
[��������](https://arxiv.org/pdf/1406.4729.pdf)

���Ľ�SPM����(Spatial pyramid matching)Ӧ�õ�CNN�У������SPP-Net������R-CNNЧ������ͬʱ������
- SPP(Spatial pyramid pooling)������ͬ�ߴ�����뻮��Ϊ�飨���趨�õ�����������ÿ����ֱ�ػ�
- SPP�����ƣ����ɱ䳤�ȵ�����ת��Ϊ�̶����ȵ�������ұ����ռ�����������ʹ�ö���ߴ��ͼƬ����ѵ������Ŀ����θ���³���ԣ������ڶ���߶ȣ������������ȡ����
- ÿ��ͼƬֻ����CNN��ȡһ����������ԭͼ��region proposals��Ӧ������ͼ�����ɻ����CNN������������ͼ��ͬ�ߴ��region proposalsͨ��SPP�ػ������õ��Ĳ�ͬ�ߴ��representionƴ�����������ɻ����ͬ�ߴ������������������FC����
- SPP-Net��ÿ��ͼƬֻ��ȡһ����������������Ч�ʣ���ѵ���׶κ�R-CNNһ�����ӣ���Fast R-CNN�õ����


## 3.Fast R-CNN(2015)
[��������](https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)

Fast R-CNN��R-CNN���죬��׼ȷ

- ѵ�����̣���һ��ͼƬ��RoIs��Ҳ��ͨ��selective search������CNN�������һ���ػ��㻻ΪRoi�ػ��㣩���ھ�������������ͼ���ҵ�ԭͼRoi��Ӧ��Roi����CNN�²����ı������ţ���������ͼ��Roi�ػ���ͳһ�ߴ磩����ȫ���Ӳ㣬�ֱ�õ�softmax�����bbox�ع��ƫ�����������ߵĹ�ͬ��ʧͬʱ���򴫲���(����ع���ʧʱʹ����smooth L1 loss)
- �����ԭ��Fast R-CNN���÷ֲ��������ÿ��ͼƬֻ��Ҫǰ�򴫵�CNNһ�α����ȡ���Rio��Ӧ��������ʹ�ö�������ʧ��ͬʱ������������Ĳ�����������R-CNNһ���ֶ���׶�
- Roi�ػ��㣬������ͼ��С��ͬ��Roi�ֿ����ػ�ͳһ�ߴ磬�������������䷴�򴫲�
- Fast R-CNNʹ����VGG����
- ����ʱȫ���Ӳ��ʱ�ܳ�����������ֵ�ֽ���٣��ֽ������С��ȫ���Ӳ㣩


## 4.Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks(2015)
[��������](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)

����Region��CNN��Ŀ����ȡ���˺ܺõ�Ч��������ͳRegion proposals��������selective search����Ϊ����ʱ��ʱ��ƿ�����������RPNs(Region Proposals Networks)���ںͼ�����磨��Fast R-CNN������CNN�����Ļ����ϣ�������������ľ����ʵ������regions proposals�����죬Ч������
- ��һ��H\*WͼƬ����VGG���õ�256άH/16\*W/16������ͼ��ÿ��������k(=9)��ê�򣬽�����ͼ����RPN������3\*3����㣨�����Ϊ256ά�����ֱ𾭹�����1\*1����㣬���Ϊ2kάH/16\*W/16��softmax����4kάH/16\*W/16���ֱ𴢴��˸�ê�����/����������ĸ��ʡ�ê��λ�õ�������������һ�����ϲ㣨���ᶪ�����ê�򣩼������region propsals���⼴ΪRPN��RPN�����proposals����Roi�ػ����������
- ������ͼ��ÿ����ѡ������k��ê��3�ִ�С�ͱ���������RPN�൱��k���������ͻع�����ͨ�����ַ�ʽ�����翼�ǲ�ͬ�ߴ��bbox
- RPN��Fast R-CNN������������ѵ��ʱ������4-Step Alternating Training��ʽ����ʧ������Fast R-CNN����


## 5.OHEM:Training region-based object detectors with online hard example mining(2016)
[��������](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)

�ڻ���region��Ŀ�������У�ǰ���ͱ�������������ƽ����һ����ս�����ⲻ��һ���µ���ս���ڹ�ȥ20��bootstrapping������hard negative mining���ڴ�ͳĿ���������У���SVM����ʹ�á��������Online Hard Example Mining�����boottrapping��������ѧϰ����ʽ���Ӷ�Ӧ�õ�CNN�����У�ȡ���˺ܺõ�Ч��
- bootstrapping��˼�����Ϊ��ѵ��ʱ������������еĽ׶Σ�ʹ��fixed modelѰ��hard example������������Υ����ǰ����߽磩�����뵽activeѵ�����У�ɾȥ������ȷ��example����ʹ��fixedѵ����ѵ��model��
- ������bootstrapping��������SGD���£�����ѧϰ������Ⱦ�����磬��������ٶȣ�OHEM����bootstrapping������ѧϰ��ʽ
- OHEM��˼��Ϊ����һ��ͼƬ����CNN��ȡ��������������RoI����ʧ��������ʧNMS��ѡ����ʧ��ߵ�������Ϊmini_batch���з��򴫲���������������ʧ��0��
- ��������ѧϰtool�����ƣ���ʹ��ʧΪ0��Ȼ��ռ���ڴ淴�򴫲����ʱ������һ�ֽϸ��ӵ�ʵ�ַ�ʽ��������һ��ֻ������

## 6.Yolo v1:You only look once: Unified, real-time object detection(2016)
[��������](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)

��R-CNNϵ�в�ͬ��Yolo��one-stageĿ���⣬����ר��Ԥ��region proposals���������Yolo v1������ʵ��ʵʱ��⣬�������дﲻ��sota���Ҷ�СĿ�����ϴ�
- ��image��ΪS\*S��grid��ÿ��grid����B��bbox��ÿ��bbox���������������x������y�����������Ŷȣ�����������x��y�������grid�ĳ�������һ�������Ϳ�ͨ������image�ĳ��Ϳ�����һ����
- ���Ŷȵļ��㣺ground truth�����Ŷ�ΪP(��bbox������grid���ж���)*IoU����ѵ��ʱ����Ŀ���������ڵ�grid�����ж���P(��bbox������grid���ж���)=1������Ϊ0
- ѵ��ʱ�������Ĳ�����ʧ�����ĵ�������ʧ�����ڸ���Ԥ���bbox��+������ʧ�����ڸ���Ԥ���bbox����ȡƽ������+���Ŷ����ֱ���ڸ���Ԥ����޶����grid��Ȩ�ز�ͬ��+��������������ж����grid��������ÿ��grid����ѵ��ʱֻѡ��ground truth IoU����bbox��responsible forԤ�⣬���඼���޶���
- ����������GoogleNet���ƣ�ÿ��ͼƬ����������ȫ���Ӳ��ĵ����ΪS\*S\*(B\*5+Class)��״��������ֱ��Ԥ��ÿ��bbox�������������Ԥ��ÿ��grid��ÿ�����ĸ��ʣ�Ԥ��ʱ��ÿ����ͨ��NMS����
- ���㳤�Ϳ����ʧʱȡƽ������ʹС�ߴ��bbox�Գ��Ϳ�ı仯������
- Ԥѵ��CNNʱ��224\*224��ͼƬ��Ϊ���룬���ʱ��448\*448���߷ֱ��ʵ�ͼƬ

## 7.SSD: Single Shot MultiBox Detector(2016)
[��������](https://arxiv.org/pdf/1512.02325.pdf%22source%22)

SSDҲ��һ��one-stageĿ���ⷽ�����ڲ�ͬ�߶ȵ�����ͼ�������������лع�ͷ���
- ��Ҫ˼�룺����������VGG��FC��Ϊ�������㣬ѡȡ�����6�����������ͼ����ÿ��cell����k����4-6���߶Ⱥͱ�����ͬ��Ĭ�Ͽ򣨼�ê�򣩣�ʹ���໥�����ľ���ˣ�3*3���ֱ�Բ�ͬ����ͼ����Ϊm\*nά���ϵ�bbox���з��ࣨN+1������Ŷȣ��ͻع飨offset�������ά��Ϊ(m\*n\*(k\*(classes+4)))�����õ������������8k+Ԥ��򣩽���NMS
- SSD���ö�߶�����ͼ���м�⣬����ʶ��ͬ�ߴ��Ŀ�꣨��Щ�����ǰ�ͼƬ����ɲ�ͬ�Ĵ�С��Ȼ���ϲ�ͬ��СͼƬ�Ľ��������Ϊ��������������ͼ��ά���½�������Ұ���󣬸�low������ͼ����ҰС��������ʶ��С�ߴ�Ŀ�ꡣ
- ��Yolo����Ҫ����1.�����˲�ͬ�߶ȵ�����ͼ��2.ʹ��������� 3.���þ���˽���Ԥ�����FC
- ѵ��ʱ�����������ᵽ��ƥ�����ȷ���������ٶԸ���������hard negative mining��������ʧ������Faster R-CNN����
- ������ͼ�����������Ĭ�Ͽ�ĳߴ�����Ĭ�Ͽ�ɰ�������Ӧ��ԭͼ
- SSD������������ǿ��ǿ³����


## 8.R-FCN: Object Detection via Region-based Fully Convolutional Networks(2016)
[��������](https://proceedings.neurips.cc/paper/2016/file/577ef1154f3240ad5b9b413aa7346a1e-Paper.pdf)

ResNet��ͼƬ������������ʣ������뽫FCNӦ�õ�Ŀ����������Ҫ�����һ���ȴ���FCNƽ�Ʋ����ԣ��ּ���Rio-wise layer�Ӷ����ٵķ��������⣬R-FCN��Ӧ����RPN��two-stage����
- ������ͼ��������µ�sota������Resnet��ʹ����ȫ����������Ȼ���뽫��Ӧ�õ�Ŀ���⡣��ֱ��Ӧ��Ч�����ã���Ϊ���**translation invariance**����ƽ�Ʋ����У��������ڷ������񣬶��������Զ����translation�����еġ���ʼ���������Ž�λ�����е�Roi pooling���뵽�����֮�䣬����**translation invariance**��������������unshared Rio-wise layers��ʹ����Ч�ʱ�ͣ���ˣ������R-FCN
- R-FCN����position-sensitive score mapsʹFCN��translation���У����ҳ�������position-sensitive Roi pooling��û�в������ܿ죩�����в㶼��shared��һ��ͼֻ��ǰ�򴫵�һ�Σ�������
- ������CNN��ȡ������ͼ��һ��������RPN����Roi����һ��������3*3���������������ͼ��Сһ�µ�$k^2(C+1)$��score map����ÿ������$k^2$�ţ�ÿһ�ŷֱ��Ӧ��һ��grid���൱�ڽ�Roi��Ϊk\*k��grid���ĵ÷ֲַ�������һ��Roi����score map���ҵ���Ӧλ�ã�����ÿ�����$k^2$������ͼ���ֱ����Ӧgridλ�õ�score grid��ƴ��һ��Rio score(��Roi��״��ͬ)����C+1��(**ԭ����ͼʾ�е���ɫ����Ҫ��**)�����ͨ��scoreͶƱ�������м򵥵Ľ�Rio score�������õ�Rio��score�����õ�C+1��score����softmax�ɵ÷�����ʡ�
- bbox�ع���������ƣ���������score map��$4k^2$������֮ǰ������
- ѵ��ʱʹ����OHEM��Ԥ��ʱʹ����NMS

## 9.YOLO9000:Better, Faster, Stronger(2017)
[��������](https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf)

yolo v2��yolo�Ļ����ϼ�����һЩ�����������Ƚ�idea����idea��������ܺ��ٶȣ�yolo9000��v2�Ļ��������÷������ݼ�����ѵ����ʹģ�Ϳ���ʶ��������𣨼�ʹ���ڼ�����ݼ��У�
- batch normalization��ʡȥ�������򻯺�dropout
- ��224\*224�ķ��༯��Ԥѵ��������448\*448��ͼ��Է�������΢���������448\*448��ͼƬѵ��������񣬴Ӷ�ʹԤѵ����ģ�͸�����Ӧ�߷ֱ��ʵ�ͼ��
- ʹ��ê��Ԥ��offset��ʹ��k-means������Ѱ��ê����������ߴ�&����
- ê��Ԥ��offset��ʽ�������仯����ê����������������grid��
- ʹ��ϸ�����������������ͼ����SSD����top����ͼΪ13\*13��������һ��paththrough�㣬��26\*26������ͼ���13\*13����top������ͼ����������ͬ��Ϊ��ȡ������
- ��ΪYolo v2ֻʹ���˾���ͳػ��㣬����ѵ��ʱʹ���˲�ͬ�ߴ��ͼƬ�����³����
- Ŀ��������ݼ�ԶС�ڷ��࣬��˱��������һ��ʹ���������ݼ�����ѵ���ķ��������ڴ�����ǩ��������������ʧ���Է����ǩ������ֻ�������ʧ���������ݰ�һ������������
- ͨ������WordTreeͳһ�������ݼ��ı�ǩ������WordNetȷ����ͬ��ǩ�Ĵ�����ϵ������������ĳ���ڵ㣨��ǩ���ľ��Ը��ʣ�ͨ����������͵�root�����и��������������˵õ���ÿ�������������ʼ���������������µĸ��ʣ�ͨ���Ը��ڵ������ӽڵ���softmax�õ�����ˣ���ѵ��ʱ��������ͬ��ʣ�����ͬһ���ࣩ��multiple softmax�����ÿ�������������ʣ��ٶԼ���õ��ľ��Ը�����loss�����ÿ��ͼƬ�����GT label�������и����Ӧ���ݶȣ��ڲ���ʱ����root��ʼ��ÿ��ѡ���������������ӽڵ㣬���ֱ�����Ը���**С��**ĳ����ֵ��
  


## 10.FPN:Feature pyramid networks for object detection(2017)
[��������](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)

���������FPN(Feature Pyramid Network)�����������ö�߶�����ͼ��ͬʱΪ�߷ֱ��ʵ�����ͼ����������Ϣ������˸�����õĽ��
- ������Ϊ��ʶ��ͬ�ߴ��Ŀ�꣬��ͳ������ͨ�����벻ͬ�ߴ��ͼƬ����ʱ����ڴ�����̫�󡣽�������SSDʹ�ò�ͬ�߶ȵ�����ͼ����Ԥ�⣬ʵ�ֲ�ͬ�ֱ��ʵ�Ԥ�⣬��һ���棬SSDΪ����ʹ��low������ͼ����������Ϊ��������������̫���ˣ�����˵̫�ֲ��ˣ����ӵ��ĸ�����㿪ʼ����Ԥ�⣬��ʹ�����СĿ���ⲻ���룬��Ϊlower��ķֱ��ʸ��ߣ���СĿ�������Ҫ����һ���漴ʹ�ӵ��Ĳ㿪ʼ�����top�㣬�ֱ��ʸߵĲ��������Ϣ���٣������ڼ��СĿ�꣬�Ҳ����֮��������gap��Ϊ�˽����ڶ���߶ȶ����зḻ��������������ͼ���������ҿ��٣����������FPN
- Bottom-up pathway:�Ե����ϵ�ͨ·����CNNǰ�򴫵��ڼ����ɵģ��߶Ȳ��ϼ�С������ͼ��ѡȡÿ���²���ǰ������ͼ�������˵�һ�㣨��Ϊ̫����ռ���ڴ棩����Щ����ͼ��Խtop�ֱ���Խ�ͣ�������ϢԽ�ḻ
- Top-down pathway and lateral connections����Bottom-up pathway��top����㾭��1\*1���������������������top�㣬����Ϊdά�����Զ�����������ÿ������ͼ���ߴ�Խ��Խ�󣩣����ɵ�����������ÿ�����״����Bottom-up pathway����ͬ��ά�ȿ��ܲ�ͬ�������ɷ�ʽΪ����toper������ϲ��������ļ򵥵�ʹ������ڣ���������lower��ߴ���ͬ��dά����ͼ���ٶ�Bottom-up pathway����Ӧ�ߴ������ͼ����1\*1���������dά����ͼ�����ϲ������dά����ͼֱ����ӣ�����һ��3\*3����������յ�����ͼ
- Ӧ����RPN����CNN����Ӧ��RPN��������������������ÿ���߶ȵ�����ͼ�ֱ�����ê��ÿ������ͼ��Ӧһ���߶ȺͶ�����������ٽ���RPN����Ԥ�⡣**���ڲ�ͬ�߶ȵ�����ͼ��Ӧ�ù�������ľ�������Ԥ��ͷ��༴��**��ԭ����Ϊ��ͬ����ͼ��������
- Ӧ����Fast F-CNN������Roi�ĳߴ罫��ָ���ͬ�߶ȵ�����ͼ����Ԥ�⣬�Ӷ�ʹ��С��Roi�ָ����߷ֱ��ʣ�Ԥ��ʱ����ÿ������ͼֱ��Ӧ��Roi pooling��������������Բ����Ԥ��ͷ��࣬�Ҳ�ͬ����ͼԤ��ͷ�Ĳ�������

## 11.RetinaNet:Focal loss for dense object detection(2017)
[��������](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

�����һ���µķ�����ʧ����Focal Loss��ʹhard example��Ӱ�����󣬽��one-stage�������������������ز��������⣬����� RetinaNet�ڱ���one stageģ�;��ȵ�ͬʱ������two stageģ�͵ľ���
- ������one stageģ�;��и���͸��򵥵�Ǳ���������Ȳ���two stage��������Ϊ��Ҫԭ����one stage��ǰ���뱳��������������ʧ�⣨�����򵥵ı�����������two stage��ͨ��RPNѡ��Roi�������򵥵ı���������Ԥ�������ʺܽӽ�1���ڱ�׼��������ʧ�����в��ɺ��Ե���ʧ����one stage�м򵥵ĸ����ܶ࣬�ᵼ��ģ��ѧϰЧ�ʲ��ʹģ���˻�������ͨ���޸ı�׼��������ʧ����������������������⣬ʵ���˱ȹ�ȥ����ʽ������OHEM�ȷ������õĽ��
- Focal loss:����˵���ӱ�׼������-lnx��תΪ$-\alpha(1-p)^\gamma log(p)$��ʹp������0ʱ��hard example������ʧ���󣬶�p����1ʱ����ʧ��С���Ӷ����ͼ򵥸�����Ӱ��
- RetinaNet��one stage��ʹ����FPN��Ϊ�������磬ʹ��ê��ع飨ѵ��ʱmatch����������ֵ�������ˣ�
- ֵ��ע���һ���ǣ���RetinaNet��ʼ��ʱ��ͨ�������һ��ȫ���Ӳ��ƫ��b���ã�ʹѵ����ʼʱ��ÿ��ê��ģ�ͱ�Ϊǰ���ĸ���Ϊ$\pi$(0.01)���Ӷ�ʹ����Ϊ������������������������ľ޴��ȶ�Ӱ�졣��Ĭ�ϳ�ʼ���Ļ���ǰ���ͱ����ĸ��ʲ�඼��0.5��

## 12.Mask r-cnn(2017)
[��������](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)

mask R-CNN��ʵ��Ŀ�����ͬʱ����ʵ������ָ����񣬲������Roi Asign������ʵ���˸��õĽ��
- ����ṹ�Ļ���Ϊ��Resnet+FPN��Faster R-CNN������ȡ����ͼ��һ��������RPN���Roi����һ����ʹ��Roi Asign��Roi���뵽�̶��ߴ��m\*m����ͼ����ԭ��bbox�ع�ͷ���Ļ����ϣ�����������ָ��֧��ʹ��FCN��m\*m����ͼ����mask����ÿ�����ؼ���C�����sigmod��δʹ��softmax�Ա������ľ�������������ʧΪƽ����Ԫ������
- ��Roi pooling��Roi��pixel-pixel���벻�ã������н���������ȡ����������Roi Asign�ɱ���������������˫���Բ�ֵ������ȡ����������������˾���
- ��ѵ��ʱ����������ָ��֧��ֻ��ÿ��mask��GT��maskֵ������ʧ
- Ԥ��ʱ����ͨ������֧Ԥ�����߷ֵ�k��Roi���ٶ����ǽ���mask��֧�����õ���m\*m\*C maskȡ����֧��Ԥ������ĵ÷֣�ά��Ϊm\*m������resize��Roi size����0.5��ֵ��ֵ��Ϊ����ָ����
> ����δ�m\*mά��mask resize��Roiԭ�߶ȴ��ɣ��ƺ�������ָ��е�dense predict����



## 13.Yolov3: An incremental improvement(2018)
[��������](https://arxiv.org/pdf/1804.02767.pdf)

�ⲻ��һƪ��ʽ�����ģ����ܹ�����arxiv�ϣ������������ӽ������ƺ���һ���������档���Ľ�����Yolo v3�����ĸ��£���һЩʵ������
- ��Ԥ��bbox����Ԥ����һ���bbox��Ŀ��ĸ��ʣ��÷֣�
- �ڷ��������У�����ʹ��yolo v2�е�multi softmax���������˶������߼���������ѵ��ʱʹ���˶����ཻ����ʧ��
- �����FPN�Ͳв�����
- Focal loss���ã��²�����ΪԤ��bbox����Ŀ��ĸ��ʣ�������ͬ��Ч����ʹ���򵥱���������������ʧ

## 14.DERT:End-to-end object detection with transformers(2020)
[��������](https://arxiv.org/pdf/2005.12872.pdf,)

����ͨ����Ԫƥ���transformer����������ṹʵ��DERT�ṹ�������˼�������pipeline����������˹���ƣ�NMS,ê�򣩣�ʵ���˼���Ԥ�⣨һ��Ԥ������ж��������λ�ã�
- �ṹ����¼���ϸ��ͼƬ�ǳ��ǳ���������������CNN��ȡ��������̶���λ�ñ�����Ӻ�����transformer encoder;��decoder����һ���ѵ����object query(Ӧ����N��)����cross-attention����Ӧ����encoder��������õ�N��Ԥ�⣨N��һ���̶�ֵ�����Դ���һ��ͼ�п��ܵ�Ŀ������������N��Ԥ������FFN���bboxԤ��ͷ�����ʣ�����"no object"��
- loss:
  - ѵ��ʱ�Ƚ�N��Ԥ����GT����Ԫƥ�䣨��GT��no object����NԪ�飩����Ѱ��һ�����У�ʹԤ���GT pair-wise��ƥ��ɱ���������͡�$y_i$��$y_{pre-i}$ƥ��ɱ�Ϊ����$y_i$����$c_i$Ϊ�޶�����0������Ϊ$-p_{pre-i}$($c_i$)+����bbox����ʧ��������
  - ѡ�����к󣬰�Ԥ����GT�Ķ�Ԫƥ�䣬����loss�����з���loss��-log��bbox��ʧʹ��L1 loss��generalized IoU loss����¼������ϸ˵�����ļ�Ȩ������ֻ��L1 loss�Գߴ����У�
- ����loss��ÿ��decoder��󶼻����FFN����Ԥ�Ⲣ������ʧ��������FFNǰ����һ����ͬ��֮��shared���һ��
- ���н��룺û��ʹ��ԭʼtransformer�е�auto-regressive���룬����ѵ��decoder�����롪��һ��object query��ֻ��cross-attention��ʹ��encoder�Ľ����ʵ���˲��н���

# Semantic Segmentation
## 1.FCN:Fully Convolutional Networks for Semantic Segmentation(2015)
[��������](https://openaccess.thecvf.com/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)

����ʹ��ȫ�������ʵ����pixel-pixel,�˵��˵�����ָ�ģ�ͣ��������һ�����ö�߶������ķ���
- ��������������CNNȡ���˺ܺõ�Ч������Ǩ�Ƶ�����ָ����񡪡�ȥ������㡢��FC��Ϊconv�������ϲ���ʵ��dense predict�������Բ�ʹCNNֻ�ܽ��̶ܹ��ߴ�����룬��ÿ��FC���Ե�ЧΪһ������㣬���ʹ��FCN���Խ�������ߴ�����롣
- ���������У������������ȡ�����Ĺ����л᲻�ϵ��²�����ʹ����ͼ�ߴ粻���½�����ʹtop��������ֱ��ʽϵͲ���Ӧ��pixel-wise������ָ�������Ҫ�÷���������Ӧdense predict�����ļ�����overFeat�������shift-and-stitch������ûʹ�ã�������ʹ�����ϲ����������������/˫���Բ�ֵ�����һ���ϲ������������ʼ��Ϊ˫���Բ�ֵ����ѧϰ������pixel lossʵ����dense predict
- ��ϸ߷ֱ���ǳ��͵ͷֱ��ʸ߲������������FPNӦ���ǶԴ�����������ڶ�top�㣨����㣩�ϲ���32��ʱ��FCN-8s���������2���ϲ������뾭��1\*����ĵ��Ĳ���ӣ������2���ϲ��������뾭��1\*����ĵ�������ӣ������8���ϲ����õ���ԭͼ�ߴ�һ�µ�������Ӷ�����˶���߶ȵ�����ͼ��������ںϸ�low�Ĳ�����ݼ���
- top����ͼ��ͨ����ΪC���������������൱������ͼ��ÿ����ΪCά������ÿ����ĵ÷֣�����Ϣ̫���ˣ������ں����߶ȵ��ϲ���������U-net�н����˸Ľ������ϲ��������Ա����˷ḻ������ͨ��

## 2.U-Net: Convolutional Networks for Biomedical Image Segmentation(2015)
[��������](https://arxiv.org/pdf/1505.04597.pdf%EF%BC%89)

����һƪ����ҽѧͼ�������ָ����ģ��������U-net��һ���㷺ȡ����������ģ��
- U-netҲ������ȫ������磬��FCN���ơ���ǰ�򴫵�һ��CNN����²�����һϵ������ͼ����top������ͼ��������3\*3�����󣬽���һϵ���ϲ���(\*2)��ÿ���ϲ����󣬽������**�²��������ж�Ӧ������ͼ**�ü���ƴ��һ��(concatenation)����������3\*3�����������һ���ϲ��������һ���ϲ�����shiyong1\*1�������ÿ�����صķ��ࡣ�ϲ������²������̱Ƚ϶Գƣ��γ�һ��U�ͽṹ�������е�ͼƬ��������
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

# С��
�����Ķ�Ч�ʣ�4 5 3 5 4 2 

3.12���տ�ʼ���ģ��ܶ���Ϊ����¹���ǡ���Ч��ƫ�ͣ�����ֻ������ƪ�루Yolo v3̫���ˣ����������������Եġ�

��һ��������������ƪ������ĵط����ڣ����������綼�����ε����ܶ���ƪ�����綼�ǿտε���ֻ����һƪ�����ϸ�������ƪ����Ч�ʵĽǶ���˵Ҳ������Ϊ����Ƚ�������������Ч�ʸߣ���Ц

���Ļ��ǱȽϸεģ�����ֻ����һƪdeeplab����Ϊֱ�Ӷ���v2���ò�̫˳�������ϻ���������Сʱ����RefineNet��������д���ظ��͹�������ĵط����٣������һ����˯���������˲ŷ��ִ��µ㲻��֮�����ViT��ͼ��ݱչݺ���ȥ��ѧ¥����д���ܽ�Ż����ᡣ�ص�������ȴ���������DERT���ˣ�д������һ��롣

�������忪ʼ����һ��С˵����Ȼ����д���賿һ��룬����С˵���������˯����������������Ȼû�ε�רע�����ޣ�ֻ����һƪSwin���������������������ǰ����SETR�����ٳ���ʱ���Ѿ���������ˡ�

�ܵ���˵���ź�����ԭ���ƻ�������һƪTransUNet���������ˣ�����ʵ��̫���ˣ������ڿ�txgs��ԭ�������ܶ�����ƪ�ģ�ԭ��Ŀ��������RefineDet�������Զ�����������ʦ�����һƪ����paper������Ϊʱ���û���������о���ɵû�����
