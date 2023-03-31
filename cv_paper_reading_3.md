- [Backbone](#backbone)
  - [1.HRNet:Deep High-Resolution Representation Learning for Human Pose Estimation(2019)](#1hrnetdeep-high-resolution-representation-learning-for-human-pose-estimation2019)
  - [2.Resnest: Split-attention networks(2022)](#2resnest-split-attention-networks2022)
  - [3.Mobilenet v2: Inverted residuals and linear bottlenecks(2018)](#3mobilenet-v2-inverted-residuals-and-linear-bottlenecks2018)
  - [4.mobilenet v3:Searching for mobilenetv3(2019)](#4mobilenet-v3searching-for-mobilenetv32019)
  - [5.Beit: Bert pre-training of image transformers(2021)](#5beit-bert-pre-training-of-image-transformers2021)
  - [6.ConNext:A convnet for the 2020s(2022)](#6connexta-convnet-for-the-2020s2022)
  - [7.MAE:Masked autoencoders are scalable vision learners(2022)](#7maemasked-autoencoders-are-scalable-vision-learners2022)
  - [8.Segnext: Rethinking convolutional attention design for semantic segmentation(2022)](#8segnext-rethinking-convolutional-attention-design-for-semantic-segmentation2022)
- [Semantic Segmentation](#semantic-segmentation)
  - [1.Deeplab v3:Rethinking atrous convolution for semantic image segmentation(2017)](#1deeplab-v3rethinking-atrous-convolution-for-semantic-image-segmentation2017)
  - [2.Bisenet: Bilateral segmentation network for real-time semantic segmentation(2018)](#2bisenet-bilateral-segmentation-network-for-real-time-semantic-segmentation2018)
  - [3.Psanet: Point-wise spatial attention network for scene parsing(2018)](#3psanet-point-wise-spatial-attention-network-for-scene-parsing2018)
  - [4.Deeplab v3+:Encoder-decoder with atrous separable convolution for semantic image segmentation(2018)](#4deeplab-v3encoder-decoder-with-atrous-separable-convolution-for-semantic-image-segmentation2018)
  - [5.Icnet:Icnet for real-time semantic segmentation on high-resolution images(2018)](#5icneticnet-for-real-time-semantic-segmentation-on-high-resolution-images2018)
  - [6.Non-local neural networks(2018)](#6non-local-neural-networks2018)
  - [7.EncNet:Context encoding for semantic segmentation(2018)](#7encnetcontext-encoding-for-semantic-segmentation2018)
  - [8.DANet:Dual attention network for scene segmentation(2019)](#8danetdual-attention-network-for-scene-segmentation2019)
  - [9.CCNet: Criss-Cross Attention for Semantic Segmentation(2019)](#9ccnet-criss-cross-attention-for-semantic-segmentation2019)
  - [10.ANN:Asymmetric non-local neural networks for semantic segmentation(2019)](#10annasymmetric-non-local-neural-networks-for-semantic-segmentation2019)
  - [13.Gcnet: Non-local networks meet squeeze-excitation networks and beyond(2019)](#13gcnet-non-local-networks-meet-squeeze-excitation-networks-and-beyond2019)
  - [14.OCRNet:Object-contextual representations for semantic segmentation(2020)](#14ocrnetobject-contextual-representations-for-semantic-segmentation2020)
  - [15.Pointrend: Image segmentation as rendering(2020)](#15pointrend-image-segmentation-as-rendering2020)
  - [16.Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation(2021)](#16bisenet-v2-bilateral-network-with-guided-aggregation-for-real-time-semantic-segmentation2021)
  - [17.DPT:Vision Transformer for Dense Prediction(2021)](#17dptvision-transformer-for-dense-prediction2021)
  - [18.Segmenter: Transformer for semantic segmentation(2021)](#18segmenter-transformer-for-semantic-segmentation2021)
  - [19.SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers(2021)](#19segformer-simple-and-efficient-design-for-semantic-segmentation-with-transformers2021)

# Backbone
## 1.HRNet:Deep High-Resolution Representation Learning for Human Pose Estimation(2019)
[��������](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.pdf)

����̬���������������һ����ȡ��߶���������������HRNet�������������б����˸߷ֱ����������������ܼ�Ԥ�⣨λ�����У�����������ɱ��϶����͡���������ֻ������߷ֱ��ʵ�����ͼ����Ԥ�⣬��Ȼ���Խ�϶�߶�һ��Ԥ�⡣

![Alt text](backbone/image/1.png)

- ��ͼ��ʾ�������ںϲ�ͬ�ֱ��������Ĳ��֣���ͷ����Ϊexchange unit���þ�����²��������ڽ��ϲ�����1\*1�����ͳһͨ����������ӵõ���Ӧ�߶�����ͼ��
- ͼ���У���һ�β������ͷֱ�������ͼʱ�����������г߶�����ͼ�������������ֻ�������ڽ��߶ȵ�����ͼ


## 2.Resnest: Split-attention networks(2022)
[��������](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Zhang_ResNeSt_Split-Attention_Networks_CVPRW_2022_paper.pdf)

�����SENet�е�channel-wiseע������ResNext�е�group convolution����SKNet�е�split-attention�������һ�ָ�ǿ��Resnest��û��ʲô�µķ���

![Alt text](backbone/image/4.png)



## 3.Mobilenet v2: Inverted residuals and linear bottlenecks(2018)
[��������](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)

��mobile v1����ȿɷ����������ϣ�������Linear Bottleneck��Inverted residuals

![Alt text](backbone/image/2.png)

- Linear Bottleneck:��������Ϊ��������X��ÿ������ռ�BX�ķֲ��൱��һ����Ȥ���Σ����ο���ӳ�䵽��ά�ռ䣨�ֲ����ܼ�������ReLU����������Ե�ͬʱ���ƻ������е���Ϣ�����磬�������ռ�BXά�ȱȽϵͣ�$B^{-1}ReLU(BX)$�ָ���X�ƻ������أ���ά�򻹺á���������Ƹ�Ч����ʱ��ϣ�������ܵؽ���ά�ȣ��ֲ�ϣ��ReLU�ƻ�̫����Ϣ����˳����ڵ�ά�ȵľ��������ȥ�������Բ㣬�Ա���������Ϣ����Ϊһ��Linear bottleneck��
- Inverted residuals����ͼ��ʾ��v2�Ƚ�������ά��Ϊ����3\*3���ʱ��ȡ���ḻ�����������ٽ�����ȿɷ����������ͨ��Linear Bottleneck��ûReLU����ά����Ϊ����bloc��ͷ���м�񣬺�residual block�෴�����Ե���



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
![Alt text](backbone/image/3.png)

- ֻ��ûmasked��patch�������������˿��Թ����ϴ�ı���������ͳһ��mask token������������룬����������������Ӧpatch����������


## 8.Segnext: Rethinking convolutional attention design for semantic segmentation(2022)
[��������](https://arxiv.org/pdf/2209.08575.pdf?trk=public_post_comment-text)

���������һ��Ϊ�˷ָ�����ļ򵥾������Segnext������˼�����Ҫ��������ǿ���encoder����߶��������ռ�ע����
- encoder��transformer���ƣ����þ��ע�����������ע����������˵����ע��������Ч�����л�����depth-wise�Ĵ�״�������׽��״����������Ұ���䣬���ٲ���
![Alt text](backbone/image/5.png)
- decoderʹ���˶�߶�����ͼ
![Alt text](backbone/image/6.png)

# Semantic Segmentation
## 1.Deeplab v3:Rethinking atrous convolution for semantic image segmentation(2017)
[��������](https://arxiv.org/pdf/1706.05587.pdf%EF%BC%8C%E6%8E%A8%E8%8D%90%E5%9C%A8%E7%9C%8B%E5%AE%8C%E6%9C%AC%E6%96%87%E4%B9%8B%E5%90%8E%E4%BB%94%E7%BB%86%E9%98%85%E8%AF%BB%E8%AE%BA%E6%96%87%E4%BB%A5%E5%8F%8A%E4%BB%A3%E7%A0%81%EF%BC%8C%E5%8F%AF%E4%BB%A5%E6%9B%B4%E5%A5%BD%E7%90%86%E8%A7%A3%E3%80%82)
��deeplab v2�Ļ����Ͻ����˸Ľ�������˼����Ļ�����Ӧ���˿ն������ģ�飬������v2
- ��������Resnet�ĺ󼸸�block�ĳɿն����������������ֱ��ʲ��䣬ÿ��block֮�䡢���ڲ��ľ����֮��ն����������ϵ����������һ�����ֹgrid problem����һ�����������Ұ
- ������ASPP�����Ľ������㣬������BN���ն����������ϵ��̫��Ļ�����Ч�㣨padding��������������ӣ��ﲻ���������Ұ��Ŀ�ģ���˼�����Image-level������ȫ�ֳػ��㣩�����1\*1������ϲ�������ASPP���ƴ����һ��


## 2.Bisenet: Bilateral segmentation network for real-time semantic segmentation(2018)
[��������](https://openaccess.thecvf.com/content_ECCV_2018/papers/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.pdf)

���������һ��˫�߷ָ�ģ��Bisenet��ʵ��Ч����Ч�ʵľ���

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/1.png)

- ��Ҫ�ص��ǣ������������·����context path��spatial path��ǰ��ͨ�����ٵ��²���pretrained��������Xception���������Ұ����ýϵͷֱ��ʵĺ��ḻ��������������ͼ�����ARM(Attention refinement module)�����а���ȫ�ֳػ������߽�����������㣬�²���8������˾��ܳߴ�󵫼��������󣩣�������ԭͼ��ḻ�Ŀռ�������
- ��Ϊ����·������Ϣ��level��ͬ�������FFM����������ֵ�������

## 3.Psanet: Point-wise spatial attention network for scene parsing(2018)
[��������](http://openaccess.thecvf.com/content_ECCV_2018/papers/Hengshuang_Zhao_PSANet_Point-wise_Spatial_ECCV_2018_paper.pdf)

����point-wiseע�������������λ�õ�ͬʱ����ȫ����Ϣ��ÿ���㶼����Ӧ��ͨ��һ����ѧϰ��ע����ӳ�����������е�����

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/2.png)

- ������һ��˫����Ϣ����·�������ע�������������֣���һ����Ϊ������j��Ŀ���iԤ�����Ҫ�ԣ��ڶ�����ΪĿ���i��������j����Ҫ�ԣ��������ֶ�����ͼ��ÿ���㶼��H\*Wά��������������һ��2H-1\*2W-1ά������ͼ��ͨ���۽������Ĳ�ͬλ�ã����ÿ����H\*Wάע�������õ�attention map����ע����ͼ����ʽ�ɵ�ÿ�����������
- (��������ͼΪH\*W\/$C_2$) collect��H\*W\*(H\*W)ά��attention map��ÿ�����H\*Wά������ʾH\*Wÿ����Ըõ��ע������������Ӧ��Ȩ���ÿ�����$C_2$ά�������ɵøõ�����������distribution���ֵ�attetion map��ÿ�����H\*Wά������ʾ�õ��H\*W�������Ҫ�ԣ��������������ʱ��ȡȫ��ÿ�����H\*Wά�����еĵ�iά��Ϊ��ȫ�ֵ��Ŀ���i��ע������Ȩ���ۼӿɵ��������

## 4.Deeplab v3+:Encoder-decoder with atrous separable convolution for semantic image segmentation(2018)
[��������](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf)

Ϊ���ڱ�֤�ֱ��ʵ�ͬʱ����������Ϣ��deeplab v3ʹ�ÿն��������ػ����Ӷ���֤�ߴ��ͬʱ�����˸���Ұ���������ַ�������encoder-decoder�Ա߽���Ϣ��ϸ�ڡ���ˣ�deeplav v3+�����encoder-decoder�ṹ����v3��Ϊһ��ǿ���encoder��֮�����һ���򵥵�decoder����̽������ȿɷ���ն������Ӧ����ASPP��decoder��

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/3.png)

## 5.Icnet:Icnet for real-time semantic segmentation on high-resolution images(2018)
[��������](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hengshuang_Zhao_ICNet_for_Real-Time_ECCV_2018_paper.pdf)

�����һ��ʵʱ����ָ���ICNet�����ü���ͼƬ���룬�ںϲ�ͬ�ߴ������ͼ��ʵ��coarse-to-fineԤ�⣬�ڵͷֱ�������ͼʹ���������磬�ڸ߷ֱ��ʲ���ʹ�����������磬�Ӷ�������С��������

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/4.png)

- ��CFF(cascade feature fusion)ģ�飬ʹ��˫���Բ�ֵ�Ϳն����ʵ�ֲ�ͬ�ߴ�����ͼ���ں�

ʹ�ø�����ʧ��ÿ���ߴ������ͼ���ᱻ����Ԥ�Ⲣ������ʧ��������ʧ���Ȩ


## 6.Non-local neural networks(2018)
[��������](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)

�������һ�� non-local ��������һ��ͨ�õ�non-local block����self-attentionͳһ��non-local�ķ�ʽ�У��������һЩ�������ܵ�ѡ��

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/5.png)

## 7.EncNet:Context encoding for semantic segmentation(2018)
[��������](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Context_Encoding_for_CVPR_2018_paper.pdf)


����� Context Encoding Module,������������Ϣ��������SENet��������ͼ��ÿ��ͨ����Ȩ

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/6.png)

- ENCģ���е�encoder layer��ͨ����ͳ�����õ�K������ʣ�����softmax��Ȩ�õ�ÿ�����ض�ÿ������ʵĲв��������ۼӵ�����ͼ��ÿ������ʵĲв�����
- ��encoder layer�����inputȫ���Ӳ㣬�õ�ÿ��ͨ����Ȩ��
- �����˸�������SE-loss��GT���Դӷָ�GT�л�ã�ÿ�����Ķ�Ԫ������


## 8.DANet:Dual attention network for scene segmentation(2019)
[��������](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf)

Ϊ�˸��õĲ�׽��������Ϣ��ȫ����Ϣ����ͨ�������ϵ�����������һ��˫ע��������DANet��ʹ������ע����ģ�����õ����õ�������ʾ

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/7.png)

position attention module����������ͼH\*Wά�ȵ���ע�������õ�(H\*W)\*(H\*W)��ע�����������󣬼����Ȩֵ��channel attention module����������ͼͨ��ά�ȵ���ע�������õ�C\*C��ע�������������ټ����Ȩֵ����󽫶����ںϡ�

## 9.CCNet: Criss-Cross Attention for Semantic Segmentation(2019)
[��������](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_CCNet_Criss-Cross_Attention_for_Semantic_Segmentation_ICCV_2019_paper.pdf)

���������������Ϣ��ע����ģ��/non-localģ�飬����Ҫ���ɳߴ�ܴ��ע������������(H\*W)\*(H\*W)����������ռ���ڴ�󡣱������һ���µľۺ�ȫ��������ע����ģ��CCA��ÿ������ڵ�ǰ�к��м���ע������ע������������ΪH\*W\*(H+W-1)������С����ͨ��ѭ������CCA����ȡȫ����Ϣ

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/8.png)

## 10.ANN:Asymmetric non-local neural networks for semantic segmentation(2019)
[��������](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhu_Asymmetric_Non-Local_Neural_Networks_for_Semantic_Segmentation_ICCV_2019_paper.pdf)
��CCNetһ�����������non-local�ı׶ˣ������ANN��Ӧ����Ҳ����ע������APNB(Asymmetric Pyramid Non-local Block)��AFNB(symmetric Fusion Non-local Block)��ǰ����ȡȫ�������������ں϶�߶�����

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/9.png)

APNB��AFNB�������ڣ���ע����ģ���key��query��ά��ͨ��������C\*HW������C\*S������SԶС��HW��������ʽΪSPPģ�飬����Ҳ�ں��˶�߶�������

## 13.Gcnet: Non-local networks meet squeeze-excitation networks and beyond(2019)
[��������](http://openaccess.thecvf.com/content_ICCVW_2019/papers/NeurArch/Cao_GCNet_Non-Local_Networks_Meet_Squeeze-Excitation_Networks_and_Beyond_ICCVW_2019_paper.pdf)

��Ȼ��Ϊ�˼�non-local����������������Ƿ���non-local��׽��ע����ͼ������query����һ�����Ӷ������һ������ע����ͼH\*W�ļ�non-local�顣���н�SENet�ͼ򻯵�non-local��ͳһ��Global context modeling framework���������ߵ����ƽ�ϣ��ȼ���ȫ��ע������query����ע����ͼ��H\*W������C\*HW��ˣ��õ�Cά������ͨ������1\*1��bottleneck����C\*H\*W��ӣ��в����ӣ����ɹ㲥���ƣ���ʵ�൱��ÿ�����ؾ�ע������Ȩ�����������ȣ����������Cά����

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/10.png)

## 14.OCRNet:Object-contextual representations for semantic segmentation(2020)
[��������](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510171.pdf)

���ľ۽��ھۺ����������������ȥ��ASPP�������non-local������ͬ�����Ľ��������������������������ḻ������

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/11.png)

��backbone���������ͼΪH\*W\*C
- soft object region:K\*H\*W.Ϊÿ��ͼԤ��K��object region������ÿ��regionΪһ��2Dͼ��ÿ�����ص�ֵ��ʾ����������������ԡ��˹������мල�ģ���GT���ɵ���GT��������
- object region representation:��soft object region�ֱ��Ȩ����������ͣ����յõ�K��Cά������ÿ��������ʾ�ö������������
- Pixel-Region Rela:ͨ�����+softmax����ÿ�����ص���ÿ���������������ԣ��õ�H\*W\*K����ԭ��������������������Ԥ��
- ���������̵��������ʱ���и�transformation,1\*1+BN+ReLU

## 15.Pointrend: Image segmentation as rendering(2020)
[��������](http://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf)

���Ľ��ָ�����������Ⱦ�������Ե���ȵ����⣬��������һ���µ��ϲ�����ʽ������ѵ����

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/12.png)

����CNN��ȡ����ϸ��������ͼ���Ȳ���һ�������ָ�����õ�coarse predict(7\*7)��Ӧ��˫���Բ�ֵ�����ֱ��ʷ���������Щ��������N�����ѵ㡱�����Ŷȵͣ�����˵�߽紦��������һ��MLP����Щ������Ԥ�⣨������������ͼ�ʹ�Ԥ��ͼ����������Ԥ��ͼ����˫���Բ�ֵ������ֱ��Ԥ��ͼ�ķֱ��ʴ��ڵ���ԭͼ��

## 16.Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation(2021)
[��������](https://arxiv.org/pdf/2004.02147)

������Bisenet v1����ƣ��ֱ��������֧��ϸ�ڷ�֧��׽������������ϸ�������������v1�����ľ�������������֧�����������²����Ĳ��о��stem block��Ӧ������ȿɷ������ۺ�������Gather-and-Expansion Layer���ͷ������׽�߲��������Context Embedding Block��ȫ�ֳػ���1\*1�ٲв���ӣ������Ļ���������������ۺ�ģ�飬��������֧����������ͬlevel�ֱ�ۺϣ��õ����õı�����

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/13.png)

## 17.DPT:Vision Transformer for Dense Prediction(2021)
[��������](http://openaccess.thecvf.com/content/ICCV2021/papers/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.pdf)

��������ܼ�Ԥ���vision transformer DPT�������˸��������ԡ�

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/14.png)

- encoder���ֺ�ViTһ����������һ�����ڷ����readout token����token��read�����б�����/��Ϊȫ�������ںϡ�
- decoder���Բ�ͬtransformer��������װ��**��ͬ�ֱ���**��������ͼ�����ʽ������ͨ��bottleneck����/�²��������н�low��transformer��ᱻ��װ�ɸ���ֱ��ʵı�ʾ����Ϊ���а�������ϸ����������֮��ʹ����������refinenet�ķ�ʽ���ں϶�߶ȵ�����ͼ

## 18.Segmenter: Transformer for semantic segmentation(2021)
[��������](https://openaccess.thecvf.com/content/ICCV2021/papers/Strudel_Segmenter_Transformer_for_Semantic_Segmentation_ICCV_2021_paper.pdf)

���������һ���µ�mask transformer decoder���Ӷ����Segmenterģ��

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/16.png)

- encoder��ViTһ��
- ��encoder�����K��patch encoding�Ϳ�ѧϰ�������ʼ����K��classǶ��һ������mask transformer��ά�Ⱦ�ΪD���������ÿ��patchǶ�����Ƕ���������(N\*D)\*(D\*K)=N\*K���Ӷ��õ�ÿ��patch�������룬��reshape���ϲ����õ�Ԥ��ͼ


## 19.SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers(2021)
[��������](https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf)

���������һ�ּ򵥸�Ч�Ļ���transformer������ָ�ģ��Segmenter

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/15.png)

- ������Swin transformer����encoder��ͨ���ϲ���������ò�ͬ�ֱ��ʵ�����ͼ���������ڣ�Segmenter�ϲ������ص��Ĳ����������ֺϲ���ľֲ�������
- ͨ����encoder��FFN�м���3\*3��Ⱦ���������ṩ�㹻��λ����Ϣ���Ӷ�ʡ����λ�ñ���
- ʹ����һ����������ALL-MLP��decoder���Ƚ���ͬ�ֱ��ʵ�����ͼͳһά�Ⱥͳߴ磬��ͨ��MLPԤ�⡣�����������Ĺؼ��ǣ�encoder�ṩ�Ķ�߶�����ͼ����Ϣ�ܷḻ������Ұ����