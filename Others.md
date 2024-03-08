- [Portrait segmentation](#portrait-segmentation)
  - [1.PortraitNet: Real-time Portrait Segmentation Network for Mobile Device(2019)](#1portraitnet-real-time-portrait-segmentation-network-for-mobile-device2019)
  - [2.Boundary-sensitive network for portrait segmentation(2019)](#2boundary-sensitive-network-for-portrait-segmentation2019)
  - [3.Sinet: Extreme lightweight portrait segmentation networks with spatial squeeze module and information blocking decoder(2020)](#3sinet-extreme-lightweight-portrait-segmentation-networks-with-spatial-squeeze-module-and-information-blocking-decoder2020)
  - [4.PP-HumanSeg: Connectivity-Aware Portrait Segmentation with a Large-Scale Teleconferencing Video Dataset(0222)](#4pp-humanseg-connectivity-aware-portrait-segmentation-with-a-large-scale-teleconferencing-video-dataset0222)
- [3D Object Detection](#3d-object-detection)
  - [1.VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection(2018)](#1voxelnet-end-to-end-learning-for-point-cloud-based-3d-object-detection2018)
  - [2.Frustum PointNets for 3D Object Detection from RGB-D Data(2018)](#2frustum-pointnets-for-3d-object-detection-from-rgb-d-data2018)
  - [3.SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation(2020)](#3smoke-single-stage-monocular-3d-object-detection-via-keypoint-estimation2020)
  - [4.Centernet:Objects as Points(2019)](#4centernetobjects-as-points2019)
  - [5. RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving(2020)](#5-rtm3d-real-time-monocular-3d-detection-from-object-keypoints-for-autonomous-driving2020)
- [RT-Series](#rt-series)
  - [1.RT-1: ROBOTICS TRANSFORMER FOR REAL-WORLD CONTROL AT SCALE(2022.12)](#1rt-1-robotics-transformer-for-real-world-control-at-scale202212)
  - [2.RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control(2023.8)](#2rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control20238)
  - [3.Open X-Embodiment: Robotic Learning Datasets and RT-X Models(2023.10)](#3open-x-embodiment-robotic-learning-datasets-and-rt-x-models202310)
- [others](#others)
  - [1.Deformable Convolutional Networks(2017)](#1deformable-convolutional-networks2017)
  - [2.Deformable ConvNets v2: More Deformable, Better Results(2019)](#2deformable-convnets-v2-more-deformable-better-results2019)
  - [3.AVP-SLAM: Semantic Visual Mapping and Localization for Autonomous Vehicles in the Parking Lot(2020)](#3avp-slam-semantic-visual-mapping-and-localization-for-autonomous-vehicles-in-the-parking-lot2020)

# Portrait segmentation
## 1.PortraitNet: Real-time Portrait Segmentation Network for Mobile Device(2019)
[��������](http://www.yongliangyang.net/docs/mobilePotrait_c&g19.pdf)

����ָ���Ϊ����ָ��һ�������������Ŷ��е���ս��1. ������ͼƬ��ռ������ܴ�2.ģ���ı߽������͸��ӵĹ�����������������˻���mobilenet v2��encoder-decoderģ��PortraitNet��ʵ��ʵʱ������ľ��Ⱥ�Ч��ƽ�⣬���л�ʹ����skip connection,depthwise conv�͸���loss

![Alt text](Paper/Portrait_Segmentation/image/1.png)

- ��һ������lossΪ�߽�loss����decoder�����һ������ͼ�󣬼���һ��Ԥ��߽�ľ���㣬ʹ��focal lossԤ��߽磨��Ϊ�߽�ռ������ռ�Ⱥ��٣����Ӷ�ʹ�ָ�Ա߽������
- һ��Լ��loss����ԭͼƬA�;���������ǿ���ı����ȡ��Աȶȡ���ȣ�������������ȣ���A'���������磬��Ԥ�⡣��ʱ��Ϊ��AΪ����ϸ�ķָ�Ӷ�ʹ��KLɢ��lossԼ��A'��A��£���������ǿ����Ը��ӹ��ջ�����³����
![Alt text](Paper/Portrait_Segmentation/image/2.png)

- ʹ��FLOPs�����������Ƚ��ٶ�

## 2.Boundary-sensitive network for portrait segmentation(2019)
[��������](https://arxiv.org/pdf/1712.08675)

������Resnet+deeplab v2�ķָ��ܣ�����Ϊ������ָ�����˶Ա߽����е�ģ��

![Alt text](Paper/Portrait_Segmentation/image/5.png)

- ��ԭͼ��GTת��Ϊsoft label��ǰ���ͱ�������one hot���߽���Ϊ��Ԫ�����������ָ�Ԥ�����������࣬�Ӷ�����������Ϊ��Ȩ�����أ��߽��и����Ȩ�أ�Ҳ�ṩ�˸���߽���Ϣ
- ��ѵ����������GTͼȡƽ��mask����һ�����ص�ľ�ֵ����0/1��������ص�����Ϊ����/ǰ����������0.5���������ڱ߽硣global boundary-sensitive kernel�㰴mask��ֵ���߽���ʸߵ����ص�ȡ���ߵ�ֵ����Ȩÿ�����ص��loss���Ӷ�ʹģ�Ͷ����ѵ����ص㣨�߽磩������
- ����ѵ����һ���߽����Եķ�����������/�̷���


## 3.Sinet: Extreme lightweight portrait segmentation networks with spatial squeeze module and information blocking decoder(2020)
[��������](http://openaccess.thecvf.com/content_WACV_2020/papers/Park_SINet_Extreme_Lightweight_Portrait_Segmentation_Networks_with_Spatial_Squeeze_Module_WACV_2020_paper.pdf)

��Portraitnet��ȣ����������Sinet����С�˲������������½����٣���Ҫ����������ģ�� Information Blocking Decoder�� Spatial Squeeze module
![Alt text](Paper/Portrait_Segmentation/image/3.png)
- Information Blocking Decoder:�ڻ�ȡϸ����Ϣʱ����ͷֱ�������ͼ�ںϸ߷ֱ�������ͼʱ��������Ϣ̫�ḻ����������/��������ģ������Ŀ�����õͷֱ��ʵ�����ͼ�ڸ߷ֱ�������ͼ�и���ע��Ҫ�ľֲ���Ϣ���ڵͷֱ�������ͼ����Ԥ��һ�����Ŷ�ͼ���߽��������Ŷȵͣ�c����1-c����Ȩ�߷ֱ�������ͼ���ٽ�����ͷֱ�������ͼ�ں�
- Spatial Squeeze module:��ģ������Ŀ���ǻ�ò�ͬ�߶ȵ�ȫ����Ϣ��S2 blockͨ���ػ���ȡ��������Ϣ��S2 module����bottleneck����ά��(����������С������)���ֲ������������ֱ��ʲ�ͬ��S2 block
- ����loss:��������߽����⡣��GT������̬ѧ���ͺ͸�ʴ��������ǿ�������ñ߽��GT���ñ߽�GT��Ԥ��ͼ��Ӧλ����loss����ǿ�Ա߽��������
  


## 4.PP-HumanSeg: Connectivity-Aware Portrait Segmentation with a Large-Scale Teleconferencing Video Dataset(0222)
[��������](https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Chu_PP-HumanSeg_Connectivity-Aware_Portrait_Segmentation_With_a_Large-Scale_Teleconferencing_Video_Dataset_WACVW_2022_paper.pdf)
���������һ����������������ָ�ģ��ConnectNet,�ü��ٵĲ���(0.13M)ʵ���˺�ǿ��Ч�����ؼ��������һ���µ�loss��ʹģ������ѧϰ��ͨ�ԡ�
- ����ܼ򵥣���ȿɷ�������ֻ����һ��skip���ϲ������bottleneck
  ![Alt text](Paper/Portrait_Segmentation/image/6.png)
- SCL: Self-supervised Connectivity-aware Learning for Portrait Segmentation:
![Alt text](Paper/Portrait_Segmentation/image/7.png)
- �Ƚ�pre��GT����ͨ�����ƥ�䣬�ټ�����ͨ��SC��loss=1-SC���ɹ�ʽ���Կ��������loss��ʹģ�������ڲ������ݸ��ٵ���ͨ���������ʹ��ͨ������Ľ����ȸ��󣬺����
- �и����⣬ѵ���տ�ʼʱ������û�н�������ʱlossΪ0�����������¡�û�н��������ʹ����һ�ֱ��loss��$loss=\frac{|P+G|}{|I|}$������ֵָ��������I������ͼƬ��ʹP��G�Ĳ�����С���������ںϡ�
# 3D Object Detection
## 1.VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection(2018)
������άĿ����ľ������ģ�����������������룬�����˹�ȥ��Ҫ�˹������������̵ı׶ˣ����������������С�˿�����
- ���ռ��еĵ���飬������ά�Ȼ��ֳ����أ�С���ӣ�������VFE(Voxel Feature encoding)�㣬���Եõ�ÿ�����ص�������ʾ��VFE�������ȡ��������ʹ�����ػ��ۺ�����������õ��ֲ��ۺ����������ֲ��ۺ����������������������������FCN�õ����ص�������ʾ��
- �����ص�������ʾ����������$C*D*H*W$�������м����㣬������RPN��ê��Ĳ���Ƶ��Ϊ����H��W��ÿ����������ȡ������������rpnԤ���ÿ���㣨����ͼ��СΪ$H/2*W/2$��������͸���ĵ÷֣���ÿ���������ê���7����������ê���ƫ�ƣ��������ĵ����꣬�����ߴ磬һ������ǣ�
  
## 2.Frustum PointNets for 3D Object Detection from RGB-D Data(2018)
Ҳ��һƪ3άĿ����ľ������ġ�����Χ��3D���ƣ����2DĿ���⣬�ﵽ�˺ܺõ�Ч��������������RGBֵ����������������ǵ���Ŀ�պ÷��ϣ�֮����Լ�������һ����ƪ���µĺ�����չ��
- �Ƚ�RGBͼ����2DĿ�������磬�õ����������������������ݺ�����Ĳ����������ڵĵ�ӳ��Ϊһ����׶���ڵĵ��ƣ�����3D�ָ�����(pointnet)������׶������ڵĵ���зָ���Ŀ�����ĵ㣻���ָ���ĵ�����3d�����ģ�飬���У�T-NetԤ��Ŀ�����ľ���������ĵĲв��һ����������磬����ΪT-netĿ����������ϵ�µĵ��ƣ�Ԥ����ʵ������T-netĿ�����ĵĲв��NS��Ԥ���ߴ��3��ά�ȵĲв�ֵ��NS���ߴ�ĵ÷֣�NH����Ԥ��ĺ���ǵĲв�͵÷֣���3+4*NS+2*NH����������տ��Եõ�3d����������꣬�ߴ�ͺ���ǡ�
## 3.SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation(2020)
��һƪ���׶ε�Ŀ3DĿ����Ĺ�����ʡ����Ԥ��2D��Ĳ��裬��֪���Ϲ�ҵ�������Գ��á�
- ���������õ�DLA���о������������֮���������֧���ؼ����֧Ԥ��Ŀ�����͹ؼ�������(��ÿ�����֣����巽ʽ��Centernet)���ؼ���Ϊ3D����ͶӰ��2D��ĵ㣬�ع��֧Ԥ��3D��Ϣ���ع��֧�У���ÿ���ؼ��㣬�ع�Ԥ��һ��ֵ�������ؼ��������ƫ��ֵ�������ߴ��ƫ��ֵ���Ǹ�ָ�������ģ����۲�ǵ�sin��cos����������yaw��ǣ���
- �ؼ����֧ʹ�õ�focal loss���ع��֧��Ԥ���3d��Ϣ����ʹ�ü����Լ��ת��һ�£���תΪ3d���8���ǵ�����꣬������GT�ǵ������L1��ʧ������ʹ��ͳһ����ʧ���������ǽ��лع顣
- ѵ��ʱ����������֧�ǲ��н��еģ���Ԥ��ʱӦ�����Ⱥ���еġ�
- ѵ��ʱ���ݶȽ����ˣ����Ԥ��3D��Ϣ�������ģ����������ƫ��ֵ��ʹ��GTͶӰ���ĵ������xy�����Ԥ�������ƫ��ع�GT����ֵ�����ڽǶȣ����˽Ƕ�֮���õĶ���GTֵ����Ҫ�����꣩�����ڳ߶�Ҳ����ˡ�

## 4.Centernet:Objects as Points(2019)
����ͨ��Ԥ��ؼ���ķ������м�⣬ʡȥ��NMS�������ұ����˶Դ�������ê�����ѵ����Ԥ�⣬�÷���Ҳ��֮��̳з�չ����SMOKE
- ��3D�߽������ģ�ʵ������ͶӰ2D������ģ�Ϊ��Ŀ��Ĺؼ��㡣���������ͼƬH\*W�����H/4\*W/4\*C����ͼ�����ն�ÿ����Ԥ��H/4\*W/4\*(C+3+1+8)��ֵ����ͬģ̬��Ԥ��ʹ�ö����ķָ�ͷ��4���������У�CάΪC�����ĵ÷֣������õ�������ͼ��peaks����Ϊ��ѡ�ؼ��㣬3άΪ3d��ĳߴ磬1άΪ��ȣ�8άΪ�ԽǶȵ�Ԥ�⣨����������һ�ֽ�����ķ��� Multi-Bin based method����2 Pi�ĽǶȷ�Χƽ�ֳ�����bins������ÿ��bins��Ԥ������bins�ĵ÷֣��õ�ƫ��ǣ��뵱ǰbin���ĽǶȵĲ�ֵ����sin��cos����
- ������㶮����ô������ͼ����ʧ��SMOKE�����һ������ͼ����ֵ��ֻ�ǹؼ����GTΪ1�������Թؼ���Ϊ���ĵĸ�˹�ֲ�������ÿ���ǹؼ��㣬ѡȡ���зֲ��е����ֵ��Ϊ��ֵ��
  
## 5. RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving(2020)

���������һ�֣�ֱ����2Dͼ����ȡ3D bbox�ؼ��㣨2D�߽�����ģ��Ͷ��㣨�����ͶӰ�����ĵ�ͶӰ������ͨ������Լ�����淶��3D��ͶӰ�ĵ�Ŀ3DĿ���ⷽ����

- �������磨�����������磩��Centernet���ƣ�Ҳ�õ�DLA����Ԥ�����һ��Cά����ͼ���ҵ��ؼ�������ͬʱ��Ԥ���9ά�Ķ�����ͼ��Ԥ��ÿ�����Ƕ���ĸ��ʣ�Ԥ���18ά����ͼ��Ϊÿ�������2άoffset�������ع鶥�����ꡣ
- Ԥ���9������󣬹�����һ������������������ͶӰ���ҵ�ͶӰ��2Dͼ����ʱ����Ԥ��������3d�򣩣�������Ƕ�������һ���������Ż���
# RT-Series
## 1.RT-1: ROBOTICS TRANSFORMER FOR REAL-WORLD CONTROL AT SCALE(2022.12)
- ����
"an we train a single, capable, large multi-task backbone model on data
consisting of a wide variety of robotic tasks? And does such a model enjoy the benefits observed in
other domains, exhibiting zero-shot generalization to new tasks, environments, and objects?"
NLP��CV��������ʼ�ڴ����������ݼ���ѵ�����/ͨ�õ�/�����޹ص�ģ�ͣ�ϣ��ģ����������������վ��顢ѧϰ��ͨ�÷�ʽ���Ӷ������������������и��õı��֡�ͬʱ��������Ԥѵ�����������Դ��������ض����ݼ���������ת��Ϊ�����������޹����ݼ�Ԥѵ��+���������ض����ݼ���΢�����ķ�ʽ
> CV�е�ImageNetԤѵ�����������������Ŀ����/�ָ������ʹ��ͼƬ��������ľ���ģ�ͣ���ResNet,ViT,MAE����ImageNet��Ԥѵ����Ϊbackbone������������ȡ������Ϊ��ЩԤѵ��ģ����ǿ��ķ���������NLP��BERT+΢����ʽ��ǰ����ʮ�����ţ�GPT��chatgpt֮ǰҲ����ΪԤѵ��ģ�ͱ�����ģ�ֻ����BERT���Ա��룬GPT���Իع飬��ǰ���ڵ�ʱЧ����ǿ��

����ϣ�������ɸ��ֻ�����������ɵ������ϣ�ѵ��һ����һ�ġ��������ġ����͵Ķ�����backboneģ�ͣ�̽��������ģ���Ƿ���������������۲쵽�ĺô����Ƿ��ܶ������񡢻����Ͷ�����ֳ�������������
>���robotic�����ȥ�Ķ�����ģ�ͣ����Ľ��ص���ӷ��ڡ������ԡ���ϣ�����һ��ͨ�õ�Ԥѵ��ģ�ͣ��Ӷ�����robotic����Դ��͵������ض����ݼ��������������Է�������������

- ��ս��The two main challenges lie in assembling the right dataset and designing the right model.

��һ����ս���ռ���ά���������ݼ���robotic���������ʱ��������ݼ���robot-specific��gather manually�ġ�����ϣ�������ݼ����㹻�������͹�ȣ�����һϵ�е���������ã����ڲ�ͬ�������г�ֵ����ӣ���ģ�Ϳ����ڽṹ���Ƶ�������̽����ʽ���Ӷ����õķ������������С�
We utilize a dataset that we gathered over the course of 17 months with a fleet of 13 robots, containing
?130k episodes and over 700 tasks, and we ablate various aspects of this dataset in our evaluation.
�ڶ�����ս�����ģ�͡�������--transformer����������ǿ���ر���ѧϰ����������ָ�������ʱ��Ȼ���������˿��ƴ���ʵʱ����Ҫ�󣬱�����������һ���µ�transformerģ��RT-1������ά������������ͼƬ��ָ�������ƣ�ѹ��Ϊtoken representations������transformer���Ӷ���֤�����ʵʱ�ԡ�RT-1������������������ʵʱ��������ļ���Ч�ʽ��������
- ���ף����RT-1������˷����Ժ�³���ԣ��������˳�ֵ�ʵ�飨���ۺ����������ģ�ͺ�ѵ������ɵ���ƣ�
- ��ع���
1. ��������˿�����������˴�������transformer�Ĺ���
2. ��ʵ��������˲�������ع��������task-specific��
3. �����������ȥ�Ķ�����ѧϰ�ͻ�����������ѧϰ�Ĺ�����
- PRELIMINARIES
1. �˽���һЩrobot learning������ѧ��AIԭ��γ��е�����agent��Щ��ϵ��ѧϰһ�����Ժ���$\pi$��������״̬ת�ƺ���������������ָ��i�͵�ǰ״̬�����һ����Ϊ�ĸ��ʷֲ��������õ������˵���һ���ж�������һ�����о��߻�����������εõ�robot����һ����Ϊ��ֱ����ֹ������Ϊһ��episode������һ����Ԫreward����robot�Ƿ���ȷִ��i��Ŀ�����������reward��
2. transformers
3. imitation learning:���ݼ�Ϊ���episode��ÿ��eposide��������ָ��i��һ��״̬+��Ϊ�ģ���ȷ�����У�ϣ��ѧϰһ������$\pi$�����������Ϊ���ƣ���ÿ��Ԥ����ж����㸺log��ʧ�����Ż���
- ����ϵͳ����a 7 degree-of-freedom arm, a two-fingered gripper, and a mobile base 
- RT-1�ṹ
RT-1��һ������ͼ��6�ţ�����Ȼ����ָ����Ϊ���룬����ÿ��ʱ�䲽��������˵Ķ�����7+3+1����11ά������ṹΪԤѵ����Efficientnet����ÿ��MBConv�������FiLMģ�飬����Ȼ����ָ������Ӧ��ָ��ͼƬ��Ϣ����ȡ����Ȼ����ָ��ͨ��universal sentence encoder����Ϊ512ά��Ƕ�루�ٷ�������δ��������ÿ��FilMģ��ά��������ƫ�õ�ȫ���Ӳ㣬��512άǶ��ӳ��Ϊ��ǰ����ͼ��ά�ȣ��õ�FiLM�ķ���任���ӣ���ͨ�����㼴�ɡ�
>Efficientnet��MBConv(mobile invert residual bottleneck Conv)Ϊ���壬ʹ��NAS�����õ���FiLM�ں��Ӿ�����Ȼ������Ϣ������Ȼ������Ϣͨ��һ��generator�õ��������������Ӿ���������ͨ������任�����Ա任����������SE

- ����6������ͼƬ������Efficientnet+FiLM���������磬�õ�6\*9\*9\*512��token������tokenlearnerģ���һ��ѹ��Ϊ6\*8\*512��token��Ϊtransformer�����롣

- ʵ��
1. ǿ��ı���/ͨ����/³���ԺͶ���ʵ����ķ�����
2. RT-1ģ�Ϳ��ԺܺõĴ��칹������ѧϰ\absorb��������ģ�⻷�������ݣ����Բ�ͬ�����˵�����.(�����µ�������ݼ���������ʧԭ���֣�����������)
3. ������long-horizon����
4. ̽�������ݼ���size��diversity��Ӱ��
- ������
1. ��Ȼ��ģ��ѧϰ�ķ������޷�����demonstrators�ı��֣�ѵ�����ݣ�
2. ����ָ��ĸ�����������ǰ�����ĸ������ϣ�����RT-1�������ƹ㵽��ǰ��δ������ȫ���˶�
3. ��������һ����ϵͳ

## 2.RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control(2023.8)
- �������Ӿ���NLP����������web-scale datasets��ѵ����LLM����Robotics��Ҫѵ��һ�������˴�ģ�ͼ��������ܣ��ò���������������ݡ���ˣ����������ֱ��ʹ��VLMs(Vision-Language)������������˶���һ��΢�����õ�VLA(vision-language-action model)��Robotic�Ĵ�ģ�͡�
- ����ѡ���VLMΪgoogle��PaLI-X �� PaLM-E������image��text���ֱ�����ͼƬ��ViT���룬����textһ������LLM�������text token����token����Եõ�text�����Ϊ����VLM���ƻ����ˣ���VLA��������˵�action�����ĺ�RT-1���ƣ���action�����token��������˵��action�İ˸�ά��(����terminate���)��ÿ��ά����ɢ��Ϊ256��ʵ����ÿ��ά����һ��language token�滻����ʵ��VLM��VLA��ת����
> PaLI-X��1000���ڵ�ÿ�����ֶ���һ����Ӧ��token�����ֻ�轫256����ɢ������256����������;PaLM-E�������ٳ��ֵ�256��tokens���ǵ����ֱ��Ӧ256����ɢ����
- ѵ��ʱ��VLA����΢����ʹ��web-scale data�ͻ���������һ����fine-tuning��������Ӧ�����˶������ֿ��԰���VLM��ϰ��Ϊ�˱�֤ʵʱ����ģ�ͱ�������TPU��Ⱥ�ϣ���֤��hz�������ٶ�
- ʵ�飺
1. RT-2��seen��������RT-1�����൱������unseen�����ϵı���Զ��RT-1���������ڷ���������
2. ���ֳ���ӿ������������symbol understanding/term reasoning(����)/human recognition(����Ϊ���ĵ�ʶ��)��
3. VLMԤѵ��Զ���ڴ��㿪ʼ��Эͬѵ������ֻʹ�û��������ݣ�ģ�Ͳ��������ʹЧ�����
4. ����˼ά�������RT-2�ܹ��ش�����ӵ����������˵����VLA��Ԥ��ʱ����ֹ���action��token�������"plan"����action����Ȼ����������
- �����ԣ������˲�û����ΪVLM��֪ʶ����κ�ִ�����˶���������ģ�͵����弼����Ȼ�����ڻ����������еļ��ֲܷ���ֻ��ѧ�������µķ�ʽ������Щ���ܣ�ģ��������������٣������������VLA��ͨ��VLM


## 3.Open X-Embodiment: Robotic Learning Datasets and RT-X Models(2023.10)
���ĵ�Ŀ����ת����ͨ�õ�Ԥѵ��ģ�ͣ�ϣ����CV,NLPһ���������ڴ��͵�/�����Ե�ͨ�����ݼ���ѵ����ģ�ͣ���Ϊ���������һ��start point.��RT-1��ͬ��ϣ�����һ�����ݶ��embodiment��ͨ�û�����ģ�ͣ�����һ��������ÿ������ÿ�������˺ͻ���ѵ��һ��ģ�ͣ�
- ������Ϊѵ���ɷ��������˲��Ե�Ŀ����Ҫ**X-embodiment(����ʵ��) training**����ʹ�����Զ��������ƽ̨�����ݣ��Ӷ����Ը��õظ��ǻ����ͻ����˵ı仯������֤���ˣ�ʹ��X-embodiment data��������Ч����ʵ����Ǩ�ơ�(��ȥ��robot embodiments�Ĺ�����������Ǩ��ѧϰ���������᲻ͬ��֮��Ĳ�࣬��ʵ������Ǩ�ƣ�������ֱ��ʹ��X-embodiment dataѵ��)�����ҿ�Դ��Open X-Embodiment Repository������һ���������ݼ���RT-Xģ�͵�checkpoint�����������΢��
- The Open X-Embodiment Dataset:����������ȫ����������ƽ̨������ 22 �ֲ�ͬ�������������ݣ�������ͳһ�ı�׼���Ͷ��루��ͬ�����˵Ĺ۲�Ͷ����ռ����ܴ�ϸ���������н��������� use the RLDS data format��Ӧ�˲�ͬ���������õĸ��ֶ����ռ������ģʽ���粻ͬ������RGB������������͵��ơ�
- RT-X����RT-1��RT-2���ø����ݼ�ѵ��
- ʵ��:
1. ��ͬembodiment֮��ķֲ��ڱ���:��RT-1-X,RT-2-X��Open X-Embodiment Dataset��ѵ������ԭʼģ�ͣ����ݼ�����ߵģ���RT-1��ֻ�ڶ�Ӧ���ݼ���ѵ�����Աȣ��ֱ���С���ݼ��ʹ����ݼ��Ͻ�����ʵ�顣���������RT-X-1��С���ݼ������������������ߣ����ڴ����ݼ��ϲ���RT-1��RT-X-2�ڴ����ݼ�������RT-1��RT-X���ԴӲ�ͬ���������ʵ����Ǩ�ƣ������ݷḻ������Ҫ������ģ�Ͳ���������ܡ�
2. ����������RT-2��RT-2-X��������unseen�����ϴ����൱��RT-2-X��ʾ���˸�ǿ��ӿ��������ͨ����Open X-Embodiment Dataset��ѵ����A embodimentѧϰ����ԭA���ݼ��в����ڵļ��ܣ�����B mbodiment����
- δ��������RT-Xû�п��Ǿ��зǳ���ͬ�Ĵ��к�����ģʽ�Ļ����ˣ�û���о����»����˵ķ�����Ҳû���ṩ��ʱ������Ǩ�ƻ򲻷�����Ǩ�Ƶľ��߱�׼
# others
## 1.Deformable Convolutional Networks(2017)

[��������](https://arxiv.org/pdf/1703.06211)

����˿ɱ��ξ�����ؼ����Ա�׼k\*k�����ȡ����ͼ���õ�ÿ�����ص�2\*k\*k��offset�����Ա�׼�����������е�����ʹ��˫���Բ�ֵ��ȡ�������Ӷ�����Ӧ�Ե���ȡ�����������Ŀ������������� Deformable RoI Pooling��Position-Sensitive (PPaper/Portrait_Segmentation/image/7.png)

## 2.Deformable ConvNets v2: More Deformable, Better Results(2019)

[��������](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.pdf)

��v1�Ļ���������� Modulated Deformable Modules��modulatedָ���Ƕ�ÿ�����ص㣬����Ԥ������oddset����Ԥ��һ�� modulation scalar��ֵ��[0,1]��������ʾ�����ص����Ҫ�̶ȡ�

## 3.AVP-SLAM: Semantic Visual Mapping and Localization for Autonomous Vehicles in the Parking Lot(2020)
ʹ��SLAM���в�����һƪ���£��������̿����ˣ����о��ȵĹؼ��Ƕ�BEVͼ��ƴ�Ӻ�����ָ���Ҷ�SLAM���Ӿ���̼ƺ;ֲ���λ�Ȼ���һ����֪������Ҫ����ѧϰ��
