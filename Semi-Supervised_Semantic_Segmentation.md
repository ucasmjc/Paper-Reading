- [Semi-supervised semantic segmentation](#semi-supervised-semantic-segmentation)
  - [1.Guided Collaborative Training for Pixel-wise Semi-Supervised Learning(2020 eccv)](#1guided-collaborative-training-for-pixel-wise-semi-supervised-learning2020-eccv)
  - [2. Mean teachers are better role models:Weight-averaged consistency targets improve semi-supervised deep learning results(2018 nips)](#2-mean-teachers-are-better-role-modelsweight-averaged-consistency-targets-improve-semi-supervised-deep-learning-results2018-nips)
  - [3.Semi-Supervised Semantic Image Segmentation with Self-correcting Networks(2020 cvpr)](#3semi-supervised-semantic-image-segmentation-with-self-correcting-networks2020-cvpr)
  - [4.Semi-Supervised Semantic Segmentation with Cross-Consistency Training(2020 cvpr)](#4semi-supervised-semantic-segmentation-with-cross-consistency-training2020-cvpr)
  - [5.Semi-supervised semantic segmentation needs strong, varied perturbations(2020 BMVC)](#5semi-supervised-semantic-segmentation-needs-strong-varied-perturbations2020-bmvc)
  - [6.ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning(2021 WACV)](#6classmix-segmentation-based-data-augmentation-for-semi-supervised-learning2021-wacv)
  - [7.Semi-Supervised Semantic Segmentation via Adaptive Equalization Learning (2021 NIPS)](#7semi-supervised-semantic-segmentation-via-adaptive-equalization-learning-2021-nips)
  - [8.Semi-supervised semantic segmentation with cross pseudo supervision(2021 cvpr)](#8semi-supervised-semantic-segmentation-with-cross-pseudo-supervision2021-cvpr)
  - [9.Semi-supervised Semantic Segmentation with Directional Context-aware Consistency(2021 cvpr)](#9semi-supervised-semantic-segmentation-with-directional-context-aware-consistency2021-cvpr)
  - [10.PseudoSeg: Designing Pseudo Labels for Semantic Segmentation(2021 iclr)](#10pseudoseg-designing-pseudo-labels-for-semantic-segmentation2021-iclr)
  - [11.ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation(2022 cvpr)](#11st-make-self-training-work-better-for-semi-supervised-semantic-segmentation2022-cvpr)
  - [12.Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels(2022 cvpr)](#12semi-supervised-semantic-segmentation-using-unreliable-pseudo-labels2022-cvpr)
  - [13.Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation(2022 cvpr)](#13perturbed-and-strict-mean-teachers-for-semi-supervised-semantic-segmentation2022-cvpr)
  - [14.Semi-supervised Semantic Segmentation with Error Localization Network(2022 cvpr)](#14semi-supervised-semantic-segmentation-with-error-localization-network2022-cvpr)
  - [15.Bootstrapping Semantic Segmentation with Regional Contrast(2022 iclr)](#15bootstrapping-semantic-segmentation-with-regional-contrast2022-iclr)
  - [16.Dmt: Dynamic mutual training for semi-supervised learning(2022 pr)](#16dmt-dynamic-mutual-training-for-semi-supervised-learning2022-pr)
  - [17.Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation(2023 cvpr)](#17revisiting-weak-to-strong-consistency-in-semi-supervised-semantic-segmentation2023-cvpr)
  - [18.Augmentation Matters: A Simple-yet-Effective Approach to Semi-supervised Semantic Segmentation(2023 cvpr)](#18augmentation-matters-a-simple-yet-effective-approach-to-semi-supervised-semantic-segmentation2023-cvpr)
  - [19.Instance-specific and Model-adaptive Supervision for Semi-supervised Semantic Segmentation(2023 cvpr)](#19instance-specific-and-model-adaptive-supervision-for-semi-supervised-semantic-segmentation2023-cvpr)
  - [20.Conflict-Based Cross-View Consistency for Semi-Supervised Semantic Segmentation(2023 cvpr)](#20conflict-based-cross-view-consistency-for-semi-supervised-semantic-segmentation2023-cvpr)
  - [21.Fuzzy Positive Learning for Semi-supervised Semantic Segmentation(2023 cvpr)](#21fuzzy-positive-learning-for-semi-supervised-semantic-segmentation2023-cvpr)

# Semi-supervised semantic segmentation
## 1.Guided Collaborative Training for Pixel-wise Semi-Supervised Learning(2020 eccv)
���������һ�������������������SSL���GCL��һ����ѵPaper/Semi-Supervised_Semantic_Segmentation/image/21.png)

T1��T2Ϊ����ģ�ͣ�task specific�������в�ͬ�ĳ�ʼ�������T1(x)��T2(x)��Ŀ��Ϊ��עy��FΪ��ȱ��̽������������Ϊx��Tk(x)�������������ΪH\*W�ĸ���ͼ��ΪTkÿ�����ص㣨Ԥ�������ȱ�ݵĸ��ʣ�CΪһ��ͼƬ����pipeline������+ģ��+���򻯡�

- ѵ���ĵ�һ�׶Σ��̶�F��ѵ��T1,T2�������б�ע�����ݣ���MSE��ƽ������ʧ�мල��ѵ�������ޱ�ע�����ݣ�������Լ����ѧϰ���е�֪ʶ��
  - Dynamic Consistency Constraint��������α��ǩ������������������ģ�͵�֪ʶ������һ����ֵ$\xi$����F�����ȱ�ݸ���ͼ�и��ʴ���$\xi$�����ص���1������T1Ϊ������F�����ȱ�ݸ���ͼ�У�ĳ����T1ȱ�ݸ��ʴ���T2������T2��ֵΪ��α��ǩ������MSE����T1��ʧ��T2ͬ��
  - Flaw Correction Constraint��ϣ��Tk�����ʹF�����������0����ĳ���ص���T1��T2�е�ȱ�ݸ��ʶ�����$\xi$������MSE��0ΪĿ�����F�������ʧ��FΪ�̶��ģ���
- ѵ���ĵڶ��׶Σ��̶�Tk��ѵ��F��ϣ�������F���������|Tk-y|��������ͨ��ϡ����sharp������ѧϰ����ˣ����ǽ�|Tk-y|����C������+ģ��+���򻯣��������ΪF����ֵ����MSEѵ����

## 2. Mean teachers are better role models:Weight-averaged consistency targets improve semi-supervised deep learning results(2018 nips)
�����ǰ�ලѧϰ�ġ�һ�������򻯡����������ݾ�����裨���ݷֲ����ɵ��ܶ�����ָ��ľ�������������ɣ��������һ��δ��ǵ�����Ӧ��ʵ�ʵ��Ŷ�����Ԥ�ⲻӦ���������仯��

���Ȼع���ƪ���෽�������¡�

- Temporal Ensembling for Semi-Supervised Learning(2017)�����Piģ�͡������б�ǩ���ݽ��мලѧϰ�������ޱ�ǩ���ݣ�ÿ�ν�������ǰ����������������ǿ������Ժ�dropout������������϶���ͬ��ʹ��MSE��ʧԼ��������ѵ�����У��ޱ�ǩ���ݵ�MSE��ʧռ��Ȩ�ؼ�С��
![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/1.png)

- Temporal Ensembling for Semi-Supervised Learning(2017)���� Pi-Model �Ļ����Ͻ�һ�������Temporal Ensembling��ֻ����һ��ǰ����������ǰԤ��������ʷԤ������ƽ��ֵ����������㣬��ʷԤ������EMA(exponential moving average��ָ������ƽ��)���㣺$y'=\alpha y'+(1-\alpha)y$
- 
![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/2.png)

Mean Teachers����Temporal Ensembling �ĸĽ��棬Temporal Ensembling ��ģ�͵�Ԥ��ֵ���� EMA����Ҫ��ѵ��ʱά���������ݵ�EMAԤ��ֵ����Mean Teachers �����˶�ģ��Ȩ�ؽ��� EMA��ÿ��batch���¼��ɡ�
![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/3.png)
�Ƚ��мලѧϰ����ģ�Ͳ�������Ϊѧ������ʦģ�͡������ޱ�ǩ���ݣ��ֱ��������������ѧ������ʦģ�ͣ��������ߵ�MSE��ʧ��ѵ��ѧ��ģ�ͣ���һ��batchѵ���꣨���򴫲��꣩��ʹ��ѧ��ģ��Ȩ�ؼ�����ʦģ��Ȩ�ص�EMA�����¡�

## 3.Semi-Supervised Semantic Image Segmentation with Self-correcting Networks(2020 cvpr)
������Եĳ���Ϊ��С��������mask��ע���������󲿷�Ŀ��bbox��ע��

![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/4.png)

- Ancillary Segmentation Model:����ΪͼƬ��bbox(ʵ����Ϊ��ά����H\*W\*(C+1)����bbox�ڵ����ص㣬��Ӧ�����ͨ����1)�����Ϊ�ָ�����ͼ��ѵ����ʼʱʹ��ȫ��ע��������ѵ����֮���̶��������ڳ��ڶ����ල�������Ƚϸߡ�
- Primary segmentation model:��Ҫģ�ͣ��б�ע�����ලѵ�����ޱ�ע������Self-correction module�����Ϊ��ֵ����ѵ����
- Linear Self-Correction:����С������ֲ���Ancillary Segmentation Model��Primary segmentation model��KLɢ��֮����Ȩ����ΪĿ�꣬�÷ֲ��н����⡣ѵ�����ڣ�Ancillary Segmentation ModelռȨ�ظߣ��𽥽��͡�
- Convolutional Self-Correction:ʹ�þ�����罨ģѧϰ��У������ǰ����ģ�͵�logit�����������Ϊ���롣�����б�ע���������ලѵ��Primary segmentation model��Convolutional Self-Correction�������ޱ�ע��������Convolutional Self-Correction�������Ϊ��ֵ�㽻������ʧ������ʧ��������Convolutional Self-Correction������ʼ��ʱ����һ���б�ע����ѵ��Ancillary Segmentation Model��ֹ����̫�ߣ�����Convolutional Self-Correctionֻ�������ģ����¸����������Convolutional Self-Correctionѧ����ô���Ancillary��primary

## 4.Semi-Supervised Semantic Segmentation with Cross-Consistency Training(2020 cvpr)
�������ָ���������İ�ල������һ����ѵ����������Ϊ��ԭʼ����ķֲ�û�б��ֳ��ָ����ĵ��ܶ����򣬶����ز��ʾ�����Ͼ�����裬�ʺϽ���һ����ѵ����

![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/5.png)

- ������һ������������������һ��main�������Ͷ����������������ʧ�����������ļල��ʧ���ޱ��������ʧ��ɣ��ޱ�ע����������������������ر�ʾzֱ������������������������Ŷ���ֱ����븨��������������������������͸���������������Ľ�������ʧ�����У���������ֻ���б�ע����ѵ����
- ÿ����������1��1���б�ע/�ޱ�ע������Ϊ�˱��������б�ע������ʹ����������OHEM�ķ���
- ��������˼��ֲ�ͬ�ģ������ر�ʾ�����Ŷ��ķ���
- �ÿ�ܻ�����Ӧ�õ����ල���񣬺Ͷ����������ڹ���������󣬽��ض���������������͸�������������Ӧ�������ѵ����Ӧ�Ľ���������

## 5.Semi-supervised semantic segmentation needs strong, varied perturbations(2020 BMVC)
�����ǽ�һ��������Ӧ�õ�����ָ������ļ�ƪ���£����ලѧϰ�е���ǿ����Cutout��CutMixӦ�õ�SSL��Ϊ�Ŷ���

- ���ĵĴ󲿷�ƪ�������۷���������һƪ��˼�����ƣ���Ϊ��ԭʼ����ķֲ�û�б��ֳ��ָ����ĵ��ܶ����򡱣�һ���һ�������򻯷����������ڷָ����񡣵�����û�󿴶���������һ���Ŷ�/���߽߱�֮��ġ�
- CutOut����ǣ����ѡ����ͼƬ��ľ���������Ϊ0��CutMix�ǣ�����������ͼƬ����һ��ͼƬ��ȡ�������룬����һ��ͼ�����ಿ��ƴ��һ����Ϊ���룬������ʧʱ��GTҲҪͬ��ƴ�ӡ�
- �����У�ʹ����Mean Teacher��ʦ����ܣ���CutMix���ͼƬ��Ϊѧ����������룬��ԭͼƬ����ʦ��������Ϊ��ֵ��ƴ�Ӻ���Ϊѧ�������α��ǩ��

## 6.ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning(2021 WACV)

��Ȼ��һ�������򻯷��������������һ���µ�����ޱ��������������ǿ����Classmix��

![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/6.png)

- Classmix��������������ޱ�ǩ����ͼƬA,B����A��Ԥ�����ͼȡargmax�����ȡһ����𣬽���Щ����Ӧ��ԭͼ���ص�cut������ճ��B�ϣ��õ���ǿ���ͼƬ����ǿ��ͼƬ��GT��A,B��Ԥ�����ͼȡargmax��ƴ�Ӷ��ɡ�
- ����ʹ����Mean teacher��ܣ��Խ�ʦ�����Ȩ������EMA���¡�����A,B���ý�ʦ������������ǿ���ͼƬѵ��������ѧ������Ĳ�����
- α��ǩ��˼�룬��A,B��Ԥ�����ͼȡargmax����������ִ�����ŵ�Ԥ�⣬�����߽�Ĳ�ȷ���ԣ��񻯣���С��Ⱦ��
- ÿ��ѵ��ȡһ���б�ǣ���ʧ�����ɼල���ֺ���ǿ������ɣ���ǿ���ֵ���ʧռ��Ȩ���ɳ��ڵĺ�С�������

## 7.Semi-Supervised Semantic Segmentation via Adaptive Equalization Learning (2021 NIPS)
�������SSL����ָ��б��ֲ��ѵ��ض���������adaptive equalization learning (AEL)�������α��ǩ/һ�������򻯷���Ԥ�ⲻ׼ȷ��������������Щ���ı��֡�

![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/7.png)

- ����ʹ����Mean teacher��ܣ����������������ǿ�����������ӱ��ֲ���������ѵ�������г��ֵ�Ƶ�ʡ���ʦģ�͸����ޱ�ǩ������α��ǩ������ѧ��ģ�͵Ĳ�������EMA���£�ѧ��ģ�͸���������Ԥ�⣬�����б��������Ԥ�����������������Ŷȣ���ָ���ޱ����������ʧ���㡣
- Confidence Bank:��ѵ�������У�ͨ���б�������ı�����ά��ÿ���������Ŷȡ���������˶��ָ�꣬���ղ�����$Conf^ {c} =  \frac {1}{N_ {l}}  \sum _ {i=1}^ {Nl} \frac {1}{N_ {i}^ {c}} \sum _ {i=1}^ {Nc}p_ {ij}^ {c} ,c \in {1,\cdots ,C}$��������ÿ��ѵ�������EMA����ÿ��������Ŷȡ�
- Adaptive CutMix:��������ޱ�ע���ݵ�������ǿ��������ԭCutMix���������ڣ�����ͼƬ������������Ŷ���������
- Adaptive Copy-Paste:��������б�ע���ݵ�������ǿ�������������Ŷȼ�����ʣ�������𣬸���Դͼ�������ڲ��������������ز�������ճ����Ŀ��ͼ���ϡ�
- Adaptive Equalization Sampling:�������Ŷȼ���һ�����Ĳ����ʣ�����ʹ�����������������޼ල��ʧ�����Ǹ������ص�Ԥ��������������һ���Ӽ���
- Dynamic Re-Weighting:α��ǩ������������Ӱ�죬����Ϊ�ޱ�ǩ������ÿ�����ص����Ȩ�أ�ʹ���Ŷȣ��˴�Ϊsoftmax������ֵ���ߵ����ص����ʧ���и��ߵ�Ȩ�ء�


## 8.Semi-supervised semantic segmentation with cross pseudo supervision(2021 cvpr)
�������￴�Ǻܼ򵥵�һƪ���£�һ����Լ������������ʵ��SOTA��Ҳ�����ѵ��ʱ��Щtrick��

- �������磨������ͬ��ͬ���������ʼ���������б�������ලѵ���������ޱ��������ͬʱ�����������磨����ͬ����ǿ������������������ķָ�����ͼת��Ϊone-hot������Ϊ��һ��ͼƬα��ǩ������ʧ�����򴫲�������ģ���ʹ����CutMix��ǿ������
- ���к��������������˶Ա����ۣ��Ҹо����trick����࣬����ƪ��Ȼ��SOTA��˵�������ֵ��
- �ڳ��ڻ���α�ලѵ������Խ��Խ���𣿼�ʹ���������ĳ�����صķ��඼�Ǵ���ģ�Ҳ�������������������ģ���𣬻�����Щ�������У�����ԽѵԽ��ʵ�������������������ġ�

## 9.Semi-supervised Semantic Segmentation with Directional Context-aware Consistency(2021 cvpr)
����Ȥ��˼·��Ҳ��һ����Լ���ķ��������������ĵ�һ���ԡ�

![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/8.png)

- �������ڰ�ල�����У�ģ�ͺ����׶��൱���޵�ѵ�����ݽ��й�����ϣ���������������������Ԥ�⣬�Ӷ����¶�ѵ���ڼ�δ�����ĳ����ķ��������ϲһ�������ǿ����Ҳ�޷��γɶ������ĵ�һ���ԡ�
- Context-Aware Consistency:Ϊ��ʵ��������һ���ԣ������ޱ�ǩ���ݣ��ü���������overlap��patch��Ϊ���룬������patch�ɿ���overlap���־��в�ͬ�����ĵ���ǿ������������һ���ԣ�����ģ�Ͷ�������patch��overlap����һ�¡�Ϊ����ģ�������յ�Ԥ�����˻���������������Ϣ�����н�encoder�������������ӳ��������ά�����󣬽���һ����Լ����
- Directional Contrastive Loss:Ϊ�˱�֤����patch�ص����ֵ�representation���룬l2��ʧ̫�����޷��ڲ����������븺�����������ʹ�������������ԡ����н����Ա�ѧϰ�ķ��������Directional Contrastive Loss��������������������ƶȣ��ּ�С�븺���������ƶȡ���ʧ�����������صģ�ÿ�����ص��������ص�����������һ��ͼƬ�Ķ�Ӧλ�ã�������Ϊ�����ġ����ң���������patchͨ���������󣬿��Եõ�ÿ�����ص����Ŷȣ��ڼ�����ʧʱ�Ը��������Ŷȸ��ߵ�Ԥ��Ϊ��׼������o1������o2��o1���루����o2�ĸ�������ֻ����o2���ݶȣ���
- Negative Sampling:���������������Ч�������ã�����ѡ��������ͬ��ͬһ�������ص㡣���и��ݷ��������������Ϊα��ǩ��ֻ��������������
- Positive Filtering�����overlap�������Ŷȸ��ߵ����ص㣬���Ŷ���С��ĳһ��ֵ������������ʧ��

## 10.PseudoSeg: Designing Pseudo Labels for Semantic Segmentation(2021 iclr)
�ǳ�trick��һƪ���£��˹�����˺ܶ๫ʽ���Ƚ��ѱ��������һ�ּ���α��ǩ�ķ�����
![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/9.png)
- ���ȣ��漰�� Class activation map (CAM)��ԭ���������Ƕ�λ���о�����������ͼ����Ϊ���������ɸ��Ӿֲ�һ�µ�mask���õ�H\*W\*Cά��������
- ����ģ�ͺ�������������ͼ��ƴ������Ϊ������Ϣ�����������������ͬ��1\*1�������ΪK,Q������ע����������CAMΪV��Ȩ���õ�SGC 
- ���գ�ͨ��У׼�ںϣ�һ���Լ���Ƶ���ѧ��ʽ������SGC�ͽ����������������õ�α��ǩ��

## 11.ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation(2022 cvpr)
��������˻���self-training�ķ���������Ӧ����ǿ������ǿ��Ч���ܺá�
- ������ѵ�����ලѵ����ʦģ�ͣ���α��ǩ�������������ޱ�ǩ����ʹ������ǿ��һ��ѵ���µ�ѧ��ģ�͡����ǣ��б������̫�ٵ���α��ǩ�������ѣ�ѧ��ģ�Ϳ��ܹ���Ͻ�ʦģ�ͣ���ѧ�����¶�����
- ST�����б�����������������ޱ������һ����ģ��һ�����ѵ����������Ϊÿ��С�����ظ�����α��ǩ����ѵ��ѧ��ģ��ʱ��Ӧ����������ǿ����Ӧ��ǿ������ǿSDA��
- ST++�����ȿ��ǿɿ���δ�����������ѵ��������һ��ͬ�ʡ�Ϊ�˱��ⳬ������������ֵ���������һ��ѡ����ԣ��ڼලѵ�������У�����K�����㣬��ÿ�������α��ǩ�����յļ������meanIoU,������ÿ��ͼƬ���м���ľ�ֵ����ѵ��ѧ��ģ��ʱ����ʹ�ñ��������meanIoU��ߵ�R���ޱ��������Խ��˵��Խ�ȶ���Ҳ��Խ�ɿ���ѵ������ʣ�µ��ޱ���������´�α��ǩ���������������ѵ��һ���µ�ѧ��ģ�͡�

## 12.Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels(2022 cvpr)
������˼��˼·��idea�Ĳ�������Ȼ����һ�����ѵ��/α��ǩ�����У�Ϊ�˱�֤α��ǩ�����������ֻȡԤ�����Ŷȸߵ�����ѵ���������Դ󲿷��������ء����Ŷȵ͵����ؿ�����top_k����Ԥ����������ʲ�ࣩ�������ڡ�����ܵ�����������ж�׼ȷ�����磬���˺�������������أ�������ȷ�����ǽ����������һ������Ϣ�����������һ�����ò�����α��ǩ�ķ�����

![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/10.png)

- ����ʹ��ʦ����ܣ���ʦ���EMA���£�ѧ������ɼල+�޼ල+�Ա���ʧѵ��
- �����ޱ�������������ʦ����Ԥ����������أ�������ֵ����ֵ����ʱ�����С����Ϊ�ɿ�/���ɿ�α��ǩ���ɿ�α��ǩֱ�Ӽ����޼ල��ʧ
- ���ò��ɿ�����(U2PL)����ÿ���࣬����Ŀ������/����/�������������ر�������Ա���ʧ
  - anchor���أ���α����ǵ����ΪC��Ԥ�����Ŷȴ���ĳһ��ֵ
  - ��������anchor�����в���
  - �����������б����������ǲ�ΪC����Ԥ��ΪC�����Ŷ�Ϊtop_k�������ޱ�������������ţ�Ԥ��ΪC�ĸ��ʼȲ�������ܵ�Ҳ��������ܵ�
  - ���ڳ�β�ֲ�����Щ���ĸ������٣�ά����һ���ڴ��


## 13.Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation(2022 cvpr)
���ĵĳ�����Ҳ������˼����һ����Լ���У�α��ǩ����ʦģ�ͣ�����ȷ�Ժ���Ҫ�������α��ǩ��ʹѧ��ģ�Ͷ�strong��ǿ��ͼƬƫ������Լ����
![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/11.png)
- ���Ļ���Mean teacher��ܣ�ʹ�ý�������ʧ�������˶����Ŷ���
- �����Ŷ�����������ʦģ�ͣ�ȡ��������ľ�ֵ��softmax��Ϊ���ǩ��ȡonehot��ΪӲ��ǩ����ѵ��ʱֻ��EMA��������һ���Ĳ�����
- �����Ŷ�������ѧ��ģ�ͱ�������������ر�ʾʩ���Ŷ������Ŷ���ͨ��T-VAT�Խ�ʦģ�͵�����Կ�ѵ���õ��ģ�û�󿴶���ô������
- �����Ŷ�����/ǿ��ǿ
- �޼ල��ʧʹ��onfidence-weighted CE loss����ʵ���Ǽ��˸����Ŷ�Ȩ�أ�α��ǩ����Ӧ��softmaxֵ������ֵʱΪȨ�أ�����Ϊ�㡣

## 14.Semi-supervised Semantic Segmentation with Error Localization Network(2022 cvpr)
����һƪ������Щ���ƣ��۽���α��ǩ���������⣬�����ELN(Error Localization Network)����֮ǰ��ECN(��������)��ȣ�������������ӣ������׹���ϣ�ELN������޹����񣬲���ֻ��ҪԤ���ֵ���룬�������ܸ��á�

![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/12.png)

- ʹ��Mean teacher��ܣ��ڽ�����ͷ�ϻ����˸�ӳ�������õ��������жԱ�ѧϰ
- ʹ���б���������ලѵ��ѧ�����磨������+������������ѵ��K��������������ֻ������ʧ����ĳ����ֵ�Ľ���������ʧ��ֻ�Ż���ʧ�ܴ�����磩���Եõ����Ƕ��ǵĴ���Ԥ�⣻����ELN��ʧ����K+1��������������ֱ����������������ͼ��������������ELN����õ���ֵ���룬�����Ȩ��������ʧ��ֵΪ1����ȷ���������ϴ󣬸���1/0�����ر�����Ȩ������Ϊѵ���ĵ�һ�׶Ρ�
- �ޱ�����������ݽ�ʦ��������������Ԥ�����ELN������Ϊ0������α��ǩ��Ϊ��Ч����ѧ��������������ල��ʧ������Ա���ʧ����ѧ����������ı���ΪĿ�꣬����ʦ�������������Ѱ�������͸���

## 15.Bootstrapping Semantic Segmentation with Regional Contrast(2022 iclr)
��12������һ����ʹ�öԱ�ѧϰ(ReCo��ʧ)���ò��������ء�

![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/13.png)
- Mean teacher��ܣ��ڱ�������ӱ�����������������Խ��жԱ�ѧϰ��ReCo��ʧ�ѶԱ���ʧ����������ƶȻ��ɵ���ˣ�������minibatch�����и������ص�ľ�ֵ��query�͸���Ϊ���������ġ�ͨ��active sample��ReCoֻ��Ҫ��С�� 5% �� pixel �� contrastive learning����ʡ�ڴ棬��Ӧ�˷ָ�����
- Active Key Sampling������ϣ����������������ĸ��࣬��ѵ��ʱ��̬ά��һ��C\*C���󣬴洢ÿ�������֮������ƶȣ����ص��ֵ֮��ĵ������softmax����ݸ÷ֲ���������
- Active Query Sampling����α��ǩ����һ����ֵ��ֻ�������Ŷȵ͵������������жԱ�ѧϰ
- ֻ�����Ŷȸߣ���ֵ��Active Query Samplingһ�£�����������α��ǩ����ල��ʧ��

## 16.Dmt: Dynamic mutual training for semi-supervised learning(2022 pr)
��ƪ������Ȼ��עα��ǩ���������⣬��Ϊͨ������ʦ��ģ��Ԥ�����Ŷ�ɸѡ����ɸȥ�����Ŷ���ȷ��ǩ/���������ŶȵĴ����ǩ������� Dynamic Mutual Training����������ģ���໥ѧϰ�������ݷ����ж�Ԥ������Ŷȣ����¼�Ȩ��ʧ��

- Dynamic Mutual Training:��A�ලѵ������α��ǩ������������ѵ��B������ÿ�����ص�Ľ�������ʧ��Ҫ����һ����̬Ȩ�أ���A��Ԥ�������B��ͬʱ����BԤ�����Ŷ�(��$\gamma _1$�η�)ΪȨ�أ���ͬʱ����A��Ԥ�����Ŷȸߣ���BԤ�����Aα��ǩ�������Ŷȵ�(��$\gamma _2$�η�)ΪȨ�أ���ͬ��A�����Ŷȵ�ʱ��Ȩ�����㣬��Ϊα��ǩ�����š�
- ģ�ͳ�ʼ����Ϊ�˱�֤�ָ������ģ�����㹻�Ĳ��죬ʹ�ò�ͬԤѵ�����ݼ����������ݼ��������б����������������Ԥѵ��ռ�Ⱥܴ�ʱ��ʹ�ø��õ�Ԥѵ��Ȩ�أ��ٲ����б�������Ĳ�ͬ�Ӽ�����ѵ����
- ����ѵ����ܣ�ÿ��ѵ��ʱ��ʹ�����Ŷ���ߵ�һ��������α��ǩ�����ŵ���������������ӵ�100%��
- �о�ͦˮ��һƪ���£���Ӳ���˸�Difference maximized sampling������PR�������ģ���֪��Ϊɶ������ͦ�ߵġ�


## 17.Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation(2023 cvpr)

���Ļع���SSL������ǿһ����Լ������������FixMatch(������ǿ��Ԥ��Լ��ǿ��ǿ����ͬһ��ģ��)�����߿��Կ���5��������10�ļ򻯣������һ���µ��Ŷ���ܡ�
![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/14.png)
- Unified Perturbations for Images and Features:����ͼ�񼶵�ǿ��ǿ����������������ǿ���򵥵�dropout���ɵõ��ܺõ�Ч��������ͬ������Ŷ��ֳɶ�����ǰ������ʹѧ���ܹ���ֱ�ӵ���ÿ������ʵ��Ŀ��һ���ԡ�
- Dual-Stream Perturbations����ͬһ������ǿ���룬�����������ǿ��ǿ���ù��������ͼ�淶����ǿ��ͼҲ���Ա���Ϊǿ��������ǿ��ͼ֮���һ���ԡ�
- ���ϱ������Ŷ������������UniMatch���ĸ�������ǰ��������ǿ/����ǿ��ǿ/�����Ŷ���

## 18.Augmentation Matters: A Simple-yet-Effective Approach to Semi-supervised Semantic Segmentation(2023 cvpr)
SSL����ָ�SOTA�𽥸��ӣ����������һ�ּ���Ч�Ŀ�ܣ���Ҫͨ����ǿ������ǿ������Ӧ��ע������Ϣ��ʹ��Լලѧϰ��������ǿ������Ӧ��ලѧϰ��

Mean teacher��ܡ��ල��ʧ+�޼ල��ʧ�������ޱ����������������µ�������ǿ��ʽ��
- Random Intensity-based Augmentations:������ǿ�ȵ���ǿ�ŵ�һ��������������ǿ�ȣ����������ǿ�����������˸����ֵ���������������ǿ��������Ϊ������ǿ�ȵ���ǿ�������ģ���������ƫ��ʺ��޼ලѧϰ��
- Adaptive Label-aided CutMix����ͼƬ��������ǿ+����ǿ�ȵ���ǿ�󣬸��ݵ�ǰģ�͵�Ԥ�����ÿ��ͼƬ�����Ŷȣ�������һ������������ޱ��ͼƬ���б��ͼƬmixup(���Ŷȵ͵��ޱ��ͼƬ���ʺϱ��б��ͼƬ����)�����ŶȸߵĲ��û�ϣ�����ͼƬ�������ģ�������ϣ������ϵģ����ͼƬ�����ޱ���������mix���õ�������ǿ��ͼƬ��

**����һ�󲿷����ڵ��Կ������Զ���GB 2312�����ʽ���ļ������UTF�����´��Ѿ��ƻ��ˣ�������ļ�֮ǰ��github��ͬ����������Ķ�ʧ��**
## 19.Instance-specific and Model-adaptive Supervision for Semi-supervised Semantic Segmentation(2023 cvpr)

![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/15.png)
- ��������ǰSSS������չ���������������������������ѵ������Ϊ���ۡ������������δ�������һ��ͬ�ʣ���ȫ������δ�������֮��Ĳ����ѧϰ���ѡ�����Ĳ���������Ŷ���������������������������������
- Ϊ�˶�̬�ĺ���ʵ�������ѳ̶ȣ����������һ�����ȨIoU��������ƽ�����⣩$\frac {|z_ {1}(c)|}{H\times W} IoU( z_ {1} (c), z_ {2} (c))$
- 11$z_1(c)$������ʦ��ģ��Ԥ��֮���wIoU�����Դ˼������������ѳ̶�
$\gamma _i=\phi ( p_ {i}^ {t}  ,  p_ {i}^ {s} )=1-[  \frac {\rho _ {i}^ {s}}{2} wIOU( p_ {i}^ {s} , p_ {i}^ {t} )+ \frac {\rho _ {i}^ {t}}{2} wIOU(p_ {i}^ {t} , p_ {i}^ {s})]$

- Model-adaptive strong augmentations:������Ϊ����������Ŷ����������Ϣ��ʧ�����������ѳ̶�����̬����ǿ��ǿ��CutMix�ͻ���ǿ�ȵģ���ǿ�ȣ��и���ʽ
- Model-adaptive unsupervised����$\gamma$��̬������������ʧȨ��
## 20.Conflict-Based Cross-View Consistency for Semi-Supervised Semantic Segmentation(2023 cvpr)
![Alt text](Paper/Semi-Supervised_Semantic_Segmentation/image/16.png)
- ��������ǰSSS��������Լ����˹��Ŷ�����ֹ������֮�以��̮���������Ա��⻥����ϣ��������һ��ǿ��������ѧϰ��ͬ��ͼ�����ķ�����
- ��������֧��Эͬѵ����ܣ�������ʼ����ͬ�����磬��������ļල��ʧ+�ޱ��������һ������ʧ����Ϊα��ǩ��+������ʧ����С������������ȡ�������������ԣ�
- ������ʧ��ϣ������������ȡ��ͬ��ͼ�����������������ƶ�����������֮��������ԣ�$c_ {dis}^ {\alpha }$ =1+ $\frac {f_ {1}^ {\alpha }\cdot \overline {f_ {2}}}{|f_ {1}||\times |f_ {2}|}$ ��1+���ƶ���Ϊ������ʧ��
- Conflict-based pseudo-labelling(CPL)���ڻ���ලʱ���������������ó�ͻ��ǩ�����䰴���Ŷȷ���������˼�Ȩ��
## 21.Fuzzy Positive Learning for Semi-supervised Semantic Segmentation(2023 cvpr)
�������α��ǩ���������⣬���ģ������ѧϰ��FPL�������ڴӶ��������ȷ�ĺ�ѡ��ǩ�����Ϣ���塣�ڱ����У��������룬�Լ��弴�õķ�ʽ����׼ȷ�� SSL ����ָĿ��������Ӧ�ع���ģ������Ԥ�Ⲣ���Ƹ߸��ʵ�����Ԥ�⡣ FPL����򵥵�ʵ������Ч���������ż������α��ǩ�ĸ��ţ�����ʵ�����������ؼ��������֡�
- Fuzzy Positive Assignment:�趨һ����ֵ����Ԥ������ŶȽ�������ۼӣ�������ֵǰ��Ϊģ������
- Fuzzy Positive Regularization:��һ�����ʱ���û������ף��������๫ʽ����Ҫ�����ģ�������������������һ����ʧ������ϣ��ʹģ�������еģ����Ŷȣ���Сֵ��󻯣����������ֵ��С��