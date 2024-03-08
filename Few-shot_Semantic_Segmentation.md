- [few shot semantic segmentation](#few-shot-semantic-segmentation)
  - [1.One-Shot Learning for Semantic Segmentation(2017 BMVC)](#1one-shot-learning-for-semantic-segmentation2017-bmvc)
  - [2.Few-Shot Semantic Segmentation with Prototype Learning(2018 bmvc)](#2few-shot-semantic-segmentation-with-prototype-learning2018-bmvc)
  - [3.Conditional networks for few-shot semantic segmentation(2018 iclr)](#3conditional-networks-for-few-shot-semantic-segmentation2018-iclr)
  - [4.CANet: Class-Agnostic Segmentation Networks With Iterative Refinement and Attentive Few-Shot Learning(2019 cvpr)](#4canet-class-agnostic-segmentation-networks-with-iterative-refinement-and-attentive-few-shot-learning2019-cvpr)
  - [5.Feature Weighting and Boosting for Few-Shot Segmentation(2019 iccv)](#5feature-weighting-and-boosting-for-few-shot-segmentation2019-iccv)
  - [6.AMP: Adaptive Masked Proxies for Few-Shot Segmentation(2019 iccv)](#6amp-adaptive-masked-proxies-for-few-shot-segmentation2019-iccv)
  - [7.Pyramid Graph Networks with Connection Attentions for Region-Based One-Shot Semantic Segmentation(2019 iccv)](#7pyramid-graph-networks-with-connection-attentions-for-region-based-one-shot-semantic-segmentation2019-iccv)
  - [8.PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment(2019 iccv)](#8panet-few-shot-image-semantic-segmentation-with-prototype-alignment2019-iccv)
  - [9.Attention-Based Multi-Context Guiding for Few-Shot Semantic Segmentation(2019 aaai)](#9attention-based-multi-context-guiding-for-few-shot-semantic-segmentation2019-aaai)
  - [10.PFENet: Prior Guided Feature Enrichment Network for Few-shot Segmentation(2020 trami)](#10pfenet-prior-guided-feature-enrichment-network-for-few-shot-segmentation2020-trami)
  - [11.FSS-1000: A 1000-Class Dataset for Few-Shot Segmentation(2020 cvpr)](#11fss-1000-a-1000-class-dataset-for-few-shot-segmentation2020-cvpr)
  - [12.Few-Shot Semantic Segmentation with Democratic Attention Networks(2020 eccv)](#12few-shot-semantic-segmentation-with-democratic-attention-networks2020-eccv)

**���ڵ��Կ������Զ���GB 2312�����ʽ���ļ������UTF�����´��Ѿ��ƻ��ˣ����沿������Ϊ����**
# few shot semantic segmentation
## 1.One-Shot Learning for Semantic Segmentation(2017 BMVC)
��һƪ��С������������ָ�����£����ڻ�ͷ���Ѿ�ûʲô��д���ˣ����漸ƪ�Ĵ����ܻ������õ������
![Alt text](Paper/Few-shot_Semantic_Segmentation/image/1.png)
- ǰmask��֧�ּ���ͼƬ����ǰ��mask�ˣ����漸ƪ���¶��ĳɺ�mask��
- ������֧����֧�ּ��������������Ϊ�ָ��֧���ȫ���Ӳ�Ĳ���
- k-shot�����н�k��ͼƬ����k����������������Ϊ���Ǿ��ȸߵ��ٻ��ʵͣ���k�����ȡ�������漸ƪ���»��������ˡ�
s


## 2.Few-Shot Semantic Segmentation with Prototype Learning(2018 bmvc)
��һƪ��ԭ��ѧϰ��˼������SSS�ָ�����£����Ǻܸ��ӵģ�ֻ��˼·дһ�¡�
![Alt text](Paper/Few-shot_Semantic_Segmentation/image/2.png)
ѵ��һ��ԭ��ѧϰ��f������Ϊ֧�ּ�ͼƬ+��Ԫmask�����һ���������˴�����ʧΪ����ѧϰ�ķ�������С����Ŀ�����ԭ�͵ľ��롣�ָ�����g������query���õ�����ͼ���������Բ�ηֱ��ںϣ��پ���1\*1ѹ����һά������һ����õ�N+1������ͼ���ֱ�����Ϊmask����queryһ������f�õ�ԭ�ͣ����������Ӧ֧�ּ�ԭ�͵ľ��룬�����һ��Ȩ��W����Ϊ���һ������ͷ�Ĳ�����



## 3.Conditional networks for few-shot semantic segmentation(2018 iclr)
iclr workshop��һƪ���ģ��о�ûɶ�¶���������˵֧�ּ��ı�ע����ֻ�Ǽ��������͸��������ص�
![Alt text](Paper/Few-shot_Semantic_Segmentation/image/5.png)
- ���µĻ���������֧��ȡ�ı�����ָ��֧��������ӣ���������Ϊ�����ˡ��������ںϷ�ʽ̫��Ӳ�ˣ���߼�ƪ����Ҳ�����Ż��ˡ�
## 4.CANet: Class-Agnostic Segmentation Networks With Iterative Refinement and Attentive Few-Shot Learning(2019 cvpr)
���֮ǰ��ƪ���£����������ܴ�
![Alt text](Paper/Few-shot_Semantic_Segmentation/image/6.png)

- ��һ���������mask+pool��ȡȫ�ֱ����ķ���������ü�ƪ����Ҳ�õ��������ȫ�ֱ����ϲ�����ԭʼ����ͼ�ߴ粢��query������ͼ����������Ϊ�ںϺ��������
- Iterative Optimization Module:���һ�ֵ����Ż��ķ����������Ż�Ԥ���mask
- ��ע���������ƹ㵽k-shot������������SEע�����ķ�����Ϊÿ��shot�ĸյ�����������ͼ����һ��Ȩ�أ�k��shot֮��softmaxһ�£��������Ȩ��ע�������Բ�׽�������������ֵ������ԣ�Խ���ƽ������Խ�ã��о���Щ��������Ȩ��Ľ����Ϊ�����������IOM��



## 5.Feature Weighting and Boosting for Few-Shot Segmentation(2019 iccv)
��ѧ����Щ�ѱ�����ƪ���£������Ż�����ı�ʽ�⡱->��ʵ�ǡ�����һ�����������С�ĵ�λ�������������⡣
![Alt text](Paper/Few-shot_Semantic_Segmentation/image/7.png)
- ѵ��ʱ����֧�ּ�ͼƬmask-pool�õ�����������������ͼ���������������ƶȣ��õ������ƶ�ͼ��uqeryԭ����ͼ�������������㡣�������һ����������������ɲ���������ǰ������λ�õ������ľ�ֵ-�����ģ���λ���õ��ģ��ڼ����������ƶ�ʱ��ͣ�٣���Ԫ�س�������ͼ��support�ı���
- Ԥ��ʱ���Ƚ�support����CNN��queryһ���õ�Ԥ�⣬�ø�Ԥ����GT��loss�����ݶ��½���������IoU��Ϊ���Ŷȣ�������N�ֵ������ݶ�ֻ������support�ı���������N�������ֱ��query����Ԥ�⣬�������ŶȽ��м�Ȩ��͵õ�����Ԥ��
- k-shot:��k������ȡ��ֵ����Ϊone-shot������к������в����������³����
![Alt text](Paper/Few-shot_Semantic_Segmentation/image/8.png)

## 6.AMP: Adaptive Masked Proxies for Few-Shot Segmentation(2019 iccv)
һƪ��ˮ�����£�����С����ѧϰûɶ���µ㣬ʹ��FCN����߶ȵ�����ͼ�㴴���𣿻���mask+pool��ֻ�ǽ�С����ѧϰ����ƹ㵽��Ƶ����ָ����������ָ�����Ǵ��������ˣ��Ͼ���ͬ�����ò�ͬ��ܲ������������
![Alt text](Paper/Few-shot_Semantic_Segmentation/image/9.png)
֮�������У��Ҹо�����Ϊȫ��Χ��"proxy"��"imprinting"����������ĸ���չ�������˸��߼��Ĺ��£�ǰ����ʵ����ȫ�ֱ��������߾�������/Ƕ��
## 7.Pyramid Graph Networks with Connection Attentions for Region-Based One-Shot Semantic Segmentation(2019 iccv)
��һƪ˼·����Ȼ�����£����������á�Ϊʲô���ڲ������ᡱ��
![Alt text](Paper/Few-shot_Semantic_Segmentation/image/10.png)
- ����������SSS�ָ�ķ��������Ϯ���������˼·�����ǣ�����಻ͬ���ָ���������ݾ��нṹ������support��ָ֧���ָ��֧ʱ��Ϊһ����Զ���Ϣ�������⡣��ȥ��support��ȡ��ȫ�ֱ�ʾ����query��ÿ�����ض�����ͬ�ģ��ܿ��ܵ�����Ϣ��ʧ�����������һ����ͼ��ģ�ָ����ݵķ�������ʹ��ע��������ʵ��support��query����Ϣ���ݡ�
- Graph Attention Unit:��CNN��ȡ����ͼ�󣬶�query��ÿ������/support��ǰ���������ؽ�ģΪͼ�еĽڵ㣬����queryͼ��ÿ���ڵ��supporttu���нڵ��ע��������������ӳ�����������support�ڵ�ļ�Ȩ�ͣ�����g������ӳ�䣩��Ϊquery�ڵ����ֵ�����õ�����ֵ��ԭ����ͼ������g�����������õ����յ�����ͼ��
- Pyramid Graph Reasoning:������Ϊ����ÿ�����ص㽨ģΪͼ���ܺ���ȫ����Ϣ�����磬�����ۺ͹���֮�佨����ϵ�������������ڵ��Ͽ��Եõ����Ӷ���ı����������б��ԣ������͹���������ģqueryͼǰ����������Ӧ�ػ����õ���ͬ�ֱ��ʵ�ͼ��
## 8.PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment(2019 iccv)
�������ܸߵ�һƪ���£�����֮��о���������϶���Ч��Ҳ�ǡ���ô���ڲ������뵽�������£�����Ϊ��Ҫ�����ǳ��������֧�ּ���
![Alt text](Paper/Few-shot_Semantic_Segmentation/image/11.png)
- ԭ��ѧϰ������һ��C way N shot���⣬��֧�ּ���ͼƬ�������mask-pool����ȡ��ֵ��Ϊÿ�����ԭ�͡�
- �ǲ����Ķ���ѧϰ������queryÿ��������ÿ��ԭ�͵ľ��룬�������Ϊ��������Ԥ�����Ϊ�����������-$\alpha$d($F_{x,y},p_c$)��softmax��
- Prototype alignment regularization:������Ϊ����ȥ��֧�ּ�mask�����ò���֣���������mask��PAR��ѵ��ʱ���ö�query��Ԥ����Ϊmask����ȡԭ�ͺ���ָ�֧�ּ�����queryһ���Խ��������ʧ��ʵ��ԭ��֮��Ķ������򻯡�
## 9.Attention-Based Multi-Context Guiding for Few-Shot Semantic Segmentation(2019 aaai)
�о�ûɶ�¶������������͵����¡�
![Alt text](Paper/Few-shot_Semantic_Segmentation/image/12.png)
- �����Ӧ��ע�������ƵĶ�߶��ں�ģ�飨A-MCG����support��֧��query��֧ÿ���׶ε�����ͼ��ע����ģ���ںϣ�SE��RAM��
- �����һ���ں�k��shot����ķ�������Conv-LSTM�ں�
## 10.PFENet: Prior Guided Feature Enrichment Network for Few-shot Segmentation(2020 trami)
��trick��һƪ���£��о���Ҫ���������
![Alt text](Paper/Few-shot_Semantic_Segmentation/image/13.png)
- Prior for Few-Shot Segmentation:��CANet [46]��ʵ�����������������ģ�͵�������������м򵥵���Ӹ߼������ᵼ�������½�����ˣ���������ͷ����������������Ϣ�ķ�ʽ�����򵥡���������ĸ߼�����ָ����backbone���һ�㣨����㣩��CANet��˽������ɾ�ˡ����Ľ��߼�������Ϊ���飬ָ���ָ����������ǣ���query��support��backbone��ȡ����������query��ÿ�����ؼ�����supportÿ�����أ�maskһ�£����������ƶȣ�ÿ������ѡ���ֵ��Ϊ�����ֵ��
- Feature Enrichment Module:�����һ���ܸ��ӵ�ģ�飬���ں�������������Inter-Source Enrichment����query������ͼ��support��mask pool������ͼ�ϲ�����������ͼ�����������Inter-Scale Interaction����query����ͼ��������Ӧ�ػ�����ȡ��߶ȵ�������������ͬ�߶ȵ���������һ����ֱ��ϸ����
- ����ϸ�ڲ�д�ˣ�ģ����Ƶĺܸ��ӡ�
## 11.FSS-1000: A 1000-Class Dataset for Few-Shot Segmentation(2020 cvpr)
�����һ�����С�����ָ�����ݼ���һ���ص�Ϊ���ࣨ1000��������һ��Ϊ����������������⣨10�������������һ��ģ�ͣ�U-Net��״�ģ�֧�ּ������ȡ��ֵ����query����ͼ������������������룬�о��ܼ򵥣���̫���ס�
## 12.Few-Shot Semantic Segmentation with Democratic Attention Networks(2020 eccv)
Ҳ����Էָ�Ľṹ���������⣬��7һ����������Ϊ����������֮���ƫ��������֧��ͼ����ͨ��������ǰ�������һС��������7��ע���������ᵼ��֧��ͼ��Ͳ�ѯͼ��֮���������һС���������������ܴ�̶���������֧��ͼ������ӡ����������һ�֡���������ͼע������ǰ�������ϵ������������������ز������ӡ�
![Alt text](Paper/Few-shot_Semantic_Segmentation/image/14.png)
- ���廹��һ������������ȡ����˫��֧��ܣ����˸���������ϸ������ǿ�����������µ�ɡ�
- Democratized Graph Attention:��Q��S��ȡ������ͼ���ֱ�ӳ�䵽kq,vq��ks,vs������kq��ks֮���ע����ͼ��hw\*hw������һά��ȡ��ֵ�ָ���ԭ�ߴ磬�����ص�ֵ���������У��ؽ�ע����ͼ��ֵ���Լ��Ĵ����ٳ���һ������HW+1������softmax��һ����õ����յ�ע����ͼ����Ȩvs�õ�ע��������ͼf����f��vq������������һģ�顣
- Multi-Scale Guidance����������ȡ����ÿһ���ֱ��ʽ׶Σ�������DGA���������գ��ڽ������׶���׶�ϸ����ע������������һ�׶ε�������������