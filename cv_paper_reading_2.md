- [��Ŀ��ȹ��ƣ�Multi-view stereo��](#��Ŀ��ȹ���multi-view-stereo)
  - [1.MVSNet: Depth inference for unstructured multi-view stereo(2018)](#1mvsnet-depth-inference-for-unstructured-multi-view-stereo2018)
  - [2.R-MVSNet:Recurrent mvsnet for high-resolution multi-view stereo depth inference(2019)](#2r-mvsnetrecurrent-mvsnet-for-high-resolution-multi-view-stereo-depth-inference2019)
  - [3.Point-based multi-view stereo network(2019)](#3point-based-multi-view-stereo-network2019)
  - [4.cascade MVSNet:Cascade cost volume for high-resolution multi-view stereo and stereo matching(2019)](#4cascade-mvsnetcascade-cost-volume-for-high-resolution-multi-view-stereo-and-stereo-matching2019)
  - [5.P-mvsnet: Learning patch-wise matching confidence aggregation for multi-view stereo(2019)](#5p-mvsnet-learning-patch-wise-matching-confidence-aggregation-for-multi-view-stereo2019)
  - [6.CVP-MVSNet:Cost volume pyramid based depth inference for multi-view stereo(2020)](#6cvp-mvsnetcost-volume-pyramid-based-depth-inference-for-multi-view-stereo2020)
  - [7.Fast-mvsnet: Sparse-to-dense multi-view stereo with learned propagation and gauss-newton refinement(2020)](#7fast-mvsnet-sparse-to-dense-multi-view-stereo-with-learned-propagation-and-gauss-newton-refinement2020)
  - [8.UCS-Net:Deep stereo using adaptive thin volume representation with uncertainty awareness(2020)](#8ucs-netdeep-stereo-using-adaptive-thin-volume-representation-with-uncertainty-awareness2020)
  - [9.Patchmatchnet: Learned multi-view patchmatch stereo(2021)](#9patchmatchnet-learned-multi-view-patchmatch-stereo2021)
  - [10.TransMVSNet: Global Context-aware Multi-view Stereo Network with Transformers(2022)](#10transmvsnet-global-context-aware-multi-view-stereo-network-with-transformers2022)
- [ͼ��ѧʰ��](#ͼ��ѧʰ��)
# ��Ŀ��ȹ��ƣ�Multi-view stereo��
## 1.MVSNet: Depth inference for unstructured multi-view stereo(2018)
[��������](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.pdf)


���ĵ�һƪ��ȹ��Ƶ����ģ����һ��ʼ����Ӧ�Ծ�֮���⣬�ȶ������ͼ��ѧ����֪ʶ���ֶ���һЩ���ͺ�pytorch���������������Ÿо�������pipeline�㶮�ˡ����������MVSNet���������ѧϰ��MVS����

- Multi-view stereo����һ�Ųο�ͼ�Ͷ���ԴͼƬ��Ԥ��ο�ͼ��pixel-wise��ȣ�ѵ��ʱÿ����ͼ�ֱ���Ϊ�ο�ͼ
- ����ṹ����һ��shared CNN��Ϊÿ��ͼƬ(3\*H\*W)��ȡ����ͼ(H/4\*W/4\*F)����ÿ������ͼ��һ����ȷ�Χ�ھ���ȡ��D�������ֱ��ն�Ӧ��ȵĵ�Ӧ����source image������ͼ�õ�Ӧ�Ա任ͳһ���ο�ͼƬ��ƽ�棬�����ƴ�����õ�$V_i$��H/4\*W/4\*D\*F�����÷������$V_i$����ͳһ�õ�һ��cost volume C���ߴ���$V_i$һ�£���C����3D�������U-Net���磬�������򻯣����վ���һ��1\*1�������$H/4\*W/4\*D'���������softmax���õ�����ͼ��ÿ�����ض�Ӧһ��Dά�ĸ�����������Ӧ��ǰ��ȵĸ��ʣ����������soft argmin����ʵ���Ƕ�ÿ�����صĸ�������ȡ��ȵ��������õ�intial depth map����ref image resize��H/4\*W/4\*3,��intial depth map(H/4\*W/4) concatenation���������������������intial depth map��ӣ����refine���depth map
> һ��û�ܽ�����ɻ��ǣ����յ����ͼ��H/4\*W/4����֪��զ�ָ���ԭ�ߴ磬���߲��ûָ�����R-MVSNet�����п����������ͼȷʵ�²������ı�
- ��ʧ��������intial depth map��refine���depth map���ֱ��ۼ���Ч���ص�Ԥ�������GT��L1����
- ����ͼ������4���������ȵĸ�����ͣ������������Ԥ����������õ�����ͼ
- �������ͼ���ˡ�����ȥ�쳣ֵ
   - photometric consistency��������������������ȥp<0.8�����ص�
   - geometric constraint�����ο�ͼ�е����ص�p1ͶӰ��һ��sourceͼ��$p_i$���ٽ�$p_i$��ͶӰ�زο�ͼ$p_1'$����$p_1$��$p_1'$������ֵ��ֵ�Ͷ�Ӧ��Ȳ�ֵ����ĳһ��ֵ�£����Ϊ����ͼ��������ʵ����ÿ�����ص���������ͼ����
- �������ͼ�ںϡ�N����ͼ�ֱ�Ԥ������ͼ����ÿ��������ͶӰ�õ���ÿ�����ͼ�����ȡ��ֵ����Ϊ������ȹ��ơ�
- �����и����ĵ�Ӧ�Ծ���Ĺ�ʽ���󣬾����https://zhuanlan.zhihu.com/p/363830541�������Ӧ��Ҳ�����ˣ����ɴ�ref��source�ˣ�

## 2.R-MVSNet:Recurrent mvsnet for high-resolution multi-view stereo depth inference(2019)
[��������](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yao_Recurrent_MVSNet_for_High-Resolution_Multi-View_Stereo_Depth_Inference_CVPR_2019_paper.pdf)

��MVSNet�Ļ����ϣ�������GRU�����R-MVSNet�����µ�����
- ������MVSNetЧ���ܺã�������ʹ��3D�����cost volume����ʱ��̫���ڴ��ˣ�Ҳ�������Ӧ�õ��߷ֱ��ʵ�ͼ��R-MVSNet��GRU�滻3D conv����cost volume���򻯣���Ч��С���ڴ�����
- ���磺������MVSNet���ƣ��ڻ��cost volume������ȷ����������GRU��������壩�����ܸ���Ұ��Ϊ��ǰ��ȼ�֮ǰ����������Ч�����ƣ����һ������ͨ����Ϊ1������softmax��ø���volume����GT���㽻������ʧ����Ϊ��������ѵ��
- Ϊ��������ȹ��Ƶķ�Χ��R-MVSNet�ڲ������ʱʹ����inverse depth��������MVSNet����ȷ�Χ�ھ��Ȳ�����inverse depth�ǰ���ȵĵ���ȡ����Ҳ���Ƕ���ȵĵ������Ȳ�����������û����ȷ˵������Ҳ��ˣ�������soft argmin���ع���Ԥ����ȣ���Ȳ��������ȣ�����������һ�ַ��࣬��refineϸ���ķ���
- ������Ϊ���շ�������ѵ����Ԥ��ʱ�൱��argmaxȡ��ȣ��޷���������ؼ���ȹ��ƣ�����Ԥ������ͼ�����������н���ЧӦ���������һ�� Variational Depth Map Refinement�������ڲ�ֵ��ʹ���ͼ���smooth��  
  ��refinement��������һ��reprojection error�����������֣�
  - photo-metric error:��source image $I_i$����ref�����ͼ$D_1$���о�������ɣ�ͶӰ���򲻶԰ɣ�ͶӰ��$I_1$,��zero-mean normalized cross-correlation�������ߵ�error
  - �������ÿһ���������������ؼ��㲢�ۼ�bilateral squared depth difference����һ���ʹ���ͼsmooth    
  
  �������ᵽ���������ʹ���error��С����������ô�������


## 3.Point-based multi-view stereo network(2019)
[��������](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Point-Based_Multi-View_Stereo_Network_ICCV_2019_paper.pdf)

����������ڵ��Ƶ�MVS����Point-MVSNet
- ������MVSNet��ʹ��3D�������cost volume��������3d�������ڴ�����̫��Point-MVSNet�����ڱ���3D conv��Ч��ͬʱ����3D��������
- ˼·���Ȼ���MVSNet����Ԥ����Ե����ͼ����ͶӰ��3d���ƣ����2D������Ӧ��PointflowԤ�����GT�Ĳв�òв�������ϸ�����ƣ��ٵ�����һ����
- coarse depth prediction:��MVSNet��ȣ������е��²���������4��Ϊ8��ͨ����Ҳ����½������3D������ڴ����Ĵ���½�
- 2D-3D�����ںϣ�
  - ��CNN��ȡ����ͼʱ��ÿ��ͼƬ��ȡ�����ߴ������������$F_i$���Ƚ���ͬͼƬ������ͼͶӰ��ͬһƽ�棨��������ڲξ������Σ����ٶԷֱ�ÿ���ߴ������ͼƬ������ͼȡ����������ͳһ��$C_i$����Ϊ2D���������ڴֲ����ͼ���ɵ�3D���ƣ���ÿ���㽫��������$X_p$��ͶӰ��$C_i$�϶�Ӧ������concatenation������Ϊ�������
  - �ɴˣ��ں��˶�߶ȵ�2D��������3D�������������������ǿ����
  - ���ң�ÿ�ε������µ��ƺ���ȡ��2D������������ͬ��ʵ��dynamic feature fetching
- Pointflow:����3D���Ƶ�ÿ���㣬��ͶӰ������sΪ�������2m������㣨����ȼ��Ϊs����ͨ����ÿ�������߾�����پ���MLP��softmax���ÿ�������������ȵĸ��ʡ���󣬶�ÿ�������ĸ��ʳ�ks����Ǽ����ļ�ࣩ���ۼӣ���ü����������Ӷ���òв����Ԥ�⡣��ԭ���ͼ��ӿ��Ի��ϸ�������ͼ���ٵ�����һ���̣��õ����еĵ�"flow"��GT��
- ÿ�ε���������ͼ�����ϲ���������ڣ����Ի�ø��߷ֱ��ʵ����ͼ������С���s���Բ�׽��ϸ������������ֻ��������





## 4.cascade MVSNet:Cascade cost volume for high-resolution multi-view stereo and stereo matching(2019)
[��������](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gu_Cascade_Cost_Volume_for_High-Resolution_Multi-View_Stereo_and_Stereo_Matching_CVPR_2020_paper.pdf)

��MVSNet�Ļ����ϣ������Cascade cost volume����Ӧ��FPN��˼�룬�Ż��ڴ��ʱ��Ч�ʵ�ͬʱ�������Ч��
- ˼·��ʹ��FPN��ȡÿ������ͼƬ�Ĳ�ͬ�ߴ������ͼ��3�������������׶Ρ�����top�㿪ʼ���ֱ�����͵ģ�1/16��������cost volume,����MVSNet���ع�õ����ͼ�������ͼ�ϲ����õ�����һ�׶�����ͼһ�µĳߴ磬����һ�׶����ͼΪ���ģ�ȷ����ȷ�Χ/���������Ƚ��в�������ÿ�����ص�p����ȼ���Ϊ$d_p+\delta$���Ӷ�����Ϊ������裩���е�Ӧ�Ա任���õ�cost volume��������MVSNet���ظ��õ������׶���������ͼ���ֱ�����ԭͼһ�£�
- ÿһ�׶α�ǰһ�׶ε���ȷ�Χ���̣�����ȼ����С���ֱ���������ˣ����ܵ�һstage�;���MVSNet����ȷ�Χ/�����࣬���ֱ��ʵ������ڴ����ĸ�С����ߵ�stage��ȷ�Χ����½����Ӷ��ڴ�����ҲС
- ��ʧ����Ϊÿ���׶����ͼ��ʧ�ļ�Ȩ��


## 5.P-mvsnet: Learning patch-wise matching confidence aggregation for multi-view stereo(2019)
[��������](http://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_P-MVSNet_Learning_Patch-Wise_Matching_Confidence_Aggregation_for_Multi-View_Stereo_ICCV_2019_paper.pdf)

���������һ���µĽ���cost volume(MCV,matching confidence volume)�ķ�����P-mvsnet�ﵽsota
- ��������ȥ��cost volumn�����ǲο�ͼ���أ���Ϊpixel-wise��������³���Բ��ã���cost volumnӦΪ������ģ���ԭ���򵥵ķ�����Ϊ����ͬ�ġ����������һ���µļ���cost volumn�ķ�����patch-wise���Ҹ����죬��Ӧ���˸���ͬ��3D U-Net��������
- ���磺
    - �Ƚ�ͼƬ�����²���4����������ȡ��
    - ��Ӧ�Ա任�󣬼���ÿ��src��ref��MSE�������ȡ����ȡָ���õ�pixel-wise��MCV��ÿ������Ԥ����ǵ�ǰ��ȼ�������Ŷȣ�����һ��patch-wise matching confidence aggregation module���ۺ�ÿ�����ص�����patch��������������ȶ�Ӧpatch���������������Ŷȣ��ۺϵĹ����ǿ�ѧϰ�ģ����������������³���ԣ�
    - ��patch-wise��MCV����3D U-Net�����а����������3D����㣨��1\*3\*3��7\*1\*1�����õ�LPV(latent probability volumn)����LPV softmax����PV(probability volumn),��Ȼع�õ����Ԥ��ֵ����������
    - ���src������ͼ����һ���������ϲ���������������������뾭���ϲ�����LPV concatenation�����������refine����Եõ����߷ֱ��ʵ����ͼ
    - ��ʧ����Ϊ�������ͼԤ����ʧ�ļ�Ȩ��
- ���������ؽ���
    - Depth-confindence:��ȥ���Բ����ŵ�Ԥ�⣬��PV�������Ŷȣ���������PV(argmax)֮��С��0.5��Ԥ��
    - Depth-consistency:�Ƚ�ref�ϵ����ص�p����Ԥ�����ͶӰ��һ��src��p'���ٽ�p'��ͶӰ������������p����Ȳ�����������ʾһ���ԡ�
- ��Depth-consistency�У�����������һ��������˼�ĵط���Ҳ����֮ǰ�ɻ��һ���㣬��ν�p'��ͶӰ��ȥ�����漰�������⣬һ��srcͼû�����ͼ��ȡ����p'����ȣ�����p'��һ���պõ����ص��ϣ������ͼҲû�á�����������������������ᵽ����Ⱦ�Ϊref�����ͼ��
    - nearest depth��ȡ��p'��������ص������ȷ�ͶӰ��̫�����ˣ�
    - bilinear depth��ȡp'�����Ϊ�ٽ��ĸ����ص���ȵ�˫���Բ�ֵ����p'��ͶӰ��ȥ�������ᵽ����GT���������֪ʱ������������õ��±������
    - depth-consisten first depth:ȡp'�ڽ����ĸ����ص��������p����ģ����䰴��Ӧ��ȷ�ͶӰ��ȥ
## 6.CVP-MVSNet:Cost volume pyramid based depth inference for multi-view stereo(2020)
[��������](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Cost_Volume_Pyramid_Based_Depth_Inference_for_Multi-View_Stereo_CVPR_2020_paper.pdf)

����image��������cost volum����������coarse-to-fine���������CVP-MVSNet(cost volum pyramid)����Cascade-MVSNetͦ���
- ���磺ȡL��level��ͼƬ���Ϲ���ͼƬ���������ֱ��ÿ��level��ȡ����ͼ������top level��ʼ��coarsest������Ӧ�Ա任����ݷ����cost volumn������3D�����ع�õ�coarse���ͼ������һ�׶����ͼ�ϲ������Դ�Ϊ����ȷ���µ���Ȳ���ƽ�棬���ݵ�ǰlevel������ͼ����cost volumn��֮���Cascade-MVSNet����Ԥ����ǲв����ͼ��������������һ�׶�
- ��Ȳ���������һ�׶��⣬��Ȳ����ķ�Χ�����������һ�׶����ͼȷ������ȼ��ͨ������0.5��������Ȳ�ľ�ֵ����ȷ�ΧΪ��ref�ϵĵ�ͶӰ��srcͼ�������ԳƵļ����ҲͶӰ��srcͼ���ɶԼ�Լ�������ڼ����ϣ�����srcͼ�����ϵ�ͶӰ��Χǡ�������أ���ʱ�ı߽����㼴��ȷ�Χ
- ��ʧΪÿ�׶����ͼ��ʧ�ļ�Ȩ��
  


## 7.Fast-mvsnet: Sparse-to-dense multi-view stereo with learned propagation and gauss-newton refinement(2020)
[��������](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Fast-MVSNet_Sparse-to-Dense_Multi-View_Stereo_With_Learned_Propagation_and_Gauss-Newton_Refinement_CVPR_2020_paper.pdf)

���ļ��MVS�����Ч�ʺ�Ч���������ɴֵ�ϸ����ϡ�赽���ܵ�Ԥ�⣬���Fast-mvsnet
- ˼·���Ⱦ���2D CNN��ȡ����������cost volumn����3D������򻯺󣬻�ø߷ֱ��ʡ�ϡ������ͼ�������ͼ���д�������ʹ�����ܣ��ȶԶ�ϡ�����ͼ������ڼ��ܣ�����refͼΪ������CNNΪÿ����Ԥ��һ��$k^2$��Ȩ�أ�����ߴ������ͼһ�£������������$k^2$������ȵļ�Ȩ�ͣ������ܵ����ͼ���и�˹-ţ��ϸ������������ͼ����������ۺ��˳�ʼ����ͼ������û����Ҫѧϰ�Ĳ���
- ϡ������ͼ���ͷֱ��ʵ����ͼû��ϸ�ڣ��߷ֱ������ͼ����ɱ�̫�ߣ���˱����ȼ���ϡ��߷ֱ������ͼ����ϸ�����Ƚ�ʡ�ɱ��ֻ�ø߷ֱ��ʡ�����ʵ��Ϊ�����ͷֱ��ʹ���cost volumn��Ԥ����ȣ���ϡ�軯�������ͼ������ó������ͼ



## 8.UCS-Net:Deep stereo using adaptive thin volume representation with uncertainty awareness(2020)
[��������](http://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Deep_Stereo_Using_Adaptive_Thin_Volume_Representation_With_Uncertainty_Awareness_CVPR_2020_paper.pdf)

��cascade MVSNet�ǳ���������ȡ��CNN�õ�U-Net���ڶ�/���׶ε���Ȳ�����Χ������ǰһ�׶����ͼ�ķ�������Ӧ�ص������б���ΪATV(adaptive thin volumn)��ͨ����׶���߷ֱ��ʣ��ϲ�������ϸ����Ȳ�������refine���ͼ

## 9.Patchmatchnet: Learned multi-view patchmatch stereo(2021)
[��������](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_PatchmatchNet_Learned_Multi-View_Patchmatch_Stereo_CVPR_2021_paper.pdf)

���Ľ�������Ӿ��е�patchmatch����Ӧ�õ�MVS�������ÿ�ѧϰ��patchmatchģ��coarse-to-fine����Ч����sota����ͬʱ��󽵵���ʱ����ڴ�����
- ˼·��������FPN��ȡ��߶�����ͼ�����ֱ�����͵�����ͼ����patchmatchģ�飬������ͼ���ϲ�����ָ����һ�׶ε�patchmatch����˼��������һ���׶ε����ͼ����ref imageϸ�����õ�ԭʼ�ߴ�����ͼ
- learning-based patchmatch:����������ʼ�������������ۣ��ڴ���+���۵������ֱ������
    - ��ʼ������R-MVSNet���ƣ���inverse depth����$F_f$����ȼ���ƽ��
    - local perturbation:�ڶ��׶ο�ʼ������һ�׶ε����ͼΪ���ģ�����$N_k$����ȼ��裬��ȷ�ΧҲ��ϸ��
    - adaptive propagation�����ģ�:��ʵҲ�ǲ�����ȼ��裬������ͼ��ÿ���㣬ȡ�ڽ���$K_p$���㣨ʹ��fixedƫ�ƣ�Ҳ�������񣩣�������ȣ���һ�׶�����ͼ����Ϊ��ȼ��衣���Ľ�һ������������Ӧ�ص㣬ϣ����������$K_p$���������ͬһƽ�棨���ƣ��������һ��2D CNNΪref image��ÿ�����ص�ѧϰ��һ��$K_p$��ƫ�ƣ��ӵ�fixed������$K_p$����������ϣ���Ϊ���ղ����ĵ㣬���������Ϊ��ȼ��衣����ϣ���������Ӧ��ƫ�ƿ�����fixed������������������λ��ͬһƽ���λ�ã��ɴ˵õ����õ���ȼ��衣
    - Matching Cost Computation:����ͨ���������������õ�����ȼ��轫����ͼͶӰ��refƽ�棬����match cost���ȷֱ��ÿ��ͼ������ͼ��W\*H\*D\*C����������C��ά�ȷֳ�G��������ƶȼ��㣬�õ�S(W\*H\*D\*G)������һ��3D����������Ŷ�P(H\*W\*D)����Pȡmax���õ�pixel-wise view weights w(H\*W)��w������һ�Σ�����ͨ���ϲ������ɡ���w��ΪS�ļ�Ȩ��������ͼƬS�ľ�ֵ$\bar S$(W\*H\*D\*G)������3D�����óɱ�C(H\*W\*D)
    - Adaptive Spatial Cost Aggregation:��ÿ��������ڽ���$K_e$���㣨���񣩣�����CNNԤ��һ��$K_e$ά��ƫ������������Ϊ���ղ����㣬����ͬ������ĳɱ������������ƶȺ�������ƶȼ�Ȩ���ֵ���õ��ۺϿռ�ɱ�
    - �ԾۺϿռ�ɱ�ʹ��softmax��ø����壬������������õ����ͼ���ٵ�������+����������̣����������׶εĵ��������ֱ�Ϊ221��

## 10.TransMVSNet: Global Context-aware Multi-view Stereo Network with Transformers(2022)
[��������](https://arxiv.org/pdf/2111.14600)

��һƪ��transformerӦ�õ�MVS�������TransMVSNet������ṹ�������е�ͼ�ر���ȷ����
- ����FPN��ȡ��ͬ�ߴ������ͼ
- ����ͬ�ߴ������ͼ�ֱ�����ARF(adaptive receptive field)��������FPN��transformer֮�����Ұ��gap������Ϊ��deformable convolution�������Ұ
- ��top�������ͼ���ֱ�����ͣ�����FMT(feature matching transformer)������λ�ñ����flatten������$N_a$��������transformer�飬��ÿ�������ȼ���ÿ��ͼƬ����ͼ��self-attention���ټ���ref��ÿ��src��cross-attention������ı����src��ֵ��src��Ϊquery����Ϊ�˱�֤��ͬsrc��ѯ��refֵ����
- Ϊ�˽�ʡ����ɱ�����top������ͼ�ᾭ��FMT��֮��ͨ��transformed feature pathway���ͷֱ��ʵ�����ͼ���Ѿ���FMT���ϲ�����͸߷ֱ��ʵ�����ͼ������ARF����������
- �ֱ��ÿ���ߴ������ͼ������ȼ�������͵�Ӧ�Ա任��ͳһ��refƽ�棬����pair-wise feature correlation�ֱ����src��ref��correlation volumn���پ�����Ȩ�ͣ�Ȩ��Ϊ��������˵������ۺ�correlation volumn
- ���ۺ�correlation volumn����3D�����õ�probability volumn��ʹ��argmax������Ԥ�⣬ʹ��focal loss
- �ͷֱ��ʵ����ͼ���ϲ��������һ�׶ε�����ͼ��ϣ�ʵ��coarse-to-fineԤ�����ͼ

# ͼ��ѧʰ��
- �����˶���2D/3D������ת/ƽ��/����/����/����/͸�ӱ任����������꣬��ת����/ŷ����/��Ԫ��
- ���ģ�ͣ�������ģ�ͣ�����/���/��������ϵ���ڲξ���/�����������
- 2D-3D�Լ����Σ��Լ�����Լ�������ߵȣ������ʾ��󣨰˵㷨��⣩����Ӧ���󣬵�Ӧ�Ա任