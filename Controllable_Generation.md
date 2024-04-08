- [1. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation(2023.5)](#1-dreambooth-fine-tuning-text-to-image-diffusion-models-for-subject-driven-generation20235)
- [2. LoRA:Low-Rank Adaption of large language model(2021.10)](#2-loralow-rank-adaption-of-large-language-model202110)
- [3. Controlnet: Adding Conditional Control to Text-to-Image Diffusion Models(2023.11)](#3-controlnet-adding-conditional-control-to-text-to-image-diffusion-models202311)
- [4. T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models(2024 aaai)](#4-t2i-adapter-learning-adapters-to-dig-out-more-controllable-ability-for-text-to-image-diffusion-models2024-aaai)
- [5. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations(2022 iclr)](#5-sdedit-guided-image-synthesis-and-editing-with-stochastic-differential-equations2022-iclr)
- [6. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models(2023.8)](#6-ip-adapter-text-compatible-image-prompt-adapter-for-text-to-image-diffusion-models20238)
- [7. DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models(2024 cvpr)](#7-dragondiffusion-enabling-drag-style-manipulation-on-diffusion-models2024-cvpr)
- [8. DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing(2024 cvpr)](#8-dragdiffusion-harnessing-diffusion-models-for-interactive-point-based-image-editing2024-cvpr)
- [9. DiffEditor: Boosting Accuracy and Flexibility on Diffusion-based Image Editing(2024 cvpr)](#9-diffeditor-boosting-accuracy-and-flexibility-on-diffusion-based-image-editing2024-cvpr)

# 1. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation(2023.5)
Google�Ĺ�������ȥ������ͼģ��ȱ��ģ�¸���subject�ο���������ȷ����۵Ĺ������ϳ��µ��ڲ�ͬ������subject��������Dreambooth��һ���ı���ͼ����ɢģ�͡����Ի������·�������Ԥѵ��������ͼģ�ͽ���΢����ʹ��ѧ�ὫΨһ��ʶ�����ض�subject�󶨣��Ϳ���ʹ��Ψһ��ʶ�����ϳɲ�ͬ������subject����ӱ����ʵ��ͼ�񣬲����Ա�����ؼ���ʶ��������
- Ŀ�꣺����һ��subject�ļ���ͼ��û���κ��ı�������Ŀ�������ɾ��и�ϸ�ڱ���Ⱥ�**�ı���ʾ����**�ı仯��subject��ͼ�񣬱仯�����ı�subjectλ�á��ı�subject���ԣ�������ɫ����״�����޸�subject�����ơ��ӵ�����������޸ġ�
![Alt text](Paper/diffusion/image/image-17.png)
- Designing Prompts for Few-Shot Personalization:Ϊ�˽�subject������ɢģ�͵ġ��ʵ��С�������Ϊfew-shot set���������prompt��a [identifier] [class noun]��class noun��Ϊ�����ù��е�������֪ʶ
- Rare-token Identifiers:���з����ִ�Ӣ�ĵ�����Ϊidentifier�Ǵ��Ż��ģ���Ϊģ����Ҫ�����������������subject��ϵ������������ض�������xx55yЧ��Ҳ���ã���Ϊ�ִ������ÿ��letter�����tokenҲ��ǿ���顣���в��õķ����ǣ����ڴʻ���ж�rare token�������������䷴�����text������С�ڵ���3ʱЧ���ã����Ӷ��õ�����unique identifier���ַ����С�
- Class-specific Prior Preservation Loss:��΢��Ԥѵ������ģ��ʱ��Ϊ�˱�������֪ʶ�����⽵�Ͷ����ԣ������һ��loss
![Alt text](Paper/diffusion/image/image-19.png)
  �ڶ�����ʧ��������$x_{pr}$=a [class noun]�����ɻ���ģ�������ͼƬ$x_{pr}$�������Լ����ɵ��������ලģ�ͣ��Ӷ��ڱ�batch�Ĳ����Ż��б��ָ����飻��һ����ʧ��Ϊƽ�����ؽ���ʧ����few-shot setΪ�ල��

<p align = "center">  
<img src="Paper/diffusion/image/image-18.png"  width="300" />
</p>

# 2. LoRA:Low-Rank Adaption of large language model(2021.10)
��������LLM΢���ķ�����ԭ��ܼ򵥣�����ɢģ����Ҳ����Ч��

<p align = "center">  
<img src="Paper/diffusion/image/image-21.png"  width="300" />
</p>

- �������̶������������ֻѵ��ĳЩ�����������������Щ����������ͨ������ֽ��ɸ��ٵĿ�ѵ��������������˵�����ڲ���$W_O$����΢��ʱѵ��$W=W_0+\Delta W=W_0+AB$������$W_0$�������ᣬֻѵ��$A\in R^{D\times r}, B\in R^{r\times D}$������r���Ժ�С����2����Ҫѵ���Ĳ���������½����ڳ�ʼ��ʱ��A��ʼ��Ϊ��׼��˹�ֲ���B��ʼ��Ϊ0.
- �÷�����Ҫ����ĳЩ������Բ��֣�����Transformer�е�QKV������ͶӰ���Լ�FFN�����Բ���
- ��ɢģ�ͣ������LoRA΢��CLIP�Լ�Unet�н���ע����������Բ��֡�
- 
# 3. Controlnet: Adding Conditional Control to Text-to-Image Diffusion Models(2023.11)
���������һ��Paper/diffusion/image/Image-to-image translation������������ɢģ�ͷ��������ø���ͼƬ�����磬��Եͼ���������ƹǼܡ��ָ�ͼ����ȡ����ߵȣ������������ɵ�ͼƬ�������ض������µ�ѵ��������������С��һ���ı���ͼ��ѵ���Ŀ������ݣ�����ֱ��ѵ��/΢�����������Controlnet����ǿԤѵ������ͼ��ɢģ�Ͷ���spatially localized, task-specifically��ͼ���������ɡ�ʵ���˺ܺõ�Ч�����������ݼ���С���кܺõ�scalabel��³���ԡ�
- ControlNet:����һ��NN�飬ControlNet��ԭ�в����̶������ƣ����Ʋ��ֿ�ѵ�����������в����ӵ�ԭ�в��֣����ƿ������Ϊx+��������������c���������Ա������ƿ��������ѵ��֮������������
<p align = "center">  
<img src="Paper/diffusion/image/image-22.png"  width="300" />
</p>

- Controlnet for SD:����SD�ı������м��ʹ��Controlnet�����ҽ�����ӵ�skip connection�ϡ�����SD��LDM��ͼƬ����c��resize��64\*64�Ҿ���һ��tiny��CNN������Controlnet
<p align = "center">  
<img src="Paper/diffusion/image/image-23.png"  width="200" />
</p>

- Training:ѵ��ʱ�����50%���ı�$c_t$�滻Ϊ���ַ��������ַ�����ǿ�� ControlNet ֱ��ʶ����������ͼ���е����壨�����Ե�����ơ���ȵȣ����������������ʾ��������������������������������ģ��Ӧ��ʼ���ܹ�Ԥ���������ͼ�񣬵��Ƕ����������Ƶ�ѧϰ������֡�ͻȻ�������󡱣�ģ�Ͳ�������ѧϰ�������ƣ�����ͻȻ�ɹ�����ѭ������������ͼ��ͨ���Ż��������� 10K��
![Alt text](Paper/diffusion/image/image-24.png)
- Classifier-free guidance resolution weighting��Ϊ��ʵ��CFG������ͼƬĬ�ϼ��뵽���������������С�����û���ı�prompt�������ȫ����ᵼ��CFGʧЧ����ֻ�������������ᵼ��guidance��ǿ��Ϊ�������CFG Resolution Weighting������$\epsilon_c$�м�������ͼƬʱ��Controlnet������뵽skip-connection֮ǰ��һ��Ȩ��$w_i=(\frac78)^{12-i}$��$i$Ϊ��i��block([0,12])����ǳ����Ȩ��������1
![Alt text](Paper/diffusion/image/image-25.png)
- Composing multiple ControlNets������ʹ�ö�������ͼƬ��ֱ�ӽ���ӦControlnet�Ľ������������




# 4. T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models(2024 aaai)
T2Iģ���У��ı������ṩ׼ȷ��structuralָ�����������ɽ���������ǵ�������������һЩ���ӳ����½������Ҳ��ȶ���������Ϊ��������Ϊ�ı��޷��ṩ׼ȷ��ָ���������ڲ�֪ʶ���ⲿ�����źţ������ֶ�����Ժ����׵��Եͳɱ�ѧϰ���Ӷ����T2I-adapter������ΪԤѵ���õ�T2Iģ���ṩ���弴�õ�adapter��ʵ��Structure control���������ؼ��㣬mask����ɫ�ȣ���
- ģ�ͣ�������SDΪ����LDM+Unetȥ��+Ԥ��������text��CLIP��ȡ�����뽻��ע������
- T2I-Adapter���ṹҲ�ܼ򵥣��Ƚ�����ͼ��Pixel unshuffle�²�����64\*64���پ����ĸ����block���ñ��뵽�ĸ��߶ȵ���������ͼ��ֱ�Ӽӵ�ȥ��Unet�������ж�Ӧ�ߴ�����ͼ�ϡ����ң�ֱ�ӽ�adapter��ȡ������ͼ��Ȩ��Ӽ���ʵ��multiple conditions control��
- Non-uniform time step sampling during training�����з��֣���adapter����ʱ�������������guidance���������󽵵���Ч�ʣ�ÿһ������������ȡһ����������ͬʱ��ʹ��DDIM����ʱ��guidance�������ڣ�t�ϴ�ʱ������ʱ��Ч�����к�������control����ˣ�������ѵ���׶β���ʱ��tʱ������Ĳ���t�ϴ�����ڽ׶Σ�$t=(1-(\frac{t}{T})^3)*T$������ǿ��guidance�����൱�ڣ�adapterĬ������t�ϴ��ʱ�������������guidance���ֲ��ý���Ч�ʣ�
![alt text](Paper/diffusion/image/image-55.png)

# 5. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations(2022 iclr)
һƪ�Ƚ����ڵĽṹ������ͼƬ���ɵĹ����������Ƚϼ�
- �������ã����ȿ�һ�±���ʵ����ʲô�����������ƣ�����ͼ��ʾ��������ģ������һ��RGBͼƬ��Ϊguidance����ȿ�����ɫ��ͼ��Ҳ��������ԭͼ��������Ӳ�Ķ��������У���ģ�ͻ����guidanceͼ������Ȼ�����е�ͼƬ����ʵ��һ��faithfulness and realism�ľ��⡣
![alt text](Paper/diffusion/image/image-56.png)
- �������ܼ򵥣�ѡ��һ��ʱ���$t_0$����guidance����$t_0$���������Դ�Ϊ��ʼ����ʼȥ��ֱ��0��$t_0$Խ��ͼƬԽ��ʵ��guidanceԽ������˿���$t_0$�Դﵽ����
- ���ۣ�����ͨ��SDE�Ƶ���֤�������������õ���$x_0$��guidance֮���L2��ʧ����һ�Ͻ磬�������������ѵ�����޸ı��ʵ��edit

# 6. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models(2023.8)
���������һ���µ�adapter����T2Iģ�ͼ���image prompt������ͼ��ʾ����structure control�ĸ���������ͬ������ȹ�ȥ��adapter������������ͬʱ����������΢��/��ͷ��ʼѵ����������ƥ�䡣
![alt text](Paper/diffusion/image/image-58.png)
- ������������Ϊ����ȥ��adapter�����������ֻ�ǽ�ͼƬ����ͨ������Ľ���ע������Ԥѵ��SDģ�ͣ��������ԭ��ֻ��text�������У��Ӷ�ʹͼ�������޷�ͨ������ע������ЧǶ�뵽Ԥѵ��ģ���С����������һ���µ�adapter��
- IP-Adapter����Unet����ԭ�н���ע�����㣬����һ���µĽ���ע���������image���������ߵ�query���ã���������Ϊ�µ���������Լ�Ȩ����Ϊ�����Ч�ʣ�ע������Q�����ã���K,V��ԭK,VΪ��ʼ����
- IP-Adapter��reusable and flexible��ѵ�õ�adapter�����ƹ㵽��ͬһ������ɢģ��΢���������Զ���ģ�ͣ����ҿ�����ControlNet ������controllable adapter���ݣ���image prompt��structure control���������
![alt text](Paper/diffusion/image/image-57.png)
# 7. DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models(2024 cvpr)
����ʵ����ͨ����ק(drag)ʵ��edit������classifier-guidance˼�룬����������Ӧ���ݶ�������ʵ�ָ�Чimage edit���˹����������ģ���΢����Ϊ�˽��༭�ź�ע�뵽ģ�͵���ɢ���̵��У������ֱ༭ǰ��ͼ�������һ���ԣ����������ַ�����˼·ֱͦ�ӵģ�������ϸ�ڲ��٣����������ܽ�
![alt text](Paper/diffusion/image/image-59.png)
- DDIM inversion with memory bank:��ԭͼǱ������$z_0$��DDIM��ʽ�𲽼��룬��ÿ��t��$z_t$�Ͷ�ӦUNet decoder��ע�������$K,V$����memory bank�С���$z_T$Ϊ��ʼ������ȥ�룬��SDEdit����
- Gradient-guidance-based editing design:classifier guidance�ڲ���ʱ�����÷���ģ�͵��ݶ�$\triangledown log\ p(y|x_t)$�������ɣ�����ģ����ʵ�����ƹ�Ϊ������ģ�͡�������x��y�����ƶȡ����Ĺ���ͨ��������Ӧ�������������������ݶ������������edit��Ŀ�ꡣ
  - content editing����ɢģ���м���������ǿ��Ӧ��ϵ����������t����memory bankȡ����ӦDDIM����$z_t$������ȥ��Unet��ȡ�м�����F���ٸ������ԭͼĿ��(original content position)��mask���Ӷ��õ�ԭͼĿ��������������ٸ���generate���̵�Ǳ�ڱ�������$z_T$��ʼȥ�룩������ȥ��Unet��ȡ�м����������� ��עtarget dragging position��mask��ȡ�õ�Ŀ�������������Ϊ��ʵ��drag��ͼƬedit������ϣ������mask���������������ƣ���cosine�������ƶȣ�������������Ϊ������ʽ��S�����ֲ���ȫ�����ֶ���������
    ![alt text](Paper/diffusion/image/image-60.png)
  ������ҵ���⣬�����������ݶ���log p���ݶ�֮���ǲ��˸����ŵģ�����ݶ�������������ʹ����������С���Ӷ�ʹS���ƶ�����
  - consistency��Ϊ�˱�֤��edit����֮�Ᵽ��һ�£��������������$m^{share}$ָorigin��targetĿ������Ĳ����Ĳ�����
  ![alt text](Paper/diffusion/image/image-61.png)
  - total��Ϊ����Ӧ�ض����񣬻�������һ���Ż����������ʽ
    ![alt text](Paper/diffusion/image/image-62.png)  
  - ���з�������ɢģ�͵��м��������ڶ�����������������������Ϣ�������ؽ���ԭʼͼ���������Ƶ�����ϸ�ڴ���һ�������ͼ�񣻵�����������������Ͳ����������޷��Ը߲������ṩ��Ч�ල�����½��ģ������˱��Ľ�����������ֱ��ʵ��м���������guidance
- �Ӿ�����ע������Ϊ�˽�һ����ǿguidance��������memory bank�е�K,V�滻��Ӧt��ȥ��ģ��decoder��ע�������K��V��ʹgenerateȥ������п��Խ��ԭͼ��DDIM����汾��������

<p align = "center">  
<img src="Paper/diffusion/image/image-63.png"  width="300" />
</p>

> ps1:����2.2���ֵĹ�ʽ2�����

> ps2:����ֱ�ӽ������������ݶȼӵ�ȥ��ģ�͵Ľ�����ˣ�������Ǵ�ģ���ʵ��û���⡣��Ϊ����ȥ��ģ��Ԥ�������������score�����߷��������ţ�$log\ p(y|x)$���ݶȺ������������ݶ�Ҳ������ţ�scoreӦ�ü������$log\ p(y|x)$���ݶȣ��������Ӧ�ü�����������������������ݶȡ�
# 8. DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing(2024 cvpr)
����Ҳ����dragGan����������Ҳͦ�񣩣�����༭drag���뵽��ɢģ�����򣬲��������һ��benchmark�����Ĳ��õķ���ֻ����һ���ض�ʱ�䲽ע��drag��Ϣ����ʵ�ֱ༭��
- Identity-preserving Fine-tuning�����ȶ�LDM����LoRA΢������ǰ�ߵĹ���һ������ԭͼ����DDIM inversion��Ϊ��ʼ���������Ӷ�΢��ʹLDMģ��ѧ���ؽ������õ�����ɢ�����б�������ͼ��
- Diffusion Latent Optimization������ϸ�ڲ�����ϸ��������Ҫ��������ʧ��������һ�֤dragǰ�������ĵ�����������Unet�������ͼ������һ�£��ڶ��֤ȥ���δ��mask������ǰ�󱣳�һ�¡��ڵ���ʱ�䲽�У����ж�θ��Ż���
![alt text](Paper/diffusion/image/image-64.png)
- Reference-latent-control�������Ӿ�����ע��������ԭͼ��ΪK,V


# 9. DiffEditor: Boosting Accuracy and Flexibility on Diffusion-based Image Editing(2024 cvpr)
����DragonDiffusion�������Ĺ���
![alt text](Paper/diffusion/image/image-65.png)
- Content Description with Image Prompt������ṹ��ΪDragonDiff��������image prompt����QFormer��ȡ�Ӿ�token����text tokenһ�����뽻��ע��������IP-Adapter��һ�µĲ���ע����������Ȼʹ����memory bank�����Ӿ�����ע����
- Sampling with Regional SDE����ȥ������ʹ��ODE��ȷ���Բ�����������Ϊ���ֲ��������˴������Ͷ����ԣ���˲�����region SDE��Ҳ������ĳЩ(region)ʱ�䲽ʹ��SDE������������>0������edit����ķ����Դ�����������
- Editing with Gradient Guidance����DragonDiff���������Ļ����ϣ������޸�ΪRegional gradient guidance����ԭ���������ݶȷֱ������ڶ�Ӧ���򣻲��Ҽ�����Time travel�����������������У�������roll back��������$z_{t-1}$���»��$z_t$������˵��Ч�������������Ҳ�ǽ����DragDiffusion���ڵ�����ɢʱ�䲽�н���ѭ��guidance��
![alt text](Paper/diffusion/image/image-66.png)