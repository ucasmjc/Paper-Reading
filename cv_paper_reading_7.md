# Semi-supervised semantic segmentation
## 1.Guided Collaborative Training for Pixel-wise Semi-Supervised Learning(2020)
本文提出了一种适用于逐像素任务的SSL框架GCL，类似于GAN。

![Alt text](%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/image/21.png)

T1和T2为任务模型（task specific），具有不同的初始化，输出T1(x)和T2(x)的目标为标注y；F为“缺陷探测器”，输入为x和Tk(x)连接起来，输出为H\*W的概率图，为Tk每个像素点（预测错误）是缺陷的概率；C为一个图片处理pipeline，膨胀+模糊+正则化。

- 训练的第一阶段：固定F，训练T1,T2。对于有标注的数据，用MSE（平方）损失有监督的训练。对无标注的数据，有两个约束来学习其中的知识。
  - Dynamic Consistency Constraint，类似于伪标签方法，集成两个任务模型的知识。设置一个阈值$\xi$，将F输出的缺陷概率图中概率大于$\xi$的像素点置1。再以T1为例，若F输出的缺陷概率图中，某像素T1缺陷概率大于T2，则以T2的值为“伪标签”，用MSE计算T1损失。T2同理。
  - Flaw Correction Constraint，希望Tk的输出使F的输出趋向于0。若某像素点在T1和T2中的缺陷概率都大于$\xi$，则用MSE以0为目标计算F输出的损失（F为固定的）。
- 训练的第二阶段：固定Tk，训练F。希望检测器F的输出趋向|Tk-y|，但后者通常稀疏且sharp，较难学习。因此，我们将|Tk-y|输入C，膨胀+模糊+正则化，将输出作为F的真值，用MSE训练。