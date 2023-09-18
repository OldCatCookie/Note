# Channel Charting

**1. 方向首先尝试ULA下的Channel Charting加波束成形的端到端模型建模**
   
   a.如何设计损失函数？

   b.模型在哪里训练？在一个BS中训练还是分布式训练(结合联邦学习？)

**2. 如何将CC从ULA单天线推广到MU与联合BF结合$\rightarrow$利用CC去决定哪些UE或者哪些BS配对。**

**3. 若CC推理的信道质量很差，可以将CC作为CSI估计的补充以提高CSI估计的复杂度与精确度。**

## Channel Charting: Locating Users within the Radio Environment using Channel State Information

我们提出了一种新的框架，我们称之为信道图，该框架将从小区中的UE获取的CSI映射到低维映射中，该低维映射捕捉真实UE在空间中位置的局部几何结构。信道制图是无监督的，即不需要来自真实UE位置的任何信息，例如，从全球导航卫星系统（GNSS）获得的信息。

 **建模流程**

 ![](image/20230918202918.png)

 ### 1.提取特征$\mathcal{F}$

 使用欧氏距离作为CSI point之间相异度的度量$d_f(F,F^{'}) = || F - F^{'} ||_F$

 使用原始二阶矩R2M来计算CSI: $\overline{H} = \displaystyle \frac{1}{T}\sum^T_{t=1}h_t h^H_t$

 **A.特征缩放**

用$\tilde{H}$来替代$\overline{H}$以解决由于路径损耗问题导致的CSI不能很好表征空间几何关系的问题。

$$
    \tilde{H} = \displaystyle \frac{B^{\beta - 1}}{||\overline{H}||^{\beta}_F} \overline{H} \ \ with \ \ \beta = 1 + \frac{1}{(2\sigma)}
$$

**B.特征变换**

最直接的令$F = \tilde{H} \  \ denote \ as \ \mathbb{C\{\cdot\}}$

其他变换：实部$\mathfrak{R}\{\cdot\}$、虚部$\mathfrak{I}\{\cdot\}$、角度域$\angle\{\cdot\}$、绝对值$|\cdot|$。

其中角域中 R2M 的绝对值被证明是所有考虑通道模型和场景的最稳健的通道特征。

### 2.Channel Charting算法

 **A.PCA主成分分析**

 $Z_{PCA} = [\sqrt{\sigma_1}u_1,.....,\sqrt{\sigma_{D^{'}}}u_{D^{'}}]^H$

  **B.Sammon's Mapping**

Sammon的映射(SM)是一种经典的非线性方法，它将高维点集映射到较低维度的点集，目的是在两个点集之间保持小的成对距离

![](image/20230918211230.png)

上式非凸，所以分解为两个凸函数迭代优化问题：

![](image/20230918211326.png)

并且使用PCA初始化Z能取得更好的效果。

**C.自动编码Autoencoder**

学习一个编码器$\mathcal{C}:\mathbb{R}^{M^{'}} \rightarrow \mathbb{R}^{D^{'}}$和一个解码器$\mathcal{C}^{-1}:\mathbb{R}^{D^{'}} \rightarrow \mathbb{R}^{M^{'}}$

示例网络架构：

![](image/20230918224325.png)

误差定义：

$$
    E = \displaystyle \frac{1}{2N}\sum^N_{n=1}||f_n - \mathcal{C}^{-1}(\mathcal{C}(f_n))||^2_2 + \frac{\beta}{2} ||W^{(5)}_{enc}||^2_F
$$

