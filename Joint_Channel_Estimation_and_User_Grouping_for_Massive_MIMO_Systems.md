## 摘要

本文提出了一种基于稀疏贝叶斯学习的方法，用于联信道估计和用户分组，以增强多用户 massive MIMO 系统中常见稀疏性的效果。文章重点关注如何处理优化问题，通过引入一个通用的稀疏模型，它包括了共享稀疏性作为一个例和个体稀疏性，以解决用户分组和信道估计之间的战。文章还提供了对所提出算法的收敛分析和计算复杂度分析。

## 问题建模

考虑一个mMIMO系统，1个BS配备N个天线，K个MUs配备一根天线，BS广播长度为T的导频序列，记为$X \in \mathbb{C}^{T \times N}$。则下行的第k个UE处的接收信号记为$y_k \in \mathbb{C}^{T \times 1}$

$$
    y_k = Xh_k + n_k,
$$

$$
    h_k = \displaystyle\sum^{N_c}_{c=1} \sum^{N_s}_{s=1} \epsilon^k_{c,s} a(\theta^k_{c,s})
$$

$$
    a(\theta) = [1,e^{-j2\pi \frac{d_2}{\lambda}sin(\theta)},...,e^{-j2\pi \frac{d_N}{\lambda}sin(\theta)}]^T
$$

> $h_k \in \mathbb{C}^{N \times 1}$: BS到第k个UE的下行信道向量
> $n_k$: 加性高斯白噪声、
> $N_c$: 散射簇的数量
> $N_s$: 每个散射簇子径的数量
> $\epsilon^k_{c,s}$:这条径上的复增益
> $\theta^k_{c,s}$: 每一径对应的方位角偏移
> $a(\theta)$: AoD对应的导向矢量

为了方便计算，将MU k的真实AOD表示为$\{\theta^k_l,l=1,2,...,L\}$，其中L=NcNs。设$\hat{\theta} = \{\hat{\theta_l}\}^{\hat{L}}_{l=1}$是均匀覆盖角域$[-\frac{\pi}{2},\frac{\pi}{2}]$的固定采样网格，其中$\hat{L}$表示网格点的数量。如果网格足够精细，使得所有真实的AoD都可以落在（或几乎靠近）网格上，此时令$A = [a(\hat{\theta}_1),a(\hat{\theta}_2),...,a(\hat{\theta}_{\hat{L}})]$，则

$$
    h_k = Aw_k
$$

$w_k \in \mathbb{C}^{\hat{L}\times 1}$中的非零元素指示了真实AoD落在网格中的位置。

$$
    y_k = XAw_k + n_k = \Phi w_k + n_k
$$

根据$w_k$的稀疏性，k个UE可以被分为G组$\{G_1,G_2,...,G_G\}$，共享同样稀疏性的UE被划分为一组，本文将稀疏表示向量划分为了两部分$w_k = w^s_k + w^v_k$分别表示共享稀疏向量与个体稀疏向量。

接下来提出了一种高效的基于SBL的方法，用于使用通用稀疏性模型进行联合信道估计和用户分组。

## 