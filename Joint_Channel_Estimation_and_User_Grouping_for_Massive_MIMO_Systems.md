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

根据$w_k$的稀疏性，k个UE可以被分为G组$\{G_1,G_2,...,G_G\}$，共享同样稀疏性的UE(即$w_k$中非零元素位置相同)被划分为一组，本文将稀疏表示向量划分为了两部分$w_k = w^s_k + w^v_k$分别表示共享稀疏向量与个体稀疏向量。

接下来提出了一种高效的基于SBL的方法，用于使用通用稀疏性模型进行联合信道估计和用户分组。

## 稀疏贝叶斯建模

根据接收信号：

$$
    y_k = \Phi(w^s_k + w^v_k) + n_k = \overline{\Phi}\overline{w}_k + n_k, \\
    \overline{\Phi} = [\Phi,\Phi]\  ;\ \overline{w}_k = [(w^s_k)^T,(w^v_k)^T]^T,
$$

假设先验分布：

$$
    p(w^s_k | \gamma^{*}_g) = \mathcal{CN}(w^s_k | 0, diag(\gamma^{*}_g)^{-1}),\forall k \in \mathcal{G}_g ,\\
    p(w^v_k | \gamma^{v}_k) = \mathcal{CN}(w^s_k | 0, \rho \cdot diag(\gamma^{v}_k)^{-1})
$$

其中$\gamma^{*}_g,\gamma^{v}_k \in R^{\hat{L}}$分别指示了$w^s_k,w^v_k$中元素的稀疏显著度，$\gamma^{*}_{g,l},\gamma^{v}_{k,l}$越大则第$l$个元素越趋于零。

设向量$z_k \in R^{G}$中唯一元素1的位置指示了第k个MU属于哪一个group，令$\Gamma^{*} = \{\gamma^{*}_g\}^G_{g=1}$，则$w^s_k$分布可写作：

$$
    p(w^s_k | z_k,\Gamma^{*}) = \prod^G_{g=1} \{\mathcal{CN}(w^s_k | 0, diag(\gamma^{*}_g)^{-1})\}^{z_{k,g}}
$$

构建参数的先验分布：

$$
    p(\gamma^{*}_g) = \prod^{\hat{L}}_{l=1} \Gamma(\gamma^{*}_{g,l} | a,b) \\
    p(\gamma^{v}_k) = \prod^{\hat{L}}_{l=1} \Gamma(\gamma^{v}_{k,l} | a,b)
$$

则接收信号的分布可以写为

$$
    p(y_k|w^s_k,w^v_k,\alpha^{-1}I) \\
    p(\alpha) = \Gamma(\alpha|a,b) 
$$

> 得到待估参数集$\Theta = \{\alpha,\overline{W},\Gamma^{*},\Gamma^{v},Z\}$
> 其中 $\overline{W} = \{\overline{w}_k\}^K_{k=1},\Gamma^v = \{\gamma^v_k\}^K_{k=1},Z = \{z_k\}^K_{k=1}$

## 基于变分贝叶斯推断的参数估计

使用$q(\Theta)$来替代后验分布$p(\Theta|Y)$，通过最小化它们之间的KL距离来逼近：

$$
    D_{KL}(q(\Theta) || p(\Theta | Y)) = -\int q(\Theta) \ln \frac{p(\Theta | Y)}{q(\Theta)}d\Theta
$$

根据平均场定理，$q(\Theta)$可以近似为：

$$
    q(\Theta) = q(\alpha) \prod^K_{k=1}q(\hat{w}_k) \prod^G_{g=1}q(\gamma^{*}_k) \prod^K_{k=1}q(\gamma^v_k) \prod^K_{k=1}q(z_k)
$$

最小化KL距离即找到$q^{*}(\Theta)$使得下式最大化：

$$
    q^{*}(\Theta) = \underset{q(\Theta)}{arg\ max} \int q(\Theta)\ln \frac{p(\Theta | Y)}{q(\Theta)}d\Theta
$$

迭代更新过程：

$$
    \begin{aligned}
        q^{(i+1)}_1 = \underset{q_1}{arg \ max}U(q_1,q^{(i)}_2,q^{(i)}_3,q^{(i)}_4,q^{(i)}_5) \\
        q^{(i+1)}_2 = \underset{q_2}{arg \ max}U(q^{(i+1)}_1,q_2,q^{(i)}_3,q^{(i)}_4,q^{(i)}_5) \\
        q^{(i+1)}_3 = \underset{q_1}{arg \ max}U(q^{(i+1)}_1,q^{(i+1)}_2,q_3,q^{(i)}_4,q^{(i)}_5) \\
        q^{(i+1)}_4 = \underset{q_1}{arg \ max}U(q^{(i+1)}_1,q^{(i+1)}_2,q^{(i+1)}_3,q_4,q^{(i)}_5) \\
        q^{(i+1)}_5 = \underset{q_1}{arg \ max}U(q^{(i+1)}_1,q^{(i+1)}_2,q^{(i+1)}_3,q^{(i+1)}_4,q_5) \\
    \end{aligned}
$$

令$<\cdot>_{p(x)}$代表求取$p(x)$对于$\cdot$的期望，令：

$$
    \hat{\phi}_k = <z_k>_{q(z_k)} \\
    \overline{\mu}_k = <\overline{w}_k>_{q(\overline{w}_k)}
$$

$$
    g^{*}_k =arg\ \underset{g}{max}\hat{\phi}_{k,g}, \\
    \mu_k \triangleq <w_k>_{q(\overline{w}_k)} = \overline{\mu}_{k,1} + \overline{\mu}_{k,2}, \\
    \Omega_k = supp(\mu_k)
$$

则估计的下行信道为：

$$
    h^e_k = A_{\Omega_k}(\Phi_{\Omega_k})^{\dagger}y_k
$$