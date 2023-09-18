# traffic prediction

## 工作流程与模型分类

**四种工作流程：** 直接预测、分类然后预测、分解然后预测、聚类然后预测。

**三种模型分类:** 统计模型、机器学习模型、深度学习模型

## 统计模型

使用的统计模型主要是时间序列、概率估计和粒子滤波模型。一些有代表性的模型包括ARIMA和Holt-Winters(HW)模型。

***1.SARIMA模型***

**ARMA**自回归移动平均模型是平稳时间序列建模中常用方法：

$$
    \begin{equation}
        \begin{aligned}
            (AR):x_t = \sum^p_{i=1}\phi_{i}x_{t-i} + \varepsilon_{t} \\
            (MA):x_t = \mu + \sum^q_{i=1}\theta_{i}\varepsilon_{t-i} + \varepsilon_{t}
        \end{aligned}
    \end{equation}
$$

则自回归移动平均模型ARMA(p,q)表示为：

$$
    \begin{equation}
        x_t - \sum^p_{i=1}\phi_{i}x_{t-i} = \sum^q_{i=1}\theta_{i}\varepsilon_{t-i} + \varepsilon_{t}
    \end{equation}
$$

使用MLE来估计模型参数，$D$为观测集,假设$\varepsilon_t \sim N(0,\sigma^2)$

$$
    \begin{equation}
        \begin{aligned}
            p(D|\phi) \propto \prod^n_{t=1}p(x_t|\phi,x_{t-1}) \\
            p(x_t|\phi , x_{t-1}) = (\frac{1}{2\pi \sigma^2})^{\frac{1}{2}} \ exp(-(\frac{1}{2\sigma^2})(x_t - \Phi^T x)^2) \\
        \end{aligned}
    \end{equation}
$$


MLE具体来说是使得下式最小化,转化为了最小二乘问题。

$$
    NLL(\theta) = \frac{n}{2}ln(2\pi \sigma^2) + \frac{1}{2\pi \sigma^2}\sum^n_{i=1}(x_t - \Phi^T x)^2
$$

缺点：要求时间序列数据平稳，而ARIMA(p,d,q)在ARMA(p,q)的基础上把差分的过程包含了进来，多了一步差分过程，对应就多了一个参数d，也因此ARIMA可以处理非平稳时间序列。

定义长度为T差分序列$X^{'}_t = [x^{'}_{t-T+1},……,x^{'}_{t-1}]$ ，其中$x^{'}_{t-j} = x_{t-j} - x_{t-j-1}$

定义符号$B$有$Bx_t \doteq x_{t-1}$，可推$x$的k阶差分：

$$
    \begin{equation}
        x^{(d)}_k = (1-B)^d x_k
    \end{equation}
$$

将上式化为

$$
    \begin{equation}
        \phi^{[p]}(B)x_t = \theta^{[q]}(B)\varepsilon_t
    \end{equation}
$$

令$\triangledown = 1 - B$ 则 **ARIMA(p,d,p,q)** 模型表示为:

$$
    \phi^{[p]}(B) \triangledown^{d} x_t = \theta^{[q]}(B)\varepsilon_t
$$

其中

$$
    \begin{equation}
        \begin{aligned}
            \phi^{[p]}(B) = 1 - \phi_1 B - \phi_2 B^2 - …… - \phi_p B^p, \\
            \theta^{[q]}(B) = 1 - \theta_1 B - \theta_2 B^2 - …… - \theta_p B^p
        \end{aligned}
    \end{equation}
$$

如果要建模的时间序列是非平稳的，并且具有周期为S的周期性分量，使用季节差分算子$\triangledown_{S} = 1 - B^{S}$，可得到$SARIMA(p,d,q)(P,D,Q)_S$的模型如下

$$
    \phi^{[p]}(B) \Phi^{[P]}(B^S) \triangledown^{d} \triangledown^D_S x_t = \theta^{[q]}(B) \Theta^{[Q]}(B^S) \varepsilon_t
$$

SARIMA流量预测流程图

![](image/20230901202459.png)

***2.预测流程***

**A.时间序列分解**

将时间序列$x_t$分解为$m_t$趋势分量；$s_t$周期分量；$r_t$随机分量

**B.时间序列聚类**

K-means、层级聚类。分类基于序列的形状、特征。常用距离为ED、DTW。

**C.确定模型阶数**

模型的阶数常用ACF、PACF来确定。

## 机器学习模型

## 深度学习模型

## 流量预测的应用

**A.基站休眠**

精确的蜂窝流量预测可以为基站设计休眠策略

**B.准入控制**

通过基于对用户流量的预测允许或防止特定用户或应用程序使用网络资源。

**C.资源分配与调度**

通过流量预测实现后续分配或调度。

**D.网络管理**

预测峰值处的流量来部署网络资源

**E.网络切片**

蜂窝流量预测在网络切片中可以最小化资源分配成本、最大化无线电资源利用率和保证网络租户的SLA。

**F.软件定义网络**

例如在控制域中预测每个单元格的流量趋势，在不同小区之间可以平衡通信负载。

**G.边缘计算**

通过预测即将到来的小时的最大、平均和最小流量需求，在MEC环境中设计相应的卸载策略。


## 关于预测模型

无论是LSTM相较于ARIMA提升了预测精度，但是以训练时间和提高预测复杂度为代价。而且跨流量类型预测性能并不理想，需要重新训练预测器。而且先分类后预测的前馈预测思路，它的预测性能依赖于分类器的性能，但是分类器性能限制、加密流量增多可能会导致策略失效。

## 关于对预测结果的应用

通过对用户来包的高峰预测，在一个用户的高峰即将到来之前，填充另外一个用户的缓冲区，来防止Qos降级。

## 几个问题

1. 用户流量数据形式？总流量？分应用流量？
   
2. 如何采集数据集？如何进行流量汇总？

3. 基站端预测一般利用业务的时空特征，用户端预测是否需要利用空间特征还是单纯的时间序列预测？
   
4. 用户流量服是否从何种分布？采集得到的数据能提供什么先验信息？
   
5. 用预测方法数值的？机器学习或者深度神经网络的或者混合的？

6. 预测出来的用户流量下一步应用是什么？

对用户的业务，多小区的干扰管理，基站的干扰规避，基站对用户做一个，基站通过对用户流量的预测，在用户最有可能发包来时做出避让。

channel charting 对H做非线性的降维，图的表征

6. 预测方法的应用场景是什么？

# 新方向 dynamic HBF 

注意和message passing的联系