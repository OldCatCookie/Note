# traffic prediction

## 工作流程与模型分类

**四种工作流程：** 直接预测、分类然后预测、分解然后预测、聚类然后预测。

**三种模型分类:** 统计模型、机器学习模型、深度学习模型

## 统计模型

***A.ARMA模型***

ARMA自回归移动平均模型是平稳时间序列建模中常用方法：

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

![](image/20230901202459.png)

## 机器学习

## 深度学习模型

## 流量预测的应用