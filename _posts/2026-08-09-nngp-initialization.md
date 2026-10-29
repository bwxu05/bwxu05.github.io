---
layout: post
title: Neural Network Initializations and NNGP
date: 2026-08-09
description: 无穷宽神经网络的随机初始化：NNGP 的 DMFT 推导
tags: pre-training scaling muP DMFT
categories: muP-Theory DMFT-Tutorial
chart:
  chartjs: true
---

对于大语言模型的预训练，神经网络的初始化不仅关系到训练是否稳定，更深刻地影响网络的 feature learning 能力。理解初始化的第一步，是回答一个看似朴素的问题：**一个随机初始化的神经网络，究竟是一个什么样的函数？**

由于权重是随机的，网络输出 $f(x)$ 对每个输入 $x$ 都是一个随机变量；参数的分布于是诱导出一个**函数空间上的分布**。经典结论是：在合适的初始化标度下，当隐藏层宽度趋于无穷时，这个函数分布收敛到一个高斯过程 (Gaussian process, GP)，其协方差核称为 **Neural Network Gaussian Process (NNGP) kernel** [<a href="#ref1">1</a>, <a href="#ref2">2</a>, <a href="#ref3">3</a>]。

Lee et al. [<a href="#ref3">3</a>] 用逐层中心极限定理 (CLT) 给出了经典推导。本文采用统计物理中更“平均场”的视角：把随机初始化的权重视为 **quenched disorder**（淬火无序），用处理高维无序系统的 Dynamical Mean-Field Theory (DMFT) 对无序求平均，再在 $N\to\infty$ 时作鞍点近似，得到单个神经元的有效理论。这一推导与 Pehlevan–Bordelon 讲义 [<a href="#ref2">2</a>] Appendix A 的思路一致。

严格地说，NNGP 只涉及初始化，并没有真实的时间变量；它可以看作完整训练动力学 DMFT 在 $t=0$ 时的“静态平均场”结果。之所以用 DMFT 这套看起来有些“重”的工具来推导一个静态结论，是因为同一套工具（MSR 共轭场、序参量、鞍点）在研究训练动力学时可以推广到 feature learning 场景。

**记号约定.**

- $\mu,\nu\in\lbrace 1,\ldots,P\rbrace$：输入（样本）下标；$i,j,k\in\lbrace 1,\ldots,N\rbrace$：神经元下标；$\ell$：层下标。
- $h$：preactivation；$\phi$：逐元素作用的非线性激活函数。
- $\Phi^{(\ell)}$：第 $\ell$ 层的特征核 (feature kernel)；$K^{(\ell)}$：第 $\ell$ 层的 preactivation 核。
- $\hat h$：与 $h$ 共轭的 MSR 场 (response field)；$\xi$：source 场；$\hat\Phi$：与 $\Phi$ 共轭的序参量。

---

## 1. 初始化标度：为什么是 1/√N？

### 1.1 网络与记号
{:data-toc-skip=''}

考虑一个全连接网络，输入维度为 $D$，所有隐藏层宽度均为 $N$。暂时忽略 bias（第 4 节再加上）：

$$
h_{\mu i}^{(1)}
=
\frac{1}{\sqrt D}\sum_{j=1}^{D}W_{ij}^{(1)}x_{\mu j},
$$

$$
h_{\mu i}^{(\ell)}
=
\frac{1}{\sqrt N}\sum_{j=1}^{N}
W_{ij}^{(\ell)}\phi\!\left(h_{\mu j}^{(\ell-1)}\right),
\qquad \ell\ge 2,
$$

其中各层权重相互独立，且

$$
W_{ij}^{(\ell)}\overset{\mathrm{i.i.d.}}{\sim}\mathcal N(0,1).
$$

为了让第一层与后续各层写成统一形式，约定

$$
\phi\bigl(h_{\mu j}^{(0)}\bigr)\equiv x_{\mu j},
\qquad
N_0\equiv D,
\qquad
N_\ell\equiv N\ (\ell\ge 1),
$$

于是对所有 $\ell\ge 1$ 都有

$$
h_{\mu i}^{(\ell)}
=
\frac{1}{\sqrt{N_{\ell-1}}}\sum_{j=1}^{N_{\ell-1}}
W_{ij}^{(\ell)}\phi\!\left(h_{\mu j}^{(\ell-1)}\right).
$$

### 1.2 标度分析
{:data-toc-skip=''}

为什么必须是 $N^{-1/2}$？考虑更一般的写法

$$
h_i^{(\ell)}
=
N^{-\alpha_\ell}\sum_{j=1}^{N}
W_{ij}^{(\ell)}\phi(h_j^{(\ell-1)}),
\qquad
\operatorname{Var}(W_{ij}^{(\ell)})=N^{-\beta_\ell}.
$$

关键观察是：$W^{(\ell)}$ 与第 $\ell-1$ 层的全部变量**独立**（后者只依赖 $W^{(1)},\ldots,W^{(\ell-1)}$）。因此以第 $\ell-1$ 层为条件，$h_i^{(\ell)}$ 是 $N$ 个独立零均值项之和：

$$
\mathbb E\bigl[h_i^{(\ell)}\,\big\vert\,\text{layer }\ell-1\bigr]=0,
$$

$$
\operatorname{Var}\bigl(h_i^{(\ell)}\,\big\vert\,\text{layer }\ell-1\bigr)
=
N^{-2\alpha_\ell}\sum_{j=1}^{N}N^{-\beta_\ell}\,\phi(h_j^{(\ell-1)})^2
=
N^{1-2\alpha_\ell-\beta_\ell}\cdot
\underbrace{\frac1N\sum_{j=1}^{N}\phi(h_j^{(\ell-1)})^2}_{O(1)}.
$$

若上一层 activation 的经验二阶矩是 $O(1)$，则

$$
\operatorname{Var}(h_i^{(\ell)})
\sim
N^{1-2\alpha_\ell-\beta_\ell}.
$$

分三种情形：

- $2\alpha_\ell+\beta_\ell>1$：preactivation 随 $N\to\infty$ 坍缩到 0，信号消失；
- $2\alpha_\ell+\beta_\ell<1$：preactivation 发散；对 ReLU 这类正齐次激活，经过 $L$ 层后方差被放大约 $N^{L(1-2\alpha-\beta)}$ 倍；
- $2\alpha_\ell+\beta_\ell=1$：方差保持 $O(1)$，且非退化。

因此有限、非退化的 preactivation 要求 [<a href="#ref2">2</a>]

$$
2\alpha_\ell+\beta_\ell=1.
$$

常见的两种写法其实等价：

$$
\frac{1}{\sqrt N}W,
\quad W_{ij}\sim\mathcal N(0,1)
\qquad(\alpha=\tfrac12,\ \beta=0),
$$

或者不显式写出前因子，而直接取

$$
W_{ij}\sim\mathcal N\!\left(0,\frac{1}{N}\right)
\qquad(\alpha=0,\ \beta=1).
$$

这就是 Xavier / He 初始化 [<a href="#ref6">6</a>, <a href="#ref7">7</a>] 中 fan-in scaling 的“平均场”本质：

> **每个神经元接收 $N$ 个独立的随机贡献，因此每一项必须缩放到 $N^{-1/2}$ 量级，才能使总方差保持 $O(1)$。**

对第一层同理，fan-in 是 $D$；这还隐含要求输入满足 $\lVert x\rVert^2/D=O(1)$，即输入的每个坐标是 $O(1)$ 量级。

> **注（参数化）.** 满足 $2\alpha+\beta=1$ 的不同选择在**初始化时**给出完全相同的分布，但它们在训练时并不等价：同样的学习率下，参数梯度的标度依赖于 $\alpha$ 与 $\beta$ 的分配。standard parameterization、NTK parameterization 与 muP 的区别正在于此 [<a href="#ref8">8</a>]。本文只讨论初始化，因此可以放心地取 $\alpha=\tfrac12,\ \beta=0$。

---

## 2. 无穷宽极限：先看结论

### 2.1 特征核
{:data-toc-skip=''}

对一组有限输入 $\lbrace x_\mu\rbrace_{\mu=1}^{P}$，定义第 $\ell$ 层的经验特征核 (empirical feature kernel)

$$
\Phi_{\mu\nu}^{(\ell)}
\equiv
\frac{1}{N}\sum_{i=1}^{N}
\phi(h_{\mu i}^{(\ell)})
\phi(h_{\nu i}^{(\ell)}),
$$

以及输入层核

$$
\Phi_{\mu\nu}^{(0)}
\equiv
\frac{1}{D}x_\mu^\top x_\nu.
$$

$\Phi^{(\ell)}$ 是一个 $P\times P$ 矩阵，它是对全部 $N$ 个神经元求平均的结果——类似于统计物理中的磁化强度，是一个**宏观量**。每个神经元的 preactivation 则记为 $P$ 维向量

$$
\mathbf h_i^{(\ell)}
=
\bigl(h_{1i}^{(\ell)},\ldots,h_{Pi}^{(\ell)}\bigr)^\top.
$$

### 2.2 主要结论
{:data-toc-skip=''}

> **命题（NNGP 极限）.** 固定输入个数 $P$ 与深度 $L$，并假设 $\phi$ 在高斯测度下有有限的二阶矩（例如 $\phi$ 至多多项式增长）。令 $N\to\infty$，则对每一层 $\ell$：
>
> 1. **自平均 (self-averaging)**：经验特征核 $\Phi^{(\ell)}$ 收敛到一个确定性矩阵；
> 2. **高斯性与独立性**：任意有限个神经元的 $\mathbf h_i^{(\ell)}$ 渐近独立同分布，且 $\mathbf h_i^{(\ell)}\sim\mathcal N(0,\Phi^{(\ell-1)})$；
> 3. **确定性递推**：极限核满足下面的递推关系。

$$
\Phi_{\mu\nu}^{(\ell)}
=
\mathbb E_{\mathbf h\sim\mathcal N(0,\Phi^{(\ell-1)})}
\left[
\phi(h_\mu)\phi(h_\nu)
\right],
\qquad
\Phi_{\mu\nu}^{(0)}=\frac{x_\mu^\top x_\nu}{D}.
$$

这已经是 NNGP 核的递推形式。

### 2.3 启发式理由：条件高斯 + 大数定律
{:data-toc-skip=''}

在进入 DMFT 之前，先看一个几乎能直接给出结论的观察。以第 $\ell-1$ 层为条件，

$$
h_{\mu i}^{(\ell)}
=
\frac{1}{\sqrt N}\sum_{j}W_{ij}^{(\ell)}\phi(h_{\mu j}^{(\ell-1)})
$$

是独立高斯变量 $W_{ij}^{(\ell)}$ 以**固定系数**作的线性组合，因此它**精确地**是高斯的（不需要 CLT，对任意 $N$ 都成立）。其条件协方差为

$$
\mathbb E\bigl[h_{\mu i}^{(\ell)}h_{\nu k}^{(\ell)}\,\big\vert\,\text{layer }\ell-1\bigr]
=
\frac1N\sum_{j,j'}
\underbrace{\mathbb E\bigl[W_{ij}^{(\ell)}W_{kj'}^{(\ell)}\bigr]}_{\delta_{ik}\delta_{jj'}}
\phi(h_{\mu j}^{(\ell-1)})\phi(h_{\nu j'}^{(\ell-1)})
=
\delta_{ik}\,\Phi_{\mu\nu}^{(\ell-1)}.
$$

也就是说：

$$
\mathbf h_i^{(\ell)}\,\big\vert\,\text{layer }\ell-1
\;\overset{\mathrm{i.i.d.}}{\sim}\;
\mathcal N\bigl(0,\Phi^{(\ell-1)}\bigr),
\qquad i=1,\ldots,N.
$$

于是唯一剩下的随机性来自经验核 $\Phi^{(\ell-1)}$ 本身。而

$$
\Phi^{(\ell)}=\frac1N\sum_{i=1}^{N}\phi(\mathbf h_i^{(\ell)})\phi(\mathbf h_i^{(\ell)})^\top
$$

是 $N$ 个（条件）独立同分布项的平均，由大数定律，它收敛到 $\mathbb E[\phi(\mathbf h)\phi(\mathbf h)^\top]$，涨落为 $O(N^{-1/2})$。从 $\Phi^{(0)}$（确定性）出发逐层归纳，就得到第 2.2 节的命题。

这个观察揭示了有限宽网络的结构：**它是一个“协方差本身随机”的高斯模型**，有限宽下的非高斯性完全来自 $\Phi$ 的涨落。下面的 DMFT 推导把这件事形式化：对 $W$ 求平均会精确地产生 $\Phi$（第 3.3 节），而鞍点近似实现了大数定律（第 3.5 节）。

---

## 3. NNGP 的 DMFT 推导

### 3.1 生成泛函
{:data-toc-skip=''}

我们希望得到所有 preactivation 的联合分布。对每个 $(\ell,\mu,i)$ 引入 source 场 $\xi_{\mu i}^{(\ell)}$，定义生成泛函（即特征函数）

$$
Z[\xi]
=
\mathbb E_{W}
\exp\left(
 i\sum_{\ell=1}^{L}
 \sum_{\mu=1}^{P}
 \sum_{i=1}^{N}
 \xi_{\mu i}^{(\ell)}
 h_{\mu i}^{(\ell)}
\right).
$$

对 source 求导即得各阶矩，例如

$$
\mathbb E\bigl[h_{\mu i}^{(\ell)}h_{\nu k}^{(\ell')}\bigr]
=
-\frac{\partial^2 Z}{\partial\xi_{\mu i}^{(\ell)}\,\partial\xi_{\nu k}^{(\ell')}}\Bigg\vert_{\xi=0}.
$$

特征函数唯一确定分布，所以只要在 $N\to\infty$ 时求出 $Z[\xi]$，就确定了 preactivation 的全部有限维分布。注意任意有限阶矩只涉及有限个非零的 $\xi_{\mu i}^{(\ell)}$——这一点在第 3.6 节很关键。

### 3.2 用 δ 函数施加前向约束
{:data-toc-skip=''}

前向传播是确定性约束：$h^{(\ell)}$ 是 $W^{(\ell)}$ 与 $h^{(\ell-1)}$ 的函数。困难在于 $h$ 通过一长串非线性复合依赖于 $W$，无法直接对 $W$ 求平均。标准做法是把 $h$ 提升为独立的积分变量，用 δ 函数施加约束。对每个 $(\ell,\mu,i)$ 插入

$$
1=
\int dh_{\mu i}^{(\ell)}
\,\delta\!\left(
h_{\mu i}^{(\ell)}
-
\frac{1}{\sqrt{N_{\ell-1}}}
\sum_j W_{ij}^{(\ell)}
\phi(h_{\mu j}^{(\ell-1)})
\right),
$$

并使用 Fourier 表示

$$
\delta(y)
=
\int\frac{d\hat y}{2\pi}
\exp\!\left(i\hat y\, y\right).
$$

每个约束因此引入一个共轭场 $\hat h_{\mu i}^{(\ell)}$，即 Martin–Siggia–Rose (MSR) 形式中的 response field [<a href="#ref9">9</a>]。记积分测度

$$
\mathcal Dh\,\mathcal D\hat h
\equiv
\prod_{\ell=1}^{L}\prod_{\mu=1}^{P}\prod_{i=1}^{N}
\frac{dh_{\mu i}^{(\ell)}\,d\hat h_{\mu i}^{(\ell)}}{2\pi},
$$

生成泛函可以精确地改写为

$$
\begin{aligned}
Z[\xi]
=\;&\mathbb E_{W}\int\mathcal Dh\,\mathcal D\hat h\;
\exp\Bigg[
i\sum_{\ell,\mu,i}\bigl(\hat h_{\mu i}^{(\ell)}+\xi_{\mu i}^{(\ell)}\bigr)h_{\mu i}^{(\ell)}
\\
&\qquad\qquad
-\,i\sum_{\ell=1}^{L}\frac{1}{\sqrt{N_{\ell-1}}}\sum_{\mu,i,j}
\hat h_{\mu i}^{(\ell)}W_{ij}^{(\ell)}\phi\bigl(h_{\mu j}^{(\ell-1)}\bigr)
\Bigg].
\end{aligned}
$$

这一改写的价值在于：$h$ 现在是**积分变量**，不再是 $W$ 的函数；而 $W$ **只线性地**出现在指数中。因此可以交换次序，先对 $W$ 求平均。

### 3.3 对随机初始化求平均
{:data-toc-skip=''}

固定 $\ell$，记

$$
a_{ij}^{(\ell)}
\equiv
\frac{1}{\sqrt{N_{\ell-1}}}\sum_{\mu}\hat h_{\mu i}^{(\ell)}\phi\bigl(h_{\mu j}^{(\ell-1)}\bigr),
$$

则与 $W^{(\ell)}$ 有关的因子是 $\exp\bigl(-i\sum_{ij}W_{ij}^{(\ell)}a_{ij}^{(\ell)}\bigr)$。由于 $W_{ij}^{(\ell)}$ 是 i.i.d. 标准高斯，可以逐元素精确积分，利用

$$
\mathbb E_{W\sim\mathcal N(0,1)}e^{-iWa}
=e^{-a^2/2},
$$

得到

$$
\mathbb E_{W^{(\ell)}}\exp\Bigl(-i\sum_{i,j}W_{ij}^{(\ell)}a_{ij}^{(\ell)}\Bigr)
=
\exp\Bigl(-\frac12\sum_{i,j}\bigl(a_{ij}^{(\ell)}\bigr)^2\Bigr).
$$

展开平方并交换求和次序：

$$
\sum_{i,j}\bigl(a_{ij}^{(\ell)}\bigr)^2
=
\sum_{\mu,\nu}
\Bigl(\sum_{i}\hat h_{\mu i}^{(\ell)}\hat h_{\nu i}^{(\ell)}\Bigr)
\underbrace{\frac{1}{N_{\ell-1}}\sum_{j}\phi\bigl(h_{\mu j}^{(\ell-1)}\bigr)\phi\bigl(h_{\nu j}^{(\ell-1)}\bigr)}_{=\ \Phi_{\mu\nu}^{(\ell-1)}}.
$$

于是

$$
Z[\xi]
=
\int\mathcal Dh\,\mathcal D\hat h\;
\exp\Bigg[
-\frac12\sum_{\ell=1}^{L}\sum_{\mu,\nu}\Phi_{\mu\nu}^{(\ell-1)}\sum_{i}\hat h_{\mu i}^{(\ell)}\hat h_{\nu i}^{(\ell)}
+i\sum_{\ell,\mu,i}\bigl(\hat h_{\mu i}^{(\ell)}+\xi_{\mu i}^{(\ell)}\bigr)h_{\mu i}^{(\ell)}
\Bigg],
$$

其中 $\Phi^{(0)}$ 是固定的输入核，而 $\Phi^{(\ell-1)}$（$\ell\ge 2$）是第 $\ell-1$ 层 $h$ 的函数。

这是整个推导中最关键的一步。原来 $N^2$ 个随机权重的全部影响，在统计平均之后被压缩成一个 $P\times P$ 的宏观对象 $\Phi^{(\ell-1)}$。换句话说，**对无序取平均之后，第 $\ell$ 层的每个神经元只通过特征核“看到”上一层**，而不再关心上一层任何单个神经元的取值——这正是“平均场”一词的含义。

### 3.4 引入序参量：神经元之间的解耦
{:data-toc-skip=''}

上式仍未在神经元之间解耦：$\Phi^{(\ell-1)}$ 是第 $\ell-1$ 层**所有**神经元的平均，把它们耦合在一起。处理办法是把 $\Phi^{(\ell)}$ 提升为独立的积分变量（**序参量**），再用 δ 函数施加它的定义。对 $\ell=1,\ldots,L-1$ 和每个 $(\mu,\nu)$，利用

$$
\delta\Bigl(\Phi-\frac{s}{N}\Bigr)=N\,\delta(N\Phi-s),
\qquad
\delta(y)=\int\frac{d\hat\Phi}{4\pi}\,e^{\frac i2\hat\Phi y},
$$

插入单位分解

$$
1
=
\frac{N}{4\pi}\int d\Phi_{\mu\nu}^{(\ell)}
\,d\hat\Phi_{\mu\nu}^{(\ell)}
\exp\left\{
\frac{i}{2}
\hat\Phi_{\mu\nu}^{(\ell)}
\left[
N\Phi_{\mu\nu}^{(\ell)}
-
\sum_{i=1}^{N}
\phi(h_{\mu i}^{(\ell)})
\phi(h_{\nu i}^{(\ell)})
\right]
\right\}.
$$

（$\Phi^{(L)}$ 不出现在作用量中，无需引入；为统一记号，约定 $\hat\Phi^{(L)}\equiv 0$。）有了 δ 函数，就可以把指数中 $\hat h\hat h$ 项里作为 $h$ 函数的 $\Phi^{(\ell-1)}$ 换成积分变量 $\Phi^{(\ell-1)}$。

此时逐一检查每个微观变量出现在哪里：

- $h_{\mu i}^{(\ell)}$ 只出现在 $i(\hat h_{\mu i}^{(\ell)}+\xi_{\mu i}^{(\ell)})h_{\mu i}^{(\ell)}$ 和 $-\frac i2\sum_{\mu\nu}\hat\Phi_{\mu\nu}^{(\ell)}\phi(h_{\mu i}^{(\ell)})\phi(h_{\nu i}^{(\ell)})$ 中；
- $\hat h_{\mu i}^{(\ell)}$ 只出现在 $-\frac12\sum_{\mu\nu}\Phi_{\mu\nu}^{(\ell-1)}\hat h_{\mu i}^{(\ell)}\hat h_{\nu i}^{(\ell)}$ 和 $i\hat h_{\mu i}^{(\ell)}h_{\mu i}^{(\ell)}$ 中。

每一项都只涉及**同一个** $(\ell,i)$。也就是说，给定序参量 $(\Phi,\hat\Phi)$ 之后，被积函数对神经元 $i$ 和层 $\ell$ 完全分解。于是生成泛函具有标准的平均场形式 [<a href="#ref2">2</a>]

$$
Z[\xi]
\propto
\int \mathcal D\Phi\,\mathcal D\hat\Phi\;
\exp\bigl(
N\,S[\Phi,\hat\Phi;\xi]
\bigr),
$$

其中作用量为

$$
S
=
\frac{i}{2}
\sum_{\ell=1}^{L-1}\sum_{\mu,\nu}
\hat\Phi_{\mu\nu}^{(\ell)}
\Phi_{\mu\nu}^{(\ell)}
+
\frac1N
\sum_{\ell=1}^{L}\sum_{i=1}^{N}
\log z_i^{(\ell)},
$$

$z_i^{(\ell)}=z_i^{(\ell)}\bigl[\Phi^{(\ell-1)},\hat\Phi^{(\ell)};\xi_i^{(\ell)}\bigr]$ 是单点 (single-site) 配分函数：

$$
\begin{aligned}
z_i^{(\ell)}
=&
\int
\prod_{\mu}
\frac{dh_{\mu}\,d\hat h_{\mu}}{2\pi}
\exp\Bigg[
-
\frac12
\sum_{\mu,\nu}
\hat h_\mu
\Phi_{\mu\nu}^{(\ell-1)}
\hat h_\nu
\\
&\qquad
+i\sum_\mu
\bigl(\hat h_\mu+\xi_{\mu i}^{(\ell)}\bigr)h_\mu
-
\frac{i}{2}
\sum_{\mu,\nu}
\hat\Phi_{\mu\nu}^{(\ell)}
\phi(h_\mu)\phi(h_\nu)
\Bigg].
\end{aligned}
$$

相应地，定义单点平均

$$
\bigl\langle O\bigr\rangle_{i,\ell}
\equiv
\frac{1}{z_i^{(\ell)}}
\int\prod_{\mu}\frac{dh_{\mu}\,d\hat h_{\mu}}{2\pi}\;
O(h,\hat h)\;e^{(\cdots)},
$$

其中 $e^{(\cdots)}$ 即 $z_i^{(\ell)}$ 中的被积函数。

几点说明：

- 在 $\xi=0$ 时所有 $z_i^{(\ell)}$ 相同，$\frac1N\sum_i\log z_i^{(\ell)}=\log z^{(\ell)}$ 与 $N$ 无关，因此 $S=O(1)$，而 $N$ 只作为整体前因子出现——这正是可以使用鞍点近似的大偏差形式。
- 原问题有 $O(LN^2)$ 个随机权重和 $O(LNP)$ 个微观变量；现在只剩 $O(LP^2)$ 个宏观序参量需要积分。**这一步就是从 $N$ 体问题到单体有效理论的核心解耦。**

### 3.5 鞍点方程
{:data-toc-skip=''}

由于整个指数正比于 $N$，

$$
N\to\infty
\quad\Longrightarrow\quad
Z[\xi]
\asymp
\exp\bigl(
N S[\Phi_\star,\hat\Phi_\star;\xi]
\bigr),
$$

其中 $(\Phi_\star,\hat\Phi_\star)$ 是 $S$ 的驻点（被积函数是复的，这里的“鞍点”是最速下降意义下的）。

**对 $\hat\Phi^{(\ell)}$ 变分.** $\hat\Phi^{(\ell)}$ 出现在 $S$ 的第一项和 $z_i^{(\ell)}$ 中，且 $\partial\log z_i^{(\ell)}/\partial\hat\Phi_{\mu\nu}^{(\ell)}=-\frac i2\langle\phi(h_\mu)\phi(h_\nu)\rangle_{i,\ell}$，故

$$
\frac{\partial S}{\partial\hat\Phi_{\mu\nu}^{(\ell)}}
=
\frac i2\Phi_{\mu\nu}^{(\ell)}
-\frac i2\cdot\frac1N\sum_{i}\bigl\langle\phi(h_\mu)\phi(h_\nu)\bigr\rangle_{i,\ell}
=0.
$$

**对 $\Phi^{(\ell)}$ 变分.** $\Phi^{(\ell)}$ 出现在 $S$ 的第一项和**下一层**的 $z_i^{(\ell+1)}$ 中，且 $\partial\log z_i^{(\ell+1)}/\partial\Phi_{\mu\nu}^{(\ell)}=-\frac12\langle\hat h_\mu\hat h_\nu\rangle_{i,\ell+1}$，故

$$
\frac{\partial S}{\partial\Phi_{\mu\nu}^{(\ell)}}
=
\frac i2\hat\Phi_{\mu\nu}^{(\ell)}
-\frac12\cdot\frac1N\sum_{i}\bigl\langle\hat h_\mu\hat h_\nu\bigr\rangle_{i,\ell+1}
=0.
$$

整理得到一对鞍点方程：

$$
\Phi_{\mu\nu}^{(\ell)}
=
\frac1N\sum_{i=1}^{N}
\bigl\langle
\phi(h_{\mu})
\phi(h_{\nu})
\bigr\rangle_{i,\ell},
\qquad
\hat\Phi_{\mu\nu}^{(\ell)}
=
-\frac iN\sum_{i=1}^{N}
\bigl\langle
\hat h_{\mu}
\hat h_{\nu}
\bigr\rangle_{i,\ell+1}.
$$

第一个方程是**前向的自洽条件**：宏观特征核等于单点有效理论中的平均，这就是大数定律的场论表述。第二个方程是**反向的**：共轭核 $\hat\Phi^{(\ell)}$ 由下一层 response field 的关联决定。在训练动力学的 DMFT 中，梯度与响应信息正是沿这条通道反向传播的；而在初始化时，我们将看到这条通道是“空”的。

### 3.6 为什么共轭核 Φ̂ 为零
{:data-toc-skip=''}

**引理.** 若 $\hat\Phi^{(\ell)}=0$，则

$$
z_i^{(\ell)}=\exp\Bigl(-\frac12\sum_{\mu,\nu}\xi_{\mu i}^{(\ell)}\Phi_{\mu\nu}^{(\ell-1)}\xi_{\nu i}^{(\ell)}\Bigr),
\qquad
\bigl\langle\hat h_\mu\hat h_\nu\bigr\rangle_{i,\ell}=\xi_{\mu i}^{(\ell)}\xi_{\nu i}^{(\ell)}.
$$

*证明.* 当 $\hat\Phi^{(\ell)}=0$ 时，$h_\mu$ 只线性地出现在指数中，对它积分给出

$$
\int\frac{dh_\mu}{2\pi}\,e^{i(\hat h_\mu+\xi_{\mu i}^{(\ell)})h_\mu}
=
\delta\bigl(\hat h_\mu+\xi_{\mu i}^{(\ell)}\bigr).
$$

于是 $\hat h$ 被钉在 $-\xi_i^{(\ell)}$ 上，代入 $-\frac12\hat h^\top\Phi^{(\ell-1)}\hat h$ 即得 $z_i^{(\ell)}$，同时 $\langle\hat h_\mu\hat h_\nu\rangle_{i,\ell}=\xi_{\mu i}^{(\ell)}\xi_{\nu i}^{(\ell)}$。$\square$

**从最后一层向前归纳.** 由约定 $\hat\Phi^{(L)}\equiv0$，对第 $L$ 层用引理，再代入第二个鞍点方程：

$$
\hat\Phi_{\mu\nu}^{(L-1)}
=
-\frac iN\sum_{i=1}^{N}\xi_{\mu i}^{(L)}\xi_{\nu i}^{(L)}.
$$

- 在 $\xi=0$ 时，它**精确为零**；
- 计算任意有限阶矩时，只有 $O(1)$ 个神经元带有非零的 $O(1)$ source，因此右边是 $O(1/N)$，在 $N\to\infty$ 时趋于零。

于是 $\hat\Phi^{(L-1)}=0$（领头阶），再对第 $L-1$ 层用引理，得 $\hat\Phi^{(L-2)}=0$……一直归纳到 $\ell=1$。结论是 [<a href="#ref2">2</a>]

$$
\hat\Phi_{\mu\nu}^{(\ell)}=0,
\qquad \ell=1,\ldots,L.
$$

$\hat\Phi$ 的 $O(1/N)$ 偏移对 $\log Z$ 的贡献会因 $S$ 在鞍点处的驻定性而相互抵消，因此不影响领头阶结果。直观上：只有 $O(1)$ 个神经元被 source 直接扰动，其余 $N-O(1)$ 个“体相”神经元在零 source 下完全相同，鞍点由体相决定，source 不会在领头阶改变宏观核。这也是 MSR 形式的一般性质：在零 source 下，response field 的关联 $\langle\hat h\hat h\rangle$ 为零，它保证了归一化 $Z[0]=1$。

### 3.7 单点有效理论：高斯过程
{:data-toc-skip=''}

令 $\hat\Phi^{(\ell)}=0$，在 $z_i^{(\ell)}$ 中先对 $\hat h$ 作高斯积分（设 $\Phi^{(\ell-1)}$ 正定）：

$$
\int\prod_{\mu}\frac{d\hat h_\mu}{2\pi}
\exp\Bigl(-\frac12\hat{\mathbf h}^\top\Phi^{(\ell-1)}\hat{\mathbf h}+i\hat{\mathbf h}^\top\mathbf h\Bigr)
=
\frac{1}{(2\pi)^{P/2}\sqrt{\det\Phi^{(\ell-1)}}}
\exp\Bigl(-\frac12\mathbf h^\top\bigl(\Phi^{(\ell-1)}\bigr)^{-1}\mathbf h\Bigr),
$$

于是单点配分函数化为

$$
\begin{aligned}
z_i^{(\ell)}
&=
\int
\frac{d^P\mathbf h}
{(2\pi)^{P/2}\sqrt{\det\Phi^{(\ell-1)}}}
\exp\left[
-
\frac12
\mathbf h^\top
(\Phi^{(\ell-1)})^{-1}
\mathbf h
+i\boldsymbol\xi_i^{(\ell)\top}\mathbf h
\right]
\\
&=
\exp\Bigl(-\frac12\boldsymbol\xi_i^{(\ell)\top}\Phi^{(\ell-1)}\boldsymbol\xi_i^{(\ell)}\Bigr),
\end{aligned}
$$

其中 $\boldsymbol\xi_i^{(\ell)}=(\xi_{1i}^{(\ell)},\ldots,\xi_{Pi}^{(\ell)})^\top$。这正是多元高斯分布 $\mathcal N(0,\Phi^{(\ell-1)})$ 的特征函数，与 3.6 节引理一致（若 $\Phi^{(\ell-1)}$ 奇异，例如有重复输入，上式第二行仍然成立，对应退化高斯分布）。

在鞍点上，$\hat\Phi=0$ 使 $S$ 的第一项消失，未被 source 扰动的神经元贡献 $\log z=0$。因此

$$
Z[\xi]
\;\xrightarrow{N\to\infty}\;
\prod_{\ell=1}^{L}\prod_{i=1}^{N}
\exp\Bigl(-\frac12\boldsymbol\xi_i^{(\ell)\top}\Phi^{(\ell-1)}\boldsymbol\xi_i^{(\ell)}\Bigr),
$$

并自动满足 $Z[0]=1$。从这一乘积形式可以直接读出：

1. 每个神经元的 preactivation 向量是高斯的：$\mathbf h_i^{(\ell)}\sim\mathcal N(0,\Phi^{(\ell-1)})$；
2. 不同神经元 $i$ 之间相互独立（乘积对 $i$ 分解）；
3. 同一神经元在不同层的 preactivation 也相互独立（乘积对 $\ell$ 分解）——第 $\ell+1$ 层只通过 $\Phi^{(\ell)}$ 依赖第 $\ell$ 层，与任何单个神经元的关联都是 $O(N^{-1/2})$。

最后，单点平均 $\langle\cdot\rangle_{i,\ell}$ 对所有神经元相同，就是在 $\mathcal N(0,\Phi^{(\ell-1)})$ 下的期望。代回第一个鞍点方程，得到

$$
\boxed{
\Phi_{\mu\nu}^{(\ell)}
=
\mathbb E_{\mathbf h\sim\mathcal N(0,\Phi^{(\ell-1)})}
\left[
\phi(h_\mu)\phi(h_\nu)
\right],
\qquad
\Phi_{\mu\nu}^{(0)}=\frac{x_\mu^\top x_\nu}{D}.
}
$$

这就是 NNGP 的递推。

### 3.8 小结：DMFT 与 CLT 的对应
{:data-toc-skip=''}

把 DMFT 推导中的每一步与第 2.3 节的启发式论证对照：

- **对 $W$ 求平均**（3.3 节）$\;\longleftrightarrow\;$ 以上一层为条件时 preactivation 精确高斯，协方差为 $\Phi^{(\ell-1)}$；
- **引入序参量并解耦**（3.4 节）$\;\longleftrightarrow\;$ 以 $\Phi$ 为条件时，各神经元条件独立；
- **鞍点近似**（3.5 节）$\;\longleftrightarrow\;$ 大数定律：$\Phi$ 自平均到确定值；
- **鞍点附近的二阶涨落** $\;\longleftrightarrow\;$ 有限宽修正：$\Phi$ 的涨落为 $O(N^{-1/2})$，由此诱导的核修正与非高斯性（如四阶连通关联）为 $O(1/N)$，参见 [<a href="#ref10">10</a>]。

Lee et al. [<a href="#ref3">3</a>] 的 CLT 推导与上面的 DMFT 推导得到完全相同的递推：CLT 逐层证明高斯性；DMFT 则把高斯性、自平均性以及（下一步的）有限宽修正统一在同一个鞍点展开的框架中。

---

## 4. 回到标准 NNGP 核

Lee et al. [<a href="#ref3">3</a>] 更常用的参数化带有权重尺度 $\sigma_w$ 和 bias 尺度 $\sigma_b$：

$$
h_i^{(\ell)}(x)
=
\frac{\sigma_w}{\sqrt{N_{\ell-1}}}
\sum_{j=1}^{N_{\ell-1}}
W_{ij}^{(\ell)}
\phi(h_j^{(\ell-1)}(x))
+
\sigma_b b_i^{(\ell)},
\qquad
W_{ij}^{(\ell)},\,b_i^{(\ell)}\overset{\mathrm{i.i.d.}}{\sim}\mathcal N(0,1).
$$

**在 DMFT 中加入 bias.** 3.2 节的推导完全照搬，只是指数中多出一项 $-i\sigma_b\sum_{\mu,i}\hat h_{\mu i}^{(\ell)}b_i^{(\ell)}$。注意 bias 对所有输入 $\mu$ 共享。对 $b_i^{(\ell)}$ 求平均：

$$
\mathbb E_{b}\exp\Bigl(-i\sigma_b b_i^{(\ell)}\sum_\mu\hat h_{\mu i}^{(\ell)}\Bigr)
=
\exp\Bigl(-\frac{\sigma_b^2}{2}\sum_{\mu,\nu}\hat h_{\mu i}^{(\ell)}\hat h_{\nu i}^{(\ell)}\Bigr),
$$

而 $\sigma_w$ 使 3.3 节的结果乘上 $\sigma_w^2$。两者合并，单点理论中的 $\Phi^{(\ell-1)}$ 被替换为

$$
K_{\mu\nu}^{(\ell)}
=
\sigma_b^2+\sigma_w^2\,\Phi_{\mu\nu}^{(\ell-1)}.
$$

其余推导不变，因此 $\mathbf h_i^{(\ell)}\sim\mathcal N(0,K^{(\ell)})$。$K^{(\ell)}$ 正是 preactivation 协方差

$$
K^{(\ell)}(x,x')
=
\mathbb E\big[h_i^{(\ell)}(x)h_i^{(\ell)}(x')\big].
$$

（无 bias、$\sigma_w=1$ 时 $K^{(\ell)}=\Phi^{(\ell-1)}$，回到第 3 节。）

**标准递推.** 第一层为

$$
K^{(1)}(x,x')
=
\sigma_b^2
+
\sigma_w^2\frac{x^\top x'}{D},
$$

后续各层满足确定性迭代 [<a href="#ref3">3</a>]

$$
K^{(\ell)}(x,x')
=
\sigma_b^2
+
\sigma_w^2\,
\mathbb E_{(u,v)\sim\mathcal N(0,\Sigma^{(\ell-1)})}
[\phi(u)\phi(v)],
$$

其中

$$
\Sigma^{(\ell-1)}
=
\begin{pmatrix}
K^{(\ell-1)}(x,x) & K^{(\ell-1)}(x,x')\\
K^{(\ell-1)}(x',x) & K^{(\ell-1)}(x',x')
\end{pmatrix}.
$$

注意递推只需要两两输入的 $2\times 2$ 协方差：因为 $\mathbb E[\phi(h_\mu)\phi(h_\nu)]$ 只依赖 $(h_\mu,h_\nu)$ 的联合分布。

**输出层.** 设网络有 $L$ 个隐藏层，输出为线性 readout

$$
f(x)
=
\frac{\sigma_w}{\sqrt N}\sum_{j=1}^{N}W_j^{(L+1)}\phi\bigl(h_j^{(L)}(x)\bigr)+\sigma_b b^{(L+1)}.
$$

它与第 $L+1$ 层的 preactivation 形式相同，因此对任意有限输入集合 $x^1,\ldots,x^P$，$(f(x^1),\ldots,f(x^P))$ 都是多元高斯的。按 Kolmogorov 相容性，这意味着在函数空间中

$$
f(\cdot)\sim\mathrm{GP}\bigl(0,K^{(L+1)}\bigr).
$$

---

## 5. 初始化即核设计

NNGP 让随机初始化的作用变得非常直接：$\sigma_w$、$\sigma_b$ 和非线性 $\phi$ 共同决定了一个核映射

$$
K^{(\ell-1)}
\mapsto
K^{(\ell)}.
$$

因此初始化不仅影响梯度的尺度，还决定了网络在训练之前偏好的函数空间。下面看两个可以精确计算的例子。

### 5.1 线性激活
{:data-toc-skip=''}

若 $\phi(u)=u$ 且 $\sigma_b=0$，则 $\mathbb E[uv]=K^{(\ell-1)}(x,x')$，递推变成

$$
K^{(\ell)}=\sigma_w^2K^{(\ell-1)}
\quad\Longrightarrow\quad
K^{(\ell)}(x,x')=\bigl(\sigma_w^2\bigr)^{\ell}\,\frac{x^\top x'}{D}.
$$

要在深度增加时保持方差尺度（既不指数爆炸也不指数衰减），需要

$$
\sigma_w^2=1,
$$

即等宽情形下的 Xavier-like initialization [<a href="#ref6">6</a>]。同时可以看到：线性网络的 NNGP 核永远是线性核，深度只改变整体尺度，不改变函数先验的“形状”。

### 5.2 ReLU：对角元与 He 初始化
{:data-toc-skip=''}

若 $\phi(u)=\max(0,u)$，记对角元

$$
q_\ell=K^{(\ell)}(x,x).
$$

对 $u\sim\mathcal N(0,q_{\ell-1})$，由于 $u$ 关于 0 对称，而 $\operatorname{ReLU}(u)^2=u^2\mathbf 1\lbrace u>0\rbrace$，

$$
\mathbb E[\operatorname{ReLU}(u)^2]
=
\frac12\mathbb E[u^2]
=
\frac{q_{\ell-1}}{2}.
$$

因此在 $\sigma_b=0$ 时

$$
q_\ell
=
\frac{\sigma_w^2}{2}q_{\ell-1}.
$$

保持方差的条件是

$$
\sigma_w^2=2,
$$

也就是

$$
\operatorname{Var}(W_{ij}^{\mathrm{effective}})
=\frac{2}{N}.
$$

这正是 He initialization [<a href="#ref7">7</a>] 的平均场解释：ReLU 把一半的输入置零，从而“丢掉”一半方差，权重方差需要加倍来补偿。

### 5.3 ReLU：非对角元与 arc-cosine 核
{:data-toc-skip=''}

ReLU 的非对角核也有闭式解。令

$$
\cos\theta
=
\frac{K(x,x')}
{\sqrt{K(x,x)K(x',x')}},
\qquad \theta\in[0,\pi],
$$

则 [<a href="#ref3">3</a>, <a href="#ref4">4</a>]

$$
\mathbb E[\operatorname{ReLU}(u)\operatorname{ReLU}(v)]
=
\frac{\sqrt{K(x,x)K(x',x')}}{2\pi}
\left[
\sin\theta+(\pi-\theta)\cos\theta
\right].
$$

*推导.* 取独立标准高斯 $z_1,z_2$，令

$$
u=\sqrt{K(x,x)}\,z_1,
\qquad
v=\sqrt{K(x',x')}\,\bigl(\cos\theta\,z_1+\sin\theta\,z_2\bigr),
$$

容易验证 $(u,v)$ 有正确的协方差。由于 ReLU 是正齐次的，$\operatorname{ReLU}(az)=a\operatorname{ReLU}(z)$（$a>0$），只需计算 $\mathbb E[\operatorname{ReLU}(z_1)\operatorname{ReLU}(\cos\theta\,z_1+\sin\theta\,z_2)]$。用极坐标 $z_1=r\cos\psi,\ z_2=r\sin\psi$，则 $\cos\theta\,z_1+\sin\theta\,z_2=r\cos(\psi-\theta)$，两者同时为正的区域是 $\psi\in(\theta-\tfrac\pi2,\tfrac\pi2)$，长度为 $\pi-\theta$。于是

$$
\mathbb E[\cdots]
=
\underbrace{\int_0^\infty\frac{r^3e^{-r^2/2}}{2\pi}\,dr}_{=1/\pi}
\;\cdot
\int_{\theta-\pi/2}^{\pi/2}\cos\psi\cos(\psi-\theta)\,d\psi.
$$

利用 $\cos\psi\cos(\psi-\theta)=\frac12\bigl[\cos(2\psi-\theta)+\cos\theta\bigr]$，角向积分为

$$
\frac12\Bigl[\frac{\sin(2\psi-\theta)}{2}\Bigr]_{\theta-\pi/2}^{\pi/2}+\frac12(\pi-\theta)\cos\theta
=
\frac12\bigl[\sin\theta+(\pi-\theta)\cos\theta\bigr].
$$

乘以 $1/\pi$ 即得结论。作为检验，$\theta=0$ 时结果为 $K(x,x)/2$，与 5.2 节一致。$\square$

因此一个随机初始化的无穷宽 ReLU 网络对应一个完全确定的、逐层复合的 arc-cosine kernel。

### 5.4 深度极限：相关性的演化
{:data-toc-skip=''}

在 He 初始化（$\sigma_w^2=2,\ \sigma_b=0$）下，对角元保持不变 $q_\ell=q$，因此只需追踪两个输入之间的相关系数 $c_\ell=K^{(\ell)}(x,x')/q$。由 5.3 节，

$$
c_\ell=f(c_{\ell-1}),
\qquad
f(c)=\frac{1}{\pi}\Bigl[\sqrt{1-c^2}+(\pi-\arccos c)\,c\Bigr].
$$

这个映射有几个简单性质：

- $f(1)=1$：$c=1$ 是不动点；
- 令 $c=\cos\theta$，则 $f(c)-c=(\sin\theta-\theta\cos\theta)/\pi\ge 0$，所以 $c_\ell$ 单调递增并收敛到 1；例如两个正交输入经过一层后相关系数变为 $f(0)=1/\pi$；
- $f'(c)=(\pi-\arccos c)/\pi$，故 $f'(1)=1$：不动点是**临界的 (marginal)**，收敛比指数慢。

也就是说，在足够深的 ReLU 网络中，任意两个输入在初始化时都会变得越来越“相似”，NNGP 核逐渐退化，丢失输入的几何信息；由于 $f'(1)=1$，这种退化只以多项式速度发生（而非指数速度），这正是深层 ReLU 网络仍然可训练的原因之一。

注意：对无 bias 的 ReLU，由于正齐次性，相关系数映射 $f$ 与 $\sigma_w$ **无关**——$\sigma_w$ 只控制对角元 $q_\ell$ 的尺度。He 初始化的作用是让 $q_\ell$ 既不爆炸也不消失，从而使上面的相关性递推成为对网络的忠实描述。

对一般的 $\phi$，不动点 $c=1$ 处的斜率 $\chi_1=\sigma_w^2\,\mathbb E_{u\sim\mathcal N(0,q^\ast)}[\phi'(u)^2]$ 区分了有序相（$\chi_1<1$，相关性指数地趋于 1）与混沌相（$\chi_1>1$，相近的输入被指数地分开），$\chi_1=1$ 称为 edge of chaos [<a href="#ref11">11</a>, <a href="#ref12">12</a>]。对 ReLU 有 $\mathbb E[\phi'(u)^2]=\tfrac12$，所以 He 初始化恰好位于 $\chi_1=1$。这就是“初始化即核设计”的具体含义：选择 $(\sigma_w,\sigma_b,\phi)$，就是在选择一个随深度演化的核，以及信号能够有效传播的深度尺度。

---

## 6. NNGP 意味着什么

NNGP 描述的是**随机初始化所诱导的函数先验**，而不是一般的 feature-learning 训练动力学。

- **NNGP**：$f(\cdot)$ 在初始化时的分布；核心对象是协方差核 $K^{(\ell)}$。Lee et al. [<a href="#ref3">3</a>] 直接用 $K^{(L+1)}$ 做 GP 回归，得到无穷宽贝叶斯神经网络的精确后验预测。
- **NTK**：参数雅可比矩阵的 Gram 核，在 lazy / kernel regime 中控制梯度流下的预测动力学 [<a href="#ref5">5</a>]。它由 NNGP 核与导数核递推构造：$\Theta^{(1)}=K^{(1)}$，以及 $\Theta^{(\ell+1)}=K^{(\ell+1)}+\sigma_w^2\,\mathbb E_{(u,v)\sim\mathcal N(0,\Sigma^{(\ell)})}[\phi'(u)\phi'(v)]\,\Theta^{(\ell)}$。
- **Feature-learning DMFT**：核本身随训练演化，需要含时的序参量，例如 $\Phi_{\mu\nu}^{(\ell)}(t,s)$、梯度核以及 response functions [<a href="#ref2">2</a>, <a href="#ref13">13</a>]。此时 3.5 节中的共轭核 $\hat\Phi$ 不再为零——反向传播的信息正是通过它进入有效理论。

因此，NNGP 可以看成完整训练动力学 DMFT 最简单的边界条件：在 $t=0$，随机高维网络已经被压缩成一个确定性的核迭代；训练开始后，如果 features 发生 $O(1)$ 的演化（例如在 muP 下 [<a href="#ref8">8</a>]），就必须把这个“静态”平均场理论扩展为真正的动力学平均场理论。

---

## References

<style>
  ol.references { padding-left: 1.8em; }
  ol.references li { margin-bottom: 0.5em; scroll-margin-top: 5rem; }
  ol.references li:target { outline: 1px dashed var(--global-theme-color, #b509ac); outline-offset: 3px; border-radius: 3px; }
</style>

<ol class="references">
  <li id="ref1">Radford M. Neal. <em>Priors for Infinite Networks</em>. In <em>Bayesian Learning for Neural Networks</em>, Lecture Notes in Statistics, vol. 118, Springer, 1996.</li>
  <li id="ref2">Cengiz Pehlevan and Blake Bordelon. <em>Lecture Notes on Infinite-Width Limits of Neural Networks</em>. Princeton Machine Learning Summer School, 2023.</li>
  <li id="ref3">Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, and Jascha Sohl-Dickstein. <em>Deep Neural Networks as Gaussian Processes</em>. ICLR, 2018. <a href="https://arxiv.org/abs/1711.00165">arXiv:1711.00165</a>.</li>
  <li id="ref4">Youngmin Cho and Lawrence K. Saul. <em>Kernel Methods for Deep Learning</em>. NeurIPS, 2009.</li>
  <li id="ref5">Arthur Jacot, Franck Gabriel, and Clément Hongler. <em>Neural Tangent Kernel: Convergence and Generalization in Neural Networks</em>. NeurIPS, 2018. <a href="https://arxiv.org/abs/1806.07572">arXiv:1806.07572</a>.</li>
  <li id="ref6">Xavier Glorot and Yoshua Bengio. <em>Understanding the Difficulty of Training Deep Feedforward Neural Networks</em>. AISTATS, 2010.</li>
  <li id="ref7">Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. <em>Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification</em>. ICCV, 2015. <a href="https://arxiv.org/abs/1502.01852">arXiv:1502.01852</a>.</li>
  <li id="ref8">Greg Yang and Edward J. Hu. <em>Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks</em>. ICML, 2021. <a href="https://arxiv.org/abs/2011.14522">arXiv:2011.14522</a>.</li>
  <li id="ref9">Paul C. Martin, Eric D. Siggia, and Harvey A. Rose. <em>Statistical Dynamics of Classical Systems</em>. Physical Review A, 8(1):423, 1973.</li>
  <li id="ref10">Daniel A. Roberts, Sho Yaida, and Boris Hanin. <em>The Principles of Deep Learning Theory</em>. Cambridge University Press, 2022. <a href="https://arxiv.org/abs/2106.10165">arXiv:2106.10165</a>.</li>
  <li id="ref11">Ben Poole, Subhaneil Lahiri, Maithra Raghu, Jascha Sohl-Dickstein, and Surya Ganguli. <em>Exponential Expressivity in Deep Neural Networks through Transient Chaos</em>. NeurIPS, 2016. <a href="https://arxiv.org/abs/1606.05340">arXiv:1606.05340</a>.</li>
  <li id="ref12">Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli, and Jascha Sohl-Dickstein. <em>Deep Information Propagation</em>. ICLR, 2017. <a href="https://arxiv.org/abs/1611.01232">arXiv:1611.01232</a>.</li>
  <li id="ref13">Blake Bordelon and Cengiz Pehlevan. <em>Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks</em>. NeurIPS, 2022. <a href="https://arxiv.org/abs/2205.09653">arXiv:2205.09653</a>.</li>
</ol>
