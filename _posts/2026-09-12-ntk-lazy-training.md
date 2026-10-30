---
layout: post
title: Neural Tangent Kernel and Lazy Training
date: 2026-09-12
description: 无穷宽神经网络的训练动力学：NTK 的 DMFT 推导
tags: pre-training scaling muP DMFT NTK
categories: muP-Theory DMFT-Tutorial
chart:
  chartjs: true
---

[上一篇]({% post_url 2026-08-09-nngp-initialization %}) 讨论了随机初始化：在无穷宽极限下，每个神经元的 preactivation 是高斯的，整个网络在初始化时是一个以 NNGP 核为协方差的高斯过程。那篇文章回答的是“训练开始之前，网络是一个什么样的函数”。本文接着回答下一个问题：**当所有层的参数都参与梯度下降时，网络的预测如何随时间演化？**

经典结论来自 Jacot et al. [<a href="#ref1">1</a>]：在 NTK parameterization 下，无穷宽网络的训练动力学由一个确定性的、在训练中保持不变的核——Neural Tangent Kernel (NTK)——完全控制，网络在函数空间中做的是以 NTK 为核的核回归梯度流 [<a href="#ref1">1</a>, <a href="#ref5">5</a>]。本文沿用上一篇的 DMFT 语言推导这一结论：先写出有限宽下精确成立的 NTK 分解，再对随机初始化求无序平均，得到前向与反向两组序参量的单点有效理论，最后说明为什么在这一标度下核会在训练中“冻结”。

与上一篇相比，新的要素是**反向传播**。计算梯度需要用到权重矩阵的转置 $W^\top$，同一个随机矩阵因此在前向与反向中被使用了两次。对无序求平均之后，这种“复用”会产生一类新的**响应 (response) 序参量**。我们将看到：在初始化时它们为零，这正是前向场与反向场可以视为相互独立的原因；而在 feature learning 标度下，它们不再能被忽略。

**记号约定.**

- $\mu,\nu\in\lbrace 1,\ldots,P\rbrace$：训练样本下标；$i,j\in\lbrace 1,\ldots,N\rbrace$：神经元下标；$\ell$：层下标；$t,s$：训练时间。
- $h$：preactivation；$z$：pre-gradient；$g=\phi'(h)z$：反向场（缩放后的梯度）。
- $\Phi^{(\ell)}$：前向特征核；$G^{(\ell)}$：反向梯度核；$\Psi^{(\ell)}$：导数核；$\Theta$：NTK。带下标 $N$ 的表示有限宽的经验量，不带下标的表示 $N\to\infty$ 的确定性极限。
- $\chi,\xi$：由初始化权重产生的前向、反向无序场；$\hat h,\hat z$：MSR 共轭场；$b,c$：source 场（上一篇把 source 记为 $\xi$，本文把 $\xi$ 留给反向无序场）。
- $\odot$：逐元素 (Hadamard) 乘积；$\mathbf 1$：全 1 向量。

---

## 1. 神经网络与训练设置

### 1.1 网络架构
{:data-toc-skip=''}

考虑 $L$ 个隐藏层、宽度均为 $N$、无 bias 的全连接网络。输入维度为 $D$，训练集为 $\lbrace(x_\mu,y_\mu)\rbrace_{\mu=1}^{P}$：

$$
\begin{aligned}
h_{\mu i}^{(1)}(t)
&=\frac{1}{\sqrt D}\sum_{j=1}^{D}W_{ij}^{(1)}(t)\,x_{\mu j},\\
h_{\mu i}^{(\ell)}(t)
&=\frac{1}{\sqrt N}\sum_{j=1}^{N}W_{ij}^{(\ell)}(t)\,\phi\bigl(h_{\mu j}^{(\ell-1)}(t)\bigr),
\qquad 2\leq\ell\leq L,\\
f_\mu(t)
&=\frac{1}{\sqrt N}\sum_{i=1}^{N} a_i(t)\,\phi\bigl(h_{\mu i}^{(L)}(t)\bigr).
\end{aligned}
$$

与上一篇相比，这里显式写出了线性 readout 层 $a$，并且所有参数都随训练时间 $t$ 变化。记全部参数为

$$
\theta=\bigl(W^{(1)},\ldots,W^{(L)},a\bigr).
$$

### 1.2 NTK 参数化与梯度流
{:data-toc-skip=''}

初始化时 $W_{ij}^{(\ell)}(0)$ 与 $a_i(0)$ 相互独立，且都服从 $\mathcal N(0,1)$。优化变量是这些 **raw parameters**，而 $D^{-1/2}$、$N^{-1/2}$ 是写在前向计算里的显式因子。对 raw parameters 使用与宽度无关的统一学习率，就是本文所说的 NTK parameterization [<a href="#ref1">1</a>, <a href="#ref5">5</a>]。

> **注（参数化与学习率）.** 上一篇第 1 节指出，满足 $2\alpha+\beta=1$ 的不同写法在**初始化时**等价。训练时则不然。例如 standard parameterization 把隐藏层写成 $h=\sum_jW'_{ij}\phi_j$，取 $W'_{ij}=W_{ij}/\sqrt N\sim\mathcal N(0,1/N)$。由链式法则 $\partial\mathcal L/\partial W'=\sqrt N\,\partial\mathcal L/\partial W$，所以对 $W'$ 用学习率 $\eta$ 做梯度下降，等价于对 raw parameter $W$ 用学习率 $N\eta$。因此改变参数坐标而保持数值相同的学习率，一般会改变训练动力学。本文始终在 raw parameter 坐标下讨论，学习率 $\eta$ 与宽度无关。

采用平均平方损失与 full-batch gradient flow。记残差 $\Delta_\mu=y_\mu-f_\mu$，则

$$
\mathcal L=\frac{1}{2P}\sum_{\mu=1}^{P}\Delta_\mu^2,
\qquad
\nabla_\theta\mathcal L=-\frac1P\sum_\mu\Delta_\mu\nabla_\theta f_\mu,
$$

于是

$$
\frac{d\theta}{dt}
=-\eta\nabla_\theta\mathcal L
=\frac{\eta}{P}\sum_{\mu=1}^{P}\Delta_\mu\nabla_\theta f_\mu.
$$

### 1.3 极限的顺序
{:data-toc-skip=''}

以下固定样本数 $P$、输入维度 $D$、深度 $L$、学习率 $\eta$ 与有限时间区间 $[0,T]$，再取 $N\to\infty$。时间区间必须先固定：许多结论在 $t\sim N^{c}$ 的时间尺度上会失效。

讨论训练动力学时，我们主要考虑具有适当光滑性和矩控制的激活函数，例如 $\tanh$。对 ReLU，初始化时的核公式依然成立（第 5.1 节会用到），但严格处理训练极限还需要控制 activation gate 在训练中的切换 [<a href="#ref5">5</a>, <a href="#ref7">7</a>]。

---

## 2. 有限宽 NTK

### 2.1 预测动力学
{:data-toc-skip=''}

由链式法则，对每个训练样本有

$$
\frac{df_\mu}{dt}
=\nabla_\theta f_\mu\cdot\frac{d\theta}{dt}
=\frac{\eta}{P}\sum_{\nu=1}^{P}
\underbrace{\nabla_\theta f_\mu\cdot\nabla_\theta f_\nu}_{\equiv\ \Theta_{N,\mu\nu}(t)}\,\Delta_\nu.
$$

于是得到精确的预测动力学 [<a href="#ref1">1</a>]：

$$
\Theta_{N,\mu\nu}(t)
=\nabla_\theta f_\mu(t)\cdot\nabla_\theta f_\nu(t),
\qquad
\frac{d\boldsymbol\Delta}{dt}
=-\frac{\eta}{P}\,\Theta_N(t)\,\boldsymbol\Delta.
$$

记 parameter Jacobian $J\in\mathbb R^{P\times\dim\theta}$，其第 $\mu$ 行为 $\nabla_\theta f_\mu^\top$，则 $\Theta_N=JJ^\top$ 是 Gram 矩阵，因而半正定。由此可知损失单调不增：

$$
\frac{d\mathcal L}{dt}
=\frac1P\boldsymbol\Delta^\top\frac{d\boldsymbol\Delta}{dt}
=-\frac{\eta}{P^2}\boldsymbol\Delta^\top\Theta_N\boldsymbol\Delta\leq0.
$$

到这里还没有做任何近似，这是有限宽网络的精确方程。难点在于 $\Theta_N(t)$ 本身依赖于 $\theta(t)$，所以这个方程一般是非线性的，核会随训练变化。

### 2.2 反向场
{:data-toc-skip=''}

为了分解 $\Theta_N$，需要输出对每个 preactivation 的导数。由于 $f$ 是 $N$ 个神经元贡献之和、每项带 $N^{-1/2}$，单个导数 $\partial f_\mu/\partial h_{\mu i}^{(\ell)}$ 的量级是 $N^{-1/2}$。乘以 $\sqrt N$ 后定义 $O(1)$ 的 backward field：

$$
g_{\mu i}^{(\ell)}
\equiv\sqrt N\,\frac{\partial f_\mu}{\partial h_{\mu i}^{(\ell)}}.
$$

**最后一层.** 直接求导：

$$
g_{\mu i}^{(L)}
=\sqrt N\cdot\frac{1}{\sqrt N}a_i\,\phi'\bigl(h_{\mu i}^{(L)}\bigr)
=\phi'\bigl(h_{\mu i}^{(L)}\bigr)\,a_i.
$$

**中间层.** $h_{\mu i}^{(\ell)}$ 通过所有 $h_{\mu j}^{(\ell+1)}$ 影响输出，且 $\partial h_{\mu j}^{(\ell+1)}/\partial h_{\mu i}^{(\ell)}=N^{-1/2}W_{ji}^{(\ell+1)}\phi'(h_{\mu i}^{(\ell)})$，于是

$$
g_{\mu i}^{(\ell)}
=\sqrt N\sum_j\frac{g_{\mu j}^{(\ell+1)}}{\sqrt N}\cdot\frac{W_{ji}^{(\ell+1)}}{\sqrt N}\phi'\bigl(h_{\mu i}^{(\ell)}\bigr)
=\phi'\bigl(h_{\mu i}^{(\ell)}\bigr)\,z_{\mu i}^{(\ell)}.
$$

两种情形统一写成

$$
g_{\mu i}^{(\ell)}=\phi'\bigl(h_{\mu i}^{(\ell)}\bigr)\,z_{\mu i}^{(\ell)},
\qquad
z_{\mu i}^{(\ell)}
=\frac{1}{\sqrt N}\sum_jW_{ji}^{(\ell+1)}g_{\mu j}^{(\ell+1)}\ \ (\ell<L),
\qquad
z_{\mu i}^{(L)}=a_i.
$$

$z$ 称为 pre-gradient。它的形式与前向的 $h$ 完全对称：$h$ 是 $W$ 作用在 $\phi(h)$ 上，$z$ 是 $W^\top$ 作用在 $g$ 上。

相应地定义前向与反向的两时间序参量 [<a href="#ref2">2</a>, <a href="#ref3">3</a>]：

$$
\begin{aligned}
\Phi_{N,\mu\nu}^{(\ell)}(t,s)
&=\frac1N\sum_{i=1}^{N}\phi\bigl(h_{\mu i}^{(\ell)}(t)\bigr)\phi\bigl(h_{\nu i}^{(\ell)}(s)\bigr),
\qquad
\Phi_{\mu\nu}^{(0)}=\frac{x_\mu^\top x_\nu}{D},\\
G_{N,\mu\nu}^{(\ell)}(t,s)
&=\frac1N\sum_{i=1}^{N}g_{\mu i}^{(\ell)}(t)\,g_{\nu i}^{(\ell)}(s).
\end{aligned}
$$

### 2.3 按参数块分解 NTK
{:data-toc-skip=''}

$\Theta_N$ 是对所有参数求和，可以按参数块拆开。先计算每一块的导数。

- **Readout：**

$$
\frac{\partial f_\mu}{\partial a_i}=\frac{1}{\sqrt N}\phi\bigl(h_{\mu i}^{(L)}\bigr).
$$

- **隐藏层 $\ell\ge2$：** 由 $\partial h_{\mu i}^{(\ell)}/\partial W_{ij}^{(\ell)}=N^{-1/2}\phi(h_{\mu j}^{(\ell-1)})$ 与 $g$ 的定义，

$$
\frac{\partial f_\mu}{\partial W_{ij}^{(\ell)}}
=\frac{g_{\mu i}^{(\ell)}}{\sqrt N}\cdot\frac{\phi\bigl(h_{\mu j}^{(\ell-1)}\bigr)}{\sqrt N}
=\frac1N\,g_{\mu i}^{(\ell)}\phi\bigl(h_{\mu j}^{(\ell-1)}\bigr).
$$

- **第一层：** 同理，

$$
\frac{\partial f_\mu}{\partial W_{ij}^{(1)}}=\frac{1}{\sqrt{ND}}\,g_{\mu i}^{(1)}x_{\mu j}.
$$

对每一块求内积。以隐藏层为例，求和对 $i$ 与 $j$ 分解：

$$
\sum_{i,j}\frac{\partial f_\mu}{\partial W_{ij}^{(\ell)}}\frac{\partial f_\nu}{\partial W_{ij}^{(\ell)}}
=\Bigl(\frac1N\sum_ig_{\mu i}^{(\ell)}g_{\nu i}^{(\ell)}\Bigr)
\Bigl(\frac1N\sum_j\phi\bigl(h_{\mu j}^{(\ell-1)}\bigr)\phi\bigl(h_{\nu j}^{(\ell-1)}\bigr)\Bigr)
=G_{N,\mu\nu}^{(\ell)}\,\Phi_{N,\mu\nu}^{(\ell-1)}.
$$

第一层与 readout 同理。把各块加起来，得到有限宽下的恒等式：

$$
\Theta_N(t)
=\underbrace{\Phi_N^{(L)}(t,t)}_{\text{readout}}
+\sum_{\ell=1}^{L}
\underbrace{\Phi_N^{(\ell-1)}(t,t)\odot G_N^{(\ell)}(t,t)}_{W^{(\ell)}},
\qquad \Phi_N^{(0)}\equiv\Phi^{(0)}.
$$

每个参数块的贡献，都是**进入该层的特征重叠** $\Phi^{(\ell-1)}$ 与**从该层到输出的敏感度重叠** $G^{(\ell)}$ 的逐元素乘积。这个恒等式没有用到无穷宽或任何独立性假设。因此只要知道 $N\to\infty$ 时 $\Phi_N$ 与 $G_N$ 的行为，就知道 NTK 的行为。

### 2.4 Scaling：lazy 的直观来源
{:data-toc-skip=''}

在进入 DMFT 之前，先用量级估计看看训练中各个量变化多快。设 $\Delta,\phi,g$ 都是 $O(1)$：

| 量 | 变化速度 | 理由 |
|:--|:--:|:--|
| 单个隐藏层权重 $W_{ij}^{(\ell)}$ | $O(N^{-1})$ | $\partial f/\partial W_{ij}=g_i\phi_j/N$ |
| 单个 readout 权重 $a_i$ | $O(N^{-1/2})$ | $\partial f/\partial a_i=\phi_i/\sqrt N$ |
| 单个 preactivation $h_{\mu i}^{(\ell)}$ | $O(N^{-1/2})$ | $N$ 个 $O(N^{-1})$ 的权重变化，经 $N^{-1/2}$ 相干叠加 |
| 核 $\Phi_N,G_N,\Theta_N$ | 至多 $O(N^{-1/2})$ | 单个场的变化通过平均传到核（实际常为 $O(N^{-1})$，见第 4.2 节） |
| 输出 $f_\mu$ | $O(1)$ | $N$ 个 $O(N^{-1/2})$ 的贡献经 $N^{-1/2}$ **相干**叠加 |

第三行的计算是：$\sum_j(dW_{ij}/dt)\phi_j/\sqrt N\sim N\cdot N^{-1}\cdot N^{-1/2}=N^{-1/2}$，因为同一误差信号驱动的各项符号一致，求和是相干的。最后一行类似：$N$ 个神经元各变化 $O(N^{-1/2})$，经过 readout 的 $N^{-1/2}$ 相干求和后得到 $O(1)$。

这张表就是 lazy training 的全部直观：**每个神经元几乎不动，但大量几乎不动的神经元相干地叠加出 $O(1)$ 的函数变化。** 下面的 DMFT 推导把第三、四行变成精确的场方程。

---

## 3. NTK 的 DMFT 推导

### 3.1 把训练写成场方程
{:data-toc-skip=''}

对隐藏层权重，梯度流给出

$$
\frac{dW_{ij}^{(\ell)}}{dt}
=\frac{\eta}{P}\sum_\nu\Delta_\nu\frac{\partial f_\nu}{\partial W_{ij}^{(\ell)}}
=\frac{\eta}{PN}\sum_\nu\Delta_\nu\,g_{\nu i}^{(\ell)}\phi\bigl(h_{\nu j}^{(\ell-1)}\bigr).
$$

积分得到

$$
W_{ij}^{(\ell)}(t)
=W_{ij}^{(\ell)}(0)
+\frac{\eta}{PN}\sum_\nu\int_0^tds\,\Delta_\nu(s)\,g_{\nu i}^{(\ell)}(s)\,\phi\bigl(h_{\nu j}^{(\ell-1)}(s)\bigr).
$$

代回前向传播 $h_{\mu i}^{(\ell)}(t)=N^{-1/2}\sum_jW_{ij}^{(\ell)}(t)\phi(h_{\mu j}^{(\ell-1)}(t))$。第二项中对 $j$ 的求和恰好凑出两时间特征核：

$$
\frac{1}{\sqrt N}\cdot\frac{\eta}{PN}\sum_\nu\int_0^tds\,\Delta_\nu(s)g_{\nu i}^{(\ell)}(s)
\underbrace{\sum_j\phi\bigl(h_{\nu j}^{(\ell-1)}(s)\bigr)\phi\bigl(h_{\mu j}^{(\ell-1)}(t)\bigr)}_{=\,N\,\Phi_{N,\mu\nu}^{(\ell-1)}(t,s)}.
$$

对反向传播 $z^{(\ell)}=N^{-1/2}W^{(\ell+1)\top}g^{(\ell+1)}$ 做同样的处理，对 $j$ 的求和凑出 $G^{(\ell+1)}$。于是整个训练过程可以写成 [<a href="#ref2">2</a>, <a href="#ref3">3</a>]

$$
\begin{aligned}
h_{\mu i}^{(\ell)}(t)
={}&\chi_{\mu i}^{(\ell)}(t)
+\frac{\eta}{P\sqrt N}\sum_\nu\int_0^t ds\,
\Delta_\nu(s)\,\Phi_{N,\mu\nu}^{(\ell-1)}(t,s)\,g_{\nu i}^{(\ell)}(s),\\
z_{\mu i}^{(\ell)}(t)
={}&\xi_{\mu i}^{(\ell)}(t)
+\frac{\eta}{P\sqrt N}\sum_\nu\int_0^t ds\,
\Delta_\nu(s)\,G_{N,\mu\nu}^{(\ell+1)}(t,s)\,\phi\bigl(h_{\nu i}^{(\ell)}(s)\bigr),
\end{aligned}
$$

其中**无序场**由初始化权重作用在当前时刻的场上给出：

$$
\chi_{\mu i}^{(\ell)}(t)
=\frac1{\sqrt N}\sum_jW_{ij}^{(\ell)}(0)\,\phi\bigl(h_{\mu j}^{(\ell-1)}(t)\bigr),
\qquad
\xi_{\mu i}^{(\ell)}(t)
=\frac1{\sqrt N}\sum_jW_{ji}^{(\ell+1)}(0)\,g_{\mu j}^{(\ell+1)}(t).
$$

边界情形如下：

- 第一层的 $\chi_{\mu i}^{(1)}=D^{-1/2}\sum_jW_{ij}^{(1)}(0)x_{\mu j}$ 与时间无关，反馈项中的核为 $\Phi^{(0)}$。代入 $dW_{ij}^{(1)}/dt=\frac{\eta}{P\sqrt{ND}}\sum_\nu\Delta_\nu g_{\nu i}^{(1)}x_{\nu j}$ 可直接验证，反馈系数仍为 $\eta/(P\sqrt N)$。
- 输出端 $z^{(L)}=a$，由 $da_i/dt=\frac{\eta}{P\sqrt N}\sum_\nu\Delta_\nu\phi(h_{\nu i}^{(L)})$ 得到 $\xi_{\mu i}^{(L)}=a_i(0)$，相当于取 $G_{N,\mu\nu}^{(L+1)}(t,s)\equiv1$。

注意 $\chi(t)$ 是**初始化权重**作用在**时刻 $t$ 的特征**上，所以它并不自动等于初始化时的 preactivation $h(0)$。

两个反馈项都带有显式的 $N^{-1/2}$，这与第 2.4 节的标度计数一致，并提示存在一个静态的 lazy 极限。但这还不是证明：$\chi(t)$ 本身也会随 $h^{(\ell-1)}(t)$ 变化，而且 $W(0)$ 同时出现在前向与反向中，二者之间存在关联。第 4.2 节会处理这个问题。

**本节余下部分先求初始化（$t=0$，时间变量省略）时前向与反向场的联合单点理论；它在训练中的适用性留到第 4.2 节说明。**

### 3.2 初始化时的生成泛函
{:data-toc-skip=''}

沿用上一篇的 Fourier-source 约定，对前向场和反向场分别引入 source $b,c$，它们只作用在有限个 site 上：

$$
Z[b,c]
=\mathbb E_{W,a}\exp\left\{
i\sum_{\ell,\mu,i}
\left[b_{\mu i}^{(\ell)}h_{\mu i}^{(\ell)}
+c_{\mu i}^{(\ell)}z_{\mu i}^{(\ell)}\right]
\right\}.
$$

与上一篇的关键区别是：内部矩阵 $W^{(\ell)}$（$2\le\ell\le L$）现在需要**同时**施加两个约束，

$$
\mathbf h_\mu^{(\ell)}
=\frac1{\sqrt N}W^{(\ell)}\phi\bigl(\mathbf h_\mu^{(\ell-1)}\bigr),
\qquad
\mathbf z_\mu^{(\ell-1)}
=\frac1{\sqrt N}W^{(\ell)\top}\mathbf g_\mu^{(\ell)}.
$$

用 $\delta(y)=\int d\hat y\,e^{i\hat y y}/(2\pi)$ 为每个约束引入共轭场 $\hat h_{\mu i}^{(\ell)}$ 与 $\hat z_{\mu j}^{(\ell-1)}$。约束 $\hat z_{\mu j}z_{\mu j}$ 中 $z_{\mu j}=N^{-1/2}\sum_iW_{ij}g_{\mu i}$，所以与 $W^{(\ell)}$ 有关的全部指数为

$$
-\frac{i}{\sqrt N}\sum_{i,j}W_{ij}^{(\ell)}\,A_{ij}^{(\ell)},
\qquad
A_{ij}^{(\ell)}
\equiv\sum_\mu\left[
\hat h_{\mu i}^{(\ell)}\phi\bigl(h_{\mu j}^{(\ell-1)}\bigr)
+g_{\mu i}^{(\ell)}\hat z_{\mu j}^{(\ell-1)}
\right].
$$

前向项与反向项乘在**同一个随机权重** $W_{ij}^{(\ell)}$ 上，因此不能预先把它们当作独立的。

第一层 $W^{(1)}$ 只有前向约束（输入不需要反向场）；readout $a$ 只出现在约束 $z^{(L)}=a$ 中。

### 3.3 对随机初始化求平均
{:data-toc-skip=''}

$W$ 只线性地出现在指数中，利用 $\mathbb E_{W\sim\mathcal N(0,1)}e^{-iWA/\sqrt N}=e^{-A^2/(2N)}$ 逐元素积分，得到

$$
\exp\left\{-\frac1{2N}\sum_{i,j}\bigl(A_{ij}^{(\ell)}\bigr)^2\right\}.
$$

把平方展开为三项。

**前向–前向项：**

$$
\frac1N\sum_{i,j}\sum_{\mu,\nu}\hat h_{\mu i}\hat h_{\nu i}\,\phi\bigl(h_{\mu j}^{(\ell-1)}\bigr)\phi\bigl(h_{\nu j}^{(\ell-1)}\bigr)
=\sum_{\mu,\nu}\Phi_{N,\mu\nu}^{(\ell-1)}\,\hat{\mathbf h}_\mu^{(\ell)}\cdot\hat{\mathbf h}_\nu^{(\ell)}.
$$

**反向–反向项：**

$$
\frac1N\sum_{i,j}\sum_{\mu,\nu}g_{\mu i}g_{\nu i}\,\hat z_{\mu j}\hat z_{\nu j}
=\sum_{\mu,\nu}G_{N,\mu\nu}^{(\ell)}\,\hat{\mathbf z}_\mu^{(\ell-1)}\cdot\hat{\mathbf z}_\nu^{(\ell-1)}.
$$

**交叉项**（乘以 2）：

$$
\frac2N\sum_{\mu,\nu}\Bigl(\sum_i\hat h_{\mu i}^{(\ell)}g_{\nu i}^{(\ell)}\Bigr)\Bigl(\sum_j\phi\bigl(h_{\mu j}^{(\ell-1)}\bigr)\hat z_{\nu j}^{(\ell-1)}\Bigr)
=2N\sum_{\mu,\nu}U_{\mu\nu}^{(\ell)}V_{\mu\nu}^{(\ell-1)},
$$

其中出现了两个新的混合序参量：

$$
U_{\mu\nu}^{(\ell)}
=\frac1N\sum_i\hat h_{\mu i}^{(\ell)}g_{\nu i}^{(\ell)},
\qquad
V_{\mu\nu}^{(\ell-1)}
=\frac1N\sum_i\phi\bigl(h_{\mu i}^{(\ell-1)}\bigr)\hat z_{\nu i}^{(\ell-1)}.
$$

合起来，无序平均后的指数为

$$
-\frac12\sum_{\mu,\nu}
\Phi_{N,\mu\nu}^{(\ell-1)}
\hat{\mathbf h}_\mu^{(\ell)}\cdot\hat{\mathbf h}_\nu^{(\ell)}
-\frac12\sum_{\mu,\nu}
G_{N,\mu\nu}^{(\ell)}
\hat{\mathbf z}_\mu^{(\ell-1)}\cdot\hat{\mathbf z}_\nu^{(\ell-1)}
-N\sum_{\mu,\nu}U_{\mu\nu}^{(\ell)}V_{\mu\nu}^{(\ell-1)}.
$$

前两项与上一篇第 3.3 节的结构完全相同，分别给出前向场与反向场的高斯协方差。第三项是新的，它记录了**同一个权重在前向与转置后的反向中被复用**所产生的 response coupling [<a href="#ref3">3</a>]。注意它是 $N\cdot O(1)$ 的作用量项，与前两项同阶：虽然原式中带有 $1/N$，但求和遍历 $N^2$ 对 $(i,j)$，不能仅凭 $1/N$ 就把它丢掉。

**边界.** 第一层的平均只给出前向项，协方差为固定的 $\Phi^{(0)}$。对 readout，约束 $z_{\mu i}^{(L)}=a_i$ 贡献 $e^{-ia_i\sum_\mu\hat z_{\mu i}^{(L)}}$，平均得

$$
\mathbb E_{a_i}e^{-ia_i\sum_\mu\hat z_{\mu i}^{(L)}}
=\exp\Bigl(-\frac12\sum_{\mu,\nu}\hat z_{\mu i}^{(L)}\hat z_{\nu i}^{(L)}\Bigr),
$$

相当于 $G^{(L+1)}=\mathbf1\mathbf1^\top$。它**不是** $I_P$：不同样本共享同一个 $a_i$，所以最后一层的 pre-gradient 在样本之间完全相关。

### 3.4 序参量与单点理论
{:data-toc-skip=''}

和上一篇一样，把 $\Phi,G,U,V$ 提升为独立的积分变量，统记为 $\mathcal Q$，并对每个独立分量插入单位分解

$$
1\propto\int dQ\,d\hat Q\,
\exp\left\{iN\hat Q\left[Q-\frac1N\sum_iO_i\right]\right\}.
$$

第 $\ell$ 层的局部观测量为

$$
\mathcal O^{(\ell)}
=\left\{
\phi(h_\mu)\phi(h_\nu),\;
g_\mu g_\nu,\;
\hat h_\mu g_\nu,\;
\phi(h_\mu)\hat z_\nu
\right\},
\qquad g_\mu=\phi'(h_\mu)z_\mu,
$$

依次对应 $\Phi^{(\ell)},G^{(\ell)},U^{(\ell)},V^{(\ell)}$。其中 $U^{(\ell)}$ 只存在于 $2\leq\ell\leq L$，$V^{(\ell)}$ 只存在于 $1\leq\ell\leq L-1$。下文的 $\hat{\mathcal Q}\cdot\mathcal Q$ 表示对各序参量的独立分量求和，$\Phi,G$ 只计对称矩阵的独立元。

给定序参量后，被积函数对神经元 $i$ 与层 $\ell$ 完全分解（检查方式与上一篇第 3.4 节相同）。于是生成泛函具有标准的鞍点形式：

$$
Z[b,c]\propto\int\mathcal D\mathcal Q\,\mathcal D\hat{\mathcal Q}\,
\exp\left\{N S[\mathcal Q,\hat{\mathcal Q};b,c]\right\},
$$

$$
S
=i\hat{\mathcal Q}\cdot\mathcal Q
-\sum_{\ell=2}^{L}\sum_{\mu,\nu}
U_{\mu\nu}^{(\ell)}V_{\mu\nu}^{(\ell-1)}
+\frac1N\sum_{\ell,i}\log\mathcal Z_i^{(\ell)}.
$$

单点生成泛函同时包含第 $\ell$ 层的前向场 $(h,\hat h)$ 与反向场 $(z,\hat z)$：

$$
\begin{aligned}
\mathcal Z_i^{(\ell)}
={}&\int\frac{d^P\mathbf h\,d^P\hat{\mathbf h}\,
 d^P\mathbf z\,d^P\hat{\mathbf z}}{(2\pi)^{2P}}
\exp\Bigg\{
i\hat{\mathbf h}^{\top}\mathbf h
+i\hat{\mathbf z}^{\top}\mathbf z
-\frac12\hat{\mathbf h}^{\top}\Phi^{(\ell-1)}\hat{\mathbf h}
-\frac12\hat{\mathbf z}^{\top}G^{(\ell+1)}\hat{\mathbf z}\\
&\qquad
-i\hat{\mathcal Q}^{(\ell)}\cdot\mathcal O^{(\ell)}
+i\mathbf b_i^{(\ell)\top}\mathbf h
+i\mathbf c_i^{(\ell)\top}\mathbf z
\Bigg\}.
\end{aligned}
$$

注意指标：第 $\ell$ 层的前向场协方差来自上一层的 $\Phi^{(\ell-1)}$，反向场协方差来自下一层的 $G^{(\ell+1)}$。神经元之间的耦合已经全部转移到宏观序参量上。有限个 source 只带来 $O(1)$ 的修正，领头阶的 bulk 鞍点由零 source 理论决定。

### 3.5 鞍点方程与响应序参量
{:data-toc-skip=''}

$N\to\infty$ 时，考察满足

$$
\frac{\partial S}{\partial\mathcal Q}=0,
\qquad
\frac{\partial S}{\partial\hat{\mathcal Q}}=0
$$

的物理鞍点。

**对共轭序参量变分**，与上一篇完全相同，给出单点自洽条件

$$
\mathcal Q^{(\ell)}=\bigl\langle\mathcal O^{(\ell)}\bigr\rangle,
$$

特别地 $\Phi^{(\ell)}=\langle\phi(\mathbf h)\phi(\mathbf h)^\top\rangle$，$G^{(\ell)}=\langle\mathbf g\mathbf g^\top\rangle$。这里尖括号表示单点有效理论中的平均，而不是对训练样本的平均。

**对 $\Phi,G$ 变分**，给出 $\hat\Phi\propto\langle\hat{\mathbf h}\hat{\mathbf h}^\top\rangle$、$\hat G\propto\langle\hat{\mathbf z}\hat{\mathbf z}^\top\rangle$。与上一篇第 3.6 节同理，在零 source 下单点积分恒归一化为 1、与协方差取值无关，因此 $\hat\Phi=\hat G=0$。

**对 $U,V$ 变分**是新的内容。$S$ 中与它们有关的部分是 $i\hat U\cdot U+i\hat V\cdot V-\sum U V$，于是

$$
i\hat U_{\mu\nu}^{(\ell)}=V_{\mu\nu}^{(\ell-1)},
\qquad
i\hat V_{\mu\nu}^{(\ell-1)}=U_{\mu\nu}^{(\ell)}.
$$

要判断 $U,V$ 是否为零，需要计算它们的自洽值 $U=\langle\hat h g\rangle$、$V=\langle\phi(h)\hat z\rangle$。这里要用到 MSR 形式的一个基本恒等式。

**引理（MSR 分部积分）.** 设零 source，且所有共轭序参量 $\hat{\mathcal Q}=0$，于是单点权重中 $\mathbf h$ 只通过 $e^{i\hat{\mathbf h}^\top\mathbf h}$ 出现、$\mathbf z$ 只通过 $e^{i\hat{\mathbf z}^\top\mathbf z}$ 出现。则对任意（适当增长的）函数 $F(\mathbf h,\mathbf z)$，

$$
\bigl\langle\hat h_\mu F\bigr\rangle=i\Bigl\langle\frac{\partial F}{\partial h_\mu}\Bigr\rangle,
\qquad
\bigl\langle\hat z_\mu F\bigr\rangle=i\Bigl\langle\frac{\partial F}{\partial z_\mu}\Bigr\rangle.
$$

*证明.* 利用 $\hat h_\mu e^{i\hat{\mathbf h}^\top\mathbf h}=-i\,\partial_{h_\mu}e^{i\hat{\mathbf h}^\top\mathbf h}$，代入积分并对 $h_\mu$ 分部积分。由于权重的其余部分不依赖 $\mathbf h$，导数只落在 $F$ 上；边界项为零，符号翻转即得。对 $\hat z$ 同理。$\square$

这正是下面要检验的零响应解所处的情形；若共轭序参量不为零，分部积分还会多出导数作用在权重上的项。

因此 $\hat h$ 与 $\hat z$ 可以理解为“对 $h$、$z$ 的扰动”，混合序参量是**响应函数**：

$$
U_{\mu\nu}^{(\ell)}=i\Bigl\langle\frac{\partial g_\nu}{\partial h_\mu}\Bigr\rangle,
\qquad
V_{\mu\nu}^{(\ell)}=i\Bigl\langle\frac{\partial\phi(h_\mu)}{\partial z_\nu}\Bigr\rangle.
$$

$U$ 度量反向场对前向场扰动的平均响应，$V$ 则相反。

**零响应解.** 尝试 $U=V=\hat U=\hat V=0$。这时单点积分中 $(h,\hat h)$ 与 $(z,\hat z)$ 之间没有任何耦合，分别对 $\hat h,\hat z$ 作高斯积分（同上一篇第 3.7 节）后，

$$
\mathbf h^{(\ell)}=\mathbf u^{(\ell)}\sim\mathcal N\bigl(0,\Phi^{(\ell-1)}\bigr),
\qquad
\mathbf z^{(\ell)}=\mathbf r^{(\ell)}\sim\mathcal N\bigl(0,G^{(\ell+1)}\bigr),
\qquad
\mathbf u^{(\ell)}\perp\mathbf r^{(\ell)}.
$$

在此分布下计算响应：

$$
\Bigl\langle\frac{\partial\phi(u_\mu)}{\partial r_\nu}\Bigr\rangle=0,
\qquad
\Bigl\langle\frac{\partial[\phi'(u_\nu)r_\nu]}{\partial u_\mu}\Bigr\rangle
=\delta_{\mu\nu}\bigl\langle\phi''(u_\mu)\bigr\rangle\bigl\langle r_\mu\bigr\rangle=0.
$$

第一式成立是因为 $\phi(u)$ 不依赖于 $r$；第二式利用了 $u\perp r$ 以及 $r$ 的均值为零，而后者归根结底来自 readout $a$ 的零均值独立初始化。所以 $U=V=0$，再由上面的鞍点方程得 $\hat U=\hat V=0$，零响应解是自洽的。

这正是文献中常用的 **gradient independence assumption**（计算 NTK 时把反向传播中的 $W^\top$ 当作与 $W$ 独立的新矩阵）的场论版本。Yang [<a href="#ref4">4</a>] 证明了：只要 readout 层独立、零均值地初始化，并且没有在前向中被别处复用，这一假设在计算初始化 NTK 时就是严格成立的。

需要强调，这里只是验证了零响应解满足鞍点方程，是一个形式上的自洽检查；它本身并不证明核在训练中冻结（见第 4.2 节）。

### 3.6 静态单点理论：前向核、导数核与反向核
{:data-toc-skip=''}

单点理论最终化为

$$
\begin{gathered}
\mathbf u^{(\ell)}\sim\mathcal N\bigl(0,\Phi^{(\ell-1)}\bigr),
\qquad
\mathbf r^{(\ell)}\sim\mathcal N\bigl(0,G^{(\ell+1)}\bigr),
\qquad \mathbf u^{(\ell)}\perp\mathbf r^{(\ell)},\\
h_\mu^{(\ell)}=u_\mu^{(\ell)},
\qquad
g_\mu^{(\ell)}=\phi'\bigl(u_\mu^{(\ell)}\bigr)\,r_\mu^{(\ell)}.
\end{gathered}
$$

高斯的是 preactivation $h$ 与 pre-gradient $z$，而反向场 $g$ 一般**不是**高斯的：它是高斯变量 $r$ 乘以另一个独立随机变量 $\phi'(u)$。

把单点解代回自洽条件。$\Phi$ 的方程与上一篇的 NNGP 递推完全相同；对 $G$，利用 $u\perp r$ 把期望拆开：

$$
G_{\mu\nu}^{(\ell)}
=\bigl\langle\phi'(u_\mu)r_\mu\,\phi'(u_\nu)r_\nu\bigr\rangle
=\bigl\langle\phi'(u_\mu)\phi'(u_\nu)\bigr\rangle\,\bigl\langle r_\mu r_\nu\bigr\rangle
=\Psi_{\mu\nu}^{(\ell)}\,G_{\mu\nu}^{(\ell+1)},
$$

其中定义了**导数核** $\Psi^{(\ell)}$。总结：

$$
\begin{aligned}
\Phi_{\mu\nu}^{(\ell)}
&=\mathbb E_{\mathbf u\sim\mathcal N(0,\Phi^{(\ell-1)})}
\bigl[\phi(u_\mu)\phi(u_\nu)\bigr],\\
\Psi_{\mu\nu}^{(\ell)}
&=\mathbb E_{\mathbf u\sim\mathcal N(0,\Phi^{(\ell-1)})}
\bigl[\phi'(u_\mu)\phi'(u_\nu)\bigr],\\
G^{(\ell)}
&=\Psi^{(\ell)}\odot G^{(\ell+1)},
\qquad G^{(L+1)}=\mathbf1\mathbf1^\top.
\end{aligned}
$$

计算顺序与反向传播一致：先**前向**从 $\Phi^{(0)}$ 出发算出所有 $\Phi^{(\ell)},\Psi^{(\ell)}$，再**反向**从 $G^{(L+1)}$ 出发算出所有 $G^{(\ell)}$。把反向递推展开，

$$
G^{(\ell)}=\Psi^{(\ell)}\odot\Psi^{(\ell+1)}\odot\cdots\odot\Psi^{(L)}.
$$

这样整个静态单点理论就闭合了 [<a href="#ref1">1</a>, <a href="#ref2">2</a>, <a href="#ref3">3</a>, <a href="#ref4">4</a>]。

---

## 4. 极限 NTK 与 lazy training

### 4.1 极限 NTK
{:data-toc-skip=''}

由自平均，$\Phi_N^{(\ell)}\to\Phi^{(\ell)}$、$G_N^{(\ell)}\to G^{(\ell)}$。代入第 2.3 节的有限宽恒等式，初始化时的 NTK 收敛到确定性极限

$$
\Theta_\infty
=\Phi^{(L)}
+\sum_{\ell=1}^{L}\Phi^{(\ell-1)}\odot G^{(\ell)}
=\Phi^{(L)}
+\sum_{\ell=1}^{L}\Phi^{(\ell-1)}\odot\Psi^{(\ell)}\odot\cdots\odot\Psi^{(L)}.
$$

**前向递推形式.** 定义

$$
\Theta^{(0)}=\Phi^{(0)},
\qquad
\Theta^{(\ell)}=\Phi^{(\ell)}+\Psi^{(\ell)}\odot\Theta^{(\ell-1)},
\qquad
\Theta_\infty=\Theta^{(L)}.
$$

把递推展开即可验证二者一致：

$$
\Theta^{(L)}
=\Phi^{(L)}+\Psi^{(L)}\odot\Phi^{(L-1)}+\Psi^{(L)}\odot\Psi^{(L-1)}\odot\Phi^{(L-2)}+\cdots
+\Psi^{(L)}\odot\cdots\odot\Psi^{(1)}\odot\Phi^{(0)}.
$$

这个递推有清楚的含义 [<a href="#ref1">1</a>, <a href="#ref4">4</a>]：$\Theta^{(\ell)}$ 是“截断在第 $\ell$ 层并接上 readout”的子网络的 NTK。增加一层时，新 readout 贡献 $\Phi^{(\ell)}$，而之前所有参数的贡献都要经过新一层的 $\phi'$ 传回来，于是乘上 $\Psi^{(\ell)}$。

它与上一篇第 6 节给出的递推一致：那里的 preactivation 核满足 $K^{(\ell+1)}=\Phi^{(\ell)}$（取 $\sigma_w=1$、$\sigma_b=0$），两篇的 $\Theta$ 只差一个层指标的平移。

### 4.2 训练中核为何冻结
{:data-toc-skip=''}

到目前为止只证明了**初始化时** $\Theta_N(0)\to\Theta_\infty$。要得到训练中的固定核，还需要说明 $\Theta_N(t)$ 在 $[0,T]$ 上几乎不变。这里给出两个互补的视角。

**视角一：参数空间（Hessian 论证）[<a href="#ref8">8</a>]。** 分两步。

*第一步：参数总位移是 $O(1)$。* 由梯度流与 $\lVert J^\top\boldsymbol\Delta\rVert\le\lambda_{\max}(\Theta_N)^{1/2}\lVert\boldsymbol\Delta\rVert$，

$$
\lVert\theta(t)-\theta(0)\rVert_2
\le\int_0^t\Bigl\lVert\frac{d\theta}{ds}\Bigr\rVert_2ds
\le\frac{\eta}{P}\int_0^t\lambda_{\max}\bigl(\Theta_N(s)\bigr)^{1/2}\lVert\boldsymbol\Delta(s)\rVert\,ds.
$$

由于损失单调下降，$\lVert\boldsymbol\Delta(s)\rVert\le\lVert\boldsymbol\Delta(0)\rVert=O(1)$（初始输出是方差 $O(1)$ 的高斯变量）。只要核在此期间保持 $O(1)$，就有 $\lVert\theta(t)-\theta(0)\rVert_2=O(1)$（“核保持 $O(1)$”与下一步的结论可以用标准的连续性 bootstrap 论证同时闭合）。相比之下 $\lVert\theta(0)\rVert_2\sim N$，所以参数的相对变化是 $O(1/N)$，与第 2.4 节的标度计数一致。

*第二步：Hessian 很小。* 对 NTK parameterization 下的光滑网络，在 $\theta(0)$ 附近半径为 $O(1)$ 的球内，输出关于参数的 Hessian 满足 $\lVert\nabla_\theta^2f_\mu\rVert_{\mathrm{op}}=\tilde O(N^{-1/2})$（$\tilde O$ 表示至多差对数因子）[<a href="#ref8">8</a>]。于是

$$
\bigl\lVert\nabla_\theta f_\mu(\theta(t))-\nabla_\theta f_\mu(\theta(0))\bigr\rVert
\le\sup_{\theta}\lVert\nabla_\theta^2f_\mu\rVert_{\mathrm{op}}\cdot\lVert\theta(t)-\theta(0)\rVert
=\tilde O(N^{-1/2}).
$$

又因为 $\lVert\nabla_\theta f_\mu\rVert^2=\Theta_{N,\mu\mu}=O(1)$，所以

$$
\bigl\lvert\Theta_{N,\mu\nu}(t)-\Theta_{N,\mu\nu}(0)\bigr\rvert=\tilde O(N^{-1/2}).
$$

直观上：参数移动了 $O(1)$ 的距离，但网络在这个尺度上几乎是线性的，所以 tangent 几乎不变。

**视角二：DMFT。** 在第 3.1 节的场方程中，显式反馈项是 $O(N^{-1/2})$。无序场 $\chi(t)$ 的变化来自两部分：上一层特征的 $O(N^{-1/2})$ 变化经 $W(0)$ 传递，以及 $W(0)$ 在前向与反向中复用导致的关联。后者在动力学 DMFT 中正是由响应序参量 $U,V$ 的含时推广来刻画的。Bordelon–Pehlevan [<a href="#ref3">3</a>] 的动力学 DMFT 表明，在 NTK 标度下这些响应项的贡献同样被 $N^{-1/2}$ 压低，动力学理论退化为第 3 节的静态理论。

两个视角给出同一结论。在固定 $P,D,L,T$、适当矩控制、前后向传播稳定的条件下，严格的结果是 [<a href="#ref1">1</a>, <a href="#ref5">5</a>, <a href="#ref7">7</a>]

$$
\sup_{0\leq t\leq T}
\bigl\lVert\Theta_N(t)-\Theta_\infty\bigr\rVert
\xrightarrow[N\to\infty]{\mathbb P}0.
$$

> **注（核变化的真实量级）.** 上面的 $N^{-1/2}$ 只是上界。更细的分析表明，核在有限时间内的实际变化通常是 $O(N^{-1})$ [<a href="#ref9">9</a>, <a href="#ref10">10</a>]。启发式的原因是：单个场的 $O(N^{-1/2})$ 变化正比于反向场 $g_i=\phi'(h_i)z_i$，而 $z_i$ 均值为零且与前向场独立（第 3.5 节）。因此把这些变化平均到核里时，领头阶的 $N^{-1/2}$ 项在神经元之间相消，只留下 $1/N$ 的系统漂移。这一 $1/N$ 修正正是有限宽 NTK 理论（neural tangent hierarchy、Feynman 图展开）的研究对象。

相应地，两时间序参量退化为与时间无关的常数：$\Phi^{(\ell)}(t,s)=\Phi^{(\ell)}$，$G^{(\ell)}(t,s)=G^{(\ell)}$。也就是说，在不同训练时刻起作用的是**同一个**初始化单点场，而不是每个时刻重新采样的白噪声。

### 4.3 平方损失下的精确解
{:data-toc-skip=''}

核冻结后，第 2.1 节的预测方程变成常系数线性 ODE，

$$
\frac{d\boldsymbol\Delta}{dt}=-\frac{\eta}{P}\Theta_\infty\boldsymbol\Delta,
$$

其解为 [<a href="#ref1">1</a>]

$$
\boldsymbol\Delta(t)
=\exp\Bigl(-\frac{\eta t}{P}\Theta_\infty\Bigr)\boldsymbol\Delta(0),
\qquad
\mathbf f(t)=\mathbf y-\boldsymbol\Delta(t).
$$

**谱分解.** 设 $\Theta_\infty v_k=\kappa_kv_k$，$\lbrace v_k\rbrace$ 正交归一，$c_k=v_k^\top\boldsymbol\Delta(0)$，则

$$
\mathcal L(t)
=\frac1{2P}\sum_kc_k^2
\exp\Bigl(-\frac{2\eta\kappa_k}{P}t\Bigr).
$$

每个核特征模态以自己的时间尺度 $P/(\eta\kappa_k)$ 衰减：大特征值方向学得快，小特征值方向学得慢，零特征值方向上的误差永远不会消失。在一定条件下（例如输入两两不平行、$\phi$ 非多项式），$\Theta_\infty$ 严格正定，训练误差指数收敛到零 [<a href="#ref1">1</a>]。

**测试点上的预测.** 对任意输入 $x$，同样的链式法则给出 $df(x)/dt=\frac{\eta}{P}\Theta_\infty(x,X)\boldsymbol\Delta(t)$，其中 $\Theta_\infty(x,X)$ 是 $x$ 与训练集之间的 $1\times P$ 核向量。代入 $\boldsymbol\Delta(t)$ 并对时间积分：

$$
f_t(x)
=f_0(x)+\Theta_\infty(x,X)\,\Theta_\infty^{-1}
\Bigl[I-\exp\Bigl(-\frac{\eta t}{P}\Theta_\infty\Bigr)\Bigr]\bigl(\mathbf y-\mathbf f_0\bigr).
$$

当 $t\to\infty$ 时，

$$
f_\infty(x)=f_0(x)+\Theta_\infty(x,X)\,\Theta_\infty^{-1}\bigl(\mathbf y-\mathbf f_0\bigr),
$$

这正是以 NTK 为核的 kernel regression（无正则），只是多了一个由随机初始函数 $f_0$ 带来的偏移 [<a href="#ref1">1</a>, <a href="#ref5">5</a>]。

**一般损失.** 对一般可微损失，仍有固定核动力学

$$
\frac{d\mathbf f}{dt}=-\eta\,\Theta_\infty\nabla_{\mathbf f}\mathcal L,
$$

但不再有上面的矩阵指数闭式解。

### 4.4 Lazy training 的含义
{:data-toc-skip=''}

**Lazy training 既不等于不训练，也不等于只训练 readout。** 由第 2.4 节和第 4.2 节，单个神经元的变化趋于零，但大量由同一误差信号驱动的贡献相干叠加，产生 $O(1)$ 的预测变化。

更精确地说，极限动力学等价于训练初始化处的 **tangent model**（线性化模型）

$$
f_{\mathrm{lin}}(x;\theta)=f(x;\theta_0)+\nabla_\theta f(x;\theta_0)\cdot(\theta-\theta_0).
$$

它对**参数增量**是线性的，对**输入**仍然是非线性的 [<a href="#ref5">5</a>, <a href="#ref6">6</a>]。Chizat et al. [<a href="#ref6">6</a>] 进一步指出，laziness 本质上来自标度而不是宽度本身：任何可微模型，只要输出被放大、学习率相应缩小，都会进入这一线性化区域。下一篇会从相反的方向看到这一点：把输出缩小、学习率放大，就能离开这一区域。

### 4.5 NNGP 与 NTK 的分工
{:data-toc-skip=''}

两个核扮演不同的角色。初始化时 $f_0(x)=N^{-1/2}\sum_ia_i(0)\phi(h_i^{(L)}(x))$，由上一篇的结果，

$$
\mathbf f(0)\;\Longrightarrow\;\mathcal N\bigl(0,\Phi^{(L)}\bigr),
\qquad
K_{\mathrm{NNGP}}=\Phi^{(L)}.
$$

- **NNGP 核** $\Phi^{(L)}$ 决定随机初始函数的**协方差**；
- **NTK** $\Theta_\infty$ 决定这个函数在训练中**如何演化**。

$\Theta_\infty$ 是确定性的，但这并不意味着 $\mathbf f(0)$ 也自平均为零。因为 $f_\infty$ 是 $f_0$ 的仿射函数，训练后的网络仍然是一个高斯过程，其均值为 $\Theta_\infty(x,X)\Theta_\infty^{-1}\mathbf y$，协方差同时依赖于 NNGP 核与 NTK [<a href="#ref5">5</a>]。

---

## 5. Toy Example

### 5.1 单隐藏层
{:data-toc-skip=''}

当 $L=1$ 时，$G^{(1)}=\Psi^{(1)}\odot\mathbf1\mathbf1^\top=\Psi^{(1)}$，因此

$$
\Theta_\infty=\underbrace{\Phi^{(1)}}_{\text{readout}}+\underbrace{\Phi^{(0)}\odot\Psi^{(1)}}_{W^{(1)}}.
$$

第一项来自 readout，第二项来自输入权重。只训练 readout 时只剩第一项，也就是 random-feature kernel $\Phi^{(1)}$；同时训练两层才得到完整的 NTK。

以 ReLU 为例，取 $\Phi_{\mu\mu}^{(0)}=1$，则 $u_\mu\sim\mathcal N(0,1)$，

$$
\Phi_{\mu\mu}^{(1)}=\mathbb E\bigl[\operatorname{ReLU}(u)^2\bigr]=\frac12,
\qquad
\Psi_{\mu\mu}^{(1)}=\mathbb E\bigl[\mathbf 1\lbrace u>0\rbrace\bigr]=\frac12,
$$

因此 $\Theta_{\infty,\mu\mu}=\frac12+1\cdot\frac12=1$。这里不需要额外的 He initialization 因子：单隐藏层网络没有“逐层衰减”的问题，上一篇的 $\sigma_w^2=2$ 只在深度方向保持方差时才是必需的。

### 5.2 深度线性网络
{:data-toc-skip=''}

当 $\phi(u)=u$ 时，$\Phi^{(\ell)}=\mathbb E[u_\mu u_\nu]=\Phi^{(\ell-1)}=\cdots=\Phi^{(0)}$，而 $\phi'\equiv1$ 给出 $\Psi^{(\ell)}=\mathbf1\mathbf1^\top$，从而 $G^{(\ell)}=\mathbf1\mathbf1^\top$。于是

$$
\Theta_\infty=\Phi^{(0)}+\sum_{\ell=1}^{L}\Phi^{(0)}\odot\mathbf1\mathbf1^\top=(L+1)\,\Phi^{(0)}.
$$

$L$ 个隐藏层与一个 readout 各贡献一份输入核，这核对了参数块计数与归一化。这也说明，lazy 区域中的深度线性网络等价于输入空间上的线性回归，深度只把有效学习率放大了 $L+1$ 倍。

---

## 6. 小结

本文在 NTK parameterization 下得到了如下物理图像：

- **初始化**：前向场 $h$ 与 pre-gradient $z$ 是相互独立的高斯场；权重复用产生的响应序参量为零。
- **极限 NTK**：$\Theta_\infty$ 由前向核 $\Phi$、导数核 $\Psi$ 与反向核 $G$ 逐层组合而成，是一个确定性的矩阵。
- **训练**：显式反馈与响应项都被 $N^{-1/2}$ 压低，核在有限时间内冻结，网络在函数空间中做以 $\Theta_\infty$ 为核的核回归。

代价也同样清楚：每个神经元的 preactivation 只变化 $O(N^{-1/2})$，**网络的表示（features）在无穷宽极限下完全不学习**。这一结论来自第 3.1 节场方程中反馈系数 $\eta/(P\sqrt N)$ 的标度，而不是来自宽度本身。下一篇博客将改变这一标度，让反馈保持 $O(1)$：此时核随训练演化，响应序参量不再为零，我们将用完整的动力学 DMFT 描述 feature learning，并由此推导出 muP。

---

## References

<style>
  ol.references { padding-left: 1.8em; }
  ol.references li { margin-bottom: 0.5em; scroll-margin-top: 5rem; }
  ol.references li:target { outline: 1px dashed var(--global-theme-color, #b509ac); outline-offset: 3px; border-radius: 3px; }
</style>

<ol class="references">
  <li id="ref1">Arthur Jacot, Franck Gabriel, and Clément Hongler. <em>Neural Tangent Kernel: Convergence and Generalization in Neural Networks</em>. NeurIPS, 2018. <a href="https://arxiv.org/abs/1806.07572">arXiv:1806.07572</a>.</li>
  <li id="ref2">Cengiz Pehlevan and Blake Bordelon. <em>Lecture Notes on Infinite-Width Limits of Neural Networks</em>. Princeton Machine Learning Theory Summer School, June 2023.</li>
  <li id="ref3">Blake Bordelon and Cengiz Pehlevan. <em>Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks</em>. NeurIPS, 2022. <a href="https://arxiv.org/abs/2205.09653">arXiv:2205.09653</a>.</li>
  <li id="ref4">Greg Yang. <em>Tensor Programs II: Neural Tangent Kernel for Any Architecture</em>. 2020. <a href="https://arxiv.org/abs/2006.14548">arXiv:2006.14548</a>.</li>
  <li id="ref5">Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. <em>Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent</em>. NeurIPS, 2019. <a href="https://arxiv.org/abs/1902.06720">arXiv:1902.06720</a>.</li>
  <li id="ref6">Lénaïc Chizat, Edouard Oyallon, and Francis Bach. <em>On Lazy Training in Differentiable Programming</em>. NeurIPS, 2019. <a href="https://arxiv.org/abs/1812.07956">arXiv:1812.07956</a>.</li>
  <li id="ref7">Greg Yang and Etai Littwin. <em>Tensor Programs IIb: Architectural Universality of Neural Tangent Kernel Training Dynamics</em>. ICML, 2021. <a href="https://arxiv.org/abs/2105.03703">arXiv:2105.03703</a>.</li>
  <li id="ref8">Chaoyue Liu, Libin Zhu, and Mikhail Belkin. <em>On the Linearity of Large Non-Linear Models: When and Why the Tangent Kernel Is Constant</em>. NeurIPS, 2020. <a href="https://arxiv.org/abs/2010.01092">arXiv:2010.01092</a>.</li>
  <li id="ref9">Ethan Dyer and Guy Gur-Ari. <em>Asymptotics of Wide Networks from Feynman Diagrams</em>. ICLR, 2020. <a href="https://arxiv.org/abs/1909.11304">arXiv:1909.11304</a>.</li>
  <li id="ref10">Jiaoyang Huang and Horng-Tzer Yau. <em>Dynamics of Deep Neural Networks and Neural Tangent Hierarchy</em>. ICML, 2020. <a href="https://arxiv.org/abs/1909.08156">arXiv:1909.08156</a>.</li>
</ol>
