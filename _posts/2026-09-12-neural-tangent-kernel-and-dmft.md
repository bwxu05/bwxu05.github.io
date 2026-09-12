---
layout: post
title: Neural Tangent Kernel and Lazy Training
date: 2026-09-12
description: 无穷宽神经网络的训练动力学：NTK 的 DMFT 推导
tags: pre-training scaling muP DMFT NTK
categories: muP-Theory
---

上一篇 *Neural Network Initializations and NNGP* 讨论了随机初始化的宽神经网络：把权重视为 quenched disorder，在无穷宽极限下，前向传播可以由单个神经元的高斯场与确定性的 feature kernels 描述。这个结果回答的是：**训练开始之前，网络代表什么样的随机函数？**

这篇继续向前一步：**当所有层都参与训练时，这个随机函数如何演化？**

Neural Tangent Kernel (NTK) 给出了一种特别清楚的答案。在合适的参数化与宽度极限下，神经网络虽然仍在更新参数，其预测动力学却等价于一个固定核控制的动力学 [1,5]。从 DMFT 的视角看，这不是另一套完全独立的理论，而是前向、反向单点场的训练反馈在 leading order 消失后，留下的一个可解极限 [2,3]。

本文沿用上一篇的 $h^{(\ell)}$、$\phi$ 和 $\Phi^{(\ell)}$，补充 backward fields 与 gradient kernels。我们先推导有限宽网络的精确恒等式，再进行无序平均与鞍点分析，最后解释核为什么冻结，以及这一结论为什么不排除 $\mu$P 中的 feature learning。以下公式统一采用本文明确给出的参数化和 loss normalization；文献中的不同层编号与缩放因子不直接混用。

## 1. From initialization to training

### 1.1 Network and parameterization

考虑 $L$ 个隐藏层、每个隐藏层宽度为 $N$ 的全连接网络。输入维度为 $D$，训练集为 $\lbrace (x_\mu,y_\mu)\rbrace_{\mu=1}^{P}$。不使用 bias，输出为标量：

$$
h_{\mu i}^{(1)}(t)
=
\frac{1}{\sqrt D}
\sum_{j=1}^{D}W_{ij}^{(1)}(t)x_{\mu j},
$$

$$
h_{\mu i}^{(\ell)}(t)
=
\frac{1}{\sqrt N}
\sum_{j=1}^{N}W_{ij}^{(\ell)}(t)
\phi\!\left(h_{\mu j}^{(\ell-1)}(t)\right),
\qquad 2\leq \ell\leq L,
$$

$$
f_\mu(t)
=
\frac{1}{\sqrt N}
\sum_{i=1}^{N}a_i(t)
\phi\!\left(h_{\mu i}^{(L)}(t)\right).
$$

初始化时，所有 $W_{ij}^{(\ell)}(0)$ 与 $a_i(0)$ 都是相互独立的 $\mathcal N(0,1)$ 随机变量。$L$ 只计隐藏层，因此网络共有 $L+1$ 个可训练的参数块。

这里真正被优化的参数是

$$
\theta=\operatorname{Vec}\!\left(
W^{(1)},\ldots,W^{(L)},a
\right),
$$

即公式中方差为 $1$ 的 **raw parameters**；$D^{-1/2}$ 与 $N^{-1/2}$ 是前向计算中的显式因子。对这些参数使用宽度无关的统一学习率，是本文采用的 NTK parameterization [1,5]。

这一约定不能省略。把 $W/\sqrt N$ 改写成一个方差为 $1/N$ 的参数，确实不改变初始化函数分布；但若随后对新参数仍使用相同数值的学习率，训练动力学一般已经改变。**初始化等价，不等于优化过程等价。**

### 1.2 Gradient flow and the order of limits

采用平方损失与 full-batch gradient flow：

$$
\mathcal L(\theta)
=
\frac{1}{2P}\sum_{\mu=1}^{P}
\left(f_\mu-y_\mu\right)^2,
\qquad
\frac{d\theta}{dt}
=-\eta\nabla_\theta\mathcal L,
$$

其中 $\eta>0$ 不随 $N$ 改变。定义 prediction error

$$
\Delta_\mu(t)=y_\mu-f_\mu(t),
$$

于是

$$
\frac{d\theta}{dt}
=
\frac{\eta}{P}
\sum_{\mu=1}^{P}
\Delta_\mu(t)\nabla_\theta f_\mu(t).
$$

本文取 $N\to\infty$，同时固定 $P,D,L$ 与有限训练时间区间 $[0,T]$。讨论动力学极限时，默认 activation 具有足够的正则性与矩控制，例如 $\tanh$；ReLU 的初始化核也可以计算，但其训练收敛论证需要额外控制 activation gates 的切换 [4,5,8]。

因此，后面得到的结论不是关于任意学习率、任意增长深度或任意长训练时间的统一断言，也不直接覆盖 Adam、weight decay 或 $P/N$ 固定的联合极限。

## 2. The NTK is an exact finite-width object

### 2.1 Prediction dynamics in function space

对 $f_\mu(t)$ 使用链式法则：

$$
\begin{aligned}
\frac{df_\mu}{dt}
&=
\nabla_\theta f_\mu(t)\cdot\frac{d\theta}{dt}
\\
&=
\frac{\eta}{P}
\sum_{\nu=1}^{P}
\left[
\nabla_\theta f_\mu(t)\cdot\nabla_\theta f_\nu(t)
\right]\Delta_\nu(t).
\end{aligned}
$$

定义 empirical NTK [1]

$$
\boxed{
\Theta_{N,\mu\nu}(t)
=
\nabla_\theta f_\mu(t)\cdot
\nabla_\theta f_\nu(t).
}
$$

则

$$
\boxed{
\frac{d\mathbf f}{dt}
=
\frac{\eta}{P}\Theta_N(t)\boldsymbol\Delta(t),
\qquad
\frac{d\boldsymbol\Delta}{dt}
=
-\frac{\eta}{P}\Theta_N(t)\boldsymbol\Delta(t).
}
$$

**这个恒等式在有限宽度下就成立。** 无穷宽理论需要回答的不是“NTK 是否存在”，而是：$\Theta_N(t)$ 能否自平均到一个可计算的对象？这个对象是否还随训练改变？

由于 $\Theta_N=JJ^\top$ 是参数 Jacobian 的 Gram matrix，它必然半正定。因此

$$
\frac{d\mathcal L}{dt}
=
-\frac{\eta}{P^2}
\boldsymbol\Delta^\top\Theta_N(t)\boldsymbol\Delta
\leq 0.
$$

这里保留 $P^{-1}$ 很重要：本文定义的是平均损失，而不是对样本直接求和的损失。

### 2.2 Forward and backward order parameters

沿用上一篇的 empirical feature kernel，并加入两个时间变量：

$$
\Phi_{N,\mu\nu}^{(\ell)}(t,s)
=
\frac{1}{N}\sum_{i=1}^{N}
\phi\!\left(h_{\mu i}^{(\ell)}(t)\right)
\phi\!\left(h_{\nu i}^{(\ell)}(s)\right),
$$

$$
\Phi_{\mu\nu}^{(0)}(t,s)
=
\Phi_{\mu\nu}^{(0)}
=
\frac{x_\mu^\top x_\nu}{D}.
$$

为了使反向传播的单点量保持 $O(1)$，定义 rescaled gradient field

$$
g_{\mu i}^{(\ell)}(t)
=
\sqrt N\,
\frac{\partial f_\mu(t)}{\partial h_{\mu i}^{(\ell)}(t)}.
$$

引入 pre-gradient field $z^{(\ell)}$，则反向传播写成

$$
g_{\mu i}^{(\ell)}(t)
=
\phi'\!\left(h_{\mu i}^{(\ell)}(t)\right)
z_{\mu i}^{(\ell)}(t),
$$

$$
z_{\mu i}^{(\ell)}(t)
=
\frac{1}{\sqrt N}
\sum_{j=1}^{N}
W_{ji}^{(\ell+1)}(t)g_{\mu j}^{(\ell+1)}(t),
\qquad \ell<L,
$$

$$
z_{\mu i}^{(L)}(t)=a_i(t).
$$

与 $\Phi$ 对应的 backward order parameter 是 gradient kernel [2,3]：

$$
G_{N,\mu\nu}^{(\ell)}(t,s)
=
\frac{1}{N}\sum_{i=1}^{N}
g_{\mu i}^{(\ell)}(t)g_{\nu i}^{(\ell)}(s).
$$

$\Phi$ 衡量两个样本在前向特征空间中的重叠；$G$ 衡量它们对同一层 preactivations 的输出敏感度重叠。这里的 $g$ 是输出的梯度，不是已经乘上 $\Delta$ 的 loss gradient。

### 2.3 Decomposing the NTK layer by layer

输出层的导数为

$$
\frac{\partial f_\mu}{\partial a_i}
=
\frac{1}{\sqrt N}\phi(h_{\mu i}^{(L)}),
$$

因此输出层对 NTK 的贡献是 $\Phi_N^{(L)}(t,t)$。

对 $2\leq\ell\leq L$，

$$
\frac{\partial f_\mu}{\partial W_{ij}^{(\ell)}}
=
\frac{1}{N}
g_{\mu i}^{(\ell)}\phi(h_{\mu j}^{(\ell-1)}),
$$

于是

$$
\begin{aligned}
\sum_{i,j}
\frac{\partial f_\mu}{\partial W_{ij}^{(\ell)}}
\frac{\partial f_\nu}{\partial W_{ij}^{(\ell)}}
&=
\left[\frac{1}{N}\sum_i
 g_{\mu i}^{(\ell)}g_{\nu i}^{(\ell)}\right]
\left[\frac{1}{N}\sum_j
 \phi(h_{\mu j}^{(\ell-1)})
 \phi(h_{\nu j}^{(\ell-1)})\right]
\\
&=
G_{N,\mu\nu}^{(\ell)}(t,t)
\Phi_{N,\mu\nu}^{(\ell-1)}(t,t).
\end{aligned}
$$

第一层要单独核对输入维度：

$$
\frac{\partial f_\mu}{\partial W_{ij}^{(1)}}
=
\frac{1}{\sqrt{ND}}g_{\mu i}^{(1)}x_{\mu j},
$$

其贡献仍然是 $G_N^{(1)}(t,t)\odot\Phi^{(0)}$。综合各层，得到精确恒等式

$$
\boxed{
\Theta_N(t)
=
\Phi_N^{(L)}(t,t)
+
\sum_{\ell=1}^{L}
G_N^{(\ell)}(t,t)\odot
\Phi_N^{(\ell-1)}(t,t).
}
$$

其中 $\odot$ 表示逐元素乘法，而不是矩阵乘法；第一层的 $\Phi_N^{(0)}$ 就是固定的 $\Phi^{(0)}$。

这一步还没有使用高斯性或前后向独立性。乘积分解来自参数导数的 outer-product 结构，是有限 $N$ 下的代数恒等式。

## 3. Rewriting training dynamics for DMFT

DMFT 要平均的是**初始化权重**，不是把每个训练时刻的权重重新抽样。为此，先把所有 $W^{(\ell)}(t)$ 改写成 $W^{(\ell)}(0)$ 与训练历史的函数。这与训练动力学 DMFT 的标准构造一致 [2,3]。

### 3.1 Integrating the parameter updates

对 hidden-to-hidden weights，

$$
\frac{dW_{ij}^{(\ell)}}{dt}
=
\frac{\eta}{PN}
\sum_{\nu=1}^{P}
\Delta_\nu(t)g_{\nu i}^{(\ell)}(t)
\phi(h_{\nu j}^{(\ell-1)}(t)),
\qquad \ell\geq 2.
$$

积分后代回前向传播。把内层的神经元求和识别为 $\Phi_N^{(\ell-1)}$，得到

$$
\boxed{
\begin{aligned}
h_{\mu i}^{(\ell)}(t)
={}&\chi_{\mu i}^{(\ell)}(t)
\\
&+\frac{\eta}{P\sqrt N}
\sum_{\nu=1}^{P}\int_0^t ds\,
\Delta_\nu(s)
\Phi_{N,\mu\nu}^{(\ell-1)}(t,s)
g_{\nu i}^{(\ell)}(s),
\end{aligned}
}
$$

其中

$$
\chi_{\mu i}^{(\ell)}(t)
=
\frac{1}{\sqrt N}
\sum_j W_{ij}^{(\ell)}(0)
\phi(h_{\mu j}^{(\ell-1)}(t)).
$$

第一层也满足同一个积分公式，只需采用

$$
\chi_{\mu i}^{(1)}
=
\frac{1}{\sqrt D}\sum_jW_{ij}^{(1)}(0)x_{\mu j},
\qquad
\Phi_N^{(0)}(t,s)=\Phi^{(0)}.
$$

注意：$\chi^{(\ell)}(t)$ 虽然使用初始化权重，却未必等于初始化 preactivation，因为它作用在时刻 $t$ 的 features 上。

### 3.2 The corresponding backward equation

对反向传播进行相同操作，得到

$$
\boxed{
\begin{aligned}
z_{\mu i}^{(\ell)}(t)
={}&\xi_{\mu i}^{(\ell)}(t)
\\
&+\frac{\eta}{P\sqrt N}
\sum_{\nu=1}^{P}\int_0^t ds\,
\Delta_\nu(s)
G_{N,\mu\nu}^{(\ell+1)}(t,s)
\phi(h_{\nu i}^{(\ell)}(s)),
\end{aligned}
}
$$

其中，对 $\ell<L$，

$$
\xi_{\mu i}^{(\ell)}(t)
=
\frac{1}{\sqrt N}
\sum_jW_{ji}^{(\ell+1)}(0)
g_{\mu j}^{(\ell+1)}(t).
$$

输出端使用边界约定

$$
\xi_{\mu i}^{(L)}(t)=a_i(0),
\qquad
G_{N,\mu\nu}^{(L+1)}(t,s)=1.
$$

这里 $G^{(L+1)}$ 是用于统一公式的边界对象，不是额外隐藏层的 empirical kernel。

至此，前向与反向动力学都由“初始化无序产生的场”与“训练历史的反馈”组成。在本文参数化下，显式反馈项都带有 $N^{-1/2}$。

### 3.3 What still needs to be shown?

这个小因子提示：在固定时间内，单个神经元可能只发生很小的变化。但目前还不能直接断言 NTK 冻结。

首先，$\chi(t)$ 与 $\xi(t)$ 的输入本身在演化，需要控制这种间接变化。其次，前向传播使用 $W(0)$，反向传播使用同一个 $W(0)^\top$，两者并不独立。无序平均必须保留这个相关性。

下面先求出初始化时、也就是候选 lazy limit 中的联合单点理论；再回到上述动力学，检查这个静态解能否在训练期间保持。

## 4. DMFT derivation of the joint single-site theory

这一节是上一篇 NNGP 推导的延伸：除了 forward field，还需要同时追踪 backward field。为突出静态 NTK 的计算，先省略初始化时刻的时间下标。完整的 time-dependent path integral 会把下面的样本指标扩展为“样本—时间”指标 [3]。

### 4.1 Sources and deterministic constraints

引入只作用在有限个 sites 上的 source fields $b,c$，定义 characteristic generating function

$$
Z[b,c]
=
\mathbb E_{W,a}
\exp\left\{
i\sum_{\ell,\mu,i}
\left[
b_{\mu i}^{(\ell)}h_{\mu i}^{(\ell)}
+c_{\mu i}^{(\ell)}z_{\mu i}^{(\ell)}
\right]
\right\}.
$$

对一个连接 $\ell-1$ 与 $\ell$ 的随机矩阵 $W^{(\ell)}$，必须同时约束

$$
\mathbf h_\mu^{(\ell)}
=
\frac{1}{\sqrt N}W^{(\ell)}
\phi(\mathbf h_\mu^{(\ell-1)}),
$$

$$
\mathbf z_\mu^{(\ell-1)}
=
\frac{1}{\sqrt N}W^{(\ell)\top}
\mathbf g_\mu^{(\ell)},
\qquad
g_\mu^{(\ell)}=\phi'(h_\mu^{(\ell)})z_\mu^{(\ell)}.
$$

使用 delta functions 与 Fourier conjugate fields $\hat h,\hat z$ 后，与这个随机矩阵有关的指数为

$$
-\frac{i}{\sqrt N}
\sum_{i,j}W_{ij}^{(\ell)}
\sum_\mu
\left[
\hat h_{\mu i}^{(\ell)}\phi(h_{\mu j}^{(\ell-1)})
+
g_{\mu i}^{(\ell)}\hat z_{\mu j}^{(\ell-1)}
\right].
$$

前向、反向项出现在**同一个** $W_{ij}^{(\ell)}$ 的系数中。这正是不能提前假定独立的地方。

### 4.2 Gaussian disorder average and the mixed term

由 $\mathbb E_{W\sim\mathcal N(0,1)}e^{-iWA}=e^{-A^2/2}$，无序平均精确给出

$$
\exp\left\{
-\frac{1}{2N}
\sum_{i,j}
\left[
\sum_\mu
\left(
\hat h_{\mu i}^{(\ell)}\phi(h_{\mu j}^{(\ell-1)})
+g_{\mu i}^{(\ell)}\hat z_{\mu j}^{(\ell-1)}
\right)
\right]^2
\right\}.
$$

将平方展开，指数包含三部分：

$$
-\frac12\sum_{\mu,\nu}
\Phi_{N,\mu\nu}^{(\ell-1)}
\hat{\mathbf h}_\mu^{(\ell)}\cdot
\hat{\mathbf h}_\nu^{(\ell)},
$$

$$
-\frac12\sum_{\mu,\nu}
G_{N,\mu\nu}^{(\ell)}
\hat{\mathbf z}_\mu^{(\ell-1)}\cdot
\hat{\mathbf z}_\nu^{(\ell-1)},
$$

以及

$$
\boxed{
-\frac1N\sum_{\mu,\nu}
\left(
\hat{\mathbf h}_\mu^{(\ell)}\cdot
\mathbf g_\nu^{(\ell)}
\right)
\left(
\phi(\mathbf h_\mu^{(\ell-1)})\cdot
\hat{\mathbf z}_\nu^{(\ell-1)}
\right).
}
$$

前两项分别产生 forward 与 backward Gaussian covariances。第三项记录了同一随机矩阵被正向与转置复用的影响；在一般训练 DMFT 中，它对应不能预先丢弃的 response coupling [3]。

例如，定义混合序参量

$$
U_{\mu\nu}^{(\ell)}
=
\frac1N\hat{\mathbf h}_\mu^{(\ell)}\cdot
\mathbf g_\nu^{(\ell)},
\qquad
V_{\mu\nu}^{(\ell-1)}
=
\frac1N\phi(\mathbf h_\mu^{(\ell-1)})\cdot
\hat{\mathbf z}_\nu^{(\ell-1)},
$$

则这一项为 $-N\sum_{\mu,\nu}U_{\mu\nu}^{(\ell)}V_{\mu\nu}^{(\ell-1)}$。它本身是 extensive 的，不能仅凭指数前面出现 $1/N$ 就把它忽略。

### 4.3 Order parameters and the large-width saddle

与上一篇一样，把 $\Phi,G,U,V$ 提升为独立序参量，并用 delta functions 约束它们的定义。例如，对 $\Phi$ 的一个独立矩阵元，

$$
\begin{aligned}
1
={}&\int d\Phi_{\mu\nu}^{(\ell)}
\frac{N\,d\hat\Phi_{\mu\nu}^{(\ell)}}{2\pi}
\\
&\times\exp\left\{
iN\hat\Phi_{\mu\nu}^{(\ell)}
\left[
\Phi_{\mu\nu}^{(\ell)}
-\frac1N\sum_i
\phi(h_{\mu i}^{(\ell)})\phi(h_{\nu i}^{(\ell)})
\right]
\right\}.
\end{aligned}
$$

对称矩阵只需对独立矩阵元积分；其他序参量采用相应约束。这样可以把神经元间的耦合转移到宏观 kernels 上，得到形式

$$
Z[b,c]
\propto
\int\mathcal D\mathcal Q\,\mathcal D\hat{\mathcal Q}\,
\exp\left\{
N S[\mathcal Q,\hat{\mathcal Q};b,c]
\right\},
$$

其中 $\mathcal Q$ 收集上述序参量，而作用量中的微观部分成为各层 single-site log partition functions 的和。省略的归一化常数由 $Z[0,0]=1$ 固定。

在 $N\to\infty$ 时，考察与原始前馈网络相对应的 physical saddle：

$$
\frac{\delta S}{\delta\mathcal Q}=0,
\qquad
\frac{\delta S}{\delta\hat{\mathcal Q}}=0.
$$

对 conjugate kernels 的变分给出 self-consistency equations

$$
\Phi_{\mu\nu}^{(\ell)}
=
\left\langle
\phi(h_\mu^{(\ell)})\phi(h_\nu^{(\ell)})
\right\rangle,
\qquad
G_{\mu\nu}^{(\ell)}
=
\left\langle
 g_\mu^{(\ell)}g_\nu^{(\ell)}
\right\rangle.
$$

尖括号是单点有效理论上的平均；不再是对训练样本求平均。有限个 source insertions 不改变 leading-order 的 bulk saddle。

### 4.4 Why the mixed response vanishes in the lazy saddle

在这个独立、零均值 readout 的全连接网络中，候选静态单点解可以写成

$$
\mathbf h^{(\ell)}=\mathbf u^{(\ell)},
\qquad
\mathbf z^{(\ell)}=\mathbf r^{(\ell)},
\qquad
g_\mu^{(\ell)}
=\phi'(u_\mu^{(\ell)})r_\mu^{(\ell)},
$$

其中 $\mathbf u^{(\ell)}$ 与 $\mathbf r^{(\ell)}$ 是相互独立的零均值高斯向量。

如何检查这不是把独立性直接当作假设？上面的混合序参量，在单点理论中可以通过 Gaussian integration by parts 解释为平均交叉响应 [3]。在这一候选解上，

$$
\left\langle
\frac{\partial\phi(h_\mu^{(\ell)})}
{\partial r_\nu^{(\ell)}}
\right\rangle
=0,
$$

因为前向场 $h=u$ 不依赖反向驱动 $r$。另一方面，

$$
\begin{aligned}
\left\langle
\frac{\partial g_\mu^{(\ell)}}
{\partial u_\nu^{(\ell)}}
\right\rangle
&=
\delta_{\mu\nu}
\left\langle
\phi''(u_\mu^{(\ell)})r_\mu^{(\ell)}
\right\rangle
\\
&=0,
\end{aligned}
$$

因为 $r$ 与 $u$ 独立且 $r$ 零均值。因此 $U=V=0$ 与混合响应的鞍点方程相容。上式的普通求导针对光滑 $\phi$；ReLU 核可由相应的 Gaussian gate expectations 或平滑近似获得。

在零 source 下，归一化的高斯单点测度还给出 $\hat\Phi=\hat G=0$。由此，前向和反向驱动在 leading-order 单点理论中解耦。

这是一个 **formal saddle-point 自洽检查**，不是所有可能鞍点的唯一性证明。对这里的网络，前后向极限计算也有 Tensor Programs 的严格支撑；但对于不满足相应条件的网络，直接使用 gradient independence assumption 可以得到错误答案 [4]。

还要区分两件事：这里消失的是未归一化的平均交叉响应；在 feature-learning DMFT 中，有些文献会先把响应除以 feature-learning 强度，其归一化后极限不必为零。

### 4.5 The effective Gaussian fields

在上述 saddle 上，单点 characteristic function 简化为

$$
\begin{aligned}
\mathcal Z^{(\ell)}(\mathbf b,\mathbf c)
&=
\mathbb E_{\mathbf u,\mathbf r}
\exp\left(i\mathbf b^\top\mathbf u+i\mathbf c^\top\mathbf r\right)
\\
&=
\exp\left[
-\frac12\mathbf b^\top\Phi^{(\ell-1)}\mathbf b
-\frac12\mathbf c^\top G^{(\ell+1)}\mathbf c
\right].
\end{aligned}
$$

所以

$$
\boxed{
\begin{gathered}
\mathbf u^{(\ell)}\sim\mathcal N(0,\Phi^{(\ell-1)}),
\qquad
\mathbf r^{(\ell)}\sim\mathcal N(0,G^{(\ell+1)}),
\\
\mathbf u^{(\ell)}\perp\mathbf r^{(\ell)},
\qquad
h_\mu^{(\ell)}=u_\mu^{(\ell)},
\qquad
g_\mu^{(\ell)}=\phi'(u_\mu^{(\ell)})r_\mu^{(\ell)}.
\end{gathered}
}
$$

这个写法也允许退化高斯分布，不需要对协方差矩阵求逆。

输出端的边界尤其值得注意：$z_{\mu i}^{(L)}=a_i$ 对所有输入 $\mu$ 都是同一个随机数。因此

$$
G_{\mu\nu}^{(L+1)}=1,
\qquad
G^{(L+1)}=\mathbf 1\mathbf 1^\top,
$$

而不是单位矩阵 $I_P$。不同样本共享同一个 readout，不能为每个样本独立抽取一个 readout。

最后，**高斯的是 $h$ 与 pre-gradient $z$，一般不是 $g$。** $g=\phi'(h)z$ 含有 activation gate，通常是非高斯变量。

## 5. Recovering the infinite-width NTK

### 5.1 Forward and backward self-consistency

把单点解代回 order parameters，前向递推仍然是上一篇的 NNGP feature-kernel recursion：

$$
\boxed{
\Phi_{\mu\nu}^{(\ell)}
=
\mathbb E_{\mathbf u\sim\mathcal N(0,\Phi^{(\ell-1)})}
\left[\phi(u_\mu)\phi(u_\nu)\right].
}
$$

另外定义 activation-derivative kernel

$$
\Psi_{\mu\nu}^{(\ell)}
=
\mathbb E_{\mathbf u\sim\mathcal N(0,\Phi^{(\ell-1)})}
\left[\phi'(u_\mu)\phi'(u_\nu)\right].
$$

这里用 $\Psi$ 而不用点号，避免与时间导数混淆。由于单点 $u$ 与 $r$ 独立，

$$
\begin{aligned}
G_{\mu\nu}^{(\ell)}
&=
\mathbb E\left[
\phi'(u_\mu)\phi'(u_\nu)r_\mu r_\nu
\right]
\\
&=
\Psi_{\mu\nu}^{(\ell)}G_{\mu\nu}^{(\ell+1)}.
\end{aligned}
$$

因此 backward recursion 为

$$
\boxed{
G^{(\ell)}
=
\Psi^{(\ell)}\odot G^{(\ell+1)},
\qquad
G^{(L+1)}=\mathbf 1\mathbf 1^\top.
}
$$

计算顺序也很自然：先从输入层前向计算 $\Phi^{(1)},\ldots,\Phi^{(L)}$ 与 $\Psi^{(\ell)}$，再从输出端反向计算 $G^{(L)},\ldots,G^{(1)}$。

### 5.2 The deterministic limiting kernel

有限宽 NTK 的层分解现在变成

$$
\boxed{
\Theta_\infty
=
\Phi^{(L)}
+
\sum_{\ell=1}^{L}
\Phi^{(\ell-1)}\odot G^{(\ell)}.
}
$$

对单个矩阵元，也可以写成

$$
\Theta_{\infty,\mu\nu}
=
\Phi_{\mu\nu}^{(L)}
+
\sum_{\ell=1}^{L}
\Phi_{\mu\nu}^{(\ell-1)}
\prod_{r=\ell}^{L}\Psi_{\mu\nu}^{(r)}.
$$

这就是标准的全连接 NTK [1,4]。每层贡献由两部分相乘：进入该层的 forward-feature overlap，以及从该层通向输出的 backward-sensitivity overlap。

若希望只写前向递推，定义 $\Theta^{(\ell)}$ 为具有 $\ell$ 个隐藏层的相应前缀网络的极限 NTK，则等价地有

$$
\boxed{
\Theta^{(0)}=\Phi^{(0)},
\qquad
\Theta^{(\ell)}
=
\Phi^{(\ell)}
+
\Psi^{(\ell)}\odot\Theta^{(\ell-1)},
\qquad
\Theta_\infty=\Theta^{(L)}.
}
$$

其中新增的 $\Phi^{(\ell)}$ 是新 readout 的参数贡献；旧参数的贡献要经过新 activation 的导数核传播。

### 5.3 Why the kernel remains frozen during training

到目前为止，已经计算了 $\Theta_N(0)$ 的无穷宽极限。但训练中的核冻结还需要回到第 3 节的动力学。

在固定 $P,L,T$、稳定的前后向传播与适当矩控制下，带有 $N^{-1/2}$ 的训练反馈导致典型单点场的变化趋于零。对光滑 activation，量级分析给出

$$
h_{\mu i}^{(\ell)}(t)-h_{\mu i}^{(\ell)}(0)
=O_{\mathbb P}(N^{-1/2}),
$$

$$
g_{\mu i}^{(\ell)}(t)-g_{\mu i}^{(\ell)}(0)
=O_{\mathbb P}(N^{-1/2})
$$

的典型有限时间尺度。这里的估计不是对所有神经元最大值、任意训练时间的无条件界；严格的训练收敛论证还需控制 Jacobian 与 activation 的稳定性 [5,8]。

结合初始化的自平均与这些稳定性条件，得到本文 setting 下的 NTK 极限：

$$
\sup_{0\leq t\leq T}
\left\lVert\Theta_N(t)-\Theta_\infty\right\rVert
\xrightarrow[N\to\infty]{\mathbb P}0.
$$

在对应的 lazy DMFT 中，两时核也退化为静态对象：

$$
\Phi^{(\ell)}(t,s)=\Phi^{(\ell)},
\qquad
G^{(\ell)}(t,s)=G^{(\ell)}.
$$

这并不是每个时刻重新采样高斯场。相反，同一个单点初始化场在不同训练时刻被保留下来。例如

$$
\mathbb E\left[u_\mu^{(\ell)}(t)u_\nu^{(\ell)}(s)\right]
=
\Phi_{\mu\nu}^{(\ell-1)}
$$

不依赖 $t,s$；它不是含有 $\delta(t-s)$ 的白噪声。本文没有 minibatch noise，DMFT 中的随机性来自 quenched initialization。

由此可以把逻辑分清：**自平均让核变成确定量；lazy scaling 与训练稳定性让这个确定量保持不变。** 第一件事本身不能推出第二件事。

## 6. Fixed-kernel prediction and loss dynamics

### 6.1 Why predictions still change by an order-one amount

单个神经元的变化趋于零，并不意味着网络无法学习。仅考虑 readout 更新，就有

$$
\left.\frac{df_\mu}{dt}\right|_{a}
=
\frac{1}{\sqrt N}\sum_i
\frac{da_i}{dt}\phi(h_{\mu i}^{(L)})
=
\frac{\eta}{P}\sum_\nu
\Phi_{N,\mu\nu}^{(L)}(t,t)\Delta_\nu(t).
$$

每个 $a_i$ 的变化是 $O(N^{-1/2})$，但 $N$ 个由同一误差信号驱动的贡献经过求和，仍能产生 $O(1)$ 的预测变化。隐藏层的变化也通过同样的参数 Jacobian 结构贡献 $O(1)$ 的核项。

因此，lazy training 不等于“不训练”，也不等于“只训练最后一层”。它表示：**所有层可以共同改变预测，但 leading-order 特征与切空间不需要发生 $O(1)$ 的演化。**

### 6.2 Closed-form solution for squared loss

在极限核固定后，误差满足线性微分方程

$$
\frac{d\boldsymbol\Delta}{dt}
=
-\frac{\eta}{P}\Theta_\infty\boldsymbol\Delta.
$$

因此 [1]

$$
\boxed{
\boldsymbol\Delta(t)
=
\exp\left(-\frac{\eta t}{P}\Theta_\infty\right)
\boldsymbol\Delta(0),
}
$$

$$
\mathbf f(t)
=
\mathbf y+
\exp\left(-\frac{\eta t}{P}\Theta_\infty\right)
\left[\mathbf f(0)-\mathbf y\right].
$$

若

$$
\Theta_\infty v_k=\kappa_kv_k,
\qquad
\boldsymbol\Delta(0)=\sum_{k=1}^{P}c_kv_k,
$$

且 $v_k$ 为正交归一特征向量，则

$$
\boxed{
\mathcal L(t)
=
\frac{1}{2P}\sum_{k=1}^{P}
c_k^2\exp\left(-\frac{2\eta\kappa_k}{P}t\right).
}
$$

这说明每个 kernel eigenmode 都有自己的衰减时间。大的 $\kappa_k$ 对应更快的拟合，小的 $\kappa_k$ 对应慢模态；$\kappa_k=0$ 的误差分量不会消失。半正定并不自动保证训练集可被完全拟合。

这个矩阵指数解依赖平方损失。对于一般可微损失，固定 NTK 仍给出

$$
\frac{d\mathbf f}{dt}
=-\eta\Theta_\infty\nabla_{\mathbf f}\mathcal L,
$$

但 $\nabla_{\mathbf f}\mathcal L$ 一般是 $\mathbf f$ 的非线性函数，不能照搬同一个指数解。

### 6.3 Linearization is in parameter space

考虑初始化处的 tangent model：

$$
f_{\mathrm{lin}}(x;\theta)
=
 f(x;\theta_0)
+
\nabla_\theta f(x;\theta_0)^\top(\theta-\theta_0).
$$

它的 parameter features 固定为 $\nabla_\theta f(x;\theta_0)$，对应的 Gram kernel 就是初始化 NTK。宽网络在上述极限中的预测动力学由这个 tangent model 描述 [5,6]。

这里的“线性”是对参数增量线性，不是说 $f(x)$ 对输入 $x$ 线性。一个深层 ReLU 网络的固定 NTK 仍然可以定义非常非线性的输入函数空间。

### 6.4 NNGP and NTK play different roles

对于本文的输出缩放，初始化输出满足

$$
\mathbf f(0)
\ \Longrightarrow\ 
\mathcal N\!\left(0,\Phi^{(L)}\right).
$$

所以 $K_{\mathrm{NNGP}}=\Phi^{(L)}$ 给出随机初始函数的协方差，而 $\Theta_\infty$ 决定这个函数随后如何在参数梯度下降下移动。

特别地，**NTK 自平均不意味着初始化预测也自平均为零。** $\Theta_\infty$ 是确定性的，但 $\mathbf f(0)$ 仍是非退化的随机向量；单次训练轨迹保留这个随机初始条件。

若训练集上的 $\Theta_\infty(X,X)$ 正定，则在已经取出的固定核模型中令 $t\to\infty$，测试点预测为

$$
\begin{aligned}
f_\infty(x)
={}&f_0(x)
\\
&+\Theta_\infty(x,X)
\Theta_\infty(X,X)^{-1}
\left[\mathbf y-\mathbf f_0(X)\right].
\end{aligned}
$$

它是围绕初始函数 $f_0$ 的 NTK interpolation；一般既不是把 $f_0$ 默认为零后的表达式，也不是使用 NNGP covariance 进行 Bayesian conditioning 的结果 [5]。

## 7. Two useful checks

### 7.1 One hidden layer: where the extra NTK term comes from

对于

$$
f(x)
=
\frac1{\sqrt N}\sum_i
 a_i\phi\!\left(\frac{w_i^\top x}{\sqrt D}\right),
$$

前面的结果退化为

$$
\boxed{
\Theta_\infty
=
\Phi^{(1)}+\Phi^{(0)}\odot\Psi^{(1)}.
}
$$

第一项来自训练 $a_i$，第二项来自训练 $w_i$。只训练 readout 时，核才退化为 $\Phi^{(1)}$；训练所有层的 lazy network 与只训练 readout 的 random-feature model 并不是同一个模型。

以 $\phi(u)=\operatorname{ReLU}(u)$ 为例，记

$$
q_\mu=\Phi_{\mu\mu}^{(0)},
\qquad
\cos\vartheta_{\mu\nu}
=
\frac{\Phi_{\mu\nu}^{(0)}}{\sqrt{q_\mu q_\nu}},
$$

并假设 $q_\mu,q_\nu>0$。由二维高斯积分，上一篇的 forward kernel 为

$$
\Phi_{\mu\nu}^{(1)}
=
\frac{\sqrt{q_\mu q_\nu}}{2\pi}
\left[
\sin\vartheta_{\mu\nu}
+(\pi-\vartheta_{\mu\nu})\cos\vartheta_{\mu\nu}
\right].
$$

由于 $\phi'(u)=\mathbf 1_{u>0}$ 几乎处处成立，导数核就是两个高斯变量同时为正的概率：

$$
\Psi_{\mu\nu}^{(1)}
=
\mathbb P(u_\mu>0,u_\nu>0)
=
\frac{\pi-\vartheta_{\mu\nu}}{2\pi}.
$$

所以

$$
\Theta_{\infty,\mu\nu}
=
\frac{\sqrt{q_\mu q_\nu}}{2\pi}
\left[
\sin\vartheta_{\mu\nu}
+2(\pi-\vartheta_{\mu\nu})\cos\vartheta_{\mu\nu}
\right].
$$

例如，归一化输入满足 $q_\mu=1$ 时，

$$
\Phi_{\mu\mu}^{(1)}=\frac12,
\qquad
\Psi_{\mu\mu}^{(1)}=\frac12,
\qquad
\Theta_{\infty,\mu\mu}=1.
$$

这提供了一个简单的 normalization check。这里用的是本文约定的单位权重方差，而没有额外加入 He initialization 的 $\sqrt2$ 因子。

### 7.2 Deep linear network

若 $\phi(u)=u$，则

$$
\Phi^{(\ell)}=\Phi^{(0)},
\qquad
\Psi_{\mu\nu}^{(\ell)}=1,
\qquad
G_{\mu\nu}^{(\ell)}=1.
$$

于是

$$
\boxed{
\Theta_\infty=(L+1)\Phi^{(0)}.
}
$$

每个可训练参数块贡献一次输入核。这再次核对了层计数：$L$ 个隐藏层，加一个 readout，共 $L+1$ 项。

在这个特定的线性、单位方差、统一 raw-parameter 学习率的 lazy 极限中，增加深度只放大固定核，不改变其特征向量。这个结论不应推广为一般深层网络的训练规律。

## 8. From the NTK limit to feature learning and muP

NTK 的结论常被概括为“无穷宽网络不学习 features”。更准确的表述是：**本文的 NTK parameterization，在固定学习率和有限训练时间的宽度极限下，不产生 $O(1)$ 的特征演化。** Lazy behavior 与 scaling 的选择有关，而不是仅由参数数量决定 [6,7]。

第 3 节的积分方程已经显示了关键小参数：

$$
\text{single-site training feedback}
\ \propto\ \frac1{\sqrt N}.
$$

要让 feature learning 在无穷宽时保留下来，需要重新协调输出缩放与参数学习率，而不是仅仅把隐藏层初始方差改大。

一个便于与 DMFT 对接的参数族是 [3]

$$
f_{\gamma_N}(x)
=
\frac{1}{\gamma_N\sqrt N}
\sum_i a_i\phi(h_i^{(L)}(x)),
\qquad
\frac{d\theta}{dt}
=-\eta\gamma_N^2\nabla_\theta\mathcal L.
$$

仍然对 raw parameters 使用这个学习率，并定义

$$
g_{\gamma_N,\mu i}^{(\ell)}
=
\gamma_N\sqrt N\,
\frac{\partial f_{\gamma_N,\mu}}
{\partial h_{\mu i}^{(\ell)}}.
$$

如此定义的 $g$ 在初始化时保持 $O(1)$。重复第 2、3 节的计算可以看到：预测动力学中的显式 $\gamma_N^2$ 因子与两个输出导数中的 $\gamma_N^{-1}$ 抵消，而单点训练反馈的系数变为

$$
\frac{\eta}{P}\frac{\gamma_N}{\sqrt N}.
$$

需要注意，在这个参数族中，真正的 Jacobian Gram kernel 为

$$
\Theta_{N,\gamma_N}^{\mathrm{Jac}}
=
\frac1{\gamma_N^2}
\left[
\Phi_N^{(L)}
+\sum_{\ell=1}^{L}
\Phi_N^{(\ell-1)}\odot G_{N,\gamma_N}^{(\ell)}
\right],
$$

而实际预测动力学使用的是乘上学习率因子后的 $\gamma_N^2\Theta_{N,\gamma_N}^{\mathrm{Jac}}$。必须区分这两个量，不能遗漏输出缩放。

当 $\gamma_N=1$ 时，回到本文的 NTK scaling。若改为

$$
\gamma_N=\gamma_0\sqrt N,
\qquad \gamma_0>0\text{ 固定},
$$

则单点反馈成为 $O(1)$。对于这里的全连接网络与梯度流，这给出与 $\mu$P 对应的 feature-learning scaling 的一种等价表达 [3,7]。此时，不能再把 $h,z$ 的训练漂移或前后向响应提前设为零；需要求解真正的 two-time order parameters

$$
\Phi^{(\ell)}(t,s),
\qquad
G^{(\ell)}(t,s),
$$

以及由权重复用产生的 causal response kernels。

两个参数化的初始输出也并不相同。在独立零均值 readout 下，$f_{\gamma_N}(0)$ 的协方差在宽度极限中按 $\Phi^{(L)}/\gamma_N^2$ 缩放：$\gamma_N=1$ 保留 $O(1)$ 的随机初始函数，而 $\gamma_N=\gamma_0\sqrt N$ 使初始输出趋于零。因此，讨论从 feature-learning DMFT 取 lazy limit 时，还必须对齐初始条件，不能只比较最终出现的核公式。

从这一角度看，本系列前两篇建立了两层不同的结论。NNGP 用 forward order parameters 描述初始化函数分布；NTK 再加入 backward order parameters，并在训练反馈消失的极限下闭合预测动力学。$\mu$P 所要保留的，恰好是这一极限中被缩放掉的单点特征反馈。

**NTK 不是对所有无穷宽训练的终点，而是完整训练 DMFT 中一个可计算、可检验的基准极限。** 下一步需要研究的，不再只是“固定核是什么”，而是“核如何在误差驱动与响应记忆的共同作用下演化”。

## References

1. <a id="ref1"></a>Arthur Jacot, Franck Gabriel, and Clément Hongler. *Neural Tangent Kernel: Convergence and Generalization in Neural Networks*. NeurIPS, 2018. [arXiv:1806.07572](https://arxiv.org/abs/1806.07572).
2. <a id="ref2"></a>Cengiz Pehlevan and Blake Bordelon. *Lecture Notes on Infinite-Width Limits of Neural Networks*. Princeton Machine Learning Theory Summer School, June 2023. [Lecture notes](https://pehlevan.seas.harvard.edu/file_url/276).
3. <a id="ref3"></a>Blake Bordelon and Cengiz Pehlevan. *Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks*. NeurIPS, 2022. [arXiv:2205.09653](https://arxiv.org/abs/2205.09653).
4. <a id="ref4"></a>Greg Yang. *Tensor Programs II: Neural Tangent Kernel for Any Architecture*. 2020. [arXiv:2006.14548](https://arxiv.org/abs/2006.14548).
5. <a id="ref5"></a>Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. *Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent*. NeurIPS, 2019. [arXiv:1902.06720](https://arxiv.org/abs/1902.06720).
6. <a id="ref6"></a>Lénaïc Chizat, Edouard Oyallon, and Francis Bach. *On Lazy Training in Differentiable Programming*. NeurIPS, 2019. [arXiv:1812.07956](https://arxiv.org/abs/1812.07956).
7. <a id="ref7"></a>Greg Yang and Edward J. Hu. *Feature Learning in Infinite-Width Neural Networks*. ICML, 2021. [arXiv:2011.14522](https://arxiv.org/abs/2011.14522).
8. <a id="ref8"></a>Greg Yang and Etai Littwin. *Architectural Universality of Neural Tangent Kernel Training Dynamics*. ICML, 2021. [arXiv:2105.03703](https://arxiv.org/abs/2105.03703).
