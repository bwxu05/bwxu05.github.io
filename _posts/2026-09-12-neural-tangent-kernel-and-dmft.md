---
layout: post
title: Neural Tangent Kernel and Lazy Training
date: 2026-09-12
description: 无穷宽神经网络的训练动力学：NTK 的 DMFT 推导
tags: pre-training scaling muP DMFT NTK
categories: muP-Theory
---

上一篇 *Neural Network Initializations and NNGP* 用单点高斯过程与特征核描述了宽网络的随机初始化。这篇继续讨论：**所有层参与训练后，网络的预测如何演化？**

在 NTK parameterization 下，无穷宽网络的训练动力学由一个固定的 Neural Tangent Kernel (NTK) 控制 [1,5]。本文从有限宽神经网络出发，通过 DMFT 计算这个核的演化动力学，解释它为什么在训练中冻结。

## 1. Network and training

考虑 $L$ 个隐藏层、宽度均为 $N$、无 bias 的全连接网络。输入维度为 $D$，训练集为 $\lbrace(x_\mu,y_\mu)\rbrace_{\mu=1}^{P}$：

$$
\begin{aligned}
h_{\mu i}^{(1)}(t)
&=\frac{1}{\sqrt D}\sum_jW_{ij}^{(1)}(t)x_{\mu j},\\
h_{\mu i}^{(\ell)}(t)
&=\frac{1}{\sqrt N}\sum_jW_{ij}^{(\ell)}(t)\phi(h_{\mu j}^{(\ell-1)}(t)),
\qquad 2\leq\ell\leq L,\\
f_\mu(t)
&=\frac{1}{\sqrt N}\sum_i a_i(t)\phi(h_{\mu i}^{(L)}(t)).
\end{aligned}
$$

初始化的 $W_{ij}^{(\ell)}(0),a_i(0)$ 相互独立且服从 $\mathcal N(0,1)$。优化变量是这些 **raw parameters**，而 $D^{-1/2},N^{-1/2}$ 是前向计算中的显式因子。对 raw parameters 使用宽度无关的统一学习率，是本文的 NTK parameterization [1,5]；改变参数坐标却保持相同数值的学习率，一般会改变训练动力学。

采用平均平方损失与 full-batch gradient flow，记 $\Delta_\mu=y_\mu-f_\mu$：

$$
\mathcal L=\frac{1}{2P}\sum_\mu\Delta_\mu^2,
\qquad
\frac{d\theta}{dt}
=-\eta\nabla_\theta\mathcal L
=\frac{\eta}{P}\sum_\mu\Delta_\mu\nabla_\theta f_\mu.
$$

以下固定 $P,D,L,\eta$ 与有限时间区间 $[0,T]$，再取 $N\to\infty$。动力学讨论以具有适当正则性和矩控制的光滑 activation 为主，例如 $\tanh$；ReLU 的训练极限还需控制 activation gates 的切换 [5,8]。

## 2. The finite-width NTK

链式法则给出精确的预测动力学 [1]：

$$
\Theta_{N,\mu\nu}(t)
=\nabla_\theta f_\mu(t)\cdot\nabla_\theta f_\nu(t),
\qquad
\frac{d\boldsymbol\Delta}{dt}
=-\frac{\eta}{P}\Theta_N(t)\boldsymbol\Delta.
$$

$\Theta_N=JJ^\top$ 是 parameter Jacobian 的 Gram matrix，因此半正定，且 $d\mathcal L/dt=-\eta\boldsymbol\Delta^\top\Theta_N\boldsymbol\Delta/P^2\leq0$。此时核仍可随训练变化。

为分解这个核，定义 $O(1)$ 的 backward field：

$$
\begin{aligned}
g_{\mu i}^{(\ell)}
&=\sqrt N\,\frac{\partial f_\mu}{\partial h_{\mu i}^{(\ell)}}
=\phi'(h_{\mu i}^{(\ell)})z_{\mu i}^{(\ell)},\\
z_{\mu i}^{(\ell)}
&=\frac{1}{\sqrt N}\sum_jW_{ji}^{(\ell+1)}g_{\mu j}^{(\ell+1)},
\qquad \ell<L,
\qquad z_{\mu i}^{(L)}=a_i.
\end{aligned}
$$

相应的 forward 与 backward order parameters 为 [2,3]

$$
\begin{aligned}
\Phi_{N,\mu\nu}^{(\ell)}(t,s)
&=\frac1N\sum_i\phi(h_{\mu i}^{(\ell)}(t))\phi(h_{\nu i}^{(\ell)}(s)),\\
G_{N,\mu\nu}^{(\ell)}(t,s)
&=\frac1N\sum_i g_{\mu i}^{(\ell)}(t)g_{\nu i}^{(\ell)}(s),
\qquad
\Phi_{\mu\nu}^{(0)}=\frac{x_\mu^\top x_\nu}{D}.
\end{aligned}
$$

由于隐藏层参数导数分别为 $g_{\mu i}^{(\ell)}\phi(h_{\mu j}^{(\ell-1)})/N$（$\ell\geq2$）与 $g_{\mu i}^{(1)}x_{\mu j}/\sqrt{ND}$，逐层求和得到有限宽恒等式：

$$
\Theta_N(t)
=\Phi_N^{(L)}(t,t)
+\sum_{\ell=1}^{L}
\Phi_N^{(\ell-1)}(t,t)\odot G_N^{(\ell)}(t,t).
$$

这里 $\odot$ 是逐元素乘法，$\Phi_N^{(0)}=\Phi^{(0)}$。第一项来自 readout，其余各项来自隐藏层；推导尚未使用无穷宽或独立性假设。

## 3. DMFT derivation of the NTK

### 3.1 Path Integral Construction

DMFT 对初始化权重取 quenched disorder average。先积分参数更新，再代回前后向传播，可将动力学写成 [2,3]

$$
\begin{aligned}
h_{\mu i}^{(\ell)}(t)
={}&\chi_{\mu i}^{(\ell)}(t)
+\frac{\eta}{P\sqrt N}\sum_\nu\int_0^t ds\,
\Delta_\nu(s)\Phi_{N,\mu\nu}^{(\ell-1)}(t,s)g_{\nu i}^{(\ell)}(s),\\
z_{\mu i}^{(\ell)}(t)
={}&\xi_{\mu i}^{(\ell)}(t)
+\frac{\eta}{P\sqrt N}\sum_\nu\int_0^t ds\,
\Delta_\nu(s)G_{N,\mu\nu}^{(\ell+1)}(t,s)\phi(h_{\nu i}^{(\ell)}(s)).
\end{aligned}
$$

其中，内部层的无序场是

$$
\chi_{\mu i}^{(\ell)}(t)
=\frac1{\sqrt N}\sum_jW_{ij}^{(\ell)}(0)\phi(h_{\mu j}^{(\ell-1)}(t)),
\qquad
\xi_{\mu i}^{(\ell)}(t)
=\frac1{\sqrt N}\sum_jW_{ji}^{(\ell+1)}(0)g_{\mu j}^{(\ell+1)}(t).
$$

第一层使用 $\chi_{\mu i}^{(1)}=D^{-1/2}\sum_jW_{ij}^{(1)}(0)x_{\mu j}$；输出端使用 $\xi_{\mu i}^{(L)}=a_i(0)$ 与 $G_{N,\mu\nu}^{(L+1)}(t,s)=1$。$\chi(t)$ 作用在时刻 $t$ 的 features 上，因此并不自动等于初始化 preactivation。

显式训练反馈都带有 $N^{-1/2}$。这提示一个静态的 lazy 极限，但尚不能代替训练稳定性论证。**下面先计算初始化的联合单点理论，省略 $t=0$；它在训练中的适用性将在 3.5 节说明。**

沿用上一篇的 Fourier-source 约定，引入只作用于有限个 sites 的 $b,c$：

$$
Z[b,c]
=\mathbb E_{W,a}\exp\left\{
i\sum_{\ell,\mu,i}
\left[b_{\mu i}^{(\ell)}h_{\mu i}^{(\ell)}
+c_{\mu i}^{(\ell)}z_{\mu i}^{(\ell)}\right]
\right\}.
$$

对内部矩阵 $W^{(\ell)}$，路径积分必须同时约束

$$
\mathbf h_\mu^{(\ell)}
=\frac1{\sqrt N}W^{(\ell)}\phi(\mathbf h_\mu^{(\ell-1)}),
\qquad
\mathbf z_\mu^{(\ell-1)}
=\frac1{\sqrt N}W^{(\ell)\top}\mathbf g_\mu^{(\ell)}.
$$

用 $\delta(y)=\int d\hat y\,e^{i\hat y y}/(2\pi)$ 引入 $\hat h,\hat z$，则与该矩阵有关的指数为

$$
-\frac{i}{\sqrt N}\sum_{i,j}W_{ij}^{(\ell)}
\sum_\mu\left[
\hat h_{\mu i}^{(\ell)}\phi(h_{\mu j}^{(\ell-1)})
+g_{\mu i}^{(\ell)}\hat z_{\mu j}^{(\ell-1)}
\right].
$$

前向与反向项乘在**同一个随机权重**上，不能预先把它们视为独立。

### 3.2 Expectation over Random Initializations

利用 $\mathbb E_{W\sim\mathcal N(0,1)}e^{-iWA}=e^{-A^2/2}$，逐元素平均后得到

$$
\exp\left\{
-\frac1{2N}\sum_{i,j}
\left[\sum_\mu
\left(
\hat h_{\mu i}^{(\ell)}\phi(h_{\mu j}^{(\ell-1)})
+g_{\mu i}^{(\ell)}\hat z_{\mu j}^{(\ell-1)}
\right)\right]^2
\right\}.
$$

除了 $\Phi_N,G_N$，平方中的交叉项还引入

$$
U_{\mu\nu}^{(\ell)}
=\frac1N\sum_i\hat h_{\mu i}^{(\ell)}g_{\nu i}^{(\ell)},
\qquad
V_{\mu\nu}^{(\ell-1)}
=\frac1N\sum_i\phi(h_{\mu i}^{(\ell-1)})\hat z_{\nu i}^{(\ell-1)}.
$$

因此，上式的指数可以重写为

$$
\begin{aligned}
&-\frac12\sum_{\mu,\nu}
\Phi_{N,\mu\nu}^{(\ell-1)}
\hat{\mathbf h}_\mu^{(\ell)}\cdot\hat{\mathbf h}_\nu^{(\ell)}\\
&-\frac12\sum_{\mu,\nu}
G_{N,\mu\nu}^{(\ell)}
\hat{\mathbf z}_\mu^{(\ell-1)}\cdot\hat{\mathbf z}_\nu^{(\ell-1)}
-N\sum_{\mu,\nu}U_{\mu\nu}^{(\ell)}V_{\mu\nu}^{(\ell-1)}.
\end{aligned}
$$

前两项产生高斯场的协方差；第三项记录权重正向与转置复用产生的 response coupling [3]。它也是 $O(N)$ 的作用量项，不能仅凭原式中的 $1/N$ 将其忽略。

第一层与 readout 的平均分别提供固定边界 $\Phi^{(0)}$ 与 $G^{(L+1)}=\mathbf1\mathbf1^\top$。后者不是 $I_P$：不同样本共享同一个 $a_i$。

### 3.3 Single-site Moment Generating Functional and DMFT Action

把 $\Phi,G,U,V$ 提升为独立序参量，统记为 $\mathcal Q$。对每个独立分量插入约束

$$
1\propto\int dQ\,d\hat Q\,
\exp\left\{iN\hat Q\left[Q-\frac1N\sum_iO_i\right]\right\}.
$$

为紧凑地写出单点积分，记第 $\ell$ 层的局部观测为

$$
\mathcal O^{(\ell)}
=\left\{
\phi(h_\mu)\phi(h_\nu),\;
g_\mu g_\nu,\;
\hat h_\mu g_\nu,\;
\phi(h_\mu)\hat z_\nu
\right\},
\qquad g_\mu=\phi'(h_\mu)z_\mu.
$$

它们分别对应 $\Phi^{(\ell)},G^{(\ell)},U^{(\ell)},V^{(\ell)}$；其中 $U$ 只存在于 $2\leq\ell\leq L$，$V$ 只存在于 $1\leq\ell<L$。以下 $\hat{\mathcal Q}\cdot\mathcal Q$ 对各序参量的独立分量求和，$\Phi,G$ 只计独立的对称矩阵元。

无序平均后的生成泛函具有标准的鞍点形式：

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

单点生成泛函为

$$
\begin{aligned}
\mathcal Z_i^{(\ell)}
={}&\int\frac{d^P\mathbf h\,d^P\hat{\mathbf h}\,
 d^P\mathbf z\,d^P\hat{\mathbf z}}{(2\pi)^{2P}}
\exp\Bigg\{
i\hat{\mathbf h}^{\top}\mathbf h
+i\hat{\mathbf z}^{\top}\mathbf z\\
&\quad-\frac12\hat{\mathbf h}^{\top}\Phi^{(\ell-1)}\hat{\mathbf h}
-\frac12\hat{\mathbf z}^{\top}G^{(\ell+1)}\hat{\mathbf z}
-i\hat{\mathcal Q}^{(\ell)}\cdot\mathcal O^{(\ell)}\\
&\quad+i\mathbf b_i^{(\ell)\top}\mathbf h
+i\mathbf c_i^{(\ell)\top}\mathbf z
\Bigg\}.
\end{aligned}
$$

这一步将神经元之间的耦合转移到了宏观序参量上。有限个 source insertions 只贡献 $O(1)$ 的修正，leading-order 的 bulk saddle 由零 source 理论决定。

### 3.4 Saddle Point Equations

当 $N\to\infty$ 时，考察满足

$$
\frac{\delta S}{\delta\mathcal Q}=0,
\qquad
\frac{\delta S}{\delta\hat{\mathcal Q}}=0
$$

的物理鞍点。对共轭序参量变分给出单点自洽条件

$$
\mathcal Q^{(\ell)}=\left\langle\mathcal O^{(\ell)}\right\rangle.
$$

特别地，$\Phi^{(\ell)}=\langle\phi(\mathbf h)\phi(\mathbf h)^\top\rangle$，$G^{(\ell)}=\langle\mathbf g\mathbf g^\top\rangle$。尖括号表示单点有效理论上的平均，不是样本平均。

现在检查混合项，而不是直接丢弃它。对 $U,V$ 变分得到

$$
i\hat U_{\mu\nu}^{(\ell)}=V_{\mu\nu}^{(\ell-1)},
\qquad
i\hat V_{\mu\nu}^{(\ell-1)}=U_{\mu\nu}^{(\ell)}.
$$

在独立、零均值 readout 对应的候选静态解上，令

$$
\mathbf h^{(\ell)}=\mathbf u^{(\ell)},
\qquad
\mathbf z^{(\ell)}=\mathbf r^{(\ell)},
\qquad
\mathbf u^{(\ell)}\perp\mathbf r^{(\ell)},
$$

其中 $\mathbf u,\mathbf r$ 是中心高斯向量。分部积分将混合序参量与平均交叉响应联系起来；在此解上，

$$
\left\langle
\frac{\partial\phi(u_\mu)}{\partial r_\nu}
\right\rangle=0,
\qquad
\left\langle
\frac{\partial[\phi'(u_\mu)r_\mu]}{\partial u_\nu}
\right\rangle
=\delta_{\mu\nu}\left\langle\phi''(u_\mu)r_\mu\right\rangle=0.
$$

因此 $U=V=\hat U=\hat V=0$ 与鞍点方程相容。此时零 source 单点积分归一化为 $1$，与协方差的取值无关，也给出 $\hat\Phi=\hat G=0$。这是静态解的 formal saddle-point 自洽检查；并非仅由此就证明训练中的核冻结。

单点理论由此化为

$$
\begin{gathered}
\mathbf u^{(\ell)}\sim\mathcal N(0,\Phi^{(\ell-1)}),
\qquad
\mathbf r^{(\ell)}\sim\mathcal N(0,G^{(\ell+1)}),
\qquad \mathbf u^{(\ell)}\perp\mathbf r^{(\ell)},\\
h_\mu^{(\ell)}=u_\mu^{(\ell)},
\qquad
g_\mu^{(\ell)}=\phi'(u_\mu^{(\ell)})r_\mu^{(\ell)}.
\end{gathered}
$$

高斯的是 $h$ 与 pre-gradient $z$，一般不是 $g$。将单点解代回自洽条件，定义导数核 $\Psi^{(\ell)}$，得到

$$
\begin{aligned}
\Phi_{\mu\nu}^{(\ell)}
&=\mathbb E_{\mathbf u\sim\mathcal N(0,\Phi^{(\ell-1)})}
[\phi(u_\mu)\phi(u_\nu)],\\
\Psi_{\mu\nu}^{(\ell)}
&=\mathbb E_{\mathbf u\sim\mathcal N(0,\Phi^{(\ell-1)})}
[\phi'(u_\mu)\phi'(u_\nu)],\\
G^{(\ell)}
&=\Psi^{(\ell)}\odot G^{(\ell+1)},
\qquad G^{(L+1)}=\mathbf1\mathbf1^\top.
\end{aligned}
$$

先前向计算 $\Phi,\Psi$，再反向计算 $G$，即可闭合整个静态单点理论 [1–4]。

### 3.5 Final NTK Theory

将单点核代回有限宽分解，得到确定性的极限 NTK：

$$
\Theta_\infty
=\Phi^{(L)}
+\sum_{\ell=1}^{L}\Phi^{(\ell-1)}\odot G^{(\ell)}.
$$

每层贡献都是进入该层的 feature overlap 与通向输出的 sensitivity overlap 之积。等价的前向递推是 [1,4]

$$
\Theta^{(0)}=\Phi^{(0)},
\qquad
\Theta^{(\ell)}=\Phi^{(\ell)}+\Psi^{(\ell)}\odot\Theta^{(\ell-1)},
\qquad
\Theta_\infty=\Theta^{(L)}.
$$

**从初始化核到训练中的固定核，还需要稳定性。** 在固定 $P,D,L,T$、适当矩控制与前后向传播稳定的条件下，3.1 节的 $N^{-1/2}$ 反馈使典型单点场的有限时间变化趋于零。结合初始化自平均，得到 [5,8]

$$
\sup_{0\leq t\leq T}
\left\lVert\Theta_N(t)-\Theta_\infty\right\rVert
\xrightarrow[N\to\infty]{\mathbb P}0.
$$

相应地，$\Phi^{(\ell)}(t,s)=\Phi^{(\ell)}$、$G^{(\ell)}(t,s)=G^{(\ell)}$。不同训练时刻保留的是同一个初始化单点场，而不是重新采样的白噪声。

平方损失下，固定核动力学因此可以直接求解 [1]：

$$
\boldsymbol\Delta(t)
=\exp\left(-\frac{\eta t}{P}\Theta_\infty\right)\boldsymbol\Delta(0),
\qquad
\mathbf f(t)=\mathbf y-\boldsymbol\Delta(t).
$$

若 $\Theta_\infty v_k=\kappa_kv_k$，$v_k$ 正交归一，且 $c_k=v_k^\top\boldsymbol\Delta(0)$，则

$$
\mathcal L(t)
=\frac1{2P}\sum_kc_k^2
\exp\left(-\frac{2\eta\kappa_k}{P}t\right).
$$

各个核特征模态按自己的时间尺度衰减；零特征值上的误差不会消失。一般可微损失仍有固定核动力学 $d\mathbf f/dt=-\eta\Theta_\infty\nabla_{\mathbf f}\mathcal L$，但不再具有上述矩阵指数解。

**Lazy training 不等于不训练，也不等于只训练 readout。** 单个神经元的变化虽趋于零，大量由同一误差信号驱动的贡献仍能累积出 $O(1)$ 的预测变化。极限动力学对应初始化处的 tangent model，即对参数增量线性化，而不是对输入线性化 [5,6]。

最后，NNGP 与 NTK 的角色不同：

$$
\mathbf f(0)\Longrightarrow\mathcal N(0,\Phi^{(L)}),
\qquad
K_{\mathrm{NNGP}}=\Phi^{(L)}.
$$

NNGP 决定随机初始函数的协方差，NTK 决定其训练演化。$\Theta_\infty$ 是确定性的，并不意味着 $\mathbf f(0)$ 也自平均为零。

## 4. Two checks

**One hidden layer.** 当 $L=1$ 时，

$$
\Theta_\infty=\Phi^{(1)}+\Phi^{(0)}\odot\Psi^{(1)}.
$$

第一项来自 readout，第二项来自输入权重。只训练 readout 才得到 random-feature kernel $\Phi^{(1)}$。对于 ReLU 与 $\Phi_{\mu\mu}^{(0)}=1$，高斯积分给出 $\Phi_{\mu\mu}^{(1)}=\Psi_{\mu\mu}^{(1)}=1/2$，因此 $\Theta_{\infty,\mu\mu}=1$；这里没有额外的 He initialization 因子。

**Deep linear network.** 当 $\phi(u)=u$ 时，$\Phi^{(\ell)}=\Phi^{(0)}$，$\Psi_{\mu\nu}^{(\ell)}=G_{\mu\nu}^{(\ell)}=1$，因此

$$
\Theta_\infty=(L+1)\Phi^{(0)}.
$$

$L$ 个隐藏层与一个 readout 各贡献一次输入核，核对了参数块计数与归一化。

## 5. From NTK to feature learning and muP

NTK 的 lazy behavior 来自 scaling，而不只是宽度。为看清被缩放掉的训练反馈，考虑参数族 [3]

$$
f_{\gamma_N}(x)
=\frac1{\gamma_N\sqrt N}\sum_i a_i\phi(h_i^{(L)}(x)),
\qquad
\frac{d\theta}{dt}=-\eta\gamma_N^2\nabla_\theta\mathcal L.
$$

定义 $g_{\gamma_N,\mu i}^{(\ell)}=\gamma_N\sqrt N\,\partial f_{\gamma_N,\mu}/\partial h_{\mu i}^{(\ell)}$，重复前面的计算，单点反馈系数变为

$$
\frac{\eta}{P}\frac{\gamma_N}{\sqrt N}.
$$

$\gamma_N=1$ 对应本文的 NTK scaling；$\gamma_N=\gamma_0\sqrt N$、固定 $\gamma_0>0$，则保留 $O(1)$ 的单点训练反馈。对于这里的全连接网络与梯度流，后者是与 $\mu$P 对应的 feature-learning scaling 的一种等价表达 [3,7]。

实际的 Jacobian Gram kernel 带有 $\gamma_N^{-2}$ 因子，预测动力学中的学习率因子 $\gamma_N^2$ 将其抵消，不能混淆这两个量。两种 scaling 的初始条件也不同：独立零均值 readout 下，初始化输出协方差按 $\Phi^{(L)}/\gamma_N^2$ 缩放，因此 feature-learning scaling 的初始输出趋于零。

当单点反馈不再消失时，静态核不再足以描述训练，必须追踪 $\Phi^{(\ell)}(t,s)$、$G^{(\ell)}(t,s)$ 与由权重复用产生的 causal response kernels。**NTK 是完整训练 DMFT 中一个可解的基准极限；下一篇将研究被这一极限缩放掉的特征演化。**

## References

1. <a id="ref1"></a>Arthur Jacot, Franck Gabriel, and Clément Hongler. *Neural Tangent Kernel: Convergence and Generalization in Neural Networks*. NeurIPS, 2018. 
2. <a id="ref2"></a>Cengiz Pehlevan and Blake Bordelon. *Lecture Notes on Infinite-Width Limits of Neural Networks*. Princeton Machine Learning Theory Summer School, June 2023. 
3. <a id="ref3"></a>Blake Bordelon and Cengiz Pehlevan. *Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks*. NeurIPS, 2022.
4. <a id="ref4"></a>Greg Yang. *Tensor Programs II: Neural Tangent Kernel for Any Architecture*. 2020.
5. <a id="ref5"></a>Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. *Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent*. NeurIPS, 2019.
6. <a id="ref6"></a>Lénaïc Chizat, Edouard Oyallon, and Francis Bach. *On Lazy Training in Differentiable Programming*. NeurIPS, 2019. 
7. <a id="ref7"></a>Greg Yang and Edward J. Hu. *Feature Learning in Infinite-Width Neural Networks*. ICML, 2021.
8. <a id="ref8"></a>Greg Yang and Etai Littwin. *Architectural Universality of Neural Tangent Kernel Training Dynamics*. ICML, 2021.
