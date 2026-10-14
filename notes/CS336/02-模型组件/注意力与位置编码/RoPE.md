# RoPE

> [!abstract] 核心定义
> RoPE（Rotary Position Embedding）不是把位置向量加到 token embedding 上，而是在**每一层 attention** 中，按照 token 位置旋转投影后的 $Q$ 和 $K$。旋转矩阵的代数性质使 attention 内积只通过相对位移 $n-m$ 依赖位置。

## 1. 设计目标

希望位置变换 $f(x,m)$ 满足：

$$
\left\langle
f(q,m),f(k,n)
\right\rangle
=
g(q,k,n-m)
$$

其中：

- $q,k\in\mathbb{R}^{d_h}$：某个 attention head 的 query 和 key；
- $m,n$：两个 token 的绝对位置；
- $d_h$：head dimension。

也就是说，attention score 应感知两个 token 相隔多远，而不是单独依赖它们各自的绝对位置。

## 2. 二维旋转

RoPE 把 hidden coordinates 两两配对。对第 $i$ 对坐标使用旋转矩阵：

$$
R(m\theta_i)
=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i)\\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
$$

常见频率定义为：

$$
\theta_i
=
\Theta^{-2i/d_h},
\qquad
i=0,\ldots,\frac{d_h}{2}-1
$$

其中：

- $\Theta$：RoPE base，原始常见值为 $10000$；
- 较小的 $i$ 对应较高频率；
- 较大的 $i$ 对应较低频率；
- $d_h$ 通常必须为偶数。

把所有二维旋转放入 block-diagonal 矩阵 $R_m$：

$$
q_m'=R_mq,
\qquad
k_n'=R_nk
$$

## 3. 为什么只依赖相对位置？

旋转矩阵满足：

$$
R_m^\top=R_{-m}
$$

以及：

$$
R_{-m}R_n=R_{n-m}
$$

因此：

$$
\begin{aligned}
(q_m')^\top k_n'
&=
(R_mq)^\top(R_nk)\\
&=
q^\top R_m^\top R_nk\\
&=
q^\top R_{n-m}k
\end{aligned}
$$

所以：

$$
\boxed{
(R_mq)^\top(R_nk)
\text{ 只通过 }n-m\text{ 依赖位置}
}
$$

这就是 RoPE 的核心性质。

同时，旋转是正交变换：

$$
\|R_mq\|_2=\|q\|_2
$$

因此 RoPE 改变方向而不改变向量的二范数。

## 4. 与其他位置编码的区别

| 方法               | 操作                            | 位置性质                                   |
| ---------------- | ----------------------------- | -------------------------------------- |
| Sinusoidal       | $x_m+PE_m$                    | 固定绝对位置；内积中含 token-position cross terms |
| Learned absolute | $x_m+u_m$                     | 学习每个绝对位置                               |
| Relative bias    | 给 attention score 加 $b_{m-n}$ | 显式相对位置，但会修改 attention score 计算         |
| RoPE             | $Q,K$ 分别乘 $R_m,R_n$           | 保留标准内积结构，并产生相对旋转                       |

RoPE 不是“只在输入 embedding 处做一次”的位置编码。每一层都会生成新的 $Q,K$，因此每一层都必须重新应用旋转。

## 5. 工程折衷

### 优点

- 通过标准点积自然表达相对位置；
- 不需要额外的 learned position embedding table；
- 容易集成进高效 attention kernel；
- 旋转不改变 $Q,K$ 的二范数。

### 局限

- 相对位置性质不等于无限长度外推能力；
- 训练长度之外的旋转角度可能未被模型见过；
- 高频维度在长距离上旋转过快，可能出现 aliasing；
- 改变 base、缩放 position 或只旋转部分维度，会改变短程分辨率与长程覆盖之间的平衡。

长上下文扩展常调整 $\Theta$、使用 position interpolation/NTK-aware scaling，或只对部分 head dimensions 使用 RoPE；这些方法都需要重新验证短上下文质量。

## 8. 常见误区

1. **旋转 $V$**

   标准 RoPE 只旋转 $Q,K$，通常不旋转 $V$。

2. **只在输入 embedding 处应用**

   RoPE 应用于每层投影后的 $Q,K$。

3. **忽略 KV Cache offset**

   decode 的 position ID 必须延续历史长度。

4. **坐标配对 convention 不一致**

   会导致加载的预训练权重完全失配。

5. **把相对位置性质等同于任意长度泛化**

   代数性质成立不代表模型在未见过的上下文长度上仍保持性能。

## 9. 一句话复习

> [!summary]
> RoPE 用位置相关的二维旋转作用于每层 $Q,K$，借助 $R_m^\top R_n=R_{n-m}$ 把绝对位置差转化为 attention 内积中的相对位移。

## 相关笔记

- [[Lec 3 Architectures, Hyperparameters]]
- [[ai infra/02-知识库/模型结构与推理机制/Cached Decode 与 KV Cache#5. Position / Slot / Length 专题|Cached Decode：Position / Slot / Length]]
