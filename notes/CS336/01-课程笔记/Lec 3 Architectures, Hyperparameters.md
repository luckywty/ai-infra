# CS336 Lecture 3：Architectures & Hyperparameters

> [!info] 资料来源
> - 课件：[[CS336/90-资料/课件/lecture_03.pdf]]
> - 课程视频：[Stanford CS336 Spring 2026 - Lecture 3](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&index=3)
> - 课程主页：[CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
>
> 本笔记以 68 页课件为主干，并对 [[RoPE]]、[[Stable Softmax and Z-Loss|z-loss]]、[[QK-Norm]]、[[MHA, MQA and GQA|GQA/MQA]] 算术强度等公式页进行了图像级核对。

> [!abstract] 一句话主线
> 现代 Dense LLM 的主体已经高度收敛为 **Pre-Norm/[[RMSNorm]] + RoPE + [[SwiGLU]] + 无 bias 的串行 Transformer**；真正需要理解的不是“背模型配置”，而是每个变体如何在**优化稳定性、模型表达力、参数/[[CS336/01-课程笔记/Lec 2 PyTorch, Resource Accounting|FLOPs]]、显存带宽和 [[Latency|推理延迟]]**之间做交换。

---

## 1. 🎯 本课核心主题（Core Theme）

本课从原始 Transformer 出发，解释现代 LLM 为什么普遍改用 Pre-Norm、RMSNorm、SwiGLU、RoPE 和无 bias 线性层，并总结 $d_{\mathrm{ff}}/d_{\mathrm{model}}$、head dimension、深宽比、词表大小和正则化等经验规律。

更重要的是，本课把“架构选择”放回系统上下文：z-loss、QK-Norm 等技巧解决大规模训练稳定性，GQA/MQA 与滑动窗口注意力则主要解决自回归推理时 KV Cache 和内存带宽瓶颈。结论是：大多数默认值都有宽阔的有效区间，最终选择必须同时考虑质量、训练稳定性和硬件效率。

---

## 2. 🔑 核心知识要点（Key Concepts & Mechanics）

### 2.1 从原始 Transformer 到现代 Dense LLM

原始 Transformer 的典型配置是：

- [[sinusoidal position embedding]]；
- [[Post-Norm LayerNorm]]；
- ReLU FFN；
- 线性层和 [[LayerNorm]] 均带 bias。

[[CS336课程作业|CS336 Assignment 1]] 采用的简洁现代变体是：

- **Pre-Norm**：归一化放在 attention/FFN 子层之前；
- **[[RMSNorm]]**：不减均值，通常也不含 bias；
- **[[RoPE]]**：在每一层 attention 中旋转 $Q,K$；
- **[[SwiGLU]]**：用 gated FFN 代替 ReLU FFN；
- **No bias**：大部分线性映射不使用 bias；
- 通常仍使用 **Attention → FFN 的串行 block**。

设第 $\ell$ 层输入为 $x_\ell$，一个标准 Pre-Norm block 可写为：

$$
h_\ell
=
x_\ell
+
\operatorname{Attn}\!\left(
\operatorname{Norm}(x_\ell)
\right)
$$

$$
x_{\ell+1}
=
h_\ell
+
\operatorname{FFN}\!\left(
\operatorname{Norm}(h_\ell)
\right)
$$

其中：

- $x_\ell,h_\ell\in\mathbb{R}^{B\times S\times d}$；
- $B$：batch size；
- $S$：sequence length；
- $d=d_{\mathrm{model}}$：residual stream / hidden dimension；
- `Norm` 通常为 RMSNorm。

现代模型的共识不是“所有细节都相同”，而是已经形成一个稳定的 LLaMA-like 主干；显著变化更多出现在 position encoding、attention pattern、QK-Norm 和混合 attention 上。

---

### 2.2 Pre-Norm、Post-Norm 与 Non-residual Post-Norm

对一个子层 $F$，三种常见结构可概括为：

#### [[Post-Norm LayerNorm]]

$$
y=\operatorname{Norm}\!\left(x+F(x)\right)
$$

#### Pre-Norm

$$
y=x+F\!\left(\operatorname{Norm}(x)\right)
$$

#### [[Non-residual Post-Norm and Double Norm]]

只归一化分支输出，而不归一化 residual stream：

$$
y=x+\operatorname{Norm}_{\text{post}}\!\left(F(x)\right)
$$

或者在分支前后都归一化：

$$
y
=
x
+
\operatorname{Norm}_{\text{post}}
\left(
F\!\left(\operatorname{Norm}_{\text{pre}}(x)\right)
\right)
$$

Grok、Gemma 2 等模型使用过 double/sandwich norm；OLMo 2 使用 non-residual post-norm。它们试图在不破坏 residual identity path 的同时，进一步约束分支输出的尺度。

> [!warning] 易混淆点
> “Post-Norm”不只是“Norm 在图上画得更靠后”。关键问题是 Norm 是否包住了 $x+F(x)$：若包住整个和，它就在 residual stream 上；若只作用于 $F(x)$，则是 non-residual post-norm。

---

### 2.3 LayerNorm、RMSNorm 与去除 bias

对一个 token 的隐藏向量 $x\in\mathbb{R}^d$，LayerNorm 定义为：

$$
\mu
=
\frac{1}{d}\sum_{i=1}^{d}x_i
$$

$$
\sigma^2
=
\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
\operatorname{LayerNorm}(x)_i
=
\gamma_i
\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
+
\beta_i
$$

其中：

- $d=d_{\mathrm{model}}$；
- $\mu,\sigma^2$：该 token 在 hidden dimension 上的均值和方差；
- $\gamma,\beta\in\mathbb{R}^d$：可学习的缩放与平移；
- $\epsilon$：防止分母为零的稳定项。

[[RMSNorm]] 不做中心化：

$$
\operatorname{RMS}(x)
=
\sqrt{
\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon
}
$$

$$
\operatorname{RMSNorm}(x)_i
=
\gamma_i\frac{x_i}{\operatorname{RMS}(x)}
$$

这也是本课再次强调的原则：

$$
\boxed{\text{FLOPs 少}\;\not\Rightarrow\;\text{wall-clock time 一定短}}
$$

矩阵乘法占据绝大多数 FLOPs，但 RMSNorm、bias addition 等算子可能是 bandwidth-bound；若它们额外触发一次完整 Tensor 的 HBM 读写，实际耗时仍不可忽略。因此现代模型也普遍删除线性层 bias：

$$
\operatorname{Linear}(x)=xW
$$

而不是：

$$
\operatorname{Linear}(x)=xW+b
$$

去掉 bias 节省的参数和 FLOPs 很小，主要收益来自更简单的 kernel、较少的数据移动和某些情况下更稳定的优化。

---

### 2.4 FFN 激活函数与 Gated Linear Unit

#### 普通 FFN

标准两层 FFN 为：

$$
\operatorname{FFN}(x)
=
\phi(xW_1)W_2
$$

其中：

- $W_1\in\mathbb{R}^{d\times d_{\mathrm{ff}}}$；
- $W_2\in\mathbb{R}^{d_{\mathrm{ff}}\times d}$；
- $\phi$：逐元素激活函数。

常见激活：

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

$$
\operatorname{GELU}(x)=x\Phi(x)
$$

$$
\operatorname{Swish}(x)=x\sigma(x)
$$

其中 $\Phi$ 是标准正态分布 CDF，$\sigma$ 是 sigmoid。

#### GLU 家族

GLU 让一个分支充当逐元素 gate：

$$
\operatorname{FFN}_{\mathrm{GLU}}(x)
=
\left[
\phi(xW_g)\odot(xW_v)
\right]W_o
$$

其中：

- $W_g,W_v\in\mathbb{R}^{d\times d_{\mathrm{ff}}}$；
- $W_o\in\mathbb{R}^{d_{\mathrm{ff}}\times d}$；
- $\odot$：逐元素乘法。

当 $\phi=\operatorname{GELU}$ 时得到 GeGLU；当 $\phi=\operatorname{Swish}$ 时得到 SwiGLU。

#### 为什么 [[SwiGLU]] 常用 $d_{\mathrm{ff}}=\frac{8}{3}d$？

普通 FFN 的主要参数量为：

$$
P_{\mathrm{FFN}}
=
dd_{\mathrm{ff}}+d_{\mathrm{ff}}d
=
2dd_{\mathrm{ff}}
$$

若使用经典 $d_{\mathrm{ff}}=4d$：

$$
P_{\mathrm{FFN}}
=
2d(4d)
=
8d^2
$$

GLU 有三块权重：

$$
P_{\mathrm{GLU}}
=
3dd_{\mathrm{ff}}^{\mathrm{GLU}}
$$

为了与 $4d$ 普通 FFN 保持近似相同的参数量和矩阵乘法计算量，令：

$$
3d d_{\mathrm{ff}}^{\mathrm{GLU}}
=
8d^2
$$

得到：

$$
\boxed{
d_{\mathrm{ff}}^{\mathrm{GLU}}
\approx
\frac{8}{3}d
}
$$

---

### 2.5 串行层与并行层

标准串行 block：

$$
h=x+\operatorname{Attn}(N_1(x))
$$

$$
y=h+\operatorname{FFN}(N_2(h))
$$

并行 block：

$$
y
=
x
+
\operatorname{Attn}(N(x))
+
\operatorname{FFN}(N(x))
$$

并行结构可共享一次 Norm，并把部分 projection 合并成更大的矩阵乘法，缩短关键路径；GPT-J、PaLM、GPT-NeoX、Command A 等模型曾采用这一设计。

代价是同一层中的 FFN 看不到该层 attention 刚产生的结果，串行 block 的层内依赖被削弱。当前多数模型仍选择串行结构，说明理论上的并行度与 kernel fusion 收益并不总能弥补表达能力或训练行为上的差异。

---

### 2.6 Position Encoding：从绝对位置到 [[RoPE]]

课件对比四类方案：

| 方法 | 注入位置 | 直觉 | 局限 |
|---|---|---|---|
| Sinusoidal | 加到 token embedding | 固定频率的绝对位置编码 | 与 token embedding 相加会产生非相对位置的 cross terms |
| Learned absolute | 加可学习向量 $u_i$ | 每个位置学习一个 embedding | 强依赖最大训练长度，天然不是相对位置 |
| Relative bias/embedding | 修改 attention score | 显式依赖 $i-j$ | 不一定仍可写为标准内积，可能影响高效 attention kernel |
| RoPE | 旋转每层的 $Q,K$ | 让 attention 内积只依赖相对位移 | 长度外推仍需频率/base scaling 等额外设计 |

#### [[RoPE]] 的目标

希望位置变换 $f(x,m)$ 满足：

$$
\left\langle
f(q,m),f(k,n)
\right\rangle
=
g(q,k,n-m)
$$

即 attention score 只依赖相对位置，而不依赖绝对位置 $m,n$。

将每两个坐标配成一组，对第 $i$ 对坐标使用二维旋转：

$$
R(m\theta_i)
=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i)\\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
$$

常见频率选择为：

$$
\theta_i
=
\Theta^{-2i/d_h},
\qquad
i=0,\ldots,\frac{d_h}{2}-1
$$

其中：

- $d_h$：单个 attention head 的维度；
- $\Theta$：RoPE base，原始常见值为 $10000$；
- $i$：二维坐标对的编号。

把所有 $2\times2$ 旋转块放到 block-diagonal 矩阵 $R_m$ 中：

$$
q_m'=R_m q,
\qquad
k_n'=R_n k
$$

由于旋转矩阵满足 $R_m^\top R_n=R_{n-m}$：

$$
(q_m')^\top k_n'
=
q^\top R_m^\top R_n k
=
q^\top R_{n-m}k
$$

因此：

$$
\boxed{
(R_mq)^\top(R_nk)
\text{ 只通过 }n-m\text{ 依赖位置}
}
$$

---

### 2.7 超参数共识一：$d_{\mathrm{ff}}/d_{\mathrm{model}}$

对非 gated FFN，最常见经验值为：

$$
\boxed{
d_{\mathrm{ff}}\approx 4d_{\mathrm{model}}
}
$$

对 GLU：

$$
\boxed{
d_{\mathrm{ff}}\approx \frac{8}{3}d_{\mathrm{model}}
}
$$

这不是严格定律。Kaplan 等实验显示，在约 $1\sim10$ 倍之间存在较宽的近最优 basin；T5 11B 甚至使用过：

$$
d_{\mathrm{ff}}=65\,536,
\qquad
d_{\mathrm{model}}=1024
$$

即：

$$
\frac{d_{\mathrm{ff}}}{d_{\mathrm{model}}}=64
$$

模型仍能训练，但后续 T5 v1.1 改回约 $2.5d$ 的 GeGLU，说明“能工作”不代表在固定参数、计算或延迟预算下最优。

---

### 2.8 超参数共识二：Attention head dimension

设：

- $H_q$：query head 数；
- $d_h$：单 head dimension；
- $d=d_{\mathrm{model}}$。

常见设置满足：

$$
\boxed{
H_qd_h\approx d
}
$$

定义比例：

$$
r_{\mathrm{head}}
=
\frac{H_qd_h}{d}
$$

多数模型的 $r_{\mathrm{head}}$ 接近 1，但这只是 projection dimension 的设计选择，不是数学要求。T5、LaMDA、PaLM 等模型曾使用明显大于 1 的值；只要输出 projection 把拼接后的 heads 映射回 $d$ 即可。

需要区分：

- $d$：residual stream dimension；
- $H_qd_h$：所有 query heads 拼接后的 projection dimension；
- 在 GQA/MQA 中，$H_{kv}d_h$ 还可小于 $H_qd_h$。

---

### 2.9 模型深宽比与参数量近似

课件使用：

$$
\rho
=
\frac{d_{\mathrm{model}}}{N_{\mathrm{layer}}}
$$

衡量模型 aspect ratio。多数 Dense LLM 的 $\rho$ 大致落在 $100\sim200$，但较新的模型也会更深。

在 $H_qd_h=d$、忽略 bias 和 Norm 参数时：

- MHA 的 $Q,K,V,O$ projection 约为 $4d^2$ 参数；
- 普通 $4d$ FFN 约为 $8d^2$ 参数；
- $\frac83d$ SwiGLU 也约为 $8d^2$ 参数。

所以每层粗略为：

$$
P_{\mathrm{layer}}
\approx
12d^2
$$

整个模型：

$$
P
\approx
12N_{\mathrm{layer}}d^2
+
Vd
$$

其中：

- $N_{\mathrm{layer}}$：Transformer 层数；
- $V$：词表大小；
- $Vd$：若输入 embedding 与 LM head 权重绑定时的 embedding 参数；若不绑定，相关项约为 $2Vd$。

在固定参数量下，增加深度意味着减小宽度。折衷为：

- **更深**：顺序依赖更长、kernel launch 更多、pipeline parallelism 更难、单 token latency 更高；
- **更宽**：单层矩阵更大，通常更利于 Tensor Core 和层内并行，但 activation、单层通信和 embedding/输出 projection 成本更高；
- 实证上有效区间较宽，因此系统约束经常决定最终 aspect ratio。

---

### 2.10 [[Lec 1 Tokenization|词表大小]]

课件给出的经验范围：

- 单语模型：约 $30\text{k}\sim50\text{k}$；
- 多语或生产系统：约 $100\text{k}\sim250\text{k}$。

词表增大会：

- 提高多语言、代码和稀有字符串的覆盖率；
- 减少同一段文本的 token 数，从而缩短 sequence；
- 增大 embedding / LM head 参数 $Vd$；
- 增大每 token 的输出 softmax 计算与显存访问；
- 使低频 token 获得更少训练样本。

因此词表不是越大越好。多语模型需要较大词表，主要是为了避免不同语言被切得过碎；单语模型通常没有这一压力。

---

### 2.11 [[Dropout]]、[[Weight Decay]] 与“正则化”

大规模预训练常常只对数据做一遍训练，而且 token 数可远大于参数量，传统意义上的训练集过拟合并不是首要问题。实践趋势是：

- 较老模型常用 dropout $\approx0.1$；
- 新模型预训练中经常设 dropout $=0$；
- AdamW weight decay 仍常见，例如 $\lambda=0.1$。

AdamW [[Weight Decay]] 的简化更新可写为：

$$
\theta_{t+1}
=
(1-\eta_t\lambda)\theta_t
-
\eta_t\widehat{u}_t
$$

其中：

- $\eta_t$：第 $t$ 步学习率；
- $\lambda$：weight decay 系数；
- $\widehat{u}_t$：Adam 的归一化更新方向。

整个训练过程的累计缩放近似为：

$$
\theta_T
\propto
\prod_{t=1}^{T}(1-\eta_t\lambda)
$$

所以 weight decay 与 cosine learning-rate schedule 强耦合。它在 LLM 预训练中的作用不应只理解为“防止训练集过拟合”，还会改变权重尺度和优化动力学。

---

### 2.12 稳定性技巧一：[[Stable Softmax and Z-Loss]]

对 logits $z\in\mathbb{R}^V$：

$$
p_i
=
\frac{e^{z_i}}{Z},
\qquad
Z=\sum_{j=1}^{V}e^{z_j}
$$

数值稳定实现应使用：

$$
m=\max_jz_j
$$

$$
\operatorname{softmax}(z)_i
=
\frac{e^{z_i-m}}
{\sum_j e^{z_j-m}}
$$

交叉熵也应通过 `logsumexp` 计算：

$$
\mathcal{L}_{\mathrm{CE}}
=
-z_y+\log\sum_{j=1}^{V}e^{z_j}
$$

Softmax 对所有 logits 同加常数 $c$ 不敏感：

$$
\operatorname{softmax}(z+c\mathbf{1})
=
\operatorname{softmax}(z)
$$

因此模型可能在不改变概率的情况下让 logits 的共同偏移不断增大。z-loss 显式约束 partition function：

$$
\log Z
=
\operatorname{logsumexp}(z)
$$

$$
\boxed{
\mathcal{L}
=
\mathcal{L}_{\mathrm{CE}}
+
\alpha(\log Z)^2
}
$$

其中 $\alpha$ 是很小的系数；PaLM 使用过 $\alpha=10^{-4}$。它把 $\log Z$ 拉向 0，抑制输出 logits 的无意义整体漂移，从而提高训练稳定性。

---

### 2.13 稳定性技巧二：[[QK-Norm]]

Attention logits 为：

$$
A
=
\frac{QK^\top}{\sqrt{d_h}}
$$

如果 $Q,K$ 的范数持续增长，点积会变得很大，softmax 进入极端饱和区。QK-Norm 在点积前对每个 head 的 query 和 key 做 LayerNorm 或 RMSNorm：

$$
\widehat{Q}
=
\operatorname{Norm}(Q)
$$

$$
\widehat{K}
=
\operatorname{Norm}(K)
$$

$$
A
=
\frac{\widehat{Q}\widehat{K}^{\top}}
{\sqrt{d_h}}
$$

它直接控制 attention logits 的尺度，已被 DCLM、OLMo 2、Gemma 2、Qwen 3 等模型采用。

QK-Norm 的成本是额外参数、逐元素运算和访存；若不能与 Q/K projection 或 RoPE 融合，可能增加 wall-clock time。它也改变了 attention 依赖向量范数的能力，因此仍是稳定性与表达力的折衷。

---

### 2.14 稳定性技巧三：[[Logit Soft-capping]]

对任意 attention 或 output logit $z$：

$$
\boxed{
\widetilde{z}
=
c\tanh\left(\frac{z}{c}\right)
}
$$

其中 $c>0$ 是 soft cap。该变换保证：

$$
-c<\widetilde{z}<c
$$

当 $|z|\ll c$ 时，$\widetilde{z}\approx z$；当 $|z|\gg c$ 时，logit 平滑饱和。其导数为：

$$
\frac{\partial\widetilde{z}}{\partial z}
=
\operatorname{sech}^2\left(\frac{z}{c}\right)
$$

---

### 2.15 [[MHA, MQA and GQA]]

设：

- $H_q$：query head 数；
- $H_{kv}$：key/value head 数；
- $d_h$：head dimension；
- 通常 $H_qd_h=d$。

投影后的形状：

$$
Q\in\mathbb{R}^{B\times S_q\times H_q\times d_h}
$$

$$
K,V\in\mathbb{R}^{B\times S_{kv}\times H_{kv}\times d_h}
$$

三种结构：

| 结构 | $H_{kv}$ | 含义 |
| :--- | ---: | :--- |
| MHA | $H_q$ | 每个 query head 有独立的 K/V head |
| GQA | $1<H_{kv}<H_q$ | 一组 query heads 共享一个 K/V head |
| MQA | $1$ | 所有 query heads 共享同一个 K/V head |

GQA 的 group size 为：

$$
G
=
\frac{H_q}{H_{kv}}
$$

实现通常要求 $H_q\bmod H_{kv}=0$。第 $h$ 个 query head 使用：

$$
g(h)
=
\left\lfloor
\frac{h}{G}
\right\rfloor
$$

号 K/V head。

#### KV Cache 大小

若缓存 $L$ 层、batch size 为 $B$、上下文长度为 $S$，每元素 $b_e$ Bytes：

$$
\boxed{
M_{\mathrm{KV}}
=
2LB S H_{kv}d_hb_e
}
$$

系数 2 对应 K 和 V。

相对 MHA 的缓存比例：

$$
\frac{M_{\mathrm{GQA}}}{M_{\mathrm{MHA}}}
=
\frac{H_{kv}}{H_q}
$$

$$
\frac{M_{\mathrm{MQA}}}{M_{\mathrm{MHA}}}
=
\frac{1}{H_q}
$$

例：$L=32$、$B=1$、$S=8192$、$H_q=32$、$d_h=128$、BF16（$b_e=2$）：

$$
M_{\mathrm{MHA}}
=
2\cdot32\cdot1\cdot8192\cdot32\cdot128\cdot2
=
4\ \mathrm{GiB}
$$

若 GQA 使用 $H_{kv}=8$：

$$
M_{\mathrm{GQA}}=1\ \mathrm{GiB}
$$

若使用 MQA：

$$
M_{\mathrm{MQA}}=128\ \mathrm{MiB}
$$

这会直接提高 serving 中可容纳的并发请求数。

---

### 2.17 [[Sliding Window, Sparse and Hybrid Attention]]

Full attention 的 score 数量和主要计算量为：

$$
N_{\mathrm{score}}
=
O(BHS^2)
$$

$$
C_{\mathrm{full}}
=
O(BS^2d)
$$

若每个 token 只关注最近 $W$ 个 token：

$$
C_{\mathrm{SWA}}
=
O(BSWd)
$$

当 $W\ll S$ 时可显著降低训练和推理成本；对于纯 SWA 层，如果旧 KV 可以安全淘汰，cache 也可从 $S$ 缩到 $W$。

代价是单层不再提供全局任意 token 对之间的直接连接。连续 $L$ 个局部层的理论 receptive field 可逐层扩张到约 $LW$，但信息必须经过多跳传播。

当前常见方案是交错：

- 大多数层使用 RoPE + sliding-window/local attention；
- 每隔若干层插入一次 full attention；
- 某些模型在 full layer 使用 NoPE，让长距离连接不受局部 RoPE 频率限制。

例如 Command A 每 4 层设置一层 full attention；LLaMA 4、Gemma 3/4、OLMo 3 等也采用局部与全局层混合。这个设计用少量 $O(S^2)$ 层维持长程信息通路，其余层换取更低的计算和 cache 成本。

---

## 3. ⚡ 重难点与工程折衷（Difficulties & Engineering Trade-offs）

### 3.1 架构共识不是理论最优证明

$4d$ FFN、$\frac83d$ SwiGLU、$H_qd_h=d$、$d/N_{\mathrm{layer}}\approx100\sim200$ 都是经过大量模型验证的稳健默认值，但并非不可违反的约束。T5 的极端配置说明很宽的区域都“能训练”；真正问题应是：

$$
\text{在固定参数、训练 FLOPs、显存和延迟下，哪种配置的 loss 最低？}
$$

不要把“很多模型这样设置”误读成“数学上必须这样设置”。

---

### 3.2 Pre-Norm 更稳，但不是免费的

- Pre-Norm 保留干净的 residual identity path，深层模型更容易训练；
- Post-Norm 可能让每层输出尺度更受控，但梯度传播更脆弱；
- double norm / QK-Norm 增加稳定性，却引入更多逐元素 kernel、参数和内存流量；
- 对小模型或已稳定训练的配置，额外 Norm 未必带来净收益。

稳定性改动必须用训练曲线、gradient norm、loss spikes 和 wall-clock [[Throughput|throughput]] 联合评估。

---

### 3.3 [[RMSNorm]] 的收益来自数据移动，而不只是 FLOPs

$$
\boxed{\text{RMSNorm 减少的 FLOPs 很少，但它能简化归约、减少参数读取，并利于减少中间张量搬运}}
$$

LayerNorm 与 RMSNorm 相比整个 Transformer 的矩阵乘法，FLOPs 差距很小；但 Norm 本身的 arithmetic intensity 低，容易被 HBM bandwidth 支配。RMSNorm 减少 reduction、bias 和中间量，更容易 fuse，因此可带来可观的 wall-clock 收益。

这也是为什么“按总 FLOPs 看占比很低”的算子仍值得优化。

---

### 3.4 [[SwiGLU]] 的宽度必须按参数预算比较

若直接把普通 FFN 的 $4d$ 宽度原样用于 SwiGLU：

$$
P_{\mathrm{SwiGLU}}=3d(4d)=12d^2
$$

比普通 $4d$ FFN 的 $8d^2$ 多 50%。此时质量提升不能全部归因于 gating，因为模型也变大了。公平比较应使用约 $\frac83d$，或者显式保持训练 FLOPs/参数量不变。

---

### 3.5 [[RoPE]] 提供相对内积，但不自动解决无限长度外推

RoPE 保证同一组旋转频率下的 attention 内积依赖相对位移，但训练中未见过的角度、低频覆盖不足和高频 aliasing 仍会影响超长上下文。改变 RoPE base、position interpolation 或只旋转部分维度会同时改变：

- 短距离位置分辨率；
- 长距离相位变化速度；
- 与训练长度分布的匹配。

因此“RoPE 是相对位置编码”与“模型可以无损扩展到任意长度”是两件事。

---

### 3.6 z-loss、QK-Norm、soft-cap 解决的是不同失稳源

#### [[Stable Softmax and Z-Loss]]

| 方法 | 主要约束对象 | 优势 | 风险 |
| :--- | :--- | :--- | :--- |
| z-loss | 输出 softmax 的 $\log Z$ | 消除 logits 公共偏移漂移 | $\alpha$ 过大时限制输出分布 |
| QK-Norm | attention 的 $Q,K$ 范数 | 限制 attention score 尺度 | 增加访存，改变范数信息 |
| soft-cap | attention/output logit 的绝对值 | 直接提供有界 logits | 饱和区梯度趋零，可能伤质量 |

它们可以组合，但不能互相替代。首先还应保证 `logsumexp`、减最大值、[[混合精度]] 等基本数值实现正确。

---

### 3.7 MHA → GQA → MQA：表达力与内存带宽

- MHA：K/V 表达最充分，但 cache 最大；
- MQA：cache 和带宽最省，但所有 query heads 共享同一 K/V 表示，可能有小幅 perplexity 损失；
- GQA：用 $H_{kv}$ 连续调节质量和系统效率，实践中往往获得更好的折衷。

在训练/prefill 阶段，大矩阵乘法可充分复用数据，GQA 的 wall-clock 优势不一定显著；在长上下文、小 batch decode 中，KV Cache 扫描是关键瓶颈，收益会放大。

---

### 3.8 Full Attention → SWA：二次复杂度与长程能力

#### [[Sliding Window, Sparse and Hybrid Attention]]

SWA 把 $S^2$ 降为 $SW$，但牺牲一步全局交互。混合 full/local 层的本质是让少量昂贵层充当全局通信总线。

工程上还应区分：

- 理论 sparse pattern 是否有高效 kernel；
- cache 能否真正淘汰窗口外 token；
- full layer 的数量是否足以支持任务所需的长程依赖；
- local 与 full layer 使用何种位置编码。

只有复杂度下降而没有对应 kernel 和 cache 管理，未必能获得实际加速。

---

### 3.9 词表大小：短序列不等于低成本

更大词表可能让文本 token 数下降，从而减少 attention 的 $S^2$ 成本；但会增加 $Vd$ embedding 参数和每 token 的 vocabulary projection/softmax。最优值取决于语言覆盖、平均压缩率、是否 weight tying、训练语料和推理任务，而不是单独最小化 token 数。

---

## 5. 🧪 课后自测三问（Self-Check Questions）

### 问题 1：公平比较 ReLU FFN 与 SwiGLU

一个模型的 $d_{\mathrm{model}}=4096$。原 FFN 使用 ReLU 和 $d_{\mathrm{ff}}=4d$，忽略 bias。

1. 原 FFN 每层有多少参数？
2. 若改成 SwiGLU，并保持 FFN 参数量不变，$d_{\mathrm{ff}}$ 应取多少？
3. 为什么直接沿用 $d_{\mathrm{ff}}=4d$ 不是公平架构对比？

> [!answer]- 参考答案
> 原 FFN：
>
> $$
> P=2dd_{\mathrm{ff}}=2d(4d)=8d^2
> $$
>
> 代入 $d=4096$：
>
> $$
> P=8\cdot4096^2=134\,217\,728
> $$
>
> 即约 134.2M 参数。
>
> SwiGLU 有三块权重：
>
> $$
> 3d d_{\mathrm{ff}}^{\mathrm{GLU}}=8d^2
> $$
>
> 所以：
>
> $$
> d_{\mathrm{ff}}^{\mathrm{GLU}}
> =\frac83d
> \approx10\,922.7
> $$
>
> 实现中应取接近且对硬件友好的整数，如 10,880 或 11,008，具体取决于对齐约束。
>
> 若直接使用 $4d$，SwiGLU 参数为 $12d^2$，比原 FFN 多 50%；质量差异会混入模型容量和计算量差异。

### 问题 2：证明 RoPE 的相对位置性质，并找出 cache bug

设 $q_m'=R_mq$、$k_n'=R_nk$，其中 $R_m,R_n$ 是 RoPE block-diagonal 旋转矩阵。

1. 证明 $(q_m')^\top k_n'$ 只依赖 $n-m$。
2. 在已有 1024 个 cached tokens 时生成第 1025 个 token，如果仍用 `position_id=0` 旋转新 $Q,K$，会发生什么？

> [!answer]- 参考答案
> 二维旋转矩阵满足：
>
> $$
> R_m^\top=R_{-m},
> \qquad
> R_{-m}R_n=R_{n-m}
> $$
>
> 因此：
>
> $$
> (q_m')^\top k_n'
> =q^\top R_m^\top R_nk
> =q^\top R_{n-m}k
> $$
>
> 只通过相对位移 $n-m$ 依赖位置。
>
> decode 时把新 token 重新设为位置 0，会让它与 cache 中历史 K 的相对角度全部错误；模型会把第 1025 个 token 当成序列开头的位置。正确做法是使用 cache length 作为 offset，即新 token 的绝对位置应为 1024（若从 0 计数）。

### 问题 3：KV Cache 与 decode 瓶颈

设模型有 $L=32$ 层、$H_q=32$、$d_h=128$，使用 BF16 KV Cache。batch size 为 1，上下文长度为 8192。

1. 分别计算 MHA（$H_{kv}=32$）、GQA（$H_{kv}=8$）和 MQA（$H_{kv}=1$）的 KV Cache 大小。
2. 为什么 MQA 能显著提高 decode 吞吐，却不会把 attention FLOPs 降低 32 倍？
3. 如果模型质量不允许使用 MQA，应如何连续调节这一折衷？

> [!answer]- 参考答案
> 使用：
>
> $$
> M_{\mathrm{KV}}
> =2LBSH_{kv}d_hb_e,
> \qquad b_e=2
> $$
>
> 得到：
>
> $$
> M_{\mathrm{MHA}}=4\ \mathrm{GiB}
> $$
>
> $$
> M_{\mathrm{GQA}}=1\ \mathrm{GiB}
> $$
>
> $$
> M_{\mathrm{MQA}}=128\ \mathrm{MiB}
> $$
>
> MQA 主要减少 K/V projection、cache 容量和每步读取 K/V 的带宽；32 个 query heads 仍各自计算 attention scores，并各自产生加权输出，因此 attention 的 query-side 计算仍存在。
>
> 可使用 GQA，把 $H_{kv}$ 设为 $1$ 与 $H_q$ 之间且能整除 $H_q$ 的数，例如 2、4、8、16；通过验证 perplexity、任务质量、tokens/s 和最大并发数选择折衷点。

---

## 复习速记

> [!summary] 现代 Dense LLM 默认配置
> - Block：Pre-Norm、串行 Attention → FFN；
> - Norm：RMSNorm，线性层通常无 bias；
> - FFN：SwiGLU，$d_{\mathrm{ff}}\approx\frac83d$；
> - Position：每层对 $Q,K$ 使用 RoPE；
> - Heads：通常 $H_qd_h\approx d$，推理模型常用 GQA；
> - Stability：stable softmax，必要时使用 z-loss、QK-Norm 或 soft-cap；
> - Attention：长上下文中常混合 full 与 sliding-window layers；
> - 原则：先保证稳定与质量，再用真实 shape、bytes moved 和 wall-clock [[benchmark]] 做系统优化。
