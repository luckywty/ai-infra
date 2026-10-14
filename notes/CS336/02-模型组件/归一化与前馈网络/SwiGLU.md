# SwiGLU

> [!abstract] 核心定义
> SwiGLU 是一种 gated FFN：一条 projection 经过 Swish/SiLU 形成 gate，另一条 projection 提供 value，两者逐元素相乘后再投影回 residual dimension。它通常比 ReLU/GELU FFN 有小而稳定的质量收益，但必须缩小 $d_{\mathrm{ff}}$ 才能进行公平的参数与计算比较。

## 1. 从普通 FFN 到 Gated FFN

普通 Transformer FFN：

$$
\operatorname{FFN}(x)
=
\phi(xW_1)W_2
$$

其中：

- $x\in\mathbb{R}^{d}$；
- $W_1\in\mathbb{R}^{d\times d_{\mathrm{ff}}}$；
- $W_2\in\mathbb{R}^{d_{\mathrm{ff}}\times d}$；
- $\phi$ 是 ReLU 或 GELU 等激活函数。

SwiGLU 使用三个矩阵：

$$
\boxed{
\operatorname{SwiGLU}(x)
=
\left[
\operatorname{Swish}(xW_g)
\odot
(xW_v)
\right]W_o
}
$$

其中：

- $W_g,W_v\in\mathbb{R}^{d\times d_{\mathrm{ff}}}$；
- $W_o\in\mathbb{R}^{d_{\mathrm{ff}}\times d}$；
- $\odot$：逐元素乘法；
- $W_g$ 产生 gate；
- $W_v$ 产生被 gate 调制的 value。

Swish/SiLU：

$$
\operatorname{Swish}(z)
=
z\sigma(z)
$$

其中 $\sigma(z)$ 是 sigmoid。

## 2. Gate 的直觉

普通 FFN 只有一条非线性通路：

$$
xW_1\rightarrow\phi(\cdot)\rightarrow W_2
$$

SwiGLU 将“生成内容”和“决定内容通过多少”拆成两条可学习投影：

$$
\underbrace{\operatorname{Swish}(xW_g)}_{\text{gate}}
\odot
\underbrace{xW_v}_{\text{value}}
$$

与固定激活函数相比，gate 能根据输入特征逐维调制 FFN 通道，提供更灵活的条件计算。

SwiGLU 与传统 GLU 不完全相同：

$$
\operatorname{GLU}(x)
=
\sigma(xW_g)\odot(xW_v)
$$

SwiGLU 使用 $z\sigma(z)$，不仅有 sigmoid gate，还保留了 $z$ 的幅度和符号信息。

## 3. 为什么常用 $d_{\mathrm{ff}}=\frac{8}{3}d$？

普通 FFN 的主要参数量：

$$
P_{\mathrm{FFN}}
=
dd_{\mathrm{ff}}+d_{\mathrm{ff}}d
=
2dd_{\mathrm{ff}}
$$

经典配置 $d_{\mathrm{ff}}=4d$：

$$
P_{\mathrm{FFN}}
=
2d(4d)
=
8d^2
$$

SwiGLU 有三块矩阵：

$$
P_{\mathrm{SwiGLU}}
=
3d d_{\mathrm{ff}}^{\mathrm{SwiGLU}}
$$

保持参数量近似不变：

$$
3d d_{\mathrm{ff}}^{\mathrm{SwiGLU}}
=
8d^2
$$

得到：

$$
\boxed{
d_{\mathrm{ff}}^{\mathrm{SwiGLU}}
\approx
\frac{8}{3}d
}
$$

因此：

- 普通 ReLU/GELU FFN：通常约 $4d$；
- SwiGLU/GeGLU：通常约 $\frac83d\approx2.67d$；
- 实际模型会取整到 Tensor Core、张量并行或 kernel 友好的倍数。


## 4. 工程折衷

### 优点

- 通常比 ReLU/GELU FFN 获得更好的 loss 或 perplexity；
- gate 能进行输入相关的逐维调制；
- 与无 bias 线性层和 fused projection 很适配；
- 已成为许多现代 Dense LLM 的默认 FFN。

### 成本

- 比普通 FFN 多一块输入 projection；
- gate/value 的逐元素乘法产生额外中间激活；
- 若不缩小 $d_{\mathrm{ff}}$，参数和 FLOPs 会明显增加；
- 若 kernel 未融合，额外 Tensor 写回会增加显存带宽压力。

训练时还需考虑 SwiGLU 中间 activation 的保存或重计算，因为 backward 同时需要 gate 和 value 通路。

## 7. 常见误区

1. **把 SwiGLU 写成 `sigmoid(gate) * value`**

   那是普通 GLU。SwiGLU 应使用：

   ```python
   F.silu(gate) * value
   ```

2. **仍使用 $4d$，却声称参数量不变**

   公平宽度约为 $\frac83d$。

3. **忘记硬件对齐**

   $\frac83d$ 往往不是整数或不是高效 tile 的倍数，实际应取邻近的硬件友好值。

4. **两条分支共用同一 projection**

   标准 SwiGLU 的 $W_g$ 与 $W_v$ 是独立可学习矩阵。

5. **忽略中间激活显存**

   参数量匹配不代表峰值 activation memory 完全相同。


## 相关笔记

- [[Lec 3 Architectures, Hyperparameters]]