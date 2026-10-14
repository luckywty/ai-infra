# QK-Norm

> [!abstract] 核心定义
> QK-Norm 在 attention 点积之前，沿每个 head 的 $d_h$ 维分别归一化 Query 和 Key。它通过控制 $\|Q\|$、$\|K\|$ 来限制 attention logits 的尺度，减少 softmax 饱和、梯度尖峰和低精度训练失稳。

## 1. Attention Logits 的尺度问题

单个 attention head 的 score：

$$
a_{mn}
=
\frac{q_m^\top k_n}{\sqrt{d_h}}
$$

其中：

- $q_m,k_n\in\mathbb{R}^{d_h}$；
- $m$：query 位置；
- $n$：key 位置；
- $d_h$：head dimension。

由 Cauchy-Schwarz：

$$
|q_m^\top k_n|
\le
\|q_m\|_2\|k_n\|_2
$$

即使除以 $\sqrt{d_h}$，如果训练过程中 $Q,K$ 的范数持续增长，attention logits 仍会变得极端。Softmax 会趋向 one-hot：

$$
\operatorname{softmax}(a)
\rightarrow
\text{近似 one-hot}
$$

这可能带来：

- softmax 饱和；
- 大部分位置梯度接近 0；
- 少数位置梯度尖峰；
- BF16/FP16 数值风险；
- 深层模型训练 loss spike。

## 2. QK-Norm 的定义

先做 projection：

$$
q=xW_Q,
\qquad
k=xW_K
$$

再按 head dimension 归一化：

$$
\widehat q
=
\operatorname{Norm}_Q(q)
$$

$$
\widehat k
=
\operatorname{Norm}_K(k)
$$

attention score 变为：

$$
\boxed{
a_{mn}
=
\frac{
\widehat q_m^\top
\widehat k_n
}{
\sqrt{d_h}
}
}
$$

`Norm` 可以是 LayerNorm 或 RMSNorm。现代 LLM 中常使用 per-head RMSNorm。

## 3. Per-head RMS QK-Norm

对 $q\in\mathbb{R}^{d_h}$：

$$
\operatorname{RMS}(q)
=
\sqrt{
\frac{1}{d_h}
\sum_{i=1}^{d_h}q_i^2
+
\epsilon
}
$$

$$
\widehat q_i
=
\gamma_{q,i}
\frac{q_i}{\operatorname{RMS}(q)}
$$

$K$ 同理，并通常有独立的 $\gamma_k$。

如果忽略可学习 scale，RMS-normalized 向量满足：

$$
\|\widehat q\|_2
\approx
\sqrt{d_h}
$$

$$
\|\widehat k\|_2
\approx
\sqrt{d_h}
$$

因此：

$$
\left|
\frac{\widehat q^\top\widehat k}
{\sqrt{d_h}}
\right|
\le
\sqrt{d_h}
$$

它不是把 logits 严格限制到 $[-1,1]$，但会消除由任意 $Q,K$ 范数增长带来的失控尺度。

## 4. Tensor Shape

常见 shape：

```text
Q: [B, H_q, S_q, D_h]
K: [B, H_kv, S_kv, D_h]
```

QK-Norm 应沿最后一维 `D_h` 进行：

```python
q = q_norm(q)  # reduction over D_h
k = k_norm(k)  # reduction over D_h
```

在 GQA/MQA 中：

- $H_q$ 与 $H_{kv}$ 可以不同；
- Q-Norm 参数对应 query heads；
- K-Norm 参数对应 KV heads；
- 不应为了归一化而先把 K 复制成 $H_q$ 份。

## 6. QK-Norm 与 RoPE 的顺序

典型流程之一：

$$
x
\rightarrow
Q/K\text{ projection}
\rightarrow
\text{QK-Norm}
\rightarrow
\text{RoPE}
\rightarrow
QK^\top
$$

也有模型使用：

$$
Q/K\text{ projection}
\rightarrow
\text{RoPE}
\rightarrow
\text{QK-Norm}
$$

纯旋转保持二范数：

$$
\|R_mq\|_2=\|q\|_2
$$

因此没有逐维可学习 scale 时，标量式 RMS normalization 与旋转在范数意义上相容。但带逐维 $\gamma$ 时，scale 与旋转通常不交换：

$$
R_m(\gamma\odot q)
\neq
\gamma\odot(R_mq)
$$

加载预训练权重时必须严格遵循目标架构的顺序。

## 7. 与其他稳定性方法的区别

| 方法 | 约束对象 | 是否改变训练目标 |
|---|---|---|
| Stable Softmax | 指数计算过程 | 否 |
| z-loss | 输出 softmax 的 $\log Z$ | 是 |
| QK-Norm | attention 的 $Q,K$ 范数 | 是，改变 attention 函数 |
| Logit Soft-Capping | attention/output logits 绝对值 | 是 |

QK-Norm 从源头限制点积尺度；soft-cap 则在点积已经得到以后压缩 logits。二者可以组合，但计算和表达力成本也会叠加。

## 8. 工程折衷

### 优点

- 降低 attention logits 爆炸和 softmax 饱和风险；
- 改善深层、大学习率和低精度训练稳定性；
- 可减少 loss spike；
- 对 $Q,K$ 的尺度漂移提供直接约束。

### 成本

- 增加两个 normalization 操作；
- Norm 属于 bandwidth-bound 算子，额外数据读写可能影响吞吐；
- 若不能与 Q/K projection 或 RoPE 融合，kernel launch 成本明显；
- 删除了 attention 利用绝对 $Q,K$ 范数表达置信度的一部分能力；
- learned scale、attention temperature 和 $1/\sqrt{d_h}$ 之间会相互作用。

## 9. 常见误区

1. **沿 head 维或 sequence 维归一化**

   QK-Norm 一般沿每个 head 内部的 `D_h` 维归一化。

2. **对 $V$ 也做 QK-Norm**

   名称中的 QK 表明标准做法只处理 $Q,K$。

3. **认为它等于 cosine attention**

   RMSNorm 后再除 $\sqrt{d_h}$ 并不严格等于单位二范数 cosine similarity，尤其还有可学习 $\gamma$。

4. **忽略 RoPE 顺序**

   有可学习逐维 scale 时，Norm 与旋转一般不可交换。

5. **在 GQA 中复制 K 后再归一化**

   应直接对原始 $H_{kv}$ 个 K heads 归一化。

## 相关笔记

- [[Lec 3 Architectures, Hyperparameters]]
