> [!abstract] 核心定义
> MHA、GQA、MQA 的主要区别是 Key/Value head 的数量。它们通常保留多个 Query heads，但逐步让更多 Query heads 共享 K/V，从而减少 K/V projection、KV Cache 容量和自回归 decode 的显存带宽；代价是 K/V 表达能力下降。

## 1. Tensor Shape

设：

- $B$：batch size；
- $S_q$：query length；
- $S_{kv}$：KV length；
- $d=d_{\mathrm{model}}$；
- $H_q$：query head 数；
- $H_{kv}$：key/value head 数；
- $d_h$：head dimension；
- 通常 $H_qd_h=d$。

投影后的 Tensor：

$$
Q
\in
\mathbb{R}^{
B\times H_q\times S_q\times d_h
}
$$

$$
K,V
\in
\mathbb{R}^{
B\times H_{kv}\times S_{kv}\times d_h
}
$$

三种结构：

| 结构  |       $H_{kv}$ | K/V 共享方式                      |
| --- | -------------: | ----------------------------- |
| MHA |          $H_q$ | 每个 Query head 有独立 K/V head    |
| GQA | $1<H_{kv}<H_q$ | 一组 Query heads 共享一个 K/V head  |
| MQA |            $1$ | 所有 Query heads 共享同一个 K/V head |

## 2. Multi-Head Attention（MHA）

MHA 通常满足：

$$
H_{kv}=H_q
$$

第 $h$ 个 Query head 与第 $h$ 个 K/V head 配对：

$$
A_h
=
\operatorname{softmax}
\left(
\frac{Q_hK_h^\top}{\sqrt{d_h}}
\right)
$$

$$
O_h=A_hV_h
$$

优点是每个 head 都能学习独立的 query、key 和 value 子空间；缺点是所有 K/V heads 都必须存入 KV Cache。

## 3. Multi-Query Attention（MQA）

MQA 设置：

$$
H_{kv}=1
$$

所有 Query heads 共享同一个 $K,V$：

$$
A_h
=
\operatorname{softmax}
\left(
\frac{Q_hK^\top}{\sqrt{d_h}}
\right)
$$

$$
O_h=A_hV
$$

Query heads 仍然不同，因此每个 head 的 attention weights 仍可不同；共享的是被检索的 key/value 表示。

## 4. Grouped-Query Attention（GQA）

GQA 介于两者之间。定义每个 KV head 服务的 Query heads 数：

$$
G
=
\frac{H_q}{H_{kv}}
$$

通常要求：

$$
H_q\bmod H_{kv}=0
$$

第 $h$ 个 Query head 使用的 KV head：

$$
g(h)
=
\left\lfloor
\frac{h}{G}
\right\rfloor
$$

所以：

$$
A_h
=
\operatorname{softmax}
\left(
\frac{
Q_hK_{g(h)}^\top
}{
\sqrt{d_h}
}
\right)
$$

GQA 通过调节 $H_{kv}$，在 MHA 的表达力与 MQA 的系统效率之间连续取值。

## 5. Projection 参数量

一般情况下：

$$
P_Q=dH_qd_h
$$

$$
P_K=P_V=dH_{kv}d_h
$$

输出 projection：

$$
P_O=H_qd_hd
$$

合计：

$$
\boxed{
P_{\mathrm{attn}}
=
dH_qd_h
+
2dH_{kv}d_h
+
H_qd_hd
}
$$

若 $H_qd_h=d$：

$$
P_{\mathrm{attn}}
=
2d^2
+
2dH_{kv}d_h
$$

MHA：

$$
P_{\mathrm{MHA}}
=
4d^2
$$

MQA：

$$
P_{\mathrm{MQA}}
=
2d^2+2dd_h
$$

GQA 位于两者之间。参数节省主要来自 K/V projection，Q 和 output projection 基本不变。

## 6. KV Cache 大小

缓存 $L$ 层、batch size $B$、上下文长度 $S$，每元素 $b_e$ Bytes：

$$
\boxed{
M_{\mathrm{KV}}
=
2LBSH_{kv}d_hb_e
}
$$

系数 2 对应 K 与 V。

相对 MHA：

$$
\frac{
M_{\mathrm{GQA}}
}{
M_{\mathrm{MHA}}
}
=
\frac{H_{kv}}{H_q}
$$

$$
\frac{
M_{\mathrm{MQA}}
}{
M_{\mathrm{MHA}}
}
=
\frac{1}{H_q}
$$

### 示例

设：

$$
L=32,\quad
B=1,\quad
S=8192,\quad
H_q=32,\quad
d_h=128
$$

使用 BF16：

$$
b_e=2
$$

MHA，$H_{kv}=32$：

$$
M_{\mathrm{MHA}}
=
4\ \mathrm{GiB}
$$

GQA，$H_{kv}=8$：

$$
M_{\mathrm{GQA}}
=
1\ \mathrm{GiB}
$$

MQA，$H_{kv}=1$：

$$
M_{\mathrm{MQA}}
=
128\ \mathrm{MiB}
$$

这会直接提高同一 GPU 上可服务的并发请求数。

## 7. 为什么主要优化 Decode？

Prefill/training 可并行处理多个 tokens，大矩阵乘法具有较高数据复用。自回归 decode 每次只能生成一个新 token，却必须读取历史 KV Cache：

$$
K_{\mathrm{cache}},V_{\mathrm{cache}}
\in
\mathbb{R}^{B\times H_{kv}\times S\times d_h}
$$

上下文越长，每一步需要扫描的 KV 数据越多。GQA/MQA 把这个读流量按：

$$
\frac{H_{kv}}{H_q}
$$

比例缩小，因此对小 batch、长上下文 decode 特别有效。

课件在简化假设下给出 MHA 增量生成的 arithmetic intensity：

$$
I_{\mathrm{decode,MHA}}
\approx
O\left[
\left(
\frac{S}{d}
+
\frac{1}{B}
\right)^{-1}
\right]
$$

小 batch 或长 sequence 时，该值很低，decode 倾向于 memory-bound。

## 8. 为什么不会把 Attention FLOPs 降低 $H_q$ 倍？

每个 Query head 仍需计算：

$$
Q_hK_{g(h)}^\top
$$

以及：

$$
A_hV_{g(h)}
$$

对于单个 decode token，QK 与 AV 的主要 FLOPs 约为：

$$
C_{\mathrm{QK+AV}}
\approx
4BSH_qd_h
$$

若 $H_qd_h=d$：

$$
C_{\mathrm{QK+AV}}
\approx
4BSd
$$

这个 query-side 计算不会因为共享 K/V 而消失。GQA/MQA 的主要收益是：

- K/V projection 更小；
- KV Cache 更小；
- K/V cache 读取流量更低；
- serving batch 与并发容量更大。

## 11. 工程折衷

| 结构 | 质量/表达力 | KV Cache | Decode 带宽 |
|---|---|---|---|
| MHA | 最高自由度 | 最大 | 最大 |
| GQA | 通常接近 MHA | 可调 | 可调 |
| MQA | K/V 共享最强 | 最小 | 最小 |

MQA 有时会带来小幅 perplexity 或任务质量损失；GQA 通常能以少量 KV heads 保留接近 MHA 的质量，因此是现代 serving-oriented LLM 的常见默认选择。

## 12. 常见误区

1. **认为 MQA 只有一个 Query head**

   MQA 仍有多个 Query heads；只有 K/V head 为 1。

2. **物理复制 K/V 后再计算**

   这会抵消 KV Cache 和带宽收益。

3. **KV Cache 公式遗漏层数或 K/V 的系数 2**

   完整公式是：

   $$
   2LBSH_{kv}d_hb_e
   $$

4. **认为 attention FLOPs 按 $H_q/H_{kv}$ 同比例下降**

   Query-side QK 与 AV 仍需对每个 Query head 计算。

5. **忽略整除关系**

   常规 GQA 要求 $H_q$ 能被 $H_{kv}$ 整除。

6. **对重复后的 K 做 QK-Norm 或存 cache**

   应直接处理并缓存原始 $H_{kv}$ 个 K heads。

## 相关笔记

- [[Lec 3 Architectures, Hyperparameters]]
- [[ai infra/02-知识库/模型结构与推理机制/Cached Decode 与 KV Cache#7. GQA Mapping|Cached Decode：GQA Mapping]]
