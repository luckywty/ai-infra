> [!abstract] 核心定义
> Sliding Window Attention（SWA）让每个 token 只关注附近 $W$ 个位置，把 full attention 的 $O(S^2)$ 降为 $O(SW)$。它属于结构化 Sparse Attention；Hybrid Attention 则交错使用 local/sparse layers 与少量 full-attention layers，以较低成本保留全局信息通路。

## 1. Full Causal Attention

设：

- $B$：batch size；
- $S$：sequence length；
- $H$：query head 数；
- $d_h$：head dimension；
- $d=Hd_h$：hidden dimension。

Full causal attention：

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
+
M_{\mathrm{causal}}
\right)
$$

$$
O=AV
$$

Causal mask：

$$
M_{ij}
=
\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$$

第 $i$ 个 token 可以访问此前的所有 token，因此 attention score 数量为：

$$
N_{\mathrm{score,full}}
=
O(BHS^2)
$$

主要 attention 计算量为：

$$
\boxed{
C_{\mathrm{full}}
=
O(BS^2d)
}
$$

朴素实现还需保存或物化 $B\times H\times S\times S$ 的 score/probability Tensor。FlashAttention 可以降低中间显存，但不会改变 full attention 的 $S^2$ 算术复杂度。

## 2. Sliding Window Attention

定义第 $i$ 个 causal query 可访问的 key 集合：

$$
\mathcal{N}(i)
=
\left\{
j
\;\middle|\;
0\le j\le i,
\quad
i-j<W
\right\}
$$

即当前 token 最多关注：

$$
i-W+1,\ldots,i
$$

对应 mask：

$$
M_{ij}^{\mathrm{SWA}}
=
\begin{cases}
0,
&
j\le i
\text{ 且 }
i-j<W
\\
-\infty,
&
\text{其他情况}
\end{cases}
$$

每个 query 只与至多 $W$ 个 keys 计算注意力，因此：

$$
N_{\mathrm{score,SWA}}
=
O(BHSW)
$$

$$
\boxed{
C_{\mathrm{SWA}}
=
O(BSWd)
}
$$

当：

$$
W\ll S
$$

计算与 score 存储都显著低于 full attention。

## 3. SWA 是一种结构化 Sparse Attention

Sparse Attention 的一般形式是：每个 query $i$ 只连接到一个稀疏集合 $\mathcal{N}(i)$：

$$
O_i
=
\sum_{j\in\mathcal{N}(i)}
\operatorname{softmax}_{j}
\left(
\frac{q_i^\top k_j}{\sqrt{d_h}}
\right)
v_j
$$

若每个 token 平均只连接 $K$ 个位置：

$$
C_{\mathrm{sparse}}
=
O(BSKd)
$$

常见 sparse pattern：

| 模式                     | 连接方式            | 特点                |
| ---------------------- | --------------- | ----------------- |
| Sliding window         | 最近 $W$ 个 token  | 局部性强、规则、容易优化      |
| Block sparse           | 只访问选定的块         | 适合 GPU tile，但粒度较粗 |
| Strided/dilated        | 按固定间隔访问         | 扩大感受野，但可能漏掉局部细节   |
| Global tokens          | 少量 token 连接所有位置 | 提供全局信息汇聚点         |
| Random/specified links | 额外长距离边          | 改善图连通性，但实现更复杂     |

SWA 的优势是 mask 规则且连续，最容易映射到高效 kernel 和环形 KV Cache。

## 4. 多层 SWA 的感受野

单层只能直接访问最近 $W$ 个 token，但堆叠后信息可以逐层向前传播。

若窗口包含当前位置，并且每层宽度均为 $W$，经过 $L$ 个 local layers 后，理论最远可访问约：

$$
\boxed{
R_L
=
L(W-1)+1
}
$$

即量级：

$$
R_L=O(LW)
$$

但“理论上可达”不等于“直接全局交互”：

- 远距离信息必须经过多层中间 token；
- 路径更长，优化和信息保真更困难；
- 在需要精确检索远处 token 的任务中，纯 SWA 可能弱于 full attention。

## 5. Hybrid Attention：交错 Full 与 Local Layers

Hybrid Attention 使用：

- 大多数层：SWA 或其他 sparse/local attention；
- 少数层：full attention；
- 有时 local layers 使用 [[RoPE]]，full layers 使用 NoPE 或不同的位置编码。

示意：

```text
Layer 1:  SWA
Layer 2:  SWA
Layer 3:  SWA
Layer 4:  Full
Layer 5:  SWA
Layer 6:  SWA
Layer 7:  SWA
Layer 8:  Full
...
```

课件以 Command A 为例：每 4 层插入一次 full attention。LLaMA 4、Gemma 3/4、OLMo 3 等模型也采用过 local/full 混合模式。

Full layer 相当于全局通信层：任意位置的信息可在该层直接交互；中间的 local layers 则以较低成本进行局部处理。

## 6. Hybrid Attention 的计算量

设：

- 总层数 $L$；
- full-attention 层数 $L_f$；
- local/SWA 层数 $L_l=L-L_f$。

忽略 projection 等共同项：

$$
\boxed{
C_{\mathrm{hybrid}}
\approx
O
\left[
Bd
\left(
L_fS^2
+
L_lSW
\right)
\right]
}
$$

相对所有层都使用 full attention：

$$
\frac{
C_{\mathrm{hybrid}}
}{
C_{\mathrm{all\ full}}
}
\approx
\frac{L_f}{L}
+
\frac{L_l}{L}\frac{W}{S}
$$

例如每 4 层有 1 个 full layer：

$$
\frac{L_f}{L}
=
\frac14
$$

当 $W\ll S$ 时，attention 部分的计算量约接近全 full 模型的四分之一，而不是纯 SWA 的最低成本。

> [!note]
> 上述公式只比较 attention 的 $QK^\top$ 与 $AV$ 主项。QKV/O projection、FFN、Norm 和通信成本仍然存在。

## 7. KV Cache

对 MHA/GQA/MQA，设：

- $H_{kv}$：KV head 数；
- 每元素 $b_e$ Bytes。

Full-attention 层必须保留长度 $S$ 的全部 K/V：

$$
M_{\mathrm{KV,full/layer}}
=
2BSH_{kv}d_hb_e
$$

纯 SWA 层若能够安全淘汰窗口外 token，只需保留最近 $W$ 个：

$$
M_{\mathrm{KV,SWA/layer}}
=
2B\min(S,W)H_{kv}d_hb_e
$$

Hybrid 模型：

$$
\boxed{
M_{\mathrm{KV,hybrid}}
=
2BH_{kv}d_hb_e
\left[
L_fS
+
L_l\min(S,W)
\right]
}
$$

这说明：

- SWA 缩短 cache 的**时间维**；
- GQA/MQA 缩小 cache 的**head 维**；
- 两者可以组合，收益近似相乘。

> [!important]
> 只有 local layers 可以丢弃窗口外 KV。Full-attention layers 仍需保存完整历史，否则后续 full layer 无法访问早期 token。

## 8. Position Encoding 的配合

Local attention 只处理短距离相对位置，常与 RoPE 配合：

$$
\text{Local layer}
\rightarrow
\text{RoPE + SWA}
$$

部分 hybrid 架构在 full layer 中使用 NoPE：

$$
\text{Full layer}
\rightarrow
\text{NoPE + Full Attention}
$$

直觉是：

- RoPE 为局部层提供精细的相对位置分辨率；
- full layer 负责长距离信息通信；
- NoPE 避免全局层的极长距离连接受固定 RoPE 频率影响。

这不是统一标准。有些模型的 full 与 local layers 都使用 RoPE；具体选择需要以目标架构为准。

## 9. 正确构造 Local Causal Mask

对绝对 query position $p_q$ 和 key position $p_k$，有效连接必须同时满足：

$$
p_k\le p_q
$$

以及：

$$
p_q-p_k<W
$$

伪代码：

```python
def sliding_causal_mask(
    query_positions,  # [S_q]
    key_positions,    # [S_kv]
    window: int,
):
    q = query_positions[:, None]
    k = key_positions[None, :]

    future = k > q
    outside_window = (q - k) >= window
    return future | outside_window
```

应用：

```python
scores = (
    q @ k.transpose(-2, -1)
) / math.sqrt(head_dim)

mask = sliding_causal_mask(
    query_positions,
    key_positions,
    window,
)

scores = scores.masked_fill(
    mask,
    -torch.inf,
)

probs = torch.softmax(
    scores.float(),
    dim=-1,
).to(q.dtype)

output = probs @ v
```

使用 KV Cache 时，`query_positions` 不能从 0 重新开始，必须包含 cache offset。

## 10. Dense Mask 不等于真正稀疏计算

下面的实现虽然语义正确：

```python
scores = q @ k.transpose(-2, -1)  # [S, S]
scores = scores.masked_fill(
    sliding_mask,
    -torch.inf,
)
```

但它仍然先计算了完整 $S\times S$ score：

$$
C=O(S^2d)
$$

只是在结果上屏蔽窗口外位置，并没有获得 $O(SW)$ 的计算收益。

真正加速需要：

- 原生 sliding-window FlashAttention；
- block-sparse/banded kernel；
- 不物化窗口外 blocks；
- decode 时使用 ring buffer 或截断后的 local KV Cache。

因此：

$$
\boxed{
\text{稀疏 Mask}
\neq
\text{稀疏 Kernel}
}
$$

## 11. 工程折衷

| 方案 | 计算量 | 长程能力 | Kernel 难度 | KV Cache |
|---|---:|---|---|---:|
| Full Attention | $O(S^2d)$ | 最直接 | 成熟 | $O(S)$ / 层 |
| SWA | $O(SWd)$ | 需多层传播 | 较规则 | $O(W)$ / 层 |
| 一般 Sparse Attention | $O(SK d)$ | 取决于连接图 | 较复杂 | 取决于模式 |
| Hybrid Full + SWA | $O(L_fS^2d+L_lSWd)$ | 少量层直接全局 | 中等 | full 层 $O(S)$、local 层 $O(W)$ |

### SWA 的优势

- 计算和显存随上下文近似线性增长；
- decode 时可限制 local-layer KV Cache；
- 连续窗口符合语言局部性；
- pattern 规则，较容易实现高效 kernel。

### SWA 的代价

- 单层缺少全局连接；
- 精确长距离检索需要多跳；
- 窗口 $W$ 太小会损失质量；
- dense masked 实现不会真正加速；
- 不同层的 cache 生命周期更复杂。

### Hybrid 的优势

- 用少量 full layers 恢复全局通信；
- 大多数层仍享受 local attention 的低成本；
- 可独立调节 full-layer 间隔与 window size。

### Hybrid 的代价

- full layers 仍有 $S^2$ 计算；
- full-layer KV Cache 仍随 $S$ 增长；
- kernel 调度、并行策略和 cache 管理更复杂；
- 不同 attention layer 的 position encoding 必须正确匹配。

## 12. 常见误区

1. **给 full attention 加 local mask 就认为复杂度变成 $O(SW)$**

   若仍计算完整 $S\times S$ 矩阵，复杂度仍是 $O(S^2)$。

2. **所有层都只保留 $W$ 个 KV**

   Hybrid 模型的 full layers 必须保留完整历史。

3. **把理论感受野当成直接访问**

   纯 SWA 中的远距离信息需要跨多层传播。

4. **忽略 decode 的绝对位置**

   local mask 和 RoPE 都必须使用带 cache offset 的 position。

5. **窗口边界 off-by-one**

   若窗口包含当前 token，条件通常是：

   $$
   0\le p_q-p_k<W
   $$

   此时最多正好有 $W$ 个有效 positions。

6. **认为 Sparse Attention 一定更快**

   不规则 pattern、较小矩阵或缺少专用 kernel 时，稀疏调度成本可能抵消理论收益。

7. **将 SWA 与 GQA 视为同一种优化**

   SWA 减少时间维 $S\rightarrow W$；GQA 减少 KV heads $H_q\rightarrow H_{kv}$。二者正交且可组合。

## 13. 一句话复习

> [!summary]
> SWA 用局部带状连接把 attention 从 $S^2$ 降为 $SW$，但牺牲单层全局交互；Hybrid Attention 用少量 full layers 作为全局通信层，并让其余 local layers 节省计算与 KV Cache。真正获得加速的前提是使用稀疏 kernel，而不只是 dense attention 加 mask。

## 相关笔记

- [[Lec 3 Architectures, Hyperparameters]]
