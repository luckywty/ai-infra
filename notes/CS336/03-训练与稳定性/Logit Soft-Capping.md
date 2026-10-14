> [!abstract] 核心定义
> Logit Soft-Capping 使用 $c\tanh(z/c)$ 将 attention 或 vocabulary logits 平滑限制在 $(-c,c)$ 内。它能阻止极端 logits 和 softmax 过度饱和，但会压缩概率差异，并让饱和区梯度趋近于 0。

## 1. 数学定义

对任意 logit $z$：

$$
\boxed{
\widetilde z
=
c\tanh\left(\frac{z}{c}\right)
}
$$

其中 $c>0$ 是 soft cap。

其范围为：

$$
-c<\widetilde z<c
$$

因此无论原始 $z$ 多大，传给 softmax 的有限 logit 都不会超过 cap。

## 2. 为什么叫 Soft Cap？

Hard clipping：

$$
\widetilde z
=
\operatorname{clip}(z,-c,c)
$$

在阈值处不可导，并且阈值外梯度严格为 0。

Soft-capping 使用平滑、单调的 `tanh`：

$$
\frac{\partial\widetilde z}{\partial z}
=
\operatorname{sech}^2
\left(
\frac{z}{c}
\right)
$$

在 $z=0$ 附近：

$$
\frac{\partial\widetilde z}{\partial z}
\approx
1
$$

在 $|z|\gg c$ 时：

$$
\frac{\partial\widetilde z}{\partial z}
\rightarrow
0
$$

因此它是平滑饱和，而不是突然截断。

## 3. 小 Logit 区域近似

利用：

$$
\tanh u
=
u-\frac{u^3}{3}+O(u^5)
$$

得到：

$$
\widetilde z
=
z
-
\frac{z^3}{3c^2}
+
O\left(\frac{z^5}{c^4}\right)
$$

当 $|z|\ll c$ 时：

$$
\widetilde z\approx z
$$

所以常规范围内的 logits 几乎不受影响；只有接近或超过 $c$ 的极端值被明显压缩。

## 4. 应用位置

### Attention Logits

原始 score：

$$
A
=
\frac{QK^\top}{\sqrt{d_h}}
$$

Soft-capped score：

$$
\widetilde A
=
c_{\mathrm{attn}}
\tanh
\left(
\frac{A}{c_{\mathrm{attn}}}
\right)
$$

然后再加入 causal/padding mask 并计算 softmax。

### Vocabulary Logits

LM head 输出：

$$
z=W_{\mathrm{LM}}h
$$

可变为：

$$
\widetilde z
=
c_{\mathrm{out}}
\tanh
\left(
\frac{z}{c_{\mathrm{out}}}
\right)
$$

再送入 cross-entropy。Attention 与 output logits 可以使用不同的 cap。

## 5. Mask 顺序是关键

错误做法：

```python
scores = scores.masked_fill(mask, -torch.inf)
scores = cap * torch.tanh(scores / cap)
```

因为：

$$
\tanh(-\infty)=-1
$$

原本的 $-\infty$ 会变成有限值 $-c$，被 mask 的位置可能重新获得非零 softmax 概率。

正确顺序：

```python
scores = cap * torch.tanh(scores / cap)
scores = scores.masked_fill(mask, -torch.inf)
probs = torch.softmax(scores.float(), dim=-1)
```

即：

$$
\boxed{
\text{先 soft-cap 有限 logits，再应用 }-\infty\text{ mask}
}
$$

## 6. 为什么有助于稳定？

Softmax 对 logit 差异呈指数敏感：

$$
\frac{p_i}{p_j}
=
e^{z_i-z_j}
$$

若某些 logit 远大于其他值，概率会极度集中，非最大项梯度接近 0。Soft-cap 压缩极端差异：

$$
|\widetilde z_i-\widetilde z_j|
<
2c
$$

从而：

- 防止 attention 过早变成近似 one-hot；
- 降低极端 outlier 对 softmax 的支配；
- 减少低精度下的数值风险；
- 限制最终输出分布的过度尖锐程度。

## 8. 工程折衷

### 优点

- logits 有显式有限范围；
- 相比 hard clip 更平滑；
- 对正常范围 logits 近似恒等；
- 可以独立调节 attention cap 和 output cap。

### 成本

- 饱和区导数趋近 0，极端 logit 难以恢复；
- 会改变 softmax 概率和训练目标；
- cap 太小会压缩模型表达能力，可能恶化 perplexity；
- `tanh` 带来额外逐元素计算和访存；
- 若 kernel 未融合，可能降低吞吐。

## 9. 与 z-loss、QK-Norm 的区别

| 方法 | 介入位置 | 主要作用 |
|---|---|---|
| z-loss | output CE 辅助目标 | 约束 $\log Z$ 与共同 logit offset |
| QK-Norm | $QK^\top$ 之前 | 从向量范数源头控制 attention score |
| Soft-cap | logits 产生之后 | 直接把 logit 平滑限制到有限范围 |

Soft-cap 最直接，但也最强地改变概率分布；QK-Norm 更偏向结构性尺度控制；z-loss 主要用于输出 softmax 的训练目标。

## 10. 常见误区

1. **在 soft-cap 前加入 $-\infty$ mask**

   `tanh(-inf)` 会把 mask 变回有限值。

2. **认为 soft-cap 后不需要 stable softmax**

   Stable softmax 仍是标准实现，尤其需要处理 mask 和低精度。

3. **attention 与 output 使用同一个 cap**

   两类 logits 的分布和作用不同，应独立调节。

4. **认为单调变换不改变 softmax**

   Soft-cap 保留 logit 排序，但会改变差值，因此会改变概率。

5. **cap 越小越稳定、效果越好**

   更小 cap 会带来更强梯度饱和和表达限制。

## 相关笔记

- [[Lec 3 Architectures, Hyperparameters]]
