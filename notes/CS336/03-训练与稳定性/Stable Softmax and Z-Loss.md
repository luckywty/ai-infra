> [!abstract] 核心定义
> Stable Softmax 通过减去最大 logit 和使用 `logsumexp`，在**不改变数学结果**的前提下避免指数溢出；z-loss 则是在交叉熵之外增加 $\alpha(\log Z)^2$，主动约束 softmax partition function。前者是数值实现技巧，后者会改变训练目标，两者不能混为一谈。

## 1. Softmax 为什么容易数值失稳？

给定 vocabulary logits：

$$
z=(z_1,\ldots,z_V)\in\mathbb{R}^{V}
$$

Softmax 为：

$$
p_i
=
\frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
$$

定义 partition function：

$$
Z(z)
=
\sum_{j=1}^{V}e^{z_j}
$$

直接计算存在两个问题：

- $z_i$ 很大时，$e^{z_i}$ 溢出为 `inf`；
- $z_i$ 很小时，$e^{z_i}$ 下溢为 0；
- 在 BF16/FP16 中，这些问题更容易出现；
- `inf / inf`、`0 / 0` 会产生 `NaN`。

## 2. Stable Softmax：减去最大值

Softmax 对所有 logits 同加或同减一个常数保持不变：

$$
\operatorname{softmax}(z+c\mathbf{1})
=
\operatorname{softmax}(z)
$$

令：

$$
m=\max_j z_j
$$

则：

$$
\boxed{
p_i
=
\frac{e^{z_i-m}}
{\sum_{j=1}^{V}e^{z_j-m}}
}
$$

此时：

$$
z_i-m\le 0
$$

至少一个指数项等于 1，其余位于 $(0,1]$，从而避免正指数溢出。

> [!important]
> 减最大值是精确恒等变换，不改变理想数学结果；它解决的是计算机浮点表示问题。

## 3. Stable Log-Softmax 与 Cross-Entropy

目标类别为 $y$ 时：

$$
\log p_y
=
z_y-\log Z(z)
$$

负对数似然：

$$
\mathcal{L}_{\mathrm{CE}}
=
-z_y+\log Z(z)
$$

稳定的 `logsumexp`：

$$
\boxed{
\operatorname{LSE}(z)
=
m
+
\log
\sum_{j=1}^{V}e^{z_j-m}
}
$$

所以：

$$
\mathcal{L}_{\mathrm{CE}}
=
-z_y+\operatorname{LSE}(z)
$$

实现中应直接使用 `cross_entropy` 或 `log_softmax`，不要先算 softmax 再取 log：

```python
# 推荐：内部使用稳定的 log-softmax。
loss = F.cross_entropy(
    logits.float(),
    targets,
)

# 不推荐：
probs = torch.softmax(logits, dim=-1)
loss = -torch.log(probs[target])
```

## 4. 仅有 Stable Softmax 为什么还不够？

交叉熵对共同平移同样不敏感：

$$
\mathcal{L}_{\mathrm{CE}}
(z+c\mathbf{1},y)
=
\mathcal{L}_{\mathrm{CE}}(z,y)
$$

也就是说，交叉熵只关心 logits 之间的相对差异，不约束它们的共同 offset。

但：

$$
\log Z(z+c\mathbf{1})
=
c+\log Z(z)
$$

模型可能在概率几乎不变的情况下，让 logits 的共同尺度或偏移逐渐漂移。稳定的 `logsumexp` 能让本次 forward 不溢出，却不会阻止网络继续产生越来越极端的 logits。

## 5. z-loss

z-loss 给 partition function 增加辅助惩罚：

$$
\mathcal{L}_{z}
=
\alpha
\left(
\log Z(z)-z_0
\right)^2
$$

课件与 PaLM 风格通常令目标 $z_0=0$：

$$
\boxed{
\mathcal{L}
=
\mathcal{L}_{\mathrm{CE}}
+
\alpha
\left[
\operatorname{logsumexp}(z)
\right]^2
}
$$

其中：

- $\alpha$：很小的权重，例如 $10^{-4}$；
- $\log Z$：每个 token 的 vocabulary log-partition；
- z-loss 主要固定 softmax 对共同平移不敏感所留下的自由度。

## 6. z-loss 的梯度

因为：

$$
\frac{\partial \log Z}{\partial z_i}
=
p_i
$$

所以：

$$
\boxed{
\frac{\partial\mathcal{L}_{z}}{\partial z_i}
=
2\alpha
\log Z
\cdot
p_i
}
$$

当 $\log Z>0$ 时，z-loss 倾向于压低 logits；当 $\log Z<0$ 时，倾向于抬高 logits。它主要阻止 partition function 无意义漂移，但在有限模型和有限 $\alpha$ 下仍会对相对 logits 产生一定影响。

## 7. PyTorch 实现

```python
def cross_entropy_with_z_loss(
    logits: torch.Tensor,   # [B, S, V]
    targets: torch.Tensor,  # [B, S]
    valid_mask: torch.Tensor,
    alpha: float = 1e-4,
):
    logits_fp32 = logits.float()

    ce = F.cross_entropy(
        logits_fp32.flatten(0, 1),
        targets.flatten(),
        reduction="none",
    ).view_as(targets)

    log_z = torch.logsumexp(
        logits_fp32,
        dim=-1,
    )

    per_token_loss = ce + alpha * log_z.square()
    return (
        per_token_loss[valid_mask].sum()
        / valid_mask.sum().clamp_min(1)
    )
```

## 8. 工程折衷

### 优点

- 抑制输出 logits 的共同偏移和无意义尺度漂移；
- 降低大规模、低精度训练中的 loss spike 风险；
- 计算成本相对较低，因为 CE 本来就需要 `logsumexp`；
- 不需要限制类别之间的相对 logit 差异到固定区间。

### 成本

- $\alpha$ 过大会干扰主交叉熵目标；
- 若实现重复计算 `logsumexp`，会增加额外开销；
- padding/ignored tokens 处理不一致会改变有效 loss 权重；
- z-loss 不是 stable softmax、gradient clipping 或正确混合精度的替代品。

## 9. 常见误区

1. **认为减最大值就是 z-loss**

   减最大值不改变目标；z-loss 增加了新的训练项。

2. **把 $Z$ 写成 softmax 概率之和**

   概率之和恒为 1。z-loss 使用的是：

   $$
   \log Z
   =
   \log\sum_j e^{z_j}
   $$

3. **先算 softmax 再取 log**

   应直接使用 `logsumexp`、`log_softmax` 或 `cross_entropy`。

4. **在低精度中计算 `logsumexp`**

   通常应转为 FP32。

5. **对 padding token 也计算 z-loss**

   z-loss 与 CE 应使用一致的有效 token mask。

## 相关笔记

- [[Lec 3 Architectures, Hyperparameters]]
