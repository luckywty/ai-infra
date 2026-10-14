# RMSNorm

> [!abstract] 核心定义
> RMSNorm（Root Mean Square Layer Normalization）只根据向量的**均方根**调整尺度，不减去均值，通常只有可学习 scale $\gamma$、没有 bias $\beta$。它保留了 LayerNorm 的尺度控制能力，同时减少 reduction、参数和数据移动，因此成为现代 LLM 的常见默认选择。

## 1. 数学定义

对某个 token 的隐藏向量：

$$
x=(x_1,\ldots,x_d)\in\mathbb{R}^{d}
$$

先计算均方根：

$$
\operatorname{RMS}(x)
=
\sqrt{
\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon
}
$$

再进行归一化与逐维缩放：

$$
\boxed{
\operatorname{RMSNorm}(x)_i
=
\gamma_i
\frac{x_i}{\operatorname{RMS}(x)}
}
$$

其中：

- $d=d_{\mathrm{model}}$：hidden dimension；
- $\epsilon$：避免除零并提高数值稳定性；
- $\gamma\in\mathbb{R}^{d}$：可学习的逐维 scale；
- 归一化对每个 token 独立进行，reduction 发生在最后一个 hidden dimension。

## 2. 与 [[LayerNorm]] 的区别

LayerNorm：

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i
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

| 对比项 | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 | 是 | 否 |
| 控制二阶尺度 | 是 | 是 |
| 可学习 scale $\gamma$ | 通常有 | 有 |
| 可学习 bias $\beta$ | 通常有 | 通常没有 |
| 可学习参数量 | $2d$ | $d$ |
| reduction | 均值与方差 | 均方根 |

LayerNorm 同时具有 **re-centering** 和 **re-scaling**；RMSNorm 只保留 re-scaling。大规模语言模型的经验表明，很多情况下控制激活尺度已经足够，强制每个 token 的 hidden mean 为 0 并非必要。

## 3. 为什么有助于训练？

若忽略 $\epsilon$，对正数 $a$：

$$
\operatorname{RMSNorm}(ax)
\approx
\operatorname{RMSNorm}(x)
$$

因此上游激活的整体尺度变化不会直接放大下游输入。它能：

- 约束不同层激活的数量级；
- 避免残差分支输出无控制地增大；
- 降低优化器必须同时处理“方向”和“绝对尺度”的难度；
- 与 Pre-Norm residual block 配合，改善深层网络的训练稳定性。

RMSNorm 不会减去 hidden mean，因此 residual stream 中的均值信息可以继续传播；这既是它与 LayerNorm 的表达差异，也是它能减少计算的原因。

## 4. 为什么可能更快？

RMSNorm 相比 LayerNorm：

- 少一次均值 reduction；
- 少一次减均值操作；
- 通常没有 $\beta$；
- **中间状态更少，更容易 kernel fusion**。

$$\boxed{\text{RMSNorm 减少的 FLOPs 很少，但它能简化归约、减少参数读取，并利于减少中间张量搬运}}$$

但关键不是“省了多少总 FLOPs”。Normalization 属于 arithmetic intensity 较低的逐元素/reduction 算子，实际延迟经常受显存带宽和 kernel launch 支配：

$$
\boxed{
\text{较少的数据移动}
\text{ 往往比 }
\text{较少的 FLOPs}
\text{ 更重要}
}
$$

若 RMSNorm 被融合进前后的 projection kernel，wall-clock 收益通常更明显；若实现产生额外 Tensor 和多次 HBM round trip，理论上的简化可能无法转化为加速。

## 5. PyTorch 实现

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        # 低精度训练时用 FP32 完成平方与 reduction。
        inv_rms = torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True)
            + self.eps
        )

        y = x * inv_rms.to(dtype=x.dtype)
        return y * self.weight.to(dtype=y.dtype)
```

## 6. 常见误区

1. **沿错误维度归一化**

   对 `[B, S, D]` Tensor，应沿 `D` 归一化，而不是 batch 或 sequence。

2. **错误地减去均值**

   一旦执行 `x - x.mean(...)`，实现就不再是 RMSNorm。

3. **在 BF16/FP16 中累计平方和**

   reduction 容易损失精度或溢出，通常应临时转为 FP32。

4. **把 $\epsilon$ 放到开方外**

   标准形式是：

   $$
   \sqrt{\operatorname{mean}(x^2)+\epsilon}
   $$

   而不是 $\sqrt{\operatorname{mean}(x^2)}+\epsilon$。

5. **认为 FLOPs 更少就必然更快**

   最终需要以真实 shape、dtype、kernel fusion 和 wall-clock benchmark 为准。

## 相关笔记

- [[Lec 3 Architectures, Hyperparameters]]
