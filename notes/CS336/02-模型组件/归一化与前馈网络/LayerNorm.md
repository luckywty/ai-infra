
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
