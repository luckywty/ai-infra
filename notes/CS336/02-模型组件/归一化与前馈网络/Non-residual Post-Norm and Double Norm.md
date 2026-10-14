

> [!abstract] 核心定义
> Non-residual Post-Norm 把归一化放在 attention/FFN 分支输出之后、残差加法之前；Double Norm 则在分支前后各放一个 Norm。两者都试图进一步控制分支尺度，同时保留一条不经过 Norm 的 residual identity path。

## 1. 先区分三种 Norm 放置方式

用 $F$ 表示 attention 或 FFN 子层。

### [[Post-Norm LayerNorm]]

$$
\boxed{
y=\operatorname{Norm}\!\left(x+F(x)\right)
}
$$

Norm 包住了 residual addition，因此主 residual stream 也经过 Norm。原始 Transformer 和 BERT 使用过这种结构。

### Pre-Norm

$$
\boxed{
y=x+F\!\left(\operatorname{Norm}(x)\right)
}
$$

Norm 只作用于子层输入，$x\rightarrow y$ 仍有直接恒等路径。这是现代 LLM 最常见的基础结构。

### Non-residual Post-Norm

$$
\boxed{
y=x+\operatorname{Norm}_{\mathrm{post}}\!\left(F(x)\right)
}
$$

Norm 只作用于分支输出，不包住 $x+F(x)$，因此不会归一化 residual stream。

### Double Norm / Sandwich Norm

$$
\boxed{
y
=
x
+
\operatorname{Norm}_{\mathrm{post}}
\left(
F\!\left(\operatorname{Norm}_{\mathrm{pre}}(x)\right)
\right)
}
$$

分支输入和分支输出各经过一次 Norm，residual identity path 仍保持不变。

## 2. 最关键的判断：Norm 是否位于 residual stream 上

不能只按图中“Norm 在前还是在后”判断结构。要问：

> `Norm` 是否包住了 $x+F(x)$？

- 若是，则 residual stream 被归一化，是传统 Post-Norm；
- 若不是，而只归一化 $F(x)$，则是 non-residual post-norm；
- 若 $F$ 的前后都有 Norm，则是 Double Norm。

对比：

$$
\underbrace{
\operatorname{Norm}(x+F(x))
}_{\text{传统 Post-Norm}}
\neq
\underbrace{
x+\operatorname{Norm}(F(x))
}_{\text{Non-residual Post-Norm}}
$$

二者在优化行为上并不等价。

## 3. 为什么 residual identity path 很重要？

Pre-Norm 的 Jacobian：

$$
\frac{\partial y}{\partial x}
=
I
+
J_FJ_{\mathrm{Norm}}
$$

Non-residual Post-Norm：

$$
\frac{\partial y}{\partial x}
=
I
+
J_{\mathrm{Norm}}J_F
$$

Double Norm：

$$
\frac{\partial y}{\partial x}
=
I
+
J_{\mathrm{Norm,post}}
J_F
J_{\mathrm{Norm,pre}}
$$

三者都显式保留 $I$，梯度可以沿 residual identity path 传播。

传统 Post-Norm：

$$
\frac{\partial y}{\partial x}
=
J_{\mathrm{Norm}}
\left(I+J_F\right)
$$

identity 项也被左乘 $J_{\mathrm{Norm}}$。当许多层堆叠时，梯度传播更依赖一连串 Norm Jacobian，容易出现梯度衰减或尖峰，因此通常需要更谨慎的 warmup 和学习率。

## 4. Non-residual Post-Norm 的作用

设分支输出为：

$$
u=F(x)
$$

Non-residual Post-Norm 先控制 $u$ 的尺度：

$$
\widetilde{u}
=
\operatorname{Norm}_{\mathrm{post}}(u)
$$

再注入 residual stream：

$$
y=x+\widetilde{u}
$$

其目标是避免某个 attention/FFN 分支突然产生过大的 update，同时不改写已存在的 residual state $x$。

可以把它理解为：

> 主干信息原样通过；新写入 residual stream 的增量先做尺度整形。

## 5. Double Norm 的作用

Double Norm 同时控制：

1. 输入给子层 $F$ 的尺度；
2. 子层输出写回 residual stream 前的尺度。

$$
u
=
F(\operatorname{Norm}_{\mathrm{pre}}(x))
$$

$$
y
=
x+\operatorname{Norm}_{\mathrm{post}}(u)
$$

Pre-Norm 让 $F$ 看到尺度稳定的输入；Post-Norm 再控制 $F$ 产生的 update。课件提到 Grok、Gemma 2 等模型使用过类似 double norm，OLMo 2 使用 non-residual post-norm。

## 6. 完整 Transformer Block 伪代码

### 普通 Pre-Norm

```python
def block(x):
    x = x + attention(attn_pre_norm(x))
    x = x + ffn(ffn_pre_norm(x))
    return x
```

### Non-residual Post-Norm

```python
def block(x):
    attn_update = attention(x)
    x = x + attn_post_norm(attn_update)

    ffn_update = ffn(x)
    x = x + ffn_post_norm(ffn_update)
    return x
```

### Double Norm

```python
def block(x):
    attn_update = attention(attn_pre_norm(x))
    x = x + attn_post_norm(attn_update)

    ffn_update = ffn(ffn_pre_norm(x))
    x = x + ffn_post_norm(ffn_update)
    return x
```

Attention 和 FFN 的 Norm 通常是四组独立参数，不能无意间共享。

## 7. 工程折衷

### 潜在收益

- 保留干净的 residual identity path；
- 控制写入 residual stream 的更新尺度；
- 可能减少 loss spike 和梯度尖峰；
- 对更深网络、更大学习率或低精度训练更稳。

### 潜在成本

- 每个 block 增加一次或两次 Norm；
- Norm 是 bandwidth-bound 操作，额外访存可能影响吞吐；
- 参数量增加虽小，但 kernel launch 和中间 Tensor 不能忽略；
- 过度限制分支尺度可能削弱表达能力；
- 不同模型对 Norm 顺序、类型和初始化敏感，不能任意替换。

如果 Norm 没有与 attention/FFN projection 融合，Double Norm 的 wall-clock 成本可能比 FLOPs 占比显示的更高。

## 8. 常见误区

1. **把 `Norm(x + F(x))` 当作 non-residual post-norm**

   这是传统 Post-Norm，因为 Norm 作用在 residual sum 上。

2. **把 Double Norm 理解为连续调用两次 Norm**

   Double Norm 的重点是分支前后各一次：

   $$
   x+\operatorname{Norm}_{post}
   (F(\operatorname{Norm}_{pre}(x)))
   $$

3. **Attention 和 FFN 共用同一套 Norm 参数**

   除非架构明确要求，否则它们应是独立模块。

4. **加载预训练权重时擅自改变位置**

   Norm 的放置会改变整个计算图，不是可以在推理时随意调整的等价变换。

5. **只比较 FLOPs**

   Norm 的主要成本可能是内存读写和 kernel launch，应测量真实 tokens/s。



## 相关笔记

- [[Lec 3 Architectures, Hyperparameters]]
