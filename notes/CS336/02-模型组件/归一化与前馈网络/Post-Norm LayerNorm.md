**Post-Norm LayerNorm** 指的是：先执行 Attention/FFN，并完成残差相加，最后再做 LayerNorm。

$$[  
\boxed{y=\operatorname{LayerNorm}\bigl(x+F(x)\bigr)}  
]$$

其中 (F(x)) 可以是 Attention，也可以是 FFN。

```text
x ───────────────┐
│                │
└→ Attention/FFN ─→ Add → LayerNorm → y
```

一个完整的 Post-Norm Transformer Block 通常写成：

$$[  
h=\operatorname{LayerNorm}\left(x+\operatorname{Attention}(x)\right)  
]$$

$$[  
y=\operatorname{LayerNorm}\left(h+\operatorname{FFN}(h)\right)  
]$$

### 为什么叫 Post-Norm？

“Post” 表示 Norm 位于子层和残差相加之后：

```text
子层计算 → Residual Add → LayerNorm
```

与 Pre-Norm 对比：

$$[  
\text{Post-Norm:}\quad y=\operatorname{LN}(x+F(x))  
]$$

$$[  
\text{Pre-Norm:}\quad y=x+F(\operatorname{LN}(x))  
]$$

### 主要问题

Post-Norm 中，残差主路径每经过一层都必须通过 LayerNorm：

$$[  
x_l\rightarrow x_l+F(x_l)\rightarrow \operatorname{LN}  
]$$

因此深层网络反向传播时，梯度要连续经过多个 LayerNorm，训练更容易出现：

- 梯度衰减或波动；
    
- gradient spike；
    
- 对 learning-rate warmup 更敏感；
    
- 大学习率下稳定性较差。
    
