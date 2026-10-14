## 1. 什么是“正则化”？

正则化不是某一种具体算法，而是一类手段：

$$[  
\boxed{\text{限制模型过度拟合训练数据，使模型对未见数据也能工作}}  
]$$

训练模型时，实际目标不只是让训练损失最低，还要避免模型依赖过于偶然、脆弱的特征。

可以把它理解为：

```text
没有正则化：
模型可以自由记忆训练数据中的细节

加入正则化：
限制模型，使它更倾向于学习稳定、通用的规律
```

Dropout 和 Weight Decay 都属于正则化，但作用对象不同：

$$[  
\boxed{\text{Dropout 作用于中间激活值}}  
]$$

$$[  
\boxed{\text{Weight Decay 作用于模型参数}}  
]$$

---

---

# 2. Weight Decay

## 2.1 基本思想

Weight Decay 不随机删除激活，而是在每次参数更新时，让参数稍微向 0 收缩。

普通梯度下降：

# $$[  
\theta_{t+1}

\theta_t-\eta g_t  
]$$

加入 Weight Decay 后：

# $$[  
\boxed{  
\theta_{t+1}

(1-\eta\lambda)\theta_t-\eta g_t  
}  
]$$

其中：

- ($\theta_t$)：当前参数；
    
- ($g_t$)：损失函数梯度；
    
- ($\eta$)：学习率；
    
- ($\lambda$)：Weight Decay 系数。
    

可以拆成两个动作：

$$[  
\theta_t  
\rightarrow  
(1-\eta\lambda)\theta_t  
]$$

先把参数缩小一点，再根据梯度更新。

---

## 2.2 一个简单例子

假设：

$$[  
\theta=10,\qquad  
\eta=0.01,\qquad  
\lambda=0.1  
]$$

仅考虑 Weight Decay：

# $$[  
\theta'

(1-0.01\times0.1)\times10  
]$$

$$[  
=0.999\times10=9.99  
]$$

每一步收缩很小，但经过大量训练步后，会持续抑制参数无限增大。

---

## 2.3 为什么较小的权重有正则化作用？

参数非常大时，输入的微小变化可能被大幅放大。

例如：

$$[  
y=wx  
]$$

当：

$$[  
w=1000  
]$$

输入变化：

$$[  
\Delta x=0.01  
]$$

输出变化为：

$$[  
\Delta y=w\Delta x=10  
]$$

Weight Decay 倾向于**避免不必要的大参数**，使模型学习到相对平滑、分散的表示。

需要注意：

$$[  
\boxed{\text{Weight Decay 不是强制所有参数都接近 0}}  
]$$

只要某个大参数能够显著降低任务损失，训练梯度仍然可以抵消衰减项。它只是在“大参数”和“任务损失”之间增加权衡。

---


---

# 3. Dropout 和 Weight Decay 的核心区别

|对比项|Dropout|Weight Decay|
|---|---|---|
|作用对象|激活值|参数|
|主要操作|随机清零一部分激活|持续缩小参数|
|是否随机|是|否|
|是否改变前向传播|是|不直接改变|
|是否在推理时生效|否|不执行参数更新，但训练结果保留|
|主要超参数|(p)|(\lambda)|
|主要效果|降低特征共适应|抑制不必要的大权重|
|常见实现|`nn.Dropout(p)`|`AdamW(weight_decay=...)`|

一句话区分：

```text
Dropout：训练时随机关闭一部分计算通路
Weight Decay：训练时让参数持续向零轻微收缩
```

---

---

# 4. Weight Decay 通常不应用于所有参数

实际实现中，常把参数分成两组：

```python
decay_params = [
    # Linear 层中的大矩阵参数
]

no_decay_params = [
    # bias
    # LayerNorm/RMSNorm 的缩放参数
]
```

原因是 bias 和 Norm 参数通常维度较低，其数值大小承担特定功能，对它们施加 Weight Decay 往往没有明显收益。

示意代码：

```python
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=learning_rate,
)
```

具体是否排除某类参数，要以模型的训练配置为准。

---

# 8. 最终知识框架

```text
正则化
│
├── Dropout
│     ├── 作用于激活
│     ├── 随机清零
│     ├── 训练时开启
│     └── 推理时关闭
│
└── Weight Decay
      ├── 作用于参数
      ├── 参数持续收缩
      ├── 通常由 AdamW 实现
      └── 训练停止后不再更新
```

核心结论：

$$[  
\boxed{  
\text{Dropout 通过随机破坏中间表示来正则化}  
}  
]$$

$$[  
\boxed{  
\text{Weight Decay 通过抑制参数规模来正则化}  
}  
]$$

$$[  
\boxed{  
\text{现代 LLM 预训练通常更依赖 Weight Decay，而非 Dropout}  
}  
]$$
