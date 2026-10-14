## 直觉

**Sinusoidal Position Embedding（正弦位置编码）**通过给每个 token 加上一组由不同频率的 ($\sin/\cos$) 组成的位置向量，让 Transformer 知道 token 位于序列中的第几个位置。

课件将它归为原始 Transformer 使用的位置编码：

$$[  
\operatorname{Embed}(x,p)=v_x+PE_p  
]$$

其中 (v_x) 是 token embedding，(PE_p) 是位置 (p) 的固定位置向量。

## 1. 具体公式

设：

- $(p)$：token 的位置，($0,1,2,\ldots$)
    
- ($d=d_{\text{model}}$)：隐藏维度
    
- (i)：维度对的编号
    

定义：

## $$[  
\boxed{  
PE(p,2i)

\sin\left(\frac{p}{10000^{2i/d}}\right)  
}  
]$$

## $$[  
\boxed{  
PE(p,2i+1)

\cos\left(\frac{p}{10000^{2i/d}}\right)  
}  
]$$



## 2. 如何加入 token embedding？

原始 token embedding：

$$[  
x_p\in\mathbb{R}^{d_{\text{model}}}  
]$$

位置向量：

$$[  
PE_p\in\mathbb{R}^{d_{\text{model}}}  
]$$

直接逐元素相加：

$$[  
\boxed{  
h_p^{(0)}=x_p+PE_p  
}  
]$$

形状不变：

$$[  
[B,T,d_{\text{model}}]  
+  
[1,T,d_{\text{model}}]  
\rightarrow  
[B,T,d_{\text{model}}]  
]$$

数据流：

```text
token ID
   ↓
Token Embedding x_p ─┐
                     Add → Transformer
位置 p → sin/cos PE ─┘
```

## 3. 为什么需要多种频率？

不同维度对使用不同的 (\theta_i)：

```text
靠前维度：变化快，高频
靠后维度：变化慢，低频
```

直观类比是钟表：

- 秒针：适合区分相邻位置；
    
- 分针：表示中等距离；
    
- 时针：表示更长距离。
    

因此一个位置会同时由短周期和长周期信号描述，类似用多个不同精度的“时钟”共同标记位置。

## 4. 为什么同时使用 sin 和 cos？

因为它们可以利用三角函数和角公式表达位置偏移。

例如：

### $$[  
\sin((p+k)\theta)

\sin(p\theta)\cos(k\theta)  
+  
\cos(p\theta)\sin(k\theta)  
]$$

### $$\sin((p+k)\theta)
=
\sin(p\theta)\cos(k\theta)
+
\cos(p\theta)\sin(k\theta)$$

写成矩阵：

### $$[  
\begin{bmatrix}  
\sin((p+k)\theta)\  
\cos((p+k)\theta)  
\end{bmatrix}

\begin{bmatrix}  
\cos(k\theta)&\sin(k\theta)\  
-\sin(k\theta)&\cos(k\theta)  
\end{bmatrix}  
\begin{bmatrix}  
\sin(p\theta)\  
\cos(p\theta)  
\end{bmatrix}  
]$$

这表示：从位置 (p) 移动到 (p+k)，可以通过一个只依赖相对距离 (k) 的线性变换描述。

这也是它比单纯使用位置编号 (p) 更有结构的原因。

## 5. 一个简化例子

假设：

$$[  
d_{\text{model}}=4  
]$$

则位置 (p) 的编码为：

$$[  
PE_p=  
\left[  
\sin(p),  
\cos(p),  
\sin\left(\frac{p}{100}\right),  
\cos\left(\frac{p}{100}\right)  
\right]  
]$$

位置 (0)：

[  
PE_0=[0,1,0,1]  
]

位置 (1)：

$$[  
PE_1=  
[\sin1,\cos1,\sin0.01,\cos0.01]  
]$$

位置 (2)：

$$[  
PE_2=  
[\sin2,\cos2,\sin0.02,\cos0.02]  
]$$

每个位置都有不同的向量。

## 6. 优点和局限

优点：

- 不需要学习参数；
    
- 任意位置都可以直接计算；
    
- 不需要存储巨大的位置 embedding 表；
    
- 位置偏移具有规则的三角函数结构。
    

局限：

- 位置信息直接加到 token 表示上，内容与位置混合；
    
- Attention 分数不保证只依赖相对距离；
    
- 长上下文外推能力有限；
    
- 现代 LLM 更多使用 RoPE。
    

## 7. 和 RoPE 的核心区别

### Sinusoidal Position Embedding

$$[  
h_p=x_p+PE_p  
]$$

位置编码加在 Transformer 输入上。

### RoPE

$$[  
q_p'=R_pq_p,\qquad k_p'=R_pk_p  
]$$

位置通过旋转注入每层 Attention 的 (Q,K)。

因此：

```json
Sinusoidal：把位置向量加到 token embedding
RoPE：按位置旋转每层的 Query 和 Key
```

课件也明确对比：原始 Transformer 使用正弦/余弦位置编码，而课程实现的现代架构采用 RoPE。

一句话记忆：

### $$[  
\boxed{  
\text{Sinusoidal PE}

\text{用多频率 }\sin/\cos\text{ 构造固定位置向量，并与 token embedding 相加}  
}  
]$$
