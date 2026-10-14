在 [CS336 Lecture 10](https://cs336.stanford.edu/lectures?trace=lecture_10&step=104) 代码中，带有门控机制的 MLP 层（Gated MLP，如 SwiGLU/GeGLU 结构）的前向计算与内存读写分为以下步骤：

- **1. 读取输入 $X$：**
    - 从 HBM 读取输入张量 $X$（维度为 $B \times T \times D$），传输量为 $2 B T D$ 字节（bf16）。
- **2. 读取三组投影权重：**
    - 从 HBM 读取三个权重矩阵：$W_{up}$（$D \times F$）、$W_{gate}$（$D \times F$）和 $W_{down}$（$F \times D$），传输量共为 $3 \times 2 D F = 6 D F$ 字节。
- **3. Up 投影计算与写回：**
    - 执行矩阵乘法 $U = X @ W_{up}$，将张量从 $D$ 维升至 $F$ 维，计算量为 $2 B T D F$ FLOPs。
    - 将中间结果 $U$（维度 $B \times T \times F$）写回 HBM，传输量为 $2 B T F$ 字节。
- **4. Gate 投影计算与写回：**
    - 执行矩阵乘法 $G = X @ W_{gate}$，计算量为 $2 B T D F$ FLOPs。
    - 将中间结果 $G$（维度 $B \times T \times F$）写回 HBM，传输量为 $2 B T F$ 字节。
- **5. 激活与逐元素乘积（Gating）：**
    - 计算激活值与门控的逐元素乘积：$\text{GeLU}(G) \cdot U$（维度仍为 $B \times T \times F$）。
- **6. Down 投影计算与写回：**
    - 执行矩阵乘法 $Y = (\text{GeLU}(G) \cdot U) @ W_{down}$，将张量从 $F$ 维降回模型隐藏层维度 $D$，计算量为 $2 B T D F$ FLOPs。
    - 将最终输出 $Y$（维度 $B \times T \times D$）写回 HBM，传输量为 $2 B T D$ 字节。
