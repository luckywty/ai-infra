# CS336 Lecture 10: Inference

[课程链接](https://cs336.stanford.edu/lectures?trace=lecture_10&step=104)

---

## 课程结构大纲

- **理解推理负载**：全景介绍、Transformer 架构复习、算术强度分析、吞吐量与延迟权衡。
- **有损优化（Lossy Shortcuts）**：减小 KV Cache（GQA、MLA、CLA、局部注意力等）、模型量化（QAT、PTQ、AWQ）、模型剪枝与蒸馏。
- **无损优化（Lossless Shortcuts）**：推测采样（Speculative Sampling）。
- **动态负载处理**：解决请求到达时间不一致、长度不同与前缀共享问题，引入连续批处理（Continuous Batching）和 PagedAttention。

---

## 推理全景与背景（landscape 部分）

### 推理的应用场景

- 实际部署场景：聊天机器人、代码补全、智能体（Agent）和批量数据处理。
- 模型评估与强化学习（RL，采样大量生成后打分）。

### 为什么效率至关重要

- 训练是一次性开销，推理是持续的重复开销。
- 对于 Agent 工作流，内部推理链会产生远多于最终人类可见的 Token 数量（Token 生成量直接对应算力消耗）。

### 主流服务商与开源框架

- **闭源模型提供商**：OpenAI、Anthropic、Google 等。
- **开源权重服务商**：Together、Fireworks、Baseten、DeepInfra、Groq、Cerebras 等。
- **开源推理引擎**：
  - vLLM（主打 PagedAttention）
  - SGLang（主打 RadixAttention，适配 Agent）
  - TensorRT-LLM（NVIDIA GPU 高度优化）
  - llama.cpp（C++ 实现，主打本地与 CPU 推理）

### 评估推理速度的核心指标

- **首字延迟（TTFT，Time-to-first-token）**：从发起请求到开始生成首个 Token 的等待时间（影响交互体验）。
- **单请求延迟（Latency，s/token）**：单条查询生成 Token 的速度（影响交互流畅度）。
- **吞吐量（Throughput，tokens/s）**：系统在多并发查询下单位时间内生成的总 Token 数（针对批处理）。

### 效率决定的本质差异（训练 vs 推理）

- **训练**：全局监督学习，所有 Token 均已知，可以在序列维度上高度并行化（矩阵乘法）。
- **推理**：依赖自回归顺序生成，无法在生成维度进行并行，因而很难充分打满硬件的计算单元。

---

## 算术强度分析

基于 FlashAttention 的注意力层在推理阶段的算术强度（Arithmetic Intensity = 计算量 / 访存字节数）推导总结如下：

### 符号与设定

- $B$：Batch 维度（并发请求数）
- $D$：模型隐藏层维度（Hidden dimension）
- $S$：上下文及历史生成的 Token 数（Key/Value 缓存长度）
- $T$：当前步骤需要生成/计算 Logits 的 Token 数
- 数据类型以 **bf16** 为例，每个元素占 **2 字节**

---

### Attention 阶段

#### 计算量与访存量分解

**计算量（FLOPs）**：

- 计算注意力得分 $A = Q K^T$：矩阵大小为 $(B, T, D) \times (B, S, D)^T$，FLOPs 为 $2 B S T D$。
- 计算输出 $Y = \text{softmax}(A) V$：矩阵大小为 $(B, T, S) \times (B, S, D)$，FLOPs 为 $2 B S T D$。
- **总 FLOPs** $= 4 B S T D$

**内存传输量（Bytes Transferred，HBM 读写）**：

- **读**：读取 $Q$ 矩阵（$2 B T D$ 字节）、读取 $K$ 缓存（$2 B S D$ 字节）、读取 $V$ 缓存（$2 B S D$ 字节）。
- **写**：写回输出 $Y$（$2 B T D$ 字节）。
- **总字节数** $= 2 B T D + 2 B S D + 2 B S D + 2 B T D = 4 B S D + 4 B T D$

#### 算术强度推导与简化

$$\text{Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}} = \frac{4 B S T D}{4 B S D + 4 B T D} = \frac{S \cdot T}{S + T}$$

#### 两阶段特征分析

**Prefill 阶段（$T = S$）**：

$$\text{Intensity}_{\text{prefill}} = \frac{S \cdot S}{S + S} = \frac{S}{2}$$

算术强度随序列长度 $S$ 线性增长，长上下文下极易达到硬件算力瓶颈，属于 **Compute-bound（算力受限）**。

**Generation / Decode 阶段（$T = 1$）**：

$$\text{Intensity}_{\text{generate}} = \frac{S}{S + 1} < 1$$

每读取 1 个字节做不到 1 次浮点运算，严重受限于内存带宽，属于极端 **Memory-bound（访存受限）**。

> [!important] Batching 无法缓解 Decode 阶段的注意力访存瓶颈
> 公式中的 $B$ 在分子与分母中直接约掉，因此单纯增大 Batch Size 无法提升注意力矩阵乘法的算术强度（不同于 MLP 层所有请求共享固定权重，注意力层中每个序列都有独立的 KV Cache 需单独读写）。

---

### MLP 阶段

根据 [CS336 Lecture 10](https://cs336.stanford.edu/lectures?trace=lecture_10&step=104) 代码中的推导，针对包含门控机制的 [[MLP 层]]（仅考虑矩阵乘法），算术强度推导总结如下：

#### 符号与设定

- $B$：Batch 维度（并发请求数）
- $T$：当前步骤处理/生成的 Token 数
- $D$：模型隐藏层维度（Model dimension）
- $F$：MLP 升维维度（Feed-forward dimension，通常 $F \approx 4D$）
- 数据类型以 **bf16** 为例，每个元素占用 **2 字节**

#### 计算量与访存量分解

**计算量（FLOPs）**：

MLP 层包含 3 个核心矩阵乘法（Up-projection、Gate-projection、Down-projection）：

1. $U = X W_{up}$：$(B \times T \times D) \times (D \times F)$，计算量为 $2 B T D F$
2. $G = X W_{gate}$：$(B \times T \times D) \times (D \times F)$，计算量为 $2 B T D F$
3. $Y = (\text{GeLU}(G) \cdot U) W_{down}$：$(B \times T \times F) \times (F \times D)$，计算量为 $2 B T D F$

- **总 FLOPs** $= 6 B T D F$

**内存传输量（Bytes Transferred，HBM 读写）**：

1. 读取输入 $X$：$2 B T D$ 字节
2. 读取权重 $W_{up}, W_{gate}, W_{down}$：$3 \times 2 D F = 6 D F$ 字节
3. 写入中间结果 $U$：$2 B T F$ 字节
4. 写入中间结果 $G$：$2 B T F$ 字节
5. 写入最终输出 $Y$：$2 B T D$ 字节

- **总字节数** $= 4 B T D + 4 B T F + 6 D F$

#### 算术强度推导与简化

$$\text{Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}} = \frac{6 B T D F}{4 B T D + 4 B T F + 6 D F}$$

在推理场景中，模型权重尺寸通常远大于激活值（即假设 $B \cdot T \ll D, F$）：

- 分母中的激活项 $4 B T D + 4 B T F$ 相比权重读取项 $6 D F$ 可以忽略不计。
- 简化计算极限：

$$\text{Intensity} \approx \frac{6 B T D F}{6 D F} = B \cdot T$$

#### 两阶段特征分析

**Prefill 阶段（$T = S$，输入 Prompt 长度）**：

- 算术强度为 $B \cdot S$。
- 序列较长或采用大 Batch 时，$B \cdot S$ 较大，极易超过硬件算力带宽比（例如 H100 约为 295），属于 **Compute-bound（算力受限）**。

**Generation / Decode 阶段（$T = 1$，单步自回归生成）**：

- 算术强度简化为 $B$。
- 算术强度完全取决于并发请求数 $B$：
  - 当 $B = 1$ 时，算术强度仅为 1，严重处于 **Memory-bound（访存受限）**，大部分时间花在从 HBM 读取模型权重上。
  - 增大并发 Batch Size $B$ 可以有效分摊权重的读取开销，从而提升 MLP 层的算术强度与硬件利用率。

---

## 注意力与 KV Cache 优化机制

### 一、减少 KV Cache 体积（结构级优化）

- **Grouped-Query Attention (GQA)**
  - **机制**：在 $N$ 个 Query 头的基础上，仅保留 $K$ 个 Key/Value 头（$K < N$），多个 Query 头共享同一组 KV 头。
  - **收益**：将 KV Cache 的显存占用直接缩减为原来的 $K / N$，降低内存读取量，提升吞吐并允许跑更大的 Batch Size。
- **Multi-Head Latent Attention (MLA)**
  - **机制**：不直接缓存完整的高维 KV 向量（$N \times H$），而是将隐层表征投影压缩为一个低维向量 $c$ 存入 Cache，计算时再投影解压；同时为适配 RoPE 额外保留少量位置编码维度。
  - **收益**：如 DeepSeek-V2 将 16384 维压缩至 512 维（外加 64 维 RoPE），大幅削减 Cache 体积，且在精度表现上更优。
- **Cross-Layer Attention (CLA)**
  - **机制**：将 GQA 在注意力头维度的共享思路拓展到网络层级，实现跨层（Across layers）共享 KV Cache。
  - **收益**：在模型精度与 KV Cache 大小的 Pareto 前沿上取得更优平衡。
- **Local / Sliding Window Attention**
  - **机制**：限制注意力仅关注局部的滑动窗口上下文，或与全局注意力交替堆叠（混合层架构）。
  - **收益**：将 KV Cache 的大小从与全序列长度线性相关解耦为固定常数窗口大小。
- **DeepSeek v4 稀疏注意力组合**
  - 针对超长上下文（1M Context），综合采用压缩稀疏注意力（CSA，多 Token 压缩为一）、局部 Top-K 挑选（DSA）与重度压缩注意力（HCA）等多级压缩策略。

---

### 二、显存系统管理与动态调度优化

- **[[PagedAttention]]（以 vLLM 为代表）**
  - **解决痛点**：传统静态预分配显存容易产生大量内部碎片（未生成满最大长度）与外部碎片。
  - **机制**：借鉴操作系统的虚拟内存分页机制，将每个序列的 KV Cache 切分为不连续的物理块（Blocks）按需动态分配。
  - **共享优化**：在处理共享前缀（如系统提示词 System Prompt）或单一 Prompt 生成多个候选时，支持在块级别实现前缀共享与写时复制（Copy-on-Write）。
- **Continuous Batching（连续批处理 / 迭代级调度）**
  - **机制**：摒弃传统请求级的同步 Batching，改为在 Step（Iteration）级别调度；一个请求生成完毕立即退出，新请求到达立即动态加入 Batch，消除了不同序列长度的 Padding 气泡开销。
  - **Selective Batching**：非注意力层（如 MLP）将各序列有效 Token 拼接为大矩阵并行计算，注意力层则保持序列隔离计算。
- **底层 Kernel 与运行时加速**
  - 利用 FlashAttention / FlashDecoding 算子优化访存与并行度。
  - 融合 Block 读取与注意力运算算子，并结合 CUDA Graphs 消除小算子反复下发（Kernel Launch）的 CPU 开销。

---

## 模型量化（Quantization）技术

根据 [CS336 Lecture 10](https://cs336.stanford.edu/lectures?trace=lecture_10&step=104) 的内容，模型量化（Quantization）技术的核心原理与主流方法总结如下：

### 一、核心目的与基本机制

- **核心目标**：推理任务通常处于访存受限（Memory-bound）状态，通过降低数值精度（减少每个数值所占字节数），能够直接减少内存搬运总量与显存占用，从而显著提升生成吞吐量并降低延迟。
- **数学映射原理**：
  - **量化（Quantize）**：$x_{\text{quant}} = \text{round}(x / \text{scale}) + \text{zero\_point}$
  - **反量化（Dequantize）**：$x_{\text{approx}} = (x_{\text{quant}} - \text{zero\_point}) \times \text{scale}$

### 二、常见数据精度与适用场景

- **FP32（4 字节）**：主要用于模型训练阶段的参数更新与优化器状态。
- **BF16（2 字节）**：当前大语言模型推理的标准默认精度。
- **FP8（1 字节）**：如 H100 支持的 `e4m3` 格式（取值范围 $[-240, 240]$），可用于激进的训练或高吞吐推理。
- **INT8（1 字节）**：取值范围 $[-128, 127]$，计算开销较 FP8 更低，但精度损失稍大，仅用于推理。
- **INT4（0.5 字节）**：取值范围 $[-8, 7]$，极致压缩显存与访存量，但对精度影响更明显。

### 三、主要量化方法

- **量化感知训练（Quantization-aware training, QAT）**
  - **机制**：在训练的前向传播过程中模拟量化与反量化误差（Quantize-and-dequantize），让模型权重在训练期主动适应精度截断。
  - **优缺点**：精度保持最好，但需要昂贵的大规模重训成本。
- **训练后量化（Post-training quantization, PTQ）**
  - **机制**：直接在已训练好的模型上操作，在小样本校准集（Sample data）上运行以确定各层或各 Tensor 的 `scale` 与 `zero_point`，成本极低。
  - **GPTQ**：利用海森矩阵（Hessian）二阶曲率信息更新未量化的权重，以动态补偿因量化产生的误差。
- **[激活感知权重量化](../02-模型组件/量化与压缩/激活感知权重量化-AWQ.md)（Activation-aware quantization, AWQ）**
  - **观察发现**：激活值（Activation）中存在少数非常显著的大通道（Outliers），与这些通道直接相乘的权重对模型精度至关重要。
  - **机制**：不搞一刀切，根据激活通道的显著性，挑选出最重要的极少数权重（约 0.1% ~ 1%）保留高精度，其余部分执行低比特量化。
  - **收益**：从 FP16 压缩至 INT3 可减少约 4 倍内存占用，带来约 3.2 倍的推理加速，且能较好地保住模型精度。

### 归纳对比

- **KV Cache 维度压缩**：**GQA** 压缩注意力头维度，**MLA** 采用低秩潜空间投影压缩通道维度，**CLA** 跨深度层共享维度。
- **超长上下文稀疏/时序压缩**：**CSA / DSA / HCA** 专为超长序列（如 DeepSeek-V4 1M 上下文）设计，分别从分组下采样、动态 Top-$k$ 稀疏和深度有损压缩三个维度削减历史 Token 带来的显存膨胀。
- **数值精度压缩**：**QAT** 依赖训练期模拟，**PTQ** 侧重训练后快速校准，而 **AWQ** 是当前兼顾硬件吞吐和精度保留的高效 PTQ 算法代表。

---

## 投机采样（Speculative Sampling / Speculative Decoding）

在 [CS336 Lecture 10](https://cs336.stanford.edu/lectures?trace=lecture_10&step=104) 的分类体系中，它属于 " 使用捷径并进行复核的无损加速策略（Lossless shortcuts）"：

- **本质**：一种针对自回归生成阶段（Memory-bound 访存受限）的加速采样算法，其数学本质是**修改版的拒绝采样（Modified Rejection Sampling）**。
- **核心机制**：
  1. 利用一个小参数量、高吞吐的草稿模型（Draft Model, $p$）自回归地快速 " 猜 " 出连续 $K$ 个候选 Token。
  2. 将这 $K$ 个 Token 喂给大参数量的目标模型（Target Model, $q$）进行一次并行的批量验证（利用 Prefill 阶段算力受限、并行度高的特性）。
  3. 依据接收准则逐个决定是否采纳；若拒绝则从残差分布中重采样修正并提前终止该轮循环。

- **关键特性（无损保证）**：通过精心设计的接受/拒绝修正概率，数学上严格保证最终生成的输出分布与直接从大模型 $q$ 中采样**完全一致（Exact sample）**。

![[Pasted image 20260905222718.png]]
