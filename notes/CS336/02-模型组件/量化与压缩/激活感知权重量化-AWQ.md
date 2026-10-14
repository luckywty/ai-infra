![[Pasted image 20260905222425.png]]

根据 [CS336 Lecture 10](https://cs336.stanford.edu/lectures?trace=lecture_10&step=104) 代码与论文 [[Lin+ 2023]](https://cs336.stanford.edu/lectures?trace=lecture_10&step=104) 的内容，激活感知权重量化（Activation-aware Weight Quantization, AWQ）的核心总结如下：

- **核心观察（Observation）：**
    - 在 Transformer 的推理过程中，激活值（Activation）并不是均匀分布的，存在极少数数值非常显著的**大激活通道（Outliers / Large Channels）**。
    - 矩阵乘法中与这些大通道相乘的权重（Weights），对最终的模型输出和精度起到主导作用。
- **核心思路（Idea & Mechanics）：**
    - **非均匀保护**：摒弃传统将所有权重一视同仁直接量化的做法，而是为重要通道分配更高的精度。
    - **通道筛选与保护**：根据输入的激活幅度，识别并筛选出极少数关键权重（约 **0.1% ~ 1%**）保留高精度，其余绝大部分权重进行低比特量化。
- **性能与压缩收益（Benefits）：**
    - **显存缩减**：将权重从 FP16 压缩至 INT3 可带来 **4 倍**的显存占用缩减。
    - **推理加速**：直接带来 **3.2 倍**的推理端到端加速。
    - **精度表现**：在大幅压缩访存带宽的同时，较好地保护了模型精度，避免了传统低比特量化导致的性能崩塌。
