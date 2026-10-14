![[Pasted image 20260905223014.png]]

根据 [CS336 Lecture 10](https://cs336.stanford.edu/lectures?trace=lecture_10&step=104) 代码中的介绍，**PagedAttention**（由 [vLLM](https://cs336.stanford.edu/lectures?trace=lecture_10&step=104) 提出）的核心原理与技术要点总结如下：

**背景痛点（传统预分配显存）**

- **内部碎片（Internal Fragmentation）**：请求到来时，系统通常按预设的最大序列长度预先分配连续的显存空间；如果实际生成 Token 数远少于最大长度，预留的显存就会被白白浪费。
- **外部碎片（External Fragmentation）**：不同请求分配的显存块之间容易产生无法利用的微小空隙。

---

**核心机制：分页式显存管理**

- **非连续物理块映射**：借鉴操作系统的虚拟内存分页机制，将序列的逻辑 KV Cache 切分成固定大小、在物理显存中非连续（Non-contiguous）的物理块（Blocks）。
- **动态按需分配**：模型在自回归生成每个 Token 时，按需动态申请新的物理块，不再预先占用大片连续显存，极大消除了显存碎片。

---

**多序列共享与写时复制（Copy-on-Write）**

- **跨序列共享前缀**：在以下场景中，多个请求的公共前缀可以在物理块级别实现直接共享，无需重复存储：
    - **System Prompt 共享**：不同请求共享相同的系统提示词。
    - **单 Prompt 多候选采样**：同一 Prompt 生成多个不同回答（如代码生成、Beam Search 等）。
- **写时复制（Copy-on-Write）**：当不同序列在共享前缀后开始分叉生成不同内容时，仅复制并开辟新的物理块存放差异部分，最大化降低显存开销。

---

**配套的工程与 Kernel 级优化**

- **算子融合**：设计专用 GPU Kernel，融合非连续物理块的读取与注意力计算（Fused Block Read & Attention），减少 Kernel Launch 的调度开销。
- **先进计算 Kernel 集成**：结合 FlashAttention 与 FlashDecoding 算子提升注意力并行度与吞吐。
- **CUDA Graphs**：利用 CUDA Graphs 固化计算图执行流，进一步消除逐步解码带来的 CPU 下发瓶颈。

---

![[Pasted image 20260905223825.png]]

根据网页中的内容，这部分讲解的是 **[PagedAttention](https://cs336.stanford.edu/lectures?trace=lecture_10&step=104)**（由 [vLLM](https://cs336.stanford.edu/lectures?trace=lecture_10&step=104) 提出）在跨序列共享 KV Cache（Sharing KV caches across sequences）**场景下的显存优化机制与**写时复制（Copy-on-Write）原理：

- **常见的多序列共享场景**
    - **共享系统提示词（Sharing the system prompt）**：如上方示意图所示，两个独立的请求（Sequence A 与 Sequence B）使用了完全相同的 Prompt 前缀（英文到法文的多样本翻译指令），仅在具体的任务输入（`"cheese"` vs `"I love you"`）上存在差异。
    - **单一 Prompt 多候选采样（Parallel Sampling）**：在代码生成（Program synthesis）或 Beam Search 等任务中，同一个 Prompt 需要并发生成多个不同的候选答案。
- **核心优化方案：块级前缀共享（Share prefixes at the block level）**
    - 多个请求相同的逻辑 Token 序列，在物理显存中直接指向同一块物理 KV Cache（Physical KV block），不需要重复存储和重复计算。
    - 物理块维护引用计数（Reference Count，例如图中显示 `Ref count: 2 -> 1`）。
- **写时复制机制（Copy-on-Write at the block level）**
    - 当两个序列在共享内容之后开始产生分叉、生成不同内容时（如下方图示中的 `Sample A1` 生成 `fathers`，而 `Sample A2` 生成 `mothers`），系统才会触发**写时复制（Copy-on-Write）**。
    - 系统开辟一个新的独立物理块（Block 3），将原有公共块（Block 1）的内容复制过去，并将当前序列写入各自专属的差异 Token，同时原公共块的引用计数减 1。
    - 这种机制最大化地复用了显存空间，消除了共享前缀带来的显存冗余。
