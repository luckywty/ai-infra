# CS336 Lecture 2：PyTorch 与 Resource Accounting

> [!info] 课件来源
> [[CS336/90-资料/课件/Trace - lecture_02.pdf.pdf]]

> [!abstract] 一句话主线
> 把 LLM 训练拆成一系列 Tensor 的**存储、搬运、计算与求导**，然后用 bytes、FLOPs、FLOP/s 和 arithmetic intensity 判断“能不能放下、要算多久、瓶颈在哪里”。

---

## 1. 🎯 本课核心主题（Core Theme）

本课建立一套面向 LLM 系统的 **resource accounting（资源记账）**方法：从 Tensor shape、dtype 和算子出发，估算参数、梯度、激活、优化器状态所需的显存，以及一次训练所需的 FLOPs。进一步借助 arithmetic intensity、roofline model 和 MFU，判断 GPU 工作负载究竟受算力还是显存带宽限制。

最终目标不是得到绝对精确的数字，而是形成可靠的数量级直觉：在固定计算量和显存下，快速判断模型能否训练、训练需要多久，以及应当使用低精度、梯度累积还是 activation checkpointing。

---

## 2. 🔑 核心知识要点（Key Concepts & Mechanics）

### 2.1 Resource Accounting：先做数量级估算

本课用两个问题说明“餐巾纸估算”的价值：

1. 70B 模型在 15T tokens 上、使用 1024 张 H100 训练，需要多久？
2. 8 张 80 GB H100 使用 AdamW，理论上最多能容纳多少参数？

#### 训练计算量近似

对矩阵乘法占主导的模型，训练总计算量可近似为：

$$
C_{\text{train}} \approx 6TP
$$

其中：

- $C_{\text{train}}$：总训练计算量，单位 FLOPs；
- $T$：训练数据点数量；语言模型中通常按训练 token 数近似；
- $P$：模型参数量；
- 系数 $6$：前向约 $2TP$，反向约 $4TP$。

若使用 $N$ 张 GPU，则训练时间近似为：

$$
t_{\text{train}}
\approx
\frac{6TP}
{N F_{\text{peak}}\cdot \mathrm{MFU}}
$$

其中：

- $N$：GPU 数量；
- $F_{\text{peak}}$：单卡在指定 dtype 下的 dense 峰值 FLOP/s；
- $\mathrm{MFU}$：Model FLOPs Utilization，即有效模型计算吞吐与理论峰值吞吐之比。

课件示例取：

$$
P=70\times 10^9,\qquad
T=15\times 10^{12}
$$

所以：

$$
C_{\text{train}}
=6\times 70\times 10^9\times 15\times 10^{12}
=6.3\times 10^{24}\ \text{FLOPs}
$$

H100 的稀疏 BF16 峰值为 1979 TFLOP/s；课件按 dense 性能减半：

$$
F_{\text{peak}}
=\frac{1979}{2}\times 10^{12}
=989.5\times 10^{12}\ \text{FLOP/s}
$$

取 $N=1024$、$\mathrm{MFU}=0.5$：

$$
t_{\text{train}}
\approx
\frac{6.3\times 10^{24}}
{1024\times 989.5\times 10^{12}\times 0.5}
\approx 1.24\times 10^7\ \text{s}
\approx 144\ \text{天}
$$

这是排期的数量级估算，不包含故障、checkpoint I/O、通信效率变化等现实因素。

---

##### 2.2 Tensor：所有资源的统一载体

训练中的数据、参数、激活、梯度和优化器状态最终都存放在 Tensor 中。一个 Tensor 的核心属性是：

- `shape`：每一维的大小；
- `rank`：维度数；
- [[tensor.dtype|dtype]]：每个元素的编码方式；
- [[tensor.device|device]]：Tensor 位于 CPU 还是某张 GPU。

[[Lec 3 Architectures, Hyperparameters|Transformer]] 中常见四维 Tensor：

$$
X\in\mathbb{R}^{B\times S\times H\times D_h}
$$

其中：

- $B$：batch size；
- $S$：sequence length；
- $H$：attention head 数；
- $D_h$：每个 head 的维度。

Tensor 的元素数和显存占用为：

$$
\operatorname{numel}(X)=\prod_{i=1}^{r}d_i
$$

$$
M_X
=\operatorname{numel}(X)\cdot b
=\left(\prod_{i=1}^{r}d_i\right)b
$$

其中：

- $r$：Tensor rank；
- $d_i$：第 $i$ 个维度的大小；
- $b$：每个元素的字节数；
- $M_X$：Tensor 占用的字节数。

对应的 PyTorch 检查方式：

```python
def tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()
```

例如 GPT-3 FFN 中一个形状为 $(4\times 12288,\ 12288)$ 的 FP32 权重矩阵约占 2304 MiB。

---

### 2.3 数值精度：动态范围、分辨率、显存与吞吐

| dtype    | 符号位 | 指数位 | 尾数位 |       每元素大小 | 核心特点                            |
| -------- | --: | --: | --: | ----------: | ------------------------------- |
| FP32     |   1 |   8 |  23 |     4 Bytes | 范围和精度较好，显存与带宽成本高                |
| FP16     |   1 |   5 |  10 |     2 Bytes | 分辨率高于 BF16，但动态范围较小              |
| BF16     |   1 |   8 |   7 |     2 Bytes | 动态范围接近 FP32，但尾数精度较低             |
| FP8 E4M3 |   1 |   4 |   3 |      1 Byte | 精度较高、范围较小，H100 上约为 $[-448,448]$ |
| FP8 E5M2 |   1 |   5 |   2 |      1 Byte | 范围更大、精度更低，约为 $[-57344,57344]$   |
| NVFP4    |   - |   - |   - | 名义 0.5 Byte | 依赖 block-wise scale 扩展可表达范围     |

FP16 的指数位较少，小数可能直接下溢：

```python
x = torch.tensor([1e-8], dtype=torch.float16)
assert x == 0
```

BF16 保留了 FP32 的 8 位指数，因此同样的数不会下溢：

```python
x = torch.tensor([1e-8], dtype=torch.bfloat16)
assert x != 0
```

#### [[混合精度|Mixed Precision Training]]

课件采用的简化记账口径是：

- 参数：BF16；
- 激活：BF16；
- 梯度：BF16；
- 优化器状态：FP32。

原因是矩阵乘法和激活通常能容忍较低尾数精度，而 Adam/AdaGrad 的滑动统计量会跨许多 step 累积，使用 FP32 更稳定 (**主要是精度更高**)。

```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    output = model(x)
    loss = loss_fn(output, target)
```

需要注意：

- autocast 按“算子”选择 dtype；它不会保证上下文里新建的所有 Tensor 都自动变成 BF16；
- `matmul` 等算子通常适合低精度，`exp` 等数值敏感算子可能保留高精度；
- 某些训练栈还会保存 FP32 master weights，此时每参数显存会高于本课的简化口径。

更低精度的本质折衷是：

$$
\text{更少显存与带宽、更高 Tensor Core 吞吐}
\quad\Longleftrightarrow\quad
\text{更大量化误差与稳定性风险}
$$

FP8/FP4 还需要 per-tensor、per-channel 或 per-block scale；scale 本身也会产生存储和计算开销。

---

### 2.4 Tensor 在 GPU 上的放置

PyTorch 默认在 CPU 上创建 Tensor：

```python
x = torch.zeros(32, 32)
assert x.device == torch.device("cpu")
```

使用 GPU 有两种基本方式：

```python
device = torch.device("cuda")

# 先创建，再搬运
x = x.to(device)

# 直接在目标设备上创建
x = torch.zeros(32, 32, device=device)
```

同一个算子的输入必须位于兼容的 device 上。反复在 CPU/GPU 间 `.to(...)` 会引入额外传输；设备搬运也应纳入资源记账。

---

### 2.5 Einops：用维度语义代替位置猜测

传统写法：

```python
z = x @ y.transpose(-2, -1)
```

`-2`、`-1` 缺少语义，Tensor rank 改变后容易操作错维度。Einops 用命名维度描述 contraction、reduction 和 reshape。

#### `einsum`：广义矩阵乘法

若：

$$
X\in\mathbb{R}^{S_1\times D},\qquad
Y\in\mathbb{R}^{D\times S_2}
$$

则：

$$
Z_{ij}=\sum_{k=1}^{D}X_{ik}Y_{kj}
$$

```python
z = einsum(
    x, y,
    "seq1 hidden, hidden seq2 -> seq1 seq2",
)
```

没有出现在输出侧的命名维度会被求和。批量 attention score 可写为：

```python
scores = einsum(
    q, k,
    "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2",
)
```

这里 `...` 表示任意数量的广播维度。

#### `reduce`：沿命名维度归约

```python
y = reduce(x, "... hidden -> ...", "sum")
```

对应：

$$
y_{\ldots}=\sum_h x_{\ldots,h}
$$

#### `rearrange`：拆分或合并维度

```python
x = rearrange(
    x,
    "... (heads head_dim) -> ... heads head_dim",
    heads=2,
)

x = rearrange(
    x,
    "... heads head_dim -> ... (heads head_dim)",
)
```

拆分必须满足：

$$
D_{\text{model}}=H D_h,
\qquad
D_{\text{model}}\bmod H=0
$$

Einops 的核心收益不是降低 FLOPs，而是显式化 shape contract，减少维度 bookkeeping 错误。

---

### 2.6 FLOPs、FLOP/s 与 MFU

- **FLOPs**：完成某次计算所需的浮点操作数，是“工作量”；
- **FLOP/s**：硬件每秒能完成多少浮点操作，是“速度”。

对矩阵乘法：

$$
X\in\mathbb{R}^{B\times D},\qquad
W\in\mathbb{R}^{D\times K},\qquad
Y=XW
$$

$Y$ 有 $BK$ 个元素，每个元素需要 $D$ 次乘法和 $D-1$ 次加法，因此：

$$
F_{\text{matmul}}
=BK(2D-1)
\approx 2BDK
$$

其中：

- $B$：输入数据点或行数；
- $D$：reduction dimension；
- $K$：输出维度；
- $F_{\text{matmul}}$：该矩阵乘法的 FLOPs。

课件中的简化 MFU 定义为：

$$
\mathrm{MFU}
=
\frac{F_{\text{actual}}}{F_{\text{peak}}}
=
\frac{\text{模型 FLOPs}/t}{F_{\text{peak}}}
$$

其中 $t$ 是实测运行时间。约 0.5 的 MFU 已经是不错的结果。

这里的 MFU 是理想化资源分析：真实模型还会受到通信、kernel launch、非矩阵算子、矩阵尺寸、pipeline bubble 和设备故障等因素影响。

---

### 2.7 Arithmetic Intensity 与 Roofline Model

一个 GPU 算子至少包含：

1. 从显存读取输入；
2. 在计算单元上执行操作；
3. 将输出写回显存。

设：

- $F$：算子需要的 FLOPs；
- $M$：算子需要从显存搬运的字节数；
- $P_{\text{peak}}$：硬件峰值 FLOP/s；
- $BW$：显存带宽，单位 Byte/s。

则：

$$
t_{\text{compute}}=\frac{F}{P_{\text{peak}}},
\qquad
t_{\text{memory}}=\frac{M}{BW}
$$

如果计算与内存搬运能完全重叠：

$$
t_{\text{total}}
\approx
\max\left(t_{\text{compute}},t_{\text{memory}}\right)
$$

#### 算术强度与硬件强度

工作负载的 arithmetic intensity：（**每个 Byte 需要 $I_{\text{arith}}$ 次计算**）

$$
I_{\text{arith}}=\frac{F}{M}
\quad [\text{FLOPs/Byte}]
$$

硬件的 accelerator intensity：

>[!info] $I_{\text{hw}}$ 表示 GPU 每获得 1 Byte/s 的显存带宽，配备了多少 FLOP/s 的计算能力。

$$
I_{\text{hw}}=\frac{P_{\text{peak}}}{BW}
\quad [\text{FLOPs/Byte}]
$$

判定规则：

$$
I_{\text{arith}}<I_{\text{hw}}
\Rightarrow \text{memory-bound}
$$

$$
I_{\text{arith}}>I_{\text{hw}}
\Rightarrow \text{compute-bound}
$$

课件取 H100 dense BF16：

$$
P_{\text{peak}}=989.5\times10^{12}\ \text{FLOP/s},
\qquad
BW=3.35\times10^{12}\ \text{Byte/s}
$$

所以：

$$
I_{\text{hw}}
=\frac{989.5}{3.35}
\approx 295\ \text{FLOPs/Byte}
$$

#### 典型算子的算术强度

以下均按 BF16（2 Bytes/element）以及“输入从 HBM 读取一次、输出写回一次”的理想模型估算。

| 算子 | FLOPs $F$ | 搬运字节 $M$ | $I_{\text{arith}}$ | H100 上的瓶颈 |
|---|---:|---:|---:|---|
| ReLU，$n$ 元素 | $\approx n$ | $4n$ | $\approx 1/4$ | memory-bound |
| GELU，$n$ 元素 | $\approx 20n$ | $4n$ | $\approx 5$ | memory-bound |
| dot product | $2n-1$ | $4n+2$ | $\approx 1/2$ | memory-bound |
| $n\times n$ matrix-vector | $n(2n-1)$ | $2n^2+4n$ | $\approx 1$ | memory-bound |
| $n\times n$ matrix-matrix | $n^2(2n-1)$ | $6n^2$ | $\approx n/3$ | 大 $n$ 时 compute-bound |

矩阵乘法的精确理想强度为：

$$
I_{\text{matmul}}
=\frac{n^2(2n-1)}{6n^2}
=\frac{2n-1}{6}
\approx \frac{n}{3}
$$

当 $n=1024$：

$$
I_{\text{matmul}}\approx 341>295
$$

因此进入 compute-bound 区域。临界矩阵规模粗略满足：

$$
\frac{n}{3}\approx 295
\Rightarrow
n\approx 885
$$

这解释了：

- 训练的大 batch 矩阵乘法数据复用高，更容易吃满 [[TF32(TensorFloat-32)|Tensor Core]]；
- 单 token、小 batch [[MHA, MQA and GQA|自回归 decode]] 更接近 matrix-vector product，每次生成 token 都需读取大量权重，因而通常受显存带宽限制；
- prefill 或大 batch 推理仍可能形成大型矩阵乘法，不能笼统地说“所有推理都 memory-bound”。

#### Roofline

Roofline model 给出的理想可达吞吐为：

$$
P_{\text{attainable}}
=
\min\left(P_{\text{peak}},\ BW\cdot I_{\text{arith}}\right)
$$

相应地，理想化 MFU 可写为：

$$
\mathrm{MFU}
\approx
\min\left(
1,\frac{I_{\text{arith}}}{I_{\text{hw}}}
\right)
$$

Roofline 的“拐点”就是 $I_{\text{hw}}$：左侧增加算术强度能提升吞吐，右侧继续增加计算复用也无法超过峰值算力。

---

### 2.8 Autograd 与训练 FLOPs：为什么是 $6TP$

#### 一个最小 autograd 例子

课件定义：

$$
\ell=\frac{1}{2}(x^\top w-5)^2
$$

其梯度为：

$$
\frac{\partial \ell}{\partial w}
=(x^\top w-5)x
$$

若 $x=[1,2,3]$、$w=[1,1,1]$，则 $x^\top w=6$，因此：

$$
\frac{\partial \ell}{\partial w}=[1,2,3]
$$

PyTorch 通过计算图和 `loss.backward()` 自动完成链式法则：

```python
x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

pred = x @ w
loss = 0.5 * (pred - 5).pow(2)
loss.backward()

assert torch.allclose(w.grad, torch.tensor([1.0, 2.0, 3.0]))
```

#### 线性层的前向与反向计算量

考虑：

$$
H_2=H_1W_2
$$

其中：

$$
H_1\in\mathbb{R}^{B\times D},
\qquad
W_2\in\mathbb{R}^{D\times D}
$$

前向传播：

$$
F_{\text{forward}}\approx 2BD^2
$$

反向传播需要计算两项。

输入激活梯度：

$$
\frac{\partial L}{\partial H_1}
=
\frac{\partial L}{\partial H_2}W_2^\top
$$

权重梯度：

$$
\frac{\partial L}{\partial W_2}
=
H_1^\top\frac{\partial L}{\partial H_2}
$$

两次矩阵乘法各约 $2BD^2$ FLOPs：

$$
F_{\text{backward}}\approx 4BD^2
$$

因此单层总计算量为：

$$
F_{\text{total}}
\approx
2BD^2+4BD^2
=6BD^2
$$

若该层参数量 $P_{\text{layer}}=D^2$：

$$
F_{\text{total}}\approx 6B P_{\text{layer}}
$$

推广到全部层和训练数据：

$$
C_{\text{train}}\approx 6TP
$$

该近似隐含的假设：

- 矩阵乘法占主要计算量；
- 每个参数对每个数据点大致参与一次乘加；
- 忽略 bias、ReLU、归一化等低阶项；
- 对短上下文 Transformer 较合理，长上下文 attention 的额外计算需要单独核算；
- Mixture-of-Experts 应使用每 token 的 active parameters，而不能直接把全部参数都代入。

---

### 2.9 [[PyTorch GPU 显存指标|训练显存]]：参数、梯度、优化器状态与激活

对 $L$ 层、隐藏维度为 $D$ 的无 bias 全连接网络：

$$
P=D^2L
$$

若 batch size 为 $B$，每层 BF16 激活形状为 $B\times D$，则玩具模型的激活显存近似为：

$$
M_{\text{activation}}\approx 2BDL\ \text{Bytes}
$$

训练总显存的基本分解是：

$$
M_{\text{total}}
=
M_{\text{param}}
+M_{\text{grad}}
+M_{\text{optimizer}}
+M_{\text{activation}}
+M_{\text{temporary}}
$$

在本课简化混合精度口径下：

| 项目 | 每参数字节数 | 总显存 |
|---|---:|---:|
| BF16 参数 | 2 | $2P$ |
| BF16 梯度 | 2 | $2P$ |
| AdaGrad FP32 二阶累积量 | 4 | $4P$ |
| Adam FP32 一阶矩和二阶矩 | 8 | $8P$ |

因此 Adam 的参数相关静态显存为：

$$
M_{\text{static, Adam}}
=2P+2P+8P
=12P\ \text{Bytes}
$$

8 张 80 GB H100 的理想参数上界：

$$
P_{\max}
\approx
\frac{8\times80\times10^9}{12}
\approx 53.3\times10^9
$$

即约 53B 参数。它只是上界，因为尚未计入：

- 激活和 attention 中间量；
- 临时 Tensor 与 CUDA workspace；
- 通信 buffer；
- allocator 碎片；
- checkpoint 读写 buffer；
- 可能额外存在的 FP32 master weights。

若额外保存每参数 4 Bytes 的 FP32 master weights，参数相关显存会变成 $16P$ Bytes，上界降为约 40B。

#### 优化器状态为何昂贵

AdaGrad：

$$
v_t=v_{t-1}+g_t^2
$$

$$
\theta_t
=
\theta_{t-1}
-\eta\frac{g_t}{\sqrt{v_t+\epsilon}}
$$

其中：

- $g_t$：第 $t$ 步梯度；
- $v_t$：逐元素累计平方梯度；
- $\eta$：学习率；
- $\epsilon$：防止除零的稳定项。

Adam 还需保存一阶矩 $m_t$ 和二阶矩 $v_t$，所以仅优化器状态就需要 $8P$ Bytes。

---

### 2.10 两种显存优化

#### Gradient Accumulation：沿 batch 维度切分

设：

- $B_{\text{eff}}$：一次参数更新对应的有效 batch size；
- $B_\mu$：micro-batch size；
- $K$：累计次数。

单个 worker 上：

$$
B_{\text{eff}}=K B_\mu
$$

若有 $N_{\text{data}}$ 个 data-parallel worker：

$$
B_{\text{eff}}=N_{\text{data}}K B_\mu
$$

激活峰值从与 $B_{\text{eff}}$ 成正比，降为与 $B_\mu$ 成正比：

$$
M_{\text{activation}}\propto B_\mu
$$

机制：

1. 对一个 micro-batch 做 forward/backward；
2. 不清空 `.grad`，让梯度继续累加；
3. 累积 $K$ 次后执行一次 `optimizer.step()`；
4. 再执行 `optimizer.zero_grad()`。

注意：课件中 `B=64`、`micro_batch_size=256` 的示意数字不满足实际切分定义；真实配置应满足 $B_\mu\le B_{\text{eff}}$ 且通常要求能整除或正确处理余数。

梯度累积：

- 降低峰值激活显存；
- 不降低固定有效 batch 对应的理论 FLOPs；
- 会产生更多小矩阵和 kernel launch，可能降低 MFU；
- 使用 mean loss 时通常要除以 $K$，否则梯度会放大 $K$ 倍。

#### Activation Checkpointing：沿网络深度切分

反向传播需要前向激活。Activation checkpointing（也称 gradient checkpointing 或 rematerialization）只保存部分边界激活，在 backward 时重新执行相应的 forward 段。

极端情况：

| 策略                |          激活显存 |      重计算 |
| ----------------- | ------------: | -------: |
| 保存每层激活            |        $O(L)$ |        无 |
| 不保存中间激活，每次从头重算    |        $O(1)$ | $O(L^2)$ |
| 每隔 $\sqrt{L}$ 层保存 | $O(\sqrt{L})$ |   $O(L)$ |

更一般地，若每 $k$ 层设置一个 checkpoint：

$$
M_{\text{peak}}
=
O\left(\frac{L}{k}+k\right)
$$

其中：

- $L/k$：需要长期保留的 checkpoint 数量；
- $k$：重算一个 segment 时的临时激活数量。

令两项平衡：

$$
\frac{L}{k}\approx k
\Rightarrow
k\approx\sqrt{L}
$$

于是：

$$
M_{\text{peak}}=O(\sqrt{L})
$$

并且每个 segment 在 backward 中只重算一次时，总额外重算仍为 $O(L)$。

---

## 3. ⚡ 重难点与工程折衷（Difficulties & Engineering Trade-offs）

### 3.1 “峰值算力高”不代表算子就快

GPU 规格表中的峰值通常要求：

- 特定 dtype；
- 足够大的矩阵；
- 合适的对齐和 tile；
- 高 occupancy；
- Tensor Core 被充分利用。

逐元素算子即使 FLOPs 很少，也要读写完整 Tensor，因此可能受带宽支配。优化时不能只看 FLOPs，必须同时看 **bytes moved**。

---

### 3.2 算术强度是“整个实现”的属性

课件的 roofline 推导假设输入各从 HBM 读取一次，真实结果还取决于：

- tile 和 cache reuse；
- 内存访问是否连续、是否 coalesced；
- [[torch.compile|kernel fusion]] 是否减少中间 Tensor 的回写；
- dtype 改变是否同时提高算力并减少搬运字节；
- batch 和 sequence 是否足够大。

例如孤立的 ReLU 与 GELU 都是 memory-bound，因此 GELU 多出的 FLOPs 未必按比例增加 [[Latency|延迟]]。若把 bias、GELU 等操作 fuse 到 matmul epilogue，甚至可以避免一次完整的 HBM round trip。

---

### 3.3 BF16、FP8、FP4：吞吐收益与数值风险

降低精度同时改变两条 roofline：

1. 每元素字节数下降，工作负载的 $I_{\text{arith}}=F/M$ 上升；
2. Tensor Core 的 $P_{\text{peak}}$ 上升，硬件的 $I_{\text{hw}}=P_{\text{peak}}/BW$ 也可能上升。

因此“精度减半就必然快一倍”并不成立。若算子已经 memory-bound，减少字节通常有效；若量化/反量化、scale 计算或数值修复成本较大，净收益会变小。

---

### 3.4 $6TP$ 是近似，不是物理定律

$6TP$ 的来源是每个权重在前向中约一次乘加，反向中再参与输入梯度和权重梯度两次矩阵乘法。它不自动覆盖：

- 长上下文 attention 的 $S^2$ 项；
- embedding、normalization、softmax 等非参数计算；
- MoE 中总参数与 active 参数的区别；
- activation recomputation；
- 稀疏计算；
- 数据并行、张量并行或流水线并行的通信。

正确用法是先以 $6TP$ 得到数量级，再对显著偏离假设的模块单独加账。

---

---

### 3.5 训练与 decode 的瓶颈不同

- 训练：大 batch 与 sequence 形成 matrix-matrix multiplication，数据复用高，通常趋向 compute-bound；
- 单 token decode：每步输入行数很小，更接近 matrix-vector multiplication，权重读取量大，通常趋向 memory-bound；
- prefill、大 batch serving：可能再次形成大矩阵，瓶颈不能只凭“这是推理”来判断。

工程分析必须基于实际 shape，而不是只看算子名称。

---

## 4. 💻 代码/伪代码关键点（Key Implementation Details）

---

### 4.1 非叶子 Tensor 默认不保留 `.grad`

```python
h1 = x @ w1
h1.retain_grad()  # 仅在需要检查中间梯度时使用
```

PyTorch 默认只把梯度累积到需要梯度的叶子 Tensor，如 `nn.Parameter`。对所有中间激活调用 `retain_grad()` 会增加显存，不应作为常规训练设置。

---

### 4.3 标准训练循环的顺序

```python
optimizer.zero_grad(set_to_none=True)

for x, y in loader:
    pred = model(x)                 # forward
    loss = loss_fn(pred, y)
    loss.backward()                # backward，梯度累加到 .grad
    optimizer.step()               # update
    optimizer.zero_grad(set_to_none=True)
```

`loss.backward()` 默认是累加而非覆盖梯度。`set_to_none=True` 避免把整块梯度 Tensor 写零，通常也便于区分“没有梯度”和“梯度恰好为零”。

---

### 4.4 正确 [[benchmark]] [[DAY02 GPU 推理延迟的正确测量|CUDA]]

[[torch.utils.benchmark.Timer]]

---

### 4.5 Gradient Accumulation 伪代码

```python
accum_steps = K
optimizer.zero_grad(set_to_none=True)

for step, (x, y) in enumerate(loader):
    pred = model(x)
    loss = loss_fn(pred, y) / accum_steps
    loss.backward()

    if (step + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

容易踩坑：

- mean loss 未除以 $K$，导致梯度放大；
- 在每个 micro-batch 后错误地 `zero_grad()`，使累积失效；
- 数据总数不能整除 $K$ 时忘记处理最后一组；
- micro-batch 太小导致矩阵规模不足，MFU 明显下降。

---

### 4.6 Activation Checkpointing

课件中的关键调用：

```python
for layer in self.layers:
    x = torch.utils.checkpoint.checkpoint(layer, x)
```

它不会压缩 Tensor，也不会把激活转移到 CPU；它改变的是 autograd 保存策略：不保存 checkpoint 区间内部的全部中间值，而在 backward 时重新运行该区间的 forward。

容易踩坑：

- checkpoint 区间中含随机算子时，要保证重算使用一致的 RNG 状态；
- 含状态副作用的 forward 可能在重算时被执行两次；
- checkpoint 粒度过细会增加调度开销，过粗则节省的显存有限；
- 某些 API 模式要求至少一个输入参与求导，应检查当前 PyTorch 版本的约束。

---

### 4.7 教学版优化器代码的 dtype 陷阱

课件的 AdaGrad 示例使用：

```python
g2 = state.get("g2", torch.zeros_like(grad))
```

若 `grad` 是 BF16，`zeros_like(grad)` 也会创建 BF16 状态，这与“优化器状态使用 FP32”的记账假设不一致。真实实现应显式控制状态 dtype，并避免直接使用 `.data` 修改参数；更安全的教学写法是在 [[torch.no_grad()]] 中更新。

```python
with torch.no_grad():
    g2 = state.get(
        "g2",
        torch.zeros_like(grad, dtype=torch.float32),
    )
    g2.add_(grad.float().square())
    p.addcdiv_(grad, (g2 + eps).sqrt(), value=-lr)
```

---

## 5. 🧪 课后自测三问（Self-Check Questions）

### 问题 1：两次餐巾纸估算

1. 一个 70B 模型在 15T tokens 上训练，使用 1024 张 H100。单卡 dense BF16 峰值为 989.5 TFLOP/s，MFU 为 50%。估算训练天数。
2. 8 张 80 GB H100 使用 BF16 参数、BF16 梯度和 FP32 Adam 一阶/二阶矩。忽略激活和临时显存，参数量上限是多少？

> [!answer]- 参考答案
> 训练 FLOPs：
>
> $$
> 6TP=6.3\times10^{24}
> $$
>
> 有效集群吞吐：
>
> $$
> 1024\times989.5\times10^{12}\times0.5
> =5.066\times10^{17}\ \text{FLOP/s}
> $$
>
> 因此训练约 $1.24\times10^7$ 秒，即 **144 天**。
>
> Adam 口径为每参数 $2+2+4+4=12$ Bytes：
>
> $$
> P_{\max}=\frac{8\times80\times10^9}{12}\approx 53.3\text{B}
> $$
>
> 这是未计激活、临时 buffer、碎片和 FP32 master weights 的上界。

### 问题 2：从线性层推导 $6BP$

设 $H_2=H_1W_2$，其中 $H_1\in\mathbb{R}^{B\times D}$、$W_2\in\mathbb{R}^{D\times D}$。分别写出前向、输入梯度和权重梯度的矩阵乘法与 FLOPs，并说明何时可以推广为 $6TP$。

> [!answer]- 参考答案
> 前向：
>
> $$
> H_2=H_1W_2,\qquad 2BD^2
> $$
>
> 输入梯度：
>
> $$
> \frac{\partial L}{\partial H_1}
> =\frac{\partial L}{\partial H_2}W_2^\top,
> \qquad 2BD^2
> $$
>
> 权重梯度：
>
> $$
> \frac{\partial L}{\partial W_2}
> =H_1^\top\frac{\partial L}{\partial H_2},
> \qquad 2BD^2
> $$
>
> 总计 $6BD^2=6BP_{\text{layer}}$。当矩阵乘法主导、上下文不太长、每 token 的 active parameters 可由 $P$ 表示时，可推广为 $6TP$。

### 问题 3：Roofline 与 checkpoint 粒度

1. H100 dense BF16 的 $I_{\text{hw}}\approx295$ FLOPs/Byte。对 BF16 的 $n\times n$ 矩阵乘法，临界 $n$ 大约是多少？为什么 matrix-vector product 不会随 $n$ 增大而进入同样的 compute-bound 区域？
2. 一个 $L=256$ 层网络每隔 $k$ 层设置 checkpoint。用 $O(L/k+k)$ 模型选择 $k$，并说明 gradient accumulation 与它减少的是不是同一类显存。

> [!answer]- 参考答案
> 矩阵乘法：
>
> $$
> I_{\text{matmul}}\approx \frac{n}{3}
> $$
>
> 令 $n/3=295$，得到 $n\approx885$，所以 $n=1024$ 的理想模型已进入 compute-bound。Matrix-vector product 的 FLOPs 和权重搬运量都按 $n^2$ 增长，强度始终约为 1 FLOP/Byte，因此仍是 memory-bound。
>
> 对 $L=256$：
>
> $$
> k\approx\sqrt{L}=16
> $$
>
> 此时峰值激活量级为 $L/k+k=16+16=32$ 个“层激活单位”，即 $O(\sqrt{L})$。Checkpointing 沿深度减少保存的中间激活；gradient accumulation 沿 batch 维度减小单次 micro-batch 激活，二者不等价但可以组合。

---
