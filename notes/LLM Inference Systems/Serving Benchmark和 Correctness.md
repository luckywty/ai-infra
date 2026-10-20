在 LLM Inference Systems（大模型推理系统）中，Serving Benchmark（在线推理服务性能基准测试）和 Correctness（正确性验证）是评价推理引擎的两个核心维度。

对于你目前研究的 mini-SGLang 项目，尤其是 Prefill/Decode 调度、Prefix Cache、Paged KV Cache 等优化，这两个维度非常重要。

可以先记住：

- **Serving Benchmark：** 优化后的推理系统是否真的更快、更高效，能够服务更多并发请求？
- **Correctness：** 优化后的推理系统是否仍然正确执行请求，生成符合预期的结果，并保持缓存与调度状态的一致性？

> 二者是互补关系：性能提升不能以错误结果为代价，而功能正确也不意味着系统具有良好的服务性能。

---

## 一、Serving Benchmark 是什么？

Serving Benchmark 是对一个正在运行的 LLM 推理服务施加请求负载，测量其在不同负载条件下的延迟、吞吐量、稳定性和服务质量。

它与单纯测量模型 Forward 时间的 Microbenchmark 有本质区别。

### 在线推理请求的完整生命周期

```text
Client 发送请求
①
Tokenization
②
Queue / Scheduler
③
KV Cache 管理
Prefill 阶段
Prompt 计算 · Prefix Cache 复用 · 首 Token 生成
Decode 阶段
Continuous Batching · Attention · Token Sampling
流式输出 Token → 请求完成
```

Serving Benchmark 在客户端测量端到端服务表现；服务端埋点可进一步拆解各阶段耗时。

### 1. 关键性能指标

| 指标 | 含义 | 主要反映什么 |
| --- | --- | --- |
| **TTFT** | Time To First Token，首 Token 延迟 | 用户等待首个输出的时间 |
| **TPOT** | Time Per Output Token，平均后续 Token 耗时 | Decode 平均速度 |
| **ITL** | Inter-Token Latency，连续输出 Token 之间的间隔 | 流式输出是否平稳 |
| **E2E Latency** | 单请求端到端延迟 | 完整请求完成时间 |
| **Output Throughput** | 每秒生成的输出 Token 数量 | 整体 Token 处理能力 |
| **Request Throughput** | 每秒完成的请求数量 | 整体请求处理能力 |
| **Goodput** | 每秒完成且满足预设 SLO 的请求数量 | 有效服务能力 |
| **Timeout / Error Rate** | 超时或失败请求占比 | 稳定性和可靠性 |

这些也是 SGLang、vLLM 等 Serving Benchmark 工具关注的主要指标。

![](https://www.google.com/s2/favicons?domain=https://docs.sglang.ai&sz=32)

SGLang

+2

其中几个指标需要特别理解。

#### TTFT：首 Token 延迟

$$
TTFT=t_{\mathrm{first\ token}}-t_{\mathrm{request\ sent}}
$$

它不只是 Prefill 计算时间，还可能包含网络传输、排队、Tokenizer、缓存处理等开销。

假设一次请求：

- 排队耗时 200 ms
- Prefill 和首 Token 计算耗时 80 ms
- 其他服务与网络耗时 20 ms

那么客户端观察到的 TTFT 约为 300 ms。

因此，优化 Prefill 调度策略通常会影响 TTFT。

#### TPOT：平均每个后续 Token 的生成耗时

对生成了 $N>1$ 个 Token 的请求：

$$
TPOT=\frac{E2E-TTFT}{N-1}
$$

例如 E2E = 1100 ms，TTFT = 300 ms，一共生成 41 个 Token：

$$
TPOT=\frac{1100-300}{40}=20\ \mathrm{ms/token}
$$

对应后续 Token 的平均生成速率约为 50 token/s。

#### ITL：逐 Token 间隔

与 TPOT 不同，ITL 更关心每两个连续输出 Token 之间的实际时间间隔。即使平均 TPOT 相同，个别 Token 出现严重卡顿，也可能导致较高的 ITL P95/P99。

在常规逐 Token 流式输出中，TPOT 和 ITL 密切相关；但在 Speculative Decoding 或一次传回多个 Token 的情况下，两者不能直接等同。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

### 2. 为什么必须看 P95/P99？

假设优化前后 TTFT 的分布为：

**示例：平均 TTFT 相同，但长尾表现不同**

虽然平均 TTFT 都是 200 ms，但优化后少部分请求可能出现严重的排队延迟。

这就是为什么 Serving Benchmark 不能只关注平均值。

对于你现在研究的 LPM、Soft Aging、防饥饿保护机制，需要重点观察：

- Hot Requests 的 TTFT P50/P95/P99
- Cold Requests 的 TTFT P50/P95/P99
- Hot 与 Cold 的排队延迟
- Decode TPOT/ITL 的尾部延迟
- Goodput 以及超时率

例如 LPM 可以优先处理缓存命中较多的请求，但如果没有公平性保护，Cold Requests 可能长期排队。此时总体吞吐量可能改善，部分用户体验却会恶化。

### 3. Goodput 为什么重要？

Throughput 只关注处理数量，Goodput 则在此基础上加入服务质量约束（SLO）。

例如定义：

$$
SLO: \begin{cases} TTFT\leq500\ \mathrm{ms}\\ TPOT\leq50\ \mathrm{ms/token} \end{cases}
$$

假设一次 Benchmark 运行 100 秒，完成 1000 个请求，但只有 800 个请求同时满足上述约束：

$$
Throughput=10\ \mathrm{req/s}
$$

$$
Goodput=8\ \mathrm{req/s}
$$

> Goodput 的核心是：不是完成的请求越多越好，而是满足服务质量要求的请求越多越好。

![](https://www.google.com/s2/favicons?domain=https://vllm-project.github.io&sz=32)

GuideLLM Docs

---

## 二、Correctness（正确性验证）是什么？

Correctness 是验证推理系统在优化前后是否保持预期行为。

对于 mini-SGLang，正确性不只是模型生成的文字看起来合理，还包括推理结果、KV Cache、请求调度、并发执行等多个层面。

### 1. 四类核心正确性

| 正确性类别 | 需要验证的内容 | 典型错误 |
| --- | --- | --- |
| **Numerical Correctness** | Logits、Attention 等数值计算是否正确 | 算子计算结果偏差过大 |
| **Functional Correctness** | Token 生成结果及请求行为是否符合预期 | 输出错误 Token、提前终止 |
| **Cache Correctness** | KV Cache 的读写、复用、释放是否正确 | 错误复用 KV、Page 被提前释放 |
| **Concurrency Correctness** | 多请求并发执行是否正确 | 请求间 KV 污染、输出串线、死锁 |

以你目前学习的 Prefix Cache 为例：

假设存在两个请求：

```text
Request A: [1, 2, 3, 4, 5]
Request B: [1, 2, 3, 6, 7]
```

二者存在公共前缀 `[1, 2, 3]`。

启用 Radix Cache 后，Request B 可以复用前三个 Token 的历史 KV。

但必须保证：

1. 被复用的 KV 与重新计算的 KV 在数值上满足预设容差。
2. Request B 的后续 Token 只能使用它自身合法的上下文。
3. Cache Page 的引用计数与释放行为正确。
4. 并发请求不会错误地覆盖正在使用的 KV。

> 即使系统的 TTFT 降低了 50%，只要出现跨请求 KV 污染，这项优化就是不正确的。

### 2. 如何验证输出结果？

最常见的方法是建立一个 Baseline，然后比较优化前后的输出。

```text
Baseline
未优化版本
Input A
Output A
Optimized
优化后版本
Input A
Output B
比较 Token IDs、Logits、状态与错误信息
检查优化前后行为是否满足对应的正确性标准
```

需要根据优化类型选择不同的比较标准。

#### 场景 A：调度算法优化

例如 FCFS → LPM + Soft Aging。

如果模型、计算路径和数值精度不变，理想情况下，同一个请求在 Greedy Decoding 下应该产生相同的 Token IDs。

但实际系统中，不同 Batch 形状可能触发不同的 Kernel，浮点计算次序的变化也可能改变极其接近的 Logits 排序。因此，不宜只依赖全序列逐 Token 完全一致。

建议同时检查数值容差、Token 一致率，以及是否存在明显的计算或状态错误。

#### 场景 B：KV Cache 量化

例如 BF16 KV Cache → FP8 KV Cache。

此时量化本身会引入误差，因此通常不能要求 Logits 完全相等。

可以采用：

$$
\text{Relative Error} =\frac{\|Y_{\mathrm{optimized}}-Y_{\mathrm{reference}}\|_2} {\|Y_{\mathrm{reference}}\|_2+\epsilon}
$$

再结合 Logits 误差、Top-1 Token 一致率、Perplexity 和下游任务质量判断量化是否可接受。

> 需要注意，Perplexity 只是质量指标之一，不能单独证明 KV Cache 实现正确。

---

## 三、Serving Benchmark 与 Correctness 有什么区别？

| 对比维度 | Serving Benchmark | Correctness |
| --- | --- | --- |
| **核心问题** | 系统有多快？ | 系统运行得对不对？ |
| **测试方式** | 压力测试、负载回放 | 单元测试、差分测试、并发测试 |
| **主要指标** | TTFT、TPOT、ITL、Throughput、Goodput | Token 一致率、数值误差、Invariant、错误数量 |
| **典型工具** | SGLang Bench Serving、vLLM Bench Serve | pytest、Reference Implementation、Sanitizer |
| **主要关注点** | 性能、服务质量、稳定性 | 功能、数值、状态一致性 |
| **通过标准** | 达到预先定义的性能目标 | 满足预先定义的正确性约束 |

它们也有交集，例如 Timeout Rate 既能反映高负载下的服务能力，也能暴露死锁、请求丢失等正确性问题，但不能仅凭超时率判断根因。

---

## 四、针对 mini-SGLang，应该怎样开展测试？

建议将测试划分为三个阶段。

### 阶段一：Correctness Gate

在跑性能测试之前，先验证优化是否破坏关键行为。

对于调度优化，测试请求完整性、合法的 Prefix Cache 复用、多请求隔离、公平性保护和请求最终完成等属性。对有限资源和持续输入的场景，需要特别检查饥饿、超时及资源泄漏。

### 阶段二：Serving Benchmark

固定模型、硬件、输入输出长度、请求到达序列、Cache 预热状态，分别测试 Baseline 和优化版本。

记录 TTFT、TPOT、Hot/Cold 排队时间、Token Throughput、Goodput 和 Timeout Rate，并在不同并发或到达速率下进行比较。

### 阶段三：Ablation Study（消融实验）

逐步启用不同优化，判断每项改动带来的独立收益与交互影响。

例如比较 FCFS、LPM、LPM + Soft Aging、LPM + Soft Aging + 防饥饿轮数保护，检查收益是否来自目标机制。

### 推荐的测试场景

| 场景 | 测试目的 |
| --- | --- |
| **单请求，无 Cache Hit** | 验证基本推理路径 |
| **单请求，高 Cache Hit** | 验证 Prefix Cache 复用 |
| **Hot/Cold 混合并发** | 验证缓存感知调度性能与公平性 |
| **高请求到达速率** | 验证队列积压与长尾延迟 |
| **Prefill/Decode 混合负载** | 验证 Decode 受到的干扰 |
| **请求取消、超时、资源紧张** | 验证异常处理与 KV 资源管理 |

这里还有两点重要的实验设计原则。

**第一，固定并发数不等于固定请求到达速率。**Closed-loop（固定并发）和 Open-loop（外部到达速率驱动）代表不同负载模型。调度公平性和高负载排队问题尤其值得采用 Open-loop 测试。

**第二，Cache 状态必须可控。**例如测试 Hot Requests 与 Cold Requests 时，明确区分 Cache Warmup、测量阶段和 Cache Reset。否则不同实验之间缓存命中率不一致，会影响调度策略的公平比较。vLLM 官方 Benchmark 文档也特别提醒需要控制重复实验中的缓存复用情况。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

---

## 五、推荐学习的 Benchmark 工具

对于你目前的项目，可以优先研究以下两个工具的测试逻辑。

- [SGLang Bench Serving](https://docs.sglang.ai/developer_guide/bench_serving?utm_source=chatgpt.com)：支持模拟在线请求、控制请求到达速率与并发，并统计 TTFT、ITL、吞吐量等指标。
- [vLLM Bench Serve](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md?utm_source=chatgpt.com)：重点研究其请求生成、客户端计时、TPOT/ITL 统计和 Benchmark 参数设计。

> mini-SGLang 不一定能直接兼容所有选项，因此应先核实 API 与响应流格式，再复用相关工具。

---

## 六、对你当前项目的建议

你已经在研究 LPM、Soft Aging、轮数保护以及 Prefill/Decode 调度。下一步应建立一个长期可复用的测试框架，把 Correctness 与 Serving Benchmark 放在同一条实验流水线中：

$$
\boxed{ \text{Code Change} \rightarrow \text{Correctness Gate} \rightarrow \text{Serving Benchmark} \rightarrow \text{Ablation} \rightarrow \text{Performance Report} }
$$

在简历项目中，这样做的价值明显高于只展示一个吞吐量提升百分比。

例如，你最终要能够回答：

> “在保证请求正确完成、KV Cache 一致性及 Cold Requests 公平性的前提下，我设计的调度算法在不同负载下，使 TTFT P95、Goodput 和 Decode TPOT 分别发生了什么变化？这些变化有没有重复实验的证据？”

> 总结：Correctness 决定优化能不能使用，Serving Benchmark 决定优化值不值得使用。 对 LLM Inference Systems 岗位而言，能够设计公平、可复现、可解释的 Benchmark，并且证明优化没有破坏正确性，是很重要的工程能力。
