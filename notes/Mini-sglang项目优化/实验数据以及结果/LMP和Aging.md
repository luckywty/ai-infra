这轮指的是 **2026 年 10 月 17 日的 FCFS / 纯 LPM / LPM+Aging 三组消融**。核心结论是：**纯 LPM 明显降低热点请求的排队和首 token 延迟，但没有建立整体吞吐收益，同时观测到生成阶段尾延迟变差；Aging N64 没有带来额外的 cold 等待收益。**

原始报告见 [FINDINGS.md](/home/wangwei/ai-infra/Promotion/benchmark-results/lpm-comparison-20261017-121500/FINDINGS.md)。

**实验如何设计**

三组使用同一份优化后的 runtime-S、相同 Cache 实现、相同阶段策略和资源，只改变 Prefill 新请求的排序方式。因此这是排序策略的消融，**FCFS 基线是 S1000，并非 U8192。**

|配置编号|策略|新请求排序|
|---|---|---|
|S1000|FCFS|按到达顺序|
|S1L00|纯 LPM|缓存匹配前缀越长，优先级越高；平局按到达顺序|
|S1100|LPM+Aging N64|按 `缓存匹配长度 + 64 × 有效等待轮数` 排序|

三组均关闭硬保护和旧 raw-step 等待回退，LPM 队列阈值为 128。已准入请求的 chunk 续算始终优先。

共同环境如下：

|项目|设置|
|---|---|
|GPU / 模型|RTX 5060 Ti 8GB / Qwen3-0.6B|
|精度 / Attention|BF16 / FI-fa2|
|KV 容量 / page size|16384 tokens / 1|
|Context 上限|2304|
|正常 Prefill 预算|**256 tokens/批**|
|最大运行请求数|**16**|
|CUDA Graph|batch size 1、2、4、8、16|
|阶段策略|**Prefill-first**|
|动态预算|关闭|
|Server / Client CPU|0–7 / 8、9|
|正式采集|light Collector|

这里没有启用 F8196K4 的 Decode-first-bounded 阶段策略。

**负载是专门构造的冷热前缀竞争场景**

每个 run 使用 W6 的 1000 个请求：

|请求类别|数量|输入 / 输出|缓存条件|
|---|---|---|---|
|hot|800|576 / 32 tokens|复用预热的 512-token 前缀|
|cold|200|576 / 32 tokens|新前缀，不刻意重复使用|

到达方式为 **32 请求一批的突发到达**，名义平均速率为 **48.812 req/s**。每批先放 cold，再放 hot，以制造 FCFS 顺序与缓存亲和排序之间的竞争。

因此，一个 hot 请求通常只需计算约 64 个新输入 tokens，而 cold 请求需要计算接近完整的 576 tokens。LPM 有机会优先服务计算量较小的 hot 请求。

实验使用 seed **31723–31732**，每组 10 个 seed，共 30 个正式对照项。全部入选，保留了 33 次尝试；额外尝试属于质量重试，不作为独立样本。每组最终均完成 **10000/10000** 个请求。

**主要实验结果**

除“成功数”和“汇总 SLO 达标率”外，下表数值均为各 run 先计算指标、再跨 seed 取中位数。

|指标|FCFS|纯 LPM|LPM+Aging N64|
|---|---|---|---|
|成功请求|10000/10000|10000/10000|10000/10000|
|hot 排队 P95|363.901 ms|**223.714 ms**|231.510 ms|
|cold 排队 P95|355.202 ms|355.938 ms|372.214 ms|
|hot TTFT P95|430.888 ms|**294.628 ms**|302.675 ms|
|cold TTFT P95|449.491 ms|454.380 ms|465.565 ms|
|输出吞吐|1554.813 token/s|1555.554 token/s|1555.397 token/s|
|Goodput|48.320 req/s|48.545 req/s|48.489 req/s|
|host TPOT P95|11.603 ms|12.003 ms|12.111 ms|
|host ITL P99|**85.476 ms**|122.502 ms|121.981 ms|
|汇总 SLO 达标率|99.03%|**99.85%**|99.64%|

本轮 SLO 为 **TTFT≤500 ms、E2E≤2550 ms**。Goodput 分母使用完整的“到达开始至最后请求结束”墙钟时间，并非最近实验的固定 80 秒窗口。

**纯 LPM 的收益与代价**

逐 seed 配对结果比直接比较表中中位数更适合判断效果：

|纯 LPM 相对 FCFS|配对中位变化|95% seed-bootstrap 区间|
|---|---|---|
|hot 排队 P95|**−39.33%**|[−41.80%, −34.06%]|
|hot TTFT P95|**−31.90%**|[−34.15%, −29.43%]|
|cold 排队 P95|−0.74%|[−3.41%, +1.69%]|
|输出吞吐|+0.037%|[−0.057%, +0.114%]|
|Goodput|+0.454%|[−0.032%, +1.545%]|
|host TPOT P95|**+3.45%**|[+2.21%, +7.42%]|
|host ITL P99|**+48.91%**|[+37.86%, +56.69%]|

所以这轮最明确的正向结果是 **hot 等待和 TTFT 下降**。cold 等待、吞吐和 Goodput 的配对区间跨零，未建立稳定改善。

同时，生成阶段存在代价：**首 token 更早，不代表后续每个 token 的间隔也更短。** ITL P99 和 TPOT 的恶化需要保留，不能把这轮写成“全面提速”。这些属于共同 light Collector ON 下的观测，完整的算法 × Collector 交互尚未验证。[完整配对数据](/home/wangwei/ai-infra/Promotion/benchmark-results/lpm-comparison-20261017-121500/analysis/paired-comparisons.json)

**为什么等待改善明显，吞吐却基本不变**

三组缓存 token 命中率都约 **71.25%**，每 run 的实际 Prefill 工作量都约 **16.63 万 tokens**，输出也固定为 32000 tokens。

LPM 主要改变了请求的服务顺序，没有明显减少总计算量。三组都在约 20.57 秒内完成，未 padding 的 Decode batch 均值约 15.7，已经接近最大运行数 16。

排序还有额外 CPU 工作：调度函数平均耗时的跨 seed 中位数从 FCFS 的 **87.94 µs** 增至 LPM 的 **93.60 µs**。这是调度函数耗时，不是完整请求耗时。

**Aging 为什么没有改善 cold 等待**

Aging 的公式是：

```
score = cached_len + 64 × waiting_passes
```

`waiting_passes` 只在一轮至少首次准入了其他新请求、而该请求仍未准入时增加；纯 Decode、全拒绝和 chunk-only 轮都不增加。它不是墙钟时间。

在独立诊断 seed 31723 中，cold 相对 hot 最多只有 5 轮年龄优势，贡献约 320 分，仍不足以抵消约 512-token 的缓存匹配差。Aging 改变了部分同类请求的顺序，却没有产生 cold 越过 hot 的跨类排序。

正式配对结果中，Aging 相比纯 LPM：

- cold 排队 P95 **增加 4.50%**；
- hot 排队 P95 **增加 4.30%**；
- 吞吐基本不变。

详细解释见 [Aging 机制复核](/home/wangwei/ai-infra/Promotion/benchmark-results/lpm-comparison-20261017-121500/analysis/aging-mechanism-20261017/READOUT.md)。

**后续硬保护消融补充了什么**

随后使用新的 10 个 seed，在相同 W6 和资源配置下比较了纯 LPM、Aging N64、纯 LPM+H4、纯 LPM+H6：

- **H4：** cold 排队 P95 约下降 **3.78%**，但 hot 排队 P95 增加 **46.86%**。
- **H6：** 没有建立稳定的 cold 排队改善。
- 没有建立整体吞吐或 Goodput 收益；该轮尾部校准未通过，相关结论也需保留限制。

这说明硬保护能重新分配等待机会，但代价需要同时评价。[硬保护消融分析](/home/wangwei/ai-infra/Promotion/benchmark-results/round-guard-comparison-20261017-144900/analysis/interpretation-20261017/READOUT.md)

之前这轮 W6 的候选队列最大仅 32，没有触发超过 128 的回退；而最近的 W4-X、25 req/s 实验约 89.5% 的候选调用处于超限区间。这是两轮 LPM 表现差异的重要背景，加上前缀长度、输出长度、运行数和阶段策略也不同，不能直接外推收益幅度。
