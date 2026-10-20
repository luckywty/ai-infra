本次针对 **LPM + Pass-based Aging**，修改了 **8 个实现文件**，新增测试、Benchmark 和报告，未删除文件。

|文件|修改内容|
|---|---|
|`scheduler/utils.py`|给 `PendingReq` 增加 `waiting_passes`；扩展策略和参数校验|
|`scheduler/prefill.py`|增加 `lpm-aging`；按“真实匹配长度 + Aging 补偿”排序；有效等待轮次只计一次；复用本轮探测计划；无调度机会时跳过排序探测|
|`scheduler/cache.py`|`try_match_and_lock()` 支持传入探测计划，准入前重新验证缓存位置和保护量|
|`kvcache/base.py`|新增 `refresh_probe()` 接口|
|`kvcache/radix_cache.py`|增加结构版本；插入、分裂、淘汰时更新版本；结构未变时复用匹配并重算保护量，结构变化时重新匹配|
|`scheduler/config.py`|增加 `lpm_aging_tokens_per_pass=0`；支持三种策略；保留默认 FCFS|
|`scheduler/scheduler.py`|将 Aging 配置传给 `PrefillManager`|
|`server/args.py`|新增 `--lpm-aging-tokens-per-pass`；扩展策略选项和非负参数校验|

新增内容：

- `tests/core/test_prefill_policy.py`：排序、Aging、降级、请求生命周期及缓存一致性测试。
- `tests/core/test_prefill_benchmark.py`：Benchmark 确定性和资源守恒测试。
- `benchmarks/bench_prefill_policy.py`：可复现 CPU 调度 Benchmark。
- `docs/lpm_aging.md`：调用链、实现说明、复现命令及结果分析。
- `artifacts/lpm_aging/`：原始 JSON、结果表、JUnit XML、本次修改 diff 和验证摘要。

行为调整：

- 支持 `fcfs`、`lpm`、`lpm-aging`，默认仍为 `fcfs`。
- Aging 默认关闭，只有有效等待轮次才增加计数。
- 原有 `fairness_max_wait_steps` 默认从 **64 改为 0**，仍可显式启用。
- 保留队列超过 128 时的 FCFS 降级和 chunked 优先续跑。
- 虚拟分数不参与真实 KV 长度和资源分配。

验证结果：**新增 66 个执行测试用例，连同原有 40 项，共 106 项通过**。模型计算、Attention、Sampling、CUDA Graph 和 Decode 逻辑未修改；既有范围外改动已保留。

使用的方法是：**只读估算缓存匹配长度，再根据策略生成本轮候选顺序**。实际准入仍检查 Token Budget、KV 容量和表槽。

三种策略的使用条件如下：

|策略|使用情况|排序依据|
|---|---|---|
|**FCFS**|默认配置；或队列超过阈值；或显式启用的等待阈值被触发|原始到达顺序|
|**LPM**|配置为 `lpm`，且未触发 FCFS 降级|缓存匹配长度，越长越优先|
|**LPM-Aging**|配置为 `lpm-aging`，且 `lpm_aging_tokens_per_pass > 0`，未触发降级|匹配长度加等待补偿|

Aging 的评分公式：

```
分数 = 缓存匹配长度 + N × 有效等待轮数
```

它适用于持续有 cache-hot 请求到来、cache-cold 请求容易被反复越过的负载。**本轮接纳了其他新请求，而该请求仍在等待，才累计一次等待轮数。**空队列、零预算、全部拒绝或只有 chunk 续跑时不累计。

具体选择逻辑是：

```
配置为 FCFS → 使用 FCFS
否则，待准入队列 > 128 → 降级为 FCFS
否则，可选等待阈值达到 → 降级为 FCFS
否则，配置为 lpm-aging 且 N > 0 → 使用 Aging 评分
否则 → 使用纯 LPM
```

当前默认是 **FCFS、N=0、等待阈值关闭**。`lpm-aging` 的 N 为 0 时等价于纯 LPM；不会自动启用 Aging。

已经准入的 chunked 请求始终优先续跑。单请求、没有新表槽或 chunk 已占满预算时，会跳过无收益的排序探测。
