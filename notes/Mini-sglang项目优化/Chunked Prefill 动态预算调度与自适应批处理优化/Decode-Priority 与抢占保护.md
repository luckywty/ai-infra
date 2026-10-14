这次计划采用：**默认保留 Prefill 优先；新增可选的“有次数限制的 Decode 优先”，当等待中的 Prefill 经历 K 个连续 Decode Batch 后，优先尝试一次小预算 Prefill。**

我已只读复核当前源码和 Git 状态。下面是待审查的实施计划，**本轮没有修改文件，也没有执行测试或 Benchmark。**

**1．明确优化作用的位置**

当前调用链是：

```
normal_loop / overlap_loop
    ↓
Scheduler._schedule_next_batch()
    ↓
优先构造 Prefill Batch，失败后构造 Decode Batch
    ↓
_prepare_batch()：分配 KV 页、准备计算元数据
    ↓
_forward()：提交模型计算
    ↓
_process_last_data()：处理结果、缓存前缀、释放资源
```

新策略主要改动 `Scheduler` 的 **Prefill/Decode 阶段选择**。

现有 LPM-Aging 继续负责 **进入 Prefill 阶段后，等待请求之间的排序**。两层策略分别解决：

|层次|决定什么|使用的方法|
|---|---|---|
|阶段选择|下一批执行 Prefill 还是 Decode|Decode 优先＋连续次数限制＋小预算 Prefill|
|Prefill 内部排序|哪些请求先进入 Prefill Batch|已有 FCFS / LPM / LPM-Aging|

一次仍然执行一个 Prefill Batch 或一个 Decode Batch，沿用现有 Batch 结构。

**2．新增配置及默认行为**

计划增加三个参数：

|配置字段|默认值|含义|
|---|---|---|
|`batch_schedule_policy`|`"prefill-first"`|阶段调度策略|
|`max_consecutive_decodes`|`16`|Prefill 等待期间，触发保护前允许连续提交的 Decode Batch 数|
|`protected_prefill_tokens`|`256`|保护轮次的 Prefill Batch 总 Token 预算|

`batch_schedule_policy` 支持：

```
prefill-first
decode-first-bounded
```

对应 CLI：

```
--batch-schedule-policy decode-first-bounded
--max-consecutive-decodes 16
--protected-prefill-tokens 256
```

配置校验：

- 未知策略报错。
- `max_consecutive_decodes` 必须为正整数。
- `protected_prefill_tokens` 必须为正整数。
- 实际保护预算取 `min(max_extend_tokens, protected_prefill_tokens)`。

`16` 和 `256` 是首轮实验的起点，最终通过 Benchmark 判断是否适合当前模型和设备。

现有 `--schedule-policy` 继续表示 Prefill 内部的 `fcfs/lpm/lpm-aging`，避免混淆两个层次。

**3．精确定义“连续 Decode 次数”**

新增 Scheduler 级状态：

```
self._consecutive_decode_batches = 0
```

它表示：

> 在有请求等待 Prefill 的期间，连续成功提交了多少个 Decode Batch，中间没有成功提交 Prefill Batch。

一个 Decode Batch 包含 1 个或 100 个会话，都只计一次。

计数规则如下：

|事件|计数处理|
|---|---|
|有 Prefill 等待，成功提交 Decode Batch|`+1`|
|成功提交 Prefill Batch，包括 Chunked Prefill 续算|清零|
|没有 Prefill 等待|清零|
|仅选择、准备 Batch，尚未提交计算|不增加|
|Prefill 尝试失败，随后成功提交 Decode|继续增加|
|空转、没有提交任何 Batch|不增加|
|Abort 导致 Prefill 队列清空|清零|

比较条件使用：

```
consecutive_decode_batches >= max_consecutive_decodes
```

这样即使资源不足导致连续 Decode 超过 K，后续仍然保持 Prefill 保护条件。

计数更新放在 `_forward()` 的成功提交路径上。**不放在 `_process_last_data()` 中**，因为 overlap 模式会先提交当前 Batch，再处理上一批结果，按结果处理顺序计数容易错位。

这里统计的是成功提交次数，不要求等待 GPU 完成，也不增加 GPU 同步。

**4．阶段选择的完整规则**

当启用 `decode-first-bounded` 时：

|当前状态|调度行为|
|---|---|
|两个阶段都没有请求|返回 `None`|
|只有 Prefill|使用正常 `prefill_budget`|
|只有 Decode|执行 Decode，保护计数保持为零|
|两者都有，连续 Decode `< K`|优先 Decode|
|两者都有，连续 Decode `>= K`|优先尝试小预算 Prefill|
|保护 Prefill 无法构造 Batch|回退 Decode，保留保护状态|

资源允许、两阶段持续有工作时，典型提交顺序为：

```
D × K → P_small → D × K → P_small → …
```

保护轮次调用已有接口：

```
budget = min(self.prefill_budget, self.protected_prefill_tokens)
batch = self.prefill_manager.schedule_next_batch(budget)
```

如果返回 `None`，再调用 Decode 调度。

实现时分成两个职责：

```
选择并构造一个最终 Batch
    ↓
只对这个 Batch 调用一次 _prepare_batch()
```

不能先准备 Decode Batch、分配其 KV 页，再决定改为 Prefill。

默认 `prefill-first` 分支保留当前调用顺序：

```
prefill_manager.schedule_next_batch(self.prefill_budget)
or decode_manager.schedule_next_batch()
```

**5．“小块 Prefill 保护”的含义和边界**

`protected_prefill_tokens=256` 表示：**保护轮次整个 Prefill Batch 最多计算 256 个新增 Token**。

它可能包含：

- 多个较短请求；
- 若干完整请求和一个部分执行的长请求；
- 已有 ChunkedReq 的续算。

如果当前没有活跃 Decode，则使用正常预算，避免在没有阶段竞争时仍然把 Prefill 切得很小。

这里需要保留三个边界：

- **限制的是本批计算量。** 新请求仍然通过原有完整 KV 需求、输出长度预留、TableManager 等准入检查。
- **保护的是 Prefill 阶段的调度机会。** 资源不足时，K 次后仍可能无法接纳请求。
- **Token 预算不能直接保证 GPU 执行时间。** 长上下文、Batch 组成和 Attention 后端都会影响耗时。

因此，计划的验收表述是：

> Prefill 持续等待时，最多连续提交 K 个 Decode Batch 后，下一次阶段选择优先尝试 Prefill；实际获得服务仍然受资源和队列排序约束。

不会把它表述为严格的 TTFT 上界或每个请求的服务保证。

**6．Chunked Prefill、KV Cache 与 Aging 的兼容要求**

实现时保持以下约束：

- ChunkedReq 仍然优先续算，复用原有 `table_idx`、`cache_handle` 和实际进度。
- Decode 穿插期间，不删除或重新创建等待续算的 ChunkedReq。
- 继续使用 `decode_manager.inflight_tokens` 预留 Decode 所需资源。
- 保留当前缓存双重检查、只读探测、实际准入后的 lock/touch 逻辑。
- Abort、完成请求的资源释放流程继续沿用现有实现。
- 没有成功构造 Prefill Batch 时，不把该尝试视为已经提供 Prefill 服务。
- Decode-only 轮次不增加现有 `waiting_passes`。

最后一点尤其重要：现有 Aging 表示 **Prefill 内部的有效等待轮次**。新的 K 计数表示 **阶段之间的等待**，两者使用独立状态。

当前可选的 `fairness_max_wait_steps` 也继续沿用原语义，不把它解释为连续 Decode 次数或墙钟时间。

第一版采用 K 控制。墙钟超时 `T_max` 放在后续评估中，因为它需要额外定义首次等待、Chunk 续算等待及多 rank 一致性。

**7．计划修改和新增的文件**

生产代码预计修改三个文件：

|文件|计划内容|
|---|---|
|[scheduler/scheduler.py](/home/wty238388/ai-infra/Serving_V1/Promotion/mini-sglang/python/minisgl/scheduler/scheduler.py)|初始化策略和计数；实现阶段选择；成功提交后更新计数；Abort 清空等待队列时重置|
|[scheduler/config.py](/home/wty238388/ai-infra/Serving_V1/Promotion/mini-sglang/python/minisgl/scheduler/config.py)|增加三个配置字段及校验|
|[server/args.py](/home/wty238388/ai-infra/Serving_V1/Promotion/mini-sglang/python/minisgl/server/args.py)|暴露 CLI 参数和准确的帮助说明|

计划新增：

|文件|作用|
|---|---|
|`tests/core/test_batch_schedule_policy.py`|阶段选择、计数、生命周期和默认兼容性测试|
|`tests/core/test_batch_schedule_benchmark.py`|验证 Benchmark 可复现及统计口径|
|`benchmarks/bench_batch_schedule_policy.py`|CPU 调度和资源生命周期 Benchmark|
|`docs/batch_scheduling.md`|策略语义、配置、限制、复现实验方法|
|`artifacts/batch_scheduling/` 下的结果文件|保存测试报告、Benchmark 数据及本次改动记录|

现有 `prefill.py`、`decode.py`、Cache 实现预计直接复用。模型计算、Attention Kernel、Sampling 和 CUDA Graph 核心代码不在修改范围内。

**8．测试计划**

测试分为四组：

|分组|必须覆盖的情况|
|---|---|
|阶段选择|默认 Prefill-first；两阶段同时存在；K−1/K/K＋1 边界；无请求；只有一个阶段；Decode 不可用时回退 Prefill|
|计数正确性|多会话 Decode 仍只加一次；只调度不提交不计数；提交失败不计数；Prefill 成功清零；保护失败不清零；新等待队列不继承已经结束的计数|
|生命周期与资源|长 Prompt 多次续算；Decode 穿插；KV 不足；Table 槽位不足；预算不足；Abort 未准入请求和 ChunkedReq；资源最终回收|
|兼容与执行模式|默认策略与修改前的阶段序列一致；normal/overlap 计数顺序一致；现有 LPM-Aging 测试继续通过；相同输入状态下调度决策确定|

重点增加两个测试：

1. **保护成功测试**：持续 Decode 与资源充足的 Prefill 同时存在，验证第 K 个 Decode 成功提交后的下一次选择优先 Prefill，且保护 Batch 不超过预算。
2. **保护失败测试**：故意让 Prefill 无法准入，验证继续执行 Decode、后续持续重试保护，且没有错误清零、请求丢失或缓存引用泄漏。

normal/overlap 的计数测试使用可控的假 Engine，CPU 环境即可验证提交顺序；缓存和生命周期测试使用真实 CPU Cache/TableManager。

执行时运行全部相关新旧测试，并保留失败信息和 JUnit 报告。当前已有 **106 项通过**的结果只作为之前工作的记录，本次改动后需要重新运行，不能直接沿用。

**9．Benchmark 计划**

比较三种行为：

|对照|用途|
|---|---|
|当前 Prefill-first|默认行为基线|
|无次数限制的 Decode-first|展示 Prefill 阶段等待问题，仅作为实验对照|
|Decode-first-bounded|评估次数限制与保护预算的效果|

实验负载包括：

- 持续长输出 Decode，同时不断加入短 Prefill。
- 持续长输出 Decode，同时加入需要多次分块的长 Prefill。
- cache-hot/cache-cold 混合请求。
- KV 或 Table 资源紧张。
- 只有 Prefill、只有 Decode 的边界负载。

参数扫描：

```
K：1、4、8、16、32
保护预算：64、128、256、512 Token
```

固定随机种子、到达序列、请求长度和资源容量，每组重复运行。阶段策略对比时固定 Prefill 内部策略，另外补充与 FCFS、LPM-Aging 的组合检查。

CPU Benchmark 记录：

- Prefill 首次执行及完成前等待的 Batch 数。
- 有 Prefill 等待时的最大连续 Decode Batch 数。
- 保护尝试次数、成功次数和资源拒绝次数。
- 调度 CPU 耗时的中位数与 P95。
- Batch 数、实际计算 Token 数、缓存复用量。
- 请求是否全部排空，Table 槽位和缓存引用是否正确释放。

CPU 结果只能证明调度语义、资源行为和 CPU 开销。

如果执行阶段具备可用模型、GPU 和请求客户端，再增加真实推理实验，记录 TTFT、Token 间延迟的 P50/P95/P99 和吞吐量。缺少这些条件时明确标记未执行，不把模拟 Batch 等待当作真实 TTFT 收益。

保护失败后的重复探测也纳入 CPU 开销统计；只有测出明显问题后，再评估重试节流机制。

**10．实施顺序与最终验收**

审查通过后按以下顺序执行：

1. 保存当前 Git 状态和任务范围内的文件基线，区分已有改动。
2. 添加配置和校验，保持默认策略。
3. 实现阶段选择、提交计数和队列清空重置。
4. 添加 CPU 测试，修复正确性和生命周期问题。
5. 运行全部相关新旧测试。
6. 执行 Benchmark，分析 K 与保护预算的取舍。
7. 检查差异，交付文件清单、测试结果、实验数据和限制说明。

验收必须同时满足：

- 默认 Prefill-first 的行为兼容。
- 连续 Decode 按 Batch 提交次数准确计数。
- 达到 K 后确实优先尝试 Prefill。
- 小预算能够通过原有 Chunked Prefill 路径执行。
- 资源不足时仍能推进 Decode，资源恢复后能够接纳 Prefill。
- 请求不丢失、不重复接纳，缓存引用和 Table 槽位正确释放。
- overlap 模式不因上一批结果处理而错误重置计数。
- Benchmark 区分调度机会改善、CPU 开销和真实 GPU 延迟收益。

本次计划中需要审查的主要选择是：**默认仍为 `prefill-first`、新策略初始 K＝16、保护预算＝256、第一版采用次数控制，并在保护失败后继续保持 Prefill 优先尝试条件。**

【

这次实现了**“有次数限制的 Decode 优先＋小预算 Prefill 保护”**，作用于 **Prefill 和 Decode 之间的阶段选择**。默认仍保持原来的 Prefill 优先。

主要修改如下：

|文件|修改内容及作用|
|---|---|
|[scheduler.py (line 296)](/home/wty238388/ai-infra/Serving_V1/Promotion/mini-sglang/python/minisgl/scheduler/scheduler.py:296)|增加阶段选择、连续 Decode Batch 计数和 Prefill 保护；处理保护失败、队列清空及 Abort 后的计数重置|
|[config.py (line 25)](/home/wty238388/ai-infra/Serving_V1/Promotion/mini-sglang/python/minisgl/scheduler/config.py:25)|增加策略、连续次数上限、保护预算三个参数，并校验非法配置|
|[args.py (line 228)](/home/wty238388/ai-infra/Serving_V1/Promotion/mini-sglang/python/minisgl/server/args.py:228)|增加对应启动参数|

另外新增了两个测试文件、CPU Benchmark 脚本和说明文档。

启用新策略后：

```
Prefill 和 Decode 同时存在
    ↓
优先执行 Decode
    ↓
连续成功提交 K 个 Decode Batch
    ↓
优先尝试一次小预算 Prefill
```

【**新策略：有 Prefill，但没有活跃 Decode 执行正常预算 Prefill，不必等到 K**】

其中：

- `max_consecutive_decodes=16`：按整个 Scheduler 提交的 Decode Batch 计数，每批只加一次。
- `protected_prefill_tokens=256`：保护轮次整个 Prefill Batch 的新增计算 Token 总预算。
- Prefill 成功提交后计数清零；准入失败则继续 Decode，并保留保护条件。
- 没有活跃 Decode 时，使用正常 Prefill 预算。
- 已有 LPM-Aging 继续负责 Prefill 内部排序；Chunked Prefill 和 KV 准入复用原有流程。

对推理的预期好处是：

1. **改善输出连续性**：Decode 优先减少生成过程被连续 Prefill 打断的机会。
2. **给 Prefill 提供调度机会**：避免无限 Decode 优先让新请求一直无法进入 Prefill 阶段。
3. **控制 Prefill 的单批计算量**：长 Prompt 分块推进，降低单次 Prefill 对 Decode 的干扰。

验证方面，**175 项测试全部通过，875 次 CPU Benchmark 全部排空并通过资源检查**。在 K=16、预算 256 的短请求模拟中：

- 相比无限 Decode 优先，首次 Prefill 等待 P95 从 **127 批降至 16 批**。
- 相比 Prefill 优先，初始 Decode 最大输出间隔从 **17 批降至 2 批**。

但长请求、混合负载中，分块续算可能增加部分请求的等待。**目前验证的是调度行为和资源正确性，尚未证明真实 GPU 的 TTFT、ITL 或吞吐改善；资源不足时，K 也不能保证请求成功准入。**

】
