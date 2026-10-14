方案是在现有 **“有限 Decode 优先＋Prefill 保护”** 中增加可选的自适应预算。第一版只调整保护轮次的计算量，保留阶段选择、LPM-Aging、ChunkedReq 和缓存准入流程。本轮不修改代码。

**1．功能生效范围。** 只有同时满足以下条件时，才计算动态预算：

```
启用了自适应预算
且阶段策略为 decode-first-bounded
且存在等待中的 Prefill 和活跃 Decode
且连续 Decode 次数达到 K
```

完整行为如下：

|情况|执行方式|
|---|---|
|自适应关闭|保持当前固定预算行为|
|两阶段都有请求，连续 Decode `< K`|优先 Decode|
|两阶段都有请求，连续 Decode `>= K`|优先尝试动态预算 Prefill|
|有 Prefill，没有活跃 Decode|使用正常 Prefill 预算|
|保护 Prefill 无法准入|继续 Decode，保留保护条件|
|Prefill 成功提交|连续 Decode 计数清零|

自适应预算不增加新的阶段策略名称。

**2．新增配置。** 建议增加三个参数：

|配置字段|默认值|作用|
|---|---|---|
|`adaptive_prefill_budget`|`False`|是否启用自适应保护预算|
|`target_batch_tokens`|`256`|用于减去 Decode 负载的目标 Token 额度|
|`min_chunk_tokens`|`64`|保护轮次 Prefill 总预算的下限|

对应 CLI：

```
--batch-schedule-policy decode-first-bounded \
--max-consecutive-decodes 16 \
--protected-prefill-tokens 256 \
--adaptive-prefill-budget \
--target-batch-tokens 256 \
--min-chunk-tokens 64
```

256 和 64 作为第一轮实验起点，后续根据模型和 GPU 测试调整。

配置校验规则：

- 目标额度和最低预算必须为正整数。
- 启用自适应时，阶段策略必须是 `decode-first-bounded`，否则报错，避免配置静默失效。
- 启用自适应时，要求：

    ```
    max_extend_tokens > 0
    
    min_chunk_tokens <= target_batch_tokens
    
    min_chunk_tokens <= min(
        max_extend_tokens,
        protected_prefill_tokens,
    )
    ```

- 自适应关闭时，不增加上述跨参数限制，保证原有小预算配置仍可使用。

**3．预算计算方法。** 在 Scheduler 中增加一个预算计算方法，概念逻辑如下：

```
def _get_protected_prefill_budget(self) -> int:
    upper = min(
        self.prefill_budget,
        self.protected_prefill_tokens,
    )

    if not self.adaptive_prefill_budget:
        return upper

    decode_tokens = len(self.decode_manager.running_reqs)

    return min(
        upper,
        max(
            self.min_chunk_tokens,
            self.target_batch_tokens - decode_tokens,
        ),
    )
```

只在需要保护 Prefill 的分支调用：

```
budget = self._get_protected_prefill_budget()
batch = self.prefill_manager.schedule_next_batch(budget)
```

每次保护尝试重新读取活跃 Decode 数量；同一个 Batch 构造过程中固定使用这次计算的预算。

这里使用 `len(running_reqs)`，不使用负责 KV 预留的 `inflight_tokens`，也不计入 CUDA Graph 的 dummy 请求。

预算仍然表示**整个 Prefill Batch 的新增计算 Token 总量**。不要求单个请求至少计算 `min_chunk_tokens`，实际剩余计算量不足时可以更少。

**4．与已有优化的关系。**

|已有功能|本次处理方式及影响|
|---|---|
|K 次 Decode 保护|保持触发和计数规则，只改变保护预算|
|LPM|保持评分和排序算法；预算会影响每批接纳数量|
|Pass-based Aging|保持有效等待轮次定义，Decode-only 和 Chunk-only 轮次不增加等待计数|
|超过 128 退化 FCFS|保持阈值和真实到达顺序|
|Chunked Prefill|复用现有续算；预算缩小时可能增加续算次数|
|Prefix Cache|复用只读探测、准入后提交和锁定流程|
|KV/Table 资源检查|保持完整检查，不用较小计算预算替代显存需求|
|normal/overlap 循环|保持提交计数位置，不增加同步或墙钟判断|

预算不按 `page_size` 强行取整，继续由现有 KV 分配器处理页对齐，并通过非页对齐预算测试验证。

**5．修改文件范围。**

|文件|计划修改|
|---|---|
|[scheduler/config.py](/home/wty238388/ai-infra/Serving_V1/Promotion/mini-sglang/python/minisgl/scheduler/config.py)|增加三个字段及配置校验|
|[server/args.py](/home/wty238388/ai-infra/Serving_V1/Promotion/mini-sglang/python/minisgl/server/args.py)|增加 CLI 参数和说明|
|[scheduler/scheduler.py](/home/wty238388/ai-infra/Serving_V1/Promotion/mini-sglang/python/minisgl/scheduler/scheduler.py)|保存配置、增加预算计算方法、接入保护分支|
|`tests/core/test_batch_schedule_policy.py`|补齐现有测试支架的配置字段，保持原有测试有效|
|`benchmarks/bench_batch_schedule_policy.py`|扩展现有 CPU 支架和实验参数，比较固定与自适应预算|

计划新增：

- `tests/core/test_adaptive_prefill_budget.py`
- `tests/core/test_adaptive_prefill_benchmark.py`
- `docs/adaptive_prefill_budget.md`
- `artifacts/adaptive_prefill_budget/` 下的测试、实验和差异记录

复用已有 Benchmark 支架，保留之前的实验结果，不复制整套调度模拟代码。

**6．测试计划。** 除重新运行现有 175 项相关测试外，新增测试覆盖：

|分组|验证内容|
|---|---|
|数学边界|目标减 Decode 数量；最低值截断；正常预算与保护上限截断；结果为正整数|
|配置校验|非法类型、零、负数、最低值超过上限、启用时策略不匹配|
|生效条件|关闭时等同固定预算；不足 K 不调用 Prefill；达到 K 使用动态预算；无 Decode 恢复正常预算|
|Batch 行为|多请求共享预算；缓存前缀不占预算；实际计算量不超过预算|
|请求生命周期|长 Prompt 跨多次动态预算续算；Decode 数量变化；Abort；KV/Table 不足后的恢复|
|兼容性|normal/overlap 行为一致；原有 LPM-Aging 与 FCFS 退化测试通过|
|页大小|`page_size=1/4`，包含非页对齐预算|
|资源正确性|请求排空；Table 槽位归还；缓存引用和物理页守恒|

还会专门测试“目标过大导致始终命中保护上限”的情况，确认这属于预算饱和，而不是计算错误。

**7．Benchmark 计划。** 固定 K，比较：

```
固定保护预算
自适应保护预算
```

第一轮参数建议：

```
K：16
正常预算：8192
保护上限：256
目标额度：128、256、512
最低预算：32、64、128
初始活跃 Decode 数：16、64、128、192
```

只运行满足配置约束的组合。覆盖短 Prompt、长 Prompt、缓存冷热混合、KV/Table 紧张，以及动态加入和完成请求的负载。

除了已有指标，还记录：

- 每次保护时的活跃 Decode 数量和选定预算。
- 实际 Prefill Token 数与预算利用率。
- Chunk 续算次数。
- 首次 Prefill、Prefill 完成的等待批次数。
- Decode 输出批间隔。
- 保护成功和资源拒绝次数。
- CPU 调度耗时、缓存复用及资源回收结果。

CPU 实验可以验证预算行为和开销。真实 GPU 实验再测量 TTFT、ITL/TPOT 和吞吐；缺少模型条件时明确标记未运行。

小预算可能减少单次 Prefill 干扰，也可能增加长请求的续算和排队，因此验收不要求所有负载都获得性能提升，要求结果能够解释。

**8．执行顺序与验收标准。** 审查通过后的实施顺序为：

1. 保存当前工作区基线。
2. 添加配置、校验和预算计算。
3. 接入保护分支，更新测试支架。
4. 添加并运行新旧测试。
5. 执行固定预算与自适应预算对照实验。
6. 检查差异，交付文档和结果。

验收必须满足：

- 默认关闭时，原有阶段和预算行为一致。
- 动态预算只在规定的保护条件下生效。
- 活跃 Decode 增多时预算不增大，并正确触及上下限。
- 没有活跃 Decode 时使用正常预算。
- Chunk 续算、缓存准入和资源释放正确。
- 实验能量化单次干扰、Prefill 进展和额外分块开销之间的取舍。
