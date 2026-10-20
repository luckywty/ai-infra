# AI Infra / LLM 推理系统实习：项目审计、岗位证据与 8 周路线

> 调研及代码复核日期：**2026-10-14，Asia/Shanghai**。本次仅调研、评估和规划；没有修改业务源码、实现新优化或执行耗时 Benchmark，也没有自动推送本报告。

对象：211 本科、985 研一，目标是第一份推理系统实习。学历和已接触技术来自本人陈述，不等于技能测评结论。新增约束为：**广东省优先，2027 年 2 月左右到岗，每周 3–4 天，连续约 3–4 个月**。最后一项原话为“可连续学习”，下文暂按可连续实习理解，投递前确认；毕业年份、导师许可和具体可到岗日仍待确认。

---

## 1. 执行摘要：当前最值得做的三件事

1. **把 mini-sglang 的两组机制做出真实证据。** 优先证明 bounded Decode-first 的 ITL/TTFT 取舍，以及 LPM-Aging 相对纯 LPM 的 cold 等待/吞吐取舍。源码和 CPU 测试已具备基础，尚缺真实 GPU、正确的端到端计量及性能解释。不把“又加一种策略”当主要进步。
2. **用限时编程与实际 trace 补齐 C++、系统和 Profiling 的证据。** 每周 3 小时 C++、3 小时算法持续训练；先学 CUDA 执行/访存/异步基础，再用一个小 Triton 算子完成正确性—计时—瓶颈分析闭环。不能由仓库存在 CUDA 代码推断你会 CUDA。
3. **沿已有项目进入正式 SGLang，并提前筛选到岗资格。** 第 6 周开始只读一条正式框架调用链，做一个有范围的小测试/问题定位；不同时学透 vLLM 和 SGLang。第 4 周即启动试投/联系，第 8 周形成主材料，2027 年 1 月再次确认 2 月 HC 和到岗安排。

> **推荐主线：mini-sglang 实验闭环 → 正式 SGLang 的调度/缓存模块。辅助线：一个独立目录内的小型 CUDA/Triton 算子实验，预算约 16–24 小时。** 不另开完整推理框架，不把多卡通信、完整论文复现、编译器、量化工具链一起纳入 8 周必修。

你的项目已经有成为简历核心项目的结构；目前最弱的是“性能与个人掌握程度的证据”，不能据此给出已达某公司录用标准的结论。到岗天数、实习月数及城市约束也可能比新增技术更早决定能否投递。

### 1.1 时间和完成标准

以 2026-10-15 开始，主计划 8 周至 12 月 9 日，延伸 12 周至 2027-01-06。每天 3–4 小时，采用**每周 24 小时**预算：主线及辅助实践 16h、C++3h、算法 3h、求职/表达 2h，合计 192h。资源阅读计入这些时段；GPU 批量运行的等待时间另计，不要求你一直盯屏。每周只有 21h 时先减辅助线和 P2，不先削掉正确性与重复实验。

“达到申请条件”的工作定义：能独立修改一个有测试的 Python 系统模块，能完成基本 C++ 手写题，能解释一条推理请求链和至少一个真实瓶颈，能拿出同条件配对实验及失败记录。它不是保证拿到实习 offer；不同 JD 的底层、研究、平台要求差别很大。

---

## 2. mini-sglang 已完成项目审计

### 2.1 本次实际检查的证据

- 阅读附件完整任务；重新检查 Git 状态、diff、当前配置、缓存/调度函数、测试及模拟 Benchmark，而非仅看旧说明。
- 本地上游 HEAD 为 `9a91cfafe754aa85daee49998176275667eb58f2`，四项改动仍在本地工作区；tracked diff 为 13 文件、687 行新增/72 行删除，含注释，不能全算算法实现。diff SHA256 为`405adc646024b672c0bb10643c369bc94c309ddd47b068e69c0c2630f7b039fc`，与此前审计一致。
- GitHub 优化快照为 [`7cd28c7`](https://github.com/luckywty/mini-sglang-lru-touch/commit/7cd28c7bda8f6881558fe810694019d3e46994b5)。它属于用户仓库独立历史，不能将该仓库的`main`或`before`自动当上游原版。新机器需从`sgl-project/mini-sglang` 获取上述原版 SHA。
- 本会话此前上传前的 CPU 回归 XML：`/tmp/minisgl-github-update-20261014-7wby2jay/tests.xml`，249 tests、0 failures/errors/skips、7.136s；本次只读取，没有再跑。该临时路径不属于可长期依赖的交付物。仓库 [历史XML](../artifacts/adaptive_prefill_budget/tests.xml) 也记录 249 项，但时间不同，不能说是本次新实验。
- [模拟器](../benchmarks/bench_batch_schedule_policy.py)用 CPU 张量生成 `100000+uid`作为输出，替换 graph/metadata/stream 等；[历史JSON](../artifacts/adaptive_prefill_budget/benchmark.json) 明确标注 no GPU TTFT/ITL/throughput。metadata 中 CUDA 可见或 GPU 名称，不代表实验执行了 GPU 模型。
- [五份实验规划](../benchmark_plan.md)已存在，拟议 `benchmarks/serving/` 尚不存在；没有证据表明另一台 5060Ti 已经完成正式实验。本机审计环境是 3060 Laptop 6GB，不能替代目标机验证。

### 2.2 项目能力映射

复杂度是模块实现的相对判断，不是对个人水平打分。状态 V=所列 CPU 语义测试已覆盖，G=真实 GPU/端到端仍待验证，P=仅规划。

| 实际模块与状态 | 关键知识 | 工程复杂度 | 已有证据 | 能力缺口/边界 | 简历价值 |
| --- | --- | --- | --- | --- | --- |
| **A：Radix 两阶段准入，V+G** | speculative probe、ref count、锁后容量、LRU、版本失效 | 中高；跨缓存/准入接口 | `MatchProbe`，probe/refresh/commit；拒绝快照、partial split、保护成本测试 | 缺旧路径独立 A/B；多个父链遍历仍在；串行提交不等于并发事务；异常回滚不完整 | 很适合讲工程正确性和资源生命周期；未测前不写 GPU 加速 |
| **B：FCFS/LPM/LPM-Aging，V+G** | 稳定排序、缓存局部性、公平、有效机会计数 | 中；队列身份和 chunk 状态易错 | `_get_candidates`、`waiting_passes`；稳定序、fallback、无效轮不老化、重排后移除测试 | 候选失败即 break、chunk 先续算、资源不足会阻塞；无墙钟服务上界 | 与缓存/调度岗位直接相关；需 cold 等待与吞吐双报 |
| **C：bounded Decode-first，V+G** | 阶段竞争、连续批处理、提交与完成次序 | 中；overlap 边界重要 | `_select_next_batch`、`_record_batch_submission`；K 边界、失败保持 streak、normal/overlap 替身测试 | K 限制尝试前的 Decode 提交批次，不保证准入；不抢占 GPU 计算 | 最值得先测真实混合负载 ITL 尾部 |
| **D：动态保护预算，V+G** | workload signal、clamp、chunk 成本/延迟权衡 | 公式低、集成中 | `min(U,max(floor,target-n_decode))`、配置约束、动态数量与 chunk 生命周期测试 | 依赖 C；只调 Prefill-only 保护批；不是混合 batch、时间反馈或动态 KV 预留；8GB 低 batch 下变化可能很小 | 可作边际实验；不是最强的独立创新点 |
| **Overlap scheduling，继承能力** | CUDA streams、event、CPU/GPU 流水 | 原机制复杂，但此次无独立改造证据 | `overlap_loop` 及既有 D2H 同步；新增 C/D 适配测试 | 没有真实 trace 证明 overlap 收益；大量 diff 为说明注释 | 可讲理解和兼容验证，不能写“独立实现 Overlap 引擎” |
| **KV 物理页管理，继承 + 局部准入改造** | 页舍入、共享页保护、future reservation、回收 | 中高 | CacheManager 与资源测试；page1/4 场景 | 已有审计记录 page>1 多新请求尾页预留缺口；没有通用异常回滚；`check_integrity()` 在 Radix 类仍为 pass | 适合展示边界分析；不应宣称已全面解决 OOM |
| **Benchmark/指标，CPU 已有、GPU 流程 P** | 计时域、trace 回放、分位数、重复与 SLO | 工具中等、方法要求高 | 既有 CPU harness/JSON；五份规划 | 旧 client 请求开始计时太晚、事件计数混用；精确 token/请求终态采集待实施 | 完成后会明显提高整个项目可信度 |
| **CUDA/Triton/TP，主要为继承** | kernel、warp、FFI、NCCL/TP | 底层高，但个人贡献未证实 | 仓库有 C++/CUDA/Triton/通信实现；本次这些目录无 tracked 改动 | 无个人 kernel、nsys/ncu 报告或多 GPU 实验可确认 | 暂不能写“熟练算子优化/分布式推理” |

**代码定位：**[cache.py](../python/minisgl/scheduler/cache.py)的 `try_match_and_lock` L40/`allocate_paged` L81；[radix_cache.py](../python/minisgl/kvcache/radix_cache.py)的 probe L175、refresh L204、commit L222、reset L289、integrity L300；[prefill.py](../python/minisgl/scheduler/prefill.py)的候选排序 L165/调度 L213；[scheduler.py](../python/minisgl/scheduler/scheduler.py) 的预算 L299、阶段 L311、提交 L340、overlap L100；[配置](../python/minisgl/scheduler/config.py)、[测试](../tests/core/test_prefill_policy.py)。行号对应本次工作区。

### 2.3 实现、测试和性能必须分别表述

A 的新请求链为 `probe/refresh → 预测锁后容量 → commit(split) → lock → touch`。commit 本身不负责 lock/touch；排序预测不能污染 LRU。旧路径先 match/split/touch 再检查，拒绝也可能改变树。A 不是“新增 Prefix Cache”，也不是单纯删除一次 if。

B 的分数为 `实际估算缓存长度+N×有效等待轮`。只有本轮准入至少一个 NEW 请求，仍等待首次准入的请求才老化；chunk-only/预算 0/全拒绝不计。新队列>128 时退回 FCFS。C 仅在两阶段竞争时用 K 提供保护尝试。D 仅改变保护批新增 Prefill token 总预算，完整剩余输入与输出的 KV 预留没有改。

因此，249 项 CPU 通过支持的是**覆盖场景内的调度语义、计数与正常资源生命周期**；不证明 GPU 数值一致、所有分页边界安全、真实吞吐提升或多卡状态一致。旧 client 在 `await create(...)` 返回后才建起点，不能据其现有 TTFT/TPOT 标签直接写简历。[client.py](../python/minisgl/benchmark/client.py)

### 2.4 原创性与个人掌握的判断

- FCFS/LPM/Aging、Decode 优先、chunk 化都是已有思想；Continuous Batching、Radix 基础设施、attention/graph/kernel/TP 主要来自上游。不能把它们都计作个人创新。
- 当前 diff 体现的工程工作包括：拒绝前无副作用预测、stale probe 刷新、按对象身份处理重排队列、有效准入轮计数、按实际提交而非选择计 K，以及相应边界测试。这些比“公式看起来复杂”更有面试价值。
- 单凭源码无法核定哪些由你独立设计、哪些参考方案或由 AI 协助。个人贡献需要设计记录、限时复写、故障定位和追问解释验证；使用辅助工具本身不减分，不能解释或验证生成代码才是风险。
- 要能主动给反例：无资源时 Aging 不保证服务；小 chunk 仍需完整未来 KV 预留；大 target 让 D 恒为固定上限；队列超阈值让 LPM 空转；S-default 依然执行 A，不能当原版。

---

## 3. 真实岗位调研：14 个条目及来源边界

### 3.1 调研口径

全部在 2026-10-14 检查。A=官网可读取职责/任职正文；B=招聘平台正文可读；C=官方页面有搜索索引正文，但直接抓取为空/受限；D=第三方聚合全文，未成功核对官网正文。**C/D 不是确认仍在招。** 对 B 也只记录页面显示状态，不保证发布者身份、实际 HC 或 2027 年 2 月能入职。

主能力样本为 J01–J11 共 11 条，供技术需求编码；另列 J12 官方列表、J13 混合课题、J14 已下线历史样本。不同地点/时长仍可作为技能参照，不能全算适合你投递的岗位。没有获得可复核正文的腾讯、MiniMax、月之暗面岗位不编入统计。阿里 ROCK 团队页主要列正式/专家岗，也未冒充实习岗位。

“页面日期”按原站明确字段记录；平台刷新不等于首发，收录不等于招聘更新。只有相对 Posted 信息的快照不换算为精确发布日期。所有岗位在正式投递前重新核对。

### J01 百度：AI Infra 实习生，J104435〔A；训练/推理混合〕

[官网全文](https://talent.baidu.com/jobs/detail/INTERN/6ba68d4d-889c-48a1-bed7-d4112c43014b)。北京，日常实习，页面日期 2026-08-13。职责横跨训练基础设施和推理效率；任职明确要求 C/C++/Python 至少一种、PyTorch、相关框架实践以及 GPU 性能分析/CUDA 等能力；KV、PD 分离、投机和量化属于工作方向。强调计算/通信效率与工程排障。C++/Python 为语言备选，不是每门都必须精通；CUDA 要求较强。**非广东，技能对标用；天数/月数未披露。**

### J02 百度：视觉模型推理开发实习生，J98660〔A；部署/异构推理〕

[官网全文](https://talent.baidu.com/jobs/detail/INTERN/88ef5f2d-93ca-445b-9b39-90b7e2f4880e)。北京，2026-07-21。部署和移植 CV/多模态模型，维护部署工具；要求研究生、Linux、C++ 或 Python、算法编码；GPU 体系结构/CUDA、容器和服务部署为优先项。没有明确要求 vLLM 或独立 CUDA kernel 开发。**非广东且要求至少 6 个月，与你当前时长不匹配；每周至少 3 天，导师知情同意。**

### J03 字节跳动：AI Infra 后端开发实习生 - 计算〔B；IaaS/平台〕

[牛客招聘页](https://www.nowcoder.com/jobs/detail/431803)。页面展示公司人力资源账号；北京、4 天/周、至少 3 个月，未见可靠发布日期，核验日为 2026-10-14。要求 2027 届、算法数据结构、一门 C/C++/Python/Go；PyTorch、vLLM/SGLang、GPU/高性能网络开发为加分。职责是 AI Infra 后端，不等于纯推理 kernel。**地点不符，毕业届别也待核；不能因为有 SGLang 关键词就当主投。**

### J04 Firmior AI：推理系统优化实习生〔B；GPU 性能/建模〕

[平台全文](https://www.shixiseng.com/intern/inn_wem2bplguili?pcm=pc_SearchList)。2026-08-18 刷新，截止字段 2026-11-18。职责为性能建模、Roofline、Nsight 及 kernel 优化；任职明确 C++/CUDA/Python、GPU 架构、PyTorch/Transformer、底层分析；适合作为较高的底层目标。**深圳可选，但页头写 5 天/周、6 个月，正文写建议 4 天以上、3–6 个月，条件冲突。必须先确认，不能取较宽条件当已匹配；2027 年 2 月机会另核。**

### J05 广和通：高性能算子实习生，J13895〔B；端侧 NPU/DSP 算子〕

[平台全文](https://www.shixiseng.com/intern/inn_rk2e67gvpbfb)。深圳，2026-05-11 刷新、截止字段 2027-03-25；硕士、5 天/周、至少 3 个月全职。开发/测试 Transformer 等算子，要求 C/C++/Python、硬件内存层次与模型计算特征；CUDA/OpenCL/TVM 等经验、Profiler 为优先项。CUDA 并非唯一可选技术，硬件也不等于 NVIDIA。**时长接近但 5 天与当前安排冲突；岗位重点是算子，不是 Serving 调度。**

### J06 中电科普天科技：实习生（AI 平台）〔B；平台/运行时集成〕

[智联正文](https://www.zhaopin.com/jobdetail/CC000269110J40898048513.htm)。广州花都，3 个月、5 天/周；页面无可靠发布时间，核验 2026-10-14。Python、Linux/Git、HTTP/API 为基础；Docker/K8s、vLLM 等为加分。职责包括推理运行时指标、服务治理、测试门禁和可观测性，另有训练平台工作。**广东但 5 天不匹配；技术门槛偏平台工程，可作为副投类型，不应为此把主线改成 Agent 应用。**

### J07 阿里巴巴：LLM 推理系统研发实习生（推理缓存方向），199909660004〔D；缓存/Serving〕

[官网入口](https://campus-talent.alibaba.com/campus/position/199909660004)直接抓取无正文；[第三方完整转录](https://jobs.ultraai.site/jobs/alibaba_campus/internship%3A199909660004) 显示 2026-09-18 收录，北京/杭州。转录职责高度贴近本项目：KV 配额/生命周期、缓存复用与淘汰、分层存储、vLLM/SGLang 集成和性能观测；要求至少一门 C++/Go/Python、Linux、OS/网络/并发/分布式基础，框架和 GPU/压测是加分。**仅作有出处的参考转录；非广东且转录要求≥6 个月，不匹配。**

### J08 蔚来：大模型推理框架工程师实习生，A125076〔D；引擎 + 平台〕

[含岗位码转录](https://www.deizao.cn/index/index/shixixq/jobid/261141) 标 2026-09-16 整理、截止 2027-01-17；正文列上海/深圳、至少 4 天/周、连续 3 个月以上、6 个月优先、尽快到岗。职责覆盖 KV/PagedAttention/连续批处理和服务平台，Python、Transformer、Linux/Git 是基础，CUDA/Triton、框架二次开发、压测/Profiling 为加分。**有可能贴近你的 4 天/3–4 个月安排，但官网 JD 未核到、2 月到岗与“尽快”存在不确定性。聚合页的每天工时字段互相不一致，不据此承诺短工时实习。**

### J09 NVIDIA：Software Engineering Intern, DLFW Comms - 2027，JR2025701〔C；分布式通信〕

[官方Workday页面](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Software-Engineering-Intern--DLFW-Comms---2027_JR2025701)。官方搜索快照显示上海/北京/深圳、相对 Posted 13 Days Ago；直接抓取正文为空，日期不换算。职责把通信技术接到 PyTorch/vLLM/SGLang 等；要求相关硕博、Python/C++/CUDA 或 DSL 原型能力、LLM 和并行基础，profiling/kernel/生产栈为加分。若干技能为方向备选，不是要求全套。**深圳可选，周天数/月数未披露；当前项目尚不足以证明通信优化能力，列挑战/长期对标。**

### J10 NVIDIA：AI Computing Software Intern, GPU Kernel Libraries - 2027，JR2026170〔C；GPU 库/编译〕

[官方页面](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/AI-Computing-Software-Intern--GPU-Kernel-Libraries---2027_JR2026170)。官方索引正文含 C/C++ 和 Python、CUDA、LLVM/MLIR 要求及 GPU 库 kernel/性能分析职责；地点上海/北京，相对 Posted 10 Days Ago，直接抓取受限。**非广东，编译与底层要求超出 8 周主目标；用作区分“算子岗”和“Serving 岗”的样本。**

### J11 影石创新：AI 开发实习生，7669741612285872434〔C；端侧部署〕

[官方飞书招聘页](https://arashivision.jobs.feishu.cn/campus/position/7669741612285872434/detail)。官方搜索索引给出深圳/实习、C++/数据结构/软件设计、模型量化/剪枝/图优化；部署、硬件体系结构及 SIMD/OpenCL/CUDA 经验为优先。职责是框架选型、模型部署和算子性能/兼容性排查。直接抓取为空，未见发布日期、周天数或月数。**广东候选线索；须核实实际条件及是否有 LLM 方向，不能把视觉音频端侧推理等同在线 LLM Serving。**

### J12 百度：大模型推理研发实习生，J106486〔官网列表可读，任职要求未取得〕

[官方实习列表](https://talent.baidu.com/jobs/list?recruitType=INTERN) 明确显示深圳、2026-10-13、日常实习，职责包含 KV、batch 调度、TTFT/吞吐/显存及多卡稳定性。列表通用说明为日常实习不少于 3 个月；**未取得该岗位独立任职正文，不推断语言硬门槛、每周天数或 2 月可到岗**。这是最贴近你主线的广东跟进线索之一，不纳入 J01–J11 任职频率分母。未采用第三方页面中 AI 生成的薪资、评分和“申请策略”作事实。

### J13 百度：Summer Camp - Agentic RL /大模型平台策略推理优化实习生，J100476〔A，混合课题〕

[官网全文](https://talent.baidu.com/jobs/detail/INTERN/52415af4-405a-4f07-8392-d4b63e76df8c)。北京，页面日期 2026-07-21；其中推理课题要求理解模型推理、vLLM/SGLang 细节和投机方案，另一部分是 Agentic RL 课题。**只提取推理方向作参考，不把 RL 要求合并成你的必修，也不将两个课题拆成两个独立岗位来增加样本。** Summer Camp 标题意味着 2 月状态需重核；非广东，未进主频率。

### J14 九天睿芯：AI 工具链开发实习生〔B，已下线历史参照〕

[平台历史正文](https://www.shixiseng.com/intern/inn_wqud8ig3egjp?pcm=pc_SearchList)。深圳、3 天/周、3 个月，但页面明确下线，截止 2026-04-12、刷新 2026-04-13，不能当当前机会。C++/Python、现代 C++、Linux、模型转换正确性、自动化测试是核心；LLVM/MLIR/量化/硬件编程属进一步方向。**仅用于说明部分工具链岗确实要求更深 C++；不纳入当前主样本频率，更不能因时间匹配就推荐直接投。**

### 3.2 广东投递取舍

| 跟进顺序 | 条目/岗位类型 | 下一步确认什么 | 当前结论 |
| --- | --- | --- | --- |
| **1** | J12 百度深圳推理、同类日常推理引擎/缓存岗 | 是否接受 2027-02 到岗、每周 4 天和 3–4 个月；独立 JD 及毕业限制 | 最贴主线，资格未完全核实 |
| **2** | J08 蔚来深圳推理框架、J11 影石框架/部署 | 先找到官方有效 JD；2 月时间、周天数、具体模型/框架范围 | 条件候选，不承诺仍有 HC |
| **3** | J09 深圳通信/框架实习 | 周天数、月数、项目 scope；是否允许以推理框架而非通信 kernel 切入 | 挑战投递，不为其提前做完整多卡项目 |
| **4** | J04 Firmior | 页头/正文冲突、时长与到岗；核实团队和岗位 | 条件确认后再决定，技术短板也较多 |
| **5** | J05/J06 算子或平台邻近岗位 | 能否接受每周 4 天 | 按当前 5 天要求不匹配；不能默认会放宽 |

> 当前样本中**没有一条已被完整核验为“广东 +2 月入职 +3–4 天 +3–4 个月全部满足”**，不代表市场不存在这种岗位。至少每两周刷新；先确认资格，再按 JD 分配学习投入。你若能稳定 4 天，通常比在“3–4 天”间摇摆更便于约定，但不能承诺做不到的时长。研一并不自动满足 2027 届定向实习，毕业年份须真实填写。

---

## 4. 岗位能力需求矩阵与统计边界

### 4.1 编码方法

只对 J01–J11 的已取得文本编码。R=任职明确列出，**包含“至少一门/某方向之一”的备选**；D=职责提到而非入职硬要求；+=明确加分/优先。每能力每岗位只记一次，优先 R、其次 D、再 +。—表示文本未显式提及，不等于工作不需要。

只要是 LLM 相关团队不自动算“熟悉 LLM”；在模型推理/Transformer/LLM 工作内容明确时才计 LLM。GPU 编程泛称不自动算 CUDA；hardware architecture 计 CPU/GPU/NPU 架构，不能解读为 NVIDIA 架构都熟悉。Profiling 是系统或 GPU 性能分析，不等于都要求 Nsight；Benchmark 是明确性能测评/压测/建模，不把所有模型效果评估都算性能 Benchmark。正式 vLLM/SGLang 项只统计明确项目名；CUDA Graph 无显式条款则不补计。

### 4.2 出现频率

| 能力 | R | D | + | 任一出现/11 | 比例 | 正文直接可读子集 J01–J06 /6 | 对本路线的解释 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Python** | 10 | 0 | 0 | 10 | 91% | 6 | 最通用；含语言备选，仍须独立编码/调试 |
| **C/C++** | 9 | 0 | 0 | 9 | 82% | 5 | 高价值基础；不能解释为 9 岗都同时要求熟练 C++ 和 Python |
| **明确现代 C++ 版本/特性** | 0 | 0 | 0 | 0 | 0% | 0 | J14 历史工具链岗明确要求；主样本不写不代表可不懂生命周期 |
| **CUDA/Triton** | 4 | 0 | 4 | 8 | 73% | 4 | 强底层岗为硬项，Serving/部署岗常为加分；CUDA 与 Triton 非人人都要 |
| **硬件体系结构/内存层次** | 2 | 1 | 2 | 5 | 45% | 3 | 算子/性能方向更重要，区分 GPU/NPU 背景 |
| **PyTorch** | 2 | 1 | 1 | 4 | 36% | 3 | 显式计数，未从“深度学习框架”泛称补计 |
| **Linux** | 4 | 0 | 0 | 4 | 36% | 2 | 文本低估通用开发前提；仍是本地 Serving 实验必需 |
| **OS/内存/并发基础显式条款** | 1 | 0 | 0 | 1 | 9% | 0 | 口径严格，不从 C++/Linux 推断；项目本身强烈要求这些能力 |
| **网络/RPC/通信基础** | 3 | 0 | 1 | 4 | 36% | 2 | 系统/平台与通信岗更突出；R 也有方向备选 |
| **分布式/并行** | 3 | 1 | 0 | 4 | 36% | 1 | 分布式基础与多卡调优不是同一级；后者不必 8 周全部实现 |
| **LLM 结构/推理** | 5 | 3 | 1 | 9 | 82% | 6 | 结构和运行特性必须能讲透 |
| **KV Cache/缓存管控** | 0 | 4 | 0 | 4 | 36% | 2 | 特定 Serving 岗职责核心；PagedAttention 未单独当多数硬项 |
| **Continuous Batching** | 0 | 2 | 0 | 2 | 18% | 1 | 更适合做项目深度证据，非所有 Infra 岗关键词 |
| **CUDA Graph** | 0 | 0 | 0 | 0 | 0% | 0 | 是当前引擎实测必要控制变量，不因为频率 0 就忽略 |
| **量化/压缩** | 1 | 2 | 0 | 3 | 27% | 1 | 部分部署/端侧核心；本轮基线不改变精度，先掌握概念 |
| **Profiling/性能分析** | 2 | 1 | 4 | 7 | 64% | 3 | 高收益补齐点，应产生 trace/定位报告而非只会工具命令 |
| **Benchmark/性能评测** | 0 | 3 | 1 | 4 | 36% | 2 | 是证明优化价值的方法；JD 不一定写统计术语 |
| **vLLM/SGLang 明确提及** | 1 | 1 | 4 | 6 | 55% | 3 | 源码/贡献通常是加分或职责，选择一个正式框架即可 |
| **测试/正确性/兼容性** | 0 | 5 | 0 | 5 | 45% | 2 | 工程岗位需要闭环；不能只展示最快路径 |
| **容器/云原生** | 0 | 2 | 2 | 4 | 36% | 2 | 平台方向重要；当前主线只做部署/隔离基础，不全学 K8s 开发 |
| **kernel 实现/调优** | 1 | 3 | 0 | 4 | 36% | 2 | 岗位分化明显，小项目补证即可，核心算子岗另需更高深度 |
| **LLVM/MLIR 明确要求** | 1 | 0 | 0 | 1 | 9% | 0 | 针对 J10，不升为全部岗位的必修 |

> 样本小、非随机、按主题搜索，且 C/D 渠道有可访问性偏差。此表是**11 条 JD 文本的描述统计，不是全国岗位比例或录用概率**。直接正文子集仅 6 条，用于看渠道差异，不能视为独立验证样本。J14 下线、J12 缺任职正文、J13 混合课题没有混进分母。核心优先级同时依据项目缺口、硬件和收益时间，不能单按词频排序。

### 4.3 复核编码表

简写：P=Python，C=C++，U=CUDA/Triton，H=硬件架构，T=PyTorch，L=Linux，S=OS/并发/内存，N=网络，D=分布式，M=LLM，K=KV，B=Continuous Batching，Q=量化，F=Profiling，E=Benchmark，V=vLLM/SGLang，X=测试，O=容器，Z=kernel，I=编译。未列项为—。R 中的 OR/方向备选按原 JD 读取。

| JD | R（含备选） | D 职责 | + 优先 |
| --- | --- | --- | --- |
| **J01** | P C U T D M F V | K Q | — |
| **J02** | P C L | M | H U O |
| **J03** | P C | M | T V N |
| **J04** | P C U H T M F | E Z | — |
| **J05** | P C H M | E X Z | U F |
| **J06** | P L N | K B M X O | V |
| **J07** | P C S N D L | K E X | M F V O |
| **J08** | P L M | D K B Q O X | U F E V |
| **J09** | P C U D M N Z | T V | F |
| **J10** | P C U I | H F Z | — |
| **J11** | C Q | X | H U |

此表不对“部署”另计频率；J02/J05/J06/J08/J11 等职责详情已列部署方向。对“必须”和“优先”有歧义时保守按原文级别，不提升要求。下一次更新岗位快照时应重新编码，而不是沿用本表比例。

---

## 5. 个人能力缺口：证据、目标与快速诊断

水平标尺：L1=能解释并用例推演，L2=能独立写小模块/测试并 debug，L3=能在真实负载定位和验证改进。下表的“现有”只是项目支持的证据；**除非完成诊断，不给你个人贴“熟练/不会”的标签**。P0 针对近期主投的 Serving/Runtime 岗位，算子专岗的门槛另见 J04/J05/J10。

### A. LLM 推理理论

| 能力点 | 目标水平 | 可确认的已有证据 | 当前判断与不确定性 | 是否补/优先级 | 快速验证（每项 1–3 问或任务） |
| --- | --- | --- | --- | --- | --- |
| **Transformer/GQA/KV 形状** | L2，能从 config 算容量 | 项目使用 GQA 模型和 KV 池；未改模型层 | 接触自述，推导能力待验证 | 必补 P0，D01 | 20min：写出 Q/K/V 与 KV cache 形状；推导 Qwen3-0.6B BF16 每 token112KiB；说明为什么用 KV heads 而非 Q heads |
| **Prefill/Decode 成本** | L2，按 shape 讨论瓶颈 | 调度按两阶段分支 | 逻辑使用已证实，计算/带宽判断待验证 | 必补 P0，D02 | 给定短/长 context、batch1/16，列权重与 KV 读写量；解释为何“Decode 总是带宽瓶颈”并非无条件成立 |
| **采样/EOS/数值正确性** | L2，能发现截断/重复/偏差 | SamplingParams、结果路径存在；核心未改采样 | 数值验收和 token 计数尚缺 GPU 证据 | 必补 P0，D03 | 用 O=1/32、自然 EOS 和 ignore_eos 三例定义实际 O；解释为何文本重分词未必还原生成 IDs；设计跨 chunk 参考验证 |
| **量化和精度概念** | L1，本轮不实现量化 | 本地有量化模型目录不算贡献；本实验固定 BF16 | 实践待验证，不是主缺口 | 概念 P1，实现 P2，D04 | 区分 weight-only、activation 和 KV 量化分别省哪块内存；解释相同 dtype/权重为何是调度对照控制变量 |

### B. GPU 底层性能

| 能力点 | 目标水平 | 可确认的已有证据 | 当前判断与不确定性 | 是否补/优先级 | 快速验证 |
| --- | --- | --- | --- | --- | --- |
| **CUDA 执行与异步** | L2 小程序，不要求复杂 kernel | 上游 store/index CUDA、stream/event 调用；无相关 diff | 个人 kernel 能力待验证 | 基础 P0，D05 | 30min 说明 grid/block/warp 与 mask；写 vector add 边界；解释 CPU 包住 launch 的计时为什么可能偏小 |
| **访存/归约/资源限制** | L1→L2 | 仓库有 warp copy，未见个人优化 | coalescing、shared memory、register/occupancy 待验证 | P1，D06 | 比较连续与跨步访问；解释高 occupancy 不必最快；给一个 block 调大反而变慢的可检验原因 |
| **nsys/ncu 与性能建模** | L2→L3 一个小范围 | 无本人 GPU trace 报告 | **明确缺证据**，是高收益补齐点 | nsys P0、ncu P1，D07 | 从 warmup 后的 trace 标出 CPU 调度、kernel、H2D/D2H 与等待；区分 host stall、launch gap 和 kernel 长；只选一个热点做 ncu |
| **Triton mask/stride/reduction** | L2 单算子 | 存在上游 fused MoE 代码，未改 | 不能由“仓库有 Triton”认定掌握 | P1，D08 | 独立写行归约/RMSNorm，覆盖非整块宽度、FP32/BF16；明确支持/拒绝的 stride，做数值对照和计时 |
| **Tensor Core/低层汇编** | L1 知道适用条件 | 没有个人 GEMM/CUTLASS/PTX 优化证据 | 暂不以此申请核心 GPU 库岗 | 概念 P1，优化 P2，D09 | 解释何时算子无法利用 Tensor Core；指出 RMSNorm 和 GEMM 在算术强度上的区别；不要求手写 SASS |

### C. 推理系统工程

| 能力点 | 目标水平 | 可确认的已有证据 | 当前判断与不确定性 | 是否补/优先级 | 快速验证 |
| --- | --- | --- | --- | --- | --- |
| **请求全链路/Batching** | L2 | API→Tokenizer→Scheduler→Engine→Detokenize 链存在，修改了部分调度 | 最接近你的项目主线；仍需独立复述 | P0，D10 | 15min 从 HTTP 入口画到首 token；指出 pending/active/chunked 在何处转移；说明 max running 与 client 并发区别 |
| **KV 生命周期/页预算** | L2，能维护不变量 | A 及页/引用测试 | 有实现线索，异常/紧内存边界仍缺验证 | P0，D11 | 手推 partial probe→split→lock→finish；复述已有 page4 反例；列资源拒绝与 GPU OOM 区别 |
| **缓存排序/公平** | L2 | B 代码与 effective pass 测试 | 可作为强项目主题；个人推理待问答验证 | P0，D12 | 无资源时为何 Aging 不保服务；chunk-only 为何不加 waiting_passes；队列 129 时走哪条路径 |
| **阶段选择/动态预算** | L2 | C/D 提交计数和 clamp 测试 | 已实现但真实收益未知 | C P0、D P1，D13 | 手推 K4 提交/失败/重试；给出 D 恒 U 及 floor 饱和的配置；解释为什么不是 Sarathi 混合 batch |
| **Overlap/CUDA Graph** | L2 一个 trace | 继承机制，C/D 在替身循环有测试 | API 使用不等于已理解设备依赖 | P0 基础，D14 | 标出两条 stream 和 event 的 happens-before；解释为何 CPU prepare 时长不能当 GPU 时长；真实 batch/padding 怎么影响计数 |
| **多 GPU/TP 与 PD 分离** | L1，能用成本模型解释 | 上游通信代码，单卡无独立验证 | 多卡性能调优待验证，8GB 单卡不是其证据 | 原理 P1、实践 P2，D15 | 画 TP 一层主要 collective；估算通信/计算量；区分同进程阶段调度与跨设备 PD 分离 |

### D. 通用工程基础

| 能力点 | 目标水平 | 可确认的已有证据 | 当前判断与不确定性 | 是否补/优先级 | 快速验证 |
| --- | --- | --- | --- | --- | --- |
| **Python 模块/测试/debug** | L2 | dataclass、排序、状态机、pytest 及替身 harness | 代码较多，但独立编码贡献不明 | P0，D16 | 60min 不用助手写稳定候选排序和“按身份移除”，含重复分数/chunk/拒绝用例；解释失败原因而非只修断言 |
| **C++ 生命周期/所有权** | L2 小模块 | 未见本人 C++ 改动；原生比较器为继承代码 | **风险高、待诊断**，不能直接判定是最大短板 | 基础 P0，D17 | 60–90min 写 move-only Buffer 或 RAII 页句柄，说明 copy 为何禁用、move 后状态和异常释放；编译并用 ASan 检查 |
| **数据结构/算法** | L2 常见中等题 | 使用 sort/dict/set，并非独立实现证明 | 面试手写能力待验证 | P0，D18 | 45min 用 list+hash 实现 LRU 并说明迭代器稳定性；另做 heap/区间合并题，分析复杂度及边界 |
| **OS/并发** | L1→L2 小队列 | 接触多进程/异步 IPC，无独立系统实验 | 操作系统/并发语义待验证 | P0 范围内补，D19 | 实现有界生产者消费者队列；解释 condition variable 谓词、spurious wakeup、进程/线程/GIL 区别及 RSS 增长排查 |
| **网络/流式服务** | L2 计量和错误处理 | 当前 API/SSE/client 代码可读，旧计量有缺口 | 可靠 client 与排队边界待验证 | P0 基础，D20 | 对首包为空、终止帧、多事件同 read、连接池等待、超时各设计一个 collector 测试；从发请求前计时 |
| **Git/可复现交付** | L2 | 有 diff、测试与工件；源码已上传 | 分支/环境/忽略文件完整性需独立演示 | P0，D21 | 在新目录还原 H/S，证明 import origin；保存 untracked/JSON；解释为什么只有 git diff 或 uv.lock 不足以复现环境 |

### 5.1 第一周诊断与调整规则

先选 D01、D10/D12、D17、D20 约 3–4 小时完成最小评估；其余随周任务验收，不再开一整周考试。关键编程在无 AI 补全条件下完成一次，再允许查文档和工具 debug；记录第一次耗时、错误、修复和口头解释，而非只保存最终通过截图。

- C++90min 连基本所有权/容器操作都无法完成：前两周 C++ 从 3h 增至 6h/周，等量减少辅助算子/正式框架阅读，不追加总负担；先保核心实验。
- Python 状态机/collector 也不能独立写：先把一小块写懂再扩展工具；暂不做第二个优化。AI 已写出的代码必须能复核。
- 已有另一台电脑完成实验工具：先同步实际 commit、trace 和 run manifest，复核计时/token/错误对账后复用，不重写另一套。当前报告不假定那边已经完成。
- 只有 CPU 数据或目标 GPU 暂不可用：先完成计数器、trace 与边界测试；GPU 结论标未验证。必要时延长对应验收到第 9 周，不用模拟数字替代。

---

## 6. 学习优先级与明确取舍

各时长是 192 小时总预算内的分配/重叠阅读范围，**不应逐行再相加**。资源编号 R1–R12 在第 11 节给出已检查链接。

| 方向/优先级 | 为什么学、对应 JD | 与现项目联系及目标深度 | 建议投入 | 怎么验证/简历价值 |
| --- | --- | --- | --- | --- |
| **P0：Serving Benchmark 与正确性** | J01/J04/J07/J08；没有实测就无法支持性能叙事 | 修正请求起点、token/终态计数、重复与 SLO；能区分 client/server/GPU 区间 | 工具 + 分析约 30–40h，分散 W2–W4/W8；R3/R5 | 两类同 trace5 次配对报告，含失败与反例；直接支撑核心项目 |
| **P0：C++ 基础 + 算法** | 主样本 C++9/11 但含 OR；J04/J05/J11 强，J03 语言任选 | RAII、move、引用/生命周期、vector/list/unordered_map、mutex/条件变量；不先攻模板元编程 | 常规 C++24h+ 算法 24h；弱项按诊断调配；R6 | LRU/页句柄/有界队列、ASan/UBSan、限时手写；主要提升面试通过能力，练习不包装成大型项目 |
| **P0：推理计算与显存模型** | J01/J05/J08/J09 | 从实际 config 推 KV bytes、context/batch 效应；能解释 Prefill/Decode 和共享页 | 6–8h 穿插主线；R8/R9/R10 | 纸笔推导与实际计数对账；解释而非背术语 |
| **P0：GPU 端到端 Profiling** | J01/J04，J05/J08 加分 | nsys 看请求链和等待；有限 ncu 解释一个 kernel，正式压测不带 profiler | 10–14h；R2/R3/R5 | 一份带时间轴/热点证据的定位报告，不把 timeline 重叠百分比当吞吐提升 |
| **P0：Linux/并发/网络的项目范围** | J02/J06/J07；project 本身需要 | 进程/线程、内存/生命周期、IPC、socket/SSE、event loop、背压/timeout | 10–14h，含 collector 及 C++ 队列；R6/R7/R11 | 能定位 client CPU 瓶颈和资源泄漏；能写正确超时/终态测试 |
| **P1：一个正式框架的源码与小改动** | J01/J03/J07/J08/J09 | 只选 SGLang，沿 scheduler→schedule policy→cache；锁 commit，读 2–3 个文件、跑一个小测试 | 12–16h，W6–W7；R8/R12 | 一份 mini→正式框架映射和一个可 review 补丁；PR 合并不作为可控 KPI |
| **P1：CUDA 基础后接 Triton 小算子** | J04/J05/J08/J10；为 Profiling 提供可控对象 | CUDA 索引/warp/coalescing/stream 基础→Triton 归约/RMSNorm；不写完整 Attention/GEMM 库 | 16–24h 上限；R1/R3/R4 | 正确性、不同 shape 计时、误差和瓶颈说明；适合作为主项目附录或辅助项目 |
| **P1：量化、TP/PD 分离原理** | J01/J08/J09；不同岗位深度不同 | 权重/KV 量化区别；TP 通信与 PD 跨设备成本；不在本轮对照中切精度 | 4–6h 概念；R9/R10 及正式框架文档 | 能回答设计题，不宣称生产调优 |
| **P2：多机通信、MoE EP、完整论文复现、编译器深挖** | J09/J10 等特定方向 | 单卡无法验证 NVLink/RDMA 性能；8 周内投入产出较低 | W9–W12 按真实 JD 择一；未选则 0h | 只有岗位确实要求再租多卡或做专项，不为凑技术栈堆项目 |

### 6.1 九个决策的直接答案

1. **mini-sglang 还是正式框架？** 前 4 周优先把现项目从 CPU 语义做到 GPU 证据，第 6 周开始正式 SGLang 同一模块。不能一直停在小框架，也不宜此刻丢掉已有积累重新通读数万行。
2. **是否系统学 CUDA/Triton？** 需要建立 GPU 执行/访存基础，但现在不排一个覆盖所有 CUDA 高级特性的长课程。8 周完成“小而独立”的闭环即可。
3. **CUDA 和 Triton 谁先？** 顺序是约 4–6h CUDA 基础与 vector add，再用 Triton 尽快产出归约/融合实验。以当前 Python Serving 目标，投入不优先倾斜到手写大 GEMM/CUTLASS。
4. **C++ 是否主要短板？** 仓库没有足够本人 C++ 证据，是高风险待验证项；不是已确诊结论。先过 D17/D18。对 Python 调度岗基础 C++ 够用，对 J04/J10 类岗位则远不够。按目标 JD 调整，不一概而论。
5. **OS/网络/并发要不要补？** 要，范围围绕队列、资源生命周期、CPU 进程、SSE、连接池和背压；先不要求完成 xv6 全套或自写高性能 RPC 框架。
6. **分布式何时学？** 现在会画 TP/DP/PP/EP 和 PD 分离边界、估算一次通信；真实多机调优推迟。CPU gloo 演示只能验证 API 语义，不能替代 GPU/NCCL 性能。
7. **要复现论文吗？** 读 SGLang/Radix 作为主论文、PagedAttention 和 Sarathi 选择关键机制做对照即可。可以称“局部机制实验/受论文启发”，不把本项目称完整复现 Sarathi，当前 D 不是其混合 batch 设计。
8. **要第二个项目吗？** 最多一个小算子辅助线；LRU/队列/C++ 习题是基础训练，不再各立一个简历项目。正式 SGLang 阅读/补丁属于主线迁移。
9. **Benchmark/Profiling 到什么程度？** 能独立解释“瓶颈在哪里、证据是什么、对照如何排除混杂、收益与代价是什么、失败如何计入”；仅会跑 nsys 或报 token/s 不够。复杂统计学、SASS 指令级优化不属于所有目标岗门槛。

---

## 7. 8 周详细学习计划

> 每周均为**主实践 16h+C++3h+ 算法 3h+ 求职/表达 2h=24h**；C++ 或算法任务特别计时的部分已在该预算里。所有编程/实验均为未来学习任务，本次没有执行。下列交付文件名为建议，并不表示已经存在。

### W1：10 月 15–21 日，建立个人能力和项目事实基线

- **目标/知识**：能从真实调用链说明四项改动、缓存不变量与证据边界；完成 D01/D10/D17/D20。
- **资源**：[SGLang论文](https://arxiv.org/abs/2312.07104)只读 Radix/Serving 部分；[C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)资源管理；[OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/) 地址空间/线程概念。资源阅读占主实践约 4h。
- **源码**：本项目 `core.py`、`scheduler/{scheduler,prefill,cache,decode}.py`、`kvcache/radix_cache.py`；对照原版对应函数与三份历史 changes.diff，不通读所有文件。
- **编程任务**：独立复写简化候选排序 + 身份移除，写 chunk/同分/拒绝测试；C++ move-only 资源句柄并跑 sanitizer。不要把练习实现接入引擎。
- **实验任务**：复核 249 项 CPU 证据来源与 fake forward；保存目标机环境清单和源码 manifest，若 GPU 就绪仅做最小加载/正确性 smoke，不压测整矩阵。
- **C++/算法**：生命周期/const/引用；哈希、链表和 LRU 思路，4 道代表题（新做 2、隔日复做 2），记录首次时间与错误。
- **投入**：主实践 16h，其中诊断/调用链 8h、阅读 4h、环境/证据 4h；另 3+3+2h。
- **交付**：`project_evidence.md`、`skills_diagnostic.md`、调用链图、环境 manifest、可独立编译的 RAII 练习。
- **验收**：不看笔记 10min 解释 A–D、说出 5 个限制；无辅助完成至少一个小模块；正确推导 112KiB/token；判定 C++ 是否需加时。不是“背出四项名称”就算通过。

### W2：10 月 22–28 日，把 Serving 计量工具做正确

- **目标/知识**：得到能回放固定请求并完整记账的最小 client/server collector；理解计时域、warmup/cache 条件与吞吐分母。
- **资源**：[PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)、[Beej网络指南](https://beej.us/guide/bgnet/html/split/)socket/partial send 部分、既有 [执行规划](../benchmark_execution.md)。
- **源码**：`benchmark/client.py`、`server/api_server.py`、`tokenizer/{server,tokenize,detokenize}.py`、scheduler 结果处理；只在独立实验分支增加必要 hook。
- **编程任务**：冻结实际 tokenized trace；实现请求前起计、token ack/文本分离、finish/timeout 终态、无重试、发送延迟；配套 mock SSE 覆盖空文本、控制帧、多 event 同 read、异常断流。服务端计数/时间戳使用已有 CPU 结果，不新加 GPU 同步。
- **实验任务**：W0 正确性/显存小档；warmup 后验证 server 生成数、client ack 数和请求终态逐项一致；做 collector 开销 pilot。暂不追求全部 5 个拟议脚本都很美观。
- **C++/算法**：`vector/list/unordered_map` 和迭代器失效；独立 LRU 练习；heap、二分各 2 次练习/复做。
- **投入/交付**：16h 主实践重点是工具和对账，另 3+3+2h；交付最小可运行 replay、冻结小 trace、collector 测试、错误台账和使用命令。
- **验收**：每个 offered 请求都有唯一终态；正常例实际 I/O 等于 manifest；能解释 TTFT-text/token-ack/server ready 区别；失败不会被当作 0ms；工具实现若已由另一台完成，则用这些标准审核后复用。

### W3：10 月 29 日–11 月 4 日，完成第一组 GPU 因果对照

- **目标/知识**：证明或否定 bounded 阶段调度的真实尾延迟价值；从 GPU 时间轴解释机制。
- **资源**：[Nsight Systems指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Sarathi-Serve论文](https://arxiv.org/abs/2403.02310) 的 chunk 与 latency tradeoff 部分；不完整复现论文。
- **源码**：`_select_next_batch`、`_record_batch_submission`、`_prepare_batch`、`_forward`、`overlap_loop`、Engine/GraphRunner 涉及的等待点。
- **编程任务**：批量启动/配置 manifest 最小 wrapper；确保 H 与 S 使用同环境与采集边界、记录真实 phase/预算/K、并区分 real/padded size。
- **实验任务**：先 pilot 确定 W4 一个绝对到达率，再冻结；H0000/S1000/S1010 各 5 次，共 15 正式 run。greedy/BF16、相同 trace/资源，保存 ITL、TTFT、TPOT、output 吞吐、failure。另做一个 20–30s 的 warm 诊断 trace，profiling 运行不作为正式性能样本。
- **C++/算法**：smart pointer、move、异常安全；二叉树/BFS/DFS、排序稳定性共 4 次练习/复做。
- **投入/交付**：16h 包含 pilot/验证 5h、自动化 3h、结果/trace 分析 8h；另 3+3+2h；交付 `stage_ablation_report.md`、15run 明细、phase 时间轨迹及一份 nsys 截图/解释。
- **验收**：能同时回答 ITL 收益与长请求 TTFT 代价；保护尝试确实发生；CI/重复不稳则结论为未确定，按预注册追加或延后，不挑最好 run。磁盘保留失败，client lag 达标；无 GPU 则此验收顺延，不拿 CPU 替代。

### W4：11 月 5–11 日，完成公平性实验并开始试投

- **目标/知识**：区分缓存优先与公平机制；把“无饥饿”收敛到可测的有限负载等待证据。
- **资源**：[SGLang论文](https://arxiv.org/abs/2312.07104)、[OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/)CPU 调度/锁与条件变量部分、[Workload设计](../workload_design.md)。
- **源码**：`_get_candidates`、effective `waiting_passes`、chunk 续算、`try_match_and_lock`、Table/KV 拒绝路径和相关测试。
- **编程任务**：hot/cold 标签、首次准入/未准入计数、资源拒绝原因；定义 cold 超 5s 与删失样本；实现有界生产者消费者队列作为 C++ 练习。
- **实验任务**：W6 同长度 hot/cold，S1000/纯 LPM/S1100 各 5 次，共 15 正式 run；比较 cold P95/max/超阈值率与 output 代价。样本不足不写 cold P99。W3 缓存压力吞吐组作为可选追加，不能用 W6 无压力命中推导缓存吞吐收益。
- **C++/算法**：mutex/condition variable 谓词；链表、heap 及单调队列，4 次练习/复做。
- **投入/交付**：主实践 16h，另 3+3+2h；交付 `fairness_report.md`、公平—吞吐图、30 个核心 run 总表、简历 v1。求职 2h 用于 3–5 个广东岗位的资格核验与少量定向试投（未来由本人执行）。
- **验收**：能说明 LPM→Aging 条件收益和资源反例，解释 N 不是秒；能独立演示 mock 数据的失败对账；形成两条不虚构数字的简历表述。C/D/全组合无需此时全部测完。

### W5：11 月 12–18 日，完成唯一辅助线：小型 GPU 算子闭环

- **目标/知识**：CUDA 执行/访存基础，再用 Triton 做一个与 LLM 层相关的 RMSNorm；展示独立计算、验证与性能分析。
- **资源**：[CUDA指南](https://docs.nvidia.com/cuda/cuda-programming-guide/)编程模型/异步章节、[Triton vector add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) 的归约/mask 思路、[ncu指南](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。
- **源码**：本项目 `layers/norm.py`了解调用形状；`kernel/csrc/jit/store.cu` 只读线程/warp copy；现有 kernel 不是你的新增成果。
- **编程任务**：先 CUDA vector add 边界，再 Triton RMSNorm：FP32 累加、epsilon、weight、mask、明确 stride 约束；有 PyTorch FP32 参考和错误输入检查。此周不替换引擎生产算子。
- **实验任务**：rows∈{1,16,256,1024}，width∈{513,1024,4096}，FP32/BF16；先固定合理误差容差再验，不按速度放宽。记录 JIT/warmup、CUDA event 或经过验证的计时、GB/s 估算、不同 block/warp 成本；只对一个形状做 ncu 短诊断。张量与临时缓冲总量控制在 256MiB 以内。
- **C++/算法**：模板基础/类型、编译链接和 sanitizer；数组/区间、滑动窗口，4 次练习/复做。
- **投入/交付**：主实践 16h 约 CUDA 基础 5h、Triton7h、验证/分析 4h；另 3+3+2h；交付 kernel、测试、CSV、短报告；复杂支持扩展最多再用 W7 的 4h，超时就缩 scope。
- **验收**：能独立解释归约、mask、stride、FP32 累加、实际访存；至少一个失败/不快形状有解释。**不要求击败 PyTorch 或成熟库，不把 microbenchmark 变化当模型 E2E 加速。** ncu 权限/硬件计数器不可用时保存原因，用 nsys/event 做较弱结论，深层指标待验证。

### W6：11 月 19–25 日，从 mini 迁移到正式 SGLang

- **目标/知识**：能区分教学框架简化与生产框架约束；只选同一个调度/缓存方向。
- **资源**：[正式scheduler](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)、[schedule_policy](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)、[贡献指南](https://docs.sglang.io/docs/developer_guide/contribution_guide)。先固定真实 commit，记录其路径；main 链接只是定位入口。
- **源码**：上述两个文件和一个 cache 实现（按固定 commit 定位）；各挑 2–3 个函数；不同时读 vLLM V1 全链。
- **编程任务**：写 mini→SGLang 的请求状态、排序、预算、缓存准入映射；复现一个 CPU 可隔离的行为，补一个有意义的小测试/定位脚本。若包导入强依赖 GPU，记录限制，在正式项目隔离纯策略 helper，不伪称全套集成已跑。
- **实验任务**：复跑一个小型 cache/policy 测试；GPU 完整 Serving 安装超出 6h 排障预算就暂停该安装，继续源码与 CPU 范围，不为了装环境临时升级主 Benchmark 环境。
- **C++/算法**：容器所有权/迭代器、复杂度；heap/哈希、拓扑排序，4 次练习/复做。
- **投入/交付**：主实践 16h 中源码 10h、验证/小测试 6h，另 3+3+2h；交付 `mini_to_sglang.md`、固定 SHA、一个最小复现及测试记录。
- **验收**：20min 能解释至少 3 个生产差异；读者能根据 SHA 找到函数；不能只展示克隆仓库或启动成功。

### W7：11 月 26 日–12 月 2 日，完成一个可 review 的工程改进包

- **目标/知识**：选择一个确有依据的小问题完成复现—修复/测试—解释闭环，不同时新增多种优化。
- **资源**：[贡献指南](https://docs.sglang.io/docs/developer_guide/contribution_guide)、[C++资源与并发指南](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)、已有 mini 审计与两个实验报告。
- **源码/任务**：优先延续 W6 的小测试或边界问题；找不到真实 bug 则交付测试/分析包，不编造 issue。另一可选是 mini 的页预算反例，二者只选一项；未来若修 mini 核心，新增 commit、回归并重跑受影响对照，旧数据不能冒充修复版。
- **实验任务**：运行该变更适用的回归；解释测试如何失败在 before、通过在 after。辅助 kernel 尚欠边界或负结果分析时最多补 4h，挤出低价值扩展项。
- **C++/算法**：两次 45min 限时题 +review，ASan/UBSan 结果复盘；算法做树/图和之前错题 4 次。
- **投入/交付**：主实践 16h，另 3+3+2h；交付自包含 diff、复现、测试和 review 说明。可准备真实上游 PR，是否提交由本人决定；社区合并时间不作为验收条件。
- **验收**：能解释每个有效改动、非适用范围和回归风险；无“把参数改小就说修复”的漏洞；至少一次他人/模拟 review 追问通过。

### W8：12 月 3–9 日，固化证据、面试和定向投递

- **目标/知识**：从“做了很多”转成两条可核查成果和一份清楚的工程故事。
- **资源**：[成果指标计划](../resume_metrics_plan.md)、本报告第 10 节 20 问、J12/J08/J11 等岗位当日有效页面；旧招聘状态重新核验。
- **源码/编程**：复核 collector 与两个关键边界测试；修复交付脚本、路径和依赖说明，不临时开新 feature。
- **实验任务**：只在新变更、失败/不稳或证据缺口下补跑；把两组核心结果和算子 microbenchmark 分开汇总，形成 README 图、绝对值/改善率、运行级离散/CI、错误台账和限制。
- **C++/算法**：两轮限时面试题，复盘此前 LRU、queue 与迭代器问题；算法保留错题复做，不临时刷大量新题。
- **投入/交付**：主实践 16h 约结果/复现 8h、两次 45min 项目 mock 及复盘 4h、材料 4h；另 3+3+2h；交付简历 v2、项目 README、结果归档、2 次 mock 记录、广东岗位资格表。
- **验收**：新目录按命令能还原至少一个核心 cell；5min 项目介绍、20min 源码追问、一个 C++ 小题均可完成；面试问题至少 15/20 达到第 10 节深度。若 GPU 仍未完成，只提交诚实的实现/CPU 结果与待测项，不填收益占位符。

---

## 8. 第 9–12 周可选延伸：按实际反馈择项

| 周/日期 | 触发条件与目标 | 任务、硬件和交付 | 上限/取舍 |
| --- | --- | --- | --- |
| **W9，12 月 10–16 日** | 主实验已可信且面试追问预算策略 | 做 E05 固定 64/固定 24/动态各 5 次；验证真实 n 与预算区间；必要时加 target 过大恒 U 负控制 | 主实践≤16h；没有动态净收益也形成报告，不强求第三条 GPU 提升 |
| **W10，12 月 17–23 日** | 缓存吞吐证据不足或目标 JD 重 cache | 追加 W3 压力组，观察真实命中、shared recompute 与吞吐；若要 cold P99，扩大到足够 cold 样本并全组配对重做 | 只选缓存或更深 kernel 方向之一，不能同时开两条 |
| **W11，12 月 24–30 日** | 收到明确通信/分布式岗位反馈 | 先纸笔估算 TP 通信；有确认需求才安排短期双 GPU 试验并预先定预算。无多卡则 CPU 语义演示 + 成本模型，标未验证 NCCL 性能 | 不默认租卡；不模拟 H100 成绩；8GB 无法做的规模明确排除 |
| **W12，12 月 31 日–2027 年 1 月 6 日** | 根据 mock/投递反馈收口 | 选择补 C++、系统排障或修一个已有 PR；复核 1 月招聘与 2 月到岗条件；整理入职前复盘 | 不新增完整框架/论文项目；保持每天 3–4h，不透支到岗承诺 |

2027 年 1 月至 2 月主要用于面试、错题和招聘资格确认，不默认还可同时新增一个大项目。若前 8 周核心未完成，W9–W12 先补核心，表中可选项顺延或删除。

---

## 9. 项目完善、实验与简历竞争力

### 9.1 现项目能否作为核心项目

> **可以作为核心项目，但目前应以“实现与 CPU 验证”描述；真实 GPU 收益还不能填写。** 相比只部署一个 API，这个项目已经涉及候选重排、资源保护、缓存副作用、异步提交和边界测试，具有面试深挖空间。相比已完成生产优化的候选人，仍缺独立的 GPU 效果、瓶颈定位和正式框架协作证据。

最有价值的三项主题，按近期收益排序：

1. C 的阶段调度：长 Prefill 如何影响 Decode 尾部，以及对新请求 TTFT 的代价。
2. B 的缓存排序/有效轮 Aging：复用效率与 cold 请求公平，尤其是资源约束下不能保证服务的边界。
3. A 的准入正确性：为什么匹配不能随意改变 LRU/树、保护成本和失效计划如何维护。优先以工程正确性讲，不预设净加速。

D 公式简单、依赖 C、低 batch 下可能作用小，适合作为条件增量实验；不能靠“自适应”三个字提高创新评价。更有价值的工作是说明哪些负载有效、哪些负载无效，以及静态小预算是否已经足够。

### 9.2 需要补的正确性与工程验收

| 待补项 | 已覆盖与仍缺部分 | 最小新增证据 |
| --- | --- | --- |
| **GPU 数值与长度** | CPU 测试覆盖状态/预算；GPU 模型输出尚缺 | 相同 greedy 输入下 cache 冷/热、chunk 大小、graph 开/关、normal/overlap 的小集；实际 IDs/长度和终态对账，差异先定位，不用“文本差不多”代替 |
| **真实 HTTP 生命周期** | CPU 正常/abort 已有若干测试；真实断连/超时未证实 | 请求断连、长 chunk abort、超长输入、单 token 输出、空文本、EOS 与重复 finish；检查请求终态与 Table/KV 回收 |
| **物理页预算/异常** | 原审计已复现 page>1 紧容量反例；当前 FI 正式主组 page1 | 后续单独复现并建失败测试；如修复逐请求物理预留/回滚，另记 commit 并重测受影响组；不在本次规划偷偷修 |
| **测量正确性** | 旧 client 存在起点/事件计数问题 | mock SSE、server 生成数/ack 数、I/O/失败对账、client lag、collector 开销校准、丢失事件计数 |
| **原版与消融路径** | A 没有 CLI 关闭；B 依赖只读 probe，D 依赖 C | 固定上游 H 和优化 S；实际路径/预算/排序计数；非法单项 LOO 写未测，旧路径 adapter 不验证就不用 |
| **故障/稳态** | 树完整性检查未完整实现、模拟资源断言范围有限 | 区分逻辑 KV 不足、GPU OOM、请求截断、超时和程序异常；保留原始失败 run；不捕获断言后继续沿用未知资源状态 |

“完整实现”只能指已有策略入口和正常路径，不代表所有异常、GPU、生产负载和多卡均完成验证。定位缺口本身有价值，报告应保留它。

### 9.3 前 8 周必做的最小实验包

完整设计仍见 [Benchmark规划](../benchmark_plan.md)、[消融矩阵](../ablation_matrix.md)、[Workload](../workload_design.md)、[执行手册](../benchmark_execution.md)。此处按学习/求职时间**缩小首轮交付范围**，不是将未测 cell 视为完成。

| 包 | 配置与负载 | 正式重复/成本 | 必要指标 | 可以支持的结论 |
| --- | --- | --- | --- | --- |
| **L0 gate** | Qwen3-0.6B BF16，page1/FI，G0→G1；32–64 个正确性/显存请求 | smoke/pilot，不计正式改善 | 实际 token、终态、峰值、余量、kernel/JIT 状态 | 目标机能否运行；不是性能收益 |
| **L1 阶段** | W4：80%128/256、20%2048/32；一个 H pilot 冻结的绝对到达率；H0000/S1000/S1010 | 3×5=15 run；GPU 等待估算每 run 数分钟到十余分钟，以 pilot 为准 | ITL P95/P99、每请求 TPOT、分组 TTFT/E2E、output 吞吐、Goodput、失败、phase/K 轨迹 | 原版净效果及 A 背景下 C 条件收益；两者分别归因 |
| **L2 公平** | W6：hot/cold 均 576/32，hot 共享 512；S1000/纯 LPM/S1100，同 trace 与 prime | 3×5=15 run | cold P95/max/超 5s/未准入率、hot 代价、output、真实缓存长度与重排/fallback 比例 | LPM→Aging 公平取舍；不能自动宣称缓存吞吐已提升 |
| **L3 可选缓存压力** | W3 固定工作集/再访问；同样三策略 | 额外 15 run，W10 可做 | token hit、shared recompute、evict、output/Goodput | 缓存选择是否真的减少重算并改变吞吐 |
| **L4 可选动态** | W5 固定 64、固定 24、dynamic32/floor8 | 额外 15 run，W9 可做 | n_decode→budget/actual chunk、ITL、TTFT、准备成本、output | 小预算效应与动态增量分离 |
| **K 辅助 micro** | RMSNorm 的冻结 shape/dtype 矩阵、PyTorch 参考 | 短计时/5 个独立重复，单一小算子 | 误差、kernel 时长、访存量与 GB/s 估算、资源指标 | 只支持算子层结论，不自动支持模型 Serving 加速 |

首轮必做**30 个正式 GPU run+smoke/pilot**，不同于完整 Benchmark 首轮 70run 建议；优先完成可解释证据，后续再扩展。若另一台已跑了部分矩阵，仅在 code/环境/trace/cache/SLO 等 key 相同且计量通过时复用。任何资源档位改变，都对该比较组共同重做。

维度扫描先放在 pilot：并发 C=1/4/8/16，I/O 选 128/32、512/128、2048/32，逐级验收，不做所有维度乘所有消融的笛卡尔积。正式两组选择能触发机制的冻结负载；C1 无竞争结果保留作负控制。若只正式验证一个到达率，简历结论就只覆盖该点；要声称跨负载稳健，必须追加至少一个较低负载的同条件配对，而不是把 pilot 最好点混进正式统计。

G1 候选为 KV16384 tokens（约 1.75GiB 池，不含额外 dummy）、max running16、graph16、seq2304、normal Prefill256；K4/U64，模型权重及 tokenizer 固定。目标机需先验证 1.5GiB 目标余量/1GiB 最低线，失败则整组统一退到较小档。使用实际请求输入长度；小 chunk 不降低完整未来 KV 预留。所有“推荐参数”均为待验候选，不是 5060Ti 上已经成功运行的结果。

### 9.4 各机制最相关的指标

- A：拒绝 split/touch 副作用、probe 失效与比较次数、CPU decision 开销；GPU 吞吐/TTFT 仅在独立对照后报告。只测排序微基准不等于测 A 的净收益。
- B：真实 cached tokens（首次成功准入统计一次）、shared recompute、cold 首次准入等待/删失比例、hot/cold TTFT 与 output 代价；队列全退回 FCFS 或没有竞争时标机制未激活。
- C：后端/客户端 token-ack ITL 尾部、每请求 TPOT、长 Prefill TTFT 和 E2E、output/Goodput；K 尝试与成功次数分别记录。
- D：预算轨迹、实际 chunk/准备次数、同负载下 ITL/TTFT/吞吐 Pareto；一定有代表性静态小预算对照。

TPOT 是每请求 `(最后token时刻-首token时刻)/(O-1)` 的均值再统计分位数；ITL 是逐 token 间隔分布，二者 P99 不可混用。client 请求前起计，server ready 与纯 GPU 事件分别命名。吞吐分子用真实生成 tokens，图 dummy 和控制帧不算；逻辑 input+output 吞吐与实算工作量分开。更细公式复用既有指标字典。

固定 trace/seed/模型/精度/参数，warmup 和正式测量分离；每个核心 cell5 次配对，报告每 run 绝对值、中位数/IQR/MAD 和 run 级 CI。P99 至少给样本数，cold 不足 1000 时以 P95/max 为主；失败不能静默删除。吞吐改善 `(O/B-1)×100%`，延迟降低`(1-O/B)×100%`，只在同 key 下计算。

### 9.5 主线与第二项目的选择评分

1–5 为任务选择的主观评分，不是录用概率。除“重复程度”越低越好外，其余越高越有利；时间效率高表示更容易在当前预算内完成。

| 选项 | 岗位相关性 | 补充能力 | 时间效率 | 8GB 可行性 | 面试展示 | 与现项目重复程度 | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **A 完善 mini-sglang** | 5 | 4 | 5 | 5 | 5 | 4 | **主线**，但必须补 GPU 证据/正确性，不再堆同类策略 |
| **B 小 CUDA/Triton 算子** | 4 | 5 | 4 | 5 | 4 | 1 | **唯一辅助线**，控制 16–24h；不是再造完整算子库 |
| **C 正式 SGLang 局部阅读/改动** | 5 | 4 | 3 | 3 | 5 | 3 | 纳入 A 的迁移验证，限定 12–16h；不作为另开的大项目 |
| **D 完整复现一篇系统论文** | 4 | 3 | 1 | 1 | 4 | 4 | 推迟；局部机制阅读/对照可留在主线，跨多卡原图不适合当前设备 |

若最想投 J04/J05/J10，后续需增加 C++/CUDA/架构和更强算子证据，并接受时长重分配；现阶段不以“同时覆盖所有岗”作为目标。若广东可投的 Runtime 岗位更合适，则维持 Python 系统主线，Kernel 是补充，不让其挤掉实验与面试。

### 9.6 可写与不可写的简历成果

**现在可写：**“围绕 mini-sglang 实现只读 Radix 探测与条件提交、有效轮 LPM-Aging、有界 Decode 优先及保护性 Prefill 预算；建立 CPU 测试验证排序、引用/回收与阶段计数边界。”249 为套件总测试数，不能全部解释为你独立新增，也不能说完成了 249 项 GPU 测试。

**实测后再写：**“在【GPU/模型/精度】【绝对到达率与混合长度】下，相对【明确基线 commit】将【server/client-ack】P99 ITL 降低【X%】，output 吞吐为基线【Y%】，长请求 TTFT 变化【Z%】，基于【重复数】次配对验证。”Aging 用 cold P95/max/超阈值率和吞吐代价；没有可靠 P99 就不用 P99。所有数字必须能从 raw run 与脚本还原。

**不能写：**独立实现了整个 SGLang/Overlap/PagedAttention；保证无饥饿；2500 次 GPU 实验；动态预算必然提高 GPU 利用率；把全部组合提升累加为单项收益。也不能把“熟悉 vLLM/SGLang”只建立在调用了启动命令上。

---

## 10. 实习面试准备：20 个问题，按当前优先级排列

前 12 题直接围绕你的改动、计量和限制。建议两轮：先 5min 项目叙述，再从表中随机抽题；必须能给代码、反例或实验，不止背定义。每题按“解释清楚/能推演/有证据/能说局限”四项记录，弱项回到对应周，不无限加书单。

| 顺序/优先级 | 问题 | 考察能力 | 回答应达到的深度与证据 |
| --- | --- | --- | --- |
| **1/P0** | 一个 HTTP 请求从进入到首 token 经历了什么？你改了哪些位置？ | 全链路与贡献边界 | 画 API/tokenizer/pending/admission/batch/engine/result 链，指出 A–D 函数及继承部分；能追 uid 和状态 |
| **2/P0** | 原版 baseline 是谁？为什么默认关闭策略仍不是原版？ | 因果实验/代码身份 | 给上游 SHA、优化 SHA、公共观测 patch 和 import origin；解释 A 始终执行、inert bookkeeping；不把 before 分支直接当上游 |
| **3/P0** | 为什么 probe 要只读？一个被拒请求改变 LRU 有什么后果？ | 缓存语义 | 手推未准入 hot probe 造成不真实热度；指出旧 match/split/touch 与新链差异，给拒绝快照测试 |
| **4/P0** | probe 后树 split/evict 或 ref count 变化，你如何确认计划仍可用？ | 生命周期/一致性 | 区分 structure_version 失效重探测和同版本保护成本刷新；commit 不 lock/touch；承认串行前提与无通用回滚 |
| **5/P0** | LPM-Aging 的“等了 N 轮”是什么？何时计数不涨？ | 公平定义 | NEW 成功准入才让仍待准入者加一；chunk-only/全拒绝不加；能在测试中定位，并解释不是墙钟 aging |
| **6/P0** | K 次 Decode 后是否保证 Prefill 运行？ | 调度/资源约束 | K 数实际提交不是选择次数；只保证保护尝试；资源失败不 reset，不能保证首次准入/TTFT 上界 |
| **7/P1** | 动态预算公式为什么减活跃 Decode 数？有哪些失效负载？ | 建模与边际归因 | 给 U/floor/target/n 公式及恒 U/floor 示例；说明 count 不含 context 成本、不是混合 batch；对比固定 24 而非只比 64 |
| **8/P0** | 小 chunk 为什么仍可能页不够？ | 显存与准入 | 说明完整未来 KV 预留与本轮计算量区别；复述多个新请求尾页舍入反例；分清逻辑 allocator 断言和 GPU OOM |
| **9/P0** | 你是否保证无饥饿？如何构造反例？ | 证明意识/边界 | Table/KV 不足、chunk 优先、失败即 break、队列 fallback；用有限 trace 的等待/删失指标作实证，不给无限服务保证 |
| **10/P0** | TTFT、TPOT、ITL 怎么测？SSE 事件能否当 token？ | 流式计量 | 请求前起计；TPOT 每请求均值、ITL 逐间隔；根据本仓库 ack 链对账后分类；空文本/finish/DONE 和传输合并分别处理 |
| **11/P0** | 为什么说你的提升可复现？什么结果会推翻结论？ | 科学实验/取舍 | 相同 trace/资源、5 次配对、cache/warmup、失败台账、client lag、thermal/graph；给无收益条件/吞吐 TTFT 代价；不能挑最佳 run |
| **12/P0** | normal/overlap 中的提交、GPU 完成、结果回收顺序是什么？ | 异步依赖 | 标 stream wait/event 与现有 D2H 同步；说明为什么选择/host 计时不能替代 GPU 执行，为什么计 K 放在 submission |
| **13/P0** | RAII、move 和智能指针如何避免页/句柄生命周期错误？ | 现代 C++ 基础 | 能写 move-only 小类、解释异常/析构、unique/shared 所有权；ref count 不自动等于线程安全；举实际悬挂引用例 |
| **14/P0** | 用 C++ 实现 LRU，哪些操作会让迭代器失效？ | 数据结构/编码 | list+hash 的小实现、O(1) 目标、容量 0/更新/淘汰/迭代器测试；不能只给伪代码 |
| **15/P0** | 有界队列如何避免忙等、丢通知与死锁？ | 并发/OS | 谓词 wait、锁范围、shutdown 和 spurious wakeup；能写测试；解释 Python GIL 和 async IO 分别解决什么 |
| **16/P0** | 为什么客户端可能限制 GPU 测试结果？ | 服务/网络排障 | event loop lag、计划/实际发送、连接池/JSON 解析/CPU 竞争；用独立校准和实际 run 证据定位；开放环不暗改闭环 |
| **17/P0** | GQA 模型 KV 占多少？Prefill/Decode 何时受计算或带宽限制？ | 模型/性能模型 | 实际 config 推 112KiB/token，给 batch/context 与权重/KV 流量估算；不把 token 数直接当 FLOPs |
| **18/P1** | CUDA 中访存合并、block/warp、shared memory 如何影响性能？ | GPU 基础 | 用 vector add/归约代码和连续/跨步对照解释；知 barrier 边界、register/occupancy 代价，不要求背所有 SM 参数 |
| **19/P1** | 你的 Triton 算子为何快/不快？如何验证数值？ | 算子/Profiling | shape、dtype、mask/stride、累加精度、JIT/warmup、真实计时、带宽估算；区分 launch-bound 和 memory-bound，不声称全面胜库 |
| **20/P1 原理/P2 实践** | TP、PP 与 PD 分离是什么？单卡实验能证明什么？ | 分布式设计 | 画一次 TP 通信与 PD KV 传输；解释带宽/延迟/负载均衡代价；明确没有多 GPU 时不能给 NCCL/RDMA 实测结论 |

算法准备以哈希/链表/堆/二分、树图遍历、区间/滑动窗口为主，每周 2 个新题 +2 次复做；记录复杂度、边界和调试。8 周约 32 次限时练习，不以刷题数量代替解释。另每周用 15min 复述一个项目反例，纳入求职/表达 2h 预算。

---

## 11. 推荐资源与阅读范围

以下链接均在本次通过浏览打开；官方框架文档可能跳转，表中优先写已到达的地址。网页版本与目标机安装版本不一定一致（例如 PyTorch 当前文档比审计 torch2.9.1 更新），执行 API/命令以锁定版本及本机 help 核对。**选章节完成任务，不要求通读所有资源。**

| 编号 | 资源及真实链接 | 本轮只读什么/用于何处 |
| --- | --- | --- |
| **R1** | [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | programming model、SIMT、memory 与 asynchronous execution；支持 W5。高级 Tile/TMA/多 GPU 章节不作 8 周必修 |
| **R2** | [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | focused profiling、NVTX、CUDA timeline；学习截取 warm 小区间，解释 CPU/GPU 依赖，W3/W8 |
| **R3** | [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | kernel 级瓶颈、memory/compute 与资源指标；先一个 kernel/少量 section，避免全量采集改变主 benchmark |
| **R4** | [Triton vector add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | program_id、mask、归约和计时；在此基础上独立写 RMSNorm，不把复制教程称自研优化 |
| **R5** | [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | warmup/schedule 与 CPU/CUDA 活动；用于排查而非给正式 Serving 测量额外加同步 |
| **R6** | [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) | R 资源管理、C 类、CP 并发相关条款，配小程序；RAII/move/容器/锁足够开始，不先追模板技巧 |
| **R7** | [OSTEP，作者公开教材](https://pages.cs.wisc.edu/~remzi/OSTEP/) | 地址空间、线程、锁/condition variables、CPU 调度相关章节；不要求整套 OS lab |
| **R8** | [SGLang论文](https://arxiv.org/abs/2312.07104) | Radix 缓存/Serving 设计和实验边界；作为主论文读关键部分，映射 A/B 及继承能力 |
| **R9** | [PagedAttention论文](https://arxiv.org/abs/2309.06180) | 物理块、共享与内存管理概念；比较 mini 的实现/约束，非复现原论文 GPU 结果 |
| **R10** | [Sarathi-Serve论文](https://arxiv.org/abs/2403.02310) | chunk 与吞吐/延迟取舍、与本项目 Prefill-only 动态预算区别；不套用其服务器参数 |
| **R11** | [Beej's Guide to Network Programming](https://beej.us/guide/bgnet/html/split/) | socket 生命周期、partial send/recv、阻塞与 poll；联系 HTTP 流/背压，非从零重写网络库 |
| **R12** | [SGLang贡献指南](https://docs.sglang.io/docs/developer_guide/contribution_guide)、[scheduler源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)、[schedule policy源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py) | W6 固定 commit 后阅读局部；了解测试/贡献边界，社区 CI 权限/资源不由个人完全控制 |

JD 来源集中在第 3 节各条，保留官方/平台/索引/转录差异。不要用搜索结果的“刚刚发布”代替可验证日期，也不要把搜索平台自动生成的岗位解读当招聘方要求。

### 11.1 持续维护的最小材料

未来建议维护一份岗位表：官方 URL/岗位码、城市、招募类型、毕业限制、每周天数、月数、最早到岗、核验日期、来源等级、是否仍有效、本人差距与下一步。每两周在求职 2h 内更新，2027 年 1 月增加频率；现在看到的 HC 不能保证四个月后还在。

项目材料以代码 SHA、运行配置、模型/tokenizer hash、输入 trace、raw results、聚合脚本、失败台账和解释报告为核心。仓库忽略 JSON/JSONL，归档时显式登记，不只看 git status；依赖 lock 也需和实际环境核对，当前 FlashInfer lock 与审计安装版本有差异。若路径或 CI 环境变化，先做小范围复现，再沿用结论。

---

## 12. 如果明天开始，每天只有 3–4 小时：前 7 天安排

从 2026-10-15 开始。前 6 天各 3.5h，第 7 天 3h，总 24h；其中主线 16h、C++3h、算法 3h、求职/表达 2h，和 W1 预算一致。目标 GPU 暂不可用时，环境项标“目标机待验”，先推进 CPU 和纸笔任务。

| 日期 | 具体行动与时间分配 | 当天必须留下的成果/验收 |
| --- | --- | --- |
| **D1，10 月 15 日** | 2h 确认本地/远端 SHA、原版来源、项目文件和一次请求调用链；0.5h C++ 生命周期小例；0.5h 哈希题；0.5h 填写广东/2 月/4 天/3–4 个月及毕业条件 | `baseline_identity.md`、一张调用链草图；能明确指出哪些是继承机制；资格表不把研一默认成 2027 届 |
| **D2，10 月 16 日** | 2.5h 阅读 probe/refresh/commit 和准入测试，纸笔算 Qwen KV；0.5h C++ 引用/const；0.5h 链表题 | 手推 partial match、ref count/evictable 变化，写出 112KiB/token；说清一个拒绝请求不应触发的副作用 |
| **D3，10 月 17 日** | 3h 读阶段 K 计数、动态预算和 Aging；分别画一次失败重试与 chunk-only pass；0.5h 复做 D1 算法题 | 两张事件序列图和 3 个反例：K 不保准入、Aging 不等秒、target 大导致恒 U |
| **D4，10 月 18 日** | 2h 审旧 client 计时/事件汇总，列 mock SSE 用例；1h 完成限时 C++ RAII 诊断；0.5h heap 题 | 找到计时起点问题和控制帧误计；保存 C++ 第一次实现、编译错误及修复；据此决定下周是否加时 |
| **D5，10 月 19 日** | 2h 独立写简化稳定候选排序 + 身份移除测试；0.5h C++ 容器/迭代器；0.5h 复做链表题；0.5h 核验 J12/J08/J11 的资格缺口 | 小测试覆盖 chunk、同分、首候选拒绝；将未知天数/月数写 NA，不能自行补成满足 |
| **D6，10 月 20 日** | 2.5h 检查 fake forward/历史 JSON、已知页舍入反例及目标机环境；0.5h C++ 所有权练习；0.5h 二分题 | `evidence_limits.md`：CPU 证据能/不能说明什么；环境版本表，目标 GPU 未测项明确标注；不跑全矩阵 |
| **D7，10 月 21 日** | 2h 整理一页 A–D 设计/限制和 W2 工具清单；1h 录一次 5min 项目介绍并做追问/资格复盘 | 不看笔记解释调用链和 5 个限制；列下一周 3 个可验任务。若 C++/Python 诊断失败，缩辅助 scope，不新增学习方向 |

> 前 7 天的最高优先级是**确认你能解释和独立验证已有改动，并把后续 GPU 实验的身份、计时和正确性定义固定下来**。完成这些之后，再开始把性能数据写成简历；不需要先把 CUDA、分布式和所有推理框架全部学完。
