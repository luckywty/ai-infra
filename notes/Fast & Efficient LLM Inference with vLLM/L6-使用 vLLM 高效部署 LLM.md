---
title: L6 - 使用 vLLM 高效部署 LLM
tags: [vLLM, LLM, inference, 部署]
---

## 一、学习目的

1. 启动一个 serving 真实模型的 vLLM 推理服务器。
2. 用 OpenAI 兼容 API 查询它
3. 用 logprobs 和采样参数探索模型行为
4. 通过实时 Prometheus 指标观察 continuous batching 和 KV Cache 使用
5. 演示前缀缓存

在之前的课程中，你用 llm-compressor 把模型权重从 16-bit 用量化到 4-bit。现在你要用 vLLM 来 serve 这个模型，并通过 OpenAI 兼容 API 与它交互。

[vLLM](https://github.com/vllm-project/vllm) is an open-source LLM inference engine that integrates key serving optimizations:

| Feature | Benefit |
|:--|:--|
| **Continuous batching** | token 级调度：不用等最长的请求，算力不空转 |
| **PagedAttention** | KV Cache 按定长 block 管理：内存浪费接近零 |
| **Prefix caching** | 跨请求复用共享 prompt 前缀的 KV Cache |
| **Quantization support** | 原生 serve GPTQ/AWQ/compressed-tensors 模型 |
| **OpenAI-compatible API** | 已用 OpenAI client 的应用可直接替换 |

## 二、启动 vLLM server

要用 vLLM 部署模型，只需在终端运行这条命令：

```bash
vllm serve Qwen/Qwen3-0.6B --dtype=bfloat16 --max-model-len 4096
```

- **vllm serve**：启动 vLLM 内置推理服务器。它把模型权重加载到引擎中（默认启用 PagedAttention、continuous batching、前缀缓存），并通过 HTTP 在 8000 端口对外提供服务。
- **Qwen/Qwen3-0.6B**：Hugging Face Hub (https://huggingface.co/Qwen/Qwen3-0.6B) 上的模型 ID。首次运行时，vLLM 会从 Hugging Face 把权重、tokenizer 和 config 下载到本地缓存（~/.cache/huggingface/hub），再加载到内存。之后再运行会直接复用缓存文件。
- **--dtype=bfloat16**：以 BF16 精度加载权重。
- **--max-model-len 4096**：把上下文窗口（prompt + 生成）上限设为 4096 tokens。

> [!info]
> 你也可以部署上一课的量化模型，但那需要 GPU，因为它的 W4A16 格式只在 GPU 上有优化的运行时支持。本学习环境是 CPU 运行，所以这里用 vLLM 部署的是原始模型。

vLLM 包了一层 OpenAI 兼容 HTTP API，实现了 SDK 调的同样路由（/v1/models 等），请求/响应形状相同。 (`/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`) with the same request and response shapes.

```python
import warnings

warnings.filterwarnings("ignore")

import time, requests, json, os, math, sys

VLLM_URL = "http://localhost:8000"

os.makedirs("outputs", exist_ok=True)

print("Waiting for vLLM server...")

for attempt in range(60):
    try:
        r = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
        if r.status_code == 200:
            MODEL = r.json()["data"][0]["id"]
            break
    except requests.ConnectionError:
        pass
    time.sleep(5)
    if attempt % 6 == 5:
        print(f"  Still waiting... ({(attempt + 1) * 5}s elapsed)")
else:
    raise RuntimeError(
        "vLLM server not reachable after 5 minutes."
    )

print(f"Connected to {VLLM_URL} — model: {MODEL}")
```

## 三、第一个本地 LLM 请求

vLLM 暴露了与 OpenAI 兼容的 API，因此使用标准的 openai Python 客户端。

```python
from openai import OpenAI

client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="unused")

# 2. base_url=f"{VLLM_URL}/v1"：覆盖默认的 https://api.openai.com/v1，
# 改指你的服务，比如 http://localhost:8000/v1。
# 必须带 /v1，否则路径对不上 vLLM 的 api_router.py。
# - VLLM_URL 是你前面定义的变量（推测），f"..." 只是字符串插值。

# 3. api_key="unused"：OpenAI 类强制要求传 api_key（会填进 Authorization: Bearer ... 头），
# 但本地 vLLM 默认不鉴权，所以填任意占位字符串即可。
# 这是功能正确 vs 性能最优无关的纯接口兼容手段。
```

Qwen3 支持一种 thinking mode（思考模式），会在回答前先生成 chain-of-thought（思维链）推理。我们用 enable_thinking: False 把它关掉，让回复更短、更可预测。

```python
start = time.time()

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user",
               "content": "What is PagedAttention in one sentence?"}],
    max_tokens=80,
    temperature=0.7,
    top_p=0.8,
    extra_body={"top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False}},
)

elapsed = time.time() - start

print(f"Response ({elapsed:.2f}s, {resp.usage.completion_tokens} tokens):")
print(resp.choices[0].message.content)
print(f"\nUsage: {resp.usage.prompt_tokens} prompt + "
      f"{resp.usage.completion_tokens} completion = {resp.usage.total_tokens} total")
```

## 四、模型行为

除了简单聊天，vLLM API 还让你能查看模型内部状态并控制生成。例如，Logprobs： 让你看到模型对生成的每个 token 的确信度，以及它考虑过的备选项。

```python
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "The capital of France is"}],
    max_tokens=15,
    temperature=0.0,       # 保证可复现
    logprobs=True,         # 开关。让 Server 把每步已算好的 log_softmax(logits) 回传，不改变生成，只加一点 CPU 拷贝。
    top_logprobs=5,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

print(f"Response: {resp.choices[0].message.content.strip()}\n")

print("Token-by-token probabilities:\n")

for tok in resp.choices[0].logprobs.content[:8]:
    print(f"  Chosen: '{tok.token}'  (logprob {tok.logprob:.2f})")
    if tok.top_logprobs:
        for alt in tok.top_logprobs[:5]:
            pct = math.exp(alt.logprob) * 100
            bar = "\u2588" * min(20, max(1, int(pct / 5)))
            print(f"    {pct:5.1f}%  {bar}  '{alt.token}'")
    print()
```

## 五、观察 vLLM 底层

vLLM 暴露了一个 Prometheus 兼容的 /metrics 端点（一种很容易被抓取的格式）。要盯的关键指标：

- `num_requests_running` / `waiting`：正在执行 vs 排队等的请求数
- `gpu_cache_usage_perc`（或 `cpu_cache_usage_perc`）：KV Cache 内存压力
- `prompt_tokens_total` / `generation_tokens_total`：累计的 prompt / 生成 token 数

```python
def get_vllm_metrics(base_url=VLLM_URL):
    """Scrape vLLM Prometheus /metrics and return {name: value}."""
    r = requests.get(f"{base_url}/metrics")
    metrics = {}
    for line in r.text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split()[0]
        try:
            metrics[name] = float(line.split()[-1])
        except (ValueError, IndexError):
            continue
    return metrics

metrics = get_vllm_metrics()

print("Current vLLM Metrics:")
for key in ["vllm:num_requests_running", "vllm:num_requests_waiting",
            "vllm:gpu_cache_usage_perc", "vllm:cpu_cache_usage_perc",
            "vllm:prompt_tokens_total", "vllm:generation_tokens_total"]:
    if key in metrics:
        print(f"  {key.replace('vllm:', '')}: {metrics[key]:g}")

with open("outputs/metrics_snapshot.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nFull metrics saved to outputs/metrics_snapshot.json")
```

## 六、连续批处理

vLLM 使用连续批处理（迭代级调度）：当一个请求生成结束，其空位立刻被下一个等待请求填上。 并发发 5 个请求，并在它们运行时观察指标。

```python
import concurrent.futures

prompts = [
    "What is quantization?",
    "Explain KV caching briefly.",
    "What is continuous batching?",
    "Why is LLM inference memory-bound?",
    "What is PagedAttention?",
]

def _ask(prompt):
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60, temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

before = get_vllm_metrics()

print(f"Sending {len(prompts)} concurrent requests...\n")

start = time.time()

with concurrent.futures.ThreadPoolExecutor(
    max_workers=len(prompts)) as pool:
    futures = {pool.submit(_ask, p): p for p in prompts}
    time.sleep(0.5)
    during = get_vllm_metrics()
    running = during.get("vllm:num_requests_running", "--")
    waiting = during.get("vllm:num_requests_waiting", "--")
    print(f"  [mid-flight]  running: {running}  |  waiting: {waiting}")

    for f in concurrent.futures.as_completed(futures):
        resp = f.result()
        print(f"  done: \"{futures[f][:40]}\" -> {resp.usage.completion_tokens} tokens")

elapsed = time.time() - start
after = get_vllm_metrics()
tokens = after.get("vllm:generation_tokens_total", 0) - before.get(
    "vllm:generation_tokens_total", 0)

print(f"\nAll {len(prompts)} completed in {elapsed:.2f}s")
if tokens > 0:
    print(f"Tokens generated: {tokens:g}  |  ~{tokens / elapsed:.1f} tokens/s")
```

> [!warning] 关于首次运行的指标
> 第一次跑这格，你可能看到 `running:0` 或 `1` 而不是 5。再跑一次就会显示 5。指标只在发出请求后一瞬间采样一次，读得太早，请求可能还没到服务端，所以显示 0（既没在跑也没排队），尽管 5 个马上就要一起跑了。重复采样才能看到真正的峰值 5。

vLLM 的调度器收到全部 5 个请求，用连续批处理处理。每步生成为 batch 里每个请求各产 token，做完的立刻让出槽位。

vLLM 的 PagedAttention 是规模化的关键：KV Cache 被切成定长 block，可放在内存任意位置。请求结束时它的 block 立刻可复用——无碎片。

## 七、前缀缓存

很多应用在请求间共享很长的 system prompt。没有前缀缓存时，vLLM 每次都要为共享前缀重算 KV Cache。

开了前缀缓存，vLLM 能识别共享前缀并复用已缓存的 KV。第一个请求付全量 prefill 代价，后续请求跳过它。

我们发 5 个带相同 system prompt 的请求，对比耗时。

```python
SYSTEM_PROMPT = (
    "You are a helpful AI teaching assistant for a course on "
    "LLM optimization. You specialize in explaining concepts like "
    "quantization, inference optimization, and model serving. Keep "
    "answers concise -- one or two sentences."
)

questions = [
    "What is weight quantization?",
    "How does vLLM handle memory?",
    "What is continuous batching?",
    "Why use prefix caching?",
    "What is GPTQ?",
]

before = get_vllm_metrics()
timings = []
tok_counts = []

print("Sending 5 requests with the SAME system prompt...\n")

for i, q in enumerate(questions):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ],
        max_tokens=60, temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    dt = time.time() - t0
    timings.append(dt)
    tok_counts.append(resp.usage.completion_tokens)
    tokens = resp.usage.completion_tokens
    ms_per_tok = (dt / tokens * 1000) if tokens > 0 else 0.0
    print(f"  [{i+1}] {q:<35} {dt:.2f}s  ({tokens} tok, {ms_per_tok:.0f} ms/tok)")

after = get_vllm_metrics()

prefix_before = before.get("vllm:prefix_cache_queries_total", 0)
prefix_after = after.get("vllm:prefix_cache_queries_total", 0)

print(f"\nPrefix cache queries: {prefix_before:g} -> {prefix_after:g}  (+{prefix_after - prefix_before:g})")

cache_keys = [k for k in after if "prefix" in k.lower()
               or "cache_hit" in k.lower()]
for k in sorted(cache_keys):
    b, a = before.get(k, 0), after.get(k, 0)
    if a != b and k != "vllm:prefix_cache_queries_total":
        print(f"  {k}: {b:g} -> {a:g}")

print("\n The increasing prefix_cache_queries count confirms vLLM is ")
print("checking and reusing cached KV blocks for the shared system prompt.")
```

## 八、思考模式

Qwen3 支持思考模式，模型先在 `<think>...</think>` 里生成内部思维链，再给可见答案。答得更好，但用多得多的 token——更多 KV Cache、更多计算、更长耗时。

我们在同一 prompt 上并排流式对比两种模式。

```python
prompt = "What makes continuous batching better than static batching?中文"

for label, thinking, max_tok in [
    ("Thinking OFF", False, 80), ("Thinking ON", True, 2000)]:
    print(f"=== {label} ===\n")
    start = time.time()
    tokens = 0
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tok, temperature=0.7, stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            sys.stdout.write(chunk.choices[0].delta.content)
            sys.stdout.flush()
            tokens += 1
    elapsed = time.time() - start
    print(f"\n  [{tokens} tokens, {elapsed:.2f}s]\n")
```
