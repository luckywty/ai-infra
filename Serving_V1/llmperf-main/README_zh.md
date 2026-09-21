# LLMPerf

一个用于评估LLM API性能的工具。

# 安装
```bash
git clone https://github.com/ray-project/llmperf.git
cd llmperf
pip install -e .
```

# 基本用法

我们实现了两种测试来评估LLM：一种是检查性能的负载测试，另一种是检查正确性的正确性测试。

## 负载测试

负载测试会向LLM API发起多个并发请求，测量每个请求的token间延迟和生成吞吐量，以及跨并发请求的指标。每个请求发送的提示格式如下：

```
Randomly stream lines from the following text. Don't generate eos tokens:
LINE 1,
LINE 2,
LINE 3,
...
```

其中的行是从莎士比亚十四行诗集合中随机采样的。无论测试哪个LLM API，token都使用`LlamaTokenizer`进行计数。这是为了确保不同LLM API之间的提示一致。

要运行最基本的负载测试，您可以使用token_benchmark_ray脚本。

### 注意事项和免责声明

- 端点提供商后端可能差异很大，因此这不能反映软件在特定硬件上的运行情况。
- 结果可能因一天中的时间而异。
- 结果可能因负载而异。
- 结果可能与用户的工作负载不相关。

### OpenAI兼容API
```bash
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE="https://api.endpoints.anyscale.com/v1"

python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api openai \
--additional-sampling-params '{}'

```

### Anthropic
```bash
export ANTHROPIC_API_KEY=secret_abcdefg

python token_benchmark_ray.py \
--model "claude-2" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api anthropic \
--additional-sampling-params '{}'

```

### TogetherAI

```bash
export TOGETHERAI_API_KEY="YOUR_TOGETHER_KEY"

python token_benchmark_ray.py \
--model "together_ai/togethercomputer/CodeLlama-7b-Instruct" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### Hugging Face

```bash
export HUGGINGFACE_API_KEY="YOUR_HUGGINGFACE_API_KEY"
export HUGGINGFACE_API_BASE="YOUR_HUGGINGFACE_API_ENDPOINT"

python token_benchmark_ray.py \
--model "huggingface/meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### LiteLLM

LLMPerf可以使用LiteLLM向LLM API发送提示。要查看为提供商设置的环境变量以及为model和additional-sampling-params设置的参数。

请参阅[LiteLLM提供商文档](https://docs.litellm.ai/docs/providers)。

```bash
python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### Vertex AI

这里，--model用于日志记录，而不是选择模型。模型在Vertex AI端点ID中指定。

GCLOUD_ACCESS_TOKEN需要定期设置，因为`gcloud auth print-access-token`生成的令牌在大约15分钟后过期。

Vertex AI不返回其端点生成的总token数，因此使用LLama tokenizer进行计数。

```bash

gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
export GCLOUD_REGION=YOUR_REGION
export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "vertexai" \
--additional-sampling-params '{}'

```

### SageMaker

SageMaker不返回其端点生成的总token数，因此使用LLama tokenizer进行计数。

```bash

export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"s
export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
export AWS_REGION_NAME="YOUR_ENDPOINTS_REGION_NAME"

python llm_correctness.py \
--model "llama-2-7b" \
--llm-api "sagemaker" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

有关参数的更多详细信息，请参阅`python token_benchmark_ray.py --help`。

## 正确性测试

正确性测试会向LLM API发起多个并发请求，格式如下：

```
Convert the following sequence of words into a number: {random_number_in_word_format}. Output just your final answer.
```

其中random_number_in_word_format可以是例如"one hundred and twenty three"。然后测试检查响应中是否包含数字格式的数字，本例中为123。

该测试对多个随机生成的数字执行此操作，并报告包含不匹配的响应数量。

要运行最基本的正确性测试，您可以运行llm_correctness.py脚本。

### OpenAI兼容API

```bash
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE=https://console.endpoints.anyscale.com/m/v1

python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--max-num-completed-requests 150 \
--timeout 600 \
--num-concurrent-requests 10 \
--results-dir "result_outputs"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=secret_abcdefg

python llm_correctness.py \
--model "claude-2" \
--llm-api "anthropic"  \
--max-num-completed-requests 5 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs"
```

### TogetherAI

```bash
export TOGETHERAI_API_KEY="YOUR_TOGETHER_KEY"

python llm_correctness.py \
--model "together_ai/togethercomputer/CodeLlama-7b-Instruct" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### Hugging Face

```bash
export HUGGINGFACE_API_KEY="YOUR_HUGGINGFACE_API_KEY"
export HUGGINGFACE_API_BASE="YOUR_HUGGINGFACE_API_ENDPOINT"

python llm_correctness.py \
--model "huggingface/meta-llama/Llama-2-7b-chat-hf" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### LiteLLM

LLMPerf可以使用LiteLLM向LLM API发送提示。要查看为提供商设置的环境变量以及为model和additional-sampling-params设置的参数。

请参阅[LiteLLM提供商文档](https://docs.litellm.ai/docs/providers)。

```bash
python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

有关参数的更多详细信息，请参阅`python llm_correctness.py --help`。

### Vertex AI

这里，--model用于日志记录，而不是选择模型。模型在Vertex AI端点ID中指定。

GCLOUD_ACCESS_TOKEN需要定期设置，因为`gcloud auth print-access-token`生成的令牌在大约15分钟后过期。

Vertex AI不返回其端点生成的总token数，因此使用LLama tokenizer进行计数。


```bash

gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
export GCLOUD_REGION=YOUR_REGION
export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--llm-api "vertexai" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### SageMaker

SageMaker不返回其端点生成的总token数，因此使用LLama tokenizer进行计数。

```bash

export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"s
export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
export AWS_REGION_NAME="YOUR_ENDPOINTS_REGION_NAME"

python llm_correctness.py \
--model "llama-2-7b" \
--llm-api "sagemaker" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

## 保存结果

负载测试和正确性测试的结果保存在`--results-dir`参数指定的结果目录中。结果保存在2个文件中，一个是测试的摘要指标，另一个是每个返回的单独请求的指标。

# 高级用法

正确性测试是按照以下工作流实现的：

```python
import ray
from transformers import LlamaTokenizerFast

from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher


# 复制环境变量并传递给ray.init()对于使任何客户端工作是必要的
ray.init(runtime_env={"env_vars": {"OPENAI_API_BASE" : "https://api.endpoints.anyscale.com/v1",
                                   "OPENAI_API_KEY" : "YOUR_API_KEY"}})

base_prompt = "hello_world"
tokenizer = LlamaTokenizerFast.from_pretrained(
    "hf-internal-testing/llama-tokenizer"
)
base_prompt_len = len(tokenizer.encode(base_prompt))
prompt = (base_prompt, base_prompt_len)

# 创建一个客户端来发起请求
clients = [OpenAIChatCompletionsClient.remote()]

req_launcher = RequestsLauncher(clients)

req_config = RequestConfig(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt=prompt
    )

req_launcher.launch_requests(req_config)
result = req_launcher.get_next_ready(block=True)
print(result)

```

# 实现新的LLM客户端

要实现新的LLM客户端，您需要实现基类`llmperf.ray_llm_client.LLMClient`并将其装饰为ray actor。

```python

from llmperf.ray_llm_client import LLMClient
import ray


@ray.remote
class CustomLLMClient(LLMClient):

    def llm_request(self, request_config: RequestConfig) -> Tuple[Metrics, str, RequestConfig]:
        """向LLM API发出单个补全请求

        Returns:
            关于请求性能特征的指标。
            请求向LLM API生成的文本。
            用于发出请求的request_config。这主要是出于日志记录目的。

        """
        ...

```

# 旧版代码库
旧的LLMPerf代码库可以在[llmperf-legacy](https://github.com/ray-project/llmval-legacy)仓库中找到。