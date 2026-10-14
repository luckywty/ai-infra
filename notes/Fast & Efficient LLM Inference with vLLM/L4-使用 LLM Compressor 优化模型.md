## 一、学习目的

1. 通过全精度模型与压缩模型的对比，学习训练后量化的工作原理
2. 使用 llm-compressor 库应用 GPTQ 方案，生成 W4A16 量化模型
3. 对量化模型与原始模型进行测试和评估对比

---

## 二、 LLM 压缩的概念

[llm-compressor](https://github.com/vllm-project/llm-compressor) 是 vLLM 项目推出的生产级量化工具包。它对已训练好的模型进行单次处理以降低精度，无需重新训练。

其核心 API 是 **oneshot**：你只需提供一个模型、一个校准数据集（用于最小化量化误差的少量真实输入样本），以及一个描述如何量化的 recipe（如 GPTQ、W4A16）。它会生成一个更小的模型，可直接由 [vLLM](https://github.com/vllm-project/vllm)（你将在下一课中使用的 LLM 推理引擎）进行服务。

```python
oneshot(

    model="model-name",           # HuggingFace 模型 ID

    dataset="dataset-name",       # 校准数据集

    recipe=recipe,                # 量化配置

    output_dir="./output",        # 保存路径

    num_calibration_samples=256,  # 校准样本数

    max_seq_length=4096,          # 序列长度

)
```

“oneshot” 这个名字体现了它只需对校准数据进行单次遍历，无需重新训练。

### 2.1 启动

```python
import warnings
warnings.filterwarnings("ignore")
import os, gc, math, pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
MODEL_DIR = "../models/Qwen3-0.6B"
OUTPUT_DIR = "../models/Qwen3-0.6B-W4A16"

print(f"Base model:      {MODEL_DIR}")
print(f"Quantized model: {OUTPUT_DIR}")
```

### 2.2 定义量化方案

一个 **Recipe（量化配方）** 告诉 LLM Compressor 应该如何对模型进行量化。它实际上是一个 **Modifier（修改器）列表**：每个 Modifier 都指定了一种算法以及相应的配置参数，这些算法会按照指定方式应用到模型上。

### 可用的 Modifier

**量化类 Modifier：**

| Modifier               | 描述                                                                 |

| :--------------------- | :----------------------------------------------------------------- |

| `GPTQModifier`         | GPTQ 算法：使用校准数据（calibration data）寻找最优的量化值                           |

| `AWQModifier`          | 激活感知权重量化（Activation-Weighted Quantization）：保留重要程度较高的权重，即对模型影响较大的权重 |

| `AutoRoundModifier`    | Intel 的量化算法，通过可学习的舍入（rounding）和截断（clipping）来优化量化效果                 |

| `QuantizationModifier` | 基础的 PTQ（训练后量化）和 QAT（量化感知训练），适用于简单的量化场景                             |

**Transform Modifier（变换类 Modifier）：**

这些 Modifier 主要用于**提高量化后的模型精度**：

| Modifier              | 描述                                                    |

| :-------------------- | :---------------------------------------------------- |

| `SmoothQuantModifier` | 在量化之前对激活值进行平滑处理，通常与 GPTQ 配合使用                         |

| `QuIPModifier`        | 使用 Hadamard 旋转（Hadamard rotations）来减少异常值（outliers）的影响 |

| `SpinQuantModifier`   | 使用 SpinQuant 风格的旋转，使权重分布更加均匀                          |

多个 Modifier 可以进行**链式组合（chaining）**。

例如，可以先应用 `SmoothQuantModifier`，再应用 `GPTQModifier`，从而改善 **W8A8** 量化的精度。

### Quantization Scheme（量化方案）

`scheme` 参数决定了**权重（W）和激活值（A）的位宽**：

| Scheme  | 权重    | 激活值          | 量化层参数/存储减少 | 对模型质量的影响 |

| :------ | :---- | :----------- | :--------- | :------- |

| `W8A16` | 8-bit | 16-bit（FP16） | ~50%       | 极小       |

| `W4A16` | 4-bit | 16-bit（FP16） | ~75%       | 低～中等     |

| `W8A8`  | 8-bit | 8-bit        | ~50%       | 较低       |

| `W4A8`  | 4-bit | 8-bit        | ~75%       | 中等       |

> **注意：**

> 上述减少比例只针对**被量化的层**。Embedding 和 `lm_head` 层保持全精度，因此整个模型最终能够减少多少，取决于这些层占整个模型参数的比例。

>

> 对于较小的模型（约 0.6B 参数），使用 W4A16 时，预计整个模型的总大小可以减少约 **40%～50%**。

### 我们的量化 Recipe：使用 GPTQModifier 进行 W4A16 量化

| 参数        | 值             | 原因                                   |

| :-------- | :------------ | :----------------------------------- |

| `scheme`  | `W4A16`       | 使用 4-bit 权重                          |

| `targets` | `Linear`      | Linear 层包含模型中的大部分参数，因此可以获得最大的压缩收益    |

| `ignore`  | `["lm_head"]` | 输出层负责将隐藏状态映射到词表（vocabulary），因此保持较高精度 |

```python
from llmcompressor.modifiers.quantization import GPTQModifier
recipe = GPTQModifier(
    scheme="W4A16",
    targets="Linear",
    ignore=["lm_head"],

)

print(f"Recipe: {recipe}")
```

## 三、量化模型

### 为什么需要校准数据集？  

GPTQ 并不是简单地把权重舍入到低精度，而是使用一小部分真实文本来衡量每个权重对模型输出的影响，然后找到能最小化误差的量化值。这正是它比朴素舍入更准确的原因。.

`dataset` 参数指定了用什么文本进行校准。这里你将使用 `WikiText-2` (https://huggingface.co/datasets/mindchain/wikitext2)，一个由维基百科文章组成的标准基准数据集，之后做困惑度评估时也会用它。校准速度很快，只需要几百个样本即可。.

```python
from llmcompressor import oneshot

if not os.path.isdir(OUTPUT_DIR):
    oneshot(
        model="Qwen/Qwen3-0.6B",
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        recipe=recipe,
        output_dir=OUTPUT_DIR,
        max_seq_length=1024,
        num_calibration_samples=128,

    )
    print(f"Quantization complete. Model saved to: {OUTPUT_DIR}")
```

## 四、 比较模型参数

```python
def folder_size(path):
    p = pathlib.Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

def format_size(nbytes):

    if nbytes < 1024**2:
        return f"{nbytes/1024:.1f} KB"
        
    if nbytes < 1024**3:
        return f"{nbytes/1024**2:.1f} MB"

    return f"{nbytes/1024**3:.2f} GB"


size_orig = folder_size(MODEL_DIR)

size_q = folder_size(OUTPUT_DIR)

reduction = (1 - size_q / size_orig) * 100 if size_orig > 0 else 0


print("Model Size Comparison")

print("=" * 45)

print(f"Original (BF16):    {format_size(size_orig)}")

print(f"Quantized (W4A16):  {format_size(size_q)}")

print(f"Reduction:          {reduction:.0f}%")
```

> **注意**：你可能会认为模型大小应该减少 **75%**，因为权重从 16-bit 降到了 4-bit，也就是理论上变成原来的 **1/4（缩小 4 倍）**。但实际上，模型整体大小只减少了约 **42%**。

原因是：只有 **Linear 层的权重**被量化成了 **Int4（4-bit 整数）**。模型的其他部分，包括 **LM Head（语言模型输出头）和归一化层（Normalization Layers）**，仍然保持较高的精度。

## 五、 Perplexity 比较

&emsp; 并排看生成只能给直觉，困惑度 [[Perplexity]] 才是标准指标：它衡量模型预测文本的能力，越低越好。如果量化搞坏了模型，困惑度会明显变高。  

$$\boxed{

PPL=\exp\left(

-\frac{1}{N}

\sum_{i=1}^{N}

\log P(x_i|x_{<i})

\right)

}$$
```python
from datasets import load_dataset


def calculate_perplexity(model, tokenizer, dataset, max_tokens=5000, stride=512):

    encodings = tokenizer(

        "\n\n".join(dataset["text"]),
        return_tensors="pt", truncation=True, max_length=max_tokens,
    )

    input_ids = encodings.input_ids

    nlls, prev_end = [], 0


    for begin_loc in range(0, input_ids.size(1), stride):

        end_loc = min(begin_loc + stride, input_ids.size(1))

        trg_len = end_loc - prev_end

        input_slice = input_ids[:, begin_loc:end_loc]

        target_slice = input_slice.clone()

        target_slice[:, :-trg_len] = -100

        with torch.no_grad():

            loss = model(input_slice, labels=target_slice).loss# 返回平均困惑

            nlls.append(loss * trg_len)

        prev_end = end_loc


    return math.exp(torch.stack(nlls).sum() / prev_end)

test_data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

print(f"Loaded {len(test_data)} test samples")
```
