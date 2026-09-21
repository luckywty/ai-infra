"""
vLLM 推理示例

本脚本演示如何使用 vLLM 库进行大语言模型（LLM）的推理。
vLLM 是一个高性能的 LLM 推理和服务引擎，支持：
- 高吞吐量的离线批量推理
- 实时在线服务
- PagedAttention 等内存优化技术

使用方法：
    python test1.py

依赖：
    - vllm: pip install vllm
    - PyTorch
    - CUDA GPU
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 导入 vLLM 核心模块：
# - LLM: 用于创建和管理大语言模型实例，支持批量推理
# - SamplingParams: 用于配置文本生成的采样参数（如温度、top_p 等）
from vllm import LLM, SamplingParams

# 定义示例提示词列表
# 这些是用于测试模型生成能力的输入文本
# 每个提示词都是一个开放式的句子，让模型续写完成
prompts = [
    "Hello, my name is",           # 自我介绍场景
    "The president of the United States is",  # 知识问答场景
    "The capital of France is",    # 事实性问题场景
    "The future of AI is",         # 开放性话题场景
]

# 创建采样参数对象，控制文本生成行为
# temperature (float): 控制生成文本的随机性
#   - 值越高（如 0.8），生成结果越多样、越有创造性
#   - 值越低（如 0.1），生成结果越确定、越保守
# top_p (float): 核采样（nucleus sampling）参数
#   - 0.95 表示从概率累计达到 95% 的词中采样
#   - 过滤掉概率很低的词，提高生成质量
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    """
    主函数：执行 LLM 推理流程

    功能：
    1. 初始化 LLM 模型
    2. 使用配置的提示词进行批量推理
    3. 格式化输出生成结果
    """
    # 创建 LLM 实例
    # 参数说明：
    #   model (str): 模型名称或路径
    #     - "facebook/opt-125m" 是 Meta 开发的 OPT 系列小型模型
    #     - vLLM 会自动从 Hugging Face Hub 下载模型
    #   gpu_memory_utilization (float): GPU 显存使用比例
    #     - 0.6 表示使用 60% 的 GPU 显存
    #     - 用于避免显存溢出（OOM），可根据实际情况调整
    #   enforce_eager (bool): 是否强制使用 eager 模式执行
    #     - True: 禁用 CUDA Graph，使用 eager 模式
    #     - 优点：更灵活，方便调试
    #     - 缺点：性能略低于 CUDA Graph 模式
    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.6,
       
    )

    # 执行批量推理
    # generate() 方法会：
    #   1. 对所有提示词进行 tokenization（分词）
    #   2. 使用 vLLM 的高效调度器进行批量推理
    #   3. 应用采样参数生成文本
    # 返回值：
    #   - outputs: RequestOutput 对象列表，包含每个请求的结果
    #     - 每个对象包含：原始提示词、生成的文本、token 信息等
    outputs = llm.generate(prompts, sampling_params)

    # 格式化并打印生成结果
    print("\nGenerated Outputs:\n" + "-" * 60)

    # 遍历每个输出结果
    for output in outputs:
        # 提取原始提示词
        prompt = output.prompt
        # 提取生成的文本
        # output.outputs[0].text 获取第一个输出序列的文本
        # （vLLM 默认可能生成多个候选，这里取第一个）
        generated_text = output.outputs[0].text

        # 使用 !r 格式化字符串，显示原始字符串表示（包含引号）
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


# Python 入口点检查
# 当脚本被直接运行时（而非作为模块导入），执行 main() 函数
# 这是 Python 的标准做法，允许脚本同时作为可执行文件和可导入模块
if __name__ == "__main__":
    main()