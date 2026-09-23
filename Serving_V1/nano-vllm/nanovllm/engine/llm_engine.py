import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        """Initialize the LLMEngine.

        Args:
            model: The model to use.
            **kwargs: Additional arguments to pass to the Config.
        """
        config_fields = {field.name for field in fields(Config)}    #取出Config类中所有字段的名称
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs) #从kwargs中取出Config类中所有字段的名称和值，创建Config对象
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")   # 获取一个使用 spawn 方式启动子进程的多进程上下文对象
        for i in range(1, config.tensor_parallel_size):
            # 创建多线程对象
            event = ctx.Event() #Event 是进程间同步工具
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start() #启动子进程
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events) #负责加载模型、管理 GPU、执行前向推理
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id    #把分词器的 EOS（结束符）token id 写进 config

        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        '''
        通知模型子进程退出、
        释放主进程里的模型运行器对象、
        等待所有子进程结束
        '''
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        '''
        推理引擎对外提供的“添加请求”接口
        '''

        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        '''
        推理引擎对外提供的“执行一步”接口

        seqs：这一步要处理的 Sequence 列表；

        is_prefill：布尔值，表示这一步是 prefill 阶段 还是 decode 阶段。
        '''
        seqs, is_prefill = self.scheduler.schedule()

        print("waiting_len =========== ",len(self.scheduler.waiting))
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill) #每个序列新生成的 1 个 token，“run”表示运行模型
        self.scheduler.postprocess(seqs, token_ids, is_prefill) #更新序列状态
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        print("waiting_len =========== ",len(self.scheduler.waiting))
        print("running_len ==========",len(self.scheduler.running))
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        '''
        prompts：多个输入，可以是字符串列表，也可以是 token id 列表的列表。

        sampling_params：可以是一个 SamplingParams（所有 prompt 共用），
        也可以是一个列表（每个 prompt 单独指定）。

        use_tqdm：是否显示进度条。

        返回值类型提示写的是 list[str]，但实际返回的是 list[dict]，
        每个 dict 含 text 和 token_ids。类型提示和实际返回不完全一致。
                
        '''

        # 1. 进度条
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 统一sampling_params格式
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 2. 提交所有请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():       #判断是否所有请求都已完成
            t = perf_counter()      #每轮计时
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })#更新进度条右侧的统计信息。
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
