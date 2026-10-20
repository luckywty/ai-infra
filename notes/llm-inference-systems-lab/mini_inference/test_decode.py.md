```python
  

from attention import CausalGQAAttention

import pytest

import torch

  

from attention import build_gqa_head_map

# def decode_no_cache(

#     attn,

#     prompt_x,

#     prompt_valid_mask,

#     prompt_position_ids,

#     decode_x,

#     decode_position_ids,

# ):

#     return torch.zeros_like(decode_x)

#     raise NotImplementedError

def decode_no_cache(

    attn: CausalGQAAttention,

    prompt_x: torch.Tensor,

    prompt_valid_mask: torch.Tensor,

    prompt_position_ids: torch.Tensor,

    decode_x: torch.Tensor,

    decode_position_ids: torch.Tensor,

) -> torch.Tensor:

  

    # # 1. 对 prompt_x 前向传播

    # prompt_context = attn(

    #     x=prompt_x,

    #     position_ids=prompt_position_ids,

    #     valid_mask=prompt_valid_mask,

    # )

  

    # 2. 初始化输出列表

    outputs = []

    combined_x = prompt_x

    combined_position_ids = prompt_position_ids

    combined_valid_mask = prompt_valid_mask

    # 3. 对 decode_x 的每个时间步进行循环

    for t in range(decode_x.shape[1]):

        current_decode_x = decode_x[:, t : t + 1, :]  # [B, 1, d_model]

        current_position_ids = decode_position_ids[:, t : t + 1]  # [B, 1]

  

        # 将 prompt context 与当前 decode_x 拼接

        combined_x = torch.cat([combined_x, current_decode_x], dim=1)

        combined_position_ids = torch.cat(

            [combined_position_ids, current_position_ids], dim=1

        )

        combined_valid_mask = torch.cat(

            [combined_valid_mask, torch.ones_like(current_position_ids, dtype=torch.bool)],

            dim=1,

        )

  

        # 前向传播

        output = attn(

            x=combined_x,

            position_ids=combined_position_ids,

            valid_mask=combined_valid_mask,

        )

  

        # 提取当前时间步的输出（最后一个位置）

        outputs.append(output[:, -1:, :])  # [B, 1, d_model]

  

    # 拼接所有时间步的输出

    return torch.cat(outputs, dim=1)  # [B, D, d_model]

  

def test_no_cache_returns_one_output_per_decode_step():

    torch.manual_seed(0)

  

    B = 1

    P = 2

    D = 3

    d_model = 8

  

    attn = CausalGQAAttention(

        d_model=8,

        num_q_heads=4,

        num_kv_heads=2,

        head_dim=2,

    )

  

    prompt_x = torch.randn(B, P, d_model)

    prompt_valid_mask = torch.tensor([[True, True]])

    prompt_position_ids = torch.tensor([[0, 1]])

  

    decode_x = torch.randn(B, D, d_model)

    decode_position_ids = torch.tensor([[2, 3, 4]])

  

    outputs = decode_no_cache(

        attn,

        prompt_x,

        prompt_valid_mask,

        prompt_position_ids,

        decode_x,

        decode_position_ids,

    )

    assert outputs.shape[1] == D

    assert outputs.shape == (B, D, d_model)

  
  

def test_no_cache_output_shape():

    torch.manual_seed(0)

  

    B = 2

    P = 3

    D = 4

    d_model = 8

  

    attn = CausalGQAAttention(

        d_model=d_model,

        num_q_heads=4,

        num_kv_heads=2,

        head_dim=2,

    )

  

    prompt_x = torch.randn(B, P, d_model)

    prompt_valid_mask = torch.tensor([[True, True, True], [True, True, False]])

    prompt_position_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])

  

    decode_x = torch.randn(B, D, d_model)

    decode_position_ids = torch.tensor([[3, 4, 5, 6], [3, 4, 5, 6]])

  

    outputs = decode_no_cache(

        attn,

        prompt_x,

        prompt_valid_mask,

        prompt_position_ids,

        decode_x,

        decode_position_ids,

    )

    assert outputs.shape == (B, D, d_model)

  
  

def test_no_cache_step_matches_direct_full_prefix_forward():

    torch.manual_seed(0)

  

    B = 1

    P = 2

    D = 2

    d_model = 8

  

    attn = CausalGQAAttention(

        d_model=d_model,

        num_q_heads=4,

        num_kv_heads=2,

        head_dim=2,

    )

  

    prompt_x = torch.randn(B, P, d_model)

    prompt_valid_mask = torch.tensor([[True, True]])

    prompt_position_ids = torch.tensor([[0, 1]])

  

    decode_x = torch.randn(B, D, d_model)

    decode_position_ids = torch.tensor([[2, 3]])

  

    outputs = decode_no_cache(

        attn,

        prompt_x,

        prompt_valid_mask,

        prompt_position_ids,

        decode_x,

        decode_position_ids,

    )

  

    # 手工构造 s=0 的完整 prefix

    current_x = torch.cat([prompt_x, decode_x[:, 0:1, :]], dim=1)  # [B, P+1, d_model]

    current_valid_mask = torch.cat([prompt_valid_mask, torch.ones((B, 1), dtype=torch.bool)], dim=1)  # [B, P+1]

    current_position_ids = torch.cat([prompt_position_ids, decode_position_ids[:, 0:1]], dim=1)  # [B, P+1]

  

    direct_output = attn(

        current_x,

        position_ids=current_position_ids,

        valid_mask=current_valid_mask,

    )

  

    expected = direct_output[:, 2, :]

  

    assert torch.allclose(

        outputs[:, 0, :],

        expected,

        atol=1e-6,

        rtol=1e-5,

    )

  
  
  
  

def test_no_cache_uses_full_prefix_each_step():

    torch.manual_seed(0)

    B, Pmax, D, d_model = 1, 2, 3, 8

  

    attn = CausalGQAAttention(

        d_model=d_model, num_q_heads=4, num_kv_heads=2, head_dim=2

    )

    prompt_x = torch.randn(B, Pmax, d_model)

    prompt_valid_mask = torch.tensor([[True, True]])

    prompt_position_ids = torch.tensor([[0, 1]])

  

    decode_x = torch.randn(B, D, d_model)

    decode_position_ids = torch.tensor([[2, 3, 4]])

  

    # 1. 建立 Spy 监听容器

    prefix_lengths = []

    original_forward = attn.forward

  

    # 2. 实现 spy_forward 偷看 x.shape[1]

    def spy_forward(*args, **kwargs):

        x = kwargs.get("x") if "x" in kwargs else args[0]

        prefix_lengths.append(x.shape[1])

        return original_forward(*args, **kwargs)

  

    # 3. 替换掉 forward 方法

    attn.forward = spy_forward

  

    # 4. 运行解码函数

    decode_no_cache(

        attn,

        prompt_x,

        prompt_valid_mask,

        prompt_position_ids,

        decode_x,

        decode_position_ids,

    )

  

    # 5. 断言检查：证明调用了 3 次，且长度依次增长为 3, 4, 5

    assert prefix_lengths == [3, 4, 5]

  
  
  

def test_no_cache_handles_padded_prompt():

    torch.manual_seed(0)

  

    B = 2

    Pmax = 4

    D = 1

    d_model = 8

  

    attn = CausalGQAAttention(

        d_model=d_model,

        num_q_heads=4,

        num_kv_heads=2,

        head_dim=2,

    )

  

    prompt_x = torch.randn(B, Pmax, d_model)

    prompt_valid_mask = torch.tensor([

        [True, True, True, True],

        [True, True, False, False],

    ])

    prompt_position_ids = torch.tensor([

        [0, 1, 2, 3],

        [7, 8, -1, -1],

    ])

  

    decode_x = torch.randn(B, D, d_model)

    decode_position_ids = torch.tensor([

        [4],

        [9],

    ])

  

    # 1. 正常运行一次

    output_before = decode_no_cache(

        attn,

        prompt_x,

        prompt_valid_mask,

        prompt_position_ids,

        decode_x,

        decode_position_ids,

    )

  

    # 2. 只大幅修改第二条序列的两个 padding 位置 (batch=1, physical_index=2,3)

    prompt_x_changed = prompt_x.clone()

    prompt_x_changed[1, 2:4, :] = 1000.0

  

    # 3. 再次运行

    output_after = decode_no_cache(

        attn,

        prompt_x_changed,

        prompt_valid_mask,

        prompt_position_ids,

        decode_x,

        decode_position_ids,

    )

  

    # 4. 验证 padding 位置数值改变不影响 Decode 输出

    assert torch.allclose(

        output_before,

        output_after,

        atol=1e-6,

        rtol=1e-5,

    )
```
