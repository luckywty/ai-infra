```python
"""Tests for attention implementation placeholder."""

  

import pytest

import torch

  

from attention import build_gqa_head_map

from attention import CausalGQAAttention

  
  
  

def test_gqa_head_map_4_to_2():

    result = build_gqa_head_map(4, 2)

    expected = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    assert torch.equal(result, expected)

  
  

def test_gqa_head_map_8_to_2():

    result = build_gqa_head_map(8, 2)

    expected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)

    assert torch.equal(result, expected)

  
  

def test_mha_head_map():

    result = build_gqa_head_map(4, 4)

    expected = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    assert torch.equal(result, expected)

  
  

def test_mqa_head_map():

    result = build_gqa_head_map(4, 1)

    expected = torch.tensor([0, 0, 0, 0], dtype=torch.int64)

    assert torch.equal(result, expected)

  
  

def test_invalid_non_divisible_heads():

    with pytest.raises(ValueError):

        build_gqa_head_map(5, 2)

  
  

def test_attention_shape_contract():

    # 配置参数

    batch_size = 2

    seq_len = 4

    d_model = 8

    num_q_heads = 4

    num_kv_heads = 2

    head_dim = 2  # d_model == num_q_heads * head_dim == 4 * 2 = 8

  

    # 初始化模型

    attn = CausalGQAAttention(

        d_model=d_model,

        num_q_heads=num_q_heads,

        num_kv_heads=num_kv_heads,

        head_dim=head_dim,

    )

  

    # 构造测试输入 Tensor

    x = torch.randn(batch_size, seq_len, d_model)

    position_ids = torch.tensor([

        [0, 1, 2, 3],

        [7, 8, -1, -1]

    ], dtype=torch.long)

    valid_mask = torch.tensor([

        [1, 1, 1, 1],

        [1, 1, 0, 0]

    ], dtype=torch.bool)

  

    # 前向传播计算

    output = attn(x, position_ids=position_ids, valid_mask=valid_mask)

  

    # 验证输出 Shape 契约：[B, T, d_model] -> [2, 4, 8]

    assert output.shape == (batch_size, seq_len, d_model)

  
  

def test_gqa_groups_use_expected_kv_heads():

        # 配置参数

    batch_size = 2

    seq_len = 4

    d_model = 8

    num_q_heads = 4

    num_kv_heads = 2

    head_dim = 2  # d_model == num_q_heads * head_dim == 4 * 2 = 8

    # 1. 构造 Shape 匹配的 q, k, v (B=1, Hq=4, Hkv=2, T=1, Dh=2)

    # q: [1, 4, 1, 2]

    q = torch.ones(1, 4, 1, 2)

    # k: [1, 2, 1, 2]

    k = torch.ones(1, 2, 1, 2)

    # v: [1, 2, 1, 2]

    # KV Head 0 设置为 [10.0, 10.0]，KV Head 1 设置为 [50.0, 50.0]

    v = torch.zeros(1, 2, 1, 2)

    v[0, 0, 0, :] = 10.0  # KV Head 0

    v[0, 1, 0, :] = 50.0  # KV Head 1

  

    # 构造极简的 position 和 mask 元数据

    query_positions = torch.tensor([[0]], dtype=torch.long)

    key_positions = torch.tensor([[0]], dtype=torch.long)

    query_valid_mask = torch.tensor([[True]], dtype=torch.bool)

    key_valid_mask = torch.tensor([[True]], dtype=torch.bool)

  
  

    attn = CausalGQAAttention(

        d_model=d_model,

        num_q_heads=num_q_heads,

        num_kv_heads=num_kv_heads,

        head_dim=head_dim,

    )

  

    # 2. 调用计划中的 attend(...) 函数

    # 注意：此时 attend() 尚未实现或导入，运行测试抛出 NameError / NotImplementedError 属于预期的 RED 状态

    context = attn.attend(

        q=q,

        k=k,

        v=v,

        query_positions=query_positions,

        key_positions=key_positions,

        query_valid_mask=query_valid_mask,

        key_valid_mask=key_valid_mask,

    )

  

    # 3. 构造期望的输出 Context: [1, 4, 1, 2]

    # Q Head 0 -> [10.0, 10.0]

    # Q Head 1 -> [10.0, 10.0]

    # Q Head 2 -> [50.0, 50.0]

    # Q Head 3 -> [50.0, 50.0]

    expected_context = torch.tensor([

        [

            [[10.0, 10.0]],

            [[10.0, 10.0]],

            [[50.0, 50.0]],

            [[50.0, 50.0]],

        ]

    ])

  

    # 4. 断言验证 GQA 连续分组映射逻辑

    assert torch.allclose(context, expected_context)

  
  
  

from attention import build_attention_mask

  

def test_causal_mask_matrix():

    # 构造输入 Tensor

    query_positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    key_positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    query_valid_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)

    key_valid_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)

  

    # 期望的 4x4 下三角布尔矩阵，扩充至 [B, 1, Tq, Tkv] 即 [1, 1, 4, 4]

    expected_mask = torch.tensor([

        [

            [

                [True, False, False, False],

                [True, True, False, False],

                [True, True, True, False],

                [True, True, True, True],

            ]

        ]

    ], dtype=torch.bool)

  

    # 调用 helper

    mask = build_attention_mask(

        query_positions=query_positions,

        key_positions=key_positions,

        query_valid_mask=query_valid_mask,

        key_valid_mask=key_valid_mask,

    )

  

    # 断言 1: Shape 必须为 [1, 1, 4, 4]

    assert mask.shape == (1, 1, 4, 4)

  

    # 断言 2: dtype 必须为 torch.bool

    assert mask.dtype == torch.bool

  

    # 断言 3: 内容严格等于下三角矩阵

    assert torch.equal(mask, expected_mask)

  
  

def test_invalid_queries_are_zero_and_finite():

    # 参数配置：B=1, Hq=4, Hkv=2, Tq=2, Tkv=2, Dh=2

    batch_size = 1

    num_q_heads = 4

    num_kv_heads = 2

    seq_len = 2

    head_dim = 2

    d_model = num_q_heads * head_dim  # 8

  

    # 初始化 Attention 模块

    attn = CausalGQAAttention(

        d_model=d_model,

        num_q_heads=num_q_heads,

        num_kv_heads=num_kv_heads,

        head_dim=head_dim,

    )

  

    # 构造简单有限值的 q, k, v ( Shape 契约: Q:[B, Hq, Tq, Dh], K/V:[B, Hkv, Tkv, Dh] )

    q = torch.ones(batch_size, num_q_heads, seq_len, head_dim)

    k = torch.ones(batch_size, num_kv_heads, seq_len, head_dim)

    v = torch.ones(batch_size, num_kv_heads, seq_len, head_dim)

  

    # 位置与掩码配置

    query_positions = torch.tensor([[0, 1]], dtype=torch.long)

    key_positions = torch.tensor([[0, 1]], dtype=torch.long)

    query_valid_mask = torch.tensor([[True, False]], dtype=torch.bool)

    key_valid_mask = torch.tensor([[True, True]], dtype=torch.bool)

  

    # 调用 attend 接口

    context = attn.attend(

        q=q,

        k=k,

        v=v,

        query_positions=query_positions,

        key_positions=key_positions,

        query_valid_mask=query_valid_mask,

        key_valid_mask=key_valid_mask,

    )

  

    # 1. 断言 Shape == (1, 4, 2, 2)

    assert context.shape == (batch_size, num_q_heads, seq_len, head_dim)

  

    # 提取 invalid query 切片 (index = 1)

    invalid_slice = context[:, :, 1, :]

  

    # 2. 断言 invalid query slice 全 0

    assert torch.all(invalid_slice == 0.0)

  

    # 3. 断言 invalid query slice 全 finite (无 NaN / Inf)

    assert torch.all(torch.isfinite(invalid_slice))

  
  
  

def test_padding_does_not_affect_valid_outputs():

    # 参数配置：B=1, Hq=4, Hkv=2, Tq=3, Tkv=3, Dh=2

    batch_size = 1

    num_q_heads = 4

    num_kv_heads = 2

    seq_len = 3

    head_dim = 2

    d_model = num_q_heads * head_dim  # 8

  

    # 初始化 Attention 模块

    attn = CausalGQAAttention(

        d_model=d_model,

        num_q_heads=num_q_heads,

        num_kv_heads=num_kv_heads,

        head_dim=head_dim,

    )

  

    # 1. 构造基础 q, k, v ( Shape 契约: Q:[B, Hq, Tq, Dh], K/V:[B, Hkv, Tkv, Dh] )

    q = torch.ones(batch_size, num_q_heads, seq_len, head_dim)

    k = torch.ones(batch_size, num_kv_heads, seq_len, head_dim)

    v = torch.ones(batch_size, num_kv_heads, seq_len, head_dim)

  

    # 位置与掩码配置：index=2 是 padding key/query

    query_positions = torch.tensor([[0, 1, 2]], dtype=torch.long)

    key_positions = torch.tensor([[0, 1, -1]], dtype=torch.long)

    query_valid_mask = torch.tensor([[True, True, False]], dtype=torch.bool)

    key_valid_mask = torch.tensor([[True, True, False]], dtype=torch.bool)

  

    # 2. 第一次运行 attend()

    context_before = attn.attend(

        q=q,

        k=k,

        v=v,

        query_positions=query_positions,

        key_positions=key_positions,

        query_valid_mask=query_valid_mask,

        key_valid_mask=key_valid_mask,

    )

  

    # 3. 构造修改后的 k_changed, v_changed（仅大幅修改 key index = 2 对应的 padding K/V）

    k_changed = k.clone()

    v_changed = v.clone()

    k_changed[:, :, 2, :] = 1000.0  # 修改 padding key 值为极大扰动值

    v_changed[:, :, 2, :] = -999.0  # 修改 padding value 值为极大扰动值

  

    # 4. 第二次运行 attend()

    context_after = attn.attend(

        q=q,

        k=k_changed,

        v=v_changed,

        query_positions=query_positions,

        key_positions=key_positions,

        query_valid_mask=query_valid_mask,

        key_valid_mask=key_valid_mask,

    )

  

    # 5. 断言：两个 valid Query (index 0 和 1) 的 Context 在修改 padding K/V 前后保持严格一致

    assert torch.allclose(context_before[..., :2, :], context_after[..., :2, :])

  
  
  

def test_future_perturbation_does_not_change_past():

    # 参数配置：B=1, T=4, d_model=8, Hq=4, Hkv=2, Dh=2

    batch_size = 1

    seq_len = 4

    num_q_heads = 4

    num_kv_heads = 2

    head_dim = 2

    d_model = num_q_heads * head_dim  # 8

  

    attn = CausalGQAAttention(

        d_model=d_model,

        num_q_heads=num_q_heads,

        num_kv_heads=num_kv_heads,

        head_dim=head_dim,

    )

  

    # 1. 生成基础输入 x 及元数据

    x = torch.randn(batch_size, seq_len, d_model)

    position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    valid_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)

  

    # 2. 第一次前向传播

    output_before = attn(x, position_ids=position_ids, valid_mask=valid_mask)

  

    # 3. clone 一份 x 并大幅修改 position 3 的输入

    x_perturbed = x.clone()

    x_perturbed[:, 3, :] += 100.0

  

    # 4. 第二次前向传播

    output_after = attn(x_perturbed, position_ids=position_ids, valid_mask=valid_mask)

  

    # 5. 比较 position 0, 1, 2 (即 [:, :3, :])，断言未来扰动不影响过去输出

    assert torch.allclose(output_before[:, :3, :], output_after[:, :3, :])

  
  
  

import math

import torch

  
  

def reference_attention(

    attn_module,

    x: torch.Tensor,

    position_ids: torch.Tensor,

    valid_mask: torch.Tensor,

) -> torch.Tensor:

    """独立于生产代码的 Reference Attention 计算逻辑。

  

    严格仅使用 stock PyTorch，不依赖任何生产端 helper。

    """

    B, T, _ = x.shape

    Hq = attn_module.num_q_heads

    Hkv = attn_module.num_kv_heads

    Dh = attn_module.head_dim

  

    # 1. q_proj / k_proj / v_proj

    # 假设 bias 为 False，直接使用 F.linear 或 Module(x)

    q_flat = attn_module.q_proj(x)  # [B, T, Hq * Dh]

    k_flat = attn_module.k_proj(x)  # [B, T, Hkv * Dh]

    v_flat = attn_module.v_proj(x)  # [B, T, Hkv * Dh]

  

    # 2. reshape + transpose

    q = q_flat.view(B, T, Hq, Dh).transpose(1, 2)  # [B, Hq, T, Dh]

    k = k_flat.view(B, T, Hkv, Dh).transpose(1, 2)  # [B, Hkv, T, Dh]

    v = v_flat.view(B, T, Hkv, Dh).transpose(1, 2)  # [B, Hkv, T, Dh]

  

    # 3. repeat_interleave K/V

    repeats = Hq // Hkv

    k_expanded = k.repeat_interleave(repeats, dim=1)  # [B, Hq, T, Dh]

    v_expanded = v.repeat_interleave(repeats, dim=1)  # [B, Hq, T, Dh]

  

    # 4. QK^T

    scores = torch.matmul(q, k_expanded.transpose(-2, -1))  # [B, Hq, T, T]

  

    # 5. / sqrt(Dh)

    scores = scores / math.sqrt(Dh)

  

    # 6. 独立构造 mask [B, 1, Tq, Tkv]

    q_pos = position_ids.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]

    k_pos = position_ids.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]

    q_valid = valid_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]

    k_valid = valid_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]

  

    causal_mask = k_pos <= q_pos

    full_mask = causal_mask & q_valid & k_valid  # [B, 1, T, T]

  

    # 应用 Mask，非法位置设为 -inf

    masked_scores = scores.masked_fill(~full_mask, float("-inf"))

  

    # 7. Softmax (dim=-1)

    probs = torch.softmax(masked_scores, dim=-1)  # [B, Hq, T, T]

  

    # 8. 安全处理 invalid Query (避免全 -inf 产生 NaN)

    # 当整行 Query 无效时，将 probabilities 强行清零

    probs = torch.where(q_valid, probs, torch.zeros_like(probs))

  

    # 9. P @ V

    context = torch.matmul(probs, v_expanded)  # [B, Hq, T, Dh]

  

    # 10. transpose + reshape (Merge Heads)

    context_merged = (

        context.transpose(1, 2).contiguous().view(B, T, Hq * Dh)

    )  # [B, T, Hq * Dh]

  

    # 11. out_proj

    output = attn_module.out_proj(context_merged)  # [B, T, d_model]

    return output

  
  

def test_attention_matches_independent_reference_fp32():

    batch_size = 2

    seq_len = 4

    num_q_heads = 4

    num_kv_heads = 2

    head_dim = 2

    d_model = num_q_heads * head_dim  # 8

  

    # 初始化生产 Attention 实例

    attn = CausalGQAAttention(

        d_model=d_model,

        num_q_heads=num_q_heads,

        num_kv_heads=num_kv_heads,

        head_dim=head_dim,

    )

  

    # 构造测试用输入

    x = torch.randn(batch_size, seq_len, d_model)

    position_ids = torch.tensor(

        [[0, 1, 2, 3], [7, 8, -1, -1]], dtype=torch.long

    )

    valid_mask = torch.tensor(

        [[True, True, True, True], [True, True, False, False]], dtype=torch.bool

    )

  

    # 1. 运行独立的 Reference 计算逻辑

    ref_output = reference_attention(

        attn_module=attn,

        x=x,

        position_ids=position_ids,

        valid_mask=valid_mask,

    )

  

    # 2. 运行生产模块前向传播

    prod_output = attn(

        x=x,

        position_ids=position_ids,

        valid_mask=valid_mask,

    )

  

    # 3. 冻结容差对比（atol=1e-6, rtol=1e-5）

    assert torch.allclose(prod_output, ref_output, atol=1e-6, rtol=1e-5)
```
