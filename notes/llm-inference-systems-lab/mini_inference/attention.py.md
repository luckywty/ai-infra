```python
"""Attention implementation placeholder."""

  
  

import torch

from torch import nn

  
  

def build_gqa_head_map(

    num_q_heads: int,

    num_kv_heads: int,

) -> torch.Tensor:

    if  num_kv_heads <= 0 or num_q_heads <= 0 or num_q_heads % num_kv_heads != 0:

        raise ValueError(

            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

        )

    head_map = torch.arange(num_q_heads) // (num_q_heads // num_kv_heads)

    return head_map

  
  

def build_attention_mask(

    query_positions: torch.Tensor,

    key_positions: torch.Tensor,

    query_valid_mask: torch.Tensor,

    key_valid_mask: torch.Tensor,

) -> torch.Tensor:

    # 1. 扩展维度至可广播 Shape：[B, 1, Tq, 1] 与 [B, 1, 1, Tkv]

    q_pos = query_positions.unsqueeze(1).unsqueeze(-1)  # [B, 1, Tq, 1]

    k_pos = key_positions.unsqueeze(1).unsqueeze(2)    # [B, 1, 1, Tkv]

  

    q_valid = query_valid_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, Tq, 1]

    k_valid = key_valid_mask.unsqueeze(1).unsqueeze(2)    # [B, 1, 1, Tkv]

  

    # 2. 计算因果布尔矩阵 (key_position <= query_position)

    causal_mask = k_pos <= q_pos  # [B, 1, Tq, Tkv]

  

    # 3. 广播与计算：同时满足因果律、Query 有效、Key 有效

    mask = causal_mask & q_valid & k_valid  # [B, 1, Tq, Tkv]

  

    return mask

class CausalGQAAttention(nn.Module):

    def __init__(

        self,

        d_model: int,

        num_q_heads: int,

        num_kv_heads: int,

        head_dim: int,

    ):

        super().__init__()

        self.d_model = d_model

        self.num_q_heads = num_q_heads

        self.num_kv_heads = num_kv_heads

        self.head_dim = head_dim

        self.q_proj = nn.Linear(d_model, num_q_heads * head_dim, bias=False)

        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)

        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)

        self.out_proj = nn.Linear(num_q_heads * head_dim, d_model, bias=False)

  
  

    def forward(

        self,

        x: torch.Tensor,

        position_ids: torch.Tensor,

        valid_mask: torch.Tensor,

    ) -> torch.Tensor:

    # 1. Q/K/V Projection

        q_flat = self.q_proj(x)  # [B, T, Hq * Dh] -> [B, T, 8]

        k_flat = self.k_proj(x)  # [B, T, Hkv * Dh] -> [B, T, 4]

        v_flat = self.v_proj(x)  # [B, T, Hkv * Dh] -> [B, T, 4]

  
  

        # 2. Shape 断言

        B, T, _ = x.shape

        assert q_flat.shape == (B, T, self.num_q_heads * self.head_dim), f"q_flat shape mismatch: {q_flat.shape}"

        assert k_flat.shape == (B, T, self.num_kv_heads * self.head_dim), f"k_flat shape mismatch: {k_flat.shape}"

        assert v_flat.shape == (B, T, self.num_kv_heads * self.head_dim), f"v_flat shape mismatch: {v_flat.shape}"

  

            # 2. Reshape + Transpose: [B, T, H * Dh] -> [B, T, H, Dh] -> [B, H, T, Dh]

        q = q_flat.view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)  # [B, Hq, T, Dh]

        k = k_flat.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, Hkv, T, Dh]

        v = v_flat.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, Hkv, T, Dh]

  

        # Reshape + Transpose Shape 断言

        assert q.shape == (B, self.num_q_heads, T, self.head_dim), f"q shape mismatch: {q.shape}"

        assert k.shape == (B, self.num_kv_heads, T, self.head_dim), f"k shape mismatch: {k.shape}"

        assert v.shape == (B, self.num_kv_heads, T, self.head_dim), f"v shape mismatch: {v.shape}"

  

        # 3. 调用 attend 方法计算注意力输出

        attended_values = self.attend(

            q=q,

            k=k,

            v=v,

            query_positions=position_ids,

            key_positions=position_ids,

            query_valid_mask=valid_mask,

            key_valid_mask=valid_mask,

        )  # [B, Hq, T, Dh]

        assert attended_values.shape == (B, self.num_q_heads, T, self.head_dim), f"attended_values shape mismatch: {attended_values.shape}"

        attended_values_flat = attended_values.transpose(1, 2).contiguous().view(B, T, self.num_q_heads * self.head_dim)  # [B, T, Hq * Dh]

        assert attended_values_flat.shape == (B, T, self.num_q_heads * self.head_dim), f"attended_values_flat shape mismatch: {attended_values_flat.shape}"

        output = self.out_proj(attended_values_flat)  # [B, T, D]

        assert output.shape == (B, T, self.d_model), f"output shape mismatch: {output.shape}"

        return output

        # # 暂时留空或抛出 NotImplementedError，等待后续 Checkpoint 逐步实现计算逻辑

        # raise NotImplementedError

  

    def attend(

        self,

        q: torch.Tensor,

        k: torch.Tensor,

        v: torch.Tensor,

        query_positions: torch.Tensor,

        key_positions: torch.Tensor,

        query_valid_mask: torch.Tensor,

        key_valid_mask: torch.Tensor,

    ) -> torch.Tensor:

  

        B, Hq, Tq, Dh = q.shape

        _, Hkv, Tkv, _ = k.shape

  

        # 1. 检查 Hq 是否能被 Hkv 整除

        if Hq % Hkv != 0:

            raise ValueError(f"num_q_heads ({Hq}) must be divisible by num_kv_heads ({Hkv})")

  

        # 2. 获取 head_map 并对齐设备

        head_map = build_gqa_head_map(Hq, Hkv).to(q.device)

  

        # 3. 沿 dim=1 索引展开 K、V

        k_expanded = k[:, head_map, :, :]  # [B, Hq, Tkv, Dh]

        v_expanded = v[:, head_map, :, :]  # [B, Hq, Tkv, Dh]

  

        # 4. Shape 断言

        assert k_expanded.shape == (B, Hq, Tkv, Dh), f"k_expanded shape mismatch: {k_expanded.shape}"

        assert v_expanded.shape == (B, Hq, Tkv, Dh), f"v_expanded shape mismatch: {v_expanded.shape}"

  

        raw_scores = torch.einsum("bhtd,bhsd->bhts", q, k_expanded)  # [B, Hq, Tq, Tkv]

        scaled_scores = raw_scores / (Dh ** 0.5)

  

        assert scaled_scores.shape == (B, Hq, Tq, Tkv), f"scaled_scores shape mismatch: {scaled_scores.shape}"

        assert raw_scores.shape == (B, Hq, Tq, Tkv), f"raw_scores shape mismatch: {raw_scores.shape}"

  
  

        mask = build_attention_mask(

            query_positions=query_positions,

            key_positions=key_positions,

            query_valid_mask=query_valid_mask,

            key_valid_mask=key_valid_mask,

        )  # [B, 1, Tq, Tkv]

        assert mask.shape == (B, 1, Tq, Tkv), f"mask shape mismatch: {mask.shape}"

        masked_scores = scaled_scores.masked_fill(~mask, float("-inf"))

        assert masked_scores.shape == (B, Hq, Tq, Tkv), f"masked_scores shape mismatch: {masked_scores.shape}"

  
  

        probs = torch.softmax(masked_scores, dim=-1)

        probabilities = probs.masked_fill(~mask, 0.0)

        assert probabilities.shape == (B, Hq, Tq, Tkv), f"probabilities shape mismatch: {probabilities.shape}"

        assert probs.shape == (B, Hq, Tq, Tkv), f"probs shape mismatch: {probs.shape}"

        assert torch.isfinite(probabilities).all()

  
  

        contextual_values = torch.einsum("bhts,bhsd->bhtd", probabilities, v_expanded)  # [B, Hq, Tq, Dh]

        assert contextual_values.shape == (B, Hq, Tq, Dh), f"contextual_values shape mismatch: {contextual_values.shape}"

        assert torch.isfinite(contextual_values).all()

        return contextual_values

        # # 5. 暂时继续抛出 NotImplementedError

        # raise NotImplementedError
```
