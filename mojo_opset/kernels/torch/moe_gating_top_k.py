# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the mojo_opset project
"""Small-operator reference for MoE gating Top-K."""

from typing import Optional
from typing import Tuple

import torch


def _stable_topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    """Descending Top-K with the lowest index winning exact ties."""
    return torch.argsort(values, dim=-1, descending=True, stable=True)[:, :k]


def moe_gating_top_k_fwd(
    x: torch.Tensor,
    k: int,
    bias: Optional[torch.Tensor],
    k_group: int,
    group_count: int,
    group_select_mode: int,
    renorm: int,
    norm_type: int,
    out_flag: bool,
    routed_scaling_factor: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """PyTorch reference for ``mojo_opset.functions.moe_gating_top_k``."""
    dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    if norm_type == 0:
        norm_out = torch.softmax(x_fp32, dim=-1)
    elif norm_type in (1, 3):
        norm_out = torch.sigmoid(x_fp32)
    else:
        norm_out = torch.sqrt(torch.nn.functional.softplus(x_fp32))

    selected_values = norm_out
    if bias is not None:
        selected_values = selected_values + bias.to(torch.float32)

    if group_count > 1:
        num_tokens, num_experts = selected_values.shape
        grouped = selected_values.reshape(num_tokens, group_count, num_experts // group_count)
        if group_select_mode == 0:
            group_scores = grouped.amax(dim=-1)
        else:
            group_scores = torch.topk(grouped, 2, dim=-1, sorted=True).values.sum(dim=-1)
        group_idx = _stable_topk_indices(group_scores, k_group)
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, True)
        selected_values = grouped.masked_fill(~group_mask.unsqueeze(-1), float("-inf")).reshape(num_tokens, num_experts)

    expert_idx = _stable_topk_indices(selected_values, k)
    weights = norm_out.gather(1, expert_idx)
    do_renorm = norm_type in (1, 2) or (norm_type in (0, 3) and renorm == 1)
    if do_renorm:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)
    weights = weights * routed_scaling_factor

    norm_result = norm_out if out_flag else None
    return weights.to(dtype), expert_idx.to(torch.int32), norm_result
