# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the mojo_opset project
"""A5 Triton MoE gating: normalization, grouped selection and Top-K."""

from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry

NEG_INF = tl.constexpr(float("-inf"))


@libentry()
@triton.jit
def _moe_gating_top_k_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    idx_ptr,
    norm_ptr,
    stride_x,
    n_rows,
    n_experts,
    k,
    k_group,
    routed_scaling_factor,
    eps,
    K_PAD: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_S: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUT_FLAG: tl.constexpr,
    RENORM: tl.constexpr,
    NORM_TYPE: tl.constexpr,
    GROUP_SELECT_MODE: tl.constexpr,
):
    """One persistent program owns a strided subset of routing rows.

    The routing table is tiled as ``[GROUP_COUNT, GROUP_SIZE]``. Padding lanes
    carry ``-inf`` for selection and are masked from every reduction, so a
    non-power-of-two expert count keeps exact group boundaries. Group scores
    and experts are selected with repeated max plus an index tie-break, which
    matches the stable descending reference without sorting the whole row.
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    group_off = tl.arange(0, BLOCK_G)
    slot_off = tl.arange(0, BLOCK_S)
    expert_off = group_off[:, None] * GROUP_SIZE + slot_off[None, :]
    valid = (group_off[:, None] < GROUP_COUNT) & (slot_off[None, :] < GROUP_SIZE)
    k_off = tl.arange(0, K_PAD)

    for row in range(pid, n_rows, grid_size):
        x = tl.load(
            x_ptr + row.to(tl.int64) * stride_x + expert_off,
            mask=valid,
            other=0.0,
        ).to(tl.float32)

        # Normalization is always FP32, matching the AscendC operator.
        if NORM_TYPE == 0:
            row_max = tl.max(tl.max(tl.where(valid, x, NEG_INF), axis=1), axis=0)
            numerator = tl.where(valid, tl.exp(x - row_max), 0.0)
            denominator = tl.sum(tl.sum(numerator, axis=1), axis=0)
            norm_values = numerator / denominator
        elif NORM_TYPE == 1 or NORM_TYPE == 3:
            norm_values = tl.where(valid, 1.0 / (1.0 + tl.exp(-x)), 0.0)
        else:
            # sqrt(softplus(x)), evaluated without overflow for large |x|.
            softplus = tl.log(1.0 + tl.exp(-tl.abs(x))) + tl.maximum(x, 0.0)
            norm_values = tl.where(valid, tl.sqrt(softplus), 0.0)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + expert_off, mask=valid, other=0.0).to(tl.float32)
            selection_values = norm_values + bias
        else:
            selection_values = norm_values

        if GROUP_COUNT > 1:
            group_values = tl.where(valid, selection_values, NEG_INF)
            if GROUP_SELECT_MODE == 0:
                group_scores = tl.max(group_values, axis=1)
            else:
                first = tl.max(group_values, axis=1)
                first_pos = tl.min(
                    tl.where(group_values == first[:, None], slot_off[None, :], BLOCK_S),
                    axis=1,
                )
                remaining = tl.where(slot_off[None, :] == first_pos[:, None], NEG_INF, group_values)
                second = tl.max(remaining, axis=1)
                group_scores = first + second

            group_selected = group_off < 0
            for _group_step in range(0, k_group):
                best = tl.max(group_scores, axis=0)
                best_group = tl.min(
                    tl.where(group_scores == best, group_off, BLOCK_G),
                    axis=0,
                )
                group_selected = group_selected | (group_off == best_group)
                group_scores = tl.where(group_off == best_group, NEG_INF, group_scores)
            selection_values = tl.where(group_selected[:, None] & valid, selection_values, NEG_INF)
        else:
            selection_values = tl.where(valid, selection_values, NEG_INF)

        gathered = tl.zeros((K_PAD,), tl.float32)
        chosen = tl.zeros((K_PAD,), tl.int32)
        for _expert_step in range(0, k):
            # The lowest expert index wins ties. The selected weight is
            # recomputed from one scalar input load instead of another
            # full-row reduction.
            best = tl.max(tl.max(selection_values, axis=1), axis=0)
            best_expert = tl.min(
                tl.min(tl.where(selection_values == best, expert_off, n_experts), axis=1),
                axis=0,
            )
            best_x = tl.load(x_ptr + row.to(tl.int64) * stride_x + best_expert).to(tl.float32)
            if NORM_TYPE == 0:
                weight = tl.exp(best_x - row_max) / denominator
            elif NORM_TYPE == 1 or NORM_TYPE == 3:
                weight = 1.0 / (1.0 + tl.exp(-best_x))
            else:
                best_softplus = tl.log(1.0 + tl.exp(-tl.abs(best_x))) + tl.maximum(best_x, 0.0)
                weight = tl.sqrt(best_softplus)
            gathered = tl.where(k_off == _expert_step, weight, gathered)
            chosen = tl.where(k_off == _expert_step, best_expert, chosen)
            selection_values = tl.where(expert_off == best_expert, NEG_INF, selection_values)

        if RENORM:
            gathered = gathered / (tl.sum(gathered, axis=0) + eps)
        weight_out = gathered * routed_scaling_factor

        store_mask = k_off < k
        tl.store(y_ptr + row.to(tl.int64) * k + k_off, weight_out.to(y_ptr.dtype.element_ty), mask=store_mask)
        tl.store(idx_ptr + row.to(tl.int64) * k + k_off, chosen, mask=store_mask)
        if OUT_FLAG:
            tl.store(norm_ptr + row.to(tl.int64) * n_experts + expert_off, norm_values, mask=valid)


def moe_gating_top_k_impl(
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
    x = x.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    n_rows, n_experts = x.shape
    y = torch.empty((n_rows, k), dtype=x.dtype, device=x.device)
    expert_idx = torch.empty((n_rows, k), dtype=torch.int32, device=x.device)
    norm_out = torch.empty((n_rows, n_experts), dtype=torch.float32, device=x.device) if out_flag else None
    if n_rows == 0:
        return y, expert_idx, norm_out

    group_size = n_experts // group_count
    block_g = triton.next_power_of_2(group_count)
    block_s = triton.next_power_of_2(group_size)
    k_pad = triton.next_power_of_2(k)
    do_renorm = norm_type in (1, 2) or (norm_type in (0, 3) and renorm == 1)

    grid = (min(get_num_cores("vector"), n_rows),)
    _moe_gating_top_k_kernel[grid](
        x,
        bias if bias is not None else x,
        y,
        expert_idx,
        norm_out if norm_out is not None else x,
        x.stride(0),
        n_rows,
        n_experts,
        k,
        k_group,
        routed_scaling_factor,
        eps,
        K_PAD=k_pad,
        GROUP_COUNT=group_count,
        GROUP_SIZE=group_size,
        BLOCK_G=block_g,
        BLOCK_S=block_s,
        HAS_BIAS=bias is not None,
        OUT_FLAG=out_flag,
        RENORM=do_renorm,
        NORM_TYPE=norm_type,
        GROUP_SELECT_MODE=group_select_mode,
    )
    return y, expert_idx, norm_out


@torch.library.custom_op(
    "mojo_npu_triton_a5::moe_gating_top_k_fwd",
    mutates_args=(),
    schema="(Tensor x, int k, Tensor? bias, int k_group, int group_count, "
    "int group_select_mode, int renorm, int norm_type, bool out_flag, "
    "float routed_scaling_factor, float eps) -> (Tensor, Tensor, Tensor?)",
)
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
    return moe_gating_top_k_impl(
        x,
        k,
        bias,
        k_group,
        group_count,
        group_select_mode,
        renorm,
        norm_type,
        out_flag,
        routed_scaling_factor,
        eps,
    )


@moe_gating_top_k_fwd.register_fake
def _moe_gating_top_k_fwd_fake(
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
    del bias, k_group, group_count, group_select_mode, renorm, norm_type, routed_scaling_factor, eps
    y = torch.empty((x.shape[0], k), dtype=x.dtype, device=x.device)
    expert_idx = torch.empty((x.shape[0], k), dtype=torch.int32, device=x.device)
    norm_out = torch.empty(x.shape, dtype=torch.float32, device=x.device) if out_flag else None
    return y, expert_idx, norm_out
