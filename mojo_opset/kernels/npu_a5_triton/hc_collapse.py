"""DeepSeek mHC residual collapse: weighted FP32 sum over BF16 hc streams."""

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry

HC_COLLAPSE_BLOCK_H = 1024


@libentry()
@triton.jit
def _hc_collapse_kernel(
    pre_ptr,
    x_ptr,
    out_ptr,
    pre_stride_t,
    pre_stride_m,
    x_stride_t,
    x_stride_m,
    x_stride_h,
    num_tokens,
    hidden_size,
    num_h_blocks,
    HC_MULT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """out[t, h] = sum_m pre[t, m] * x[t, m, h] accumulated in FP32.

    One task collapses one (token, hidden-block) tile. The persistent
    grid-stride loop keeps the logical program count at the vector-core
    count instead of one program per token tile.
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_H)
    total_tasks = num_tokens * num_h_blocks

    for task_id in range(pid, total_tasks, grid_size):
        token_idx = (task_id // num_h_blocks).to(tl.int64)
        block_idx = task_id % num_h_blocks
        offsets = block_idx * BLOCK_H + cols
        col_mask = offsets < hidden_size

        x_row = x_ptr + token_idx * x_stride_t
        pre_row = pre_ptr + token_idx * pre_stride_t

        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for mix_idx in tl.static_range(HC_MULT):
            pre = tl.load(pre_row + mix_idx * pre_stride_m).to(tl.float32)
            x = tl.load(
                x_row + mix_idx * x_stride_m + offsets * x_stride_h,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            acc += pre * x

        out_row = out_ptr + token_idx * hidden_size
        tl.store(out_row + offsets, acc.to(out_ptr.dtype.element_ty), mask=col_mask)


def hc_collapse_impl(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    num_tokens, hc_mult, hidden_size = x.shape
    out = torch.empty(num_tokens, hidden_size, dtype=x.dtype, device=x.device)
    if num_tokens == 0 or hidden_size == 0:
        return out

    block_h = min(HC_COLLAPSE_BLOCK_H, triton.next_power_of_2(hidden_size))
    num_h_blocks = triton.cdiv(hidden_size, block_h)
    total_tasks = num_tokens * num_h_blocks
    grid = (min(get_num_cores("vector"), total_tasks),)

    _hc_collapse_kernel[grid](
        pre_mix,
        x,
        out,
        pre_mix.stride(0),
        pre_mix.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        num_tokens,
        hidden_size,
        num_h_blocks,
        HC_MULT=hc_mult,
        BLOCK_H=block_h,
    )
    return out


@torch.library.custom_op("mojo_npu_triton_a5::hc_collapse_fwd", mutates_args=())
def hc_collapse_fwd(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return hc_collapse_impl(x, pre_mix)


@hc_collapse_fwd.register_fake
def _hc_collapse_fwd_fake(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[0], x.shape[2], dtype=x.dtype, device=x.device)
