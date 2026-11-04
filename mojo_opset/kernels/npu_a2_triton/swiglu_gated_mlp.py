from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry

"""
SwigluGatedMlp for NPU A2, following the staged ops-transformer reference
(experimental/ffn/swiglu_gated_mlp): ONE kernel entry per direction whose
STAGE selector runs a cube GEMM stage (the reference's
SwigluGatedMlpCubeMatmulStage: MM1 / MM2 / wgrads) or the vector SwiGLU stage
(SwigluGatedMlpSwigluStage); the host sequences the stage launches.

    gate = x @ Wg^T;  up = x @ Wu^T
    gate = gate.clamp(max=swiglu_limit);  up = up.clamp(+-swiglu_limit)   (Glm5NextTextMLP)
    h = silu(gate) * up;  y = h @ Wd^T

glm-5.3 uses hidden 6144 / intermediate 12288 (dense) and 2048 (shared
experts); glm-5.3-flash uses hidden 4096 with swiglu_limit=10.0.

Modifications for NPU architecture by triton-x team, 2025.
"""

STAGE_GEMM = tl.constexpr(0)
STAGE_SWIGLU = tl.constexpr(1)
STAGE_SWIGLU_GRAD = tl.constexpr(2)

# A2: the 192 KiB UB bounds both the dot tiles and the vector-stage tiles
# (situ style); both budgets halve A5's BLOCK_K.
UB_BYTES_A2 = 196608
UB_RESERVE_BYTES = 1024
_BLOCK_BYTES = 32
_GEMM_BLOCK_K = {4: 32, 2: 64}


def _swiglu_stage_tiling(n_rows: int, n_cols: int, esize: int) -> Tuple[int, int]:
    """Rows x cols tiles for the elementwise stages inside one AIV's UB."""
    bytes_per_col = 3 * esize + 6 * 4  # gate/up reads, one write, FP32 scratch
    ub_cols = (UB_BYTES_A2 - UB_RESERVE_BYTES) // bytes_per_col
    align = max(1, _BLOCK_BYTES // min(esize, 4))
    if n_cols <= ub_cols and (n_cols * esize) % _BLOCK_BYTES == 0:
        return max(1, ub_cols // n_cols), n_cols
    block = min(n_cols, ub_cols)
    return 1, max(align, block // align * align)


def _gemm_block_m(rows: int) -> int:
    """Smallest GEMM tile row band >= rows so decode never wastes a 256-row tile."""
    if rows <= 16:
        return 16
    if rows <= 32:
        return 32
    if rows <= 64:
        return 64
    return 128


def _gemm_block_n(rows: int, cols: int) -> int:
    """Narrower column tiles for GEMV-like shapes so tiles cover all cube cores."""
    return 64 if rows <= 32 and cols % 64 == 0 else 128


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def _swiglu_gated_mlp_fwd_kernel(
    a, b, c,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    limit,
    HAS_LIMIT: tl.constexpr,
    ACCUM: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """STAGE_GEMM: c = a @ b (+ c when ACCUM) with arbitrary strides, FP32 accumulate.

    STAGE_SWIGLU: c = silu(clamp(a, max=limit)) * clamp(b, +-limit) elementwise,
    where a is the gate projection [M, N] and b the up projection [M, N].
    """
    if STAGE == STAGE_GEMM:
        start_pid = tl.program_id(axis=0)
        num_cores = tl.num_programs(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)

        for tile in range(start_pid, num_pid_m * num_pid_n, num_cores):
            # Grouped ordering keeps one BLOCK_M row band of a hot in L2.
            group_id = tile // (8 * num_pid_n)
            first_m = group_id * 8
            group_m = tl.minimum(num_pid_m - first_m, 8)
            pid_m = first_m + (tile % group_m)
            pid_n = (tile % (8 * num_pid_n)) // group_m

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            m_mask = offs_m[:, None] < M
            n_mask = offs_n[None, :] < N

            a_ptrs = a + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                k_left = K - k * BLOCK_K
                a_tile = tl.load(a_ptrs, mask=m_mask & (offs_k[None, :] < k_left), other=0.0)
                b_tile = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & n_mask, other=0.0)
                accumulator = tl.dot(a_tile, b_tile, acc=accumulator)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk

            c_ptrs = c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            if ACCUM:
                accumulator += tl.load(c_ptrs, mask=m_mask & n_mask, other=0.0).to(tl.float32)
            tl.store(c_ptrs, accumulator.to(c.dtype.element_ty), mask=m_mask & n_mask)
    else:
        pid = tl.program_id(axis=0)
        grid_size = tl.num_programs(axis=0)
        num_row_tasks = (M + BLOCK_M - 1) // BLOCK_M

        for row_task_id in range(pid, num_row_tasks, grid_size):
            rows_off = row_task_id * BLOCK_M + tl.arange(0, BLOCK_M)
            rows_mask = rows_off < M
            for col_offset in range(0, N, BLOCK_N):
                cols_off = col_offset + tl.arange(0, BLOCK_N)
                block_mask = rows_mask[:, None] & (cols_off < N)[None, :]
                offs = rows_off[:, None] * stride_am + cols_off[None, :]

                gate = tl.load(a + offs, mask=block_mask, other=0.0).to(tl.float32)
                up = tl.load(b + offs, mask=block_mask, other=0.0).to(tl.float32)
                if HAS_LIMIT:
                    gate = tl.minimum(gate, limit)
                    up = tl.minimum(tl.maximum(up, -limit), limit)
                out = gate * tl.sigmoid(gate) * up
                offs_c = rows_off[:, None] * stride_cm + cols_off[None, :]
                tl.store(c + offs_c, out.to(c.dtype.element_ty), mask=block_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def _swiglu_gated_mlp_bwd_kernel(
    a, b, c,
    dh, gate, up, h,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    limit,
    HAS_LIMIT: tl.constexpr,
    ACCUM: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """STAGE_GEMM: as the forward kernel's GEMM stage.

    STAGE_SWIGLU_GRAD: gate/up are the recomputed projections [M, N]; writes
    the [gate | up] halves of the activation gradient into c [M, 2N] and the
    h intermediate into h (for dWd), with inclusive clamp gradient masks.
    """
    if STAGE == STAGE_GEMM:
        start_pid = tl.program_id(axis=0)
        num_cores = tl.num_programs(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)

        for tile in range(start_pid, num_pid_m * num_pid_n, num_cores):
            group_id = tile // (8 * num_pid_n)
            first_m = group_id * 8
            group_m = tl.minimum(num_pid_m - first_m, 8)
            pid_m = first_m + (tile % group_m)
            pid_n = (tile % (8 * num_pid_n)) // group_m

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            m_mask = offs_m[:, None] < M
            n_mask = offs_n[None, :] < N

            a_ptrs = a + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                k_left = K - k * BLOCK_K
                a_tile = tl.load(a_ptrs, mask=m_mask & (offs_k[None, :] < k_left), other=0.0)
                b_tile = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & n_mask, other=0.0)
                accumulator = tl.dot(a_tile, b_tile, acc=accumulator)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk

            c_ptrs = c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            if ACCUM:
                accumulator += tl.load(c_ptrs, mask=m_mask & n_mask, other=0.0).to(tl.float32)
            tl.store(c_ptrs, accumulator.to(c.dtype.element_ty), mask=m_mask & n_mask)
    else:
        pid = tl.program_id(axis=0)
        grid_size = tl.num_programs(axis=0)
        num_row_tasks = (M + BLOCK_M - 1) // BLOCK_M

        for row_task_id in range(pid, num_row_tasks, grid_size):
            rows_off = row_task_id * BLOCK_M + tl.arange(0, BLOCK_M)
            rows_mask = rows_off < M
            for col_offset in range(0, N, BLOCK_N):
                cols_off = col_offset + tl.arange(0, BLOCK_N)
                block_mask = rows_mask[:, None] & (cols_off < N)[None, :]
                offs = rows_off[:, None] * stride_am + cols_off[None, :]

                gate_chunk = tl.load(gate + offs, mask=block_mask, other=0.0).to(tl.float32)
                up_chunk = tl.load(up + offs, mask=block_mask, other=0.0).to(tl.float32)
                dh_chunk = tl.load(dh + offs, mask=block_mask, other=0.0).to(tl.float32)

                gate_mask = 1.0
                up_mask = 1.0
                if HAS_LIMIT:
                    gate_mask = tl.where(gate_chunk <= limit, 1.0, 0.0)
                    up_mask = tl.where((up_chunk >= -limit) & (up_chunk <= limit), 1.0, 0.0)
                    gate_chunk = tl.minimum(gate_chunk, limit)
                    up_chunk = tl.minimum(tl.maximum(up_chunk, -limit), limit)

                sigmoid_gate = tl.sigmoid(gate_chunk)
                silu_gate = gate_chunk * sigmoid_gate
                dsilu_gate = sigmoid_gate * (1 + gate_chunk * (1 - sigmoid_gate))

                dgate_chunk = dh_chunk * up_chunk * dsilu_gate * gate_mask
                dup_chunk = dh_chunk * silu_gate * up_mask
                offs_c = rows_off[:, None] * stride_cm + cols_off[None, :]
                tl.store(c + offs_c, dgate_chunk.to(c.dtype.element_ty), mask=block_mask)
                tl.store(c + offs_c + N, dup_chunk.to(c.dtype.element_ty), mask=block_mask)
                tl.store(h + offs, (silu_gate * up_chunk).to(h.dtype.element_ty), mask=block_mask)


def _check_swiglu_gated_mlp_inputs(x, gate_weight, up_weight, down_weight, swiglu_limit):
    if x.dim() < 1:
        raise ValueError("swiglu_gated_mlp input must have at least one dimension.")
    if gate_weight.dim() != 2 or up_weight.dim() != 2 or down_weight.dim() != 2:
        raise ValueError("swiglu_gated_mlp weights must be 2D.")
    if gate_weight.shape != up_weight.shape or gate_weight.shape[1] != x.shape[-1]:
        raise ValueError("swiglu_gated_mlp gate/up weights must share shape [intermediate, hidden].")
    if down_weight.shape != (x.shape[-1], gate_weight.shape[0]):
        raise ValueError("swiglu_gated_mlp down weight must be [hidden, intermediate].")
    if not (x.dtype == gate_weight.dtype == up_weight.dtype == down_weight.dtype):
        raise ValueError("swiglu_gated_mlp input and weights must share dtype.")
    if not (x.device == gate_weight.device == up_weight.device == down_weight.device):
        raise ValueError("swiglu_gated_mlp input and weights must share device.")
    if swiglu_limit < 0:
        raise ValueError("swiglu_gated_mlp swiglu_limit must be nonnegative.")


def swiglu_gated_mlp_train_fwd_impl(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """y = down_proj(silu(clamp(gate_proj(x))) * clamp(up_proj(x)))."""
    _check_swiglu_gated_mlp_inputs(x, gate_weight, up_weight, down_weight, swiglu_limit)
    ori_shape = x.shape
    hidden_size = ori_shape[-1]
    intermediate_size = gate_weight.shape[0]
    rows = x.numel() // hidden_size
    if rows == 0:
        return x.new_empty(ori_shape)

    x_2d = x.reshape(-1, hidden_size).contiguous()
    gate_weight = gate_weight.contiguous()
    up_weight = up_weight.contiguous()
    down_weight = down_weight.contiguous()
    gemm_cores = get_num_cores("cube")
    block_k = _GEMM_BLOCK_K[x.element_size()]

    # MM1 stage: gate = x @ Wg^T (B tile read from Wg via swapped strides).
    gate = x_2d.new_empty((rows, intermediate_size))
    num_tiles = triton.cdiv(rows, _gemm_block_m(rows)) * triton.cdiv(intermediate_size, _gemm_block_n(rows, intermediate_size))
    _swiglu_gated_mlp_fwd_kernel[(min(gemm_cores, num_tiles),)](
        x_2d, gate_weight.t(), gate, rows, intermediate_size, hidden_size,
        x_2d.stride(0), x_2d.stride(1), gate_weight.stride(1), gate_weight.stride(0),
        gate.stride(0), gate.stride(1),
        0.0, False, False, STAGE_GEMM,
        BLOCK_M=_gemm_block_m(rows), BLOCK_N=_gemm_block_n(rows, intermediate_size), BLOCK_K=block_k,
    )
    # MM1 stage: up = x @ Wu^T.
    up = x_2d.new_empty((rows, intermediate_size))
    _swiglu_gated_mlp_fwd_kernel[(min(gemm_cores, num_tiles),)](
        x_2d, up_weight.t(), up, rows, intermediate_size, hidden_size,
        x_2d.stride(0), x_2d.stride(1), up_weight.stride(1), up_weight.stride(0),
        up.stride(0), up.stride(1),
        0.0, False, False, STAGE_GEMM,
        BLOCK_M=_gemm_block_m(rows), BLOCK_N=_gemm_block_n(rows, intermediate_size), BLOCK_K=block_k,
    )

    # SwiGLU stage with the Glm5NextTextMLP clamp.
    h = x_2d.new_empty((rows, intermediate_size))
    block_m, block_n = _swiglu_stage_tiling(rows, intermediate_size, x.element_size())
    _swiglu_gated_mlp_fwd_kernel[(min(get_num_cores("vector"), triton.cdiv(rows, block_m)),)](
        gate, up, h, rows, intermediate_size, 0,
        gate.stride(0), 0, 0, 0, h.stride(0), 0,
        float(swiglu_limit), swiglu_limit > 0.0, False, STAGE_SWIGLU,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=1,
    )

    # MM2 stage: y = h @ Wd^T.
    out = x_2d.new_empty((rows, hidden_size))
    num_tiles = triton.cdiv(rows, _gemm_block_m(rows)) * triton.cdiv(hidden_size, _gemm_block_n(rows, hidden_size))
    _swiglu_gated_mlp_fwd_kernel[(min(gemm_cores, num_tiles),)](
        h, down_weight.t(), out, rows, hidden_size, intermediate_size,
        h.stride(0), h.stride(1), down_weight.stride(1), down_weight.stride(0),
        out.stride(0), out.stride(1),
        0.0, False, False, STAGE_GEMM,
        BLOCK_M=_gemm_block_m(rows), BLOCK_N=_gemm_block_n(rows, hidden_size), BLOCK_K=block_k,
    )
    return out.reshape(ori_shape)


def swiglu_gated_mlp_train_bwd_impl(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    swiglu_limit: float,
    need_x: bool,
    need_gate: bool,
    need_up: bool,
    need_down: bool,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Analytic gradients for y = down_proj(silu(clamp(gate)) * clamp(up))."""
    _check_swiglu_gated_mlp_inputs(x, gate_weight, up_weight, down_weight, swiglu_limit)
    if grad_output.shape != x.shape:
        raise ValueError(f"swiglu_gated_mlp grad_output must have shape {tuple(x.shape)}, got {grad_output.shape}.")
    if grad_output.dtype != x.dtype or grad_output.device != x.device:
        raise ValueError("swiglu_gated_mlp grad_output must share dtype and device with the input.")

    ori_shape = x.shape
    hidden_size = ori_shape[-1]
    intermediate_size = gate_weight.shape[0]
    rows = x.numel() // hidden_size

    x_2d = x.reshape(-1, hidden_size).contiguous()
    grad_2d = grad_output.reshape(-1, hidden_size).contiguous()
    gate_weight = gate_weight.contiguous()
    up_weight = up_weight.contiguous()
    down_weight = down_weight.contiguous()
    gemm_cores = get_num_cores("cube")
    block_k = _GEMM_BLOCK_K[x.element_size()]
    gemm_grid_rows = triton.cdiv(rows, _gemm_block_m(rows))
    wgrad_block_n = _gemm_block_n(intermediate_size, hidden_size)
    wgrad_tiles = triton.cdiv(intermediate_size, _gemm_block_m(intermediate_size)) * triton.cdiv(hidden_size, wgrad_block_n)
    wgrad_grid = (min(gemm_cores, wgrad_tiles),)

    # Recompute the gate/up projections and dh = dOut @ Wd.
    gate = x_2d.new_empty((rows, intermediate_size))
    _swiglu_gated_mlp_bwd_kernel[(min(gemm_cores, gemm_grid_rows * triton.cdiv(intermediate_size, _gemm_block_n(rows, intermediate_size))),)](
        x_2d, gate_weight.t(), gate, x_2d, x_2d, x_2d, x_2d, rows, intermediate_size, hidden_size,
        x_2d.stride(0), x_2d.stride(1), gate_weight.stride(1), gate_weight.stride(0),
        gate.stride(0), gate.stride(1),
        0.0, False, False, STAGE_GEMM,
        BLOCK_M=_gemm_block_m(rows), BLOCK_N=_gemm_block_n(rows, intermediate_size), BLOCK_K=block_k,
    )
    up = x_2d.new_empty((rows, intermediate_size))
    _swiglu_gated_mlp_bwd_kernel[(min(gemm_cores, gemm_grid_rows * triton.cdiv(intermediate_size, _gemm_block_n(rows, intermediate_size))),)](
        x_2d, up_weight.t(), up, x_2d, x_2d, x_2d, x_2d, rows, intermediate_size, hidden_size,
        x_2d.stride(0), x_2d.stride(1), up_weight.stride(1), up_weight.stride(0),
        up.stride(0), up.stride(1),
        0.0, False, False, STAGE_GEMM,
        BLOCK_M=_gemm_block_m(rows), BLOCK_N=_gemm_block_n(rows, intermediate_size), BLOCK_K=block_k,
    )
    dh = x_2d.new_empty((rows, intermediate_size))
    _swiglu_gated_mlp_bwd_kernel[(min(gemm_cores, gemm_grid_rows * triton.cdiv(intermediate_size, _gemm_block_n(rows, intermediate_size))),)](
        grad_2d, down_weight, dh, grad_2d, grad_2d, grad_2d, grad_2d, rows, intermediate_size, hidden_size,
        grad_2d.stride(0), grad_2d.stride(1), down_weight.stride(0), down_weight.stride(1),
        dh.stride(0), dh.stride(1),
        0.0, False, False, STAGE_GEMM,
        BLOCK_M=_gemm_block_m(rows), BLOCK_N=_gemm_block_n(rows, intermediate_size), BLOCK_K=block_k,
    )

    # SwiGLU grad stage: d(gate | up) [rows, 2I] plus h for dWd.
    d_gate_up = x_2d.new_empty((rows, 2 * intermediate_size))
    h = x_2d.new_empty((rows, intermediate_size))
    block_m, block_n = _swiglu_stage_tiling(rows, intermediate_size, x.element_size())
    _swiglu_gated_mlp_bwd_kernel[(min(get_num_cores("vector"), triton.cdiv(rows, block_m)),)](
        gate, up, d_gate_up, dh, gate, up, h,
        rows, intermediate_size, 0,
        gate.stride(0), 0, 0, 0, d_gate_up.stride(0), 0,
        float(swiglu_limit), swiglu_limit > 0.0, False, STAGE_SWIGLU_GRAD,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=1,
    )
    d_gate = d_gate_up[:, :intermediate_size].contiguous()
    d_up = d_gate_up[:, intermediate_size:].contiguous()

    # dX = dGate @ Wg + dUp @ Wu; the second GEMM stage accumulates in place.
    grad_x = None
    if need_x:
        grad_x = x_2d.new_empty((rows, hidden_size))
        dx_grid = (min(gemm_cores, gemm_grid_rows * triton.cdiv(hidden_size, _gemm_block_n(rows, hidden_size))),)
        _swiglu_gated_mlp_bwd_kernel[dx_grid](
            d_gate, gate_weight, grad_x, d_gate, d_gate, d_gate, d_gate, rows, hidden_size, intermediate_size,
            d_gate.stride(0), d_gate.stride(1), gate_weight.stride(0), gate_weight.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            0.0, False, False, STAGE_GEMM,
            BLOCK_M=_gemm_block_m(rows), BLOCK_N=_gemm_block_n(rows, hidden_size), BLOCK_K=block_k,
        )
        _swiglu_gated_mlp_bwd_kernel[dx_grid](
            d_up, up_weight, grad_x, d_up, d_up, d_up, d_up, rows, hidden_size, intermediate_size,
            d_up.stride(0), d_up.stride(1), up_weight.stride(0), up_weight.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            0.0, False, True, STAGE_GEMM,
            BLOCK_M=_gemm_block_m(rows), BLOCK_N=_gemm_block_n(rows, hidden_size), BLOCK_K=block_k,
        )
        grad_x = grad_x.reshape(ori_shape)

    # dW stages: dAct^T @ x and dOut^T @ h via swapped A strides.
    dgate_weight = None
    if need_gate:
        dgate_weight = gate_weight.new_empty(gate_weight.shape)
        _swiglu_gated_mlp_bwd_kernel[wgrad_grid](
            d_gate, x_2d, dgate_weight, d_gate, d_gate, d_gate, d_gate, intermediate_size, hidden_size, rows,
            d_gate.stride(1), d_gate.stride(0), x_2d.stride(0), x_2d.stride(1),
            dgate_weight.stride(0), dgate_weight.stride(1),
            0.0, False, False, STAGE_GEMM,
            BLOCK_M=_gemm_block_m(intermediate_size), BLOCK_N=wgrad_block_n, BLOCK_K=block_k,
        )
    dup_weight = None
    if need_up:
        dup_weight = up_weight.new_empty(up_weight.shape)
        _swiglu_gated_mlp_bwd_kernel[wgrad_grid](
            d_up, x_2d, dup_weight, d_up, d_up, d_up, d_up, intermediate_size, hidden_size, rows,
            d_up.stride(1), d_up.stride(0), x_2d.stride(0), x_2d.stride(1),
            dup_weight.stride(0), dup_weight.stride(1),
            0.0, False, False, STAGE_GEMM,
            BLOCK_M=_gemm_block_m(intermediate_size), BLOCK_N=wgrad_block_n, BLOCK_K=block_k,
        )
    ddown_weight = None
    if need_down:
        ddown_weight = down_weight.new_empty(down_weight.shape)
        ddown_block_n = _gemm_block_n(hidden_size, intermediate_size)
        ddown_grid = (min(gemm_cores, triton.cdiv(hidden_size, _gemm_block_m(hidden_size)) * triton.cdiv(intermediate_size, ddown_block_n)),)
        _swiglu_gated_mlp_bwd_kernel[ddown_grid](
            grad_2d, h, ddown_weight, grad_2d, grad_2d, grad_2d, grad_2d, hidden_size, intermediate_size, rows,
            grad_2d.stride(1), grad_2d.stride(0), h.stride(0), h.stride(1),
            ddown_weight.stride(0), ddown_weight.stride(1),
            0.0, False, False, STAGE_GEMM,
            BLOCK_M=_gemm_block_m(hidden_size), BLOCK_N=ddown_block_n, BLOCK_K=block_k,
        )

    return grad_x, dgate_weight, dup_weight, ddown_weight


@torch.library.custom_op("mojo_npu_triton_a2::swiglu_gated_mlp_fwd", mutates_args=())
def swiglu_gated_mlp_fwd(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    return swiglu_gated_mlp_train_fwd_impl(x, gate_weight, up_weight, down_weight, swiglu_limit)


@swiglu_gated_mlp_fwd.register_fake
def _swiglu_gated_mlp_fwd_fake(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    del gate_weight, up_weight, down_weight, swiglu_limit
    return torch.empty(x.shape, dtype=x.dtype, device=x.device, memory_format=torch.contiguous_format)


@torch.library.custom_op("mojo_npu_triton_a2::swiglu_gated_mlp_bwd", mutates_args=())
def swiglu_gated_mlp_bwd(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    swiglu_limit: float,
    need_x: bool,
    need_gate: bool,
    need_up: bool,
    need_down: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Registered as a custom op so AOT backward tracing treats it as opaque
    # (see situ_and_mul / swiglu). Skipped gradients come back as empty
    # placeholders; the autograd Function maps them to None.
    grads = swiglu_gated_mlp_train_bwd_impl(
        grad_output, x, gate_weight, up_weight, down_weight, swiglu_limit,
        need_x, need_gate, need_up, need_down,
    )
    placeholders = (x, gate_weight, up_weight, down_weight)
    return tuple(
        grad if grad is not None else torch.empty(0, dtype=like.dtype, device=like.device)
        for grad, like in zip(grads, placeholders)
    )


@swiglu_gated_mlp_bwd.register_fake
def _swiglu_gated_mlp_bwd_fake(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    swiglu_limit: float,
    need_x: bool,
    need_gate: bool,
    need_up: bool,
    need_down: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del grad_output, swiglu_limit, need_x, need_gate, need_up, need_down
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(gate_weight, memory_format=torch.contiguous_format),
        torch.empty_like(up_weight, memory_format=torch.contiguous_format),
        torch.empty_like(down_weight, memory_format=torch.contiguous_format),
    )
