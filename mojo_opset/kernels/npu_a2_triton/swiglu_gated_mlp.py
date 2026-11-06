import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry

"""
SwigluGatedMlp for NPU A2 (forward only): two stage launches (P0 plan,
swiglu_gated_mlp_fusion.md section 6).

    stage 1: act = silu(clamp(x @ Wg^T, max=L)) * clamp(x @ Wu^T, +-L)
        One K loop drives two tl.dot accumulators sharing the x tile loads;
        the SwiGLU epilogue runs on the vector cores of the same launch, so
        gate/up never hit GM. Wg/Wu enter as two B pointers (virtual
        half-packing; HF weight layout kept as-is).
    stage 2: y = act @ Wd^T.

Clamp follows Glm5NextTextMLP (swiglu_limit=0 disables; flash uses 10.0).

Modifications for NPU architecture by triton-x team, 2025.
"""

# Tile selection, gmm-style: A2 stages dot tiles through the 192-KiB UB
# (budget = UB/2); blocks take real (non-pow2) values. Budget-model derived,
# pending on-device validation (evidence: a5 twin).
#   decode rows<=16 -> (16, 64, deepest legal BK)
#   base            -> BM=min(rows,128), BN=min(N,cap); BN=64 when BM<=32
#   BK              -> budget//((BM+num_b*BN)*esize), 16-aligned
_TILE_BUDGET_BYTES = 192 * 1024 // 2  # UB / 2
_BK_MAX = 256
_BK_MIN = 16
_WIDE_N_BK_MAX = 128  # BK cap when BN >= 128
_DECODE_ROWS = 16


def _align16(value: int) -> int:
    return (value + 15) // 16 * 16


def _stage_tiles(num_b: int, bn_cap: int, rows: int, n: int, k: int, esize: int):
    """Deterministic per-stage (BM, BN, BK), gmm-style."""
    if rows <= _DECODE_ROWS:
        block_m, block_n = 16, 64
        per_k = (block_m + num_b * block_n) * esize
        block_k = min((_TILE_BUDGET_BYTES // per_k) & ~15, _BK_MAX, k & ~15 or _BK_MIN)
        return block_m, block_n, max(block_k, _BK_MIN)

    block_m = min(rows, 128)
    if num_b > 1 and block_m <= 32:
        bn_cap = min(bn_cap, 64)
    block_n = max(min(n, bn_cap), 16)

    per_k = (_align16(block_m) + num_b * _align16(block_n)) * esize
    block_k = min((_TILE_BUDGET_BYTES // per_k) & ~15, _BK_MAX, k & ~15 or _BK_MIN)
    if block_n >= 128:
        block_k = min(block_k, _WIDE_N_BK_MAX)
    return block_m, block_n, max(block_k, _BK_MIN)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def _swiglu_gated_mlp_mm1_swiglu_kernel(
    x, gate_weight, up_weight, act,
    M, N, K,
    stride_xm, stride_xk, stride_gk, stride_gn, stride_uk, stride_un, stride_am,
    limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Stage 1: act = silu(clamp(x@Wg^T)) * clamp(x@Wu^T) over [M, N=intermediate].

    Dual tl.dot accumulators share each x tile; the elementwise epilogue runs
    on the compiler-dispatched vector cores of the same launch. FP32 math.
    """
    pid = tl.program_id(axis=0)
    num_cores = tl.num_programs(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    for tile in range(pid, num_pid_m * num_pid_n, num_cores):
        # Grouped ordering keeps one BLOCK_M row band of x hot in L2.
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

        x_ptrs = x + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        g_ptrs = gate_weight + offs_k[:, None] * stride_gk + offs_n[None, :] * stride_gn
        u_ptrs = up_weight + offs_k[:, None] * stride_uk + offs_n[None, :] * stride_un
        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_left = K - k * BLOCK_K
            x_tile = tl.load(x_ptrs, mask=m_mask & (offs_k[None, :] < k_left), other=0.0)
            g_tile = tl.load(g_ptrs, mask=(offs_k[:, None] < k_left) & n_mask, other=0.0)
            u_tile = tl.load(u_ptrs, mask=(offs_k[:, None] < k_left) & n_mask, other=0.0)
            acc_gate = tl.dot(x_tile, g_tile, acc=acc_gate)
            acc_up = tl.dot(x_tile, u_tile, acc=acc_up)
            x_ptrs += BLOCK_K * stride_xk
            g_ptrs += BLOCK_K * stride_gk
            u_ptrs += BLOCK_K * stride_uk

        gate = acc_gate
        up = acc_up
        if HAS_LIMIT:
            gate = tl.minimum(gate, limit)
            up = tl.minimum(tl.maximum(up, -limit), limit)
        result = gate * tl.sigmoid(gate) * up
        act_ptrs = act + offs_m[:, None] * stride_am + offs_n[None, :]
        tl.store(act_ptrs, result.to(act.dtype.element_ty), mask=m_mask & n_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def _swiglu_gated_mlp_mm2_kernel(
    act, down_weight, out,
    M, N, K,
    stride_am, stride_ak, stride_wk, stride_wn, stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Stage 2: out = act @ down_weight^T over [M, N=hidden], FP32 accumulate."""
    pid = tl.program_id(axis=0)
    num_cores = tl.num_programs(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    for tile in range(pid, num_pid_m * num_pid_n, num_cores):
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

        a_ptrs = act + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        w_ptrs = down_weight + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_left = K - k * BLOCK_K
            a_tile = tl.load(a_ptrs, mask=m_mask & (offs_k[None, :] < k_left), other=0.0)
            w_tile = tl.load(w_ptrs, mask=(offs_k[:, None] < k_left) & n_mask, other=0.0)
            accumulator = tl.dot(a_tile, w_tile, acc=accumulator)
            a_ptrs += BLOCK_K * stride_ak
            w_ptrs += BLOCK_K * stride_wk

        out_ptrs = out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, accumulator.to(out.dtype.element_ty), mask=m_mask & n_mask)


def _check_swiglu_gated_mlp_inputs(x, gate_weight, up_weight, down_weight, swiglu_limit):
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("swiglu_gated_mlp supports float16/bfloat16 only.")
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


def swiglu_gated_mlp_fwd_impl(
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

    # Stage 1: fused gate/up projections + SwiGLU (gate/up never hit GM).
    # B tiles address Wg^T / Wu^T through swapped strides (virtual half-packing).
    act = x_2d.new_empty((rows, intermediate_size))
    block_m, block_n, block_k = _stage_tiles(2, 128, rows, intermediate_size, hidden_size, x.element_size())
    num_tiles = triton.cdiv(rows, block_m) * triton.cdiv(intermediate_size, block_n)
    _swiglu_gated_mlp_mm1_swiglu_kernel[(min(gemm_cores, num_tiles),)](
        x_2d, gate_weight, up_weight, act, rows, intermediate_size, hidden_size,
        x_2d.stride(0), x_2d.stride(1), gate_weight.stride(1), gate_weight.stride(0),
        up_weight.stride(1), up_weight.stride(0), act.stride(0),
        float(swiglu_limit), swiglu_limit > 0.0,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )

    # Stage 2: down projection.
    out = x_2d.new_empty((rows, hidden_size))
    block_m, block_n, block_k = _stage_tiles(1, 128, rows, hidden_size, intermediate_size, x.element_size())
    num_tiles = triton.cdiv(rows, block_m) * triton.cdiv(hidden_size, block_n)
    _swiglu_gated_mlp_mm2_kernel[(min(gemm_cores, num_tiles),)](
        act, down_weight, out, rows, hidden_size, intermediate_size,
        act.stride(0), act.stride(1), down_weight.stride(1), down_weight.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return out.reshape(ori_shape)


@torch.library.custom_op("mojo_npu_triton_a2::swiglu_gated_mlp_fwd", mutates_args=())
def swiglu_gated_mlp_fwd(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    return swiglu_gated_mlp_fwd_impl(x, gate_weight, up_weight, down_weight, swiglu_limit)


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
