import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry
from mojo_opset.kernels._weight_cache import PreparedWeightCache

"""
GmmSwigluQuant for NPU A5 (forward only): one fused launch. The former
swiglu_gated_mlp down projection is gone; the output now feeds a quantized
consumer directly (ops-transformer gmm/grouped_matmul_swiglu_quant_v2
semantics, 950PR MX path):

    C     = (x . w) with MXFP8 dequant fused into the dot
    act, lin = C[:, :I], C[:, I:]          (weight halves, I = N // 2)
    S     = silu(clamp(act, max=L)) * clamp(lin, +-L),  L=0 disables
    y, y_scale = MX quant of S             (OCP: e = floor(log2 amax) - 8,
                                            32-element groups, e8m0 scales)

I/O (all contiguous; the wrapper repacks weight_scale once per live weight
into the contiguous [k_blocks, N] layout the kernel consumes):
    x           [M, K]         float8_e4m3fn
    weight      [1, K, N]      float8_e4m3fn   (N = 2I; first half = act)
    x_scale     [M, cdiv(K,64), 2]       uint8 e8m0, per 32 K elements
    weight_scale[1, cdiv(K,64), N, 2]    uint8 e8m0, per 32 K per column
    y           [M, I]         float8_e4m3fn
    y_scale     [M, cdiv(I,64), 2]       uint8 e8m0

One K loop drives two tl.dot_scaled accumulators sharing the x tile loads
(the two B pointers address the two N halves of one weight tensor, virtual
half-packing); the SwiGLU + MX quant epilogue runs on the vector cores of
the same launch, so the SwiGLU halves never hit GM.

Modifications for NPU architecture by triton-x team, 2025.
"""

# Tile selection, gmm-style; BM keeps real (non-pow2) values, but BN/BK are
# powers of two: the tl.dot_scaled scale operands are [BM, BK // 32] /
# [BN, BK // 32] tiles and the MX groups reshape to [BM, BN // 32, 32], and
# non-pow2 scale-tile extents (BK=96 or BN=160) fault the AIC cube
# instruction on this triton-ascend build (evidence: scripts/
# gmm_swiglu_quant_{rint,tile}_probe.py). Otherwise derived from the
# swiglu_gated_mlp stage-1 rules (perf / gap-analysis reports, repo root)
# with fp8 element size (1 byte) and the MX quant epilogue budgeted like
# the clamp epilogue (BM capped at 128):
#   decode rows<=16 -> (16, 64, deepest legal BK)
#   base            -> BM=min(rows,128), BN=min(I,cap); BN=64 when BM<=32
#   BK              -> L1/2 // ((BM+2*BN)*esize), then pow2 floor, BN>=128
#                      caps at 256
_TILE_BUDGET_BYTES = 512 * 1024 // 2  # L1 / 2
_BK_MAX = 512
_BK_MIN = 32  # one MX scale block: tl.dot_scaled K granularity
_WIDE_N_BK_MAX = 256  # MLIR cap for BN >= 128
_DECODE_ROWS = 16
_BM_CAP = 128  # fused quant epilogue UB budget (2.5*BM*BN*4B <= 248 KiB)
_NUM_B = 2  # dual accumulators (act / lin halves)


def _align32(value: int) -> int:
    return (value + 31) // 32 * 32


def _pow2_floor(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def _stage_tiles(rows: int, n: int, k: int, cores: int = 28):
    """Deterministic (BM, BN, BK) for the dual-dot MX kernel, gmm-style."""
    if rows <= _DECODE_ROWS:
        block_m, bn_cap = 16, 64
    else:
        block_m = min(rows, _BM_CAP)
        bn_cap = 64 if block_m <= 32 else 128
    block_n = max(min(triton.next_power_of_2(n), bn_cap), 32)

    per_k = _align32(block_m) + _NUM_B * _align32(block_n)  # fp8: 1 byte/elem
    block_k = min(_TILE_BUDGET_BYTES // per_k, _BK_MAX, k & ~(_BK_MIN - 1) or _BK_MIN)
    if block_n >= 128:
        block_k = min(block_k, _WIDE_N_BK_MAX)
    return block_m, block_n, max(_pow2_floor(block_k), _BK_MIN)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def _gmm_swiglu_quant_kernel(
    x, weight, x_scale, weight_scale, y, y_scale,
    M, N, K,
    limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y, y_scale = mxfp8_quant(silu(clamp(x @ W[:, :I])) * clamp(x @ W[:, I:])).

    Two tl.dot_scaled accumulators share each x tile (fp8e4m3 operands,
    e8m0 per-32 scales); the SwiGLU + MX quant epilogue runs on the
    compiler-dispatched vector cores of the same launch. FP32 math.
    """
    pid = tl.program_id(axis=0)
    num_cores = tl.num_programs(axis=0)
    half_n = N // 2
    k_pairs = tl.cdiv(K, 64)
    i_pairs = tl.cdiv(half_n, 64)
    k_blocks = tl.cdiv(K, 32)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(half_n, BLOCK_N)

    for tile in range(pid, num_pid_m * num_pid_n, num_cores):
        # Grouped ordering keeps one BLOCK_M row band of x hot in L2.
        group_id = tile // (8 * num_pid_n)
        first_m = group_id * 8
        group_m = tl.minimum(num_pid_m - first_m, 8)
        pid_m = first_m + (tile % group_m)
        pid_n = (tile % (8 * num_pid_n)) // group_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # over I (output cols)
        offs_k = tl.arange(0, BLOCK_K)
        offs_kb = tl.arange(0, BLOCK_K // 32)
        m_mask = offs_m < M
        n_mask = offs_n < half_n

        x_ptrs = x + offs_m[:, None] * K + offs_k[None, :]
        xs_base = x_scale + offs_m[:, None] * (k_pairs * 2)
        # First half (act) at column n, second half (lin) at column n + I.
        w1_ptrs = weight + offs_k[:, None] * N + offs_n[None, :]
        w2_ptrs = w1_ptrs + half_n
        # weight_scale arrives repacked by the host as contiguous
        # [k_blocks, N]: byte (kb, n) at kb * N + n, vectorizable along n
        # (the reference (kp, N, 2) pair packing interleaves n with stride 2
        # and halves the MX dot rate; evidence:
        # scripts/gmm_swiglu_quant_variants_bench.py). Tiles are
        # [BN, BK // 32] in dot_scaled rhs_scale order.
        ws_base = weight_scale + offs_n[:, None]

        acc_act = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_lin = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_left = K - k * BLOCK_K
            kb = k * (BLOCK_K // 32) + offs_kb
            k_mask = offs_k < k_left
            kb_mask = kb < k_blocks
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            xs_tile = tl.load(xs_base + kb[None, :], mask=m_mask[:, None] & kb_mask[None, :], other=0)
            ws_ptr = kb[None, :] * N
            w1_tile = tl.load(w1_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            w2_tile = tl.load(w2_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            ws1_tile = tl.load(ws_base + ws_ptr, mask=n_mask[:, None] & kb_mask[None, :], other=0)
            ws2_tile = tl.load(ws_base + half_n + ws_ptr, mask=n_mask[:, None] & kb_mask[None, :], other=0)
            acc_act = tl.dot_scaled(x_tile, xs_tile, "e4m3", w1_tile, ws1_tile, "e4m3", acc=acc_act)
            acc_lin = tl.dot_scaled(x_tile, xs_tile, "e4m3", w2_tile, ws2_tile, "e4m3", acc=acc_lin)
            x_ptrs += BLOCK_K
            w1_ptrs += BLOCK_K * N
            w2_ptrs += BLOCK_K * N

        act = acc_act
        lin = acc_lin
        if HAS_LIMIT:
            act = tl.minimum(act, limit)
            lin = tl.minimum(tl.maximum(lin, -limit), limit)
        s = act * tl.sigmoid(act) * lin
        s = tl.where(n_mask[None, :], s, 0.0)  # tail cols must not inflate amax

        # ---- MX dynamic quant, 32-element groups, OCP scale algorithm ----
        # e = floor(log2(amax)) - emax with emax = 8 (E4M3FN max 448); the
        # no-carry OCP scale may saturate values to +-448. e is clamped at
        # the e8m0 encoding floor (-127): a sub-2^-119 amax would otherwise
        # wrap the byte cast and decode as a huge scale.
        groups = s.reshape(BLOCK_M, BLOCK_N // 32, 32)
        amax = tl.max(tl.abs(groups), axis=2)
        e = tl.where(amax > 0.0, tl.floor(tl.log2(amax)) - 8.0, -127.0)
        e = tl.maximum(e, -127.0)
        q = groups * tl.exp2(-e)[:, :, None]
        # rint (round half to even) inlined: libdevice.rint crashes the AIC
        # on this triton-ascend build, floor + parity does not.
        q_floor = tl.floor(q)
        q_frac = q - q_floor
        q_odd = q_floor - 2.0 * tl.floor(q_floor * 0.5) > 0.5
        q = q_floor + (q_frac > 0.5).to(tl.float32) + (q_odd & (q_frac == 0.5)).to(tl.float32)
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        y_ptrs = y + offs_m[:, None] * half_n + offs_n[None, :]
        tl.store(y_ptrs, q.reshape(BLOCK_M, BLOCK_N).to(y.dtype.element_ty),
                 mask=m_mask[:, None] & n_mask[None, :])
        offs_g = tl.arange(0, BLOCK_N // 32)
        g_start = pid_n * BLOCK_N + offs_g * 32  # first column of each MX group
        ys_ptrs = y_scale + offs_m[:, None] * (i_pairs * 2) + g_start[None, :] // 32
        tl.store(ys_ptrs, (e + 127.0).to(tl.int32).to(tl.uint8),
                 mask=m_mask[:, None] & (g_start[None, :] < half_n))


def _repack_weight_scale(weight_scale, transposed=False):
    """(1, kp, N, 2) pair packing -> contiguous (k_blocks, N) for the kernel."""
    del transposed
    kp, n = weight_scale.shape[1], weight_scale.shape[2]
    return weight_scale.view(kp, n, 2).transpose(1, 2).reshape(kp * 2, n).contiguous()


_prepare_weight_scale = PreparedWeightCache(_repack_weight_scale)


def _check_gmm_swiglu_quant_inputs(x, weight, weight_scale, x_scale, group_list, swiglu_limit):
    if x.dtype is not torch.float8_e4m3fn:
        raise ValueError("gmm_swiglu_quant x must be float8_e4m3fn.")
    if x.dim() < 1:
        raise ValueError("gmm_swiglu_quant input must have at least one dimension.")
    k = x.shape[-1]
    rows = x.numel() // k if k else 0
    if weight.dim() != 3 or weight.shape[0] != 1 or weight.shape[1] != k:
        raise ValueError(
            f"gmm_swiglu_quant weight must be [1, K, N]=[1, {k}, N] (dense single expert), got {list(weight.shape)}."
        )
    n = weight.shape[2]
    if n % 2:
        raise ValueError("gmm_swiglu_quant weight N must be even (two SwiGLU halves).")
    k_pairs = (k + 63) // 64
    if weight_scale.shape != (1, k_pairs, n, 2) or weight_scale.dtype not in (torch.uint8, torch.int8):
        raise ValueError(
            f"gmm_swiglu_quant weight_scale must be uint8/int8 e8m0 of shape [1, {k_pairs}, {n}, 2], "
            f"got {tuple(weight_scale.shape)} {weight_scale.dtype}."
        )
    if x_scale.dim() != 3 or x_scale.shape[0] != rows or x_scale.shape[1:] != (k_pairs, 2) \
            or x_scale.dtype not in (torch.uint8, torch.int8):
        raise ValueError(
            f"gmm_swiglu_quant x_scale must be uint8/int8 e8m0 of shape [{rows}, {k_pairs}, 2], "
            f"got {tuple(x_scale.shape)} {x_scale.dtype}."
        )
    if group_list.dtype != torch.int64 or group_list.numel() != 1:
        raise ValueError("gmm_swiglu_quant group_list must be int64 [1] (cumsum == [rows]; dense only).")
    if not (x.device == weight.device == weight_scale.device == x_scale.device == group_list.device):
        raise ValueError("gmm_swiglu_quant tensors must share device.")
    if swiglu_limit < 0:
        raise ValueError("gmm_swiglu_quant swiglu_limit must be nonnegative.")


def gmm_swiglu_quant_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(y, y_scale) = mxfp8_quant(silu(clamp(x@W_left)) * clamp(x@W_right))."""
    _check_gmm_swiglu_quant_inputs(x, weight, weight_scale, x_scale, group_list, swiglu_limit)
    k = x.shape[-1]
    n = weight.shape[2]
    inter = n // 2
    rows = x.numel() // k
    if rows == 0:
        return (
            x.new_empty((0, inter), dtype=torch.float8_e4m3fn),
            x.new_zeros((0, triton.cdiv(inter, 64), 2), dtype=torch.uint8),
        )

    x_2d = x.reshape(rows, k).contiguous()
    weight = weight.contiguous().view(k, n)
    x_scale = x_scale.contiguous().view(rows, -1)
    if weight_scale.dtype is not torch.uint8:
        weight_scale = weight_scale.view(torch.uint8)
    if x_scale.dtype is not torch.uint8:
        x_scale = x_scale.view(torch.uint8)
    # One cached repack per live weight: the interleaved reference packing
    # halves the MX dot rate in-kernel (see _repack_weight_scale).
    weight_scale = _prepare_weight_scale(weight_scale.contiguous(), transposed=False).view(-1)

    y = x_2d.new_empty((rows, inter), dtype=torch.float8_e4m3fn)
    y_scale = x_2d.new_zeros((rows, triton.cdiv(inter, 64), 2), dtype=torch.uint8)
    gemm_cores = get_num_cores("cube")

    block_m, block_n, block_k = _stage_tiles(rows, inter, k, gemm_cores)
    num_tiles = triton.cdiv(rows, block_m) * triton.cdiv(inter, block_n)
    _gmm_swiglu_quant_kernel[(min(gemm_cores, num_tiles),)](
        x_2d, weight, x_scale, weight_scale, y, y_scale,
        rows, n, k,
        float(swiglu_limit), swiglu_limit > 0.0,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return y, y_scale


@torch.library.custom_op("mojo_npu_triton_a5::gmm_swiglu_quant_fwd", mutates_args=())
def gmm_swiglu_quant_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return gmm_swiglu_quant_fwd_impl(x, weight, weight_scale, x_scale, group_list, swiglu_limit)


@gmm_swiglu_quant_fwd.register_fake
def _gmm_swiglu_quant_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.numel() // x.shape[-1]
    inter = weight.shape[2] // 2
    pairs = -(-inter // 64)
    return (
        torch.empty((rows, inter), dtype=torch.float8_e4m3fn, device=x.device,
                    memory_format=torch.contiguous_format),
        torch.empty((rows, pairs, 2), dtype=torch.uint8, device=x.device,
                    memory_format=torch.contiguous_format),
    )
