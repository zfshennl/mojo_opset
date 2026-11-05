import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry

"""
SwigluGatedMlp for NPU A5 (forward only), implementing the P0 fusion plan
(swiglu_gated_mlp_fusion.md section 6): the forward is TWO stage kernels,

    stage 1 (one launch): act = silu(clamp(x @ Wg^T, max=L)) * clamp(x @ Wu^T, +-L)
        Each output tile's K loop drives TWO tl.dot accumulators that share the
        x tile loads; the SwiGLU vec epilogue is auto-dispatched to the vector
        cores by the compiler (dot lands on the cube core), so the gate/up
        projections never touch GM. Wg/Wu enter as two B pointers (virtual
        half-packing; no physical concat, HF weight layout is kept as-is).
    stage 2 (one launch): y = act @ Wd^T.

    gate = x @ Wg^T;  up = x @ Wu^T
    gate = gate.clamp(max=swiglu_limit);  up = up.clamp(+-swiglu_limit)   (Glm5NextTextMLP)
    act = silu(gate) * up;  y = act @ Wd^T

glm-5.3 uses hidden 6144 / intermediate 12288 (dense) and 2048 (shared
experts); glm-5.3-flash uses hidden 4096 with swiglu_limit=10.0.

Modifications for NPU architecture by triton-x team, 2025.
"""

# Tile selection follows the gmm (grouped_matmul_swiglu_quant_v2) tiling
# methodology, mapped onto triton block semantics: triton BLOCK_M/N/K are the
# L1-level tile allocation (per cube core), while the L0A/L0B/L0C split of a
# tile is done by the compiler and is deliberately NOT modelled here.
# triton-ascend does not require power-of-two or 16-aligned blocks, so all
# three take REAL values (verified on 950PR: odd BM/BN/BK compile and match
# the reference within bf16 tolerance).
#   1. base blocks from the shape (their CalBasicBlock): BM = min(rows, cap)
#      exactly (no rounding -- e.g. rows=96 stages 96 rows instead of a
#      masked 128 block), BN = min(N, cap) exactly (caps below); stage-1
#      additionally narrows BN to 64 for BM<=32 (the dual B tiles leave
#      little L1 for width there; fresh-process sweeps: (32,64,512) beats
#      (32,128,256) by ~2.5%, while stage-2 prefers the wide form).
#      For rows>128 both stages go BM<=256 / BN=128: the M band count halves
#      (weights are streamed half as many times -- the aclnn gap analysis,
#      swiglu_gated_mlp_gap_analysis.md). The L1 staging stays within budget
#      ((256 + num_b*128)*BK*esize <= 256 KiB), but the fp32 ACCUMULATORS no
#      longer fit L0C with the default double buffer: stage-1's dual accs
#      are 2*BM*128*4B = 256 KiB, so those launches MUST pass
#      multibuffer_mode={"gm":2,"l1":2,"l0c":1,"ub":2} (single-buffered L0C,
#      compiler default is l0c=2 -> cc overflow). Measured: mm1 rows=512
#      324->291 us (-10%), rows=4096 2318->2240 us; mm2 keeps the default db
#      (acc 128 KiB x2 fits L0C; 256x256 with l0c=1 measured SLOWER, 245 vs
#      190 us -- aclnn's 256x256 edge is its ASW/L2 scheduling, not the db).
#      rows<=128 keeps the default buffers (dual accs 2*128*128*4*2 = 256 KiB
#      exactly fits L0C with db).
#   2. K staging fills the tile budget exactly (their CalL1Depth): BK is
#      budget // ((BM + num_b*BN) * esize), where num_b=2 for the dual-B
#      stage-1 and 1 for stage-2. fp32 doubles esize, so BK halves
#      automatically. Budget = L1/2: Ascend950 AIC L1 is 512 KiB (catlass
#      arch.hpp) and the other half is left for the compiler's double
#      buffering of the K loop.
#   3. decode profile (their small-M path gating): rows<=16 is a GEMV whose
#      cube pipe waits on the weight stream (measured MAC 13-16%), so it
#      takes the deepest legal K staging with narrow BN=64.
# No underfill resplit: like gmm's ReSplitBaseN (which only rebalances tails
# at equal wave count), idle cores in a partial wave are tolerated -- fresh
# sweeps show splitting BN to fill the grid never wins warm (e.g. mm2 rows=64
# runs fastest at 24/28 tiles: 14 us vs 22 us for the split form).
# Empirical MLIR constraint: BN>=128 with BK=512 fails to compile, so wide-N
# tiles cap BK at 256; BK is capped at 512 overall (1024 unverified) and
# floored at 16 (tl.dot minimum K). The budget check uses 16-ALIGNED staging:
# the compiler pads each tile dim (e.g. BK 455 -> 456) and double-buffers the
# K loop in cbuf (= L1), so 2 * aligned_staging must fit -- the selector
# keeps aligned_staging <= L1/2 (checked on-device: a (16,64,455) fp32 tile
# overflowed cbuf by exactly the padding).
_TILE_BUDGET_BYTES = 512 * 1024 // 2  # L1 (512 KiB) / 2, see above
_BK_MAX = 512
_BK_MIN = 16
_WIDE_N_BK_MAX = 256  # BK cap when BN >= 128 (MLIR)
_DECODE_ROWS = 16
_BM_CAP = 128  # default BM cap; rows>128 raises it to 256 with BN=128


def _align16(value: int) -> int:
    return (value + 15) // 16 * 16


def _stage_tiles(num_b: int, bn_cap: int, rows: int, n: int, k: int, esize: int, clamp_epilogue: bool = False):
    """Deterministic per-stage (BM, BN, BK) with real values, gmm-style.

    clamp_epilogue=True (stage 1 with swiglu_limit>0) keeps BM<=128: the
    clamp/SwiGLU vector epilogue's UB working set scales with BM*BN and
    256x128 fp32 chunks overflow UB (248 KiB) at compile time, while the
    UB-legal 256x64 forms measure far slower (fresh: 478 vs 323 us).
    """
    # 3. decode profile first (rows <= 16): deepest legal K staging, BN=64.
    if rows <= _DECODE_ROWS:
        block_m, block_n = 16, 64
        per_k = (block_m + num_b * block_n) * esize
        block_k = min((_TILE_BUDGET_BYTES // per_k) & ~15, _BK_MAX, k & ~15 or _BK_MIN)
        return block_m, block_n, max(block_k, _BK_MIN)

    # 1. base blocks from the shape, at their real values (no rounding).
    # rows>128 (clamp-free only): BM cap 256 with BN=128 (M-band halving).
    block_m = min(rows, _BM_CAP if (rows <= _BM_CAP or clamp_epilogue) else 256)
    if num_b > 1 and block_m <= 32:
        bn_cap = min(bn_cap, 64)
    elif rows > _BM_CAP and not clamp_epilogue:
        bn_cap = min(bn_cap, 128)
    block_n = max(min(n, bn_cap), 16)

    # 2. exact budget fill for K staging (16-aligned, compiler pads+double
    #    buffers in cbuf), then the empirical caps.
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
    # rows>128 tiles carry dual 256x128 fp32 accumulators = 256 KiB, so the
    # L0C double buffer must be off for those (see tile-selection comment).
    act = x_2d.new_empty((rows, intermediate_size))
    block_m, block_n, block_k = _stage_tiles(
        2, 128, rows, intermediate_size, hidden_size, x.element_size(), swiglu_limit > 0.0)
    num_tiles = triton.cdiv(rows, block_m) * triton.cdiv(intermediate_size, block_n)
    mm1_kwargs = {"multibuffer_mode": {"gm": 2, "l1": 2, "l0c": 1, "ub": 2}} if block_m > 128 else {}
    _swiglu_gated_mlp_mm1_swiglu_kernel[(min(gemm_cores, num_tiles),)](
        x_2d, gate_weight, up_weight, act, rows, intermediate_size, hidden_size,
        x_2d.stride(0), x_2d.stride(1), gate_weight.stride(1), gate_weight.stride(0),
        up_weight.stride(1), up_weight.stride(0), act.stride(0),
        float(swiglu_limit), swiglu_limit > 0.0,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, **mm1_kwargs,
    )

    # Stage 2: down projection.
    out = x_2d.new_empty((rows, hidden_size))
    block_m, block_n, block_k = _stage_tiles(1, 256, rows, hidden_size, intermediate_size, x.element_size())
    num_tiles = triton.cdiv(rows, block_m) * triton.cdiv(hidden_size, block_n)
    _swiglu_gated_mlp_mm2_kernel[(min(gemm_cores, num_tiles),)](
        act, down_weight, out, rows, hidden_size, intermediate_size,
        act.stride(0), act.stride(1), down_weight.stride(1), down_weight.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return out.reshape(ori_shape)


@torch.library.custom_op("mojo_npu_triton_a5::swiglu_gated_mlp_fwd", mutates_args=())
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
