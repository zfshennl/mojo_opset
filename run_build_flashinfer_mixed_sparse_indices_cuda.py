#!/usr/bin/env python3
"""Run the ORIGINAL vLLM implementation of build_flashinfer_mixed_sparse_indices on a CUDA host.

Everything is patched in memory only — this script never writes to or modifies
repository files. The triton kernel, its remap helper and the launcher below
are COPIED VERBATIM from vLLM (vllm/models/deepseek_v41/common/ops/cache_utils.py);
unrelated vllm code is not copied and nothing from the vllm package or repo is
imported, read or otherwise used at runtime. The copy is bound into the repo
dispatch:

 1. mojo_opset.functions._dispatch.resolve_implementation:
    the op is configured only for npu.a2/npu.a5; report "triton" on ilu too.
 2. mojo_opset.functions._dispatch._load_kernel_binding:
    the ilu triton provider has no binding for the op; return an adapter that
    forwards the repo's normalized kernel-binding signature to the embedded
    launcher (grid=(num_tokens,), num_warps by block size — the original
    per-token program, runtime div/mod, unconditional remap).
 3. tests._fixtures.check_implementation: accept the patched-in op on ilu.generic.
 4. tests.performance.timing.kernel_spans: with CUDA activity tracing, Kineto
    mirrors every record_function range onto the GPU timeline as a same-named
    "gpu_user_annotation" event; the duplicate-range detector would reject the
    mirrored sample ranges ("duplicate profiler range"), so the mirrors are
    filtered out before analysis. Inert on NPU traces (no mirrored events).
 5. tests.performance._adapters.ilu: add make_l2_flush (buffer-sum eviction
    mirroring the npu adapter) so --perf-l2-flush works on CUDA; the harness
    otherwise fails the session ("only supported by the npu adapter"). Direct
    mode passes the flag by default for cold-cache numbers comparable to the
    NPU baseline runs.

Usage (from the repository root):
    python3 run_build_flashinfer_mixed_sparse_indices_cuda.py [pytest args...]   # direct mode
    python -m pytest -p run_build_flashinfer_mixed_sparse_indices_cuda \
        tests/performance/functions/test_build_flashinfer_mixed_sparse_indices.py \
        --mojo-target ilu.generic --perf-output out.json                        # plugin mode

Plugin mode applies the patches at import time, before any conftest/test module
binds the original names (use `python -m pytest` or PYTHONPATH=. so the module
resolves). The suite runs with --mojo-target ilu.generic, the stock
profiler+e2e timers and --perf-l2-flush (patch 5 makes it CUDA-capable; drop
it on the CLI only if you must). Extra CLI arguments are forwarded to pytest
in direct mode (e.g. "-k dsv41_mixed_ratio1" or "-q").

    --verify-patches   apply the patches, smoke-check the wiring and exit
                       (direct mode only; works without a GPU).
"""

import functools
import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Snapshot taken before this module imports anything heavy: proves that OUR
# machinery never loads the vllm package (anything preloaded by the embedder
# is not our doing and is excluded from the check).
_PRELOADED_VLLM = frozenset(
    m for m in list(sys.modules) if m == "vllm" or m.startswith("vllm.")
)


def assert_no_vllm_imported():
    """Fail hard if this script's machinery pulled the vllm package in."""
    leaked = [
        m
        for m in sys.modules
        if (m == "vllm" or m.startswith("vllm.")) and m not in _PRELOADED_VLLM
    ]
    assert not leaked, f"vllm was imported by this machinery: {leaked}"


OP_ID = "build_flashinfer_mixed_sparse_indices"
PERF_TEST = "tests/performance/functions/test_build_flashinfer_mixed_sparse_indices.py"
DEFAULT_OUTPUT = str(REPO_ROOT / "build_flashinfer_mixed_sparse_indices_cuda_vllm.json")

# ---------------------------------------------------------------------------
# vLLM original implementation — copied verbatim from
# vllm/models/deepseek_v41/common/ops/cache_utils.py. Only this op's remap
# helper, kernel and launcher are copied; unrelated vllm code removed.
# ---------------------------------------------------------------------------


@triton.jit
def _remap_flashinfer_index(values, block_size, block_span):
    # FlashInfer's DSv4 kernel indexes sparse KV by physical token stride, so
    # packed pages (#44577) need block*block_size+off -> block*block_span+off.
    # The launcher declares these indices with sparse_indices_are_storage_offsets=True.
    is_valid = values >= 0
    safe_values = tl.where(is_valid, values, 0)
    values = (safe_values // block_size) * block_span
    values += safe_values % block_size
    return tl.where(is_valid, values, -1)


@triton.jit(
    do_not_specialize=[
        "sparse_indices_stride",
        "decode_swa_stride",
        "decode_compressed_stride",
        "prefill_topk_stride",
        "swa_block_table_stride",
        "swa_block_size",
        "swa_block_span",
        "compressed_block_table_stride",
        "compressed_block_size",
        "compressed_block_span",
        "NUM_DECODE_TOKENS",
        "PREFILL_TOPK_STRIDE",
    ]
)
def _build_flashinfer_mixed_sparse_indices_kernel(
    sparse_indices_ptr,
    sparse_indices_stride,
    sparse_topk_lens_ptr,
    decode_swa_indices_ptr,
    decode_swa_stride,
    decode_compressed_indices_ptr,
    decode_compressed_stride,
    decode_compressed_topk_lens_ptr,
    decode_is_valid_token_ptr,
    prefill_topk_indices_ptr,
    prefill_topk_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    replay_start_ptr,
    swa_block_table_ptr,
    swa_block_table_stride,
    swa_block_size,
    swa_block_span,
    compressed_block_table_ptr,
    compressed_block_table_stride,
    compressed_block_size,
    compressed_block_span,
    NUM_DECODE_TOKENS,
    WINDOW_SIZE: tl.constexpr,
    SWA_INDEX_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOP_K: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    PREFILL_TOPK_STRIDE,
    DECODE_COMPRESSED_TOPK: tl.constexpr,
    DECODE_COMPRESSED_INDICES_ARE_LOCAL: tl.constexpr,
    HAS_DECODE_COMPRESSED_LENS: tl.constexpr,
    WINDOW_BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    if token_idx < NUM_DECODE_TOKENS:
        for i in range(0, SWA_INDEX_WIDTH, WINDOW_BLOCK_SIZE):
            offset = i + tl.arange(0, WINDOW_BLOCK_SIZE)
            mask = offset < SWA_INDEX_WIDTH
            values = tl.load(
                decode_swa_indices_ptr + token_idx * decode_swa_stride + offset,
                mask=mask,
                other=-1,
            )
            values = _remap_flashinfer_index(values, swa_block_size, swa_block_span)
            tl.store(
                sparse_indices_ptr + token_idx * sparse_indices_stride + offset,
                values,
                mask=mask,
            )

        compressed_len = tl.zeros((), dtype=tl.int32)
        for i in range(0, PADDED_TOP_K, TOPK_BLOCK_SIZE):
            offset = i + tl.arange(0, TOPK_BLOCK_SIZE)
            mask = offset < PADDED_TOP_K
            values = tl.load(
                decode_compressed_indices_ptr
                + token_idx * decode_compressed_stride
                + offset,
                mask=offset < DECODE_COMPRESSED_TOPK,
                other=-1,
            )
            if DECODE_COMPRESSED_INDICES_ARE_LOCAL:
                token_valid = tl.load(decode_is_valid_token_ptr + token_idx)
                is_valid = values >= 0
                req_idx = tl.load(token_to_req_indices_ptr + token_idx)
                block_indices = values // compressed_block_size
                block_numbers = tl.load(
                    compressed_block_table_ptr
                    + req_idx * compressed_block_table_stride
                    + block_indices,
                    mask=mask & is_valid,
                    other=-1,
                )
                block_offsets = values % compressed_block_size
                values = block_numbers * compressed_block_size + block_offsets
                values = tl.where(is_valid, values, -1)
                compressed_len += tl.sum((is_valid & token_valid).to(tl.int32), axis=0)
            values = _remap_flashinfer_index(
                values, compressed_block_size, compressed_block_span
            )
            tl.store(
                sparse_indices_ptr
                + token_idx * sparse_indices_stride
                + SWA_INDEX_WIDTH
                + offset,
                values,
                mask=mask,
            )

        if DECODE_COMPRESSED_TOPK == 0:
            compressed_len = tl.zeros((), dtype=tl.int32)
        elif not DECODE_COMPRESSED_INDICES_ARE_LOCAL:
            if HAS_DECODE_COMPRESSED_LENS:
                compressed_len = tl.load(decode_compressed_topk_lens_ptr + token_idx)
            else:
                compressed_len = tl.full((), DECODE_COMPRESSED_TOPK, dtype=tl.int32)

        tl.store(sparse_topk_lens_ptr + token_idx, SWA_INDEX_WIDTH + compressed_len)
        return

    prefill_idx = token_idx - NUM_DECODE_TOKENS
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)
    start_pos = seq_len - query_len
    token_idx_in_query = token_idx - query_start
    pos = start_pos + token_idx_in_query
    swa_start_pos = tl.maximum(pos - (WINDOW_SIZE - 1), 0)
    # SWA bounded replay: no window KV exists below the request's replay start.
    swa_start_pos = tl.maximum(swa_start_pos, tl.load(replay_start_ptr + req_idx))
    swa_len = pos - swa_start_pos + 1
    if COMPRESS_RATIO > 0:
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
    else:
        # SWA-only layers (ratio 0): no indexer topk; div by constexpr 0 is UB.
        topk_len = 0

    for i in range(0, SWA_INDEX_WIDTH, WINDOW_BLOCK_SIZE):
        offset = i + tl.arange(0, WINDOW_BLOCK_SIZE)
        mask = offset < SWA_INDEX_WIDTH
        pos_offset = swa_start_pos + offset
        block_indices = pos_offset // swa_block_size
        block_numbers = tl.load(
            swa_block_table_ptr + req_idx * swa_block_table_stride + block_indices,
            mask=mask & (offset < swa_len),
            other=-1,
        )
        block_offsets = pos_offset % swa_block_size
        slot_ids = block_numbers * swa_block_size + block_offsets
        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        slot_ids = _remap_flashinfer_index(slot_ids, swa_block_size, swa_block_span)
        tl.store(
            sparse_indices_ptr + token_idx * sparse_indices_stride + offset,
            slot_ids,
            mask=mask,
        )

    for i in range(0, PADDED_TOP_K, TOPK_BLOCK_SIZE):
        offset = i + tl.arange(0, TOPK_BLOCK_SIZE)
        mask = offset < PADDED_TOP_K
        local_idx = tl.load(
            prefill_topk_indices_ptr + prefill_idx * prefill_topk_stride + offset,
            mask=(offset < PREFILL_TOPK_STRIDE) & (offset < topk_len),
            other=-1,
        )
        is_valid = local_idx >= 0
        block_indices = local_idx // compressed_block_size
        block_numbers = tl.load(
            compressed_block_table_ptr
            + req_idx * compressed_block_table_stride
            + block_indices,
            mask=mask & is_valid,
            other=-1,
        )
        block_offsets = local_idx % compressed_block_size
        slot_ids = block_numbers * compressed_block_size + block_offsets
        slot_ids = tl.where((offset < topk_len) & is_valid, slot_ids, -1)
        slot_ids = _remap_flashinfer_index(
            slot_ids, compressed_block_size, compressed_block_span
        )
        tl.store(
            sparse_indices_ptr
            + token_idx * sparse_indices_stride
            + SWA_INDEX_WIDTH
            + offset,
            slot_ids,
            mask=mask,
        )

    tl.store(sparse_topk_lens_ptr + token_idx, SWA_INDEX_WIDTH + topk_len)


def build_flashinfer_mixed_sparse_indices(
    decode_swa_indices: torch.Tensor,
    decode_compressed_indices: torch.Tensor | None,
    decode_compressed_topk_lens: torch.Tensor | None,
    prefill_topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    swa_block_table: torch.Tensor,
    swa_block_size: int,
    compressed_block_table: torch.Tensor | None,
    compressed_block_size: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    decode_compressed_indices_are_local: bool = False,
    decode_is_valid_token: torch.Tensor | None = None,
    swa_block_span: int | None = None,
    compressed_block_span: int | None = None,
    *,
    replay_start: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the FlashInfer DSV4 sparse-index matrix for decode-first batches.

    Produces ``sparse_indices`` of shape ``[num_tokens, swa_total_width +
    padded_topk]`` (the first ``swa_total_width`` columns are SWA slot ids, the
    rest are compressed/top-k slot ids) and ``sparse_topk_lens`` (active length
    per token). Decode tokens read precomputed SWA/compressed indices; prefill
    tokens derive their SWA window from the position and translate local
    compressed indices to global slots via the block tables.

    ``replay_start`` ([num_reqs], SWA bounded replay) lower-bounds every
    prefill token's window: positions below it hold no window KV.
    """
    assert decode_swa_indices.dtype == torch.int32
    assert decode_swa_indices.dim() == 2
    swa_total_width = decode_swa_indices.shape[-1]
    assert swa_total_width >= window_size
    if decode_compressed_topk_lens is not None:
        assert decode_compressed_topk_lens.dtype == torch.int32
    assert prefill_topk_indices.dtype == torch.int32
    assert prefill_topk_indices.dim() == 2
    assert query_start_loc.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert token_to_req_indices.dtype == torch.int32
    assert swa_block_table.dtype == torch.int32

    num_decode_tokens = decode_swa_indices.shape[0]
    num_prefill_tokens = prefill_topk_indices.shape[0]
    num_tokens = num_decode_tokens + num_prefill_tokens
    assert token_to_req_indices.shape[0] >= num_tokens
    if decode_compressed_topk_lens is not None:
        assert decode_compressed_topk_lens.shape[0] >= num_decode_tokens

    decode_compressed_topk = 0
    if decode_compressed_indices is None:
        decode_compressed_indices = prefill_topk_indices
    else:
        assert decode_compressed_indices.dtype == torch.int32
        assert decode_compressed_indices.dim() == 2
        assert decode_compressed_indices.shape[0] == num_decode_tokens
        decode_compressed_topk = decode_compressed_indices.shape[-1]
    if decode_compressed_topk > 0 and decode_compressed_indices_are_local:
        assert decode_is_valid_token is not None
        assert decode_is_valid_token.dtype == torch.bool
        assert decode_is_valid_token.shape[0] >= num_decode_tokens
    else:
        decode_is_valid_token = token_to_req_indices

    if compressed_block_table is None:
        compressed_block_table = swa_block_table
    assert compressed_block_table.dtype == torch.int32
    has_decode_compressed_lens = decode_compressed_topk_lens is not None
    if decode_compressed_topk_lens is None:
        decode_compressed_topk_lens = token_to_req_indices

    # The FlashInfer TRTLLM-gen sparse-MLA kernels require every per-token topk
    # index row to start on a 16-byte boundary: the kernel loads the compressed
    # indices with 128-bit (16-byte) vectorized loads, so a misaligned row would
    # fault or read across rows. 16 bytes = 4 int32 indices, so round the topk
    # width (and hence the row stride, since the SWA columns are fixed-width) up
    # to a multiple of 4. The extra columns are filled with -1 (invalid) and bounded
    # by ``sparse_topk_lens``, so padding never changes the attention result.
    padded_topk = max(topk, decode_compressed_topk)
    padded_topk = (padded_topk + 3) // 4 * 4
    sparse_indices = torch.empty(
        (num_tokens, swa_total_width + padded_topk),
        dtype=torch.int32,
        device=decode_swa_indices.device,
    )
    sparse_topk_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=decode_swa_indices.device
    )
    if num_tokens == 0:
        return sparse_indices, sparse_topk_lens

    window_block_size = triton.next_power_of_2(max(swa_total_width, 1))
    topk_block_size = triton.next_power_of_2(max(padded_topk, 1))
    max_block_size = max(window_block_size, topk_block_size)
    num_warps = 4 if max_block_size >= 256 else 1

    # block_span = page_stride / token_stride; == block_size (no-op) for unpacked KV.
    swa_span = swa_block_size if swa_block_span is None else swa_block_span
    compressed_span = (
        compressed_block_size
        if compressed_block_span is None
        else compressed_block_span
    )
    _build_flashinfer_mixed_sparse_indices_kernel[(num_tokens,)](
        sparse_indices,
        sparse_indices.stride(0),
        sparse_topk_lens,
        decode_swa_indices,
        decode_swa_indices.stride(0),
        decode_compressed_indices,
        decode_compressed_indices.stride(0),
        decode_compressed_topk_lens,
        decode_is_valid_token,
        prefill_topk_indices,
        prefill_topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        replay_start,
        swa_block_table,
        swa_block_table.stride(0),
        swa_block_size,
        swa_span,
        compressed_block_table,
        compressed_block_table.stride(0),
        compressed_block_size,
        compressed_span,
        NUM_DECODE_TOKENS=num_decode_tokens,
        WINDOW_SIZE=window_size,
        SWA_INDEX_WIDTH=swa_total_width,
        COMPRESS_RATIO=compress_ratio,
        TOP_K=topk,
        PADDED_TOP_K=padded_topk,
        PREFILL_TOPK_STRIDE=prefill_topk_indices.shape[-1],
        DECODE_COMPRESSED_TOPK=decode_compressed_topk,
        DECODE_COMPRESSED_INDICES_ARE_LOCAL=decode_compressed_indices_are_local,
        HAS_DECODE_COMPRESSED_LENS=has_decode_compressed_lens,
        WINDOW_BLOCK_SIZE=window_block_size,
        TOPK_BLOCK_SIZE=topk_block_size,
        num_warps=num_warps,
    )
    return sparse_indices, sparse_topk_lens


# --- end of verbatim vLLM copy ---------------------------------------------


def make_repo_fwd(vllm_launcher):
    """Adapt the repo kernel-binding signature to the embedded vLLM launcher.

    The functions layer has already normalized optional inputs (dummy tensors,
    spans defaulting to block sizes, replay_start), which the launcher accepts
    unchanged; decode_compressed_topk/has_decode_compressed_lens are re-derived
    inside it exactly as upstream does.
    """

    def build_flashinfer_mixed_sparse_indices_fwd(
        decode_swa_indices,
        decode_compressed_indices,
        decode_compressed_topk_lens,
        decode_is_valid_token,
        prefill_topk_indices,
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        replay_start,
        swa_block_table,
        swa_block_size,
        swa_block_span,
        compressed_block_table,
        compressed_block_size,
        compressed_block_span,
        window_size,
        compress_ratio,
        topk,
        decode_compressed_topk,
        has_decode_compressed_lens,
        decode_compressed_indices_are_local,
    ):
        return vllm_launcher(
            decode_swa_indices=decode_swa_indices,
            decode_compressed_indices=decode_compressed_indices,
            decode_compressed_topk_lens=decode_compressed_topk_lens,
            prefill_topk_indices=prefill_topk_indices,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            token_to_req_indices=token_to_req_indices,
            swa_block_table=swa_block_table,
            swa_block_size=swa_block_size,
            compressed_block_table=compressed_block_table,
            compressed_block_size=compressed_block_size,
            window_size=window_size,
            compress_ratio=compress_ratio,
            topk=topk,
            decode_compressed_indices_are_local=decode_compressed_indices_are_local,
            decode_is_valid_token=decode_is_valid_token,
            swa_block_span=swa_block_span,
            compressed_block_span=compressed_block_span,
            replay_start=replay_start,
        )

    return build_flashinfer_mixed_sparse_indices_fwd


def apply_patches():
    """Patch dispatch and the implementation check in memory; idempotent.

    Returns (dispatch, repo_fwd). In plugin mode this runs at module import
    (pytest's -p loads the module before conftests), so conftest-level
    ``from ... import resolve_implementation`` bindings capture the patches.
    """
    global _PATCH_STATE
    if _PATCH_STATE is not None:
        return _PATCH_STATE

    import mojo_opset.functions._dispatch as dispatch

    fwd = make_repo_fwd(build_flashinfer_mixed_sparse_indices)

    # (1) configuration lookup: the op resolves to triton everywhere
    original_resolve = dispatch.resolve_implementation

    def resolve_implementation(op_id, target, snapshot=None):
        if op_id == OP_ID:
            return "triton"
        return original_resolve(op_id, target, snapshot)

    dispatch.resolve_implementation = resolve_implementation

    # (2) provider binding: redirect the op to the embedded vLLM implementation
    original_binding = dispatch._load_kernel_binding

    def _load_kernel_binding(provider_package, target_key, op_id):
        if op_id == OP_ID:
            return fwd, None
        return original_binding(provider_package, target_key, op_id)

    dispatch._load_kernel_binding = _load_kernel_binding

    # (3) performance-harness gate: the patched-in binding is not declared by
    # the ilu provider, let it through
    import tests._fixtures as fixtures
    from mojo_opset import Target

    original_check = fixtures.check_implementation

    def check_implementation(op_ids, implementation, target):
        if implementation is None and set(op_ids) == {OP_ID} and Target.parse(target).arch_key == "ilu.generic":
            return
        return original_check(op_ids, implementation, target)

    fixtures.check_implementation = check_implementation

    # (4) profiler-trace analysis: drop CUDA's same-named GPU-timeline mirrors
    # of the sample ranges so duplicate-range detection sees each sample once.
    import tests.performance.timing as timing

    original_spans = timing.kernel_spans

    @functools.wraps(original_spans)
    def kernel_spans(trace, is_kernel, repeats, **kwargs):
        events = trace if isinstance(trace, list) else trace["traceEvents"]
        if any(isinstance(e, dict) and e.get("cat") == "gpu_user_annotation" for e in events):
            kept = [
                e
                for e in events
                if not (isinstance(e, dict) and e.get("cat") == "gpu_user_annotation")
            ]
            trace = kept if isinstance(trace, list) else {"traceEvents": kept}
        return original_spans(trace, is_kernel, repeats, **kwargs)

    timing.kernel_spans = kernel_spans

    # (5) CUDA L2 flush: --perf-l2-flush is gated on the adapter exposing
    # make_l2_flush (only npu ships one); mirror the npu buffer-sum eviction
    # so cold-cache measurement works on CUDA. A 256 MiB read exceeds every
    # current CUDA L2 (H100 50 MiB, B200 126 MiB), matching triton's do_bench
    # eviction-size convention. Inert unless the ilu adapter is actually used.
    import tests.performance._adapters.ilu as ilu_adapter

    def make_l2_flush():
        buffer = torch.empty(
            ilu_adapter.L2_FLUSH_BYTES // 4,
            dtype=torch.float32,
            device=f"cuda:{torch.cuda.current_device()}",
        )

        def flush():
            buffer.sum()  # a full-buffer read replaces the cached lines

        flush()
        torch.cuda.synchronize()  # shake out any cuda error before measurement
        return flush

    ilu_adapter.make_l2_flush = make_l2_flush
    ilu_adapter.L2_FLUSH_METHOD = "buffer_sum_read_v1"
    ilu_adapter.L2_FLUSH_BYTES = 256 * 1024 * 1024
    assert_no_vllm_imported()  # patches + repo/test imports pulled in no vllm
    _PATCH_STATE = (dispatch, fwd)
    return _PATCH_STATE


# Bind at import: direct mode needs the patches before its smoke call, and
# plugin mode (-p <module>) needs them before pytest imports any conftest.
_PATCH_STATE = None
apply_patches()


def pytest_unconfigure(config):
    # Plugin-mode session guard: patches, harness and kernel JIT must finish
    # the whole run without ever loading the vllm package.
    assert_no_vllm_imported()


def verify_patches():
    from mojo_opset import Target
    from mojo_opset.functions import _dispatch

    dispatch, fwd = apply_patches()
    assert _dispatch.resolve_implementation(OP_ID, Target.parse("ilu.generic")) == "triton"
    assert _dispatch.resolve_implementation("rms_norm_infer", Target.parse("ilu.generic")) == "triton"  # untouched op
    bound, backward = dispatch._load_kernel_binding("mojo_opset.kernels.ilu_triton", "ilu.generic", OP_ID)
    assert bound is fwd and backward is None
    import inspect

    expected = 21  # repo kernel-binding signature parameter count
    assert len(inspect.signature(fwd).parameters) == expected
    import tests._fixtures as fixtures

    fixtures.check_implementation({OP_ID}, None, "ilu.generic")  # must not raise or skip

    # the embedded kernel must be a triton JITFunction (a real copy, not a stub)
    kernel = _build_flashinfer_mixed_sparse_indices_kernel
    assert hasattr(kernel, "run") and hasattr(kernel, "__getitem__"), "kernel lost its @triton.jit decorator"

    # duplicate-range fix: CUDA traces mirror each sample range onto the GPU
    # timeline under the same name; the analysis must ignore the mirrors.
    import tests.performance.timing as timing

    mirrored = [
        {"ph": "X", "cat": "user_annotation", "name": "mojo_perf_sample_0", "ts": 1000, "dur": 100},
        {"ph": "X", "cat": "kernel", "name": "triton_kernel", "ts": 1020, "dur": 40},
        {"ph": "X", "cat": "gpu_user_annotation", "name": "mojo_perf_sample_0", "ts": 1020, "dur": 40},
        {"ph": "X", "cat": "user_annotation", "name": "mojo_perf_sample_1", "ts": 2000, "dur": 100},
        {"ph": "X", "cat": "kernel", "name": "triton_kernel", "ts": 2010, "dur": 30},
        {"ph": "X", "cat": "gpu_user_annotation", "name": "mojo_perf_sample_1", "ts": 2010, "dur": 30},
    ]
    is_kernel = lambda e: e.get("cat") == "kernel"
    try:
        timing.kernel_spans.__wrapped__(mirrored, is_kernel, 2)
        raise AssertionError("unpatched analysis should reject the mirrored trace")
    except ValueError as error:
        assert "Duplicate profiler range" in str(error), error
    assert timing.kernel_spans(mirrored, is_kernel, 2) == [40.0, 30.0]

    # CUDA L2 flush wiring (mirrors the npu adapter's buffer-sum eviction)
    import tests.performance._adapters.ilu as ilu_adapter

    assert callable(ilu_adapter.make_l2_flush)
    assert ilu_adapter.L2_FLUSH_METHOD == "buffer_sum_read_v1" and ilu_adapter.L2_FLUSH_BYTES > 0
    if torch.cuda.is_available():
        ilu_adapter.make_l2_flush()  # exercise allocation + prime + sync once
        torch.cuda.synchronize()

    assert_no_vllm_imported()
    print("patch wiring verified (vLLM original embedded verbatim):",
          "CUDA available" if torch.cuda.is_available() else "no CUDA device (kernel launch would fail)")
    return 0


def main(argv):
    if "--verify-patches" in argv:
        return verify_patches()

    if not torch.cuda.is_available():
        print("error: a CUDA device is required", file=sys.stderr)
        return 2

    dispatch, fwd = apply_patches()

    # Plain-NVIDIA boxes detect as cpu.generic; pin the process target so the
    # ilu triton provider (and our binding) is selected. The pytest benchmark
    # fixture re-pins this per test via --mojo-target ilu.generic.
    from mojo_opset import Target
    from mojo_opset.utils import target as target_utils

    target_utils._AUTO_TARGET = Target.parse("ilu.generic")

    # Fail fast with a direct smoke call before the harness starts; compare
    # against the device-agnostic torch reference.
    from mojo_opset import functions
    from tests.functions.test_build_flashinfer_mixed_sparse_indices import make_mixed_sparse_case

    device = f"cuda:{torch.cuda.current_device()}"
    case = make_mixed_sparse_case(device, seed=0,
        decode_query_lens=[1] * 8, prefill_query_lens=[64, 32],
        seq_lens=[256] * 8 + [512, 384], compress_ratio=1, topk=512,
        window_size=128, swa_block_size=64)
    indices, lens = functions.build_flashinfer_mixed_sparse_indices(**case, implementation="triton")
    ref_indices, ref_lens = functions.build_flashinfer_mixed_sparse_indices(**case, implementation="torch")
    torch.cuda.synchronize()
    assert torch.equal(indices, ref_indices) and torch.equal(lens, ref_lens), "vllm-on-cuda smoke mismatch"
    assert_no_vllm_imported()  # even the triton JIT compile pulled in no vllm
    print(f"smoke ok on {torch.cuda.get_device_name()}: {tuple(indices.shape)} int32")

    import pytest

    args = [
        PERF_TEST,
        "--mojo-target", "ilu.generic",
        "--perf-l2-flush",
        "--perf-output", DEFAULT_OUTPUT,
        *[a for a in argv if a != "--verify-patches"],
    ]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
