from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry

"""
FlashInfer DSV4.1 mixed sparse-index build for DeepSeek-V4.1 sparse MLA on NPU:
one row per token; SWA slot ids first, then compressed/top-k slot ids, plus a
per-token active length. Decode tokens copy precomputed indices; prefill tokens
derive their SWA window from the position and translate local compressed
indices to global slots via the block tables.

Reference: vLLM build_flashinfer_mixed_sparse_indices
(vllm/models/deepseek_v41/common/ops/cache_utils.py).

Modifications for NPU architecture by triton-x team.
"""

UB_BYTES = 196608
UB_RESERVE_BYTES = 1024
# Live int32 lanes per column in the widest tile: values + is_valid +
# block_indices + block_numbers + block_offsets + slot_ids.
_WORKING_LANES = 6


def _ub_capped_block(width: int) -> int:
    """Largest power-of-two tile whose int32 working set fits one AIV's UB."""
    budget = (UB_BYTES - UB_RESERVE_BYTES) // (_WORKING_LANES * 4)
    block = triton.next_power_of_2(max(width, 1))
    while block > budget and block > 16:
        block //= 2
    return block


@triton.jit
def _remap_flashinfer_index(values, block_size, block_span):
    # FlashInfer's DSv4 kernel indexes sparse KV by physical token stride, so
    # packed pages (#44577) need block*block_size+off -> block*block_span+off.
    is_valid = values >= 0
    safe_values = tl.where(is_valid, values, 0)
    values = (safe_values // block_size) * block_span
    values += safe_values % block_size
    return tl.where(is_valid, values, -1)


@triton.jit
def _maybe_remap_flashinfer_index(values, block_size: tl.constexpr, block_span: tl.constexpr,
                                  do_remap: tl.constexpr):
    # span == block_size makes the remap an identity; skipping it saves a full
    # div+mod per element (13% of the kernel on 957b, see the perf report).
    if do_remap:
        values = _remap_flashinfer_index(values, block_size, block_span)
    return values


@libentry()
@triton.jit(
    do_not_specialize=[
        "sparse_indices_stride",
        "decode_swa_stride",
        "decode_compressed_stride",
        "prefill_topk_stride",
        "swa_block_table_stride",
        "compressed_block_table_stride",
        "num_tokens",
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
    swa_block_size: tl.constexpr,
    swa_block_span: tl.constexpr,
    compressed_block_table_ptr,
    compressed_block_table_stride,
    compressed_block_size: tl.constexpr,
    compressed_block_span: tl.constexpr,
    num_tokens,
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
    SWA_REMAP: tl.constexpr,
    COMPRESSED_REMAP: tl.constexpr,
    SWA_SCALAR_LOOKUP: tl.constexpr,
    SWA_BLOCK_SPAN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    # Persistent programs: the AIV count stays fixed while token counts vary per step.
    for token_idx in range(pid, num_tokens, grid_size):
        if token_idx < NUM_DECODE_TOKENS:
            for i in range(0, SWA_INDEX_WIDTH, WINDOW_BLOCK_SIZE):
                offset = i + tl.arange(0, WINDOW_BLOCK_SIZE)
                mask = offset < SWA_INDEX_WIDTH
                values = tl.load(
                    decode_swa_indices_ptr + token_idx * decode_swa_stride + offset,
                    mask=mask,
                    other=-1,
                )
                values = _maybe_remap_flashinfer_index(values, swa_block_size, swa_block_span, SWA_REMAP)
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
                    decode_compressed_indices_ptr + token_idx * decode_compressed_stride + offset,
                    mask=offset < DECODE_COMPRESSED_TOPK,
                    other=-1,
                )
                if DECODE_COMPRESSED_INDICES_ARE_LOCAL:
                    token_valid = tl.load(decode_is_valid_token_ptr + token_idx).to(tl.int1)
                    is_valid = values >= 0
                    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
                    block_indices = values // compressed_block_size
                    block_numbers = tl.load(
                        compressed_block_table_ptr + req_idx * compressed_block_table_stride + block_indices,
                        mask=mask & is_valid,
                        other=-1,
                    )
                    block_offsets = values % compressed_block_size
                    values = block_numbers * compressed_block_size + block_offsets
                    values = tl.where(is_valid, values, -1)
                    compressed_len += tl.sum((is_valid & token_valid).to(tl.int32), axis=0)
                values = _maybe_remap_flashinfer_index(
                    values, compressed_block_size, compressed_block_span, COMPRESSED_REMAP
                )
                tl.store(
                    sparse_indices_ptr + token_idx * sparse_indices_stride + SWA_INDEX_WIDTH + offset,
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
        else:
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
                if SWA_SCALAR_LOOKUP:
                    # A window tile touches at most SWA_BLOCK_SPAN distinct
                    # blocks: that many masked scalar loads on the SIMT path
                    # beat a WINDOW_BLOCK_SIZE-lane indirect load (~27% of the
                    # kernel on 957b, see the perf report).
                    first_block = swa_start_pos // swa_block_size
                    last_block = pos // swa_block_size
                    block_numbers = tl.zeros((WINDOW_BLOCK_SIZE,), dtype=tl.int32)
                    for b in tl.static_range(0, SWA_BLOCK_SPAN):
                        valid = (first_block + b) <= last_block
                        val = tl.load(
                            swa_block_table_ptr + req_idx * swa_block_table_stride + first_block + b,
                            mask=valid,
                            other=0,
                        )
                        block_numbers += tl.where(block_indices == (first_block + b), val, 0)
                else:
                    block_numbers = tl.load(
                        swa_block_table_ptr + req_idx * swa_block_table_stride + block_indices,
                        mask=mask & (offset < swa_len),
                        other=-1,
                    )
                block_offsets = pos_offset % swa_block_size
                slot_ids = block_numbers * swa_block_size + block_offsets
                slot_ids = tl.where(offset < swa_len, slot_ids, -1)
                slot_ids = _maybe_remap_flashinfer_index(slot_ids, swa_block_size, swa_block_span, SWA_REMAP)
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
                    compressed_block_table_ptr + req_idx * compressed_block_table_stride + block_indices,
                    mask=mask & is_valid,
                    other=-1,
                )
                block_offsets = local_idx % compressed_block_size
                slot_ids = block_numbers * compressed_block_size + block_offsets
                slot_ids = tl.where((offset < topk_len) & is_valid, slot_ids, -1)
                slot_ids = _maybe_remap_flashinfer_index(
                    slot_ids, compressed_block_size, compressed_block_span, COMPRESSED_REMAP
                )
                tl.store(
                    sparse_indices_ptr + token_idx * sparse_indices_stride + SWA_INDEX_WIDTH + offset,
                    slot_ids,
                    mask=mask,
                )

            tl.store(sparse_topk_lens_ptr + token_idx, SWA_INDEX_WIDTH + topk_len)


@torch.library.custom_op("mojo_npu_triton_a2::build_flashinfer_mixed_sparse_indices_fwd", mutates_args=())
def build_flashinfer_mixed_sparse_indices_fwd(
    decode_swa_indices: torch.Tensor,
    decode_compressed_indices: torch.Tensor,
    decode_compressed_topk_lens: torch.Tensor,
    decode_is_valid_token: torch.Tensor,
    prefill_topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    replay_start: torch.Tensor,
    swa_block_table: torch.Tensor,
    swa_block_size: int,
    swa_block_span: int,
    compressed_block_table: torch.Tensor,
    compressed_block_size: int,
    compressed_block_span: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    decode_compressed_topk: int,
    has_decode_compressed_lens: bool,
    decode_compressed_indices_are_local: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    swa_total_width = decode_swa_indices.shape[-1]
    num_decode_tokens = decode_swa_indices.shape[0]
    num_tokens = num_decode_tokens + prefill_topk_indices.shape[0]

    # The FlashInfer TRTLLM-gen sparse-MLA kernels require every per-token topk
    # index row to start on a 16-byte boundary: the kernel loads the compressed
    # indices with 128-bit (16-byte) vectorized loads. 16 bytes = 4 int32
    # indices, so round the topk width up to a multiple of 4. The extra columns
    # are filled with -1 (invalid) and bounded by ``sparse_topk_lens``.
    padded_topk = max(topk, decode_compressed_topk)
    padded_topk = (padded_topk + 3) // 4 * 4

    sparse_indices = torch.empty(
        (num_tokens, swa_total_width + padded_topk), dtype=torch.int32, device=decode_swa_indices.device
    )
    sparse_topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=decode_swa_indices.device)
    if num_tokens == 0:
        return sparse_indices, sparse_topk_lens

    window_block_size = _ub_capped_block(swa_total_width)
    topk_block_size = _ub_capped_block(padded_topk)
    # Distinct SWA blocks touched by one window tile; scalar lookups only pay
    # off while this stays small (8 is conservative, DSV4.1 needs 3-5).
    swa_block_span_tiles = window_block_size // swa_block_size + 2

    _build_flashinfer_mixed_sparse_indices_kernel[(get_num_cores("vector"),)](
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
        swa_block_span,
        compressed_block_table,
        compressed_block_table.stride(0),
        compressed_block_size,
        compressed_block_span,
        num_tokens,
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
        SWA_REMAP=swa_block_span != swa_block_size,
        COMPRESSED_REMAP=compressed_block_span != compressed_block_size,
        SWA_SCALAR_LOOKUP=swa_block_span_tiles <= 8,
        SWA_BLOCK_SPAN=swa_block_span_tiles,
    )
    return sparse_indices, sparse_topk_lens


@build_flashinfer_mixed_sparse_indices_fwd.register_fake
def _build_flashinfer_mixed_sparse_indices_fwd_fake(
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
    num_tokens = decode_swa_indices.shape[0] + prefill_topk_indices.shape[0]
    padded_topk = max(topk, decode_compressed_topk)
    padded_topk = (padded_topk + 3) // 4 * 4
    return (
        torch.empty(
            (num_tokens, decode_swa_indices.shape[-1] + padded_topk),
            dtype=torch.int32,
            device=decode_swa_indices.device,
        ),
        torch.empty(num_tokens, dtype=torch.int32, device=decode_swa_indices.device),
    )
