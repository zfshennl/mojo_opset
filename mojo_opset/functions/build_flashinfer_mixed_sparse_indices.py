"""FlashInfer DSV4.1 mixed sparse-index matrix construction."""

from typing import Optional
from typing import Tuple

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def build_flashinfer_mixed_sparse_indices(
    decode_swa_indices: torch.Tensor,
    decode_compressed_indices: Optional[torch.Tensor],
    decode_compressed_topk_lens: Optional[torch.Tensor],
    prefill_topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    swa_block_table: torch.Tensor,
    swa_block_size: int,
    compressed_block_table: Optional[torch.Tensor],
    compressed_block_size: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    decode_compressed_indices_are_local: bool = False,
    decode_is_valid_token: Optional[torch.Tensor] = None,
    swa_block_span: Optional[int] = None,
    compressed_block_span: Optional[int] = None,
    *,
    replay_start: Optional[torch.Tensor] = None,
    implementation: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the FlashInfer DSV4 sparse-index matrix for decode-first batches.

    Produces ``sparse_indices`` of shape ``[num_tokens, swa_total_width +
    padded_topk]`` (the first ``swa_total_width`` columns are SWA slot ids, the
    rest are compressed/top-k slot ids) and ``sparse_topk_lens`` (active length
    per token). Decode tokens read precomputed SWA/compressed indices; prefill
    tokens derive their SWA window from the position and translate local
    compressed indices to global slots via the block tables.

    ``replay_start`` ([num_reqs], SWA bounded replay) lower-bounds every
    prefill token's window: positions below it hold no window KV. ``None``
    substitutes zeros, matching vLLM's graph-capture metadata builder.
    """

    if decode_swa_indices.dtype != torch.int32 or decode_swa_indices.dim() != 2:
        raise ValueError("decode_swa_indices must be a 2-D int32 tensor.")
    swa_total_width = decode_swa_indices.shape[-1]
    if swa_total_width < window_size:
        raise ValueError(f"decode_swa_indices width {swa_total_width} < window_size {window_size}.")
    if decode_compressed_topk_lens is not None and decode_compressed_topk_lens.dtype != torch.int32:
        raise ValueError("decode_compressed_topk_lens must be int32.")
    if prefill_topk_indices.dtype != torch.int32 or prefill_topk_indices.dim() != 2:
        raise ValueError("prefill_topk_indices must be a 2-D int32 tensor.")
    for name, tensor in (
        ("query_start_loc", query_start_loc),
        ("seq_lens", seq_lens),
        ("token_to_req_indices", token_to_req_indices),
        ("swa_block_table", swa_block_table),
    ):
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32.")
    if query_start_loc.shape[0] < seq_lens.shape[0] + 1:
        raise ValueError("query_start_loc needs num_reqs + 1 entries.")

    num_decode_tokens = decode_swa_indices.shape[0]
    num_prefill_tokens = prefill_topk_indices.shape[0]
    num_tokens = num_decode_tokens + num_prefill_tokens
    if token_to_req_indices.shape[0] < num_tokens:
        raise ValueError("token_to_req_indices must cover every batched token.")
    if decode_compressed_topk_lens is not None and decode_compressed_topk_lens.shape[0] < num_decode_tokens:
        raise ValueError("decode_compressed_topk_lens must cover every decode token.")

    decode_compressed_topk = 0
    if decode_compressed_indices is None:
        # Fully masked loads: an empty dummy keeps the kernel specialization
        # shared with batches that do carry decode indices.
        decode_compressed_indices = prefill_topk_indices
    else:
        if decode_compressed_indices.dtype != torch.int32 or decode_compressed_indices.dim() != 2:
            raise ValueError("decode_compressed_indices must be a 2-D int32 tensor.")
        if decode_compressed_indices.shape[0] != num_decode_tokens:
            raise ValueError("decode_compressed_indices must have one row per decode token.")
        decode_compressed_topk = decode_compressed_indices.shape[-1]
    if decode_compressed_topk > 0 and decode_compressed_indices_are_local:
        if decode_is_valid_token is None:
            raise ValueError("local decode_compressed_indices require decode_is_valid_token.")
        if decode_is_valid_token.dtype != torch.bool or decode_is_valid_token.shape[0] < num_decode_tokens:
            raise ValueError("decode_is_valid_token must be a bool tensor covering every decode token.")
    else:
        decode_is_valid_token = token_to_req_indices

    if compressed_block_table is None:
        compressed_block_table = swa_block_table
    if compressed_block_table.dtype != torch.int32:
        raise ValueError("compressed_block_table must be int32.")
    has_decode_compressed_lens = decode_compressed_topk_lens is not None
    if decode_compressed_topk_lens is None:
        decode_compressed_topk_lens = token_to_req_indices
    if replay_start is None:
        replay_start = torch.zeros(seq_lens.shape[0], dtype=torch.int32, device=decode_swa_indices.device)
    elif replay_start.dtype != torch.int32:
        raise ValueError("replay_start must be int32.")
    # block_span = page_stride / token_stride; == block_size (no-op) for unpacked KV.
    swa_span = swa_block_size if swa_block_span is None else swa_block_span
    compressed_span = compressed_block_size if compressed_block_span is None else compressed_block_span

    _require_inference(
        "build_flashinfer_mixed_sparse_indices",
        decode_swa_indices,
        decode_compressed_indices,
        decode_compressed_topk_lens,
        decode_is_valid_token,
        prefill_topk_indices,
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        swa_block_table,
        compressed_block_table,
        replay_start,
    )
    forward, _ = load_impl("build_flashinfer_mixed_sparse_indices", implementation)
    return forward(
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
        swa_span,
        compressed_block_table,
        compressed_block_size,
        compressed_span,
        window_size,
        compress_ratio,
        topk,
        decode_compressed_topk,
        has_decode_compressed_lens,
        decode_compressed_indices_are_local,
    )
