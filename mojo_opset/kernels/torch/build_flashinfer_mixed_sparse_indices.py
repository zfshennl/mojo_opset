"""Reference FlashInfer DSV4.1 mixed sparse-index build composed of torch ops."""

import torch


def _remap_flashinfer_index(values: torch.Tensor, block_size: int, block_span: int) -> torch.Tensor:
    """FlashInfer's DSv4 kernel indexes sparse KV by physical token stride, so
    packed pages need block*block_size+off -> block*block_span+off. The -1
    invalid sentinel passes through unchanged."""
    is_valid = values >= 0
    safe = torch.where(is_valid, values, torch.zeros_like(values))
    remapped = (safe // block_size) * block_span + torch.remainder(safe, block_size)
    return torch.where(is_valid, remapped, torch.full_like(remapped, -1))


def _translate_local_indices(local, block_table, req_idx, block_size, block_span):
    """Local compressed coordinates -> global paged slot ids (or -1)."""
    is_valid = local >= 0
    safe_blocks = torch.where(is_valid, local // block_size, torch.zeros_like(local))
    block_numbers = block_table[req_idx[:, None], safe_blocks.long()]
    slots = torch.where(
        is_valid,
        block_numbers * block_size + torch.remainder(local, block_size),
        torch.full_like(local, -1),
    )
    return _remap_flashinfer_index(slots, block_size, block_span), is_valid


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
) -> tuple[torch.Tensor, torch.Tensor]:
    device = decode_swa_indices.device
    swa_total_width = decode_swa_indices.shape[-1]
    num_decode_tokens = decode_swa_indices.shape[0]
    num_prefill_tokens = prefill_topk_indices.shape[0]
    num_tokens = num_decode_tokens + num_prefill_tokens

    # The FlashInfer TRTLLM-gen sparse-MLA kernels load the compressed indices
    # with 128-bit vectorized loads, so every topk row must start 16-byte
    # aligned: round the width up to a multiple of 4 int32 columns and pad -1.
    padded_topk = max(topk, decode_compressed_topk)
    padded_topk = (padded_topk + 3) // 4 * 4

    sparse_indices = torch.empty((num_tokens, swa_total_width + padded_topk), dtype=torch.int32, device=device)
    sparse_topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=device)
    if num_tokens == 0:
        return sparse_indices, sparse_topk_lens

    cols = torch.arange(padded_topk, device=device, dtype=torch.int32)

    if num_decode_tokens > 0:
        # SWA columns: copy the precomputed decode slot ids, remapping packed pages.
        sparse_indices[:num_decode_tokens, :swa_total_width] = _remap_flashinfer_index(
            decode_swa_indices, swa_block_size, swa_block_span
        )

        # Top-k columns; entries past the decode width stay -1.
        values = torch.full((num_decode_tokens, padded_topk), -1, dtype=torch.int32, device=device)
        width = min(decode_compressed_topk, decode_compressed_indices.shape[-1])
        values[:, :width] = decode_compressed_indices[:, :width]

        if decode_compressed_indices_are_local:
            slots, is_valid = _translate_local_indices(
                values,
                compressed_block_table,
                token_to_req_indices[:num_decode_tokens].long(),
                compressed_block_size,
                compressed_block_span,
            )
            token_valid = decode_is_valid_token[:num_decode_tokens].to(torch.bool)
            compressed_len = (is_valid & token_valid[:, None]).sum(dim=1, dtype=torch.int32)
        else:
            slots = _remap_flashinfer_index(values, compressed_block_size, compressed_block_span)
            if decode_compressed_topk == 0:
                compressed_len = torch.zeros(num_decode_tokens, dtype=torch.int32, device=device)
            elif has_decode_compressed_lens:
                compressed_len = decode_compressed_topk_lens[:num_decode_tokens].to(torch.int32)
            else:
                compressed_len = torch.full(
                    (num_decode_tokens,), decode_compressed_topk, dtype=torch.int32, device=device
                )

        sparse_indices[:num_decode_tokens, swa_total_width:] = slots
        sparse_topk_lens[:num_decode_tokens] = swa_total_width + compressed_len

    if num_prefill_tokens > 0:
        token_idx = torch.arange(num_decode_tokens, num_tokens, device=device, dtype=torch.int32)
        req_idx = token_to_req_indices[token_idx].long()
        query_start = query_start_loc[req_idx]
        query_len = query_start_loc[req_idx + 1] - query_start
        seq_len = seq_lens[req_idx]
        pos = (seq_len - query_len) + (token_idx - query_start)
        swa_start_pos = torch.clamp(pos - (window_size - 1), min=0)
        # SWA bounded replay: no window KV exists below the request's replay start.
        swa_start_pos = torch.maximum(swa_start_pos, replay_start[req_idx])
        swa_len = pos - swa_start_pos + 1
        if compress_ratio > 0:
            topk_len = torch.clamp((pos + 1) // compress_ratio, max=topk)
        else:
            # SWA-only layers (ratio 0): no indexer topk.
            topk_len = torch.zeros(num_prefill_tokens, dtype=torch.int32, device=device)

        # SWA columns: derive the token's window from its position via the block table.
        window_cols = torch.arange(swa_total_width, device=device, dtype=torch.int32)
        in_window = window_cols[None, :] < swa_len[:, None]
        pos_offset = swa_start_pos[:, None] + window_cols[None, :]
        safe_blocks = torch.where(in_window, pos_offset // swa_block_size, torch.zeros_like(pos_offset))
        block_numbers = swa_block_table[req_idx[:, None], safe_blocks.long()]
        slot_ids = torch.where(
            in_window,
            block_numbers * swa_block_size + torch.remainder(pos_offset, swa_block_size),
            torch.full_like(pos_offset, -1),
        )
        sparse_indices[num_decode_tokens:, :swa_total_width] = _remap_flashinfer_index(
            slot_ids, swa_block_size, swa_block_span
        )

        # Top-k columns: translate the local compressed top-k indices to slots.
        stride = prefill_topk_indices.shape[-1]
        local = torch.full((num_prefill_tokens, padded_topk), -1, dtype=torch.int32, device=device)
        width = min(stride, padded_topk)
        local[:, :width] = prefill_topk_indices[:, :width]
        read = (cols[None, :] < stride) & (cols[None, :] < topk_len[:, None])
        local = torch.where(read, local, torch.full_like(local, -1))
        slots, _ = _translate_local_indices(
            local, compressed_block_table, req_idx, compressed_block_size, compressed_block_span
        )
        sparse_indices[num_decode_tokens:, swa_total_width:] = slots
        sparse_topk_lens[num_decode_tokens:] = (swa_total_width + topk_len).to(torch.int32)

    return sparse_indices, sparse_topk_lens
