"""Tests for functions.build_flashinfer_mixed_sparse_indices."""

from functools import partial
from itertools import accumulate
from typing import Optional
from typing import Sequence

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def make_mixed_sparse_case(
    device,
    *,
    decode_query_lens: Sequence[int] = (),
    prefill_query_lens: Sequence[int] = (),
    seq_lens: Sequence[int],
    window_size: int,
    compress_ratio: int,
    topk: int,
    swa_block_size: int = 64,
    compressed_block_size: Optional[int] = None,
    swa_width: Optional[int] = None,
    decode_indices: str = "local",
    decode_topk_width: Optional[int] = None,
    invalid_decode_fraction: float = 0.0,
    decode_is_valid: Optional[Sequence[bool]] = None,
    replay_starts: Optional[Sequence[int]] = None,
    swa_block_span: Optional[int] = None,
    compressed_block_span: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Build a full kwarg set for ``functions.build_flashinfer_mixed_sparse_indices``.

    Decode requests come first (decode-first batches), then prefill requests.
    ``decode_indices`` selects the flavor: "local" (V4.1 topk_indices_buffer),
    "global_lens"/"global_const" (V4 C128A global decode indices with or without
    per-token lens), or "none" (SWA-only layers).
    """
    decode_query_lens = list(decode_query_lens)
    prefill_query_lens = list(prefill_query_lens)
    seq_lens = list(seq_lens)
    num_reqs = len(seq_lens)
    if num_reqs != len(decode_query_lens) + len(prefill_query_lens):
        raise ValueError("seq_lens must cover decode requests followed by prefill requests")
    query_lens = decode_query_lens + prefill_query_lens
    num_decode_tokens = sum(decode_query_lens)
    num_prefill_tokens = sum(prefill_query_lens)
    swa_width = window_size if swa_width is None else swa_width
    if compressed_block_size is None:
        compressed_block_size = swa_block_size if compress_ratio <= 1 else swa_block_size // compress_ratio
    generator = torch.Generator().manual_seed(seed)

    query_start_loc = torch.tensor(list(accumulate(query_lens, initial=0)), dtype=torch.int32)
    token_to_req = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32), torch.tensor(query_lens, dtype=torch.int64)
    )

    # (req_idx, pos) for every token, decode requests first.
    token_pos = []
    for req, (seq_len, query_len) in enumerate(zip(seq_lens, query_lens)):
        start = seq_len - query_len
        token_pos.extend((req, start + offset) for offset in range(query_len))

    max_seq = max(seq_lens, default=0)
    swa_blocks = max(1, _ceil_div(max_seq, swa_block_size))
    swa_table = torch.randperm(num_reqs * swa_blocks, generator=generator).reshape(num_reqs, swa_blocks).to(torch.int32)

    def swa_slot(req, pos):
        return int(swa_table[req, pos // swa_block_size]) * swa_block_size + pos % swa_block_size

    # Decode SWA indices: the window slot ids the sparse-SWA metadata precomputed.
    decode_swa = torch.full((num_decode_tokens, swa_width), -1, dtype=torch.int32)
    for token in range(num_decode_tokens):
        req, pos = token_pos[token]
        start = max(pos - (window_size - 1), 0)
        for column, p in enumerate(range(start, pos + 1)):
            decode_swa[token, column] = swa_slot(req, p)

    compressed_counts = [_ceil_div(seq, compress_ratio) for seq in seq_lens] if compress_ratio > 0 else [0] * num_reqs
    if compress_ratio > 0:
        compressed_blocks = max(1, _ceil_div(max(compressed_counts, default=0), compressed_block_size))
        compressed_table = (
            torch.randperm(num_reqs * compressed_blocks, generator=generator)
            .reshape(num_reqs, compressed_blocks)
            .to(torch.int32)
        )
    else:
        compressed_table = None

    def sample_locals(req, count):
        if count <= 0:
            return []
        return torch.randperm(compressed_counts[req], generator=generator)[:count].sort().values.tolist()

    are_local = decode_indices == "local"
    decode_compressed = None
    decode_topk_lens = None
    decode_valid = None
    if decode_indices != "none":
        width = decode_topk_width if decode_topk_width is not None else topk
        decode_compressed = torch.full((num_decode_tokens, width), -1, dtype=torch.int32)
        if decode_indices == "global_lens":
            decode_topk_lens = torch.zeros(num_decode_tokens, dtype=torch.int32)
        for token in range(num_decode_tokens):
            req, pos = token_pos[token]
            count = min((pos + 1) // compress_ratio, width) if compress_ratio > 0 else 0
            values = sample_locals(req, count)
            if invalid_decode_fraction > 0:
                values = [
                    -1 if torch.rand(1, generator=generator).item() < invalid_decode_fraction else value
                    for value in values
                ]
            for column, value in enumerate(values):
                if value < 0:
                    continue
                if are_local:
                    decode_compressed[token, column] = value
                else:
                    decode_compressed[token, column] = (
                        int(compressed_table[req, value // compressed_block_size]) * compressed_block_size
                        + value % compressed_block_size
                    )
            if decode_indices == "global_lens":
                decode_topk_lens[token] = len([value for value in values if value >= 0])
        if are_local:
            if decode_is_valid is not None and len(decode_is_valid) != num_decode_tokens:
                raise ValueError("decode_is_valid must have one entry per decode token")
            decode_valid = torch.tensor(
                decode_is_valid if decode_is_valid is not None else [True] * num_decode_tokens, dtype=torch.bool
            )

    prefill_topk = torch.full((num_prefill_tokens, topk), -1, dtype=torch.int32)
    for prefill_idx in range(num_prefill_tokens):
        req, pos = token_pos[num_decode_tokens + prefill_idx]
        count = min((pos + 1) // compress_ratio, topk) if compress_ratio > 0 else 0
        picks = sample_locals(req, count)
        prefill_topk[prefill_idx, : len(picks)] = torch.tensor(picks, dtype=torch.int32)

    replay = torch.tensor(replay_starts if replay_starts is not None else [0] * num_reqs, dtype=torch.int32)

    return {
        "decode_swa_indices": decode_swa.to(device),
        "decode_compressed_indices": None if decode_compressed is None else decode_compressed.to(device),
        "decode_compressed_topk_lens": None if decode_topk_lens is None else decode_topk_lens.to(device),
        "prefill_topk_indices": prefill_topk.to(device),
        "query_start_loc": query_start_loc.to(device),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32, device=device),
        "token_to_req_indices": token_to_req.to(device),
        "swa_block_table": swa_table.to(device),
        "swa_block_size": swa_block_size,
        "compressed_block_table": None if compressed_table is None else compressed_table.to(device),
        "compressed_block_size": compressed_block_size,
        "window_size": window_size,
        "compress_ratio": compress_ratio,
        "topk": topk,
        "decode_compressed_indices_are_local": are_local,
        "decode_is_valid_token": None if decode_valid is None else decode_valid.to(device),
        "swa_block_span": swa_block_span,
        "compressed_block_span": compressed_block_span,
        "replay_start": replay.to(device),
    }


def reference_mixed_sparse_indices(case: dict) -> tuple[list[list[int]], list[int]]:
    """Independent serial oracle mirroring the vLLM Triton kernel token by token."""
    decode_swa = case["decode_swa_indices"].cpu().tolist()
    decode_compressed = case["decode_compressed_indices"]
    decode_compressed = decode_compressed.cpu().tolist() if decode_compressed is not None else None
    decode_topk_lens = case["decode_compressed_topk_lens"]
    decode_topk_lens = decode_topk_lens.cpu().tolist() if decode_topk_lens is not None else None
    is_valid_token = case["decode_is_valid_token"]
    is_valid_token = is_valid_token.cpu().tolist() if is_valid_token is not None else None
    prefill_topk = case["prefill_topk_indices"].cpu().tolist()
    query_start_loc = case["query_start_loc"].cpu().tolist()
    seq_lens = case["seq_lens"].cpu().tolist()
    token_to_req = case["token_to_req_indices"].cpu().tolist()
    swa_table = case["swa_block_table"].cpu().tolist()
    compressed_table = case["compressed_block_table"]
    compressed_table = compressed_table.cpu().tolist() if compressed_table is not None else None
    if compressed_table is None:
        compressed_table = swa_table
    replay = case["replay_start"]
    replay = replay.cpu().tolist() if replay is not None else None

    swa_block_size = case["swa_block_size"]
    swa_span = case["swa_block_span"] or swa_block_size
    compressed_block_size = case["compressed_block_size"]
    compressed_span = case["compressed_block_span"] or compressed_block_size
    window_size = case["window_size"]
    compress_ratio = case["compress_ratio"]
    topk = case["topk"]
    are_local = case["decode_compressed_indices_are_local"]

    num_decode = len(decode_swa)
    num_tokens = num_decode + len(prefill_topk)
    swa_width = case["decode_swa_indices"].shape[-1]
    decode_width = len(decode_compressed[0]) if decode_compressed else 0
    padded_topk = _ceil_div(max(topk, decode_width), 4) * 4

    def remap(value, block_size, span):
        if value < 0:
            return -1
        return value // block_size * span + value % block_size

    rows, lens = [], []
    for token in range(num_tokens):
        if token < num_decode:
            row = [remap(value, swa_block_size, swa_span) for value in decode_swa[token]]
            compressed_len = 0
            topk_row = []
            for column in range(padded_topk):
                value = decode_compressed[token][column] if column < decode_width else -1
                if are_local:
                    valid = value >= 0
                    if valid:
                        req = token_to_req[token]
                        block = compressed_table[req][value // compressed_block_size]
                        slot = block * compressed_block_size + value % compressed_block_size
                    else:
                        slot = -1
                    token_valid = bool(is_valid_token[token]) if is_valid_token is not None else True
                    compressed_len += int(valid and token_valid)
                else:
                    slot = value
                topk_row.append(remap(slot, compressed_block_size, compressed_span))
            if decode_width == 0:
                compressed_len = 0
            elif not are_local:
                compressed_len = decode_topk_lens[token] if decode_topk_lens is not None else decode_width
            row.extend(topk_row)
            rows.append(row)
            lens.append(swa_width + compressed_len)
        else:
            prefill_idx = token - num_decode
            req = token_to_req[token]
            query_start = query_start_loc[req]
            query_len = query_start_loc[req + 1] - query_start
            pos = (seq_lens[req] - query_len) + (token - query_start)
            req_replay = replay[req] if replay is not None else 0
            swa_start = max(pos - (window_size - 1), 0, req_replay)
            swa_len = pos - swa_start + 1
            topk_len = min((pos + 1) // compress_ratio, topk) if compress_ratio > 0 else 0
            row = []
            for column in range(swa_width):
                if column < swa_len:
                    window_pos = swa_start + column
                    slot = swa_table[req][window_pos // swa_block_size] * swa_block_size + window_pos % swa_block_size
                else:
                    slot = -1
                row.append(remap(slot, swa_block_size, swa_span))
            stride = len(prefill_topk[prefill_idx])
            for column in range(padded_topk):
                local = prefill_topk[prefill_idx][column] if column < min(stride, topk_len) else -1
                if local >= 0:
                    slot = (
                        compressed_table[req][local // compressed_block_size] * compressed_block_size
                        + local % compressed_block_size
                    )
                else:
                    slot = -1
                row.append(remap(slot, compressed_block_size, compressed_span))
            rows.append(row)
            lens.append(swa_width + topk_len)
    return rows, lens


def reference_mixed_sparse_tensors(case: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Oracle results as int32 tensors shaped like the kernel outputs."""
    rows, lens = reference_mixed_sparse_indices(case)
    swa_width = case["decode_swa_indices"].shape[-1]
    decode_width = case["decode_compressed_indices"].shape[-1] if case["decode_compressed_indices"] is not None else 0
    padded_topk = _ceil_div(max(case["topk"], decode_width), 4) * 4
    columns = swa_width + padded_topk
    indices = torch.tensor(rows, dtype=torch.int32).reshape(len(lens), columns)
    return indices, torch.tensor(lens, dtype=torch.int32)


# Real DeepSeek-V4.1-Flash scenario (deepseek-ai/DeepSeek-V4.1-Flash config):
# sliding_window=128, index_topk=512, compress_ratios only {0, 1, 2}
# ([0, 0, 2*18, 1*20, 0, 0, 0] + 3 MTP layers), SWA cache block size 64 and
# compressed_block_size = block_size // compress_ratio (64 / 32 / 64).
DSV41 = dict(window_size=128, swa_block_size=64)

CASES = [
    # Small hand-checkable geometries across every decode-index flavor.
    (
        "decode_local_small",
        dict(
            decode_query_lens=[1, 1, 2],
            seq_lens=[9, 300, 133],
            compress_ratio=2,
            window_size=8,
            topk=6,
            swa_block_size=4,
        ),
    ),
    (
        "decode_global_const_small",
        dict(
            decode_query_lens=[2, 1],
            seq_lens=[70, 9],
            compress_ratio=1,
            window_size=8,
            topk=0,
            swa_block_size=4,
            decode_indices="global_const",
            decode_topk_width=6,
        ),
    ),
    (
        "decode_global_lens_small",
        dict(
            decode_query_lens=[1, 1, 1],
            seq_lens=[40, 41, 8],
            compress_ratio=2,
            window_size=8,
            topk=4,
            swa_block_size=8,
            decode_indices="global_lens",
            decode_topk_width=5,
        ),
    ),
    (
        "prefill_ratio1_small",
        dict(
            prefill_query_lens=[5, 20],
            seq_lens=[30, 25],
            compress_ratio=1,
            window_size=8,
            topk=6,
            swa_block_size=8,
            replay_starts=[0, 12],
        ),
    ),
    (
        "prefill_swa_only_small",
        dict(
            prefill_query_lens=[10, 3],
            seq_lens=[40, 33],
            compress_ratio=0,
            topk=0,
            window_size=8,
            swa_block_size=8,
            replay_starts=[0, 5],
        ),
    ),
    (
        "mixed_packed_spans",
        dict(
            decode_query_lens=[1, 3],
            prefill_query_lens=[9, 7],
            seq_lens=[60, 33, 100, 45],
            compress_ratio=2,
            window_size=16,
            topk=8,
            swa_block_size=16,
            swa_block_span=24,
            compressed_block_span=12,
            replay_starts=[0, 0, 64, 0],
            invalid_decode_fraction=0.25,
            decode_is_valid=[True, True, False, True],
        ),
    ),
    (
        "dspark_padded_width",
        dict(
            decode_query_lens=[1],
            prefill_query_lens=[10],
            seq_lens=[300, 200],
            compress_ratio=1,
            window_size=8,
            swa_width=12,
            topk=6,
            swa_block_size=4,
        ),
    ),
    # DeepSeek-V4.1-Flash real serving steps: pure decode (ratio 0/1/2 layers),
    # SWA-only mixed chunk, decode+prefill steps with bounded replay.
    (
        "dsv41_decode_ratio1",
        dict(
            decode_query_lens=[1] * 32,
            seq_lens=[128 + 37 * i for i in range(32)],
            compress_ratio=1,
            topk=512,
            **DSV41,
        ),
    ),
    (
        "dsv41_decode_ratio2",
        dict(
            decode_query_lens=[1] * 32,
            seq_lens=[128 + 37 * i for i in range(32)],
            compress_ratio=2,
            topk=512,
            **DSV41,
        ),
    ),
    (
        "dsv41_swa_only",
        dict(
            decode_query_lens=[1] * 16,
            prefill_query_lens=[256],
            seq_lens=[8192] * 17,
            compress_ratio=0,
            topk=0,
            **DSV41,
        ),
    ),
    (
        "dsv41_mixed_ratio1",
        dict(
            decode_query_lens=[1] * 24,
            prefill_query_lens=[512, 256],
            seq_lens=[4096] * 24 + [2048, 6144],
            compress_ratio=1,
            topk=512,
            replay_starts=[0] * 24 + [0, 1500],
            **DSV41,
        ),
    ),
    (
        "dsv41_spec_mixed_ratio2",
        dict(
            decode_query_lens=[6] * 8,
            prefill_query_lens=[192],
            seq_lens=[2048] * 8 + [2560],
            compress_ratio=2,
            topk=512,
            decode_is_valid=[token % 6 != 5 for token in range(48)],
            **DSV41,
        ),
    ),
]

CASE_IDS = [name for name, _ in CASES]


@pytest.mark.api("functions.build_flashinfer_mixed_sparse_indices")
@pytest.mark.parametrize("name,spec", CASES, ids=CASE_IDS)
@pytest.mark.accuracy
def test_build_flashinfer_mixed_sparse_indices_accuracy(accuracy_backend, name, spec):
    implementation, _, device = accuracy_backend
    case = make_mixed_sparse_case(device, seed=7, **spec)
    expected_indices, expected_lens = reference_mixed_sparse_tensors(case)

    actual_indices, actual_lens = F.build_flashinfer_mixed_sparse_indices(**case, implementation=implementation)
    assert_close(actual_indices.cpu(), expected_indices, rtol=0, atol=0)
    assert_close(actual_lens.cpu(), expected_lens, rtol=0, atol=0)

    # The device-side torch reference must also match the oracle independently.
    reference_indices, reference_lens = F.build_flashinfer_mixed_sparse_indices(**case, implementation="torch")
    assert_close(reference_indices.cpu(), expected_indices, rtol=0, atol=0)
    assert_close(reference_lens.cpu(), expected_lens, rtol=0, atol=0)


@pytest.mark.api("functions.build_flashinfer_mixed_sparse_indices")
@pytest.mark.accuracy
def test_build_flashinfer_mixed_sparse_indices_empty_batch(accuracy_backend):
    implementation, _, device = accuracy_backend
    case = make_mixed_sparse_case(
        device,
        decode_query_lens=[],
        prefill_query_lens=[],
        seq_lens=[],
        window_size=8,
        compress_ratio=1,
        topk=4,
        swa_block_size=4,
    )
    expected_indices, expected_lens = reference_mixed_sparse_tensors(case)
    actual_indices, actual_lens = F.build_flashinfer_mixed_sparse_indices(**case, implementation=implementation)
    assert_close(actual_indices.cpu(), expected_indices, rtol=0, atol=0)
    assert_close(actual_lens.cpu(), expected_lens, rtol=0, atol=0)


@pytest.mark.api("functions.build_flashinfer_mixed_sparse_indices")
@pytest.mark.parametrize("name,spec", [CASES[0], CASES[5], CASES[9]], ids=[CASE_IDS[i] for i in (0, 5, 9)])
@pytest.mark.bitwise
def test_build_flashinfer_mixed_sparse_indices_bitwise(accuracy_backend, name, spec):
    implementation, _, device = accuracy_backend
    case = make_mixed_sparse_case(device, seed=3, **spec)
    assert_repeatable(
        partial(F.build_flashinfer_mixed_sparse_indices, **case, implementation=implementation),
        (),
        backward=False,
    )


@pytest.mark.api("functions.build_flashinfer_mixed_sparse_indices")
def test_build_flashinfer_mixed_sparse_indices_fake_metadata(accuracy_backend):
    implementation, target, _ = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("build_flashinfer_mixed_sparse_indices", implementation="triton", target=target)
    # Meta tensors dispatch the triton wrapper to its registered fake, which
    # must derive [num_tokens, swa_width + 16B-aligned topk] outputs.
    case = make_mixed_sparse_case(
        "meta",
        decode_query_lens=[1],
        prefill_query_lens=[3],
        seq_lens=[70, 40],
        window_size=8,
        compress_ratio=1,
        topk=6,
        swa_block_size=4,
    )
    indices, lens = F.build_flashinfer_mixed_sparse_indices(**case, implementation="triton")
    assert indices.device.type == "meta" and indices.dtype == torch.int32
    assert lens.device.type == "meta" and lens.dtype == torch.int32
    assert indices.shape == (4, case["decode_swa_indices"].shape[-1] + 8)  # topk 6 -> 16B-aligned 8
    assert lens.shape == (4,)


@pytest.mark.api("functions.build_flashinfer_mixed_sparse_indices")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_build_flashinfer_mixed_sparse_indices_compiled(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU Triton compile path")
    implementation = implementation or "triton"
    preload("build_flashinfer_mixed_sparse_indices", implementation=implementation)
    case = make_mixed_sparse_case(
        device,
        decode_query_lens=[1, 1],
        prefill_query_lens=[5],
        seq_lens=[70, 33, 40],
        window_size=8,
        compress_ratio=2,
        topk=6,
        swa_block_size=4,
    )

    def run():
        return F.build_flashinfer_mixed_sparse_indices(**case, implementation=implementation)

    try:
        compiled = compile_fullgraph(run, backend=backend)
        actual_indices, actual_lens = compiled()
        expected_indices, expected_lens = F.build_flashinfer_mixed_sparse_indices(**case, implementation="torch")
        assert_close(actual_indices, expected_indices, rtol=0, atol=0)
        assert_close(actual_lens, expected_lens, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()


@pytest.mark.reference
@pytest.mark.accuracy
@pytest.mark.parametrize("name,spec", CASES, ids=CASE_IDS)
def test_build_flashinfer_mixed_sparse_indices_reference_matches_oracle(name, spec):
    # A CPU oracle catches dispatch bugs in the torch reference itself; this
    # runs without an accelerator.
    case = make_mixed_sparse_case("cpu", seed=11, **spec)
    indices, lens = F.build_flashinfer_mixed_sparse_indices(**case, implementation="torch")
    expected_indices, expected_lens = reference_mixed_sparse_tensors(case)
    assert_close(indices, expected_indices, rtol=0, atol=0)
    assert_close(lens, expected_lens, rtol=0, atol=0)
