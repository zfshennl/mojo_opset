"""Performance tests for functions.build_flashinfer_mixed_sparse_indices."""

import pytest

from mojo_opset import functions
from tests.functions.test_build_flashinfer_mixed_sparse_indices import make_mixed_sparse_case

# Real DeepSeek-V4.1-Flash serving workloads (deepseek-ai/DeepSeek-V4.1-Flash):
# sliding_window=128, index_topk=512, per-layer compress_ratios in {0, 1, 2},
# SWA cache block size 64, compressed block size 64 // ratio, decode-first
# batches under max_num_batched_tokens=4096, and DSpark spec queries of
# 6 tokens with their last draft token invalid.
WINDOW = 128
TOPK = 512
SWA_BLOCK = 64

WORKLOADS = {
    "dsv41_decode_ratio0": dict(
        decode_query_lens=[1] * 256,
        prefill_query_lens=[],
        seq_lens=[256 + 31 * i for i in range(256)],
        compress_ratio=0,
        topk=0,
        window_size=WINDOW,
        swa_block_size=SWA_BLOCK,
    ),
    "dsv41_decode_ratio1": dict(
        decode_query_lens=[1] * 256,
        prefill_query_lens=[],
        seq_lens=[256 + 31 * i for i in range(256)],
        compress_ratio=1,
        topk=TOPK,
        window_size=WINDOW,
        swa_block_size=SWA_BLOCK,
    ),
    "dsv41_decode_ratio2": dict(
        decode_query_lens=[1] * 256,
        prefill_query_lens=[],
        seq_lens=[256 + 31 * i for i in range(256)],
        compress_ratio=2,
        topk=TOPK,
        window_size=WINDOW,
        swa_block_size=SWA_BLOCK,
    ),
    "dsv41_spec_decode_ratio1": dict(
        decode_query_lens=[6] * 128,
        prefill_query_lens=[],
        seq_lens=[1024 + 17 * i for i in range(128)],
        compress_ratio=1,
        topk=TOPK,
        decode_is_valid=[token % 6 != 5 for token in range(768)],
        window_size=WINDOW,
        swa_block_size=SWA_BLOCK,
    ),
    "dsv41_mixed_ratio1": dict(
        decode_query_lens=[1] * 128,
        prefill_query_lens=[2048, 1792],
        seq_lens=[4096] * 128 + [6144, 8192],
        compress_ratio=1,
        topk=TOPK,
        replay_starts=[0] * 128 + [4096, 0],
        window_size=WINDOW,
        swa_block_size=SWA_BLOCK,
    ),
    "dsv41_mixed_ratio2": dict(
        decode_query_lens=[1] * 128,
        prefill_query_lens=[2048, 1792],
        seq_lens=[4096] * 128 + [6144, 8192],
        compress_ratio=2,
        topk=TOPK,
        replay_starts=[0] * 128 + [4096, 0],
        window_size=WINDOW,
        swa_block_size=SWA_BLOCK,
    ),
    "dsv41_prefill_ratio1": dict(
        decode_query_lens=[],
        prefill_query_lens=[2048, 1024, 1024],
        seq_lens=[4096, 3072, 1024],
        compress_ratio=1,
        topk=TOPK,
        window_size=WINDOW,
        swa_block_size=SWA_BLOCK,
    ),
    "dsv41_long_chunk_ratio2": dict(
        decode_query_lens=[],
        prefill_query_lens=[4096],
        seq_lens=[32768],
        compress_ratio=2,
        topk=TOPK,
        window_size=WINDOW,
        swa_block_size=SWA_BLOCK,
    ),
}


@pytest.mark.api("functions.build_flashinfer_mixed_sparse_indices")
@pytest.mark.parametrize("workload", list(WORKLOADS))
def test_build_flashinfer_mixed_sparse_indices(benchmark, perf_environment, workload):
    _, device, _, implementation = perf_environment
    spec = WORKLOADS[workload]
    num_tokens = sum(spec["decode_query_lens"]) + sum(spec["prefill_query_lens"])

    def factory():
        case = make_mixed_sparse_case(device, seed=0, **spec)
        return lambda: functions.build_flashinfer_mixed_sparse_indices(**case, implementation=implementation)

    benchmark(
        factory=factory,
        op="build_flashinfer_mixed_sparse_indices",
        workload=workload,
        num_tokens=num_tokens,
        num_decode_tokens=sum(spec["decode_query_lens"]),
        window_size=spec["window_size"],
        topk=spec["topk"],
        compress_ratio=spec["compress_ratio"],
        swa_block_size=spec["swa_block_size"],
        dtype="int32",
    )
