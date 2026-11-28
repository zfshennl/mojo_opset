# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the mojo_opset project
"""Performance cases for MoE gating Top-K on routed MoE model shapes."""

import pytest
import torch

from mojo_opset import functions

# Qwen3.8-Flash-Next / Qwen4Exp routing: 512 experts, Top-10, FP32 softmax
# and renormalization. Qwen3.5-MoE uses 256 experts and Top-8; Qwen3-30B-A3B
# uses 128 experts and Top-8. Decode cases are launch-bound, while chunked
# prefill cases cover 2K and 8K routed token tiles. DeepSeek-V3 adds the
# grouped sigmoid route (8 groups, top-4 groups, top-2 group score).
GATING_CASES = [
    pytest.param(
        1,
        512,
        10,
        dict(renorm=1, norm_type=0),
        torch.bfloat16,
        id="qwen3.8-flash-next-decode-b1",
    ),
    pytest.param(
        64,
        512,
        10,
        dict(renorm=1, norm_type=0),
        torch.bfloat16,
        id="qwen3.8-flash-next-decode-b64",
    ),
    pytest.param(
        256,
        512,
        10,
        dict(renorm=1, norm_type=0),
        torch.bfloat16,
        id="qwen3.8-flash-next-decode-b256",
    ),
    pytest.param(
        2048,
        512,
        10,
        dict(renorm=1, norm_type=0),
        torch.bfloat16,
        id="qwen3.8-flash-next-prefill-2048",
    ),
    pytest.param(
        8192,
        512,
        10,
        dict(renorm=1, norm_type=0),
        torch.bfloat16,
        id="qwen3.8-flash-next-prefill-8192",
    ),
    pytest.param(
        1,
        256,
        8,
        dict(renorm=1, norm_type=0),
        torch.bfloat16,
        id="qwen3.5-moe-decode-b1",
    ),
    pytest.param(
        128,
        256,
        8,
        dict(renorm=1, norm_type=0),
        torch.bfloat16,
        id="qwen3.5-moe-decode-b128",
    ),
    pytest.param(
        8192,
        256,
        8,
        dict(renorm=1, norm_type=0),
        torch.bfloat16,
        id="qwen3.5-moe-prefill-8192",
    ),
    pytest.param(
        64,
        128,
        8,
        dict(renorm=1, norm_type=0),
        torch.bfloat16,
        id="qwen3-30b-a3b-decode-b64",
    ),
    pytest.param(
        2048,
        128,
        8,
        dict(renorm=1, norm_type=0),
        torch.bfloat16,
        id="qwen3-30b-a3b-prefill-2048",
    ),
    pytest.param(
        32,
        256,
        8,
        dict(
            k_group=4, group_count=8, group_select_mode=1, renorm=0, norm_type=1, routed_scaling_factor=2.5, bias=True
        ),
        torch.bfloat16,
        id="deepseek-v3-decode-b32-grouped",
    ),
    pytest.param(
        2048,
        256,
        8,
        dict(
            k_group=4, group_count=8, group_select_mode=1, renorm=0, norm_type=1, routed_scaling_factor=2.5, bias=True
        ),
        torch.bfloat16,
        id="deepseek-v3-prefill-2048-grouped",
    ),
    pytest.param(
        2048,
        256,
        8,
        dict(renorm=0, norm_type=0, out_flag=True),
        torch.float32,
        id="norm-output-2048x256-k8-fp32",
    ),
]


@pytest.mark.api("functions.moe_gating_top_k")
@pytest.mark.parametrize("rows,experts,k,options,dtype", GATING_CASES)
def test_moe_gating_top_k(benchmark, perf_environment, rows, experts, k, options, dtype):
    _, device, _, implementation = perf_environment
    options = dict(options)
    generator = torch.Generator(device="cpu").manual_seed(20261128 + rows + experts)
    x = torch.randn(rows, experts, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    needs_bias = options.pop("bias", False)
    bias = (
        (torch.randn(experts, generator=generator, dtype=torch.float32) * 0.2).to(device=device, dtype=dtype)
        if needs_bias
        else None
    )

    def factory():
        return lambda: functions.moe_gating_top_k(
            x,
            k,
            bias,
            implementation=implementation,
            **options,
        )

    benchmark(
        factory=factory,
        op="moe_gating_top_k",
        rows=rows,
        experts=experts,
        k=k,
        dtype=str(dtype),
        bias=needs_bias,
        **options,
    )
