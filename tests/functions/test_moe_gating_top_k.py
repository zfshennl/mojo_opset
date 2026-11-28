# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the mojo_opset project
"""Accuracy cases for MoE gating Top-K, including Qwen3.8 model routing."""

import pytest
import torch

from mojo_opset import functions
from mojo_opset import preload
from tests._checks import assert_close
from tests._compile import compile_fullgraph

# Original op-plugin cases for the AscendC operator, plus the reference
# constraints: group_select_mode 0/1, norm_type 0..3, optional bias and both
# renorm settings. The IDs name the model scenario where one applies.
ROUTING_CASES = [
    pytest.param(
        8,
        256,
        6,
        dict(
            k_group=4, group_count=8, group_select_mode=1, renorm=0, norm_type=1, routed_scaling_factor=1.0, bias=True
        ),
        torch.float32,
        id="ascendc-8x256-k6-group-top2-sigmoid",
    ),
    pytest.param(
        1002,
        256,
        1,
        dict(
            k_group=4, group_count=8, group_select_mode=1, renorm=0, norm_type=1, routed_scaling_factor=1.0, bias=True
        ),
        torch.float16,
        id="ascendc-1002x256-k1-group-top2",
    ),
    pytest.param(
        128,
        256,
        16,
        dict(
            k_group=4, group_count=8, group_select_mode=1, renorm=0, norm_type=1, routed_scaling_factor=0.0, bias=True
        ),
        torch.bfloat16,
        id="ascendc-zero-scale",
    ),
    pytest.param(
        33,
        256,
        8,
        dict(
            k_group=4, group_count=8, group_select_mode=0, renorm=0, norm_type=1, routed_scaling_factor=2.5, bias=True
        ),
        torch.bfloat16,
        id="deepseek-v3-33x256-k8-group-max-sigmoid",
    ),
    pytest.param(
        1,
        512,
        10,
        dict(renorm=1, norm_type=0, routed_scaling_factor=1.0),
        torch.bfloat16,
        id="qwen3.8-flash-next-decode-1x512-k10",
    ),
    pytest.param(
        257,
        512,
        10,
        dict(renorm=1, norm_type=0, routed_scaling_factor=1.0),
        torch.bfloat16,
        id="qwen3.8-flash-next-257x512-k10",
    ),
    pytest.param(
        17,
        256,
        8,
        dict(renorm=1, norm_type=0, routed_scaling_factor=1.0),
        torch.bfloat16,
        id="qwen3.5-moe-17x256-k8",
    ),
    pytest.param(
        129,
        128,
        8,
        dict(renorm=1, norm_type=0, routed_scaling_factor=1.0),
        torch.float16,
        id="qwen3-30b-a3b-129x128-k8",
    ),
    pytest.param(
        19,
        384,
        6,
        dict(
            k_group=3,
            group_count=8,
            group_select_mode=0,
            renorm=0,
            norm_type=2,
            routed_scaling_factor=1.0,
            bias=True,
            out_flag=True,
        ),
        torch.bfloat16,
        id="sqrt-softplus-grouped-tail-384",
    ),
    pytest.param(
        7,
        128,
        4,
        dict(renorm=0, norm_type=3, routed_scaling_factor=1.5, out_flag=True),
        torch.float32,
        id="sigmoid-norm-type3-no-renorm",
    ),
    pytest.param(
        31,
        513,
        7,
        dict(
            k_group=2,
            group_count=3,
            group_select_mode=1,
            renorm=1,
            norm_type=0,
            routed_scaling_factor=1.0,
            bias=True,
            out_flag=True,
        ),
        torch.float32,
        id="odd-expert-count-513-grouped",
    ),
    pytest.param(
        0,
        256,
        8,
        dict(renorm=1, norm_type=0, out_flag=True),
        torch.bfloat16,
        id="empty-rows",
    ),
]


def _make_inputs(rows, experts, options, dtype, device):
    generator = torch.Generator(device="cpu").manual_seed(20261128 + rows + experts)
    x = torch.randn(rows, experts, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    bias = None
    if options.get("bias"):
        bias = (torch.randn(experts, generator=generator, dtype=torch.float32) * 0.2).to(device=device, dtype=dtype)
    return x, bias


def _call(x, k, bias, options, implementation):
    kwargs = {key: value for key, value in options.items() if key != "bias"}
    return functions.moe_gating_top_k(x, k, bias, implementation=implementation, **kwargs)


@pytest.mark.api("functions.moe_gating_top_k")
@pytest.mark.accuracy
@pytest.mark.parametrize("rows,experts,k,options,dtype", ROUTING_CASES)
def test_moe_gating_top_k_matches_torch_reference(accuracy_backend, rows, experts, k, options, dtype):
    """Normalization, grouped selection, Top-K and optional norm output match torch."""
    implementation, _, device = accuracy_backend
    x, bias = _make_inputs(rows, experts, options, dtype, device)
    expected = _call(x, k, bias, options, "torch")
    actual = _call(x, k, bias, options, implementation)

    assert actual[0].shape == (rows, k) and actual[0].dtype == dtype
    assert actual[1].shape == (rows, k) and actual[1].dtype == torch.int32
    assert_close(actual[0], expected[0], dtype, name="y")
    assert_close(actual[1], expected[1], rtol=0, atol=0, name="expert_idx")
    if options.get("out_flag"):
        assert actual[2].shape == x.shape and actual[2].dtype == torch.float32
        assert_close(actual[2], expected[2], torch.float32, name="norm_out")
    else:
        assert actual[2] is None and expected[2] is None


@pytest.mark.api("functions.moe_gating_top_k")
@pytest.mark.accuracy
def test_moe_gating_top_k_strided_inputs(accuracy_backend):
    """Non-contiguous rows and bias keep the same routing result."""
    implementation, _, device = accuracy_backend
    torch.manual_seed(7)
    x_storage = torch.randn(35, 512, device=device, dtype=torch.bfloat16)
    bias_storage = torch.randn(512, device=device, dtype=torch.bfloat16)
    x = x_storage[:, ::2]
    bias = bias_storage[::2]
    options = dict(
        k_group=4, group_count=8, group_select_mode=1, renorm=0, norm_type=1, routed_scaling_factor=2.5, out_flag=True
    )
    expected = _call(x, 6, bias, options, "torch")
    actual = _call(x, 6, bias, options, implementation)
    assert actual[0].is_contiguous() and actual[1].is_contiguous()
    assert_close(actual[0], expected[0], torch.bfloat16, name="y")
    assert_close(actual[1], expected[1], rtol=0, atol=0, name="expert_idx")
    assert_close(actual[2], expected[2], torch.float32, name="norm_out")


@pytest.mark.api("functions.moe_gating_top_k")
@pytest.mark.accuracy
def test_moe_gating_top_k_ties_keep_lowest_index(accuracy_backend):
    """Equal scores select the lowest expert and lowest group deterministically."""
    implementation, _, device = accuracy_backend
    x = torch.zeros(3, 256, device=device, dtype=torch.float32)
    zeros = torch.zeros(256, device=device, dtype=torch.float32)
    options = dict(k_group=4, group_count=8, group_select_mode=0, renorm=0, norm_type=0, out_flag=True)
    expected = _call(x, 6, zeros, options, "torch")
    actual = _call(x, 6, zeros, options, implementation)
    assert_close(actual[1], expected[1], rtol=0, atol=0, name="expert_idx")
    assert_close(actual[0], expected[0], torch.float32, name="y")


@pytest.mark.api("functions.moe_gating_top_k")
@pytest.mark.accuracy
def test_moe_gating_top_k_validates_arguments(accuracy_backend):
    """Reject routing configurations outside the AscendC operator contract."""
    _, _, device = accuracy_backend
    x = torch.randn(2, 128, device=device)
    invalid = [
        dict(k_group=3, group_count=2, k=1),
        dict(group_count=3, k=4),
        dict(group_count=8, k_group=8, k=129),
        dict(group_select_mode=2, k=4),
        dict(norm_type=4, k=4),
        dict(renorm=2, k=4),
    ]
    for kwargs in invalid:
        k = kwargs.pop("k")
        with pytest.raises(ValueError):
            functions.moe_gating_top_k(x, k, implementation="torch", **kwargs)


@pytest.mark.api("functions.moe_gating_top_k")
@pytest.mark.accuracy
def test_moe_gating_top_k_custom_op_supports_compile(accuracy_backend):
    """The registered custom op keeps schema/fake metadata and graph capture."""
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu"):
        pytest.skip("NPU Triton custom-op compile contract")
    preload("moe_gating_top_k", implementation="triton")
    namespace = "mojo_npu_triton_a5" if target.startswith("npu.a5") else "mojo_npu_triton_a2"
    x = torch.randn(4, 512, device=device, dtype=torch.bfloat16)
    bias = torch.randn(512, device=device, dtype=torch.bfloat16)
    torch.library.opcheck(
        getattr(torch.ops, namespace).moe_gating_top_k_fwd.default,
        (x, 10, bias, 4, 8, 1, 0, 1, True, 2.5, 1e-20),
        test_utils=("test_schema", "test_faketensor"),
    )

    def run(x, bias):
        return functions.moe_gating_top_k(
            x,
            10,
            bias,
            k_group=4,
            group_count=8,
            group_select_mode=1,
            renorm=0,
            norm_type=1,
            routed_scaling_factor=2.5,
            out_flag=True,
            implementation="triton",
        )

    try:
        eager = run(x, bias)
        compiled = compile_fullgraph(run)
        actual = compiled(x, bias)
        for a, e in zip(actual, eager):
            if a is None:
                assert e is None
            else:
                assert_close(a, e, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()
