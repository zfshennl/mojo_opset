"""Tests for functions.swiglu_gated_mlp (forward only)."""

from functools import partial

import pytest
import torch

import torch.nn.functional as F

from mojo_opset import functions
from mojo_opset import preload
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

# swiglu_gated_mlp consumes [.., hidden] rows against gate/up weights of
# [intermediate, hidden] and a down weight of [hidden, intermediate].
# glm-5.3 (GlmMoeDsaMLP): hidden 6144, dense intermediate 12288, shared
# experts 2048, no clamp. glm-5.3-flash (Glm5NextTextMLP): hidden 4096,
# intermediate 12288 / shared 2048, swiglu_limit=10.0.
SWIGLU_MLP_SHAPES = [
    (7, 512, 320),  # odd tail rows/columns
    (128, 512, 320),
    (37, 1024, 768),
    (1, 6144, 12288),  # glm-5.3 dense MLP decode
    (128, 6144, 12288),  # glm-5.3 dense MLP prefill
    (1, 4096, 12288),  # glm-5.3-flash dense MLP decode
    (128, 4096, 12288),  # glm-5.3-flash dense MLP prefill
    (7, 6144, 2048),  # glm-5.3 shared experts decode
    (64, 4096, 2048),  # glm-5.3-flash shared experts prefill
]

# 0 disables the clamp (GlmMoeDsaMLP); 10.0 is glm-5.3-flash (Glm5NextTextMLP).
SWIGLU_MLP_LIMITS = [0.0, 10.0]

# transformers GlmMoeDsaConfig.initializer_range.
WEIGHT_STD = 0.02

# Chained-GEMM forward budgets. The kernel rounds its gate/up/h intermediates
# at the I/O dtype while the fp32 oracle does not; values within that noise of
# a rounding boundary snap to a different code point.
SWIGLU_MLP_FORWARD_TOLERANCES = {
    torch.float16: (2**-6, 0.25),
    torch.bfloat16: (2**-6, 0.25),
}


def _mlp_inputs(tokens, hidden, intermediate, dtype, device):
    return (
        torch.randn(tokens, hidden, device=device, dtype=dtype),
        torch.randn(intermediate, hidden, device=device, dtype=dtype) * WEIGHT_STD,
        torch.randn(intermediate, hidden, device=device, dtype=dtype) * WEIGHT_STD,
        torch.randn(hidden, intermediate, device=device, dtype=dtype) * WEIGHT_STD,
    )


def _reference_forward(x, gate_weight, up_weight, down_weight, swiglu_limit):
    """Independent oracle: the Glm5NextTextMLP forward with FP32 math."""
    x = x.to(torch.float32)
    gate = F.linear(x, gate_weight.to(torch.float32))
    up = F.linear(x, up_weight.to(torch.float32))
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.linear(F.silu(gate) * up, down_weight.to(torch.float32))


@pytest.mark.api("functions.swiglu_gated_mlp")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("limit", SWIGLU_MLP_LIMITS)
@pytest.mark.parametrize("shape", SWIGLU_MLP_SHAPES)
@pytest.mark.accuracy
def test_swiglu_gated_mlp_accuracy(accuracy_backend, dtype, limit, shape):
    implementation, target, device = accuracy_backend
    inputs = _mlp_inputs(*shape, dtype, device)

    expected = _reference_forward(*inputs, limit).to(dtype)
    actual = functions.swiglu_gated_mlp(*inputs, swiglu_limit=limit, implementation=implementation)

    rtol, atol = SWIGLU_MLP_FORWARD_TOLERANCES.get(dtype, (None, None))
    assert_close(actual, expected, dtype, rtol=rtol, atol=atol)


@pytest.mark.api("functions.swiglu_gated_mlp")
@pytest.mark.parametrize(
    "shape,limit",
    [((128, 512, 320), 0.0), ((128, 512, 320), 10.0), ((64, 4096, 2048), 10.0)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.bitwise
def test_swiglu_gated_mlp_bitwise(accuracy_backend, dtype, shape, limit):
    implementation, _, device = accuracy_backend
    inputs = _mlp_inputs(*shape, dtype, device)
    options = {"swiglu_limit": limit}
    # Forward-only op: no differentiable outputs, so skip the backward phase.
    assert_repeatable(partial(functions.swiglu_gated_mlp, implementation=implementation, **options), inputs, backward=False)


@pytest.mark.api("functions.swiglu_gated_mlp")
@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_swiglu_gated_mlp_fake_metadata(accuracy_backend, limit):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("swiglu_gated_mlp", implementation="triton", target=target)
    ops = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    x = torch.randn(14, 18, device=device, dtype=torch.bfloat16).T  # non-contiguous [18, 14]
    inputs = (x, torch.randn(32, 14, device=device, dtype=torch.bfloat16),
              torch.randn(32, 14, device=device, dtype=torch.bfloat16),
              torch.randn(14, 32, device=device, dtype=torch.bfloat16), limit)
    torch.library.opcheck(ops.swiglu_gated_mlp_fwd.default, inputs, test_utils=("test_faketensor",))


@pytest.mark.api("functions.swiglu_gated_mlp")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_swiglu_gated_mlp_compiled_layout(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU contiguous-output contract")
    implementation = implementation or "triton"
    preload("swiglu_gated_mlp", implementation=implementation)
    x = torch.randn(14, 18, device=device, dtype=torch.bfloat16).T  # [18, 14]
    # GlmMoeDsaConfig.initializer_range keeps activations in the trained range.
    inputs = (x,
              torch.randn(32, 14, device=device, dtype=torch.bfloat16).mul_(0.02),
              torch.randn(32, 14, device=device, dtype=torch.bfloat16).mul_(0.02),
              torch.randn(14, 32, device=device, dtype=torch.bfloat16).mul_(0.02))

    def run(*args):
        output = functions.swiglu_gated_mlp(*args, swiglu_limit=10.0, implementation=implementation)
        outputs = output if isinstance(output, tuple) else (output,)
        return outputs, tuple((y.stride(), y.is_contiguous()) for y in outputs)

    try:
        _, metadata = run(*inputs)
        actual, captured_metadata = compile_fullgraph(run, backend=backend)(*inputs)
        assert captured_metadata == metadata
        assert all(y.is_contiguous() for y in actual)
        reference = functions.swiglu_gated_mlp(*inputs, swiglu_limit=10.0, implementation="torch")
        reference = reference if isinstance(reference, tuple) else (reference,)
        for y, expected in zip(actual, reference):
            torch.testing.assert_close(y, expected, rtol=2e-2, atol=2e-2)
    finally:
        torch._dynamo.reset()


@pytest.mark.api("functions.swiglu_gated_mlp")
def test_swiglu_gated_mlp_contracts(accuracy_backend):
    implementation, _, device = accuracy_backend
    x = torch.randn(5, 16, device=device, dtype=torch.bfloat16)
    gate = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
    up = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
    down = torch.randn(8, 16, device=device, dtype=torch.bfloat16)  # expected [16, 8]
    with pytest.raises(ValueError):
        functions.swiglu_gated_mlp(x, gate, up, down, implementation=implementation)
    mismatched = torch.randn(8, 16, device=device, dtype=torch.float32)
    with pytest.raises(ValueError):
        functions.swiglu_gated_mlp(x, mismatched, up, torch.randn(16, 8, device=device, dtype=torch.bfloat16),
                                   implementation=implementation)
    with pytest.raises(ValueError):
        functions.swiglu_gated_mlp(x.to(torch.float32), gate.to(torch.float32), up.to(torch.float32),
                                   torch.randn(16, 8, device=device, dtype=torch.float32),
                                   implementation=implementation)
    with pytest.raises(ValueError):
        functions.swiglu_gated_mlp(x, gate, up, torch.randn(16, 8, device=device, dtype=torch.bfloat16),
                                   swiglu_limit=-1.0, implementation=implementation)


@pytest.mark.reference
@pytest.mark.accuracy
@pytest.mark.parametrize("limit", SWIGLU_MLP_LIMITS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_swiglu_gated_mlp_reference_matches_cpu_fp64(accuracy_backend, limit, dtype):
    # A CPU FP64 oracle catches vendor-specific dispatch bugs in the reference itself.
    device = accuracy_backend[2]
    inputs = _mlp_inputs(9, 64, 48, dtype, device)
    actual = functions.swiglu_gated_mlp(*inputs, swiglu_limit=limit, implementation="torch")

    reference_inputs = tuple(t.detach().cpu().double() for t in inputs)
    expected = _reference_forward(*reference_inputs, limit).to(dtype)

    assert_close(actual, expected.to(device), dtype)
