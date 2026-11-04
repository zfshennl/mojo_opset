"""Tests for modules.SwigluGatedMLP."""

from functools import partial

import pytest
import torch

import torch.nn.functional as F

from mojo_opset import modules
from tests._checks import assert_close

# glm-5.3 dense MLP (6144/12288), glm-5.3-flash dense MLP (4096/12288), and
# the shared experts of both models (hidden / moe_intermediate_size=2048).
SWIGLU_MLP_MODULE_CASES = [
    (7, 512, 320),  # tail shapes
    (64, 6144, 2048),  # glm-5.3 shared experts
    (64, 4096, 2048),  # glm-5.3-flash shared experts
    (5, 6144, 12288),  # glm-5.3 dense MLP
    (5, 4096, 12288),  # glm-5.3-flash dense MLP
]

# transformers GlmMoeDsaConfig.initializer_range.
WEIGHT_STD = 0.02

# FP32 gradients keep blocked-GEMM accumulation order noise above the
# elementwise default; half dtypes add 1-ulp snapping of the rounded
# intermediates, budgeted like the grouped-GEMM matrix (see
# tests/functions/test_swiglu_gated_mlp.py).
SWIGLU_MLP_GRAD_TOLERANCES = {
    torch.float32: (1e-3, 1e-3),
    torch.float16: (2**-6, 1.0),
    torch.bfloat16: (2**-6, 1.0),
}

SWIGLU_MLP_FORWARD_TOLERANCES = {
    torch.float16: (2**-6, 0.25),
    torch.bfloat16: (2**-6, 0.25),
}


def _reference_forward(x, gate_weight, up_weight, down_weight, swiglu_limit):
    """Independent oracle: the Glm5NextTextMLP forward with FP32 math."""
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x = x.to(compute_dtype)
    gate = F.linear(x, gate_weight.to(compute_dtype))
    up = F.linear(x, up_weight.to(compute_dtype))
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.linear(F.silu(gate) * up, down_weight.to(compute_dtype))


def _reference_backward(x, gate_weight, up_weight, down_weight, swiglu_limit, grad_output):
    """Analytic Glm5NextTextMLP gradients with FP32 math."""
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x = x.to(compute_dtype)
    grad_output = grad_output.to(compute_dtype)

    gate = x @ gate_weight.to(compute_dtype).t()
    up = x @ up_weight.to(compute_dtype).t()
    gate_mask = torch.ones_like(gate)
    up_mask = torch.ones_like(up)
    if swiglu_limit > 0:
        gate_mask = (gate <= swiglu_limit).to(compute_dtype)
        up_mask = ((up >= -swiglu_limit) & (up <= swiglu_limit)).to(compute_dtype)
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)

    sigmoid_gate = torch.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    dsilu_gate = sigmoid_gate * (1 + gate * (1 - sigmoid_gate))

    dh = grad_output @ down_weight.to(compute_dtype)
    dg = dh * up * dsilu_gate * gate_mask
    du = dh * silu_gate * up_mask

    return (
        dg @ gate_weight.to(compute_dtype) + du @ up_weight.to(compute_dtype),
        dg.t() @ x,
        du.t() @ x,
        grad_output.t() @ (silu_gate * up),
    )


def _init_module(hidden, intermediate, limit, implementation, device, dtype):
    module = modules.SwigluGatedMLP(hidden, intermediate, limit, implementation=implementation)
    module.to(device=device, dtype=dtype)
    with torch.no_grad():
        for linear in (module.gate_proj, module.up_proj, module.down_proj):
            linear.weight.normal_(0, WEIGHT_STD)
    return module


@pytest.mark.api("modules.SwigluGatedMLP", ops=["swiglu_gated_mlp"])
@pytest.mark.parametrize("hidden,intermediate", [case[1:] for case in SWIGLU_MLP_MODULE_CASES])
@pytest.mark.parametrize("limit", [0.0, 10.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.accuracy
def test_swiglu_gated_mlp_module_accuracy(accuracy_backend, hidden, intermediate, limit, dtype):
    implementation, target, device = accuracy_backend
    module = _init_module(hidden, intermediate, limit, implementation, device, dtype)

    tokens = 7 if hidden == 512 else 5
    x = torch.randn(tokens, hidden, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn(tokens, hidden, device=device, dtype=dtype)

    actual = module(x)
    weights = (module.gate_proj.weight, module.up_proj.weight, module.down_proj.weight)
    expected = _reference_forward(x, *weights, limit).to(dtype)
    expected_grads = _reference_backward(x, *weights, limit, grad_output)

    check = partial(assert_close, dtype=dtype)
    frtol, fatol = SWIGLU_MLP_FORWARD_TOLERANCES.get(dtype, (None, None))
    check(actual, expected, rtol=frtol, atol=fatol)

    parameters = list(module.parameters())
    actual_grads = torch.autograd.grad(actual, [x, *parameters], grad_output)
    rtol, atol = SWIGLU_MLP_GRAD_TOLERANCES.get(dtype, (None, None))
    for actual_grad, expected_grad, name in zip(
        actual_grads, expected_grads, ("x", "gate_proj", "up_proj", "down_proj")
    ):
        check(actual_grad, expected_grad.to(dtype), name=f"grad_{name}", rtol=rtol, atol=atol)


@pytest.mark.api("modules.SwigluGatedMLP", ops=["swiglu_gated_mlp"])
@pytest.mark.accuracy
def test_swiglu_gated_mlp_module_hf_state_dict(accuracy_backend):
    """State-dict parity with transformers GlmMoeDsaMLP / Glm5NextTextMLP."""
    implementation, _, device = accuracy_backend
    hidden, intermediate = 64, 48
    hf_state_dict = {
        "gate_proj.weight": torch.randn(intermediate, hidden, device=device, dtype=torch.bfloat16) * WEIGHT_STD,
        "up_proj.weight": torch.randn(intermediate, hidden, device=device, dtype=torch.bfloat16) * WEIGHT_STD,
        "down_proj.weight": torch.randn(hidden, intermediate, device=device, dtype=torch.bfloat16) * WEIGHT_STD,
    }
    module = modules.SwigluGatedMLP(hidden, intermediate, swiglu_limit=10.0, implementation=implementation)
    module.load_state_dict(hf_state_dict, strict=True, assign=True)
    for name, parameter in hf_state_dict.items():
        assert torch.equal(getattr(module, name.split(".")[0]).weight, parameter)

    x = torch.randn(3, hidden, device=device, dtype=torch.bfloat16)
    expected = _reference_forward(x, module.gate_proj.weight, module.up_proj.weight, module.down_proj.weight, 10.0)
    assert_close(module(x), expected.to(x.dtype), torch.bfloat16)
