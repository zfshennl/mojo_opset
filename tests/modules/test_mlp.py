"""Tests for modules.SwigluGatedMLP (forward only)."""

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

SWIGLU_MLP_FORWARD_TOLERANCES = {
    torch.float16: (2**-6, 0.25),
    torch.bfloat16: (2**-6, 0.25),
}


def _reference_forward(x, gate_weight, up_weight, down_weight, swiglu_limit):
    """Independent oracle: the Glm5NextTextMLP forward with FP32 math."""
    x = x.to(torch.float32)
    gate = F.linear(x, gate_weight.to(torch.float32))
    up = F.linear(x, up_weight.to(torch.float32))
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.linear(F.silu(gate) * up, down_weight.to(torch.float32))


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
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.accuracy
def test_swiglu_gated_mlp_module_accuracy(accuracy_backend, hidden, intermediate, limit, dtype):
    implementation, target, device = accuracy_backend
    module = _init_module(hidden, intermediate, limit, implementation, device, dtype)

    tokens = 7 if hidden == 512 else 5
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)

    actual = module(x)
    weights = (module.gate_proj.weight, module.up_proj.weight, module.down_proj.weight)
    expected = _reference_forward(x, *weights, limit).to(dtype)

    check = partial(assert_close, dtype=dtype)
    rtol, atol = SWIGLU_MLP_FORWARD_TOLERANCES.get(dtype, (None, None))
    check(actual, expected, rtol=rtol, atol=atol)


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
