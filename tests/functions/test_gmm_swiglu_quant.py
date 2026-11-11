"""Tests for functions.gmm_swiglu_quant (forward only, MXFP8 in/out)."""

from functools import partial

import pytest
import torch
import torch.nn.functional as F

from mojo_opset import functions
from mojo_opset import preload
from mojo_opset.kernels.torch.gmm_swiglu_quant import mxfp8_quantize
from mojo_opset.kernels.torch.gmm_swiglu_quant import mxfp8_quantize_weight
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

# gmm_swiglu_quant consumes [.., K] rows against a fused [1, K, 2I] MXFP8
# weight (two SwiGLU halves along N). glm-5.3 (GlmMoeDsaMLP): hidden 6144,
# dense intermediate 12288, shared experts 2048, no clamp. glm-5.3-flash
# (Glm5NextTextMLP): hidden 4096, intermediate 12288, swiglu_limit=10.0.
GMM_SWIGLU_QUANT_SHAPES = [
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
GMM_SWIGLU_QUANT_LIMITS = [0.0, 10.0]

# transformers GlmMoeDsaConfig.initializer_range.
WEIGHT_STD = 0.02

# V3 numerics bound, in units of the group scale 2^e: the BF16 intermediate
# view rounds by <= 1 (half ULP of [256, 512)), one RTNE cast onto the fp8
# grid adds <= 16 at the head binade, and the no-carry OCP scale saturates
# up to 512 - 448 = 64 further steps; accumulation-order noise and amax
# scale-boundary flips are covered by the remaining margin.
# Bound: |y * 2^e - S| <= 82 * 2^e + eps.
_QUANT_STEP_BOUND = 82.0


def _quantize(value):
    data, scales = mxfp8_quantize(value)
    return data, scales


def _gmm_inputs(tokens, hidden, intermediate, device):
    x, x_scale = _quantize(torch.randn(tokens, hidden, device=device))
    weight, weight_scale = mxfp8_quantize_weight(
        torch.randn(hidden, 2 * intermediate, device=device) * WEIGHT_STD
    )
    group_list = torch.tensor([tokens], dtype=torch.int64, device=device)
    return x, weight.view(1, hidden, 2 * intermediate), weight_scale.view(1, 2 * intermediate, -1), \
        x_scale, group_list


def _decode_e8m0(scale_bytes):
    return torch.pow(2.0, scale_bytes.to(torch.float32) - 127.0)


def _reference_silu_half(x, weight, x_scale, weight_scale, swiglu_limit):
    """Independent oracle: FP32 dequant-GEMM + SwiGLU, pre-quantization S."""
    rows, k = x.reshape(-1, x.shape[-1]).shape
    n = weight.shape[-1]
    inter = n // 2
    xs = _decode_e8m0(x_scale.reshape(rows, -1))
    ws = _decode_e8m0(weight_scale.reshape(n, -1)).t()
    xd = x.reshape(rows, k).to(torch.float32) * xs.repeat_interleave(32, dim=1)[:, :k]
    wd = weight.reshape(k, n).to(torch.float32) * ws.repeat_interleave(32, dim=0)[:k, :]
    c = xd @ wd
    act, lin = c[:, :inter], c[:, inter:]
    if swiglu_limit > 0:
        act = act.clamp(max=swiglu_limit)
        lin = lin.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(act) * lin


def _assert_quantized_close(y, y_scale, expected):
    """Dequantized y must match the FP32 oracle within MX quantization steps."""
    rows, inter = y.shape
    scale = _decode_e8m0(y_scale.reshape(rows, -1)[:, : (inter + 31) // 32])
    # repeat_interleave, not a [:, arange // 32] gather: the all-equal index
    # tensor of a single-group column hits a wrong-result gather bug on
    # torch_npu (evidence: scripts/gmm_swiglu_quant_gather_probe.py).
    scale = scale.repeat_interleave(32, dim=1)[:, :inter]
    error = (y.to(torch.float32) * scale - expected).abs()
    bound = scale * _QUANT_STEP_BOUND + 1e-6
    assert bool((error <= bound).all()), (
        f"gmm_swiglu_quant quantization error {(error / bound).max().item():.3f}x bound"
    )


@pytest.mark.api("functions.gmm_swiglu_quant", ops=["gmm_swiglu_quant"])
@pytest.mark.parametrize("limit", GMM_SWIGLU_QUANT_LIMITS)
@pytest.mark.parametrize("shape", GMM_SWIGLU_QUANT_SHAPES)
@pytest.mark.accuracy
def test_gmm_swiglu_quant_accuracy(accuracy_backend, limit, shape):
    implementation, target, device = accuracy_backend
    inputs = _gmm_inputs(*shape, device)

    expected = _reference_silu_half(inputs[0], inputs[1], inputs[3], inputs[2], limit)
    y, y_scale = functions.gmm_swiglu_quant(*inputs, swiglu_limit=limit, implementation=implementation)

    assert y.dtype is torch.float8_e4m3fn and y.shape == expected.shape
    assert y_scale.dtype is torch.uint8 and y_scale.shape == (
        shape[0], (shape[2] + 31) // 32,
    )
    _assert_quantized_close(y, y_scale, expected)


@pytest.mark.api("functions.gmm_swiglu_quant", ops=["gmm_swiglu_quant"])
@pytest.mark.parametrize(
    "shape,limit",
    [((128, 512, 320), 0.0), ((128, 512, 320), 10.0), ((64, 4096, 2048), 10.0)],
)
@pytest.mark.bitwise
def test_gmm_swiglu_quant_bitwise(accuracy_backend, shape, limit):
    implementation, _, device = accuracy_backend
    inputs = _gmm_inputs(*shape, device)
    options = {"swiglu_limit": limit}
    # Forward-only op: no differentiable outputs, so skip the backward phase.
    assert_repeatable(partial(functions.gmm_swiglu_quant, implementation=implementation, **options),
                      inputs, backward=False)


@pytest.mark.api("functions.gmm_swiglu_quant", ops=["gmm_swiglu_quant"])
@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_gmm_swiglu_quant_fake_metadata(accuracy_backend, limit):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu.a5"):
        pytest.skip("NPU A5 Triton fake metadata")
    preload("gmm_swiglu_quant", implementation="triton", target=target)
    ops = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    inputs = _gmm_inputs(14, 32, 16, device)  # K/I tails inside one scale block
    torch.library.opcheck(ops.gmm_swiglu_quant_fwd.default, (*inputs, limit), test_utils=("test_faketensor",))


@pytest.mark.api("functions.gmm_swiglu_quant", ops=["gmm_swiglu_quant"])
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_gmm_swiglu_quant_compiled_layout(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU contiguous-output contract")
    implementation = implementation or "triton"
    preload("gmm_swiglu_quant", implementation=implementation)
    inputs = _gmm_inputs(18, 32, 16, device)

    def run(*args):
        outputs = functions.gmm_swiglu_quant(*args, swiglu_limit=10.0, implementation=implementation)
        return outputs, tuple((y.stride(), y.is_contiguous()) for y in outputs)

    try:
        _, metadata = run(*inputs)
        actual, captured_metadata = compile_fullgraph(run, backend=backend)(*inputs)
        assert captured_metadata == metadata
        assert all(y.is_contiguous() for y in actual)
        expected = _reference_silu_half(inputs[0], inputs[1], inputs[3], inputs[2], 10.0)
        _assert_quantized_close(actual[0], actual[1], expected)
    finally:
        torch._dynamo.reset()


@pytest.mark.api("functions.gmm_swiglu_quant", ops=["gmm_swiglu_quant"])
def test_gmm_swiglu_quant_contracts(accuracy_backend):
    implementation, _, device = accuracy_backend
    x, weight, weight_scale, x_scale, group_list = _gmm_inputs(5, 64, 32, device)
    with pytest.raises(ValueError):  # weight must be [1, K, N]
        functions.gmm_swiglu_quant(x, weight.view(64, 64), weight_scale, x_scale, group_list,
                                   implementation=implementation)
    with pytest.raises(ValueError):  # two experts are dense-only for now
        functions.gmm_swiglu_quant(x, weight, weight_scale, x_scale,
                                   torch.tensor([3, 5], dtype=torch.int64, device=device),
                                   implementation=implementation)
    with pytest.raises(ValueError):  # odd N has no SwiGLU halves
        functions.gmm_swiglu_quant(x, weight[..., :63].contiguous(),
                                   weight_scale[..., :63].contiguous(), x_scale, group_list,
                                   implementation=implementation)
    with pytest.raises(ValueError):  # e8m0 carriers are uint8/int8 only
        functions.gmm_swiglu_quant(x, weight, weight_scale.float(), x_scale, group_list,
                                   implementation=implementation)
    with pytest.raises(ValueError):  # x must be fp8
        functions.gmm_swiglu_quant(x.to(torch.bfloat16), weight, weight_scale, x_scale, group_list,
                                   implementation=implementation)
    with pytest.raises(ValueError):
        functions.gmm_swiglu_quant(x, weight, weight_scale, x_scale, group_list,
                                   swiglu_limit=-1.0, implementation=implementation)


@pytest.mark.reference
@pytest.mark.accuracy
@pytest.mark.parametrize("limit", GMM_SWIGLU_QUANT_LIMITS)
def test_gmm_swiglu_quant_reference_matches_cpu_fp64(accuracy_backend, limit):
    # A CPU FP64 oracle catches vendor-specific dispatch bugs in the reference itself.
    device = accuracy_backend[2]
    tokens, hidden, intermediate = 9, 64, 48
    inputs = _gmm_inputs(tokens, hidden, intermediate, device)

    actual = functions.gmm_swiglu_quant(*inputs, swiglu_limit=limit, implementation="torch")
    cpu_inputs = tuple(t.cpu() for t in inputs)
    x64 = cpu_inputs[0].to(torch.float64)
    w64 = cpu_inputs[1].to(torch.float64).reshape(hidden, 2 * intermediate)
    xs64 = _decode_e8m0(cpu_inputs[3].to(torch.float64)).reshape(tokens, -1)
    ws64 = _decode_e8m0(cpu_inputs[2].to(torch.float64)).reshape(2 * intermediate, -1).t()
    xd = x64 * xs64.repeat_interleave(32, dim=1)[:, :hidden]
    wd = w64 * ws64.repeat_interleave(32, dim=0)[:hidden, :]
    c = xd @ wd
    act, lin = c[:, :intermediate], c[:, intermediate:]
    if limit > 0:
        act = act.clamp(max=limit)
        lin = lin.clamp(min=-limit, max=limit)
    expected = F.silu(act) * lin

    _assert_quantized_close(actual[0].cpu(), actual[1].cpu(), expected)
