import pytest
import torch

from mojo_opset import functions
from mojo_opset.kernels.torch.gmm_swiglu_quant import mxfp8_quantize
from mojo_opset.kernels.torch.gmm_swiglu_quant import mxfp8_quantize_weight


# GmmSwigluQuant workloads at the glm-5.3 / glm-5.3-flash model shapes:
# dense MLP layers use intermediate_size=12288 with hidden 6144 (glm-5.3,
# GlmMoeDsaMLP) or 4096 (glm-5.3-flash, Glm5NextTextMLP with
# swiglu_limit=10.0); shared experts use moe_intermediate_size=2048.
@pytest.mark.api("functions.gmm_swiglu_quant", ops=["gmm_swiglu_quant"])
@pytest.mark.parametrize(
    "tokens,hidden,intermediate,limit",
    [
        (1, 6144, 12288, 0.0),  # glm-5.3 dense decode
        (512, 6144, 12288, 0.0),  # glm-5.3 dense prefill
        (4096, 6144, 12288, 0.0),  # glm-5.3 dense prefill (full batch)
        (1, 4096, 12288, 10.0),  # glm-5.3-flash dense decode
        (512, 4096, 12288, 10.0),  # glm-5.3-flash dense prefill
        (4096, 4096, 12288, 10.0),  # glm-5.3-flash dense prefill (full batch)
        (64, 6144, 2048, 0.0),  # glm-5.3 shared experts
        (512, 4096, 2048, 10.0),  # glm-5.3-flash shared experts prefill
    ],
)
def test_gmm_swiglu_quant(benchmark, perf_environment, tokens, hidden, intermediate, limit):
    _, device, _, implementation = perf_environment

    def factory():
        x, x_scale = mxfp8_quantize(torch.randn(tokens, hidden, device=device))
        weight, weight_scale = mxfp8_quantize_weight(
            torch.randn(hidden, 2 * intermediate, device=device) * 0.02  # GlmMoeDsaConfig.initializer_range
        )
        inputs = (
            x,
            weight.view(1, hidden, 2 * intermediate),
            weight_scale.view(1, 2 * intermediate, -1),
            x_scale,
            torch.tensor([tokens], dtype=torch.int64, device=device),
        )
        return lambda: functions.gmm_swiglu_quant(*inputs, swiglu_limit=limit, implementation=implementation)

    benchmark(
        factory=factory,
        op="gmm_swiglu_quant",
        shape=[tokens, hidden, intermediate],
        dtype="float8e4m3fn",
        phase="forward",
        swiglu_limit=limit,
    )
