import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training


# SwigluGatedMlp workloads at the glm-5.3 / glm-5.3-flash model shapes:
# dense MLP layers use intermediate_size=12288 with hidden 6144 (glm-5.3,
# GlmMoeDsaMLP) or 4096 (glm-5.3-flash, Glm5NextTextMLP with
# swiglu_limit=10.0); shared experts use moe_intermediate_size=2048.
@pytest.mark.api("functions.swiglu_gated_mlp", ops=["swiglu_gated_mlp"])
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
@pytest.mark.parametrize("phase", ["forward", "backward", "forward_backward"])
def test_swiglu_gated_mlp(benchmark, perf_environment, tokens, hidden, intermediate, limit, phase):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16, requires_grad=phase != "forward")
        weights = []
        for shape in ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate)):
            weight = torch.empty(shape, device=device, dtype=torch.bfloat16, requires_grad=phase != "forward")
            torch.nn.init.normal_(weight, 0, 0.02)  # GlmMoeDsaConfig.initializer_range
            weights.append(weight)
        inputs = (x, *weights)
        call = lambda: functions.swiglu_gated_mlp(*inputs, swiglu_limit=limit, implementation=implementation)
        return training(call, inputs, phase)

    benchmark(
        factory=factory,
        op="swiglu_gated_mlp",
        shape=[tokens, hidden, intermediate],
        dtype="bfloat16",
        phase=phase,
        swiglu_limit=limit,
    )
