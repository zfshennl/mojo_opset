import pytest
import torch

from mojo_opset import modules


# Module-level SwigluGatedMLP workloads at the glm-5.3 / glm-5.3-flash model
# shapes: dense MLP (12288 intermediate) and shared experts (2048).
@pytest.mark.api("modules.SwigluGatedMLP", ops=["swiglu_gated_mlp"])
@pytest.mark.parametrize(
    "tokens,hidden,intermediate,limit",
    [
        (1, 6144, 12288, 0.0),  # glm-5.3 dense decode
        (512, 6144, 12288, 0.0),  # glm-5.3 dense prefill
        (1, 4096, 12288, 10.0),  # glm-5.3-flash dense decode
        (512, 4096, 12288, 10.0),  # glm-5.3-flash dense prefill
        (64, 6144, 2048, 0.0),  # glm-5.3 shared experts
        (512, 4096, 2048, 10.0),  # glm-5.3-flash shared experts prefill
    ],
)
def test_swiglu_gated_mlp(benchmark, perf_environment, tokens, hidden, intermediate, limit):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.SwigluGatedMLP(hidden, intermediate, limit, implementation=implementation).to(
            device=device, dtype=torch.bfloat16
        )
        for linear in (module.gate_proj, module.up_proj, module.down_proj):
            torch.nn.init.normal_(linear.weight, 0, 0.02)
        x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
        return lambda: module(x)

    benchmark(
        factory=factory,
        op="swiglu_gated_mlp",
        shape=[tokens, hidden, intermediate],
        dtype="bfloat16",
        phase="forward",
        swiglu_limit=limit,
    )
