"""Reference SwigluGatedMlp forward used by accuracy checks and dispatch fallbacks."""

import torch

import torch.nn.functional as F


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype


def swiglu_gated_mlp_fwd(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """Glm5NextTextMLP forward with FP32 math; swiglu_limit=0 disables the clamp."""
    compute_dtype = _compute_dtype(x)
    ori_shape = x.shape
    hidden_size = ori_shape[-1]

    x_2d = x.reshape(-1, hidden_size).to(compute_dtype)
    gate = F.linear(x_2d, gate_weight.to(compute_dtype))
    up = F.linear(x_2d, up_weight.to(compute_dtype))
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    intermediate = F.silu(gate) * up
    out = F.linear(intermediate, down_weight.to(compute_dtype))
    return out.to(x.dtype).reshape(ori_shape)
