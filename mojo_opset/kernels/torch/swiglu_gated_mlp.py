"""Reference SwigluGatedMlp used by accuracy checks and dispatch fallbacks."""

from typing import Optional

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


def swiglu_gated_mlp_bwd(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    swiglu_limit: float,
    need_x: bool,
    need_gate: bool,
    need_up: bool,
    need_down: bool,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Analytic gradients; clamp boundaries pass their gradient (torch convention)."""
    compute_dtype = _compute_dtype(x)
    hidden_size = x.shape[-1]

    x_2d = x.reshape(-1, hidden_size).to(compute_dtype)
    grad_2d = grad_output.reshape(-1, hidden_size).to(compute_dtype)

    gate = x_2d @ gate_weight.to(compute_dtype).t()
    up = x_2d @ up_weight.to(compute_dtype).t()
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
    h = silu_gate * up

    grad_intermediate = grad_2d @ down_weight.to(compute_dtype)
    grad_gate = grad_intermediate * up * dsilu_gate * gate_mask
    grad_up = grad_intermediate * silu_gate * up_mask

    grad_x = grad_gate @ gate_weight.to(compute_dtype) + grad_up @ up_weight.to(compute_dtype) if need_x else None
    dgate_weight = grad_gate.t() @ x_2d if need_gate else None
    dup_weight = grad_up.t() @ x_2d if need_up else None
    ddown_weight = grad_2d.t() @ h if need_down else None

    return (
        None if grad_x is None else grad_x.to(x.dtype).reshape(x.shape),
        None if dgate_weight is None else dgate_weight.to(gate_weight.dtype),
        None if dup_weight is None else dup_weight.to(up_weight.dtype),
        None if ddown_weight is None else ddown_weight.to(down_weight.dtype),
    )
