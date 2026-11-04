"""SwigluGatedMlp (SiLU-gated MLP with optional clamp) dispatched to one exact provider."""

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


class SwigluGatedMlpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight, swiglu_limit, implementation, require_backward):
        _check_swiglu_gated_mlp(x, gate_weight, up_weight, down_weight, swiglu_limit)
        forward, ctx.backward_kernel = load_impl("swiglu_gated_mlp", implementation, require_backward=require_backward)
        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(x, gate_weight, up_weight, down_weight)
        return forward(x, gate_weight, up_weight, down_weight, swiglu_limit)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, gate_weight, up_weight, down_weight = ctx.saved_tensors
        needs = ctx.needs_input_grad[:4]
        grads = ctx.backward_kernel(
            grad_output.contiguous(),
            x,
            gate_weight,
            up_weight,
            down_weight,
            ctx.swiglu_limit,
            *needs,
        )
        # Backends whose backward is a registered custom op return placeholder
        # empty tensors for skipped gradients; autograd requires None there.
        # (Real gradients of empty inputs stay 2D [0, N] and are kept.)
        mapped = []
        for grad, need in zip(grads, needs):
            if grad is not None and need and not (grad.dim() == 1 and grad.numel() == 0):
                mapped.append(grad)
            else:
                mapped.append(None)
        return (*mapped, None, None, None)


def _check_swiglu_gated_mlp(x, gate_weight, up_weight, down_weight, swiglu_limit):
    hidden_size = x.shape[-1]
    if x.dim() < 1:
        raise ValueError("swiglu_gated_mlp input must have at least one dimension.")
    if gate_weight.dim() != 2 or up_weight.dim() != 2 or down_weight.dim() != 2:
        raise ValueError("swiglu_gated_mlp weights must be 2D [out_features, in_features].")
    if gate_weight.shape != up_weight.shape or gate_weight.shape[1] != hidden_size:
        raise ValueError(
            "swiglu_gated_mlp gate/up weights must share shape [intermediate, hidden] matching the input."
        )
    if down_weight.shape != (hidden_size, gate_weight.shape[0]):
        raise ValueError(
            f"swiglu_gated_mlp down weight must be [hidden, intermediate]=[{hidden_size}, {gate_weight.shape[0]}], "
            f"got {list(down_weight.shape)}."
        )
    if not (x.dtype == gate_weight.dtype == up_weight.dtype == down_weight.dtype):
        raise ValueError("swiglu_gated_mlp input and weights must share dtype.")
    if not (x.device == gate_weight.device == up_weight.device == down_weight.device):
        raise ValueError("swiglu_gated_mlp input and weights must share device.")
    if swiglu_limit < 0:
        raise ValueError("swiglu_gated_mlp swiglu_limit must be nonnegative.")


def swiglu_gated_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    swiglu_limit: float = 0.0,
    implementation: str | None = None,
) -> torch.Tensor:
    """Gated MLP from transformers ``GlmMoeDsaMLP`` / ``Glm5NextTextMLP``:

    ``down_proj(silu(clamp(gate_proj(x), max=swiglu_limit)) * clamp(up_proj(x), +-swiglu_limit))``
    over ``nn.Linear``-layout weights (``gate``/``up``: ``[intermediate, hidden]``,
    ``down``: ``[hidden, intermediate]``, no bias). ``swiglu_limit=0`` disables the
    clamp (glm-5.3: hidden 6144 / intermediate 12288); glm-5.3-flash (hidden 4096)
    sets ``swiglu_limit=10.0``.
    """
    return SwigluGatedMlpFunction.apply(
        x,
        gate_weight,
        up_weight,
        down_weight,
        float(swiglu_limit),
        implementation,
        torch.is_grad_enabled() and any(t.requires_grad for t in (x, gate_weight, up_weight, down_weight)),
    )
