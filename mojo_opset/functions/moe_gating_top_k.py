"""MoE gating with optional grouped expert selection and Top-K routing."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def moe_gating_top_k(
    x: torch.Tensor,
    k: int,
    bias: Optional[torch.Tensor] = None,
    *,
    k_group: int = 1,
    group_count: int = 1,
    group_select_mode: int = 0,
    renorm: int = 0,
    norm_type: int = 0,
    out_flag: bool = False,
    routed_scaling_factor: float = 1.0,
    eps: float = 1e-20,
    implementation: Optional[str] = None,
):
    """Select the Top-K experts of every MoE routing row.

    ``x`` is ``[num_tokens, num_experts]`` with ``num_experts <= 2048`` and
    FP16, BF16 or FP32 dtype. The routing values are normalized with softmax
    (``norm_type=0``), sigmoid (``1`` or ``3``) or sqrt-softplus (``2``).
    ``bias`` only perturbs expert selection and is never gathered into the
    returned weights. When ``group_count > 1``, each group is scored by its
    maximum (``group_select_mode=0``) or its two largest values
    (``group_select_mode=1``); only the best ``k_group`` groups may contribute
    experts.

    Selected weights are renormalized for ``norm_type`` 1 and 2. For
    ``norm_type`` 0 and 3 renormalization follows ``renorm``. Every weight is
    then scaled by ``routed_scaling_factor``.

    Returns ``(y, expert_idx, norm_out)`` where ``y`` has ``x``'s dtype,
    ``expert_idx`` is INT32, and ``norm_out`` is the FP32 normalized routing
    table when ``out_flag`` is true, otherwise ``None``. Ties keep the lowest
    expert or group index. Inference only; there is no backward.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError("moe_gating_top_k expects x to be a torch.Tensor")
    _require_inference("moe_gating_top_k", x, bias)
    if x.ndim != 2:
        raise ValueError(f"moe_gating_top_k expects a 2D tensor, got {tuple(x.shape)}")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"moe_gating_top_k supports FP16, BF16 and FP32, got {x.dtype}")
    num_tokens, num_experts = x.shape
    if not 1 <= num_experts <= 2048:
        raise ValueError(f"moe_gating_top_k requires 1 <= num_experts <= 2048, got {num_experts}")

    k = int(k)
    k_group = int(k_group)
    group_count = int(group_count)
    group_select_mode = int(group_select_mode)
    renorm = int(renorm)
    norm_type = int(norm_type)
    if not 1 <= group_count <= num_experts or num_experts % group_count:
        raise ValueError(
            f"group_count({group_count}) must divide num_experts({num_experts}) and be in [1, num_experts]"
        )
    group_size = num_experts // group_count
    if group_size <= group_select_mode:
        raise ValueError(
            f"num_experts // group_count({group_size}) must be greater than group_select_mode({group_select_mode})"
        )
    if not 1 <= k_group <= group_count:
        raise ValueError(f"k_group({k_group}) must be in [1, group_count({group_count})]")
    if not 1 <= k <= group_size * k_group:
        raise ValueError(f"k({k}) must be in [1, (num_experts // group_count) * k_group]")
    if group_select_mode not in (0, 1):
        raise ValueError(f"group_select_mode must be 0 or 1, got {group_select_mode}")
    if renorm not in (0, 1):
        raise ValueError(f"renorm must be 0 or 1, got {renorm}")
    if norm_type not in (0, 1, 2, 3):
        raise ValueError(f"norm_type must be in 0..3, got {norm_type}")

    if bias is not None:
        if bias.ndim != 1 or bias.numel() != num_experts:
            raise ValueError(f"bias must have shape ({num_experts},), got {tuple(bias.shape)}")
        if bias.dtype != x.dtype:
            raise TypeError(f"bias dtype {bias.dtype} must match x dtype {x.dtype}")
        if bias.device != x.device:
            raise ValueError("bias must be on the same device as x")

    if not isinstance(out_flag, (bool, int)) or int(out_flag) not in (0, 1):
        raise ValueError("out_flag must be a boolean")
    if not isinstance(routed_scaling_factor, (int, float)):
        raise TypeError("routed_scaling_factor must be a number")
    if not isinstance(eps, (int, float)):
        raise TypeError("eps must be a number")

    forward, _ = load_impl("moe_gating_top_k", implementation)
    return forward(
        x,
        k,
        bias,
        k_group,
        group_count,
        group_select_mode,
        renorm,
        norm_type,
        bool(out_flag),
        float(routed_scaling_factor),
        float(eps),
    )
