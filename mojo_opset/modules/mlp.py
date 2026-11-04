"""Weight-owning MLP modules."""

from typing import Optional

import torch

from mojo_opset import functions


class SwigluGatedMLP(torch.nn.Module):
    """Gated MLP block: ``down_proj(silu(clamp(gate_proj(x))) * clamp(up_proj(x)))``.

    Mirrors transformers ``GlmMoeDsaMLP`` (glm-5.3 dense MLP: hidden 6144 /
    intermediate 12288; shared experts: ``intermediate_size=
    moe_intermediate_size=2048``) and ``Glm5NextTextMLP`` (glm-5.3-flash:
    hidden 4096 / intermediate 12288 with ``swiglu_limit=10.0``). A
    ``swiglu_limit`` of ``0`` disables the clamp.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float = 0.0,
        *,
        implementation: Optional[str] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False, **factory_kwargs)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False, **factory_kwargs)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False, **factory_kwargs)
        self.swiglu_limit = swiglu_limit
        self.implementation = implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functions.swiglu_gated_mlp(
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            swiglu_limit=self.swiglu_limit,
            implementation=self.implementation,
        )
