"""Reference GmmSwigluQuant forward (MXFP8 in/out) used by accuracy checks and dispatch fallbacks."""

import torch
import torch.nn.functional as F

# E4M3FN: 3 mantissa bits, maximum finite value 448, emax = 8.
_E4M3_EMAX = 8
_E4M3_MAX = 448.0
_MX_GROUP = 32


def e8m0_to_scale(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Decode e8m0 scale bytes (uint8/int8 carrier) to 2^(byte - 127)."""
    return torch.pow(2.0, scale_bytes.to(torch.float32) - 127.0)


def mxfp8_quantize(values: torch.Tensor):
    """Quantize ``[..., K]`` to MXFP8 (OCP scales): e4m3 data + e8m0 scales.

    V3 numerics: values are rint-rounded (RNE) to BF16 first and the group
    amax / element quantization run on that BF16 view; elements are then
    quantized by one RTNE cast straight onto the fp8 grid (the reference
    ``cast_E4M3FN,rint``), with an explicit +-448 clamp providing the
    satfinite half of the OCP no-carry contract.

    Returns ``(data, scale_bytes)`` where data keeps the input shape and
    scale_bytes packs one e8m0 byte per 32 trailing elements as
    ``[..., ceil(K / 64), 2]`` (the unused byte of an odd final pair is 0),
    matching the gmm grouped_matmul_swiglu_quant_v2 scale packing.
    """
    shape = values.shape
    k = shape[-1]
    v2 = values.to(torch.bfloat16).to(torch.float32).reshape(-1, k)
    pairs = -(-k // 64)
    pad = pairs * 2 * _MX_GROUP - v2.shape[1]
    if pad:
        v2 = F.pad(v2, (0, pad))
    blocks = v2.reshape(-1, pairs * 2, _MX_GROUP)
    amax = blocks.abs().amax(dim=-1)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax)) - _E4M3_EMAX, torch.full_like(amax, -127.0))
    e = e.clamp_(min=-127.0)  # e8m0 encoding floor; keeps the byte cast in range
    q = (blocks * torch.exp2(-e)[..., None]).clamp_(-_E4M3_MAX, _E4M3_MAX)
    data = q.reshape(q.shape[0], -1)[:, :k].to(torch.float8_e4m3fn).reshape(shape)
    scales = (e + 127.0).to(torch.uint8).reshape(*shape[:-1], pairs, 2)
    return data, scales


def mxfp8_quantize_weight(values: torch.Tensor):
    """MXFP8-quantize a ``[K, N]`` weight with 32-element groups along K.

    Returns ``(data [K, N], scale_bytes [ceil(K/64), N, 2])`` matching the
    gmm_swiglu_quant weight_scale packing (per-32-K, per-column e8m0).
    """
    data, scales = mxfp8_quantize(values.t())
    return data.t().contiguous(), scales.transpose(0, 1).contiguous()


def gmm_swiglu_quant_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 grouped matmul + SwiGLU + MX quant with FP32 math.

    C = (x * x_scale) @ (w * w_scale) with the per-32-K scales expanded
    before the reduction; S = silu(clamp(C[:, :I], max=L)) *
    clamp(C[:, I:], +-L); (y, y_scale) = MXFP8 quant of S.
    """
    del group_list  # dense single expert; validated by the public wrapper
    k = x.shape[-1]
    rows = x.numel() // k
    n = weight.shape[-1]
    inter = n // 2
    k_pairs = -(-k // 64)

    x2 = x.reshape(rows, k).to(torch.float32)
    w2 = weight.reshape(k, n).to(torch.float32)
    # x_scale [rows, kp, 2] -> logical per-32-block scale [rows, kp*2].
    xs = e8m0_to_scale(x_scale.reshape(rows, k_pairs * 2))
    # weight_scale [kp, n, 2] -> logical per-32-block per-column [kp*2, n].
    ws = e8m0_to_scale(weight_scale.reshape(k_pairs, n, 2)).transpose(1, 2).reshape(k_pairs * 2, n)

    c = (x2 * xs.repeat_interleave(_MX_GROUP, dim=1)[:, :k]) @ (
        w2 * ws.repeat_interleave(_MX_GROUP, dim=0)[:k, :]
    )
    act, lin = c[:, :inter], c[:, inter:]
    if swiglu_limit > 0:
        act = act.clamp(max=swiglu_limit)
        lin = lin.clamp(min=-swiglu_limit, max=swiglu_limit)
    return mxfp8_quantize(F.silu(act) * lin)
