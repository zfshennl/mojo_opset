# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the mojo_opset project
"""Experimental TLA implementation of ``moe_gating_top_k`` for A5.

The kernel follows the TLA ``topk_gate`` vector style (explicit UB buffers and
64-lane SIMD registers) and uses a real `BLOCK_M` batched row path:

- one 2D ``GM -> UB`` load for ``BLOCK_M`` rows;
- one 2D ``UB -> GM`` store for y / expert_idx / optional norm;
- per-row SIMD processing inside the loaded UB tile;
- a per-row copy fallback only for the last partial batch.

Supported constraints
----------------------
- ``num_experts % 64 == 0`` and ``num_experts <= 4096`` (TLA tail DMA not used).
- ``group_count <= 64`` and ``k <= 64``.
- FP16 / BF16 / FP32 input, optional same-dtype bias.
- All ``norm_type`` values, both group-select modes, renorm and ``out_flag``.

The Top-K algorithm is still repeated max/min, so prefill remains slower than
AscendC's single ``Sort32 + MrgSort`` path. See
``moe_gating_top_k_tla_experiment.md`` for measurements.
"""

from typing import Optional

import torch
import triton.language as tl
import triton.language.extra.tla as tla

__all__ = ["moe_gating_top_k_tla"]


@tla.jit
def _moe_gating_top_k_tla_kernel(
    X,
    BIAS,
    Y,
    IDX,
    NORM_OUT,
    n_rows,
    STRIDE_X: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    K_GROUP: tl.constexpr,
    G: tl.constexpr,
    S: tl.constexpr,
    NORM_TYPE: tl.constexpr,
    OUT_FLAG: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RENORM: tl.constexpr,
    GROUP_MODE: tl.constexpr,
    SCALING: tl.constexpr,
    EPS: tl.constexpr,
    X_IS_F32: tl.constexpr,
    Y_IS_F32: tl.constexpr,
    X_DTYPE: tl.constexpr,
    Y_DTYPE: tl.constexpr,
    CORES: tl.constexpr,
):
    """One persistent program owns strided batches of ``M`` routing rows."""
    E: tl.constexpr = G * S
    NC: tl.constexpr = E // 64
    Y_PAD: tl.constexpr = 64 if Y_IS_F32 else 128
    tl.static_assert(E % 64 == 0, "TLA experiment requires E multiple of 64")
    tl.static_assert(E <= 4096, "TLA experiment requires E <= 4096")
    tl.static_assert(G <= 64 and K <= 64, "TLA experiment requires G<=64 and K<=64")
    tl.static_assert(M >= 1 and M <= 16, "TLA experiment requires 1 <= BLOCK_M <= 8")

    if X_IS_F32:
        x_ub = tla.alloc_buffer((M, E), tl.float32, space="ub", versions=2)
    else:
        x_raw = tla.alloc_buffer((M, E), X_DTYPE, space="ub", versions=2)
        x_ub = tla.alloc_buffer((M, E), tl.float32, space="ub", versions=2)
    norm_ub = tla.alloc_buffer((M, E), tl.float32, space="ub", versions=2)
    if HAS_BIAS:
        if X_IS_F32:
            bias_ub = tla.alloc_buffer((E,), tl.float32, space="ub")
        else:
            bias_raw = tla.alloc_buffer((E,), X_DTYPE, space="ub")
            bias_ub = tla.alloc_buffer((E,), tl.float32, space="ub")
    if HAS_BIAS or G > 1:
        sel_ub = tla.alloc_buffer((M, E), tl.float32, space="ub", versions=2)
    y_ub = tla.alloc_buffer((M, Y_PAD), Y_DTYPE, space="ub")
    idx_ub = tla.alloc_buffer((M, 64), tl.int32, space="ub")

    if HAS_BIAS:
        bv = tla.view(BIAS, (E,), strides=(1,), offsets=(0,))
        if X_IS_F32:
            tla.copy(bv, bias_ub)
        else:
            tla.copy(bv, bias_raw)
            with tla.scope.simd():
                for c in tl.static_range(NC):
                    bc = tla.simd.vcvt(tla.simd.vlds(bias_raw[c * 64], dist="UNPK_B16"), tl.float32)
                    tla.simd.vsts(bias_ub[c * 64], bc, dist="NORM_B32")

    for base in range(tl.program_id(0) * M, n_rows, CORES * M):
        # Batch GM->UB load. The tail falls back to per-row copies, but the
        # main path is one 2D DMA for M rows.
        if base + M <= n_rows:
            xv = tla.view(X, (M, E), strides=(STRIDE_X, 1), offsets=(base, 0))
            if X_IS_F32:
                tla.copy(xv, x_ub)
            else:
                tla.copy(xv, x_raw)
        else:
            for tr in tl.static_range(M):
                if base + tr < n_rows:
                    xv = tla.view(X, (1, E), strides=(STRIDE_X, 1), offsets=(base + tr, 0))
                    if X_IS_F32:
                        tla.copy(xv, tla.subview(x_ub, offsets=(tr, 0), sizes=(1, E)))
                    else:
                        tla.copy(xv, tla.subview(x_raw, offsets=(tr, 0), sizes=(1, E)))

        if not X_IS_F32:
            with tla.scope.simd():
                for tr in tl.static_range(M):
                    for c in tl.static_range(NC):
                        xc = tla.simd.vcvt(tla.simd.vlds(x_raw[tr, c * 64], dist="UNPK_B16"), tl.float32)
                        tla.simd.vsts(x_ub[tr, c * 64], xc, dist="NORM_B32")

        rows_this = tl.minimum(M, n_rows - base)
        for br in range(0, rows_this):
            with tla.scope.simd():
                neg_inf = tla.simd.vsplat(float("-inf"), tl.float32)
                zero = tla.simd.vsplat(0.0, tl.float32)
                one = tla.simd.vsplat(1.0, tl.float32)
                k_lane = tla.simd.vci(0)
                gid_lane = tla.simd.vci(0)

                xs = ()
                for c in tl.static_range(NC):
                    xs = xs + (tla.simd.vlds(x_ub[br, c * 64], dist="NORM"),)

                norm = ()
                if NORM_TYPE == 0:
                    m = neg_inf
                    for c in tl.static_range(NC):
                        m = tla.simd.vmax(m, tla.simd.vreduce_max(xs[c]))
                    m = tla.simd.vbroadcast(m)
                    numer = ()
                    den = zero
                    for c in tl.static_range(NC):
                        e = tla.simd.vexp(tla.simd.vsub(xs[c], m))
                        numer = numer + (e,)
                        den = tla.simd.vadd(den, tla.simd.vreduce_sum(e))
                    inv = tla.simd.vbroadcast(tla.simd.vdiv(one, den))
                    for c in tl.static_range(NC):
                        nv = tla.simd.vmul(numer[c], inv)
                        norm = norm + (nv,)
                        tla.simd.vsts(norm_ub[br, c * 64], nv, dist="NORM_B32")
                elif NORM_TYPE == 1 or NORM_TYPE == 3:
                    for c in tl.static_range(NC):
                        nv = tla.simd.vdiv(one, tla.simd.vadds(tla.simd.vexp(tla.simd.vneg(xs[c])), 1.0))
                        norm = norm + (nv,)
                        tla.simd.vsts(norm_ub[br, c * 64], nv, dist="NORM_B32")
                else:
                    for c in tl.static_range(NC):
                        ax = tla.simd.vabs(xs[c])
                        sp = tla.simd.vadd(
                            tla.simd.vln(tla.simd.vadds(tla.simd.vexp(tla.simd.vneg(ax)), 1.0)),
                            tla.simd.vmax(xs[c], zero),
                        )
                        nv = tla.simd.vsqrt(sp)
                        norm = norm + (nv,)
                        tla.simd.vsts(norm_ub[br, c * 64], nv, dist="NORM_B32")

                sel = ()
                if HAS_BIAS:
                    for c in tl.static_range(NC):
                        b = tla.simd.vlds(bias_ub[c * 64], dist="NORM")
                        sv = tla.simd.vadd(norm[c], b)
                        sel = sel + (sv,)
                        tla.simd.vsts(sel_ub[br, c * 64], sv, dist="NORM_B32")
                else:
                    sel = norm
                    if G > 1:
                        for c in tl.static_range(NC):
                            tla.simd.vsts(sel_ub[br, c * 64], norm[c], dist="NORM_B32")

                if G > 1:
                    tla.simd.membar("VST_VLD")
                    gscore = neg_inf
                    for g in tl.static_range(G):
                        first = neg_inf
                        for off in tl.static_range(0, S, 64):
                            chunk = tla.simd.vlds(sel_ub[br, g * S + off], dist="NORM")
                            valid = tla.simd.vcmps(tla.simd.vci(off), S, op=tla.simd.CMP.LT)
                            chunk = tla.simd.vsel(valid, chunk, neg_inf)
                            first = tla.simd.vmax(first, tla.simd.vreduce_max(chunk))
                        first = tla.simd.vbroadcast(first)
                        if GROUP_MODE == 0:
                            gs = first
                        else:
                            pos = tla.simd.vsplat(S, tl.int32)
                            for off in tl.static_range(0, S, 64):
                                chunk = tla.simd.vlds(sel_ub[br, g * S + off], dist="NORM")
                                valid = tla.simd.vcmps(tla.simd.vci(off), S, op=tla.simd.CMP.LT)
                                chunk = tla.simd.vsel(valid, chunk, neg_inf)
                                pos_cand = tla.simd.vsel(
                                    tla.simd.vcmp(chunk, first, op=tla.simd.CMP.EQ),
                                    tla.simd.vci(off),
                                    tla.simd.vsplat(S, tl.int32),
                                )
                                pos = tla.simd.vmin(pos, tla.simd.vreduce_min(pos_cand))
                            first_pos = tla.simd.vbroadcast(pos)
                            second = neg_inf
                            for off in tl.static_range(0, S, 64):
                                chunk = tla.simd.vlds(sel_ub[br, g * S + off], dist="NORM")
                                lanes = tla.simd.vci(off)
                                valid = tla.simd.vcmps(lanes, S, op=tla.simd.CMP.LT)
                                chunk = tla.simd.vsel(valid, chunk, neg_inf)
                                chunk2 = tla.simd.vsel(
                                    tla.simd.vcmp(lanes, first_pos, op=tla.simd.CMP.EQ), neg_inf, chunk
                                )
                                second = tla.simd.vmax(second, tla.simd.vreduce_max(chunk2))
                            gs = tla.simd.vadd(first, tla.simd.vbroadcast(second))
                        gscore = tla.simd.vsel(tla.simd.vcmps(gid_lane, g, op=tla.simd.CMP.EQ), gs, gscore)

                    gsel = tla.simd.vsel(tla.simd.vcmps(gid_lane, 0, op=tla.simd.CMP.LT), one, zero)
                    gwork = gscore
                    for step in tl.static_range(K_GROUP):
                        best_g = tla.simd.vbroadcast(tla.simd.vreduce_max(gwork))
                        bg = tla.simd.vbroadcast(
                            tla.simd.vreduce_min(
                                tla.simd.vsel(
                                    tla.simd.vcmp(gwork, best_g, op=tla.simd.CMP.EQ),
                                    gid_lane,
                                    tla.simd.vsplat(G, tl.int32),
                                )
                            )
                        )
                        gsel = tla.simd.vsel(tla.simd.vcmp(gid_lane, bg, op=tla.simd.CMP.EQ), one, gsel)
                        gwork = tla.simd.vsel(tla.simd.vcmp(gid_lane, bg, op=tla.simd.CMP.EQ), neg_inf, gwork)
                    new_sel = ()
                    for c in tl.static_range(NC):
                        lidx = tla.simd.vci(c * 64)
                        gm = zero
                        for g in tl.static_range(G):
                            gflag = tla.simd.vbroadcast(
                                tla.simd.vreduce_max(
                                    tla.simd.vsel(tla.simd.vcmps(gid_lane, g, op=tla.simd.CMP.EQ), gsel, zero)
                                )
                            )
                            in_g = tla.simd.pand(
                                tla.simd.vcmps(lidx, g * S, op=tla.simd.CMP.GE),
                                tla.simd.vcmps(lidx, (g + 1) * S, op=tla.simd.CMP.LT),
                            )
                            gm = tla.simd.vsel(in_g, gflag, gm)
                        new_sel = new_sel + (
                            tla.simd.vsel(tla.simd.vcmps(gm, 0.5, op=tla.simd.CMP.GT), sel[c], neg_inf),
                        )
                    sel = new_sel

                sc = sel
                y_vec = zero
                idx_vec = tla.simd.vsplat(0, tl.int32)
                for step in range(0, K):
                    best = tla.simd.vbroadcast(tla.simd.vreduce_max(sc))
                    top_cand = ()
                    for c in tl.static_range(NC):
                        top_cand = top_cand + (
                            tla.simd.vsel(
                                tla.simd.vcmp(sc[c], best, op=tla.simd.CMP.EQ),
                                tla.simd.vci(c * 64),
                                tla.simd.vsplat(E, tl.int32),
                            ),
                        )
                    winner = tla.simd.vbroadcast(tla.simd.vreduce_min(top_cand))
                    w = zero
                    for c in tl.static_range(NC):
                        w = tla.simd.vadd(
                            w,
                            tla.simd.vsel(
                                tla.simd.vcmp(tla.simd.vci(c * 64), winner, op=tla.simd.CMP.EQ), norm[c], zero
                            ),
                        )
                    weight = tla.simd.vbroadcast(tla.simd.vreduce_sum(w))
                    kmask = tla.simd.vcmps(k_lane, step, op=tla.simd.CMP.EQ)
                    y_vec = tla.simd.vsel(kmask, weight, y_vec)
                    idx_vec = tla.simd.vsel(kmask, winner, idx_vec)
                    new_sc = ()
                    for c in tl.static_range(NC):
                        new_sc = new_sc + (
                            tla.simd.vsel(
                                tla.simd.vcmp(tla.simd.vci(c * 64), winner, op=tla.simd.CMP.EQ), neg_inf, sc[c]
                            ),
                        )
                    sc = new_sc

                if RENORM:
                    ysum = tla.simd.vbroadcast(
                        tla.simd.vreduce_sum(tla.simd.vsel(tla.simd.vcmps(k_lane, K, op=tla.simd.CMP.LT), y_vec, zero))
                    )
                    y_vec = tla.simd.vdiv(y_vec, tla.simd.vadds(ysum, EPS))
                y_vec = tla.simd.vmuls(y_vec, SCALING)
                if Y_IS_F32:
                    tla.simd.vsts(y_ub[br, 0], y_vec, dist="NORM_B32")
                else:
                    y_narrow = tla.simd.vcvt(y_vec, Y_DTYPE)
                    y_packed, _ = tla.simd.vdintlv(y_narrow, y_narrow)
                    tla.simd.vsts(y_ub[br, 0], y_packed, dist="NORM_B16")
                tla.simd.vsts(idx_ub[br, 0], idx_vec, dist="NORM_B32")

        if base + M <= n_rows:
            # Batch UB->GM store: y/idx use a compact (M,K) subview with the
            # UB row pitch, so the GM write is still one DMA per tensor.
            ysrc = tla.subview(y_ub, offsets=(0, 0), sizes=(M, K))
            isrc = tla.subview(idx_ub, offsets=(0, 0), sizes=(M, K))
            yv = tla.view(Y, (M, K), strides=(K, 1), offsets=(base, 0))
            iv = tla.view(IDX, (M, K), strides=(K, 1), offsets=(base, 0))
            tla.copy(ysrc, yv)
            tla.copy(isrc, iv)
            if OUT_FLAG:
                nv = tla.view(NORM_OUT, (M, E), strides=(E, 1), offsets=(base, 0))
                tla.copy(norm_ub, nv)
        else:
            for tr in tl.static_range(M):
                if base + tr < n_rows:
                    tla.copy(
                        tla.subview(y_ub, offsets=(tr, 0), sizes=(1, K)),
                        tla.view(Y, (1, K), strides=(K, 1), offsets=(base + tr, 0)),
                    )
                    tla.copy(
                        tla.subview(idx_ub, offsets=(tr, 0), sizes=(1, K)),
                        tla.view(IDX, (1, K), strides=(K, 1), offsets=(base + tr, 0)),
                    )
                    if OUT_FLAG:
                        tla.copy(
                            tla.subview(norm_ub, offsets=(tr, 0), sizes=(1, E)),
                            tla.view(NORM_OUT, (1, E), strides=(E, 1), offsets=(base + tr, 0)),
                        )


def moe_gating_top_k_tla(
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
    cores: int = 56,
    block_m: Optional[int] = None,
):
    """Run the experimental TLA routing kernel with batched rows.

    ``block_m`` defaults to the largest value in ``{1, 2, 4, 8}`` that keeps
    all ``cores`` busy, capped by a conservative UB estimate. Pass it
    explicitly to reproduce a particular batch configuration.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError("moe_gating_top_k_tla expects a 2D torch.Tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("moe_gating_top_k_tla supports FP16, BF16 and FP32")
    n_rows, num_experts = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    if num_experts % 64 != 0:
        raise ValueError("TLA experiment requires num_experts divisible by 64")
    if num_experts > 4096:
        raise ValueError("TLA experiment requires num_experts <= 4096")
    if num_experts % group_count != 0:
        raise ValueError("group_count must divide num_experts")
    if group_count > 64 or k > 64:
        raise ValueError("TLA experiment requires group_count<=64 and k<=64")
    if not 1 <= k <= (num_experts // group_count) * k_group:
        raise ValueError("k is outside the selected-group expert range")
    if not 1 <= k_group <= group_count:
        raise ValueError("k_group must be in [1, group_count]")
    if group_select_mode not in (0, 1):
        raise ValueError("group_select_mode must be 0 or 1")
    if norm_type not in (0, 1, 2, 3):
        raise ValueError("norm_type must be in 0..3")
    if bias is not None:
        if bias.ndim != 1 or bias.numel() != num_experts:
            raise ValueError("bias must have shape (num_experts,)")
        if bias.dtype != x.dtype or bias.device != x.device:
            raise TypeError("bias dtype/device must match x")
        bias = bias.contiguous()

    y = torch.empty((n_rows, k), dtype=x.dtype, device=x.device)
    idx = torch.empty((n_rows, k), dtype=torch.int32, device=x.device)
    norm = torch.empty((n_rows, num_experts), dtype=torch.float32, device=x.device) if out_flag else None
    if n_rows == 0:
        return y, idx, norm

    # Keep all cores busy and stay inside a conservative UB budget.
    ub_cap = max(1, min(8, 8192 // num_experts))
    if block_m is None:
        block_m = 1
        for candidate in (8, 4, 2):
            if (n_rows + candidate - 1) // candidate >= cores:
                block_m = candidate
                break
    block_m = max(1, min(int(block_m), ub_cap, n_rows, 8))

    do_renorm = norm_type in (1, 2) or (norm_type in (0, 3) and renorm == 1)
    x_is_f32 = x.dtype == torch.float32
    y_is_f32 = y.dtype == torch.float32
    x_dtype = tl.float32 if x_is_f32 else (tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16)
    y_dtype = tl.float32 if y_is_f32 else x_dtype
    grid = min(int(cores), (n_rows + block_m - 1) // block_m)
    _moe_gating_top_k_tla_kernel[(grid,)](
        x,
        bias if bias is not None else x,
        y,
        idx,
        norm if norm is not None else x,
        n_rows,
        STRIDE_X=x.stride(0),
        K=int(k),
        M=block_m,
        K_GROUP=int(k_group),
        G=int(group_count),
        S=num_experts // int(group_count),
        NORM_TYPE=int(norm_type),
        OUT_FLAG=bool(out_flag),
        HAS_BIAS=bias is not None,
        RENORM=do_renorm,
        GROUP_MODE=int(group_select_mode),
        SCALING=float(routed_scaling_factor),
        EPS=float(eps),
        X_IS_F32=x_is_f32,
        Y_IS_F32=y_is_f32,
        X_DTYPE=x_dtype,
        Y_DTYPE=y_dtype,
        CORES=grid,
    )
    return y, idx, norm
