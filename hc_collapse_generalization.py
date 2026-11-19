#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the mojo_opset project.
"""hc_collapse 泛化用例驱动脚本（单文件，放于仓库根目录，不改动仓库代码）。

为 ``mojo_opset.functions.hc_collapse`` 生成大规模泛化精度用例：
对每个 dtype 独立生成一套确定的用例矩阵（shape + 布局 + 数据种子），
在被测设备上运行算子并与标杆实现对比，输出逐用例结果与汇总报告。

特点
----
* 不识别机型：脚本自身不含任何 a2/a5/910B 等机型分支。NPU 模式直接调用
  ``functions.hc_collapse(x, pre)``，由仓库配置按目标自动分发实现；
  CPU 模式固定 ``implementation="torch"``。
* 每个 dtype 默认 700 条用例（ curated 边界/代表性 shape + 确定性随机补充）；
  ``--quick`` 快速模式每个 dtype 精选 60 条代表性用例。
* 默认 ``--device npu``：被测算子（triton，按仓库配置分发）与标杆（torch 参考实现）
  都在 NPU 上执行，同设备对比。
* ``--device cpu``：数据生成与标杆（torch 参考实现）都在 CPU 上执行；被测算子
  仍在 NPU 上运行（不切换实现），actual 搬回 CPU 与标杆对比。两种模式均需
  NPU 可用，``--device`` 只决定标杆位置。
* 用例矩阵与数据完全由 ``--seed`` 决定，可复现；``--case-id`` 可重放单条用例。
* 进程级崩溃隔离：Ascend 设备上个别 shape 可能直接终止进程（ERR99999）。
  默认每个 dtype 的用例在子进程中成批执行、逐条落盘；子进程崩溃时自动
  单条重试定位崩溃用例并记录为 crash，随后续跑剩余用例。
  ``--in-process`` 可关闭隔离（快，但崩溃会中断整轮）。
* UB 预算自适应：内核侧将 ``hc_mult × BLOCK_H`` 的 Unified Buffer 预算并入
  tile 宽度计算（大 ``hc_mult`` 自动缩小 tile），仅当 ``hc_mult`` 本身超预算时
  抛 ``AssertionError``（消息含 "exceeds Unified Buffer budget"）；脚本识别
  到该校验即按 pass 记录（带 ``guard`` 标记并在汇总中单独计数）。

用法示例
--------
::

    # 完整泛化（3 dtype × 700 用例，NPU，默认标杆=torch 参考实现@NPU）
    python hc_collapse_generalization.py

    # 快速模式（3 dtype × 60 代表性用例）
    python hc_collapse_generalization.py --quick

    # CPU 标杆模式（数据生成与标杆在 CPU，被测算子仍在 NPU）
    python hc_collapse_generalization.py --device cpu --quick

    # 自定义用例数 / dtype / 种子 / 报告
    python hc_collapse_generalization.py --cases-per-dtype 100 \
        --dtypes bfloat16 float16 --seed 2026 --report /tmp/hc_gen.json

    # 重放单条用例（编号见 --list 或报告）
    python hc_collapse_generalization.py --case-id bfloat16:0007

    # 仅打印用例矩阵不执行
    python hc_collapse_generalization.py --list

退出码：0=全部通过；1=存在 fail/error/crash 用例；2=参数或环境错误。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()

DTYPES = ("bfloat16", "float16", "float32")

# 输入布局：contiguous / x 最后一维跨步 / x 流维度跨步 / pre 转置跨步 / 组合。
LAYOUTS = (
    "contiguous",
    "x_inner",           # x[..., ::2]：最内层 stride=2
    "x_stream",          # x 由 [m, T, H] 转置而来：流维度不连续
    "pre_transposed",    # pre.t().contiguous().t()：token 维 stride=1
    "x_inner+pre_transposed",
)
_LAYOUT_BASE = ("contiguous", "x_inner", "x_stream", "pre_transposed", "x_inner+pre_transposed")
_LAYOUT_WEIGHTS = (0.50, 0.15, 0.15, 0.10, 0.10)

FULL_CASES_PER_DTYPE = 700
QUICK_CASES_PER_DTYPE = 60

# 随机用例的 x 张量元素预算（分配量；x_inner 布局翻倍）。
RANDOM_ELEMENT_BUDGET = 1 << 24        # 16.7M 元素（bf16 约 32MiB）
# curated 用例超出该元素量时降级为 contiguous 布局。
CURATED_ELEMENT_BUDGET = 1 << 26       # 67.1M 元素

# 数值容差 (rtol, atol)，口径沿用 tests/_checks.py 与既有精度用例。
TOLERANCES = {
    "bfloat16": (1e-2, 1.6e-2),
    "float16": (1e-3, 1e-3),
    "float32": (1e-4, 1e-4),
}


# --------------------------------------------------------------------------
# 用例模型与矩阵生成（纯标准库，父/子进程共用，完全由 --seed 决定）
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    index: int
    num_tokens: int
    hc_mult: int
    hidden_size: int
    layout: str
    origin: str

    @property
    def key(self):
        return (self.num_tokens, self.hc_mult, self.hidden_size, self.layout)

    def case_id(self, dtype):
        return f"{dtype}:{self.index:04d}"


def _quick_matrix():
    """60 条代表性用例：边界、上游矩阵、hc_mult 扫描、hidden 扫描（含
    9216/12288 tile 上限与超限多 block）、token 尾部、serving shape、
    宽 token 窄 hidden、跨步布局。"""
    matrix = [
        # 退化与最小边界
        (0, 4, 5120, "contiguous"),
        (3, 4, 0, "contiguous"),
        (1, 1, 1, "contiguous"),
        (2, 1, 513, "contiguous"),
        # 上游 vLLM mHC 代表 shape
        (1, 4, 5120, "contiguous"),
        (1, 4, 5120, "x_inner"),
        (7, 8, 5137, "contiguous"),
        (7, 8, 5137, "pre_transposed"),
        (33, 4, 5137, "contiguous"),
        (33, 4, 5137, "x_inner"),
        (128, 4, 5120, "contiguous"),
        (128, 4, 4096, "contiguous"),
        # serving 规模
        (256, 4, 5120, "contiguous"),
        (1024, 4, 5120, "contiguous"),
        (1024, 4, 7168, "contiguous"),
        (2048, 4, 5120, "contiguous"),
        # hc_mult 扫描（含 1、素数与 16）
        (17, 1, 1023, "contiguous"),
        (17, 2, 1023, "contiguous"),
        (17, 3, 1024, "contiguous"),
        (17, 5, 511, "contiguous"),
        (17, 6, 1025, "x_inner"),
        (17, 7, 65, "contiguous"),
        (17, 9, 127, "contiguous"),
        (17, 12, 256, "contiguous"),
        (17, 16, 512, "pre_transposed"),
        # hidden 宽度扫描：极小 / 2^n±1 / 常见模型宽度 / tile 上下限与超限
        (9, 4, 1, "contiguous"),
        (9, 4, 2, "contiguous"),
        (9, 4, 3, "contiguous"),
        (9, 4, 17, "contiguous"),
        (9, 4, 31, "contiguous"),
        (9, 4, 33, "x_inner"),
        (9, 4, 63, "contiguous"),
        (9, 4, 64, "contiguous"),
        (9, 4, 65, "contiguous"),
        (9, 4, 127, "contiguous"),
        (9, 4, 128, "contiguous"),
        (9, 4, 129, "contiguous"),
        (9, 4, 255, "pre_transposed"),
        (9, 4, 256, "contiguous"),
        (9, 4, 257, "contiguous"),
        (9, 4, 511, "contiguous"),
        (9, 4, 513, "contiguous"),
        (9, 4, 1023, "contiguous"),
        (9, 4, 1024, "contiguous"),
        (9, 4, 1025, "contiguous"),
        (9, 4, 2048, "contiguous"),
        (9, 4, 4095, "contiguous"),
        (9, 4, 4097, "x_inner"),
        (9, 4, 5120, "contiguous"),
        (9, 4, 6144, "contiguous"),
        (9, 4, 9216, "contiguous"),
        (9, 4, 12288, "contiguous"),
        (9, 4, 12289, "contiguous"),
        (9, 4, 16384, "contiguous"),
        (9, 4, 18432, "x_inner"),
        (9, 4, 24576, "contiguous"),
        # 宽 token 窄 hidden（序列并行尾 block）
        (65537, 4, 65, "contiguous"),
        (131072, 2, 33, "contiguous"),
        # 跨步布局补充
        (2, 4, 5120, "x_stream"),
        (65, 4, 65, "x_inner+pre_transposed"),
    ]
    assert len(matrix) == QUICK_CASES_PER_DTYPE, len(matrix)
    assert len({(t, m, h, lo) for t, m, h, lo in matrix}) == QUICK_CASES_PER_DTYPE
    return matrix


_HIDDEN_SWEEP = (
    [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
    + [255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049]
    + [4095, 4096, 4097, 5119, 5120, 5121, 5137, 6144, 7168, 8191, 8192, 8193]
    + [9216, 10240, 12287, 12288, 12289, 13569, 16384, 18432, 20480, 24576]
)

_TOKEN_TAILS = (
    [2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257]
    + [999, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4097]
)


def _curated_full_matrix():
    """完整模式的 curated 用例（确定性，无随机），返回 (T, m, H, tag) 列表。"""
    entries = []

    def add(shapes, tag):
        for t, m, h in shapes:
            entries.append((int(t), int(m), int(h), tag))

    add(
        [
            (0, 1, 64), (0, 4, 5120), (0, 16, 65), (3, 4, 0), (1, 4, 0), (0, 1, 0),
            (1, 1, 1), (1, 1, 2), (1, 1, 3), (2, 1, 513), (1, 2, 1),
        ],
        "degenerate",
    )
    add(
        [
            (0, 4, 5120), (1, 4, 5120), (33, 4, 5137), (128, 4, 5120),
            (1024, 4, 5120), (2, 1, 513), (7, 8, 5137), (65537, 4, 65),
        ],
        "upstream",
    )
    add([(17, m, 1023) for m in range(1, 17)], "mult-sweep")
    add([(9, 4, h) for h in _HIDDEN_SWEEP], "hidden-sweep")
    add([(t, 4, 5120) for t in _TOKEN_TAILS], "token-tails")
    add(
        [
            (1, 4, 5120), (64, 4, 5120), (256, 4, 5120), (1280, 4, 5120),
            (2048, 4, 5120), (4096, 4, 5120), (33, 4, 5137), (7, 8, 5137),
            (128, 4, 4096), (1024, 4, 7168), (2048, 4, 4096), (512, 4, 7168),
        ],
        "serving",
    )
    add(
        [
            (64, 4, 9216), (64, 4, 12288), (33, 4, 12289), (16, 4, 16384),
            (16, 4, 18432), (9, 4, 24576), (100, 3, 12288), (9, 5, 12289),
            # hc_mult × tile 压力阶梯（tile=12288 时 m 增大）
            (2, 5, 12288), (2, 6, 12288), (2, 8, 12288), (5, 16, 12288),
            (1, 8, 24576), (9, 6, 13569), (2, 16, 24576),
        ],
        "tile-boundary",
    )
    add(
        [(65537, 4, 65), (131072, 2, 33), (100000, 1, 7), (65536, 8, 17)],
        "wide-token",
    )
    return entries


def _effective_elements(num_tokens, hc_mult, hidden_size, layout):
    multiplier = 2 if "x_inner" in layout else 1
    return num_tokens * hc_mult * hidden_size * multiplier


def _random_case(rng, seen):
    """确定性随机采样一条满足内存预算且不重复的用例；失败返回 None。"""
    layout = rng.choices(_LAYOUT_BASE, weights=_LAYOUT_WEIGHTS, k=1)[0]
    mult_pool = [1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 10, 11, 12, 12, 14, 16]
    hc_mult = rng.choice(mult_pool)
    tokens = max(1, min(65536, int(2 ** rng.uniform(0.0, 16.0))))
    if rng.random() < 0.3:  # 部分 Hidden 采样偏向大宽度（覆盖多 block 路径）
        hidden = max(1, min(24576, int(2 ** rng.uniform(10.0, 14.58))))
    else:
        hidden = max(1, min(24576, int(2 ** rng.uniform(0.0, 14.58))))
    if _effective_elements(tokens, hc_mult, hidden, layout) > RANDOM_ELEMENT_BUDGET:
        return None
    key = (tokens, hc_mult, hidden, layout)
    if key in seen:
        return None
    return key


def build_cases(dtype, *, quick, cases_per_dtype, seed):
    """构建某个 dtype 的用例列表（对所有 dtype 使用同一份确定性 shape 矩阵，
    数据种子按 dtype 区分）。"""
    del dtype  # shape 矩阵与 dtype 无关；数据种子在运行期按 dtype 混入。
    if quick:
        rows = [(t, m, h, lo, "curated/quick") for t, m, h, lo in _quick_matrix()]
    else:
        seen = set()
        rows = []
        # curated：轮转分配布局，超预算降级为 contiguous。
        for pos, (t, m, h, tag) in enumerate(_curated_full_matrix()):
            layout = _LAYOUT_BASE[pos % len(_LAYOUT_BASE)]
            if _effective_elements(t, m, h, layout) > CURATED_ELEMENT_BUDGET:
                layout = "contiguous"
            key = (t, m, h, layout)
            if key in seen:
                continue
            seen.add(key)
            rows.append((t, m, h, layout, f"curated/{tag}"))
        if len(rows) > cases_per_dtype:
            rows = rows[:cases_per_dtype]
        rng = random.Random(seed)
        attempts = 0
        while len(rows) < cases_per_dtype and attempts < cases_per_dtype * 500:
            attempts += 1
            key = _random_case(rng, seen)
            if key is None:
                continue
            seen.add(key)
            rows.append((*key, "random"))
        if len(rows) < cases_per_dtype:
            raise RuntimeError(
                f"仅能生成 {len(rows)}/{cases_per_dtype} 条不重复用例，请降低 --cases-per-dtype"
            )
    return tuple(Case(i, t, m, h, lo, origin) for i, (t, m, h, lo, origin) in enumerate(rows))


# --------------------------------------------------------------------------
# 用例执行（worker 侧；torch / mojo_opset 在此惰性导入）
# --------------------------------------------------------------------------


def _torch_dtype(name):
    import torch

    return getattr(torch, name)


def _data_seed(seed, dtype, index):
    return (
        seed * 1_000_003 + index * 10_007 + DTYPES.index(dtype) * 101_390_423
    ) & 0x7FFF_FFFF_FFFF_FFFF


def _make_base_inputs(case, dtype_name, seed):
    """在 CPU 上生成 contiguous 输入原件（跨设备确定性；标杆与被测共用）。"""
    import torch

    dtype = _torch_dtype(dtype_name)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_data_seed(seed, dtype_name, case.index))

    tokens, mult, hidden = case.num_tokens, case.hc_mult, case.hidden_size
    width = hidden * 2 if "x_inner" in case.layout else hidden
    if case.layout == "x_stream":
        base = torch.randn(mult, tokens, hidden, dtype=torch.float32, generator=generator)
    else:
        base = torch.randn(tokens, mult, width, dtype=torch.float32, generator=generator)
    x = base.to(dtype)
    pre = torch.rand(tokens, mult, dtype=torch.float32, generator=generator)
    return x, pre


def _apply_layout(case, x, pre):
    """在张量所在设备上应用布局变换，保持跨步 stride 语义。"""
    if case.layout == "x_stream":
        x = x.transpose(0, 1)
    if "x_inner" in case.layout:
        x = x[..., ::2]
    if "pre_transposed" in case.layout:
        pre = pre.t().contiguous().t()
    return x, pre


# 内核 UB 预算校验的识别标记（AssertionError + 稳定消息子串）：命中视为预期行为。
_UB_GUARD_MARKER = "exceeds Unified Buffer budget"


def _is_tile_budget_guard(exc):
    return (
        isinstance(exc, AssertionError)
        and _UB_GUARD_MARKER in str(exc)
    )


def _evaluate(actual, expected, case, dtype_name):
    """返回 (problems, stats, worst_samples)。"""
    import torch

    dtype = _torch_dtype(dtype_name)
    problems = []
    if tuple(actual.shape) != (case.num_tokens, case.hidden_size):
        problems.append(f"输出 shape {tuple(actual.shape)} != ({case.num_tokens}, {case.hidden_size})")
    if actual.dtype != dtype:
        problems.append(f"输出 dtype {actual.dtype} != {dtype}")
    if not actual.is_contiguous():
        problems.append("输出不是 contiguous")

    stats = {"mismatches": 0, "max_abs_err": 0.0, "max_rel_err": 0.0}
    worst = []
    a = actual.detach().float()
    e = expected.detach().float()
    if a.numel():
        finite = torch.isfinite(a)
        n_nonfinite = int((~finite).sum().item())
        if n_nonfinite:
            problems.append(f"{n_nonfinite}/{a.numel()} 个非有限值")
        rtol, atol = TOLERANCES[dtype_name]
        close = torch.isclose(a, e, rtol=rtol, atol=atol, equal_nan=False)
        bad = int((~close).sum().item())
        diff = (a - e).abs()
        stats["max_abs_err"] = float(diff.max().item())
        stats["max_rel_err"] = float((diff / e.abs().clamp_min(1e-6)).max().item())
        if bad:
            flat_diff = diff.flatten()
            flat_bad = (~close).flatten()
            positions = flat_bad.nonzero().flatten()[:3].tolist()
            a_flat = a.flatten()
            e_flat = e.flatten()
            worst = [
                {
                    "flat_index": int(i),
                    "actual": float(a_flat[i].item()),
                    "expected": float(e_flat[i].item()),
                    "abs_err": float(flat_diff[i].item()),
                }
                for i in positions
            ]
            problems.append(
                f"{bad}/{a.numel()} 个元素超差 (rtol={rtol:g}, atol={atol:g})"
            )
    return problems, stats, worst


def _sanitize(value):
    """递归把非有限浮点数替换为 None，保证 JSON 报告对严格解析器合法。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def _run_one_case(case, dtype_name, args):
    """执行单条用例并返回完整记录 dict（不抛异常）。"""
    import torch

    from mojo_opset.functions import hc_collapse

    record = {"dtype": dtype_name, "case_id": case.case_id(dtype_name), **{
        field: getattr(case, field)
        for field in ("index", "num_tokens", "hc_mult", "hidden_size", "layout", "origin")
    }}
    start = time.perf_counter()
    try:
        x_base, pre_base = _make_base_inputs(case, dtype_name, args.seed)
        with torch.no_grad():
            # 被测算子始终在 NPU 上执行（不因标杆位置切换实现）。
            x_dev, pre_dev = _apply_layout(case, x_base.to("npu"), pre_base.to("npu"))
            impl = None if args.implementation == "auto" else args.implementation
            try:
                actual = (
                    hc_collapse(x_dev, pre_dev)
                    if impl is None
                    else hc_collapse(x_dev, pre_dev, implementation=impl)
                )
            except AssertionError as exc:
                if not _is_tile_budget_guard(exc):
                    raise
                # 命中内核 UB 预算校验：配置超出硬件预算属预期拒绝，按 pass 记录。
                record.update(
                    {
                        "status": "pass",
                        "guard": True,
                        "message": f"命中 UB 预算校验: {exc}"[:2000],
                        "mismatches": 0,
                        "max_abs_err": None,
                        "max_rel_err": None,
                        "worst": [],
                    }
                )
                record["elapsed_s"] = round(time.perf_counter() - start, 4)
                return record
            if args.device == "npu":
                expected = hc_collapse(x_dev, pre_dev, implementation="torch")
            else:
                # cpu 模式：标杆在 CPU 上对同一份逻辑输入计算，actual 搬回 CPU 对比。
                x_ref, pre_ref = _apply_layout(case, x_base, pre_base)
                expected = hc_collapse(x_ref, pre_ref, implementation="torch")
                actual = actual.to("cpu")
        problems, stats, worst = _evaluate(actual, expected, case, dtype_name)
        record.update(stats)
        record["worst"] = worst
        record["status"] = "pass" if not problems else "fail"
        if problems:
            record["message"] = "; ".join(problems)[:2000]
    except Exception as exc:  # 进程内存活的可捕获错误
        record.update({"mismatches": -1, "max_abs_err": None, "max_rel_err": None, "worst": []})
        record["status"] = "error"
        record["message"] = f"{type(exc).__name__}: {exc}"[:2000]
    record["elapsed_s"] = round(time.perf_counter() - start, 4)
    return _sanitize(record)


def _worker_main(args):
    import torch

    # 被测算子始终在 NPU 上执行，--device 仅决定标杆位置，两种模式都需要 NPU。
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("worker: NPU 不可用", file=sys.stderr)
        return 3
    torch.npu.set_device(args.device_id)
    results_path = Path(args.results_file)
    for dtype in args.dtypes:
        cases = build_cases(
            dtype, quick=args.quick, cases_per_dtype=args.cases_per_dtype, seed=args.seed
        )
        if args.only_ids is not None:
            wanted = set(args.only_ids)
            selected = [c for c in cases if c.index in wanted]
        else:
            selected = cases[args.start : args.end]
        with results_path.open("a", encoding="utf-8") as handle:
            for case in selected:
                record = _run_one_case(case, dtype, args)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if args.verbose:
                    print(
                        f"[worker] {record['case_id']} status={record['status']} "
                        f"elapsed={record['elapsed_s']}s",
                        flush=True,
                    )
    return 0


# --------------------------------------------------------------------------
# 父进程：调度、崩溃隔离、汇总
# --------------------------------------------------------------------------


def _worker_command(args, dtype, *, start=None, end=None, only_ids=None):
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--worker",
        "--results-file",
        str(args.results_file),
        "--dtype",
        dtype,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--implementation",
        args.implementation,
    ]
    if args.quick:
        cmd += ["--quick"]
    else:
        cmd += ["--cases-per-dtype", str(args.cases_per_dtype)]
    if args.device == "npu":
        cmd += ["--device-id", str(args.device_id)]
    if only_ids is not None:
        cmd += ["--only-ids", ",".join(str(i) for i in only_ids)]
    else:
        cmd += ["--start", str(start), "--end", str(end)]
    if args.verbose:
        cmd += ["--verbose"]
    return cmd


def _read_records(path):
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break  # 崩溃时最后一行可能被截断
        records[record["index"]] = record
    return records


def _tail(path, limit=1200):
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace").strip()


# KeyboardInterrupt 哨兵：通过 missing 通道传回，调用方据此保留在途记录后停止。
_INTERRUPTED = "__interrupted__"


def _launch_batch(args, dtype, *, start=None, end=None, only_ids=None):
    """在子进程中运行一批用例。

    返回 (records, first_missing, teardown_crash, stderr_tail)；
    first_missing 为 [start,end) 内首个无记录的下标（全有记录则为 None）；
    用户中断时 first_missing 为 _INTERRUPTED（records 含已落盘的记录）。
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="hc_collapse_gen_"))
    results_path = tmpdir / "results.jsonl"
    results_path.touch()
    stdout_path = tmpdir / "stdout.log"
    stderr_path = tmpdir / "stderr.log"
    batch_args = argparse.Namespace(**vars(args))
    batch_args.results_file = str(results_path)
    cmd = _worker_command(batch_args, dtype, start=start, end=end, only_ids=only_ids)

    env = dict(os.environ)
    env.setdefault("PYTHONFAULTHANDLER", "1")
    timed_out = False
    interrupted = False
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=env)
        last_size = -1
        last_progress = time.monotonic()
        next_report = time.monotonic() + 30
        try:
            while proc.poll() is None:
                time.sleep(2)
                size = results_path.stat().st_size
                if size != last_size:
                    last_size = size
                    last_progress = time.monotonic()
                elif time.monotonic() - last_progress > args.idle_timeout:
                    proc.kill()
                    proc.wait()
                    timed_out = True
                    break
                if args.progress_every > 0 and time.monotonic() >= next_report:
                    done = len(_read_records(results_path))
                    total = len(only_ids) if only_ids is not None else (end - start)
                    print(f"  [{args.device}|{dtype}] 批次进行中 {done}/{total} 条", flush=True)
                    next_report = time.monotonic() + 30
        except KeyboardInterrupt:
            proc.kill()
            proc.wait()
            interrupted = True
    code = proc.returncode
    records = _read_records(results_path)

    wanted = sorted(only_ids) if only_ids is not None else list(range(start, end))
    first_missing = None
    for index in wanted:
        if index not in records:
            first_missing = index
            break
    if interrupted:
        return records, _INTERRUPTED, False, ""
    stderr_tail = _tail(stderr_path)
    keep_dir = first_missing is not None or timed_out
    if keep_dir:
        # 崩溃/超时批次保留诊断目录（含 worker stdout/stderr 与结果文件）。
        print(f"  [{dtype}] 批次异常，诊断目录保留于 {tmpdir}", flush=True)
    else:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if code == 0 and first_missing is None:
        return records, None, False, ""
    reason = f"worker 退出码={code}"
    if timed_out:
        reason = f"worker 无进度超时（>{args.idle_timeout:g}s）被终止"
    if first_missing is not None:
        return records, first_missing, False, f"{reason}\n{stderr_tail}" if stderr_tail else reason
    return records, None, True, stderr_tail


def _run_dtype_isolated(args, dtype, cases, emit, should_stop=lambda: False):
    by_index = {case.index: case for case in cases}
    pending = sorted(by_index)
    empty_batches = 0
    while pending:
        # 取出 pending 开头的连续段：整段合并为一个批次，避免越段执行未选用例。
        run_len = 1
        while run_len < len(pending) and pending[run_len] == pending[run_len - 1] + 1:
            run_len += 1
        segment = pending[:run_len]
        if run_len > 1:
            records, missing, _, _ = _launch_batch(
                args, dtype, start=segment[0], end=segment[-1] + 1
            )
        else:
            records, missing, _, _ = _launch_batch(args, dtype, only_ids=[segment[0]])
        for index in sorted(records):
            if index in pending:  # 只记录被请求的用例，防止重复/越选
                emit(records[index])
                pending.remove(index)
        if missing == _INTERRUPTED:
            return
        if should_stop():
            return  # fail-fast：批次内已出现非 pass 结果
        if records:
            empty_batches = 0
        else:
            empty_batches += 1
            if empty_batches >= 20:
                print(
                    f"  [{dtype}] 连续 {empty_batches} 个批次零进度，疑似环境故障，中止该 dtype",
                    flush=True,
                )
                return
        if missing is None:
            continue  # 本批次用例均已记录；剩余 pending 由循环继续处理
        # 批次在 missing 处崩溃：单条重试定位。
        probe_records, probe_missing, _, probe_tail = _launch_batch(
            args, dtype, only_ids=[missing]
        )
        if probe_missing == _INTERRUPTED:
            for index in sorted(probe_records):
                if index in pending:
                    emit(probe_records[index])
                    pending.remove(index)
            return
        if missing in probe_records:
            record = probe_records[missing]
            record["note"] = "批次子进程崩溃后单条重试通过/复现（见 status）"
            emit(record)
        else:
            case = by_index[missing]
            emit(
                {
                    "dtype": dtype,
                    "case_id": case.case_id(dtype),
                    "index": case.index,
                    "num_tokens": case.num_tokens,
                    "hc_mult": case.hc_mult,
                    "hidden_size": case.hidden_size,
                    "layout": case.layout,
                    "origin": case.origin,
                    "status": "crash",
                    "message": f"设备进程级崩溃/挂起，单条重试仍失败\n{probe_tail}"[:2000],
                    "mismatches": -1,
                    "max_abs_err": None,
                    "max_rel_err": None,
                    "worst": [],
                    "elapsed_s": None,
                }
            )
        if missing in pending:
            pending.remove(missing)
        if should_stop():
            return


def _run_in_process(args, dtype, cases, emit, should_stop=lambda: False):
    for case in cases:
        emit(_run_one_case(case, dtype, args))
        if should_stop():
            return


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="hc_collapse 泛化用例驱动（详见脚本头部说明）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu",
                        help="标杆位置：npu=标杆(torch参考实现)在 NPU（默认，同设备对比）；"
                             "cpu=数据生成与标杆在 CPU，被测算子仍在 NPU 上执行")
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES),
                        help="参与的 dtype（默认全部三种）")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：每 dtype 60 条代表性用例")
    parser.add_argument("--cases-per-dtype", type=int, default=None, metavar="N",
                        help="每 dtype 用例数（默认 700；不可与 --quick 同用）")
    parser.add_argument("--seed", type=int, default=1234, help="随机用例与数据的种子（默认 1234）")
    parser.add_argument("--implementation", choices=("auto", "torch", "triton"), default="auto",
                        help="被测实现：auto=仓库配置分发（默认），torch/triton=强制指定（triton 仅 NPU）")
    parser.add_argument("--case-id", action="append", default=None, metavar="DTYPE:INDEX",
                        help="只重放指定用例，如 bfloat16:0007，可重复传入")
    parser.add_argument("--report", default=None, metavar="PATH", help="JSON 报告输出路径")
    parser.add_argument("--list", action="store_true", help="仅打印用例矩阵后退出")
    parser.add_argument("--in-process", action="store_true",
                        help="不启用子进程崩溃隔离（快，但设备级崩溃会中断整轮）")
    parser.add_argument("--fail-fast", action="store_true", help="出现首条非 pass 结果后停止后续用例")
    parser.add_argument("--progress-every", type=int, default=50, help="进度打印间隔（默认 50 条）")
    parser.add_argument("--verbose", action="store_true", help="打印每条用例")
    parser.add_argument("--idle-timeout", type=float, default=300.0,
                        help="子进程无进度超时秒数（默认 300）")
    parser.add_argument("--device-id", type=int, default=0, help="NPU 设备序号（默认 0）")
    # worker 专用参数（父进程内部使用）
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--results-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dtype", choices=DTYPES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--end", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--only-ids", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.quick and args.cases_per_dtype is not None:
        parser.error("--quick 与 --cases-per-dtype 互斥")
    if args.cases_per_dtype is not None and args.cases_per_dtype <= 0:
        parser.error("--cases-per-dtype 必须为正整数")
    if not args.quick and args.cases_per_dtype is None:
        args.cases_per_dtype = FULL_CASES_PER_DTYPE
    if args.quick:
        args.cases_per_dtype = QUICK_CASES_PER_DTYPE

    if args.worker:
        if not args.results_file or not args.dtype:
            parser.error("worker 需要 --results-file 与 --dtype")
        args.dtypes = [args.dtype]
        if args.only_ids is not None:
            args.only_ids = [int(i) for i in args.only_ids.split(",") if i != ""]
        return _worker_main(args)

    # ---- 父进程 ----
    if args.idle_timeout <= 0:
        parser.error("--idle-timeout 必须为正数")
    selected = {}
    for spec in args.case_id or []:
        try:
            dtype, _, index = spec.partition(":")
            index = int(index)
        except ValueError:
            parser.error(f"非法 --case-id：{spec!r}（应为 DTYPE:INDEX，如 bfloat16:0007）")
        if dtype not in DTYPES:
            parser.error(f"--case-id dtype 非法：{spec!r}（可选 {', '.join(DTYPES)}）")
        if dtype not in args.dtypes:
            parser.error(
                f"--case-id {spec!r} 的 dtype 未包含在 --dtypes {args.dtypes} 中"
            )
        selected.setdefault(dtype, set()).add(index)

    matrices = {
        dtype: build_cases(
            dtype, quick=args.quick, cases_per_dtype=args.cases_per_dtype, seed=args.seed
        )
        for dtype in args.dtypes
    }

    if args.list:
        for dtype in args.dtypes:
            cases = matrices[dtype]
            print(f"\n[{dtype}] {len(cases)} 条用例（seed={args.seed}, "
                  f"{'quick' if args.quick else 'full'}）")
            for case in cases:
                print(
                    f"  {case.case_id(dtype)}  T={case.num_tokens:<6d} m={case.hc_mult:<2d} "
                    f"H={case.hidden_size:<5d} layout={case.layout:<24s} {case.origin}"
                )
        return 0

    if args.device == "npu":
        import torch

        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            print("错误：--device npu 但当前环境 NPU 不可用；可切换 --device cpu", file=sys.stderr)
            return 2
    else:
        import torch

        # cpu 模式被测算子仍在 NPU 上执行，同样要求 NPU 可用。
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            print("错误：--device cpu 仅将数据生成与标杆放到 CPU，被测算子仍在 NPU 上执行，"
                  "需要 NPU 可用", file=sys.stderr)
            return 2

    reference_desc = (
        "torch 参考实现 @NPU" if args.device == "npu"
        else "torch 参考实现 @CPU（数据生成与标杆在 CPU，被测仍在 NPU）"
    )
    if selected:
        planned = sum(len(ids) for dtype, ids in selected.items() if dtype in matrices)
        total_all = planned
        scope = f"重放 {planned} 条指定用例"
    else:
        total_all = sum(len(cases) for cases in matrices.values())
        scope = f"每dtype={args.cases_per_dtype} 条 共 {total_all} 条"
    print(
        f"hc_collapse 泛化用例 | device={args.device} implementation={args.implementation} "
        f"标杆={reference_desc} | dtypes={','.join(args.dtypes)} "
        f"{scope} | seed={args.seed} "
        f"隔离={'关' if args.in_process else '开'}"
    )

    report = {
        "op": "hc_collapse",
        "meta": {
            "device": args.device,
            "implementation": args.implementation,
            "reference": reference_desc,
            "quick": bool(args.quick),
            "cases_per_dtype": args.cases_per_dtype,
            "seed": args.seed,
            "dtypes": list(args.dtypes),
            "isolation": not args.in_process,
            "started": datetime.now().isoformat(timespec="seconds"),
        },
        "dtypes": {},
    }

    overall_bad = 0
    started_wall = time.monotonic()
    stop_all = False
    for dtype, cases in matrices.items():
        if stop_all:
            break
        if selected and dtype not in selected:
            continue  # --case-id 模式下跳过没有选中用例的 dtype
        run_cases = cases
        if selected:
            wanted = selected.get(dtype, set())
            unknown = wanted - {case.index for case in cases}
            if unknown:
                print(f"错误：{dtype} 用例编号不存在：{sorted(unknown)}（有效范围 0~{len(cases) - 1}）",
                      file=sys.stderr)
                return 2
            run_cases = [case for case in cases if case.index in wanted]
        counters = {"pass": 0, "fail": 0, "error": 0, "crash": 0}
        records = []
        done = [0]
        stop_flag = [False]
        dtype_start = time.monotonic()

        def emit(record, dtype=dtype, run_cases=run_cases, counters=counters,
                 records=records, done=done, dtype_start=dtype_start,
                 stop_flag=stop_flag):
            records.append(record)
            counters[record["status"]] = counters.get(record["status"], 0) + 1
            done[0] += 1
            if record["status"] != "pass":
                stop_flag[0] = True
            if record["status"] != "pass" or args.verbose:
                detail = record.get("message", "")
                if detail:
                    # 控制台只显示首行摘要，完整信息见 JSON 报告。
                    detail = detail.splitlines()[0][:160] + (
                        f" …（完整信息见 --report，共 {len(record.get('message', ''))} 字符）"
                        if len(detail) > 160 or "\n" in record.get("message", "")
                        else ""
                    )
                if record.get("worst"):
                    detail += f" | worst={record['worst'][0]}"
                print(
                    f"  [{dtype}] {record['case_id']} status={record['status']} "
                    f"T={record['num_tokens']} m={record['hc_mult']} H={record['hidden_size']} "
                    f"layout={record['layout']} {detail}",
                    flush=True,
                )
            if args.progress_every > 0 and done[0] % args.progress_every == 0:
                elapsed = time.monotonic() - dtype_start
                print(
                    f"  [{args.device}|{dtype}] {done[0]}/{len(run_cases)} "
                    f"pass={counters['pass']} fail={counters['fail']} error={counters['error']} "
                    f"crash={counters['crash']} elapsed={elapsed:.0f}s",
                    flush=True,
                )

        try:
            # fail-fast：任何非 pass 记录后立即停止；正常模式下持续运行。
            should_stop = (lambda: stop_flag[0]) if args.fail_fast else (lambda: False)
            if args.in_process:
                _run_in_process(args, dtype, run_cases, emit, should_stop)
            else:
                _run_dtype_isolated(args, dtype, run_cases, emit, should_stop)
        except KeyboardInterrupt:
            print("\n中断：已保留当前结果", file=sys.stderr)
            stop_all = True

        bad = counters["fail"] + counters["error"] + counters["crash"]
        overall_bad += bad
        if args.fail_fast and bad:
            stop_all = True
        stats = {
            "total": len(records),
            "passed": counters["pass"],
            "guarded": sum(1 for r in records if r.get("guard")),
            "failed": counters["fail"],
            "errors": counters["error"],
            "crashed": counters["crash"],
        }
        for key in ("max_abs_err", "max_rel_err"):
            valid = [r for r in records if r.get(key) is not None]
            if valid:
                top = max(valid, key=lambda r: r[key])
                stats[key] = {"case": top["case_id"], "value": top[key]}
        report["dtypes"][dtype] = {**stats, "cases": records}
        print(
            f"[{dtype}] 完成：{stats['passed']}/{stats['total']} 通过（其中 {stats['guarded']} 条命中 UB 预算校验），"
            f"fail={stats['failed']} error={stats['errors']} crash={stats['crashed']} "
            f"({time.monotonic() - dtype_start:.0f}s)"
        )

    report["meta"]["finished"] = datetime.now().isoformat(timespec="seconds")
    report["meta"]["wall_s"] = round(time.monotonic() - started_wall, 1)
    report["meta"]["overall_passed"] = overall_bad == 0
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"JSON 报告已写入 {report_path}")

    verdict = "全部通过" if overall_bad == 0 else f"{overall_bad} 条用例未通过"
    print(f"总结：{verdict}（总耗时 {report['meta']['wall_s']}s）")
    return 0 if overall_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
