"""Run the mojo_opset ``hc_collapse`` cases on CUDA without touching repo code.

Why this is needed
------------------
The packaged Triton provider for ``hc_collapse`` is NPU-only
(``npu_a2_triton``/``npu_a5_triton``): its wrapper queries NPU vector-core
counts, and its persistent schedule is NPU-tuned. The CUDA device in this
repo is exposed through the ``ilu.generic`` target, and that provider package
(``ilu_triton``) has no ``hc_collapse`` binding. So on CUDA:

* ``implementation="torch"`` already works with zero changes.
* ``implementation="triton"`` needs a CUDA kernel, a provider binding and a
  config entry -- this script patches all three at runtime.

It also patches two CUDA-only harness quirks:

* With the kineto profiler, a ``record_function`` that wraps device work is
  exported twice in the chrome trace (CPU ``user_annotation`` plus a
  same-named GPU ``gpu_user_annotation``, and the tail samples can
  additionally be duplicated by the final activity flush). The repo's
  ``kernel_spans`` treats that as ``Duplicate profiler range:
  mojo_perf_sample_N``.
* The ilu adapter provides no ``kernel_owners`` launch-flow correlation, so
  sample kernels are selected by device-time containment and any stray
  kernel executing inside a sample window (aten fill/init noise) is swept
  into ``profiler_span_us``. The NPU adapter correlates kernels through
  torch_to_npu flows and is immune.

The timing patch fixes both and, for traces that contain the hc_collapse
kernel, pins the profiler span to that kernel only -- on every platform.
Cross-platform numbers then compare the same operator kernel; ``e2e``/``event``
timers stay unfiltered (whole-call host time).

Cross-platform operator consistency
-----------------------------------
* The CUDA kernel is the vLLM original: identical FP32 accumulation order
  over ``hc_mult`` streams, identical tail masking (BLOCK_H=1024) and
  identical BF16 store rounding as the NPU kernels. Scheduling (grid shape,
  warps) is platform-appropriate by design, like every per-target kernel in
  this repo.
* The shared accuracy suite (including the bitwise pinned rows) validates the
  same math on both platforms; run it on CUDA with this plugin (see below).
* For identical measurement code paths on NPU, load this plugin there too:
  ``python -m pytest tests/performance/functions/test_hc_collapse.py \
    -p scripts.bench_hc_collapse_cuda``

What the script patches (process-local, never writes files):
1. Registers the original vLLM CUDA kernel (``_hc_head_reduce_store_kernel``
   from ``vllm/model_executor/kernels/mhc/triton.py``, 2D grid, num_warps=4,
   ``enable_fp_fusion=False`` on backends that support it) as the dynamic
   module ``mojo_opset.kernels.ilu_triton.hc_collapse`` exporting
   ``hc_collapse_fwd``.
2. Adds ``"hc_collapse": "hc_collapse"`` to ``ilu_triton.OPS``.
3. Overlays ``ilu.generic: {hc_collapse: triton}` onto the dispatch config.
4. Wraps ``tests.performance.timing.kernel_spans`` to drop duplicate
   ``mojo_perf_sample_*`` ranges and to pin profiler spans to the
   hc_collapse kernel itself (``_hc_head_reduce_store_kernel`` on CUDA,
   ``_hc_collapse_kernel`` on NPU), so stray aten kernels never inflate
   cross-platform comparisons.

Usage
-----
From the repository root on a CUDA machine:

    # Everything (patches + pytest), defaults: --mojo-target ilu.generic
    #                               --mojo-implementation triton
    python scripts/bench_hc_collapse_cuda.py
    python scripts/bench_hc_collapse_cuda.py --perf-repeats 10 -k v41

    # Or load it as a pytest plugin and keep full CLI control:
    pytest tests/performance/functions/test_hc_collapse.py \
        -p scripts.bench_hc_collapse_cuda \
        --mojo-target ilu.generic --mojo-implementation triton

    # The same plugin also enables the accuracy cases on CUDA:
    pytest tests/functions/test_hc_collapse.py \
        -p scripts.bench_hc_collapse_cuda \
        --mojo-target ilu.generic --mojo-implementation triton

Note: on NVIDIA GPUs the automatic platform probe reports no accelerator
(``get_platform`` only auto-detects Iluvatar/NPU/MLU), so pass
``--mojo-target ilu.generic`` explicitly; the ``ilu`` adapter is plain
``torch.cuda`` and works on any CUDA device.
"""

import sys

from pathlib import Path

import torch

PROVIDER_PACKAGE = "mojo_opset.kernels.ilu_triton"
PATCHED_MODULE = f"{PROVIDER_PACKAGE}.hc_collapse"

# ---------------------------------------------------------------------------
# CUDA Triton kernel: the vLLM original (Apache-2.0), self-contained.
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on non-triton hosts
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _hc_head_reduce_store_kernel(
        pre_ptr,
        x_ptr,
        out_ptr,
        hidden_size: tl.constexpr,
        hc_mult: tl.constexpr,
        pre_stride_t: tl.constexpr,
        pre_stride_m: tl.constexpr,
        x_stride_t: tl.constexpr,
        x_stride_m: tl.constexpr,
        x_stride_h: tl.constexpr,
        out_stride_t: tl.constexpr,
        out_stride_h: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1)
        offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offsets < hidden_size

        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for mix_idx in tl.static_range(0, hc_mult):
            pre = tl.load(pre_ptr + token_idx * pre_stride_t + mix_idx * pre_stride_m).to(
                tl.float32
            )
            x = tl.load(
                x_ptr
                + token_idx * x_stride_t
                + mix_idx * x_stride_m
                + offsets * x_stride_h,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += pre * x

        tl.store(
            out_ptr + token_idx * out_stride_t + offsets * out_stride_h,
            acc,
            mask=mask,
        )


def _fp_fusion_supported() -> bool:
    """enable_fp_fusion is an NVIDIA/AMD backend option, not a portable one.

    Cached: probing the active target on every call would add host overhead
    to each timed invocation.
    """

    from functools import lru_cache

    @lru_cache(maxsize=1)
    def _probe() -> bool:
        try:
            backend = triton.runtime.driver.active.get_current_target().backend
        except Exception:
            return False
        return backend in ("cuda", "nvidia", "amd", "rocm")

    return _probe()


def hc_collapse_cuda_impl(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse BF16 residual streams with FP32 pre-mix coefficients (CUDA)."""
    num_tokens, hc_mult, hidden_size = x.shape
    out = torch.empty(num_tokens, hidden_size, dtype=x.dtype, device=x.device)
    if num_tokens == 0:
        return out

    block_h = 1024
    launch = {"num_warps": 4}
    if _fp_fusion_supported():
        # Preserve the separate FP32 multiply and sum in the Torch reference.
        launch["enable_fp_fusion"] = False
    _hc_head_reduce_store_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        pre_mix,
        x,
        out,
        hidden_size,
        hc_mult,
        pre_mix.stride(0),
        pre_mix.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        **launch,
    )
    return out


# ---------------------------------------------------------------------------
# Runtime patching
# ---------------------------------------------------------------------------

def install(*, require_cuda: bool = True) -> bool:
    """Patch the ilu_triton provider and config for hc_collapse on CUDA.

    Idempotent; returns True when the patches are active after the call.
    ``require_cuda=False`` registers the binding without a device check, for
    dry-run validation on non-CUDA hosts (kernel launch still needs CUDA).
    """
    if triton is None:
        raise RuntimeError("triton is required for the hc_collapse CUDA patch")
    if require_cuda and not torch.cuda.is_available():
        return False
    if PATCHED_MODULE in sys.modules:  # already installed in this process
        return True

    import types

    provider = __import__(PROVIDER_PACKAGE, fromlist=["OPS"])

    module = types.ModuleType(PATCHED_MODULE)
    module.hc_collapse_fwd = hc_collapse_cuda_impl
    module.hc_collapse_impl = hc_collapse_cuda_impl
    module.hc_collapse_kernel = _hc_head_reduce_store_kernel
    sys.modules[PATCHED_MODULE] = module
    setattr(provider, "hc_collapse", module)
    provider.OPS["hc_collapse"] = "hc_collapse"

    # Overlay the config entry so runs without --mojo-implementation resolve too.
    from mojo_opset import config

    snapshot = config.get_config()
    snapshot.setdefault("ilu.generic", {})["hc_collapse"] = "triton"
    config.configure(snapshot)

    print(
        f"[bench_hc_collapse_cuda] patched: {PATCHED_MODULE} + ilu_triton.OPS "
        "+ config ilu.generic hc_collapse=triton",
        file=sys.stderr,
    )
    return True


# Kernel-name selectors for every hc_collapse implementation (CUDA script
# kernel plus the packaged NPU kernels), matched with "contains" semantics.
OP_KERNEL_SELECTORS = ("_hc_head_reduce_store_kernel", "_hc_collapse_kernel")


def _dedupe_sample_ranges(trace):
    """Drop repeated mojo_perf_sample_* complete events from a chrome trace.

    kineto on CUDA emits the same record_function name twice (CPU
    ``user_annotation`` + GPU ``gpu_user_annotation``), and the final CUPTI
    flush can duplicate tail samples. Keep the earliest-start occurrence: the
    CPU annotation opens before any device work, so its window is the one
    that contains the sample's kernels regardless of trace ordering.
    """
    from tests.performance.timing import SAMPLE_PREFIX

    events = trace.get("traceEvents") if isinstance(trace, dict) else trace
    if not events:
        return trace
    best = {}
    for index, event in enumerate(events):
        name = event.get("name", "")
        if event.get("ph") != "X" or not isinstance(name, str) or not name.startswith(SAMPLE_PREFIX):
            continue
        start = float(event.get("ts", 0))
        if name not in best or start < best[name][0]:
            best[name] = (start, index)
    duplicate_indexes = set()
    counted = set()
    for index, event in enumerate(events):
        name = event.get("name", "")
        if event.get("ph") != "X" or not isinstance(name, str) or not name.startswith(SAMPLE_PREFIX):
            continue
        if index != best[name][1]:
            duplicate_indexes.add(index)
    if not duplicate_indexes:
        return trace
    filtered = [event for index, event in enumerate(events) if index not in duplicate_indexes]
    return {**trace, "traceEvents": filtered} if isinstance(trace, dict) else filtered


def _pin_span_to_op_kernel(trace, is_kernel):
    """Remove non-hc_collapse kernel events, but only from this op's traces.

    Without launch-flow correlation (ilu/CUDA adapter), kernel_spans selects
    sample kernels by device-time containment, so stray aten kernels inside a
    sample window would inflate the span. Traces without an hc_collapse
    kernel are returned untouched, which keeps other operators' runs intact.
    """
    events = trace.get("traceEvents") if isinstance(trace, dict) else trace
    if not events:
        return trace

    def is_op_kernel(event):
        name = event.get("name", "")
        return isinstance(name, str) and any(selector in name for selector in OP_KERNEL_SELECTORS)

    if not any(is_kernel(event) and is_op_kernel(event) for event in events):
        return trace

    import os

    debug = bool(os.getenv("MOJO_HC_CUDA_DEBUG_KERNELS"))
    dropped = {}
    filtered = []
    for event in events:
        if is_kernel(event) and not is_op_kernel(event):
            name = event.get("name", "")
            dropped[name] = dropped.get(name, 0) + 1
            continue
        filtered.append(event)
    count = sum(dropped.values())
    print(
        "[bench_hc_collapse_cuda] profiler span pinned to the hc_collapse kernel "
        + (f"({count} non-op kernel events dropped" if count else "(no non-op kernels found")
        + ("; set MOJO_HC_CUDA_DEBUG_KERNELS=1 to list them)" if count and not debug else ")"),
        file=sys.stderr,
    )
    if debug:
        for name, occ in sorted(dropped.items()):
            print(f"[bench_hc_collapse_cuda]   dropped {occ}x {name[:120]}", file=sys.stderr)
    return {**trace, "traceEvents": filtered} if isinstance(trace, dict) else filtered


def patch_timing_dedup() -> bool:
    """Wrap kernel_spans so duplicated profiler sample ranges are ignored.

    Idempotent; returns True when the wrapper is active (or already was).
    Safe on every platform: genuinely distinct sample ranges are never merged.
    """
    from tests.performance import timing

    if getattr(timing.kernel_spans, "_hc_cuda_dedup", False):
        return True
    original = timing.kernel_spans

    def kernel_spans(trace, *args, **kwargs):
        trace = _dedupe_sample_ranges(trace)
        if args and callable(args[0]):  # is_kernel is the first positional arg
            trace = _pin_span_to_op_kernel(trace, args[0])
        return original(trace, *args, **kwargs)

    kernel_spans._hc_cuda_dedup = True
    timing.kernel_spans = kernel_spans
    print(
        "[bench_hc_collapse_cuda] patched: kernel_spans duplicate-sample dedup "
        "+ op-kernel span pinning",
        file=sys.stderr,
    )
    return True


def pytest_configure(config):
    """Hook entry when loaded as a plugin via -p (after pythonpath applies)."""
    import os

    if torch.cuda.is_available() or os.getenv("MOJO_HC_CUDA_PATCH_FORCE"):
        install(require_cuda=False)
    try:
        patch_timing_dedup()
    except Exception as error:  # non-perf runs never touch timing
        print(f"[bench_hc_collapse_cuda] timing patch skipped: {error}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Standalone entry: patch in-process, then run the repo's perf cases.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest

    REPO_ROOT = Path(__file__).resolve().parent.parent
    PERF_CASES = "tests/performance/functions/test_hc_collapse.py"

    argv = list(sys.argv[1:])
    injected = []
    if "--mojo-target" not in argv:
        injected += ["--mojo-target", "ilu.generic"]
    if "--mojo-implementation" not in argv:
        injected += ["--mojo-implementation", "triton"]
    if not any(arg.endswith(".py") or "::" in arg for arg in argv):
        argv = [str(REPO_ROOT / PERF_CASES)] + argv

    install(require_cuda=False)
    patch_timing_dedup()
    raise SystemExit(pytest.main(injected + argv))
