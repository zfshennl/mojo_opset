# moe_gating_top_k TLA 实验报告

> 参考：`triton-ascend/third_party/ascend/unittest/tla_ut/vector/topk_gate.py`
>
> 实验实现：`mojo_opset/kernels/npu_a5_triton/moe_gating_top_k_tla.py`
>
> 目标：把当前 Triton 版 `moe_gating_top_k` 改写成 TLA（Tile Language
> Ascend）算子，验证 TLA 的 UB/SIMD 显式编程能否缩小与 AscendC 原生
> `torch_npu.npu_moe_gating_top_k` 的差距。

## 1. 结论先行

TLA 版本已经完成 **真正的 batched_row 改造**，并在真实 A5 上跑通完整语义、
精度和 msprof 性能验证。

- **Decode 档已经达到或超过 native**：
  - Qwen3.8 B1：TLA `3.47 µs`，native `3.81 µs`，TLA 更快；
  - Qwen3.8 B64：TLA `5.73 µs`，native `5.51 µs`，`1.06×`；
  - Qwen3.8 B256：TLA `13.75 µs`，native `8.96 µs`，`1.54×`；
  - Qwen3.5 B1：TLA `2.69 µs`，native `3.49 µs`，TLA 更快；
  - Qwen3.5 B128：TLA `5.94 µs`，native `5.60 µs`，`1.06×`；
  - Qwen3-30B B64：TLA `4.90 µs`，native `4.82 µs`，`1.02×`；
  - DeepSeek grouped B32：TLA `3.88 µs`，native `4.08 µs`，TLA 更快。
- **Prefill 档差距缩小到 1.17-1.69×**：
  - Qwen3.8 2048：TLA `58.7 µs`，native `36.4 µs`，`1.61×`；
  - Qwen3.8 8192：TLA `214.8 µs`，native `127.0 µs`，`1.69×`；
  - Qwen3.5 8192：TLA `103.2 µs`，native `86.9 µs`，`1.19×`；
  - Qwen3-30B 2048：TLA `27.0 µs`，native `17.6 µs`，`1.53×`；
  - DeepSeek grouped 2048：TLA `45.3 µs`，native `36.7 µs`，`1.23×`；
  - norm-out 2048：TLA `30.7 µs`，native `26.3 µs`，`1.17×`。
- 相对当前 Triton 版，batched TLA 全档提速 **2.3-6.2×**：
  - Qwen3.8 2048：`230.8 → 58.7 µs`（`3.93×`）；
  - Qwen3.8 8192：`899.4 → 214.8 µs`（`4.19×`）；
  - Qwen3.5 8192：`643.4 → 103.2 µs`（`6.23×`）；
  - Qwen3-30B 2048：`149.3 → 27.0 µs`（`5.54×`）；
  - DeepSeek grouped 2048：`265.5 → 45.3 µs`（`5.86×`）。

Batched_row 改造本身带来了明显收益，尤其是 prefill：

| Case | 单行 TLA µs | batched TLA µs | 提升 |
| --- | ---: | ---: | ---: |
| Qwen3.8 2048 | 89.5 | 58.7 | **1.52×** |
| Qwen3.8 8192 | 347.2 | 214.8 | **1.62×** |
| Qwen3.5 8192 | 192.0 | 103.2 | **1.86×** |
| Qwen3-30B 2048 | 46.9 | 27.0 | **1.74×** |
| DeepSeek 2048 | 62.2 | 45.3 | **1.37×** |
| norm-out 2048 | 58.1 | 30.7 | **1.89×** |

现在的剩余差距已经非常清楚：

- Qwen3.8 prefill 的 Vector pipe 达到 **0.90-0.92**，MTE2/MTE3 只有
  **0.03-0.04**；搬运已经不是瓶颈；
- TLA 仍然是 repeated-max Top-K，K=10 就要扫 10 次整行；
- native 是 `Sort32 + MrgSort` 一次排序后取前 K。
- 所以下一步的核心是 **硬件 merge sort 的 value+index 输出**，
  而不是继续做 batch row 或搬运优化。

## 2. TLA 算子设计

TLA 实现遵循 `topk_gate.py` 的“显式 UB + SIMD”风格：

```python
@tla.jit
def _moe_gating_top_k_tla_kernel(X, BIAS, Y, IDX, NORM_OUT, n_rows,
                                 STRIDE_X: tl.constexpr, K: tl.constexpr,
                                 M: tl.constexpr, ...):
    x_ub = tla.alloc_buffer((M, E), tl.float32, space="ub", versions=2)
    norm_ub = tla.alloc_buffer((M, E), tl.float32, space="ub", versions=2)
    y_ub = tla.alloc_buffer((M, Y_PAD), Y_DTYPE, space="ub")
    idx_ub = tla.alloc_buffer((M, 64), tl.int32, space="ub")
    for base in range(tl.program_id(0) * M, n_rows, CORES * M):
        if base + M <= n_rows:
            xv = tla.view(X, (M, E), strides=(STRIDE_X, 1), offsets=(base, 0))
            tla.copy(xv, x_ub)          # 一次 batch load M 行
        else:
            ...                          # 尾部逐行 load
        rows_this = tl.minimum(M, n_rows - base)
        for br in range(0, rows_this):  # 动态行循环，避免 M 份 body 展开
            with tla.scope.simd():
                ...
        if base + M <= n_rows:
            ysrc = tla.subview(y_ub, (0, 0), (M, K))
            tla.copy(ysrc, y_view(base)) # 一次 batch store
```

### 2.1 BLOCK_M 行 batch ld/st（本轮已落地）

当前 TLA 是 **真正的 batched_row**：

- `x_ub` 是 `(M, E)`，`norm_ub` 是 `(M, E)`，`y_ub` 是 `(M, Y_PAD)`，
  `idx_ub` 是 `(M, 64)`；
- main path 用一次 2D `GM -> UB` copy 搬入 `M` 行；
- 用一次 2D `UB -> GM` copy 写出 `M` 行的 y / expert_idx / norm；
- batch 内部再按行做 SIMD 处理；
- 只有最后一个不满 `M` 行的 tail 才退回逐行 copy。

`M` 的默认策略：

| rows | 默认 `BLOCK_M` | 说明 |
| ---: | ---: | --- |
| 1 | 1 | 单行，grid=1 |
| 32/64 | 1 | `cdiv(rows, 2) < 56`，保持多核并行 |
| 128 | 2 | `cdiv(128,2)=64 >= 56` |
| 256 | 4 | `cdiv(256,4)=64 >= 56` |
| 2048/8192 | 8 | 56 核都有足够 batch |

同时用 `8192 // E` 做 UB 上限保护，避免大 E 时 batch 过大。

实现细节：

- `STRIDE_X` 必须作为 `tl.constexpr` 传入；TLA 的 2D `tla.view` 不接受
  runtime stride；
- 行处理循环必须用运行时 `for br in range(0, rows_this)`，
  不能用 `tl.static_range(M)`：后者会把完整 body 展开 M 份，
  group 路径会触发 BiSheng VF stack overflow（实测 9-18 KB > 8 KB）；
- batch store 用 `tla.subview(y_ub, (0,0), (M,K))` 生成带 UB row pitch
  的 `(M,K)` 视图，再 copy 到 GM；这样不需要把输出 padding 到 `K_PAD`。

### 2.2 归一化

所有归一化都在 64-lane FP32 寄存器上完成：

| `norm_type` | TLA 实现 |
| --- | --- |
| 0 softmax | `vreduce_max -> vbroadcast -> vexp -> vreduce_sum -> vbroadcast -> vdiv` |
| 1 sigmoid | `vdiv(1, 1 + vexp(-x))` |
| 2 sqrt-softplus | `vsqrt(vln(1 + vexp(-abs(x))) + vmax(x, 0))` |
| 3 sigmoid | 同 1 |

关键点：`vreduce_max/sum` 的结果只在 lane 0 有效，参与 `vsub/vdiv/vmul`
前必须 `vbroadcast`，否则其余 lane 是 0，会算出 inf/NaN。

### 2.3 分组选择

- 组内 score 仍用 64-lane 向量计算；
- `group_select_mode=0` 取组内 max；
- `group_select_mode=1` 取组内 top-2 之和，tie 时只 mask 第一个最大值位置，
  与 torch 参考一致；
- `k_group` 次 `vreduce_max + vreduce_min(index)` 选出 group；
- 选中 group 的 mask 不再写回 GM，而是在每个 64-lane expert chunk 上，
  用 `group_id` 区间比较和 `gflag` 广播生成，避免额外的 group mask buffer。

### 2.4 Top-K

与 `topk_gate.py` 相同，使用 repeated max，但 K 循环用运行时
`for step in range(0, K)`（不用 `tl.static_range`，避免 K>=24 时
VF stack overflow）：

```text
best = vreduce_max(selection)
winner = vreduce_min(where(selection == best, expert_id, E))
weight = sum(where(expert_id == winner, norm, 0))
selection = where(expert_id == winner, -inf, selection)
```

差别是：

- `selection`、`norm` 都是 64-lane 寄存器 tuple，不再做 tensor reduction；
- 权重直接从寄存器里的 `norm` 取，不需要重新标量 load；
- K 个结果先写进 `y_vec/idx_vec` 的 lane，再一次性写 UB/GM。

### 2.5 输出打包

BF16/FP16 的窄化有一个关键 layout 细节：

- `vcvt(f32, bf16)` 的 64 个 f32 lane 会落到 128-lane bf16 的
  **偶数 lane**（`0, 2, 4, ...`）；
- 直接 `vsts(..., dist="NORM_B16")` 会得到间隔为 2 的权重，y 全错；
- 需要再做一次 `vdintlv(y_narrow, y_narrow)` 把偶数 lane 压紧，
  然后写 `[K]` 输出。

FP32 输出不需要这一步，直接 `NORM_B32`。

## 3. 实现中踩到的 TLA 坑

| 问题 | 现象 | 解决 |
| --- | --- | --- |
| `vextf` binding 不一致 | 编译报 `unknown vcvt kind: vextf` | 改用 `tla.simd.vcvt(src, tl.float32)` |
| reduce 结果未广播 | softmax 分母只有 lane 0 有效，y 出现 NaN | `vbroadcast(vreduce_*)` 后再做 elementwise |
| UB 写后立即读 | group 路径 idx 全错 | `sel_ub` 写完后加 `tla.simd.membar("VST_VLD")` |
| BF16 窄化 lane 交错 | idx 对、y 误差 0.1-0.6 | `vcvt + vdintlv` 压紧偶数 lane |
| 2D view 的 runtime stride | 编译报 `buffer shape must contain positive compile-time integers` | `STRIDE_X` 改为 `tl.constexpr` |
| `tl.static_range(M)` 展开 body | group 路径 VF stack overflow（9-18 KB > 8 KB） | 行循环改为运行时 `for br in range(rows_this)` |
| `tl.static_range(K)` 展开 Top-K | K>=24 时 VF stack overflow（9.4 KB > 8 KB） | Top-K 主循环改为运行时 `for step in range(0,K)` |
| K 输出 copy | `tla.subview(y_ub, (0,0), (M,K))` 可用 | 不需要把 y/idx padding 到 32B |

其中 **`VST_VLD` membar** 是最隐蔽的：group 路径先写 `sel_ub`，随后在同一
SIMD scope 里读回计算 group score；没有 membar 时读到的还是旧值，单 group
路径不受影响，所以很容易误判为 group 算法错误。

## 4. 精度验证

验证方式：与 `mojo_opset.functions.moe_gating_top_k(..., implementation="torch")`
逐元素比较。BF16/FP16 的 idx 和 y 都是**完全相等**；norm_out 只有
FP32 舍入级误差。batched 版本对 `block_m=2/4/8` 分别跑了同一组完成度
对比，结果一致；下表是自适应 `block_m` 路径的结果。

| Case | idx | y max err | norm max err |
| --- | --- | ---: | ---: |
| Qwen3.8 257×512 K10, softmax, renorm, out_flag | 完全一致 | 0 | 7.5e-9 |
| Qwen3.8 64×512 K10, softmax, bias, no renorm | 完全一致 | 0 | — |
| Qwen3.5 129×256 K8, softmax, renorm, out_flag | 完全一致 | 0 | 7.5e-9 |
| Qwen3-30B 65×128 K8, softmax, renorm | 完全一致 | 0 | — |
| DeepSeek 33×256 K8, sigmoid, G8/Top4, bias, out_flag | 完全一致 | 0 | 0 |
| DeepSeek 64×256 K8, sigmoid, G8/Top4, bias, FP16 | 完全一致 | 0 | — |
| sqrt-softplus 19×384 K6, G8/Top3, bias, out_flag | 完全一致 | 0 | 2.4e-7 |
| sigmoid norm_type=3, renorm=0 | 完全一致 | 0 | — |
| sigmoid norm_type=3, renorm=1 | 完全一致 | 0 | — |
| K=1 / K=16 | 完全一致 | 0 | — |
| K=24 / K=32 / K=64 | 完全一致 | 0 | — |
| 非连续输入（row stride > E） | 完全一致 | 0 | — |
| FP32 out_flag | 完全一致 | 3.0e-8 | 3.7e-9 |
| 全相等 tie（无 group） | 完全一致 | 0 | — |
| 全相等 tie（grouped） | 完全一致 | 0 | — |

覆盖了：

- `norm_type` 0/1/2/3；
- `group_count` 1/8，`group_select_mode` 0/1，`k_group` 1/4；
- bias / no bias；
- renorm 0/1；
- `out_flag` True/False；
- BF16 / FP16 / FP32；
- K=1/8/10/16；
- 非连续输入和 tie。

## 5. 性能验证

数据来源：`msprof op --aic-metrics=PipeUtilization`，
`OpBasicInfo.csv` 的 `Task Duration(us)`；TLA 与 native 都取
`--launch-skip-before-match=4` 后的一次 kernel。
TLA 使用自适应 `BLOCK_M`（B1/B64→1，B128→2，B256→4，2048/8192→8）。

### 5.1 Decode

| Case | TLA µs | 当前 Triton µs | native µs | TLA/Triton | TLA/native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 B1 | 3.47 | 8.1 | 3.81 | **0.43×** | **0.91×** |
| Qwen3.8 B64 | 5.73 | 15.6 | 5.51 | **0.37×** | 1.04× |
| Qwen3.8 B256 | 13.75 | 34.5 | 8.96 | **0.40×** | 1.54× |
| Qwen3.5 B1 | 2.69 | 6.5 | 3.49 | **0.41×** | **0.77×** |
| Qwen3.5 B128 | 5.94 | 16.4 | 5.60 | **0.36×** | 1.06× |
| Qwen3-30B B64 | 4.90 | 11.0 | 4.82 | **0.45×** | 1.02× |
| DeepSeek B32 grouped | 3.88 | 9.8 | 4.08 | **0.40×** | **0.95×** |

Decode 已经进入 native 同级或更快：

- Qwen3.8 B1、Qwen3.5 B1、DeepSeek B32 都快于 native；
- Qwen3.8 B64 只有 1.04× 差距，Qwen3.5 B128 为 1.06×，
  Qwen3-30B B64 为 1.02×；
- 唯一明显落后的是 Qwen3.8 B256（1.54×），原因是当前 TLA 对 256 行
  仍按 `BLOCK_M=4`、56 核处理，而 native 会根据 batch 调整 block dim。

### 5.2 Prefill

| Case | TLA µs | 当前 Triton µs | native µs | TLA/Triton | TLA/native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 2048,E512,K10 | 58.70 | 230.8 | 36.44 | **0.25×** | 1.61× |
| Qwen3.8 8192,E512,K10 | 214.79 | 899.4 | 127.00 | **0.24×** | 1.69× |
| Qwen3.5 8192,E256,K8 | 103.22 | 643.4 | 86.88 | **0.16×** | 1.19× |
| Qwen3-30B 2048,E128,K8 | 26.95 | 149.3 | 17.60 | **0.18×** | 1.53× |
| DeepSeek 2048,E256,G8,K8 | 45.31 | 265.5 | 36.71 | **0.17×** | 1.23× |
| norm-out 2048,E256,K8,FP32 | 30.70 | 171.7 | 26.26 | **0.18×** | 1.17× |

TLA 相对当前 Triton 稳定快 **3.9-6.2×**，prefill 与 native 的差距
已经收敛到 **1.17-1.69×**。Batched row 把单行 TLA 又提升了
**1.37-1.89×**，这是本轮最大的性能增量。

### 5.3 流水占用

| Case | vec | scalar | mte2 | mte3 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8 B1 | 0.576 | 0.154 | 0.157 | 0.120 |
| Qwen3.8 B64 | 0.426 | 0.368 | 0.125 | 0.087 |
| Qwen3.8 B256 | 0.732 | 0.170 | 0.058 | 0.055 |
| Qwen3.8 2048 | 0.900 | 0.034 | 0.038 | 0.036 |
| Qwen3.8 8192 | 0.917 | 0.018 | 0.038 | 0.035 |
| Qwen3.5 B1 | 0.390 | 0.202 | 0.230 | 0.185 |
| Qwen3.5 B128 | 0.416 | 0.349 | 0.135 | 0.119 |
| Qwen3.5 8192 | 0.858 | 0.036 | 0.075 | 0.049 |
| Qwen3-30B B64 | 0.236 | 0.473 | 0.128 | 0.167 |
| Qwen3-30B 2048 | 0.796 | 0.091 | 0.078 | 0.052 |
| DeepSeek B32 | 0.431 | 0.240 | 0.191 | 0.174 |
| DeepSeek 2048 | 0.891 | 0.047 | 0.050 | 0.029 |
| norm-out 2048 | 0.792 | 0.071 | 0.076 | 0.085 |

解释：

- Qwen3.8 prefill 的 Vector pipe 达到 **0.90-0.92**，MTE2/MTE3 只有
  0.035-0.038，说明搬运已经彻底不是瓶颈；
- Qwen3.5 8192、Qwen3-30B 2048、DeepSeek 2048 的 Vector 也到
  0.80-0.89，剩余时间主要花在 repeated-max 的 Vector 计算上；
- decode 小档 scalar 比例较高（0.15-0.47），但 kernel 只有 3-14 µs，
  行数太少，launch/标量开销占比自然高；
- norm-out 档 MTE3 0.085 仍然很低，额外写 FP32 norm 表没有成为瓶颈。

## 6. 和当前 Triton 版差异

| 维度 | 当前 Triton | TLA 实验 |
| --- | --- | --- |
| 行内表示 | `[BLOCK_G, BLOCK_S]` 2D tensor | 64-lane vector register tuple |
| 归约 | `tl.max/tl.sum(axis=1)` | `vreduce_max/sum + vbroadcast` |
| Top-K | K 次 tensor max/min/mask | K 次 vector max/min/mask |
| bias | 每 program 一次 | 每 kernel 一次 |
| 数据搬运 | `tl.load/tl.store`，`BLOCK_M=2` batch | `tla.copy` 显式 GM/UB，自适应 `BLOCK_M=1/2/4/8` |
| batch load | `[TILE,2]` + `tl.split` | 2D `(M,E)` 一次 copy，UB 内按行取 |
| batch store | `[K,2]` / `[TILE,2]` where 累加 | `(M,K)` subview 一次 copy，保留 UB row pitch |
| UB 控制 | 编译器分配 | `tla.alloc_buffer` 显式声明 |
| 代码约束 | E<=2048 任意 | E%64==0，G<=64，K<=64 |
| Decode | 8.1-34.5 µs | 2.69-13.75 µs |
| Prefill | 149-899 µs | 27.0-214.8 µs |

TLA 的收益主要来自：

1. 消除 2D tile 的 layout 转换和 reduction 开销；
2. 整行常驻 64-lane 寄存器，Top-K 每次只做向量 compare/select；
3. **真正的 batched_row**：一次搬入/搬出最多 8 行，减少 DMA 指令和
   行启动开销；prefill 因此再提升 1.37-1.89×；
4. bias 和 index 在寄存器/UB 中复用，减少重复 load；
5. 行处理使用运行时循环而不是 static unroll，避免代码膨胀和 VF stack
   overflow。

## 7. 下一步

1. **最高优先级：TLA + 硬件 `sort_with_index`**
   - 用 `vbitsort + vmrgsort4` 一次排序，替代 K 次 repeated-max；
   - 排序后直接 `extract_slice` 取前 K 个 value/index；
   - 当前 prefill 的 Vector pipe 已经 0.81-0.92，只剩算法优化空间；
     换排序后预计能从 1.17-1.69× 差距继续压缩到接近 native。
2. **batch 继续扩展 / double buffer**
   - batched_row 已经落地（1/2/4/8 行自适应）；
   - 下一层可显式做多 buffer 轮转，把 `BLOCK_M` 与 UB 容量、
     `batchRows` 反推绑定，进一步降低 prefill 的 MTE 启动开销；
   - 当前 prefill 的 MTE2/MTE3 只有 0.03-0.08，收益会比 Top-K
     算法改造小。
3. **尾块 DMA**
   - 当前要求 `E%64==0`；补 tail DMA 后支持 513 等奇数 expert。
4. **decode B256 的 block dim**
   - Qwen3.8 B256 TLA `13.75 µs`，native `8.96 µs`；
   - 当前 TLA 对 B256 用 56 核、`BLOCK_M=4`，native 用 52 核；
   - 可继续调 grid/每核行数和 batch 策略，但这是 decode 单点问题，
     不影响 prefill 主结论。

## 8. 复现

```python
from mojo_opset.kernels.npu_a5_triton.moe_gating_top_k_tla import moe_gating_top_k_tla

y, expert_idx, norm_out = moe_gating_top_k_tla(
    x,
    k=10,
    bias=None,
    renorm=1,
    norm_type=0,
    out_flag=False,
    cores=56,
    # block_m=None 时自动选择 1/2/4/8；也可显式指定
    block_m=8,
)
```

该实现**没有**注册到 `mojo_opset.functions` 的 provider dispatch 中，
需要显式 import；这样不会影响现有 Triton/torch_npu 路径和 CI。
