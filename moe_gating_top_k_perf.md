# moe_gating_top_k 性能报告

> 算子：`mojo_opset.functions.moe_gating_top_k`（A5 Triton 实现）
>
> 注：本报告采集于 batch row 改造前。当前实现已引入 `BLOCK_M=2` 真 batch ld/st
> （x `[TILE,2]` load + `tl.split`，y/idx/norm batch store），kernel 变化
> +0.8~8.2%；当前算法/tiling 与 AscendC Top-K 能力缺口见
> `moe_gating_top_k_algorithm_tiling.md`。
> TLA batched_row 实验见 `moe_gating_top_k_tla_experiment.md`：
> decode 已达到/超过 native，prefill 差距缩小到 1.17-1.69×，
> 剩余瓶颈是 repeated-max Top-K。
>
> 模型场景：Qwen3.8-Flash-Next / Qwen4Exp（512 experts, Top-10）、
> Qwen3.5-MoE（256 experts, Top-8）、Qwen3-30B-A3B（128 experts, Top-8）、
> DeepSeek-V3 grouped routing（256 experts, Top-8, 8 groups / Top-4 groups）。
>
> 原始数据：`moe_gating_top_k_perf_l2flush.json`、
> `moe_gating_top_k_baselines.json`、
> `moe_gating_top_k_msprof_pipeline.json`；msprof 原始 `PipeUtilization` 归档
> 位于 `/tmp/opencode/msprof_moe/*`。

## 0. 现状速览

- 当前实现是**纯 Vector routing kernel**：一次加载行内全部 expert logits，
  在 kernel 内完成 softmax/sigmoid/sqrt-softplus、可选 grouped selection、
  Top-K 与 renorm；不使用 Cube，也没有跨核通信。
- **Qwen3.8-Flash-Next**：
  - decode B1：kernel span **8.1 µs**，e2e **61.6 µs**；
  - decode B256：kernel span **34.5 µs**，e2e **114.1 µs**；
  - prefill 2048 / 8192：kernel span **230.8 / 899.4 µs**，
    e2e **290.5 / 963.3 µs**。
- **瓶颈形态**：msprof `PipeUtilization` 显示
  - AIV Vector pipe 占用 **0.57-0.82**；
  - AIV Scalar pipe 占用 **0.33-0.54**；
  - AIV MTE2 / MTE3 仅 **0.02-0.08 / 0.02-0.05**。
  → 当前 kernel 是 **Vector + Scalar 受限**，不是 HBM 带宽受限。
- **语义 ALU 利用率很低**：MFU（ALU-eq 口径）最高仅 **3.71%**
  （Qwen3.8 prefill 8192）。原因是 Top-K 由 K 次 `max/min/reduce` 选择构成，
  大量时间花在 reduction、mask、索引和标量控制上；语义上的有效算术量很小。
  pipe 表中 Vector 占用高、MFU 低并不矛盾：前者包含实现开销，后者只统计
  公式语义的 ALU-eq。
- **对比基线**（同场次冷 L2、NPU event 中位）：
  - decode 小档：多数场景 Triton 比 torch 小算子快 **1.1-3.7×**
    （Qwen3.8 B1、Qwen3.5 B1/B128、Qwen3-30B B64、DeepSeek grouped decode）；
    Qwen3.8 B64/B256 例外，慢 1.28-1.36×；
  - prefill 大档：Triton 比 torch 小算子慢 **1.8-6.0×**；
  - 相对 `torch_npu.npu_moe_gating_top_k`：Triton 慢 **3.6-8.2×**，
    大档差距集中在 prefill（Qwen3.8 8192 为 **6.95×**）。
  - 因此当前定位仍是**可移植 fallback**，不是 CANN 原生 routing 的性能替代。

## 1. 环境与口径

| 项目 | 值 |
| --- | --- |
| 目标 / 设备 | `npu.a5` / Ascend 950PR（56 VectorCore，1.65 GHz） |
| 软件 | CANN 9.2.0、torch 2.13.0、torch_npu 2.13.0、triton 3.6.0 |
| kernel 计时 | `torch_npu.profiler` NPU kernel event，冷 L2，单调用 span 中位 |
| e2e 计时 | 冷 L2 synchronized wall，包含 Python/custom-op host 路径与同步 |
| 冷 L2 | 每次调用前对 192 MiB FP32 buffer 求和；flush 在计时窗口之外 |
| 基线计时 | 同场次 NPU event，冷 L2，21 次中位；Triton / torch / torch_npu 用同一输入 |
| 流水采集 | `msprof op --aic-metrics=PipeUtilization`，`--launch-skip-before-match=3` |
| MFU 峰值 | **11.8272 TALU-eq/s** = 56 cores × 1.65 GHz × 128 FP32 elem/cycle |
| MBU 峰值 | **1.439 TB/s**（本机实测 1 GiB copy：2 GiB r+w / 1.492 ms） |

> 注：本轮只在 A5（Ascend 950PR）上采集；A2 与 A5 共用同一套 Triton
> kernel 结构，但未在物理 A2 设备上实测，A2 数据不纳入本报告。

`alu-eq` / MFU 的指令吞吐权重来自
`~/.cannbot/repo/ops/ops-theoretical-performance-estimation/references/vector_instruction_types.json`
与 `hardware/ascend950pr.json`：

```text
FP32 Add/Sub/Mul/Max/Min/Select/Compare/Cast(lowp->fp32):
    issue_slots=2, throughput_cycles=1, 256B/4B=64 elem/instr
    -> 2*64 = 128 elem/cycle/core = 1 alu-eq/element

FP32 Exp/Log/Div/Sqrt:
    issue_slots=2, throughput_cycles=4
    -> 2*64/4 = 32 elem/cycle/core = 4 alu-eq/element

FP32 ReduceMax/ReduceSum:
    issue_slots=1, throughput_cycles=1
    -> 1*64 = 64 elem/cycle/core = 2 alu-eq/element
```

## 2. 计算流与 alu-eq 口径

本 kernel 每行执行：

```text
x load -> FP32 cast (lowp)
       -> softmax / sigmoid / sqrt-softplus
       -> optional bias add
       -> optional group score + top-k_group group mask
       -> K 次 (max reduce + compare/select + min reduce + mask update)
       -> renorm + routed_scaling_factor
       -> y / expert_idx (+ optional norm_out)
```

本报告只统计**语义计算**，不把 mask/broadcast/索引管理等实现开销计入
`alu-eq`。令 `E=num_experts`、`K=k`、`G=group_count`、`Kg=k_group`、
`R=rows`，则每行：

```text
alu_eq_row =
    cast_in(E)                                      # lowp -> FP32，FP32 输入为 0
  + norm(E)                                         # softmax=13E, sigmoid=10E, sqrtsoftplus=17E
  + bias(E)                                         # 仅 bias != None
  + group_select(E, G, Kg)                          # G=1 时为 0
  + 8*E*K                                           # 每次 max/compare/select/min/mask
  + 2*K                                             # renorm / scale

group_select =
    G=1: 0
    G>1, mode=0: 2E + 5*G*Kg + 2E
    G>1, mode=1: 6E + G + 5*G*Kg + 2E
```

其中 softmax 的 `13E` 为 `ReduceMax(2E)+Sub(1E)+Exp(4E)+ReduceSum(2E)+Div(4E)`；
sigmoid 的 `10E` 为 `Neg+Exp(4E)+Add+Div(4E)`；sqrt-softplus 的 `17E` 为
`Abs+Neg+Exp(4E)+Add+Log(4E)+Max+Add+Sqrt(4E)`。

```text
TALU-eq/s = alu_eq / span
MFU       = TALU-eq/s / 11.8272e12

GM_bytes  = R*E*bytes(x)
          + (bias ? E*bytes(x) : 0)
          + R*K*bytes(y) + R*K*4
          + (out_flag ? R*E*4 : 0)

MBU       = GM_bytes / span / 1.439e12
```

`GM_bytes` 是逻辑必需流量，不包含 Top-K 选中元素的标量重读（这些重读来自
同一行、通常命中 L2/UB，不作为 HBM 必需流量）。

## 3. 表 1：端到端、span 与效率指标

| Case | alu-eq | GM bytes | e2e µs | span µs | TALU-eq/s | TB/s | MFU | MBU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen3.8-flash-next-decode-b1` | 0.05 MF-eq | 1.08 KB | 61.6 | 8.1 | 0.006 | 0.000 | 0.05% | 0.01% |
| `qwen3.8-flash-next-decode-b64` | 3.08 MF-eq | 69.38 KB | 68.6 | 15.6 | 0.197 | 0.004 | 1.67% | 0.31% |
| `qwen3.8-flash-next-decode-b256` | 12.33 MF-eq | 277.50 KB | 114.1 | 34.5 | 0.357 | 0.008 | 3.02% | 0.56% |
| `qwen3.8-flash-next-prefill-2048` | 98.61 MF-eq | 2.22 MB | 290.5 | 230.8 | 0.427 | 0.010 | 3.61% | 0.67% |
| `qwen3.8-flash-next-prefill-8192` | 394.43 MF-eq | 8.88 MB | 963.3 | 899.4 | 0.439 | 0.010 | 3.71% | 0.69% |
| `qwen3.5-moe-decode-b1` | 0.02 MF-eq | 0.56 KB | 63.4 | 6.5 | 0.003 | 0.000 | 0.03% | 0.01% |
| `qwen3.5-moe-decode-b128` | 2.56 MF-eq | 71.68 KB | 115.9 | 16.4 | 0.156 | 0.004 | 1.32% | 0.30% |
| `qwen3.5-moe-prefill-8192` | 163.71 MF-eq | 4.59 MB | 742.1 | 643.4 | 0.254 | 0.007 | 2.15% | 0.50% |
| `qwen3-30b-a3b-decode-b64` | 0.64 MF-eq | 19.46 KB | 64.0 | 11.0 | 0.058 | 0.002 | 0.49% | 0.12% |
| `qwen3-30b-a3b-prefill-2048` | 20.48 MF-eq | 622.59 KB | 203.3 | 149.3 | 0.137 | 0.004 | 1.16% | 0.29% |
| `deepseek-v3-decode-b32-grouped` | 0.69 MF-eq | 18.43 KB | 79.1 | 9.8 | 0.071 | 0.002 | 0.60% | 0.13% |
| `deepseek-v3-prefill-2048-grouped` | 44.42 MF-eq | 1.15 MB | 319.8 | 265.5 | 0.167 | 0.004 | 1.41% | 0.30% |
| `norm-output-2048x256-k8-fp32` | 40.40 MF-eq | 4.33 MB | 265.4 | 171.7 | 0.235 | 0.025 | 1.99% | 1.75% |

读法：

- 所有 case 的 **MBU < 1.8%**，GM 流量远不是瓶颈；
- MFU 最高 3.71%，且 prefill 大档随 rows 增长趋于饱和（Qwen3.8 2048→8192
  仅 3.61%→3.71%），说明语义 ALU 不是限制项；
- e2e − span 的差值主要是 Python/custom-op 与同步开销：
  decode B1 为 53.5 µs（8.1 µs kernel），prefill 8192 为 63.9 µs；
- `norm-output` FP32 档 MBU 1.75% 最高，因为要额外写 `[R,E]` FP32 norm 表，
  但仍然是 Vector/Scalar 受限。

## 4. 表 2：msprof 流水占用

> 纯 Vector kernel：`cube/mte1/fixp` 不适用，表中记 `—`；`mte2` 为 AIV MTE2
> （GM→UB），`aiv-scalar` 为 AIV Scalar，`aiv-mte3` 为 UB→GM。`dur` 取
> `OpBasicInfo.csv` 的 Task Duration；ratio 为 56 个 sub_block 的均值。

| Case | dur µs | cube | scalar | mte1 | mte2 | fixp | vec | aiv-scalar | aiv-mte3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen3.8-flash-next-decode-b1` | 8.2 | — | — | — | 0.05 | — | 0.64 | 0.46 | 0.05 |
| `qwen3.8-flash-next-decode-b64` | 15.7 | — | — | — | 0.05 | — | 0.57 | 0.54 | 0.04 |
| `qwen3.8-flash-next-decode-b256` | 33.9 | — | — | — | 0.06 | — | 0.67 | 0.50 | 0.03 |
| `qwen3.8-flash-next-prefill-2048` | 228.9 | — | — | — | 0.06 | — | 0.70 | 0.48 | 0.03 |
| `qwen3.8-flash-next-prefill-8192` | 897.3 | — | — | — | 0.06 | — | 0.71 | 0.48 | 0.02 |
| `qwen3.5-moe-prefill-8192` | 641.4 | — | — | — | 0.07 | — | 0.69 | 0.53 | 0.03 |
| `qwen3-30b-a3b-prefill-2048` | 148.6 | — | — | — | 0.07 | — | 0.68 | 0.54 | 0.04 |
| `deepseek-v3-prefill-2048-grouped` | 265.7 | — | — | — | 0.04 | — | 0.82 | 0.33 | 0.02 |
| `norm-output-2048x256-k8-fp32` | 171.4 | — | — | — | 0.08 | — | 0.65 | 0.53 | 0.05 |

读法与结论：

- **Vector pipe 是主占用**：0.57-0.82；DeepSeek grouped 档最高（0.82），
  因为 group score / group top-k 增加了一组 reduction 与 mask；
- **Scalar pipe 是第二占用**：0.33-0.54，来自 K 次 Top-K 的索引比较、
  标量 load、循环控制和 `tl.where` 更新；
- **MTE2 只有 0.04-0.08**：每行只读一次 `[E]` logits（外加少量标量重读），
  远低于 HBM 上限；Qwen3.8 8192 的 kernel 并非权重/激活流送型 kernel；
- **MTE3 只有 0.02-0.05**：输出 `[R,K]` 很小，norm-output 档也只增加
  `[R,E]` FP32 写；
- decode B1 的 `Block Dim = 1`，单核执行，因此 8.2 µs 主要是单核
  reduction + 控制链；decode B64 起铺满 56 个 VectorCore，但每核只处理
  1-2 行，仍受 reduction 延迟与尾块影响。

## 5. 表 3：与 torch / torch_npu 对比

> 同场次冷 L2、NPU event、21 次中位。`Triton/torch` 与 `Triton/torch_npu`
> 为耗时比：**<1 表示 Triton 更快，>1 表示 Triton 更慢**。
> torch 为 `implementation="torch"` 的小算子参考；torch_npu 为
> `torch_npu.npu_moe_gating_top_k`（CANN 原生实现）。

| Case | Triton µs | torch µs | torch_npu µs | Triton/torch | Triton/torch_npu |
|---|---:|---:|---:|---:|---:|
| `qwen3.8-flash-next-decode-b1` | 52.8 | 67.9 | 13.8 | **0.78×** | 3.83× |
| `qwen3.8-flash-next-decode-b64` | 60.2 | 46.9 | 11.3 | 1.28× | 5.33× |
| `qwen3.8-flash-next-decode-b256` | 69.3 | 50.9 | 15.1 | 1.36× | 4.60× |
| `qwen3.8-flash-next-prefill-2048` | 264.7 | 73.2 | 41.5 | 3.62× | 6.38× |
| `qwen3.8-flash-next-prefill-8192` | 932.0 | 159.5 | 134.2 | 5.84× | 6.95× |
| `qwen3.5-moe-decode-b1` | 48.0 | 53.6 | 10.9 | **0.90×** | 4.40× |
| `qwen3.5-moe-decode-b128` | 51.0 | 58.8 | 14.2 | **0.87×** | 3.60× |
| `qwen3.5-moe-prefill-8192` | 679.5 | 113.0 | 92.6 | 6.01× | 7.34× |
| `qwen3-30b-a3b-decode-b64` | 57.3 | 66.2 | 15.3 | **0.87×** | 3.76× |
| `qwen3-30b-a3b-prefill-2048` | 181.7 | 53.4 | 22.2 | 3.40× | 8.17× |
| `deepseek-v3-decode-b32-grouped` | 47.0 | 172.9 | 12.4 | **0.27×** | 3.78× |
| `deepseek-v3-prefill-2048-grouped` | 303.4 | 167.8 | 45.4 | 1.81× | 6.68× |
| `norm-output-2048x256-k8-fp32` | 208.4 | 44.0 | 32.8 | 4.73× | 6.35× |

读法：

- **decode 小档 Triton 有优势**：Qwen3.8 B1、Qwen3.5 B1/B128、
  Qwen3-30B B64、DeepSeek grouped decode 均快于 torch 小算子
  （1.1-3.7×；DeepSeek grouped decode 为 0.27×，即 3.7× faster），
  因为 torch 路径要启动 softmax/topk/gather/renorm 多个 kernel；
  Qwen3.8 B64/B256 则被 torch 反超 1.28-1.36×；
- **prefill 大档 torch 反超**：torch 的 `softmax + torch.topk` 使用高度优化
  的原生算子，而本 kernel 的 K 次 repeated max/min 与标量控制随 rows 线性
  放大；Qwen3.8 8192 上 torch 159.5 µs、Triton 932.0 µs；
- **torch_npu 原生实现仍显著领先**：3.6-8.2×。CANN routing 是单 kernel
  且针对 Top-K 选择做了平台优化，本 Triton 版本目前没有对应指令级优化；
- 当前实现适合作为**非 CANN / 可移植 routing fallback**，或在 decode 小档
  作为 fused softmax+topk+renorm 的低 launch 开销实现。

## 6. 结论与后续优化方向

1. **性能瓶颈已明确**：不是 MTE2/MTE3，而是 AIV Vector + Scalar 的
   repeated reduction / Top-K 选择链。优化优先级应从“减少 GM 字节”转向
   “减少 K 次全行 reduce 与标量索引控制”。
2. **Top-K 选择是主要成本**：当前每行 K 次执行
   `max -> compare/select -> min -> mask update`，每次都是全行 reduction。
   后续可评估：
   - 基于分块 top-k + merge 的选择网络，减少全行 reduction 次数；
   - 将 value/index 打包为可比较 key，用一次 max 同时得到 value 与 index，
     替代 value max + index min 两次 reduction；
   - 对 Qwen 的 `G=1` 路径做专用 kernel，去掉 group tile 与 group mask 开销；
   - decode 小 rows 继续走单核/少核低 launch 路径，prefill 大 rows 评估
     更多核或分块并行。
3. **MFU 口径说明**：本报告 MFU 仅衡量语义 ALU-eq，Top-K 的 reduction、
   mask、索引和标量控制不计入分子，因此 MFU 低（≤3.71%）不代表 Vector pipe
   空闲；pipe 表显示 Vector 实际占用 0.57-0.82。若后续需要“指令级
   utilization”，应以 msprof Vector/Scalar pipe 占用为主指标。
4. **与 CANN 原生差距**：prefill 大档 6-8×，decode 3.6-5.3×。短期内建议
   保持 CANN 原生 routing 为生产路径，Triton 版本定位为可移植 fallback；
   若要以性能替换为目标，需要专项重写 Top-K 选择算法，而不是继续微调
   load/store。

## 7. 复现命令

```bash
# 精度 + 性能用例（冷 L2 e2e/event/span）
pytest tests/performance/functions/test_moe_gating_top_k.py \
  --perf-timers profiler,e2e,event --perf-l2-flush \
  --perf-output /tmp/opencode/moe_gating_top_k_perf_l2flush.json

# msprof 流水（以 Qwen3.8 prefill 8192 为例）
msprof op --output=/tmp/opencode/msprof_moe/qwen38_p8192 \
  --aic-metrics=PipeUtilization --launch-count=1 \
  --launch-skip-before-match=3 --kernel-name='*moe_gating_top_k*' \
  python /tmp/opencode/moe_gating_msprof_runner.py \
  8192 512 10 1 1 0 0 1 1.0 bf16 0 0
```
