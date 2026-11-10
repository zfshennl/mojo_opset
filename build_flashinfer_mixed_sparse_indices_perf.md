# build_flashinfer_mixed_sparse_indices 基线性能报告

生成于 2026-11-10 (UTC)，主机 `ascend-950pr-223-2-2`（950pr / **957b** SKU，经 `ASCEND_RT_VISIBLE_DEVICES=6` 挂卡；0/2 号卡健康 Alarm，0 号卡上 192 MiB L2 flush 的 ReduceSum 即向量核超时，6 号卡一切正常）。

```sh
ASCEND_RT_VISIBLE_DEVICES=6 pytest tests/performance/functions/test_build_flashinfer_mixed_sparse_indices.py \
       --perf-l2-flush --perf-trace-dir bfmsi_traces \
       --perf-output tests/performance/results/build_flashinfer_mixed_sparse_indices_baseline_triton.json
```

## 环境

| 项目 | 值 |
| --- | --- |
| 目标 / 设备 | `npu.a5.950pr` / Ascend950PR_9579（957b：56 个 vector core × 1.65 GHz） |
| 设备显存 / L2 | 123.2 GiB / 128 MiB（torch 上报；957b ini 记 112 MiB） |
| 实现 | `triton`（`_build_flashinfer_mixed_sparse_indices_kernel`，persistent 56 program，`mix_simd_simt`） |
| 软件 | python 3.11.17, torch 2.13.0+cpu, torch_npu 2.13.0, triton_ascend 3.6.0+dev20261103220353 |
| 版本 | `83e7cce91575`（[Feat] add npu build_sparse_indices kernels，工作树仅多未跟踪报告/轨迹） |
| 原始数据 | `bfmsi_perf.tar.gz`（harness 轨迹 + msprof op_summary + 全部基准脚本）、上述 JSON |

## 测量方式

| 设置 | 值 |
| --- | --- |
| 预热 / 重复次数 / 实例数 | 10 / 20 / 2（输入轮换） |
| L2 flush | 每次被测调用前对 192 MiB buffer 求和清空 L2，位于计时窗口之外 |
| 内核时长 | ① harness profiler span（`trace_view.json` 逐 launch，20 次）；② msprof `--aic-metrics=PipeUtilization --aic-mode=task-based`（每 case 2 warmup + 5 采集，取中位），aiv_* 管线占比与 `aiv_total_cycles` 均取自 `op_summary_*.csv` |
| e2e | profiler 开启下的主机同步墙钟时间，包含插桩开销 |

## 负载（DeepSeek-V4.1-Flash 服务形态，int32 元数据）

Case 为 `tests/performance/functions/test_build_flashinfer_mixed_sparse_indices.py` 的参数化 id。
全部取 `sliding_window=128`（SWA 段宽 128）、`index_topk=512`、SWA/压缩块 64（ratio2 压缩块 32）、
decode 局部压缩索引（V4.1 `topk_indices_buffer` 路径）。

| 参数 id | decode tok | prefill tok | ratio | topk | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| `dsv41_decode_ratio0` | 256 | 0 | 0 | 0 | 纯 SWA 层（无 topk 段） |
| `dsv41_decode_ratio1` | 256 | 0 | 1 | 512 | decode batch |
| `dsv41_decode_ratio2` | 256 | 0 | 2 | 512 | decode batch |
| `dsv41_spec_decode_ratio1` | 768 | 0 | 1 | 512 | 128 请求 × 6 token（末 token 无效） |
| `dsv41_mixed_ratio1` | 128 | 3840 | 1 | 512 | decode + 2 条 prefill（含 replay） |
| `dsv41_mixed_ratio2` | 128 | 3840 | 2 | 512 | 同上，ratio2 |
| `dsv41_prefill_ratio1` | 0 | 4096 | 1 | 512 | chunked prefill |
| `dsv41_long_chunk_ratio2` | 0 | 4096 | 2 | 512 | 32K 长序列单 chunk |

## 结果

vec / scalar / mte2 / mte3 = msprof 报告的对应管线时间占比中位。kernel 列同时给
harness span 中位（p10–p90）与 msprof Task Duration 中位。

| Case | kernel msprof µs | kernel span µs (p10–p90) | vec | scalar | mte2 | mte3 | e2e µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_ratio0` | 7.1 | 7.3 (7.0–7.7) | ≈0 | ≈0 | ≈0 | ≈0 | 65.0 |
| `decode_ratio1` | 12.0 | 13.3 (12.7–13.9) | 67% | 43% | 9% | 35% | 70.5 |
| `decode_ratio2` | 11.8 | 13.5 (12.8–14.0) | 68% | 41% | 9% | 41% | 88.1 |
| `spec_decode_ratio1` | 24.3 | 27.2 (26.7–27.7) | 85% | 34% | 13% | 49% | 107.3 |
| `mixed_ratio1` | 121.1 | 122.4 (121.8–123.0) | **98%** | 25% | 7% | 46% | 192.1 |
| `mixed_ratio2` | 121.4 | 122.9 (122.4–123.4) | **98%** | 27% | 7% | 49% | 184.4 |
| `prefill_ratio1` | 126.9 | 130.6 (129.9–131.2) | **97%** | 25% | 7% | 47% | 192.3 |
| `long_chunk_ratio2` | 126.2 | 127.1 (126.6–127.9) | **98%** | 27% | 7% | 42% | 186.1 |

`aiv_time ≈ duration`（核占用 ~99%），`aiv_total_cycles/时长/56 ≈ 1.61 GHz`，自洽。

## Bound 分析

该算子输出 `[N, 128+512]` int32 索引矩阵，每列的工作 = 块号换算（1 div + 1 mod + mul + add，
块表 gather）+ FlashInfer 槽位 remap（再 1 div + 1 mod + mul + add）+ 掩码 cmp/select。分解依据
（复现脚本在 `bfmsi_perf.tar.gz`）：

- **ttadapter IR**：块表访问降级为 `triton_indirect_load(i64 索引张量)`（512/128 lane 逐 lane 间接
  读），标量元数据走 SIMT（`parallel_mode=mix_simd_simt`，scalar 管线 25–43%）。
- **int32 div/mod 链式微基准**（25.2M 元素，ITERS=64，56 program）：div = **7.8×** add、
  mod = **8.2×** add（链式，含时延放大）——A5 标定库里 Div 按 fp32 超越管线 `valu_2slot_t4` 计
  4×，对 int32 除法是低估，本报告按 14/15 ALU-eq 修正。
- **组件 knockout**（prefill_ratio1 同负载逐项关闭，事件计时中位）：full 144.2 µs → 去 gather
  79.6 µs（**gather ≈ 45%**）→ div/mod 换 constexpr 移位 105.3 µs（**索引 div/mod ≈ 27%**）→ 跳过
  remap（span==size 恒等）125.7 µs（**remap ≈ 13%**）→ 全关 52.0 µs；纯流（load+store）44.8 µs
  （≈ 18.9 MB / 44.8 µs ≈ 422 GB/s，2.5 KB/行 × 56 program 的行粒度访问）。

结论：

- **大负载（mixed/prefill/long_chunk）为向量发射受限**：vec 管线 97–98% 占用，mte2 仅 7%，
  有效带宽 ≈ 150 GB/s（MBU ≈ 11%），远未触及 HBM。单列成本 ≈ 4.3 cycles（≈ 550 issue-slot
  ALU-eq/列），其中接近一半是 `triton_indirect_load` 的逐 lane 间接读 + i64 索引张量
  （extsi/addi 64 位）展开，其余是 int32 div/mod 模拟与掩码逻辑——不是数学意义上的"计算量"，
  而是地址计算与发射带宽。
- **小 decode 负载为访存时延受限**：256 token / 56 program ≈ 5 token/program，每 token 一趟
  load→换算→store 的串行链。`decode_ratio0` 各管线占比全部 ≈0、核占用 99%（aiv_time 6.58 /
  6.83 µs），12.1 cycles/列远超算术模型 0.28——纯 GM 往返时延堆叠；7.1 µs 也高于 1.6 µs 的
  launch floor。ratio1/2（有 topk 段 + gather）vec 升到 67%，开始转为发射/时延混合。
- **e2e 由主机 glue 主导**：kernel 7–131 µs vs e2e 65–192 µs。这是每层每步一次的小元数据算子，
  真实收益看 kernel 侧；e2e 差值基本是 launch/插桩开销。

---

# MBU / MFU

按 situ_and_mul_perf.md 的成本模型契约组织：**数据量** = `bytes_from_io`（声明 I/O 张量全量，
int32）。两处口径注意：① decode 路径只 remap 预计算 SWA 索引、不读 `swa_block_table`
（ratio0 的声明量含 128 KB 从不读的表，其真实流量 ≈ 261 KB）；② 块表按全量计，gather 实际
只触碰子集（decode 负载块表占声明量 ~16%，偏差上限 ±10%）。
**计算量** = ALU-eq 两口径：

- **JSON 口径**：ops-theoretical-performance-estimation 的执行模型——int32 Add/Sub/Mul =
  `alu_2slot_t2` → 2；cmp/select/and/or = `alu_2slot_t1` → 1；Div = `valu_2slot_t4` → 4（Mod 同）；
  ReduceSum = `reduce_1slot_t1` → 2；gather 的 i64 `extsi+addi` 按 dtype 宽度加倍 → 8。
- **CAL 口径**：div/mod 换成上机实测 14/15（链式微基准，含 ~2.2× 时延放大，偏保守的上界估计）。

每列 ALU-eq（JSON/CAL）：decode-SWA 15/36；decode-topk 44/86；prefill-SWA 42/84；
prefill-topk 42/84。每 token = 128×SWA + 512×topk。

**分母**：向量峰值 = 56 核 × 1.65 GHz × 128 元素-op/周期 = **11.83 T ALU-eq/s**（标定模型自洽
口径）；HBM 峰值 = **1.4 TB/s**（skill 957b 档；本机 320 MiB d2d copy 实测 1.431 TB/s）。

## 数据量与 MBU

| Case | 数据量 (in+out) | 有效带宽 | **MBU** (÷1.4 TB/s) | 访存下界 |
| --- | ---: | ---: | ---: | ---: |
| `decode_ratio0` | 389 KB | 56 GB/s | 4.0% | 0.3 µs |
| `decode_ratio1` | 1.51 MiB | 132 GB/s | 9.4% | 1.1 µs |
| `decode_ratio2` | 1.51 MiB | 133 GB/s | 9.5% | 1.1 µs |
| `spec_decode_ratio1` | 3.81 MiB | 165 GB/s | 11.8% | 2.9 µs |
| `mixed_ratio1` | 17.66 MiB | 153 GB/s | 10.9% | 13.2 µs |
| `mixed_ratio2` | 17.66 MiB | 153 GB/s | 10.9% | 13.2 µs |
| `prefill_ratio1` | 18.03 MiB | 149 GB/s | 10.6% | 13.5 µs |
| `long_chunk_ratio2` | 18.04 MiB | 150 GB/s | 10.7% | 13.5 µs |

（kernel 时长取 msprof 中位。）

## 计算量与 MFU

| Case | ALU-eq (JSON) | ALU-eq (CAL) | **MFU** (JSON) | **MFU** (CAL) | cycles/列 实测 | cycles/列 模型(CAL) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode_ratio0` | 491,520 | 1,179,648 | 0.6% | 1.4% | 12.06 | 0.28 |
| `decode_ratio1` | 6,258,688 | 12,451,840 | 4.4% | 8.8% | 5.29 | 0.59 |
| `decode_ratio2` | 6,258,688 | 12,451,840 | 4.5% | 8.9% | 5.19 | 0.59 |
| `spec_decode_ratio1` | 18,776,064 | 37,355,520 | 6.5% | 13.0% | 4.04 | 0.59 |
| `mixed_ratio1` | 106,348,544 | 212,664,320 | 7.4% | 14.9% | 4.31 | 0.65 |
| `mixed_ratio2` | 106,348,544 | 212,664,320 | 7.4% | 14.8% | 4.32 | 0.65 |
| `prefill_ratio1` | 110,100,480 | 220,200,960 | 7.3% | 14.7% | 4.38 | 0.66 |
| `long_chunk_ratio2` | 110,100,480 | 220,200,960 | 7.4% | 14.8% | 4.34 | 0.66 |

（若沿用 situ_and_mul_perf.md 的外部峰值 27 TOPS 作分母，MFU 数值 ×0.44，如 prefill CAL
14.7% → 6.4%。）

## 口径校验与结论

- **MFU 低 ≠ 硬件空闲**：大 case vec 管线占用 97–98%、核占用 ~99%，向量发射口已满。CAL 模型
  只解释实测周期的 ~15%（cycles/列 4.3 vs 0.65），缺口 ≈ 550 ALU-eq/列的隐含发射量，来自
  `triton_indirect_load` 逐 lane 间接读、i64 索引运算、`linalg.fill` 常量广播与掩码逻辑——
  knockout 实测其中 gather 一项就占 ~45%。对这类地址计算受限的元数据算子，**利用率以管线
  占用为准**，MFU 反映的是"纯数学流模型 vs 实际指令流"的差距。
- **ratio0（纯 SWA 层）是例外**：无 topk 段、无 gather，纯 remap 流水在 5 token/program 的
  串行链上跑不满任何管线（12.1 cycles/列全是访存时延），属 launch/时延地板。
- **对 DSV4.1 服务的影响**：这是每个稀疏 MLA 层每步都要跑一次的元数据准备算子，
  decode batch 12 µs、mixed prefill ~121 µs 每层每步固定计入，且当前 ~45% 花在块表 gather 上。

## 优化方向（均有 knockout 实测背书，prefill_ratio1 负载）

| 改动 | 预期 | 依据 |
| --- | ---: | --- |
| span == block_size 时跳过 remap（本仓默认正是恒等） | −13% | 144.2 → 125.7 µs |
| 块大小 constexpr 化 / 2 的幂走移位（64/32 均为 2 的幂） | −27% | 144.2 → 105.3 µs |
| 两者叠加 | ≈ −40% | 外推 ≈ 87 µs（105.3 − 18.5） |
| 块表 gather 结构优化（行驻 UB / 预乘 block_size / 去间接读） | 至多 −45% | 全关后 52.0 µs（保留流式地板 44.8 µs） |

## 说明

- 31 个功能测试全部通过；8 个 perf case 全部通过、无跳过。
- `decode_ratio0/2` 与 `decode_ratio1` 在 op_summary 中 Input Shapes 相同，故 aiv 计数器按
  workload 分会话采集（每 case 独立 msprof 目录）。
- harness span 比 msprof Task Duration 系统性高 1–3 µs（插桩口径差异），两列均已给出。
- A2 版本（npu_a2_triton）含相同逻辑与 192 KiB UB 预算，需 A2 主机补测。
