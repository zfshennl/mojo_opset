# gmm_swiglu_quant 性能报告（首轮）

> 由 `swiglu_gated_mlp` 重构而来：去除第二次 mm、MXFP8 输入输出、融合 MX 量化
> （参考 ops-transformer `gmm/grouped_matmul_swiglu_quant_v2`，950PR MX 路径语义）。

## 环境

| 项目 | 值 |
| --- | --- |
| 目标 / 设备 | `npu.a5` / Ascend 950PR（共享环境） |
| 实现 | `triton`（`_gmm_swiglu_quant_kernel`：双 `tl.dot_scaled` + 片上 SwiGLU/MX-OCP 量化 epilogue，单一 masked 路径） |
| 软件 | triton_ascend 3.6.0+dev（`tl.dot_scaled` → `hfusion.matmul_mx`） |
| 原始数据 | `gmm_swiglu_quant_perf{,_v2,_v3,_v4}.json`（perf 套件） |

## 测量方式

- kernel 时间：perf 套件同历史口径（warmup 3 / repeats 5 / instances 2、每次计时前
  192 MiB buffer 求和清空 L2、profiler 逐 launch 均值）；
- 与小算子基线对比：同口径冷 L2、NPU event 计时、9 次取中位
  （`scripts/gmm_swiglu_quant_vs_torch.py`）；
- 基线为 **vllm 小算子实现（冻结，torch 官方 API）**：取自 `~/vllm`
  `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py`，逐字调用、
  零改动——`dequant_mxfp8_to_bf16`（MXFP8 反量化）与
  `_mxfp8_e4m3_quantize_torch`（MXFP8 量化，ceil/cuBLAS 式 scale）；中间
  matmul/clamp/silu/mul 为 torch 官方算子。不走其 triton 镜像/分发器，
  不含任何 npu_* 私有算子。权重静态布局（`[N,K]`+`[N,K/32]`）在计时窗口外
  一次性完成。数值已对拍：基线与本算子对同一 fp32 oracle 的最大相对误差
  一致（3.9%，各自合法 MX 量化步）。

## 结果 1：kernel 时间演进（冷 L2，µs）

| Case | v1 初版 | v4 终版 | 收益 |
|---|---:|---:|---:|
| `1-6144-12288-0.0` | 1634.9 | **240.7** | 6.8× |
| `512-6144-12288-0.0` | 7155.8 | **1236.7** | 5.8× |
| `4096-6144-12288-0.0` | 54284.0 | **9676.8** | 5.6× |
| `1-4096-12288-10.0` | 1103.8 | **167.0** | 6.6× |
| `512-4096-12288-10.0` | 4901.6 | **876.1** | 5.6× |
| `4096-4096-12288-10.0` | 36980.6 | **6838.5** | 5.4× |
| `64-6144-2048-0.0` | 243.9 | **77.6** | 3.1× |
| `512-4096-2048-10.0` | 698.7 | **192.2** | 3.6× |

（v2/v3 为中间实验版：v2 曾引入 EVEN_M/N/K 无 mask 分支、v3 叠加 scale 重排；
v4 按评审移除无 mask 分支后**普遍快于 v3**——masked staging 的代价在重排后的
kernel 中不再成立，EVEN 分支反致代码劣化。收益几乎全部来自下述重排。v4 终版
epilogue 同时对齐了 V3 数值：S 先 rint→BF16 再量化、元素单次 RTNE 直 cast，
比旧手写整数格 rint 少一组向量指令，速度再快 0.2-2%。）

## 结果 2：vs vllm 小算子实现（冻结基线，冷 L2 event 中位，µs）

| Case | triton | vllm 小算子 | vllm/triton |
|---|---:|---:|---:|
| `1-6144-12288-0.0`（decode） | 322.4 | 2370.2 | **7.4×** |
| `512-6144-12288-0.0` | 1302.1 | 2758.1 | **2.1×** |
| `4096-6144-12288-0.0` | 9752.0 | 7369.3 | 0.76× |
| `1-4096-12288-10.0`（decode） | 232.1 | 1601.2 | **6.9×** |
| `512-4096-12288-10.0` | 952.4 | 1919.5 | **2.0×** |
| `4096-4096-12288-10.0` | 6931.8 | 5864.6 | 0.85× |
| `64-6144-2048-0.0` | 134.2 | 419.6 | **3.1×** |
| `512-4096-2048-10.0` | 253.5 | 321.4 | **1.3×** |

- 融合收益集中在小/中 M：省去 dequant 物化（`[M,K]`+`[N,K]` bf16 中间量）与
  独立量化 pass（vllm 量化器为多趟 elementwise），decode 达 **6.9-7.4×**；
- 全批 M=4096 两档落后（0.76-0.85×）：基线的大 bf16 matmul 满速运行，而本
  kernel 的 MX dot 受下述平台路径限速，融合省下的中间量流量不足以反超。

## 结果 3：vs vendor 融合算子（`torch_npu.npu_grouped_matmul_swiglu_quant_v2`）

平台原生 MX MMAD 融合实现的同形状实测（`scripts/gmm_swiglu_quant_vs_npu_v2.py`，
同口径冷 L2 event 中位；该算子原生接受本算子的张量打包，权重零转换直喂；
数值 sanity：对同一 fp32 oracle 最大相对误差 3.9%，与本算子一致）：

| Case | triton | vendor | triton/vendor |
|---|---:|---:|---:|
| `1-6144-12288-0.0`（decode） | 316.6 | 100.5 | 3.15× |
| `512-6144-12288-0.0` | 1305.3 | 283.6 | 4.60× |
| `4096-6144-12288-0.0` | 9750.4 | 1756.0 | 5.55× |
| `1-4096-12288-10.0`（decode） | 226.8 | 68.0 | 3.34× |
| `512-4096-12288-10.0` | 939.3 | 201.2 | 4.67× |
| `4096-4096-12288-10.0` | 6901.4 | 1183.8 | 5.83× |
| `64-6144-2048-0.0` | 137.9 | 47.2 | 2.92× |
| `512-4096-2048-10.0` | 252.2 | 73.3 | 3.44× |

- 交叉验证"原生 MX ≈ 2× bf16"：vendor 对旧 bf16 mm1 kernel（历史 v4 表）恰为
  1.9-2.0×（1756 vs 3333、100.5 vs 204.4 µs）——与 §结论中微基准的推断一致；
- 本 kernel 落后 vendor 2.9-5.8×，其中大 M 档差距最大（dot 吞吐主导），decode
  档最小（带宽主导、fp8 权重字节减半部分兑现）。

## 核心修复：weight_scale 一次性重排（贡献全部 ~5×）

参考仓 `(kp, N, 2)` pair 打包在 kernel 内沿 N 步长 2 交错寻址，无法向量化，
微基准单独 ~2×、生产综合 ~5×。修复：host 端经 `PreparedWeightCache`（权重静态，
每权重仅一次）重排为连续 `(k_blocks, N)`，kernel 按 `kb * N + n` 连续读；
**对外 API 布局不变**。证据：`scripts/gmm_swiglu_quant_variants_bench.py`。

## 穷尽过的其他杠杆（无收益/负收益，均留证据）

| 杠杆 | 结果 |
|---|---|
| EVEN_M/N/K 无 mask 分支 | 移除后反而更快（见上）；已按评审删除 |
| tile 扫描（BM/BN/BK 8 档，`scripts/gmm_swiglu_quant_sweep_probe.py`） | BM=128 各配置持平平台期；BM=64 减半 |
| `dot_scaled(fast_math=True)` | 慢 2.2× |
| 手工 32 块小 dot + epilogue 融合 scale | 慢 2.5×（K=32 小 dot 无效率） |

## 结论与差距定性

1. 相对 vllm 小算子实现（冻结基线）：**6/8 档 1.3-7.4× 领先，全批两档
   0.76-0.85×**——中小 batch 的融合价值已兑现。
2. 相对 vendor 融合算子（原生 MX MMAD，实测见结果 3）：**落后 2.9-5.8×**。
   经源码核查（triton-ascend + AscendNPU-IR，2026-11），该 triton-ascend 构建的
   MX 路径**已是原生实现而非软件模拟**：`tl.dot_scaled`（fp8 操作数 + e8m0
   scale、2D、f32 累加、k_pack）→ `hfusion.matmul_mx` →（950PR/c310 属
   RegBased 架构）`hivm.mmadmxL1` → catlass `L1MxMmad`（AtlasA5 模板，硬件
   MMAD + `loadMXScale`）；本 kernel 最终 npubin 中实测含 `mmadmxL1` 符号。
   软件解码 fallback 确实存在（TritonOpConverter `useNativeMx` 条件不满足时：
   逐元素上抛 bf16 + scale 展开 + 普通 matmul），但本 kernel 形状未命中。
   差距的真实根因是 triton-ascend 侧 MX 降级栈（HFusion/HIVM 调度 + catlass
   模板）的**实现效率**（模板内 L0A/L0B ping-pong 减半、额外 scale 加载与
   布局转换），相对 vendor ACLNN 直写 AscendC（对旧 bf16 mm1 实测 1.9-2.0×，
   即 fp8 理论 2× 兑现）仍有 2.9-5.8× 空间，优化主体在平台侧模板/管线。

## 附：本轮踩到的后端缺陷清单（均有复现脚本）

| 现象 | 复现 |
|---|---|
| `libdevice.rint` 打挂 AIC（极简 kernel 亦然） | `scripts/gmm_swiglu_quant_rint_probe.py`（kernel 内联 banker's rint 绕开） |
| `dot_scaled` 非 2 幂 scale tile（BK=96/BN=160）触发 cube 指令异常 | `scripts/gmm_swiglu_quant_tile_probe.py`（tile 规则 BN/BK 取 2 幂） |
| torch_npu 全常量索引 gather 返回错值 | `scripts/gmm_swiglu_quant_gather_probe.py`（测试用 `repeat_interleave` 绕开） |
| aclnn fp32 matmul `[512,6144]@[6144,4096]` 新进程稳定挂起 | `scripts/gmm_swiglu_quant_refbis_probe.py`（torch 参考实现路径受影响，bf16 不受影响） |
| pair 打包 scale 交错寻址 ~2-5× | `scripts/gmm_swiglu_quant_variants_bench.py`（重排修复依据） |
