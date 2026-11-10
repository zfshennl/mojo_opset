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
- 与 torch 对比：同口径冷 L2、NPU event 计时、9 次取中位
  （`scripts/gmm_swiglu_quant_vs_torch.py`）；
- torch 基线为**小算子组合**（无本算子时的真实替代路径）：
  `dequant(x/w×scale) → bf16 aclnn matmul → clamp/silu/mul → MX 量化`
  （bf16 matmul 路径；fp32 组合在本机部分形状触发 aclnn 挂起，见文末）。

## 结果 1：kernel 时间演进（冷 L2，µs）

| Case | v1 初版 | v4 终版 | 收益 |
|---|---:|---:|---:|
| `1-6144-12288-0.0` | 1634.9 | **241.3** | 6.8× |
| `512-6144-12288-0.0` | 7155.8 | **1253.3** | 5.7× |
| `4096-6144-12288-0.0` | 54284.0 | **9809.7** | 5.5× |
| `1-4096-12288-10.0` | 1103.8 | **167.8** | 6.6× |
| `512-4096-12288-10.0` | 4901.6 | **893.5** | 5.5× |
| `4096-4096-12288-10.0` | 36980.6 | **6981.0** | 5.3× |
| `64-6144-2048-0.0` | 243.9 | **78.1** | 3.1× |
| `512-4096-2048-10.0` | 698.7 | **195.5** | 3.6× |

（v2/v3 为中间实验版：v2 曾引入 EVEN_M/N/K 无 mask 分支、v3 叠加 scale 重排；
v4 按评审移除无 mask 分支后**普遍快于 v3**——masked staging 的代价在重排后的
kernel 中不再成立，EVEN 分支反致代码劣化。收益几乎全部来自下述重排。）

## 结果 2：vs torch 小算子组合（冷 L2 event 中位，µs）

| Case | triton | torch 组合 | torch/triton |
|---|---:|---:|---:|
| `1-6144-12288-0.0`（decode） | 314.0 | 3202.4 | **10.2×** |
| `512-6144-12288-0.0` | 1337.0 | 3622.6 | **2.7×** |
| `4096-6144-12288-0.0` | 9895.2 | 8908.9 | 0.90× |
| `1-4096-12288-10.0`（decode） | 241.4 | 2161.2 | **9.0×** |
| `512-4096-12288-10.0` | 983.1 | 2487.3 | **2.5×** |
| `4096-4096-12288-10.0` | 7062.8 | 7061.4 | 1.00× |
| `64-6144-2048-0.0` | 151.6 | 624.5 | **4.1×** |
| `512-4096-2048-10.0` | 269.7 | 489.7 | **1.8×** |

- 融合收益集中在小/中 M：省去 dequant 物化（`[M,K]`+`[K,2I]` bf16 中间量）、
  clamp/silu/mul 逐算子发射与独立量化 pass，decode 达 **9-10×**；
- 全批 M=4096 两档打平/略负（0.90-1.00×）：组合基线的大 bf16 matmul 满速运行，
  而本 kernel 的 MX dot 受下述平台路径限速，融合省下的中间量流量不足以反超。

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

1. 相对 torch 小算子组合：**6/8 档 1.8-10× 领先，全批两档打平**——融合价值已兑现。
2. 全批档剩余差距在平台侧：微基准显示本 triton-ascend 构建 fp8/MX dot
   （`hfusion.matmul_mx`）单核吞吐仅 bf16 dot 的 ~0.55×（13.5 vs 8.7 µs/同
   tile/单 dot），符合上游"无原生 MX 支持时软件模拟：先上抛 bf16"描述；vendor
   AscendC cgmct 路径（`kernel_gmm_swiglu_hif4quant.h`）以 `SetQuantVector`/MMAD
   原生跑满 MX，triton 侧暂无对应暴露。平台补齐后本 kernel 结构可直接受益。

## 附：本轮踩到的后端缺陷清单（均有复现脚本）

| 现象 | 复现 |
|---|---|
| `libdevice.rint` 打挂 AIC（极简 kernel 亦然） | `scripts/gmm_swiglu_quant_rint_probe.py`（kernel 内联 banker's rint 绕开） |
| `dot_scaled` 非 2 幂 scale tile（BK=96/BN=160）触发 cube 指令异常 | `scripts/gmm_swiglu_quant_tile_probe.py`（tile 规则 BN/BK 取 2 幂） |
| torch_npu 全常量索引 gather 返回错值 | `scripts/gmm_swiglu_quant_gather_probe.py`（测试用 `repeat_interleave` 绕开） |
| aclnn fp32 matmul `[512,6144]@[6144,4096]` 新进程稳定挂起 | `scripts/gmm_swiglu_quant_refbis_probe.py`（torch 参考实现/fp32 组合基线受影响，bf16 不受影响） |
| pair 打包 scale 交错寻址 ~2-5× | `scripts/gmm_swiglu_quant_variants_bench.py`（重排修复依据） |
