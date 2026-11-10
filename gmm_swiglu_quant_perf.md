# gmm_swiglu_quant 性能报告（首轮）

> 由 `swiglu_gated_mlp` 重构而来：去除第二次 mm、MXFP8 输入输出、融合 MX 量化
> （参考 ops-transformer `gmm/grouped_matmul_swiglu_quant_v2`，950PR MX 路径语义）。
> 历史 bf16 两段式数据见 `swiglu_gated_mlp_perf.md`（stage-1/mm1 列可对照）。

## 环境

| 项目 | 值 |
| --- | --- |
| 目标 / 设备 | `npu.a5` / Ascend 950PR（共享环境） |
| 实现 | `triton`（`_gmm_swiglu_quant_kernel`：双 `tl.dot_scaled` + 片上 SwiGLU/MX-OCP 量化 epilogue） |
| 软件 | triton_ascend 3.6.0+dev（`tl.dot_scaled` → `hfusion.matmul_mx`） |
| 原始数据 | `gmm_swiglu_quant_perf{,_v2,_v3}.json`（仓库根目录，perf 套件同口径） |

## 测量方式

与历史报告同口径：warmup 3 / repeats 5 / instances 2（rotate）、每次计时前 192 MiB
buffer 求和清空 L2（窗口外）、kernel 时间取 profiler 逐 launch 均值。负载与历史
8 档一致（M-K-I-L，fp8 版 weight 为 MXFP8 随机量化、swiglu_limit 不变）。

## 结果（kernel µs，冷 L2）

| Case | v1 首版 | v3 优化后 | 收益 | 旧 bf16 mm1 | v3/旧 |
|---|---:|---:|---:|---:|---:|
| `1-6144-12288-0.0` | 1634.9 | 400.8 | 4.1× | 204.4 | 0.51 |
| `512-6144-12288-0.0` | 7155.8 | 1792.8 | 4.0× | 430.5 | 0.24 |
| `4096-6144-12288-0.0` | 54284.0 | 14053.3 | 3.9× | 3332.6 | 0.24 |
| `1-4096-12288-10.0` | 1103.8 | 275.1 | 4.0× | 146.1 | 0.53 |
| `512-4096-12288-10.0` | 4901.6 | 1303.1 | 3.8× | 326.5 | 0.25 |
| `4096-4096-12288-10.0` | 36980.6 | 10186.5 | 3.6× | 2318.9 | 0.23 |
| `64-6144-2048-0.0` | 243.9 | 65.7 | 3.7× | 39.2 | 0.60 |
| `512-4096-2048-10.0` | 698.7 | 191.7 | 3.6× | 72.1 | 0.38 |

（注：旧 mm1 输出 bf16 act、无量化；新算子多出 MX 量化 epilogue 与 fp8 I/O，
对照仅作量级参考。）

## v1 → v3 的两项修复（各约 2×）

1. **全 masked load 降级**：masked fp8/`matmul_mx` load 在本后端降级为
   alloc + zero-fill + copy 的 staging 路径，微基准 ~18× 慢于直接 load
   （`scripts/gmm_swiglu_quant_variants_bench.py`）。修复：EVEN_M/N/K
   constexpr 快速路径——形状对齐 tile 时无 mask load/store（模型形状全部命中；
   尾部形状仍走 masked，精度用例覆盖两路径）。
2. **weight_scale pair 打包交错寻址**：参考仓 `(kp, N, 2)` 打包沿 N 步长 2
   交错，kernel 内 gather 无法向量化，微基准单独贡献 ~2×。修复：host 一次性
   重排为连续 `(k_blocks, N)`（`PreparedWeightCache` 缓存，权重静态只排一次），
   对外 API 布局不变。

## 穷尽过的 kernel 侧杠杆（均无收益或负收益）

| 杠杆 | 结果 |
|---|---|
| tile 扫描（BM/BN/BK 8 档，`scripts/gmm_swiglu_quant_sweep_probe.py`） | BM=128 各配置持平 86–87 TF/s 平台期；BM=64 减半 |
| `dot_scaled(fast_math=True)` | 慢 2.2× |
| 手工 32 块小 dot + epilogue 融合 scale | 慢 2.5×（K=32 小 dot 无效率） |
| 双累加器 vs 单（微基准对照） | 无差（双 dot 共享 A tile 免费保持） |

## 结论与差距定性

1. **剩余 2–4× 差距在平台侧**：微基准显示本 triton-ascend 构建 fp8/MX dot
   （`hfusion.matmul_mx`）单核吞吐仅为 bf16 dot 的 ~0.55×（13.5 vs 8.7 µs/同
   tile/单 dot），符合上游文档"无原生 MX 支持时软件模拟：先上抛 bf16 再 dot"
   的描述——逐元素转换开销盖过了 fp8 的理论 2× 算力与一半权重流量。vendor
   AscendC cgmct 路径（`kernel_gmm_swiglu_hif4quant.h`）在 950PR 上以
   `SetQuantVector`/MMAD 原生跑满 MX，triton 侧暂无对应暴露。
2. decode 档（M=1）带宽优势也未兑现（0.51×）：bf16 时代该档 MAC 仅 16%、
   纯权重带宽瓶颈；fp8 权重字节减半但被 dot 速率盖过。
3. 后续方向：(a) 平台暴露原生 MX MMAD（对齐 cgmct 的 `SetQuantVector` 融合），
   预期直接吃满 ~2× bf16 算力；(b) 在此之前如需极致吞吐，可评估 bf16 权重 +
   动态 MX 量化激活的混合方案（语义不同，需产品侧决策）。

## 附：本轮踩到的后端缺陷清单（均有复现脚本）

| 现象 | 复现 |
|---|---|
| `libdevice.rint` 打挂 AIC（极简 kernel 亦然） | `scripts/gmm_swiglu_quant_rint_probe.py`（kernel 内联 banker's rint 绕开） |
| `dot_scaled` 非 2 幂 scale tile（BK=96/BN=160）触发 cube 指令异常 | `scripts/gmm_swiglu_quant_tile_probe.py`（tile 规则 BN/BK 取 2 幂） |
| torch_npu 全常量索引 gather 返回错值 | `scripts/gmm_swiglu_quant_gather_probe.py`（测试用 `repeat_interleave` 绕开） |
| aclnn fp32 matmul `[512,6144]@[6144,4096]` 新进程稳定挂起 | `scripts/gmm_swiglu_quant_refbis_probe.py`（仅影响 torch 参考实现路径） |
| masked fp8 load staging ~18×；pair 打包 scale 寻址 ~2× | `scripts/gmm_swiglu_quant_variants_bench.py`（本文两项修复的依据） |
