# gmm_swiglu_quant 性能报告

> 由 `swiglu_gated_mlp` 重构而来：去除第二次 mm、MXFP8 输入输出、融合 MX 量化
> （参考 ops-transformer `gmm/grouped_matmul_swiglu_quant_v2`，950PR MX 路径语义）。
> 本报告按工作时间序组织：§3-§6 四轮工作，§7 现产数据，§8 平台修复项。

## 0. 现状速览

- **现产 kernel**：`_gmm_swiglu_quant_kernel`，API 定稿 N-major +
  **`K_TAIL` 分支**——`K%BLOCK_K==0` 时 scale load 只保留免费的 m/n 行轴
  mask（kb 连续方向 mask 静态恒真，按轴拆分结论省去，`scale_masked/repro.py`）；
  m/n mask 无条件保留，K 尾形状走全 mask 分支（新增测试 `(8,96,32)` 覆盖）；
- **现产 msprof 占比**（4096-6144-12288，`msprof op` sub_block 粒度）：

| sub_block | 时长 µs | cube | aic_mte2 | **aiv vec** |
|---|---:|---:|---:|---:|
| cube0 ×28 | 3482 | 1634 (0.47) | 3472 | — |
| vector0/1 ×56 | 3482 | — | — | **692 (0.20)** |

  → 分支前 vec 8740/0.99 饱和（kb mask × 8B 行逐元素化）；分支后只剩
  SwiGLU+量化 epilogue 的 692µs，**关键路径转为 MTE2 3472µs（G4 碎段）**；
- 现产 vs vllm 冻结基线：**8/8 全档领先（2.03-8.30×，大档 2.03×）**；
  vs vendor：落后 1.68-2.80×（从最初 9.1× 收敛）；
- 历史峰值（v5，含已被指令移除的无 mask 特化与 host 重排）：3654µs；
- 剩余差距的平台降级缺口与修复项见 §8（`gmm_swiglu_quant_ascendc_gap.md`）。

## 1. 环境与口径

| 项目 | 值 |
| --- | --- |
| 目标 / 设备 | `npu.a5` / Ascend 950PR（共享环境） |
| 软件 | triton_ascend 3.6.0+dev（`tl.dot_scaled` → `hfusion.matmul_mx` → `hivm.mmadmxL1`） |
| 原始数据 | `gmm_swiglu_quant_perf{,_v3,_v4,_v5,_masked}.json` |

- kernel 时间：perf 套件历史口径（warmup 3 / repeats 5 / instances 2，计时前
  192 MiB 求和清空 L2，逐 launch 均值）；
- 小算子对比：同口径冷 L2、NPU event、9 次中位（`scripts/gmm_swiglu_quant_vs_torch.py`）；
- 基线为 **vllm 小算子实现（冻结，torch 官方 API）**：`~/vllm`
  `.../quantization/utils/mxfp8_utils.py` 逐字调用，不含 npu_* 私有算子；
  权重静态布局在计时窗口外完成；数值对拍一致（对同一 fp32 oracle 最大相对
  误差均 3.9%）；
- vendor 对比：`scripts/gmm_swiglu_quant_vs_npu_v2.py`
  （`torch_npu.npu_grouped_matmul_swiglu_quant_v2`，原生接受本算子张量打包）。

## 2. 时间线总览

| 阶段 | 内容 | 大档时长（4096-6144-12288，µs） |
|---|---|---:|
| v1 初版 | 直连 pair 打包 scale，交错寻址 | 54284 |
| v4 | host k-major 重排 + masked 单路径 + V3 数值 epilogue | 9969 |
| 定位轮 | msprof op + 消融：AIV vec 饱和（scale gather 3.5ms + mask staging 5.2ms） | — |
| v5 | N-major scale 重排 + EVEN 无 mask 特化 | **3654（峰值）** |
| 评审回撤 | EVEN 移除（禁无 mask）；raw 布局 in-kernel 消费穷尽（全败） | 8840-9084 |
| 规范化 | 禁 `+=` 指针步进、禁指针隐式转置（显式 `tl.trans`，scale 侧 fold 缺失实测 +2.4ms） | 11422 |
| 禁预处理 | raw pair 直连（3D `[BN,P,2]` 块 + reshape，`K_PAD` constexpr）；动态 3D mask 谓词化排障（66.7→15.9ms） | 15892 |
| 现产 | **API 重构：去 pair、scale 轴序随数据张量**（`weight_scale [1,⌈K/32⌉,N]` 等），`[BK/32,BN]` 自然 tile + 显式 `tl.trans`，零预处理 | **11408（现产）** |

### API 演进记录（2026-11-11，本轮定案）

评审决策：**放弃 vendor 的 pair 打包，scale 布局与数据张量轴序一致**——
`weight_scale [1, ⌈K/32⌉, N]`（N 内，随 weight `[1,K,N]`）、`x_scale
[M, ⌈K/32⌉]`（随 x 行主序）、`y_scale [M, ⌈I/32⌉]`。动机：triton 的
`dot_scaled` 前端是逐字节 `[N, K/32]` tile 契约，pair 只服务 MTE 专用道
（`gmm_swiglu_quant_ascendc_gap.md` §1.4/§1.5）；轴序一致使 kernel 的
scale tile 与 fp8 tile 同构（`[BK/32, BN]` 自然加载），x/y 侧与旧布局
字节等价（仅去 pair 视图、尾部精确化）。vendor 对比路径改为脚本内一次性
pair 重组（计时外）。注意：若平台 M1（pair 直通 nd2nz）落地，k-major 字节
布局仍可按 `[kp, N]` b16 reinterpret（相邻两字节即 pair），无需再改 API。

## 3. 第一轮（v1→v4）：host 重排时代，5.3-6.8×

- **问题**：参考仓 `(kp, N, 2)` pair 打包在 kernel 内沿 N 步长 2 交错寻址，
  无法向量化（微基准单独 ~2×、生产综合 ~5×）；
- **修复**：host 端 `PreparedWeightCache`（权重静态，每权重一次）重排为连续
  `(k_blocks, N)` k-major，kernel 按 `kb * N + n` 读；对外 API 布局不变。
  证据：`scripts/gmm_swiglu_quant_variants_bench.py`；
- v4 终版 epilogue 对齐 V3 数值：S 先 rint→BF16 再量化、单次 RTNE 直 cast +
  ±448 clamp（cast 无 satfinite，证据 `scripts/gmm_swiglu_quant_cast_probe.py`）；
- v2/v3 曾引入 EVEN_M/N/K 无 mask 分支，v4 按当时评审移除（其真实收益被
  scale gather 的 vector 饱和掩盖，见 §4-§5）。

| Case | v1 | v4 | 收益 |
|---|---:|---:|---:|
| `1-6144-12288-0.0` | 1634.9 | 518.9 | 3.1× |
| `512-6144-12288-0.0` | 7155.8 | 1506.7 | 4.7× |
| `4096-6144-12288-0.0` | 54284.0 | 9969.2 | 5.4× |
| `4096-4096-12288-10.0` | 36980.6 | 6973.8 | 5.3× |

（v4 列取 `gmm_swiglu_quant_perf_v4.json` 同 rig 复测。）

## 4. 第二轮：pipe 级定位（AIV vector 饱和）

方法：`msprof op --aic-metrics=PipeUtilization`（sub_block 粒度）+ 消融实验。
**教训**：torch profiler `kernel_details.csv` 的 `aiv_time(us)` 列对 mix kernel
为空，须看 `aiv_*_ratio` 列——首轮分析据此误判"AIV 空闲"。

关键事实（v4，4096-6144-12288，时长 9678µs）：
1. **双方 cube busy 一字不差均为 1634µs**——MMAD 与 vendor 完全同速，差距
   100% 在 AIV vector pipe（56 子块 vec 100% 打满 9543µs）；
2. 消融（无 mask + 平凡 epilogue 剥离 kernel）：

| 消融变体 | µs | 归因 |
|---|---:|---|
| 剥离基线 | 6340 | — |
| w-scale 常量化 | 2809 | **w-scale 通路 ≈ 3.5ms** |
| 全 scale 常量化 | 2523 | x-scale 另 ≈ 0.3ms |
| 去掉整条 w2 fp8 流 | 仅 −130 | fp8 字节量几乎无关 |

3. **根因 1**：staged k-major `[K/32, N]` 与 dot_scaled 要求的 `[BN, BK//32]`
   错配 → `n + kb*N` 跨 24KB stride gather+转置，降级 AIV 逐元素
   （llvm-objdump 见 `vector_dma_unalign_vv_2d_vf`/`copy_ubuf_to_ubuf_*`/
   `SIMTIndirectStore2D<u8>` 符号）；2KB/迭代的 uint8 比 96KB/迭代的 fp8
   贵 ~50×/字节；
4. **根因 2**：masked load 的 fill+copy staging 同样落 AIV vector pipe。

## 5. 第三轮（v5）：N-major scale + EVEN 特化 → 峰值 3654µs

- **修复 1**：staged weight_scale 改 N-major `[N, 2kp]`（与 x_scale 同构，
  行内连续）→ scale 通路 AIV vec 6195µs → 32µs（msprof 实证）；
- **修复 2**：EVEN constexpr 特化（M%BM==I%BN==K%BK==0 走无 mask 加载）→
  vec 只剩 epilogue 692µs（0.19），瓶颈转 AIC MTE2（3644µs 打满，w 的 ND
  行跨步 128B 碎段）；
- **峰值数据**（`gmm_swiglu_quant_perf_v5.json` / 两 vs 脚本）：

| Case | v1 | v4 | v5 | v4→v5 |
|---|---:|---:|---:|---:|
| `1-6144-12288-0.0` | 1634.9 | 518.9 | **331.9** | 1.56× |
| `512-6144-12288-0.0` | 7155.8 | 1506.7 | **680.3** | 2.21× |
| `4096-6144-12288-0.0` | 54284.0 | 9969.2 | **3804.0** | 2.62× |
| `1-4096-12288-10.0` | 1103.8 | 371.9 | **247.6** | 1.50× |
| `512-4096-12288-10.0` | 4901.6 | 998.4 | **470.3** | 2.12× |
| `4096-4096-12288-10.0` | 36980.6 | 6973.8 | **2479.6** | 2.81× |
| `64-6144-2048-0.0` | 243.9 | 215.0 | **126.3** | 1.70× |
| `512-4096-2048-10.0` | 698.7 | 324.1 | **156.9** | 2.07× |

  - vs vllm：**8/8 领先 1.93×-7.54×**（原两档 0.76×/0.85× 反超为 1.93×/2.38×）；
  - vs vendor：**1.44×-3.19×**（原 2.9-5.8×），大档收敛 ~2.1×。

## 6. 第四轮：评审约束回撤与 raw 布局穷尽

评审指令：**禁止无 mask 路径**（EVEN 特化移除）、**禁止 host 侧预处理**。

1. **EVEN 移除**：现产回归 always-masked，9678 时代的 masked staging
   （~5.2ms AIV vec）重新成为最大成本（现产 vec 8744µs / 0.99）；
2. **raw `[kp, N, 2]` 布局 in-kernel 消费穷尽**（全败，见矩阵；探针
   `rawpair_probe` / `raw3d_probe` / `subk_probe`）：

| 写法 | masked | µs | msprof |
|---|---|---:|---|
| int16 整块 + 移位 + `tl.interleave` | ✗ | 34974 | — |
| `tl.join`+reshape | ✗ | 35115 | — |
| 3D load `[BN,P,2]`+reshape | ✗ | 13276 | AIV vec 12966 / 0.99 |
| 2D pair 平面+`tl.permute`；指针增量式 | ✗ | 13277 / 13276 | — |
| **sub-block-K（BK=64，`[rows,2]` 纯 2D）** | ✓ | **57768** | — |
| 同上 scale 常量化（隔离 BK=64 本身） | ✓ | 21876 | — |
| 3D load + reshape（全约束合规形态） | ✓ | 16009 | — |
| （对照）staged N-major + 无 mask | ✗ | 3654-3778 | cube 0.45 / MTE2 打满 |

   - sub-block-K 最差：96 次迭代 × 小 tile masked staging + 小 dot 双重崩坏；
   - 寄存器重排类（interleave/join/permute/reshape）一律降级 AIV 逐元素；
     `[kp,N,2]` 中每列 8 个 scale 字节是 4 个 2 字节碎片（间隔 2N），地址
     分布无法用 K 重排改善——in-kernel 无解；
   - xs 无需处理：`[M,kp,2]` 展开即 `[M,2kp]` 行内连续（kb 线性寻址）。

3. **vendor AscendC 的加载机制**（`gmm/common/cgmct/block/block_mmad_mx.h`）：
   pair 字节按 **b16 视图**、`Nd2NzParams{dValue=K对数, nValue=N, srcDValue=n_}`
   由 **MTE1 `DataCopy` GM→L1 直达**，ND→NZ 转换由 MTE 硬件在搬运中完成，
   L1 双缓冲——无拆包、无 vector pipe、无掩码（边界由 tiling 精确长度处理）。
   triton `dot_scaled` 强制 `[BN, BK//32]` 逐字节寄存器 tile 是与硬件能力
   错配的前端格式约束；
4. **现源码中 `BLOCK_K // 32` 的剩余出现**：tile 形状（dot_scaled API 要求
   `[rows, BK//32]`）与步进计数（每 BK 个 K 元素 = BK/32 个 MX 组），列
   stride=1 行内连续——**离散访实根源 `kb * N`（stride）已删**，msprof 实证
   scale 通路 vec 6195→32µs。输出侧 `BN//32, 32` / `g_start//32` 是 MX
   32 元素组的格式语义，不可移除。

## 7. 现产数据（N-major API + K 对齐分支）

sweep（`gmm_swiglu_quant_perf_v8.json`；对照 v7 = 分支前 N-major）：

| Case | v7（全 mask） | **v8（K 对齐分支）** |
|---|---:|---:|
| `1-6144-12288-0.0` | 305.6 | **339.1** |
| `512-6144-12288-0.0` | 1199.4 | **632.6** |
| `4096-6144-12288-0.0` | 9115.8 | **3685.8** |
| `1-4096-12288-10.0` | 222.2 | **226.1** |
| `512-4096-12288-10.0` | 894.4 | **402.8** |
| `4096-4096-12288-10.0` | 6438.5 | **2237.4** |
| `64-6144-2048-0.0` | 137.5 | **123.1** |
| `512-4096-2048-10.0` | 249.2 | **198.9** |

（decode 档本就不受 mask 主导，±10% 为噪声；中大档 1.9-2.9×。）

vs vllm 冻结基线（冷 L2 event 中位，µs）：

| Case | 现产 | vllm | vllm/现产 |
|---|---:|---:|---:|
| `1-6144-12288-0.0`（decode） | 286.8 | 2380.3 | **8.30×** |
| `512-6144-12288-0.0` | 637.6 | 2790.1 | **4.38×** |
| `4096-6144-12288-0.0` | 3634.2 | 7388.3 | **2.03×** |
| `1-4096-12288-10.0`（decode） | 225.3 | 1677.3 | **7.45×** |
| `512-4096-12288-10.0` | 403.8 | 1965.7 | **4.87×** |
| `4096-4096-12288-10.0` | 2263.5 | 5945.4 | **2.63×** |
| `64-6144-2048-0.0` | 115.3 | 463.0 | **4.02×** |
| `512-4096-2048-10.0` | 161.4 | 364.6 | **2.26×** |

vs vendor 融合算子（vendor 吃 pair，脚本一次性重组不计入计时）：

| Case | 现产 | vendor | 现产/vendor |
|---|---:|---:|---:|
| `1-6144-12288-0.0`（decode） | 293.5 | 106.2 | 2.76× |
| `512-6144-12288-0.0` | 583.3 | 289.7 | 2.01× |
| `4096-6144-12288-0.0` | 3571.6 | 1765.2 | 2.02× |
| `1-4096-12288-10.0`（decode） | 212.9 | 76.0 | 2.80× |
| `512-4096-12288-10.0` | 372.3 | 201.9 | 1.84× |
| `4096-4096-12288-10.0` | 2323.0 | 1185.6 | 1.96× |
| `64-6144-2048-0.0` | 96.6 | 50.5 | 1.91× |
| `512-4096-2048-10.0` | 123.4 | 73.5 | 1.68× |

- **8/8 档全面领先 vllm**（此前仅 6/8，全批两档 0.82×/0.92× → 2.03×/2.63×）；
- vs vendor 收敛至 1.68-2.80×；剩余差距 = G4 MTE2 碎段（现 3472µs 打满）
  + epilogue 692µs，均为平台项；
- 交叉验证"原生 MX ≈ 2× bf16"：vendor 对旧 bf16 mm1 kernel 恰 1.9-2.0×。

