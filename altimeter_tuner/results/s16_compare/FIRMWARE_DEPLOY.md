# 方案 15 / 16 固件部署说明

> 自动调参（仿真同一算法 `altimeter_tuner/algorithm.py` 的 S15/S16 分支）得到的最优参数，
> 已写入单片机固件并可直接编译使用。

## 1. 改动文件
| 文件 | 改动 |
|---|---|
| `Core/Inc/fusion_scheme_15_16_params.h` | **新增**：S15/S16 全部调参宏（由 `compare_s16.py` 自动生成，勿手工改） |
| `Core/Inc/main.h` | `#include` 上述参数头文件 |
| `Core/Src/main.c` | ① `FUSION_SCHEME` 注释补充 15/16；② 新增 15/16 融合状态变量；③ 融合分支 `elif FUSION_SCHEME==15/16`；④ 高度改为直接积分（`h15_lock`/`h16_lock`）；⑤ EMA 平滑按方案切换为调参值；⑥ 校准阶段用调参最优 KF 参数覆盖标定经验值；⑦ 方案16 跳过 NN 联合模型推理（纯 KF）；⑧ `STATUS`/`SCENE` 输出按方案区分 |

## 2. 如何切换方案
在 `Core/Src/main.c` 修改（编译期选择，单固件只含一个方案）：
```c
#define FUSION_SCHEME 15   // NN 主导场景门控增量锁定
// 或
#define FUSION_SCHEME 16   // KF 主导场景门控增量锁定（纯 KF，无 NN）
```

## 3. 使用的调参参数（已写入 `fusion_scheme_15_16_params.h`）
**共享全局 KF/EMA（自动调参最优，两方案共用，校准阶段覆盖）**
- `MS5611_KF_Q=0.3181, MS5611_KF_R=9.532, BMP280_KF_Q=0.0525, BMP280_KF_R=0.4502`
- `PRESSURE_EMA_ALPHA=0.2468, HEIGHT_EMA_ALPHA=0.7106`

**方案 15（NN 主导）**
- `gate_open=0.7027, gate_close=0.5522, lock_integ=0.7994, hold_anchor=0.000343`
- `delta_lp_alpha=0.2610, motion_lp_alpha=0.1808, delta_conf_eps=0.9913`

**方案 16（KF 主导）**
- `gate_open_kf=0.01177, gate_close_kf=0.01507, scene16_lp_alpha=0.3096, scene16_delta_alpha=0.1198`
- `lock_integ=1.1909, hold_anchor=0.01982, delta_lp_alpha=0.1497, motion_lp_alpha=0.5522, delta_conf_eps=0.7961`

> 完整记录见 `tuned_params_record.json`。复现调参：`cd altimeter_tuner && python compare_s16.py --fus 60 --glob 50 --sens 12`

## 4. 设计要点与注意事项
- **方案 15 = NN 主导**：融合入口用联合多任务模型 NN 降噪高度，`g_mt_scene[1]`（p_elevation）做 Schmitt 门控；需要 NN 模型（X-CUBE-AI）。
- **方案 16 = KF 主导**：融合入口用自适应卡尔曼滤波高度 `height_filtered_m`，场景由 KF 自身 Δh 幅度（`|Δh|` EMA + Schmitt）判定；**不运行 NN 推理**（省算力），是低端 MCU 纯 KF 部署版本。
- 两条方案结构同构（逆方差置信加权 Δh + 门控积分：静止锁死、升降积分），与 Python 仿真一一对应，固件即该算法的 C 移植。
- **已知 quirks（忠于调参结果）**：方案 16 调出的 `gate_close_kf(0.01507) > gate_open_kf(0.01177)`，即 Schmitt 迟滞为“反向”，属于随机搜索得到的局部参数；仿真已用相同值验证（S16 调参后 RMSE=0.2159），固件与之完全一致即可复现。若希望更“标准”的迟滞（开>关），可手动把两者调换，但会偏离已记录的调参结果。
- 高度输出对 15/16 为**直接积分高度**（非气压→高度公式），与方案 7 同机制；EMA 平滑用于显示去抖。

## 5. 编译
本机无 ARM 工具链，请在 **Keil uVision（MDK-ARM/QMSX 工程）** 或 **STM32CubeIDE** 中打开工程，
将 `FUSION_SCHEME` 改为 15 或 16 后重新编译下载。
