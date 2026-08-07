# QMSX — 基于 STM32 的智能气压高度计（多传感器融合）

QMSX 是一套以 **STM32F4（STM32F411xE）** 为核心的智能气压高度计系统，使用 **BMP280 + MS5611** 双气压传感器，结合卡尔曼滤波 / 扩展卡尔曼滤波（EKF）与多种融合算法，实现高精度、低漂移的相对/绝对高度测量，并通过 OLED 实时显示。系统基于 **FreeRTOS** 进行多任务调度，并可选用 **X-CUBE-AI** 的神经网络滤波路径。



## 特性

- **双气压传感器融合**：BMP280 + MS5611，提供 22 种融合方案（含 HPF、自适应权重、Delta 累积、二阶互补、逆方差、EKF+BP 多页对照等）。
- **推荐方案 21（对照 + 融合）**：对两路传感器各并行跑 EKF 与 BP 神经网络去噪，OLED 翻页对照 4 路结果，并额外给出"逆方差加权的 FUSED"融合页（详见「方案设计 → 方案 21」）。
- **自适应卡尔曼滤波**：基于残差滑动窗口 STD 的运动检测，动态调节过程噪声 Q。
- **温漂补偿 / EMA 平滑 / 校准**：参考气压 + BMP280 偏置校准。
- **实时显示**：SSD1306 OLED（I2C）。
- **RTOS 调度**：FreeRTOS，按 `SAMPLE_PERIOD_MS=100ms` 周期采样。
- **可选 NN 滤波**：通过 X-CUBE-AI 部署轻量神经网络（运行时可选，未启用时退化为 KF）。

## 目录结构

| 目录 / 文件 | 说明 |
|-------------|------|
| `Core/` | 固件应用代码（`Src/` + `Inc/`）：`main.c`、`kalman_filter.c`、`altitude_convert.c`、`baro_ekf.c`、`bsp_bmp280.c`、`bsp_ms5611.c`、OLED/SSD1306 驱动等 |
| `Drivers/` | STM32 HAL / CMSIS 底层驱动（ST 提供） |
| `Middlewares/` | 第三方中间件（FreeRTOS、X-CUBE-AI 运行时头文件等） |
| `X-CUBE-AI/` | X-CUBE-AI 应用层源码 |
| `MDK-ARM/` | Keil MDK 工程文件（`QMSX.uvprojx` 等；构建产物已被忽略） |
| `QMSX.ioc` | STM32CubeMX 工程，可重新生成初始化代码 |
| `simulation/` | 算法仿真（Python）：复现并验证固件融合/滤波行为 |
| `altimeter_tuner/` | PC 端 GUI 调参仿真器，忠实复现固件算法，可视化调参 |
| `serial_tool/` | 串口 / 数据采集与模型相关工具 |
| `LICENSE_X-CUBE-AI.txt` | X-CUBE-AI 组件许可 |

## 方案设计

### 1. 系统架构与任务划分

系统以 **STM32F411**（ARM Cortex-M4）为控制核心，运行 **FreeRTOS**，按 `SAMPLE_PERIOD_MS = 20ms` 周期采样。两个气压传感器分工明确：

- **BMP280**（I²C）：低噪声（std≈0.35 Pa）、低漂移，作为**绝对高度基准**，并以其温度作为融合温度。
- **MS5611**（SPI，24-bit ADC）：噪声较大（std≈3.05 Pa）但动态响应快，提供**高频变化增量**。

任务划分（`Core/Src/main.c`）：

| 任务 | 职责 |
|------|------|
| `StartMS5611Task` | 读 MS5611 → 自适应 KF 降噪；把读数推入联合模型窗口 |
| `StartBMP280Task` | 读 BMP280 → KF 降噪 + 全局偏置补偿 + 温漂补偿（方案14）→ 调用 `Fusion_Compute()` → 显示/高度 EMA 平滑 |
| `StartOLEDTask` | SSD1306（I²C）实时显示融合高度/温度/场景 |
| `StartUARTOutputTask` / `StartUARTCommandTask` | 串口输出与命令（校零、重锚、切预设海拔） |
| `StartDefaultTask` | 上电热稳定等待、参考气压 `reference_pressure_pa` 初始化 |

为保证数据一致性：所有共享数据由 `SensorDataMutex` 互斥锁保护；**统一参考气压** `reference_pressure_pa`，MS5611 高度用 BMP280 温度重算，确保两传感器高度同一基准。

### 2. 气压 → 高度转换（`altitude_convert.c`）

提供两种大气压公式，由宏 `ALTITUDE_FORMULA_ISA` 切换：

- **ISA 国际标准大气公式**：`h = (T0/L)·(1 − (P/P0)^(L·R/g))`，适用于 0~11000 m 对流层（温度随高度递减）。
- **等温模型**：`h = −(R·T/g)·ln(P/P0)`，简化但忽略温度递减。

带温度补偿版 `PressureToAltitudeWithTemp()` 用实测温度对标准温度偏差做一阶修正（每度温差约修正 3.5 cm），抑制温度带来的高度偏差。

### 3. 单传感器降噪（自适应卡尔曼滤波，`kalman_filter.c`）

标准一维卡尔曼滤波的基础上引入**运动自适应**：

- 维护 5 帧残差滑动窗口，计算其**标准差 STD**；
- 残差 STD 大（传感器快速变化）→ `Q` 按 `KF_Q_INCREASE` 放大，快速跟踪；
- 残差 STD 小（平稳）→ `Q` 按 `KF_Q_DECREASE` 衰减回 `q_base`，加强平滑；
- `Q` 限制在 `[q_base, KF_Q_MAX]` 之间，避免振荡。

BMP280 噪声更小，使用专用阈值/放大系数的 `KalmanFilter_Update_Adaptive_BMP280()`，保证运动跟踪能力。另设全局偏置 `bmp_bias_pa`，校准后把 BMP280 绝对气压对齐到 MS5611 基准。

### 4. 双传感器融合引擎 `Fusion_Compute()`（`main.c`）

全部融合方案集中在此函数，由 `main.h` 的编译期宏 `FUSION_SCHEME` 选择**唯一生效方案**（其余方案不参与编译，不占代码空间）。融合统一在**气压域**进行，再由统一参考气压换算高度。各方案核心思想：

| 方案 | 思路 |
|------|------|
| 1 / 3 | 气压域双传感器加权融合（MS5611 仅用 KF，BMP280 用 NN） |
| 2 | 四路直接加权（MS5611_KF + BMP280_NN + BMP280_KF） |
| 4 | **BMP280 主导 + MS5611 高频增强**：`P = BMP280 + HPF(MS5611_KF − BMP280)`，高通只提取 MS5611 快速变化，不污染绝对精度 |
| 5 | 自适应权重：静止时 BMP280 95%，运动时升至 MS5611 40%，权重平滑过渡 |
| 6 | MS5611 主导（85%）+ BMP280 KF（15%），**不用 NN** |
| 7 | 高度变化量加权累积 + 静止「拉力」缓慢回锚，抑制噪声累积漂移 |
| 8 / 9 | 纯单传感器（仅 MS5611 KF / 仅 BMP280 NN+KF） |
| 10 | **逆方差加权**（创新）：权重 = `1/(σ²+ε)`，MS5611 跳变时方差暴增→权重归零，完全信任 BMP280 |
| 11 | **Delta 置信度加权累积** + 泄漏锚：融合帧间气压变化量，`1/(var+ε)` 定权，静止缓慢拉回 BMP280 消除漂移 |
| 13 | **二阶互补融合**：状态含气压与变化率，MS5611 主导速度估计、BMP280 慢速锚定 |
| 14 | 方案 4 + **BMP280 实时温漂线性补偿**（按温度偏离校准值修正气压） |
| 15 / 16 | **场景门控增量锁定**（默认新架构，详见下） |
| 17 | 无场景模式：BMP280 KF 直接算高度 |
| 20 / 21 / 22 | sahixi 气压域 EKF + BP 去噪多页对照显示（方案 21 详细讲解见下文） |

#### 方案 15 / 16：场景门控增量锁定（核心创新架构）

针对静止/升降场景下传统 KF 固定增益导致约 0.4 m 系统误差的问题，设计了「**门控积分**」结构：

1. **场景判定（静止 vs 升降）**：用**原始气压**的「双窗口 STD 门控」——长窗口（20 帧）捕捉持续/慢速运动，短窗口（5 帧）捕捉快速小幅（~25 cm）运动；两者各自独立 Schmitt 迟滞，最终 `门控 = 长 OR 短`。相对高度改用**原始气压**两点 ISA 作差，绕开 KF 启动滞后、升降对称、路径无关。
2. **减滞后延时锁定**：判定静止后不立即冻结，继续积分 `S15_SETTLE_FRAMES` 帧沉降等待期，待气压完全沉降再冻结，避免锁在半路值。
3. **逆方差置信加权 Δh**：两传感器帧间 Δh 按窗口方差 `1/(var+ε)` 融合。
4. **NN 多任务联合模型**（方案 15/16 同构，仅降噪与场景信号来源不同）：
   - 单一模型双输入（各 10 点相对气压窗口）→ 三输出（MS5611 滤波 / BMP280 滤波 / 场景概率），一次推理同时完成滤波与场景识别；
   - 方案 15 用 NN 滤波 + NN 场景；方案 16 用 KF 滤波 + NN/KF 散度判场景，**不依赖 NN 数值**，更适合算力受限 MCU。

### 5. 扩展滤波 / 去噪路径

- **EKF**（`altitude_ekf.c` / `baro_ekf.c`）：
  - `AltitudeEKF`：高度–速度状态模型，以气压为观测（含气压→高度雅可比），输出高度与垂直速度；
  - `BaroEKF`：气压–变化率状态模型，以 ISA 高度残差做观测更新（雅可比替代常数 H），方案 20/21/22 用于气压域滤波。
- **BP 去噪 MLP**（`bp_denoise.c`）：5 点滑动窗 → 双层 MLP（`5→8(tanh)→1`）去噪，权重由 X-CUBE-AI 反解自参考项目 sahixi 的 `_real` 模型；输出再经 EMA 后处理消除高频抖动。方案 20/21/22 将其与 EKF 对照显示（单 EKF / EKF+BP、MS5611 / BMP280、双 EKF 方差倒数加权融合）。

#### 方案 21：sahixi 气压域 EKF + BP 去噪多页对照 + FUSED 融合（详细）

方案 21 属于 `FUSION_SCHEME == 20/21/22` 这一家子，思路源自参考项目 **sahixi** 的"气压域 EKF + BP 去噪"。与方案 1~17 最大的不同：**它不急着只给一个高度，而是先把各滤波方法的"成绩"摆出来对照，再把最好的合在一起。** OLED 上因此能逐页翻看 5 个结果，这也是它被称为"B1 多页对照"的原因。

**整体数据流（每采样周期，`main.c` 中 `FUSION_SCHEME == 21` 分支）：**

```
对 MS5611 / BMP280 各跑两路降噪：
   ├─ BaroEKF_Update()   → 气压域 EKF 滤波值  (EKF 路)
   └─ BP_Denoise_Update()→ 5 点滑窗 MLP 去噪值 (BP 路)
        ↓ 各自用 PressureToAltitudeWithTemp() 换算高度
   ├─ MS5611-EKF / BMP280-EKF / MS5611-BP / BMP280-BP  ← 4 个对照页
        ↓ 两路 EKF 再经"对齐基准 + 逆方差加权 + 低通"合成
   └─ FUSED 融合页（方案 21 额外第 5 页）
```

**1. 两个传感器 × 两路滤波 = 4 个"选手"**

| 选手 | EKF 路（扩展卡尔曼滤波） | BP 路（小神经网络去噪） |
|------|------|------|
| BMP280 | `BMP280-EKF` | `BMP280-BP` |
| MS5611 | `MS5611-EKF` | `MS5611-BP` |

- **EKF 路**（`baro_ekf.c` 的 `BaroEKF`）：在**气压域**直接对 Pa 建模，状态含"气压 + 气压变化率"，以 ISA 高度残差做观测更新。它既平滑抖动、又跟踪真实升降（"会预判的平滑器"）。
- **BP 路**（`bp_denoise.c`）：提前训练好的迷你神经网络（5 点滑动窗 → `5→8(tanh)→1` MLP），权重由 X-CUBE-AI 反解自 sahixi 的 `_real` 模型；输出再经 EMA 抹平高频抖动（"跟过大量数据、知道噪声长啥样"的去噪笔）。

**2. 第 5 页 FUSED：把两个 EKF 聪明地合二为一**

FUSED 页把 `MS5611-EKF` 与 `BMP280-EKF` 的结果按"谁更靠谱就多听谁的"融合，做了三件关键处理：

- **(a) 先对齐基准（在线偏置估计）**：BMP280 与 MS5611 天生读数有固定差值（零点几到几帕），直接融合会"打架"导致输出持续漂移。方案 21 用一段很慢的 EMA 在 `S21_BIAS_CONVERGE_FRAMES` 帧内估出该差值，减掉后让两路站在同一基准，随后**冻结**不再更新（避免把噪声学进去）。
- **(b) 逆方差加权 + 权重平滑限幅**：每帧算两路 EKF 的残差方差 `var`（衡量当前噪声/异常水平），权重 `w = 1/(var+ε)` 归一化；再对权重做长 EMA（`S21_W_EMA`）并限幅到 `[S21_W_MIN, S21_W_MAX]`，消除帧间权重抖动（这是融合"波动巨大"的主因）。最终 `p_fus = w·ms_ekf + (1−w)·aligned_bm_ekf`。`S21_FUSE_METHOD==1` 时可退化为固定权重 `S21_FIXED_W_MS`。
- **(c) 输出低通（EMA）**：融合气压再过一道 `S21_OUTPUT_EMA` 指数平滑，得到最终融合气压，并用 `ms_p0 + 温度补偿` 换算高度。温度来源由 `S21_FUSE_TEMP_SRC` 选择（BMP280 温度 / 两路平均）。

**3. 显示与操作（OLED + 板载按键）**

- **短按按键**（`g_s20_page = (g_s20_page+1)%5`）：在 5 页间循环——`0=MS5611 EKF`、`1=BMP280 EKF`、`2=MS5611 BP`、`3=BMP280 BP`、`4=FUSED`。每页显示标题、温度 `T`、气压 `P (hPa)`、高度 `Alt (m)`；FUSED 页（第 4 页）额外显示**相对高度 `Rel`**。
- **长按 3 秒**：把当前融合高度设为"相对高度零点"（`rel_height_ref`），FUSED 页 `Rel` 归零。
- **启动保护**：上电头约 10 s 融合尚在"热身"（偏置收敛 + 低通到位），相对高度基准特意延迟约 5 s（25 个 OLED 循环）才锁定，避免把启动漂移固化下来。

**4. 与方案 20 / 22 的区别**

| 方案 | 对照页 | 融合页 | 输出低通 |
|------|--------|--------|----------|
| 20 | 4 页（EKF×2 + BP×2） | 无（以 `BMP280-EKF` 为代表值） | — |
| **21** | 4 页 + **FUSED**（5 页） | 逆方差加权 | **固定** EMA（`S21_OUTPUT_EMA`） |
| 22 | 同 21（5 页） | 同 21 | **自适应**：静止用最平滑慢滤波保精度，运动切快滤波跟手 |

> 注：方案 21 的完整实现位于 `Core/Src/main.c`（`FUSION_SCHEME == 21` 分支，约 2236–2424 行处理、2904–3053 行显示）、`Core/Src/baro_ekf.c`、`Core/Src/bp_denoise.c`。

### 6. 显示、输出与调参一致性

- **OLED / 串口**：实时输出融合高度、温度、场景（static/elevation）；方案 15/16 静止时冻结 EMA 显示输出，避免噪声引入缓慢漂移。
- **PC 端忠实复现**：`simulation/` 与 `altimeter_tuner/` 完整复现固件 `main.c` + `kalman_filter.c` 的融合/滤波算法，可调参（调参结果自动生成 `fusion_scheme_*_params.h`），确保板端与上位机算法一致。

## 固件构建

1. 使用 **STM32CubeMX** 打开 `QMSX.ioc` 可重新生成/修改初始化代码。
2. 使用 **Keil MDK** 打开 `MDK-ARM/QMSX.uvprojx` 编译并下载。
3. 关键宏（融合方案、采样周期等）位于 `Core/Inc/` 与 `Core/Src/kalman_filter.c`、`main.c` 中。

> 注：X-CUBE-AI 的神经网络运行时库（`.lib`）及训练好的模型权重不属于源码，未纳入本仓库；NN 路径为可选项。

## PC 端工具

### 仿真（simulation/）

```bash
cd simulation
pip install -r requirements.txt   # 若提供；否则安装 numpy/matplotlib 等
python <脚本>.py
```

### 高度融合算法调参仿真器（altimeter_tuner/）

在 PC 端忠实复现固件 `main.c` + `kalman_filter.c` 的气压高度融合算法，支持 GUI 实时调参、三类数据集（static / translation / elevation）及真实 CSV 加载。

```bash
cd altimeter_tuner
pip install -r requirements.txt
python app.py
```

### 串口工具（serial_tool/）

数据采集与模型相关辅助工具，见 `serial_tool/README.rst`。

## 许可

- 固件与自研源码按仓库默认许可（见 LICENSE，如有）。
- `Middlewares/`、`X-CUBE-AI/` 及 `Drivers/` 中来自 STMicroelectronics 的组件遵循其各自许可，详见 `LICENSE_X-CUBE-AI.txt` 及各组件内声明。
