# QMSX — 基于 STM32 的智能气压高度计（多传感器融合）

QMSX 是一套以 **STM32F4（STM32F411xE）** 为核心的智能气压高度计系统，使用 **BMP280 + MS5611** 双气压传感器，结合卡尔曼滤波 / 扩展卡尔曼滤波（EKF）与多种融合算法，实现高精度、低漂移的相对/绝对高度测量，并通过 OLED 实时显示。系统基于 **FreeRTOS** 进行多任务调度，并可选用 **X-CUBE-AI** 的神经网络滤波路径。

仓库仅包含**源代码（含固件与 PC 端仿真/调参工具）与说明文档**，不含个人隐私、课程过程材料及构建产物。

## 特性

- **双气压传感器融合**：BMP280 + MS5611，提供 14 种融合方案（含 HPF、自适应权重、Delta 累积、二阶互补、逆方差等）。
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
