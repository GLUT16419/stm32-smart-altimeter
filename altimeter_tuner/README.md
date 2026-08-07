# 高度融合算法调参仿真器 (altimeter_tuner)

在 PC 端**忠实复现嵌入式固件**（`Core/Src/main.c` + `kalman_filter.c`）的气压高度融合算法，
用于不刷机即可对单片机算法进行**可视化调参**。

## 特点

- **算法一致**：复刻固件的 KF / 自适应 KF / 14 种融合方案（含 HPF、自适应权重、Delta 累积、
  二阶互补、逆方差）/ 校准（参考气压 + BMP280 偏置）/ 温漂补偿 / EMA 平滑。
- **三类数据集**：
  - `static` 静止（验证平滑与噪声）
  - `translation` 平移（验证不引入虚假高度）
  - `elevation` 升降（验证跟踪、回零、过冲）
  - 也支持**加载真实采集 CSV**（合并单文件 或 MS5611/BMP280 双文件）。
- **GUI 调参**：所有 KF 参数、融合权重、阈值、EMA 系数均可滑动/输入实时调整。
- **对比模式**：勾选「对比(保留上次)」可把上一次融合曲线叠加，直观比较参数差异。
- **指标**：RMSE、静态噪声、漂移率、跟踪延迟、回零误差、过冲。
- **导出**：一键导出参数片段（可贴回 `kalman_filter.h` / `main.c` 宏），或导出对比图。

## 安装

```bash
cd altimeter_tuner
pip install -r requirements.txt
```

> 可选：安装 `tflite_runtime` 后，在 `AlgoParams.use_nn=True` 并指定模型路径时启用真实 NN 滤波；
> 未安装时 NN 自动退化为 KF（等价于固件 `WORK_MODE!=0` 路径）。

## 运行

```bash
python app.py
```

操作：
1. 顶部「数据集」选择 `static` / `translation` / `elevation`，或「从文件」加载真实 CSV。
2. 左侧分组调整参数（校准 / MS5611 KF / BMP280 KF / 融合 / 平滑）。
3. 点「运行 ▶」刷新曲线与指标。
4. 勾「对比(保留上次)」再点运行，可叠加上一次结果。
5. 「导出参数」生成 C 头宏片段；「生成示例CSV」写出三类合成数据集。

## 文件说明

| 文件 | 作用 |
|------|------|
| `algorithm.py` | 固件算法 Python 移植（KF、融合、校准、EMA、NN 退化） |
| `datasets.py`  | 合成数据集生成 + CSV 加载（合并/双文件） |
| `metrics.py`   | RMSE / 噪声 / 漂移 / 延迟 / 过冲 等指标 |
| `app.py`       | Tkinter GUI + matplotlib 绘图 |

## 与固件的一致性说明

- 气压→高度使用 `PressureToAltitudeWithTemp`（ISA + 温度补偿），与 `altitude_convert.c` 一致。
- 自适应 KF 的 5 帧残差滑动窗口 STD 运动检测、Q 增大/衰减系数、Q 上下限，完全对齐 `kalman_filter.c`。
- 融合方案编号与 `main.c` 的 `FUSION_SCHEME` 宏一一对应；默认 14（BMP280 主导 + MS5611 高频增强 + 温漂补偿）。
- 高度统一用 BMP280 温度 + 同一参考气压重算，与固件「所有高度统一基准」一致。
- 已知差异：固件为 RTOS 多任务按 `SAMPLE_PERIOD_MS=100ms` 调度；本工具按数据集采样周期顺序处理，
  数值等价（单帧串行等价于逐帧推进）。

## 典型调参建议

- 静止噪声大 → 调大 `BMP280 KF` 的 `R` 或减小 `Q_base`（更平滑）。
- 升降跟踪慢/滞后 → 调大运动阈值或提高运动权重（方案 5/7 的 `运动权重MS`）。
- 回零漂移 → 启用方案 11/13 的锚定，或检查 `BMP280偏置` 与 `参考气压` 校准。
- 平移时出现虚假高度 → 检查两传感器一致性（偏置 `bmp_bias`）与融合方案。
