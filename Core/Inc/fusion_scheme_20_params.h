/*
 * fusion_scheme_20_params.h
 * 方案 20（sahixi 气压域 EKF + BP 去噪）参数。
 * 直接移植自参考项目 sahixi，系数/权重沿用其经验值，请勿手工大幅改动。
 * 复现：参考项目 参考文档/sahixi/code/Src/kalman.c 与 Core/Src/freertos.c。
 */
#ifndef FUSION_SCHEME_20_PARAMS_H
#define FUSION_SCHEME_20_PARAMS_H

/* ===== 气压域 EKF 过程/测量噪声 ===== */
/* 权衡要点：
 *   R_MEAS 越大 → EKF 更平滑 → 静止噪声低 → 但对真实高度变化响应更慢。
 *   Q_PRESS 越大 → 预测更自信 → 跟手更好 → 静止噪声略升。
 *   R=0.2 噪声大但跟手好，R=1.0 平滑与跟手均衡，R=5.0 极度平滑但严重滞后。
 * 经测试：R=1.0 + Q=2.0 在静止噪声~0.04m 与 75cm 升降 1s 内到位之间取得平衡。 */
#define S20_EKF_Q_PRESS  2.0f      /* 气压过程噪声 (Pa^2)：越大越跟手 */
#define S20_EKF_Q_RATE   5.0f      /* 气压率过程噪声 */
#define S20_EKF_R_MEAS   1.0f      /* 测量噪声 (高度 m^2)：均衡值，兼顾平滑与跟手 */

/* ===== BP 去噪网络后处理 ===== */
#define S20_BP_EMA_ALPHA 0.1f      /* EMA: out = α*new + (1-α)*old (α越小越平滑) */
#define S20_BP_WIN_SCALE 5.0f      /* 滑窗归一化尺度: (x-center)/SCALE */

/* ===== 双传感器独立 P0 基准 ===== */
#define S20_BM_P0_OFFSET 100.0f    /* BMP280 相对 MS5611 的系统偏差 (~+1 hPa), Pa */

#endif /* FUSION_SCHEME_20_PARAMS_H */
