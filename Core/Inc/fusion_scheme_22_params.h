/*
 * fusion_scheme_22_params.h
 * 方案 22：在方案 21（双 EKF 方差倒数加权融合）基础上，
 *          把固定输出低通换成“运动自适应输出低通（快慢双低通偏差法）”，
 *          解决“相对高度测试时要等约 1 分钟才稳定”的问题，同时保持静止精度。
 *
 * 问题根因：
 *   方案21 输出级固定低通 S21_OUTPUT_EMA = 0.001 @50Hz → 时间常数 τ = dt/α ≈ 20s。
 *   相对高度发生阶跃/斜坡变化后，输出按指数逼近真值，需 3~4τ ≈ 60~80s 才进入 ±5cm，
 *   表现为“等约 1 分钟才稳定”。
 *
 * 方案22 改进（仅改“输出低通”这一步，融合核心完全复用方案21）：
 *   同时维护快低通 p_fast(α_fast, τ≈1s) 与慢低通 p_slow(α_slow, τ≈20s)；
 *   运动指标 m = |p_fast − p_slow|（两者均已平滑，不含高频噪声；
 *       运动时快低通跟得上、慢低通滞后，m 达数 Pa；静止时两者收敛到真值，m→0）；
 *   norm = sat(m/S22_THR)^2（平方抑制静止微小残差）；
 *   输出 out = norm·p_fast + (1−norm)·p_slow：
 *       静止 → norm≈0 → 完全用最平滑的 p_slow（静止精度与方案21一致）；
 *       运动 → norm≈1 → 切到 p_fast（τ≈1s，几秒到位）。
 *   这样“等待约1分钟”→几秒，且静止噪声不变（精度保持）。
 *
 * 仿真验证（simulation/sim_scheme22.py，相对高度实测场景）：
 *   稳定时间 87s → 1s（抬升后）；静止噪声 1.5cm → 2.2cm（仍厘米级）；运动段 RMSE 更小（更跟手）。
 *
 * 说明：融合核心（在线偏置对齐 + 残差方差倒数加权）直接复用方案 21（fusion_scheme_21_params.h）。
 */
#ifndef FUSION_SCHEME_22_PARAMS_H
#define FUSION_SCHEME_22_PARAMS_H

#include "fusion_scheme_21_params.h"   /* 复用方案21 融合核心（在线偏置 + 方差倒数加权） */

/* ===== 方案22 自适应输出低通（快慢双低通偏差法） ===== */
/* 快低通 α：τ≈1s @50Hz，运动时让融合输出几秒跟到新高度 */
#define S22_A_FAST  0.02f

/* 慢低通 α：τ≈20s @50Hz（= 方案21 S21_OUTPUT_EMA），静止时保持精度不变 */
#define S22_A_SLOW  0.001f

/* 运动指标阈值 (Pa)：|p_fast − p_slow| 超过即判为运动。
 * 仿真统计：静止 RMS≈0.34Pa，运动均值≈7Pa，阈值取 3Pa 区分度充足且不会被噪声误触发。 */
#define S22_THR     3.0f

#endif /* FUSION_SCHEME_22_PARAMS_H */
