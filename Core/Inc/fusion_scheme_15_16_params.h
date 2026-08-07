/*
 * fusion_scheme_15_16_params.h
 * 方案 15（NN 主导场景门控增量锁定）与 方案 16（KF 主导场景门控增量锁定）
 * 自动调参结果。由 altimeter_tuner/compare_s16.py 生成，请勿手工修改。
 * 复现：python compare_s16.py --fus 60 --glob 50 --sens 12
 * 单位：气压相关阈值 m/样本；KF Q/R（Pa²）；EMA 为无量纲平滑系数。
 */
#ifndef FUSION_SCHEME_15_16_PARAMS_H
#define FUSION_SCHEME_15_16_PARAMS_H

/* ===== 方案 15 融合参数（Schmitt 门控 + 逆方差置信加权 + 门控积分） ===== */
#define S15_GATE_OPEN        0.702661f   /* NN 场景 p_elevation 开门阈值 */
#define S15_GATE_CLOSE       0.552226f   /* NN 场景 p_elevation 关门阈值（迟滞） */
/* 方案B 手动覆写（相对高度精度优化）：S15_LOCK_INTEG 0.79944→1.0 消除 ~20% 系统性缩放误差 */
#define S15_LOCK_INTEG       1.0f      /* 升降段门控积分增益（原 0.79944，放大导致相对高度偏小） */
#define S15_HOLD_ANCHOR      0.000342523f  /* 静止段锚定增益（≈0 即纯锁定） */
#define S15_DELTA_LP_ALPHA   0.26097f  /* 静止 Δh 低通系数 */
/* 方案B 手动覆写：S15_MOTION_LP_ALPHA 0.180758→0.45 运动段更快跟踪、少衰减 */
#define S15_MOTION_LP_ALPHA  0.45f     /* 升降 Δh 低通系数（原 0.180758，低通过狠拖累积分） */
/* 方案B 手动覆写：S15_DELTA_CONF_EPS 0.99131→0.2 逆方差加权更有效，信任低噪声传感器 */
#define S15_DELTA_CONF_EPS   0.2f      /* 逆方差置信正则项（原 0.99131，过大近似等权） */
/* 方案15 优化（场景判定改用原始气压窗口方差 STD，单位 Pa）：
 *   静止 STD≈传感器噪声(~0.35Pa)；运动 STD 达数~数十 Pa，与速度/幅度无关。
 *   OPEN> CLOSE 为正向迟滞。
 *   以下值为 sim_scheme15.py 用 serial_tool/data/raw 真实数据(3 场景 6 数据集)
 *   调优结果：默认 OPEN=3.0/CLOSE=1.5 太钝，升降运动时门控几乎不触发(F1≈0)，
 *   导致相对高度冻结在起点、测不准(~0.49m 去偏移 RMSE)；调优后去偏移 RMSE 0.135m、
 *   含偏移 0.19m、升降段 0.10m。 */
#define S15_MOT_STD_OPEN   1.19f  /* 长窗口气压 STD 开门阈值 (Pa) — 调优自 3.0 */
#define S15_MOT_STD_CLOSE  0.84f  /* 长窗口气压 STD 关门阈值 (Pa，迟滞) — 调优自 1.5 */
/* 方案15 增强（sim_scheme15_v2.py 调优）：新增短窗口 STD 门控，捕捉「25cm 小幅度 + 快速」变化。
 *   25cm 仅约 3Pa 气压偏移，被长窗口(20帧)平均稀释后 STD 低于开门阈值→原方案漏检；
 *   短窗口(5帧)横跨跳变、STD 陡升~1.2Pa→触发；短窗口对真实传感器低频漂移免疫。
 *   最终门控 = 长窗口STD门控 OR 短窗口STD门控（各自独立 Schmitt 迟滞）。
 *   仅在「连续静止≥S15_REANCHOR_MIN 帧后转运动」才重锚，避免快速运动抖动(flicker)反复重锚
 *   导致高度重复计算/卡在半路。 */
#define S15_FAST_WIN        5      /* 短窗口帧数（快速小幅检测） */
#define S15_FAST_STD_OPEN   0.50f  /* 短窗口气压 STD 开门阈值 (Pa) */
#define S15_FAST_STD_CLOSE  0.35f  /* 短窗口气压 STD 关门阈值 (Pa，迟滞) */
#define S15_REANCHOR_MIN    4      /* 连续静止至少 N 帧后转运动才重锚（防抖动重锚） */

/* ===== 方案 16 融合参数（KF 衍生场景：|Δh| EMA + Schmitt 门控） ===== */
#define S16_GATE_OPEN_KF     0.0117748f   /* KF Δh 幅度开门阈值 */
#define S16_GATE_CLOSE_KF    0.0150701f   /* KF Δh 幅度关门阈值（迟滞） */
#define S16_SCENE_LP_ALPHA   0.309619f  /* |Δh| EMA 系数 */
#define S16_SCENE_DELTA_ALPHA 0.119765f /* 与门控解耦的 Δh 低通 α */
#define S16_LOCK_INTEG       1.1909f   /* 升降段门控积分增益 */
#define S16_HOLD_ANCHOR      0.0198151f  /* 静止段锚定增益 */
#define S16_DELTA_LP_ALPHA   0.149704f  /* 静止 Δh 低通系数 */
#define S16_MOTION_LP_ALPHA  0.552181f  /* 升降 Δh 低通系数 */
#define S16_DELTA_CONF_EPS   0.796067f  /* 逆方差置信正则项 */

/* ===== 全局 KF（方案15/16 共用，自动调参最优） ===== */
#define S15S16_MS5611_KF_Q   0.318144f
#define S15S16_MS5611_KF_R   9.53199f
#define S15S16_BMP280_KF_Q   0.0524646f
#define S15S16_BMP280_KF_R   0.450249f

/* ===== EMA 显示平滑（方案15/16 使用，与调参一致） ===== */
#define S15S16_PRESSURE_EMA_ALPHA  0.246802f
#define S15S16_HEIGHT_EMA_ALPHA    0.710595f

#endif /* FUSION_SCHEME_15_16_PARAMS_H */
