/*
 * fusion_scheme_tuned_params.h
 * 方案 1-14 自动调参结果（两级：共享 KF/EMA 全局搜索 + 各方案融合参数精修）。
 * 由 altimeter_tuner/tune_all_params.py 生成，请勿手工修改。
 * 复现：cd altimeter_tuner && python tune_all_params.py
 * 用法：本头在 main.h 中「FUSION_SCHEME 定义之后」被包含；
 *       main.h 的融合宏优先取 TUNED_*，未定义时回退各方案默认值。
 */
#ifndef FUSION_SCHEME_TUNED_PARAMS_H
#define FUSION_SCHEME_TUNED_PARAMS_H

/* ===== 共享全局 KF / EMA（方案1-14 通用，两级调参最优） ===== */
#define TUNED_MS5611_KF_Q        0.25901f
#define TUNED_MS5611_KF_R        23.8569f
#define TUNED_BMP280_KF_Q        0.0879115f
#define TUNED_BMP280_KF_R        2.82916f
#define TUNED_PRESSURE_EMA_ALPHA 0.607815f
#define TUNED_HEIGHT_EMA_ALPHA   0.946719f

/* ===== 各方案融合参数（仅当前选中的 FUSION_SCHEME 生效） ===== */
#if FUSION_SCHEME == 1
#define TUNED_FUSION_WEIGHT_MS5611  0.0296786f
#define TUNED_FUSION_WEIGHT_BMP280  0.970321f
#elif FUSION_SCHEME == 3
#define TUNED_FUSION_WEIGHT_MS5611  0.767441f
#define TUNED_FUSION_WEIGHT_BMP280  0.232559f
#elif FUSION_SCHEME == 4
#define TUNED_HPF_ALPHA            0.0434126f
#elif FUSION_SCHEME == 5
#define TUNED_MOTION_THRESHOLD_PA  0.587598f
#define TUNED_WEIGHT_STATIC_MS     0.165901f
#define TUNED_WEIGHT_STATIC_BMP    0.834099f
#define TUNED_WEIGHT_MOTION_MS     0.605535f
#define TUNED_WEIGHT_MOTION_BMP    0.394465f
#define TUNED_WEIGHT_SMOOTH_ALPHA  0.158084f
#elif FUSION_SCHEME == 7
#define TUNED_W_DELTA_MS_STATIC        0.194107f
#define TUNED_W_DELTA_BMP_STATIC       0.805893f
#define TUNED_W_DELTA_MS_MOTION        0.527679f
#define TUNED_W_DELTA_BMP_MOTION       0.472321f
#define TUNED_DELTA_WEIGHT_SMOOTH_ALPHA 0.289711f
#elif FUSION_SCHEME == 10
#define TUNED_IVAR_EPSILON         2.96291f
#elif FUSION_SCHEME == 11
#define TUNED_DELTA_CONF_EPS       0.563543f
#define TUNED_ANCHOR_ALPHA         0.00517625f
#elif FUSION_SCHEME == 12
#define TUNED_FUSION_WEIGHT_MS5611  0.68536f
#define TUNED_FUSION_WEIGHT_BMP280  0.31464f
#elif FUSION_SCHEME == 13
#define TUNED_COMP_ALPHA           0.0117728f
#define TUNED_COMP_BETA            0.709907f
#elif FUSION_SCHEME == 14
#define TUNED_HPF_ALPHA            0.036028f
#define TUNED_TC_COEFF             2.34345f
#endif

#endif /* FUSION_SCHEME_TUNED_PARAMS_H */
