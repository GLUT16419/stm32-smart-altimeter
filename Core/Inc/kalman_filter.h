#ifndef _KALMAN_FILTER_H_
#define _KALMAN_FILTER_H_

#include "stm32f4xx_hal.h"

/* 自适应卡尔曼滤波参数（MS5611 适用） */
#define KF_Q_MIN         0.01f
#define KF_Q_MAX         2.0f    /* 运动时 Q 上限 1.49 已够，取 2.0 */
#define KF_RESIDUAL_TH   5.0f    /* 残差阈值：MS5611 静止噪声 ±3Pa，设为 5Pa 避免误触发 */
#define KF_Q_INCREASE    1.05f  /* 残差大时 Q 放大系数 */
#define KF_Q_DECREASE    0.98f  /* 残差小时 Q 衰减系数，快速回归平滑 */

/* BMP280 专用自适应参数（噪声远小于 MS5611，需单独配置） */
#define BMP280_RESIDUAL_TH  1.2f   /* BMP280 残差阈值：静止噪声 ±0.35Pa，设为 1.2Pa */
#define BMP280_Q_INCREASE   1.08f  /* BMP280 Q 放大系数：放大更快，弥补小 R 的慢跟踪 */
#define BMP280_Q_DECREASE   0.97f  /* BMP280 Q 衰减系数：衰减略快，快速回归平滑 */
#define BMP280_Q_MAX        5.0f   /* BMP280 Q 上限：运动时允许更大 Q 加速跟踪 */

/* 传感器卡尔曼滤波推荐参数（从实际数据标定获得） */
/* 注意：启用增强校准(CALIB_ENABLE)后，以下参数将被自动调参结果覆盖 */
/* R = 测量噪声方差 = noise_std²: MS5611 ~9.3, BMP280 ~0.12 */
/* Q_base 取静止场景噪声小值，加强平滑 */
#define MS5611_KF_Q      0.03f   /* 静止 Q 取小值，加强平滑 */
#define MS5611_KF_R      9.3f    /* MS5611 静止噪声 std≈3.05Pa, 方差≈9.3 */
#define BMP280_KF_Q      0.005f  /* BMP280: Q 从 0.003 调整为 0.005，增强运动跟踪能力 */
#define BMP280_KF_R      1.0f    /* BMP280: R 从 0.10 增大到 1.0，大幅增强静止平滑，卡尔曼增益从 0.18→0.07 */

/* 初始估计误差协方差：设为较大值让滤波器快速收敛 */
#define KF_INIT_P        1000.0f

typedef struct {
    float x;         /* 状态估计值 */
    float p;         /* 估计误差协方差 */
    float q;         /* 过程噪声协方差 */
    float r;         /* 测量噪声协方差 */
    float k;         /* 卡尔曼增益 */
    float q_base;    /* 基础 Q 值，自适应恢复的基准 */
    float residual_window[5];  /* 滑动窗口残差（最近5帧），用于 STD 运动检测 */
    int   residual_idx;        /* 滑动窗口写入位置 */
} KalmanFilter_TypeDef;

void KalmanFilter_Init(KalmanFilter_TypeDef *kf, float init_x, float init_p, float q, float r);
float KalmanFilter_Update(KalmanFilter_TypeDef *kf, float z);
float KalmanFilter_Update_Adaptive(KalmanFilter_TypeDef *kf, float z);

/* BMP280 专用自适应更新：使用独立的自适应阈值和 Q 变化速率 */
float KalmanFilter_Update_Adaptive_BMP280(KalmanFilter_TypeDef *kf, float z);

#endif
