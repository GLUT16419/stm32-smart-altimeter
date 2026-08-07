#include "kalman_filter.h"
#include <math.h>

void KalmanFilter_Init(KalmanFilter_TypeDef *kf, float init_x, float init_p, float q, float r)
{
    kf->x = init_x;
    kf->p = init_p;
    kf->q = q;
    kf->q_base = q;
    kf->r = r;
    kf->k = 0;
    kf->residual_idx = 0;
    for(int i = 0; i < 5; i++)
        kf->residual_window[i] = 0.0f;
}

float KalmanFilter_Update(KalmanFilter_TypeDef *kf, float z)
{
    kf->p = kf->p + kf->q;
    
    kf->k = kf->p / (kf->p + kf->r);
    
    kf->x = kf->x + kf->k * (z - kf->x);
    
    kf->p = (1 - kf->k) * kf->p;
    
    return kf->x;
}

/**
  * @brief  自适应卡尔曼滤波更新
  * @note   根据残差动态调整 Q 值：
  *         残差大（传感器快速变化）→ Q 增大 → 快速跟踪
  *         残差小（传感器平稳）   → Q 减小 → 平滑更多
  *         Q 在 [Q_MIN, Q_MAX] 范围内自适应
  * @param  kf: 卡尔曼滤波器结构体指针
  * @param  z: 当前测量值
  * @retval 滤波后的估计值
  */
float KalmanFilter_Update_Adaptive(KalmanFilter_TypeDef *kf, float z)
{
    float residual = z - kf->x;
    float residual_abs = fabsf(residual);

    /* ---- 更新 5 帧滑动窗口残差（用于 STD 运动检测） ---- */
    kf->residual_window[kf->residual_idx] = residual_abs;
    kf->residual_idx = (kf->residual_idx + 1) % 5;

    /* 计算窗口残差标准差：比单帧残差更稳健，可抑制偶发跳变误触发 */
    float mean = 0.0f;
    for(int i = 0; i < 5; i++)
        mean += kf->residual_window[i];
    mean /= 5.0f;
    float var = 0.0f;
    for(int i = 0; i < 5; i++) {
        float d = kf->residual_window[i] - mean;
        var += d * d;
    }
    float residual_std = sqrtf(var / 5.0f);

    /* ---- 自适应调整 Q（基于残差 STD） ---- */
    if(residual_std > KF_RESIDUAL_TH)
    {
        /* 残差 STD 大 → 快速跟踪，增大 Q */
        kf->q = kf->q * KF_Q_INCREASE;
        if(kf->q > KF_Q_MAX)
            kf->q = KF_Q_MAX;
    }
    else
    {
        /* 残差 STD 小 → 缓慢衰减到基础 Q，加强平滑 */
        kf->q = kf->q * KF_Q_DECREASE;
        if(kf->q < kf->q_base)
            kf->q = kf->q_base;
    }

    /* 标准卡尔曼更新 */
    kf->p = kf->p + kf->q;
    
    kf->k = kf->p / (kf->p + kf->r);
    
    kf->x = kf->x + kf->k * residual;
    
    kf->p = (1 - kf->k) * kf->p;
    
    return kf->x;
}

/**
  * @brief  自适应卡尔曼滤波更新（BMP280 专用版）
  * @note   使用 BMP280 专用的自适应阈值和 Q 变化速率。
  *         BMP280 噪声远小于 MS5611（std=0.35Pa vs 3.05Pa），
  *         需要用更小的残差阈值、更快的 Q 放大系数来保证运动跟踪能力。
  * @param  kf: 卡尔曼滤波器结构体指针
  * @param  z: 当前测量值
  * @retval 滤波后的估计值
  */
float KalmanFilter_Update_Adaptive_BMP280(KalmanFilter_TypeDef *kf, float z)
{
    float residual = z - kf->x;
    float residual_abs = fabsf(residual);

    /* ---- 更新 5 帧滑动窗口残差（用于 STD 运动检测） ---- */
    kf->residual_window[kf->residual_idx] = residual_abs;
    kf->residual_idx = (kf->residual_idx + 1) % 5;

    /* 计算窗口残差标准差 */
    float mean = 0.0f;
    for(int i = 0; i < 5; i++)
        mean += kf->residual_window[i];
    mean /= 5.0f;
    float var = 0.0f;
    for(int i = 0; i < 5; i++) {
        float d = kf->residual_window[i] - mean;
        var += d * d;
    }
    float residual_std = sqrtf(var / 5.0f);

    /* ---- 自适应调整 Q（基于残差 STD，BMP280 专用参数） ---- */
    if(residual_std > BMP280_RESIDUAL_TH)
    {
        /* 残差 STD 大：快速跟踪，增大 Q */
        kf->q = kf->q * BMP280_Q_INCREASE;
        if(kf->q > BMP280_Q_MAX)
            kf->q = BMP280_Q_MAX;
    }
    else
    {
        /* 残差 STD 小：缓慢衰减到基础 Q，加强平滑 */
        kf->q = kf->q * BMP280_Q_DECREASE;
        if(kf->q < kf->q_base)
            kf->q = kf->q_base;
    }

    /* 标准卡尔曼更新 */
    kf->p = kf->p + kf->q;
    
    kf->k = kf->p / (kf->p + kf->r);
    
    kf->x = kf->x + kf->k * residual;
    
    kf->p = (1 - kf->k) * kf->p;
    
    return kf->x;
}
