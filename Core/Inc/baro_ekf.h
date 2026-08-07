#ifndef _BARO_EKF_H_
#define _BARO_EKF_H_

#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * 气压域扩展卡尔曼滤波 (移植自参考项目 sahixi 的 KalmanEKF)
 *
 * 状态:   x = [气压 P (Pa), 气压变化率 dP/dt (Pa/s)]
 * 预测:   匀速模型 F = [[1, dt],[0, 1]] (状态转移线性)
 * 更新:   测量方程 h(P) = (T0/L)*(1-(P/P0)^exp) 高度公式 (非线性)
 *         雅可比 H = ∂h/∂P, 用高度残差 y = h(z)-h(x⁻) 修正气压状态
 *
 * 与现有 altitude_ekf.c (高度域 [高度, 速度]) 状态空间与测量方程根本不同，
 * 因此独立成模块，互不干扰，也便于答辩对比"气压域 vs 高度域 EKF"。
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float p;          /* 气压状态估计 (Pa)                         */
    float dp;         /* 气压变化率 (Pa/s)                          */
    float P[2][2];    /* 误差协方差 2x2 (对称)                      */
    float q_press;    /* 气压过程噪声方差 (Pa^2)                    */
    float q_rate;     /* 气压率过程噪声方差 (Pa^2/s^3)              */
    float r_meas;     /* 测量噪声方差 (高度 m^2)                    */
    float p0_pa;      /* 海平面气压基准 (含各传感器偏差, Pa)         */
    float dt;         /* 时间步长 (s)                               */
} BaroEKF_TypeDef;

void BaroEKF_Init(BaroEKF_TypeDef *k, float p0_pa, float dt,
                  float q_press, float q_rate, float r_meas);
void BaroEKF_Reset(BaroEKF_TypeDef *k, float press_pa);
void BaroEKF_SetP0(BaroEKF_TypeDef *k, float p0_pa);
void BaroEKF_Update(BaroEKF_TypeDef *k, float pressure_pa, float temp_c);
float BaroEKF_GetPressure(const BaroEKF_TypeDef *k);
float BaroEKF_GetRate(const BaroEKF_TypeDef *k);
float BaroEKF_GetP00(const BaroEKF_TypeDef *k);   /* 气压状态方差 P[0][0]，方案21 融合加权用 */

#endif /* _BARO_EKF_H_ */
