#include "baro_ekf.h"
#include <math.h>

/* 温度递减率 (K/m) 与 R·L/(g·M) 物理常数组合 (ISO 2533 标准大气) */
#define BARO_EKF_LAPSE     0.0065f
#define BARO_EKF_EXPONENT  0.190263f

void BaroEKF_Init(BaroEKF_TypeDef *k, float p0_pa, float dt,
                  float q_press, float q_rate, float r_meas)
{
    k->p = p0_pa;
    k->dp = 0.0f;
    k->P[0][0] = 100.0f;  k->P[0][1] = 0.0f;
    k->P[1][0] = 0.0f;    k->P[1][1] = 100.0f;
    k->q_press = q_press;
    k->q_rate  = q_rate;
    k->r_meas  = r_meas;
    k->p0_pa   = p0_pa;
    k->dt      = dt;
}

void BaroEKF_Reset(BaroEKF_TypeDef *k, float press_pa)
{
    k->p = press_pa;
    k->dp = 0.0f;
    k->P[0][0] = 100.0f;  k->P[0][1] = 0.0f;
    k->P[1][0] = 0.0f;    k->P[1][1] = 100.0f;
}

void BaroEKF_SetP0(BaroEKF_TypeDef *k, float p0_pa)
{
    k->p0_pa = p0_pa;
}

void BaroEKF_Update(BaroEKF_TypeDef *k, float pressure_pa, float temp_c)
{
    float dt  = k->dt;
    float dt2 = dt * dt;
    float dt3 = dt2 * dt;

    /* ═══ 预测步 (匀速模型 F=[[1,dt],[0,1]]) ═══ */
    float x0 = k->p + dt * k->dp;     /* 气压预测     */
    float x1 = k->dp;                 /* 变化率不变   */
    float P00 = k->P[0][0];
    float P01 = k->P[0][1];
    float P10 = k->P[1][0];
    float P11 = k->P[1][1];

    P00 = P00 + dt * (P10 + P01) + dt2 * P11;
    P01 = P01 + dt * P11;
    P10 = P10 + dt * P11;
    P11 = P11;

    /* 过程噪声 Q (连续白噪声积分) */
    P00 += k->q_rate * dt3 / 3.0f + k->q_press * dt;
    P01 += k->q_rate * dt2 / 2.0f;
    P10 += k->q_rate * dt2 / 2.0f;
    P11 += k->q_rate * dt;

    /* ═══ 更新步 (EKF 核心: 雅可比代替常数 H) ═══
     * h(P) = (T0/L)*(1-(P/P0)^exp)
     * ∂h/∂P = -(T0/L)*exp*P^(exp-1)/P0^exp
     */
    float T0 = temp_c + 273.15f;                  /* °C → K */
    float Pabs = (x0 > 100.0f) ? x0 : k->p0_pa;    /* 防零除 */
    float ratio = powf(Pabs / k->p0_pa, BARO_EKF_EXPONENT - 1.0f);
    float Hjac = -(T0 / BARO_EKF_LAPSE) * BARO_EKF_EXPONENT * ratio / k->p0_pa;

    /* 高度残差 y = h(测量) - h(预测) */
    float h_pred = (T0 / BARO_EKF_LAPSE) * (1.0f - powf(x0 / k->p0_pa, BARO_EKF_EXPONENT));
    float h_meas = (T0 / BARO_EKF_LAPSE) * (1.0f - powf(pressure_pa / k->p0_pa, BARO_EKF_EXPONENT));
    float y = h_meas - h_pred;

    /* 创新协方差 S = H·P·Hᵀ + R (测量仅依赖气压 x[0]) */
    float S = Hjac * P00 * Hjac + k->r_meas;
    float Si = (S != 0.0f) ? 1.0f / S : 0.0f;

    /* 卡尔曼增益 K = P·Hᵀ / S */
    float K0 = P00 * Hjac * Si;
    float K1 = P10 * Hjac * Si;

    /* 状态修正 (用高度残差修正气压状态) */
    k->p  = x0 + K0 * y;
    k->dp = x1 + K1 * y;

    /* 协方差更新 (对称简化形式) */
    k->P[0][0] = (1.0f - K0 * Hjac) * P00;
    k->P[0][1] = (1.0f - K0 * Hjac) * P01;
    k->P[1][0] = k->P[0][1];
    k->P[1][1] = P11 - K1 * Hjac * P01;
}

float BaroEKF_GetPressure(const BaroEKF_TypeDef *k) { return k->p; }
float BaroEKF_GetRate(const BaroEKF_TypeDef *k)     { return k->dp; }
float BaroEKF_GetP00(const BaroEKF_TypeDef *k)      { return k->P[0][0]; }
