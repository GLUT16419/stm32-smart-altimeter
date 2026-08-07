#ifndef _ALTITUDE_CONVERT_H_
#define _ALTITUDE_CONVERT_H_

#include "stm32f4xx_hal.h"

#define SEA_LEVEL_PRESSURE_PA 101325.0f

/* 高度计算公式选择：
 *   0 = 等温模型: h = -(R*T/g) * ln(P/P0)  (当前默认)
 *   1 = ISA 标准大气: h = 44330 * (1 - (P/P0)^(1/5.255))
 *   ISA 公式在 0~11000m 范围内精度更高（考虑了温度递减率 0.0065 K/m）
 */
#define ALTITUDE_FORMULA_ISA  1

/* ISA 国际标准大气常数（供 main.c 等外部使用） */
#define ISA_T0    288.15f   /* 海平面标准温度 (K) */
#define ISA_L     0.0065f   /* 温度递减率 (K/m) */
#define ISA_G     9.80665f  /* 重力加速度 (m/s^2) */
#define ISA_R     287.05f   /* 干空气气体常数 (J/(kg·K)) */

float PressureToAltitude(float pressure_pa, float reference_pressure_pa);
float PressureToAltitudeWithTemp(float pressure_pa, float reference_pressure_pa, float temperature_c);
float PressureToAltitudeISA(float pressure_pa, float reference_pressure_pa);

#endif
