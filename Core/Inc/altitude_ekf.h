#ifndef _ALTITUDE_EKF_H_
#define _ALTITUDE_EKF_H_

#include "stm32f4xx_hal.h"

#define SEA_LEVEL_PRESSURE_PA 101325.0f

typedef struct {
    float height_m;
    float velocity_mps;
    float p00;
    float p01;
    float p10;
    float p11;
    float reference_pressure_pa;
    float accel_var;
    float pressure_var;
} AltitudeEKF_TypeDef;

void AltitudeEKF_Init(AltitudeEKF_TypeDef *ekf, float reference_pressure_pa);
void AltitudeEKF_SetNoise(AltitudeEKF_TypeDef *ekf, float accel_sigma, float pressure_sigma);
void AltitudeEKF_Update(AltitudeEKF_TypeDef *ekf, float pressure_pa, float dt_s);
float AltitudeEKF_PressureToAltitude(float pressure_pa, float reference_pressure_pa);
float AltitudeEKF_GetHeight(AltitudeEKF_TypeDef *ekf);
float AltitudeEKF_GetVelocity(AltitudeEKF_TypeDef *ekf);

#endif

