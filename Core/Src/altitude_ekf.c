#include "altitude_ekf.h"
#include <math.h>

#define GRAVITY 9.81f
#define TEMPERATURE_LAPSE_RATE 0.0065f
#define SEA_LEVEL_TEMP 288.15f
#define GAS_CONSTANT 287.05f

static float AltitudeEKF_PressureModel(float height_m, float reference_pressure_pa)
{
    float exponent = -(GRAVITY * height_m) / (GAS_CONSTANT * SEA_LEVEL_TEMP);
    return reference_pressure_pa * expf(exponent);
}

static float AltitudeEKF_PressureJacobian(float height_m, float reference_pressure_pa)
{
    float exponent = -(GRAVITY * height_m) / (GAS_CONSTANT * SEA_LEVEL_TEMP);
    return reference_pressure_pa * expf(exponent) * (-GRAVITY / (GAS_CONSTANT * SEA_LEVEL_TEMP));
}

void AltitudeEKF_Init(AltitudeEKF_TypeDef *ekf, float reference_pressure_pa)
{
    ekf->height_m = 0.0f;
    ekf->velocity_mps = 0.0f;
    ekf->p00 = 4.0f;
    ekf->p01 = 0.0f;
    ekf->p10 = 0.0f;
    ekf->p11 = 1.0f;
    ekf->reference_pressure_pa = reference_pressure_pa;
    ekf->accel_var = 2.25f;
    ekf->pressure_var = 16.0f;
}

void AltitudeEKF_SetNoise(AltitudeEKF_TypeDef *ekf, float accel_sigma, float pressure_sigma)
{
    ekf->accel_var = accel_sigma * accel_sigma;
    ekf->pressure_var = pressure_sigma * pressure_sigma;
}

void AltitudeEKF_Update(AltitudeEKF_TypeDef *ekf, float pressure_pa, float dt_s)
{
    float dt2 = dt_s * dt_s;
    float dt3 = dt2 * dt_s;
    float dt4 = dt3 * dt_s;
    
    // Predict state
    float predicted_height = ekf->height_m + ekf->velocity_mps * dt_s;
    float predicted_velocity = ekf->velocity_mps;
    
    // Predict covariance
    float p00_pred = ekf->p00 + ekf->p01 * dt_s + ekf->p10 * dt_s + ekf->p11 * dt2 + 0.25f * dt4 * ekf->accel_var;
    float p01_pred = ekf->p01 + ekf->p11 * dt_s + 0.5f * dt3 * ekf->accel_var;
    float p10_pred = ekf->p10 + ekf->p11 * dt_s + 0.5f * dt3 * ekf->accel_var;
    float p11_pred = ekf->p11 + dt2 * ekf->accel_var;
    
    // Compute observation prediction
    float predicted_pressure = AltitudeEKF_PressureModel(predicted_height, ekf->reference_pressure_pa);
    
    // Compute observation Jacobian
    float h0 = AltitudeEKF_PressureJacobian(predicted_height, ekf->reference_pressure_pa);
    float h1 = 0.0f;
    
    // Compute residual
    float residual = pressure_pa - predicted_pressure;
    
    // Compute residual covariance
    float s = h0 * h0 * p00_pred + h0 * h1 * p01_pred + h1 * h0 * p10_pred + h1 * h1 * p11_pred + ekf->pressure_var;
    
    // Compute Kalman gain
    float k0 = (p00_pred * h0 + p01_pred * h1) / s;
    float k1 = (p10_pred * h0 + p11_pred * h1) / s;
    
    // Update state
    ekf->height_m = predicted_height + k0 * residual;
    ekf->velocity_mps = predicted_velocity + k1 * residual;
    
    // Update covariance (Joseph form)
    float kh0 = k0 * h0;
    float kh1 = k0 * h1;
    float k1h0 = k1 * h0;
    float k1h1 = k1 * h1;
    
    ekf->p00 = (1.0f - kh0) * p00_pred - kh1 * p10_pred;
    ekf->p01 = (1.0f - kh0) * p01_pred - kh1 * p11_pred;
    ekf->p10 = (1.0f - k1h0) * p10_pred - k1h1 * p00_pred;
    ekf->p11 = (1.0f - k1h0) * p11_pred - k1h1 * p01_pred;
}

float AltitudeEKF_PressureToAltitude(float pressure_pa, float reference_pressure_pa)
{
    if (pressure_pa <= 0.0f || reference_pressure_pa <= 0.0f)
        return 0.0f;
    
    float ratio = pressure_pa / reference_pressure_pa;
    if (ratio >= 1.0f)
        return 0.0f;
    
    return -(GAS_CONSTANT * SEA_LEVEL_TEMP / GRAVITY) * logf(ratio);
}

float AltitudeEKF_GetHeight(AltitudeEKF_TypeDef *ekf)
{
    return ekf->height_m;
}

float AltitudeEKF_GetVelocity(AltitudeEKF_TypeDef *ekf)
{
    return ekf->velocity_mps;
}

