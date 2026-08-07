#include "altitude_convert.h"
#include "math.h"

float PressureToAltitude(float pressure_pa, float reference_pressure_pa)
{
    if(pressure_pa <= 0 || reference_pressure_pa <= 0) return 0.0f;

    float ratio = pressure_pa / reference_pressure_pa;
    if(ratio <= 0) return 0.0f;

#if ALTITUDE_FORMULA_ISA
    /* ISA 国际标准大气公式：h = (T0/L) * (1 - (P/P0)^(L*R/g)) */
    /* 适用于 0~11000m 对流层，考虑了温度随高度递减 */
    float altitude = (ISA_T0 / ISA_L) * (1.0f - powf(ratio, ISA_L * ISA_R / ISA_G));
#else
    /* 等温模型 */
    float altitude = -(287.05f * 288.15f / 9.80665f) * logf(ratio);
#endif

    return altitude;
}

float PressureToAltitudeWithTemp(float pressure_pa, float reference_pressure_pa, float temperature_c)
{
    if(pressure_pa <= 0 || reference_pressure_pa <= 0) return 0.0f;
    if(temperature_c < -100.0f || temperature_c > 100.0f) return 0.0f;

#if ALTITUDE_FORMULA_ISA
    /* ISA 公式（带温度补偿版）：用实测温度做一阶修正 */
    float ratio = pressure_pa / reference_pressure_pa;
    if(ratio <= 0.0f || ratio > 10.0f) return 0.0f;

    float altitude = (ISA_T0 / ISA_L) * (1.0f - powf(ratio, ISA_L * ISA_R / ISA_G));

    /* 用实测温度修正标准温度偏差 */
    float temperature_k = temperature_c + 273.15f;
    float delta_t = temperature_k - (ISA_T0 - ISA_L * altitude);
    altitude += delta_t * 0.035f;  /* 经验系数：每度温差约修正 3.5cm */

    if(altitude < -1000.0f || altitude > 20000.0f) return 0.0f;
    return altitude;
#else
    if(pressure_pa <= 0 || reference_pressure_pa <= 0) return 0.0f;
    if(temperature_c < -100.0f || temperature_c > 100.0f) return 0.0f;

    float temperature_k = temperature_c + 273.15f;

    float ratio = pressure_pa / reference_pressure_pa;
    if(ratio <= 0.0f || ratio > 10.0f) return 0.0f;

    float altitude = -(287.05f * temperature_k / 9.80665f) * logf(ratio);

    if(altitude < -1000.0f || altitude > 20000.0f) return 0.0f;

    return altitude;
#endif
}

/* ISA 国际标准大气公式版本（无温度补偿） */
float PressureToAltitudeISA(float pressure_pa, float reference_pressure_pa)
{
    if(pressure_pa <= 0.0f || reference_pressure_pa <= 0.0f) return 0.0f;

    float ratio = pressure_pa / reference_pressure_pa;
    if(ratio <= 0.0f || ratio > 10.0f) return 0.0f;

    return (ISA_T0 / ISA_L) * (1.0f - powf(ratio, ISA_L * ISA_R / ISA_G));
}
