#ifndef _BSP_BMP280_H_
#define _BSP_BMP280_H_

#include "stm32f4xx_hal.h"
#include <stdbool.h>  // ????
#define BMP280_DEFAULT_CHIP_ID 0x58

#define BMP280_CHIP_ID_REG 0xD0
#define BMP280_RST_REG 0xE0
#define BMP280_CTRL_MEAS_REG 0xF4
#define BMP280_CONFIG_REG 0xF5
#define BMP280_PRESSURE_MSB_REG 0xF7
#define BMP280_TEMPERATURE_MSB_REG 0xFA
#define BMP280_TEMPERATURE_CALIB_DIG_T1_LSB_REG 0x88

#define BMP280_SLEEP_MODE 0x00
#define BMP280_FORCED_MODE 0x01
#define BMP280_NORMAL_MODE 0x03

#define BMP280_OVERSAMP_SKIPPED 0x00
#define BMP280_OVERSAMP_1X 0x01
#define BMP280_OVERSAMP_2X 0x02
#define BMP280_OVERSAMP_4X 0x03
#define BMP280_OVERSAMP_8X 0x04
#define BMP280_OVERSAMP_16X 0x05

typedef struct {
    uint16_t dig_T1;
    int16_t dig_T2;
    int16_t dig_T3;
    uint16_t dig_P1;
    int16_t dig_P2;
    int16_t dig_P3;
    int16_t dig_P4;
    int16_t dig_P5;
    int16_t dig_P6;
    int16_t dig_P7;
    int16_t dig_P8;
    int16_t dig_P9;
    int32_t t_fine;
} BMP280_Calib_TypeDef;

void BMP280_IIC_Init(void);
bool BMP280_Init(void);
void BMP280_Read_Data(float *pressure_pa, float *temperature_c);

#endif
