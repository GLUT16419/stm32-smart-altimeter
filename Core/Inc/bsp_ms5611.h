#ifndef _BSP_MS5611_H_
#define _BSP_MS5611_H_

#include "stm32f4xx_hal.h"
#include <stdbool.h>

/* ===== MS5611 软件 SPI 引脚定义 ===== */
/* 复用原 I2C 引脚 PB6(SCL)→SCLK, PB7(SDA)→SDI(MOSI) */
/* 新增 PB5→SDO(MISO), PB4→CSB */
#define MS5611_SPI_PORT      GPIOB

#define MS5611_SPI_SCLK_PIN  GPIO_PIN_6   /* 原 I2C SCL */
#define MS5611_SPI_SDI_PIN   GPIO_PIN_7   /* 原 I2C SDA → MOSI */
#define MS5611_SPI_SDO_PIN   GPIO_PIN_5   /* 新增 MISO */
#define MS5611_SPI_CSB_PIN   GPIO_PIN_4   /* 新增 片选 */

/* SPI 操作宏 */
#define MS5611_SPI_SCLK(x)   HAL_GPIO_WritePin(MS5611_SPI_PORT, MS5611_SPI_SCLK_PIN, (x ? GPIO_PIN_SET : GPIO_PIN_RESET))
#define MS5611_SPI_SDI(x)    HAL_GPIO_WritePin(MS5611_SPI_PORT, MS5611_SPI_SDI_PIN,  (x ? GPIO_PIN_SET : GPIO_PIN_RESET))
#define MS5611_SPI_SDO_GET() HAL_GPIO_ReadPin(MS5611_SPI_PORT, MS5611_SPI_SDO_PIN)
#define MS5611_SPI_CSB(x)    HAL_GPIO_WritePin(MS5611_SPI_PORT, MS5611_SPI_CSB_PIN, (x ? GPIO_PIN_SET : GPIO_PIN_RESET))

void MS5611_GPIO_Init(void);
char MS5611_Reset(void);
void MS5611_Read_PROM(void);
float MS5611_Get_Temperature(void);
float MS5611_Get_Pressure(void);
void MS5611_Read_Data(float *pressure_pa, float *temperature_c);

extern uint16_t Cal_C1_6[8];

#endif
