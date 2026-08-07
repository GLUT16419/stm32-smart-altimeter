#ifndef __OLED_H
#define __OLED_H

#include "stm32f4xx_hal.h"

/* I2C 引脚定义 — 使用 PC0(SDA), PC1(SCL) 避免与 MS5611/BMP280 冲突 */
#define OLED_SDA_PORT   GPIOC
#define OLED_SDA_PIN    GPIO_PIN_0
#define OLED_SCL_PORT   GPIOC
#define OLED_SCL_PIN    GPIO_PIN_1

/* I2C 操作宏 */
#define OLED_SDA_Set()  HAL_GPIO_WritePin(OLED_SDA_PORT, OLED_SDA_PIN, GPIO_PIN_SET)
#define OLED_SDA_Clr()  HAL_GPIO_WritePin(OLED_SDA_PORT, OLED_SDA_PIN, GPIO_PIN_RESET)
#define OLED_SCL_Set()  HAL_GPIO_WritePin(OLED_SCL_PORT, OLED_SCL_PIN, GPIO_PIN_SET)
#define OLED_SCL_Clr()  HAL_GPIO_WritePin(OLED_SCL_PORT, OLED_SCL_PIN, GPIO_PIN_RESET)

#define OLED_CMD  0  /* 命令 */
#define OLED_DATA 1  /* 数据 */

/* ====== 基本操作 ====== */
extern uint8_t OLED_GRAM[128][8];
void OLED_Init(void);
void OLED_Clear(void);
void OLED_Refresh(void);

/* ====== 显示控制 ====== */
void OLED_DisPlay_On(void);
void OLED_DisPlay_Off(void);
void OLED_ColorTurn(uint8_t i);    /* 0=正常, 1=反色 */
void OLED_DisplayTurn(uint8_t i);  /* 0=正常, 1=旋转180度 */

/* ====== 绘图 ====== */
void OLED_DrawPoint(uint8_t x, uint8_t y, uint8_t t);      /* t=1 点亮, t=0 熄灭 */
void OLED_ClearPoint(uint8_t x, uint8_t y);
void OLED_DrawLine(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t mode);
void OLED_DrawCircle(uint8_t x, uint8_t y, uint8_t r);

/* ====== 字符/字符串显示 ====== */
void OLED_ShowChar(uint8_t x, uint8_t y, uint8_t chr, uint8_t size1, uint8_t mode);
void OLED_ShowChar6x8(uint8_t x, uint8_t y, uint8_t chr, uint8_t mode);
void OLED_ShowString(uint8_t x, uint8_t y, uint8_t *chr, uint8_t size1, uint8_t mode);
void OLED_ShowNum(uint8_t x, uint8_t y, uint32_t num, uint8_t len, uint8_t size1, uint8_t mode);
void OLED_ShowFloat(uint8_t x, uint8_t y, float num, uint8_t intLen, uint8_t decLen, uint8_t size1, uint8_t mode);

/* ====== 扩展功能 ====== */
void OLED_ShowChinese(uint8_t x, uint8_t y, uint8_t num, uint8_t size1, uint8_t mode);
void OLED_ShowPicture(uint8_t x, uint8_t y, uint8_t sizex, uint8_t sizey, uint8_t BMP[], uint8_t mode);
void OLED_ScrollDisplay(uint8_t num, uint8_t space, uint8_t mode);

#endif
