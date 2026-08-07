/**
 * @file    ssd1306.h
 * @brief   SSD1306 OLED 0.96" 128x64 I2C Driver for STM32 HAL
 *
 * Uses software bit-bang I2C on PC0(SDA) / PC1(SCL).
 *
 * I2C address: 0x3C (7-bit) -> 0x78 (8-bit HAL write address)
 */

#ifndef __SSD1306_H
#define __SSD1306_H

#include "stm32f4xx_hal.h"
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

/* ======================== Hardware Configuration ======================== */

#define SSD1306_I2C_ADDR        0x3C    /* 7-bit I2C address */
#define SSD1306_WIDTH           128
#define SSD1306_HEIGHT          64
#define SSD1306_BUFFER_SIZE     (SSD1306_WIDTH * SSD1306_HEIGHT / 8)  /* 1024 bytes */

/* ======================== Enumerations ======================== */

typedef enum {
    SSD1306_COLOR_BLACK = 0,
    SSD1306_COLOR_WHITE = 1
} SSD1306_Color_t;

typedef enum {
    SSD1306_FONT_6X8 = 0,
    SSD1306_FONT_8X16 = 1
} SSD1306_FontSize_t;

/* ======================== API Functions ======================== */

/**
 * @brief  Initialize SSD1306 OLED display (software I2C on PC0/PC1)
 */
void SSD1306_Init(void);

/**
 * @brief  Fill entire screen with specified color
 * @param  color: SSD1306_COLOR_BLACK or SSD1306_COLOR_WHITE
 */
void SSD1306_Fill(uint8_t color);

/**
 * @brief  Update screen by sending framebuffer via I2C
 */
void SSD1306_UpdateScreen(void);

/**
 * @brief  Draw a single pixel
 */
void SSD1306_DrawPixel(uint8_t x, uint8_t y, uint8_t color);

/**
 * @brief  Draw a character at specified position
 * @param  x, y: Top-left corner of character
 * @param  ch: ASCII character
 * @param  font: SSD1306_FONT_6X8 or SSD1306_FONT_8X16
 * @param  color: SSD1306_COLOR_BLACK or SSD1306_COLOR_WHITE
 * @return Character width in pixels
 */
uint8_t SSD1306_DrawChar(uint8_t x, uint8_t y, char ch, SSD1306_FontSize_t font, uint8_t color);

/**
 * @brief  Draw a string at specified position
 * @return Total width drawn in pixels
 */
uint8_t SSD1306_DrawString(uint8_t x, uint8_t y, const char *str, SSD1306_FontSize_t font, uint8_t color);

/**
 * @brief  Draw a line using Bresenham's algorithm
 */
void SSD1306_DrawLine(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1, uint8_t color);

/**
 * @brief  Draw a rectangle
 */
void SSD1306_DrawRectangle(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint8_t color);

/**
 * @brief  Draw a filled rectangle
 */
void SSD1306_DrawFilledRectangle(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint8_t color);

/**
 * @brief  Formatted print to OLED (like printf)
 * @note   Uses internal static buffer, limited to 64 chars
 */
void SSD1306_Printf(uint8_t x, uint8_t y, SSD1306_FontSize_t font, uint8_t color, const char *fmt, ...);

/**
 * @brief  Display a floating-point number with specified decimal places
 */
void SSD1306_DrawFloat(uint8_t x, uint8_t y, float value, uint8_t decimals, SSD1306_FontSize_t font, uint8_t color);

/**
 * @brief  Display an integer
 */
void SSD1306_DrawInt(uint8_t x, uint8_t y, int32_t value, SSD1306_FontSize_t font, uint8_t color);

/**
 * @brief  Draw horizontal bargraph (for mini trend display)
 */
void SSD1306_DrawBar(uint8_t x, uint8_t y, uint8_t w, uint8_t h, float value, float min, float max);

/**
 * @brief  Set contrast (0-255)
 */
void SSD1306_SetContrast(uint8_t value);

/**
 * @brief  Turn display ON
 */
void SSD1306_DisplayOn(void);

/**
 * @brief  Turn display OFF
 */
void SSD1306_DisplayOff(void);

#endif /* __SSD1306_H */
