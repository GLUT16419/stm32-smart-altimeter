/**
 * @file    ssd1306_fonts.h
 * @brief   Font definitions for SSD1306 OLED
 *
 * Fonts included:
 *   - Font_6x8: 6x8 pixel monospace font (ASCII 0x20-0x7E)
 *   - Font_8x16: 8x16 pixel monospace font (ASCII 0x20-0x7E)
 */

#ifndef __SSD1306_FONTS_H
#define __SSD1306_FONTS_H

#include <stdint.h>

/* ======================== Font Structure ======================== */

typedef struct {
    const uint8_t width;
    const uint8_t height;
    const uint16_t *data;   /* Pointer to font bitmap data */
} SSD1306_FontDef_t;

/* ======================== 6x8 Font ======================== */

extern const SSD1306_FontDef_t Font_6x8;

/* ======================== 8x16 Font ======================== */

extern const SSD1306_FontDef_t Font_8x16;

#endif /* __SSD1306_FONTS_H */
