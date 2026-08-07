/**
 * @file    ssd1306.c
 * @brief   SSD1306 OLED 0.96" 128x64 I2C Driver for STM32 HAL
 *
 * Based on afiskon/stm32-ssd1306 library.
 * Uses software bit-bang I2C on PC0(SDA) / PC1(SCL).
 *
 * I2C address: 0x3C (7-bit) -> 0x78 (8-bit write address)
 */

#include "ssd1306.h"
#include "ssd1306_fonts.h"
#include "oled.h"  /* for OLED_SDA/SCL pin definitions and macros */

/* ======================== Static Variables ======================== */

static uint8_t ssd1306_buffer[SSD1306_BUFFER_SIZE];

/* ======================== Software I2C Bit-Bang ======================== */

static void IIC_Delay(void)
{
    volatile uint8_t t = 8;
    while (t--);
}

static void I2C_Start(void)
{
    __disable_irq();
    OLED_SDA_Set();
    IIC_Delay();
    OLED_SCL_Set();
    IIC_Delay();
    OLED_SDA_Clr();
    IIC_Delay();
    OLED_SCL_Clr();
    IIC_Delay();
    __enable_irq();
}

static void I2C_Stop(void)
{
    __disable_irq();
    OLED_SDA_Clr();
    IIC_Delay();
    OLED_SCL_Set();
    IIC_Delay();
    OLED_SDA_Set();
    IIC_Delay();
    __enable_irq();
}

static void I2C_SendByte(uint8_t dat)
{
    uint8_t i;
    __disable_irq();
    for (i = 0; i < 8; i++)
    {
        if (dat & 0x80)
            OLED_SDA_Set();
        else
            OLED_SDA_Clr();
        IIC_Delay();
        OLED_SCL_Set();
        IIC_Delay();
        OLED_SCL_Clr();
        IIC_Delay();
        dat <<= 1;
    }
    __enable_irq();
}

static void I2C_Ack(void)
{
    __disable_irq();
    OLED_SDA_Set();   /* Release SDA */
    IIC_Delay();
    OLED_SCL_Set();   /* 9th clock */
    IIC_Delay();
    OLED_SCL_Clr();
    IIC_Delay();
    __enable_irq();
}

/* ======================== Low-Level I2C Functions ======================== */

static void SSD1306_WriteCommand(uint8_t cmd)
{
    I2C_Start();
    I2C_SendByte(0x78);   /* 0x3C << 1, write */
    I2C_Ack();
    I2C_SendByte(0x00);   /* Co = 0, D/C# = 0 (command) */
    I2C_Ack();
    I2C_SendByte(cmd);
    I2C_Ack();
    I2C_Stop();
}

/* ======================== Initialization ======================== */

void SSD1306_Init(void)
{
    /* Initialize GPIO */
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    __HAL_RCC_GPIOC_CLK_ENABLE();
    GPIO_InitStruct.Pin = OLED_SDA_PIN | OLED_SCL_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
    OLED_SDA_Set();
    OLED_SCL_Set();

    /* Wait for display to power up */
    HAL_Delay(100);

    /* Initialization command sequence */
    static const uint8_t init_cmds[] = {
        0xAE,           /* Display OFF */
        0x81, 0xCF,     /* Contrast control */
        0xA1,           /* Segment re-map: column 127 -> SEG0 (mirrored) */
        0xC8,           /* COM output scan direction: reversed */
        0xA6,           /* Normal display (not inverted) */
        0xA8, 0x3F,     /* Multiplex ratio: 64 (for 128x64) */
        0xD3, 0x00,     /* Display offset: 0 */
        0xD5, 0x80,     /* Oscillator frequency / clock divide */
        0xD9, 0xF1,     /* Pre-charge period */
        0xDA, 0x12,     /* COM pins hardware configuration */
        0xDB, 0x30,     /* VCOMH deselect level */
        0x20, 0x00,     /* Memory addressing mode: Horizontal (0x00) */
        0x21, 0x00, 0x7F, /* Column address range: 0 to 127 */
        0x22, 0x00, 0x07, /* Page address range: 0 to 7 */
        0x8D, 0x14,     /* Charge pump enable */
        0xAF            /* Display ON */
    };

    for (size_t i = 0; i < sizeof(init_cmds); i++) {
        SSD1306_WriteCommand(init_cmds[i]);
    }

    /* Clear framebuffer and display */
    SSD1306_Fill(SSD1306_COLOR_BLACK);
    SSD1306_UpdateScreen();
}

/* ======================== Display Control ======================== */

void SSD1306_SetContrast(uint8_t value)
{
    SSD1306_WriteCommand(0x81);
    SSD1306_WriteCommand(value);
}

void SSD1306_DisplayOn(void)
{
    SSD1306_WriteCommand(0xAF);
}

void SSD1306_DisplayOff(void)
{
    SSD1306_WriteCommand(0xAE);
}

/* ======================== Drawing Functions ======================== */

void SSD1306_Fill(uint8_t color)
{
    memset(ssd1306_buffer, color ? 0xFF : 0x00, SSD1306_BUFFER_SIZE);
}

void SSD1306_UpdateScreen(void)
{
    /* Horizontal addressing mode: set column range once */
    SSD1306_WriteCommand(0x21);
    SSD1306_WriteCommand(0x00);
    SSD1306_WriteCommand(0x7F);
    /* Set page range once */
    SSD1306_WriteCommand(0x22);
    SSD1306_WriteCommand(0x00);
    SSD1306_WriteCommand(0x07);

    IIC_Delay();

    /* Send all 1024 bytes of framebuffer in one I2C transaction */
    I2C_Start();
    I2C_SendByte(0x78);
    I2C_Ack();
    I2C_SendByte(0x40);  /* Data mode */
    I2C_Ack();
    for (uint16_t i = 0; i < SSD1306_BUFFER_SIZE; i++) {
        I2C_SendByte(ssd1306_buffer[i]);
        I2C_Ack();
    }
    I2C_Stop();
}

void SSD1306_DrawPixel(uint8_t x, uint8_t y, uint8_t color)
{
    if (x >= SSD1306_WIDTH || y >= SSD1306_HEIGHT) return;

    if (color) {
        ssd1306_buffer[x + (y / 8) * SSD1306_WIDTH] |= (1 << (y % 8));
    } else {
        ssd1306_buffer[x + (y / 8) * SSD1306_WIDTH] &= ~(1 << (y % 8));
    }
}

void SSD1306_DrawLine(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1, uint8_t color)
{
    int16_t dx = (int16_t)x1 - (int16_t)x0;
    int16_t dy = (int16_t)y1 - (int16_t)y0;
    int16_t abs_dx = (dx >= 0) ? dx : -dx;
    int16_t abs_dy = (dy >= 0) ? dy : -dy;
    int16_t sx = (dx >= 0) ? 1 : -1;
    int16_t sy = (dy >= 0) ? 1 : -1;
    int16_t err = abs_dx - abs_dy;
    int16_t cx = (int16_t)x0;
    int16_t cy = (int16_t)y0;

    while (1) {
        SSD1306_DrawPixel((uint8_t)cx, (uint8_t)cy, color);
        if (cx == x1 && cy == y1) break;
        int16_t e2 = err * 2;
        if (e2 > -abs_dy) {
            err -= abs_dy;
            cx += sx;
        }
        if (e2 < abs_dx) {
            err += abs_dx;
            cy += sy;
        }
    }
}

void SSD1306_DrawRectangle(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint8_t color)
{
    SSD1306_DrawLine(x, y, x + w - 1, y, color);
    SSD1306_DrawLine(x, y + h - 1, x + w - 1, y + h - 1, color);
    SSD1306_DrawLine(x, y, x, y + h - 1, color);
    SSD1306_DrawLine(x + w - 1, y, x + w - 1, y + h - 1, color);
}

void SSD1306_DrawFilledRectangle(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint8_t color)
{
    for (uint8_t yi = y; yi < y + h; yi++) {
        for (uint8_t xi = x; xi < x + w; xi++) {
            SSD1306_DrawPixel(xi, yi, color);
        }
    }
}

/* ======================== Character / String Drawing ======================== */

uint8_t SSD1306_DrawChar(uint8_t x, uint8_t y, char ch, SSD1306_FontSize_t font, uint8_t color)
{
    const SSD1306_FontDef_t *font_def;
    uint8_t char_width, char_height;

    if (font == SSD1306_FONT_6X8) {
        font_def = &Font_6x8;
        char_width = 6;
        char_height = 8;
    } else {
        font_def = &Font_8x16;
        char_width = 8;
        char_height = 16;
    }

    /* Only printable ASCII characters */
    if (ch < 0x20 || ch > 0x7E) {
        ch = '?';
    }

    uint16_t char_index = (uint16_t)(ch - 0x20);

    for (uint8_t row = 0; row < char_height; row++) {
        uint16_t row_data = font_def->data[char_index * char_height + row];

        for (uint8_t col = 0; col < char_width; col++) {
            if (row_data & (1 << (char_width - 1 - col))) {
                SSD1306_DrawPixel(x + col, y + row, color);
            } else {
                SSD1306_DrawPixel(x + col, y + row, !color);
            }
        }
    }

    return char_width;
}

uint8_t SSD1306_DrawString(uint8_t x, uint8_t y, const char *str, SSD1306_FontSize_t font, uint8_t color)
{
    uint8_t cursor_x = x;
    uint8_t char_w = (font == SSD1306_FONT_6X8) ? 6 : 8;

    while (*str) {
        SSD1306_DrawChar(cursor_x, y, *str, font, color);
        cursor_x += char_w;
        /* Wrap to next line if needed */
        if (cursor_x + char_w > SSD1306_WIDTH) {
            cursor_x = x;
            y += (font == SSD1306_FONT_6X8) ? 9 : 17;
        }
        str++;
    }

    return cursor_x - x;
}

/* ======================== Formatted Output ======================== */

void SSD1306_Printf(uint8_t x, uint8_t y, SSD1306_FontSize_t font, uint8_t color, const char *fmt, ...)
{
    char buf[64];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    SSD1306_DrawString(x, y, buf, font, color);
}

void SSD1306_DrawFloat(uint8_t x, uint8_t y, float value, uint8_t decimals, SSD1306_FontSize_t font, uint8_t color)
{
    char buf[16];
    int32_t int_part = (int32_t)value;
    float frac = value - (float)int_part;
    if (frac < 0) frac = -frac;

    float multiplier = 1.0f;
    for (uint8_t d = 0; d < decimals; d++) multiplier *= 10.0f;
    uint32_t frac_part = (uint32_t)(frac * multiplier + 0.5f);

    if (decimals > 0) {
        snprintf(buf, sizeof(buf), "%+ld.%0*lu", (long)int_part, decimals, (unsigned long)frac_part);
    } else {
        snprintf(buf, sizeof(buf), "%+ld", (long)int_part);
    }
    SSD1306_DrawString(x, y, buf, font, color);
}

void SSD1306_DrawInt(uint8_t x, uint8_t y, int32_t value, SSD1306_FontSize_t font, uint8_t color)
{
    char buf[12];
    snprintf(buf, sizeof(buf), "%ld", (long)value);
    SSD1306_DrawString(x, y, buf, font, color);
}

/* ======================== Bar Graph ======================== */

void SSD1306_DrawBar(uint8_t x, uint8_t y, uint8_t w, uint8_t h, float value, float min, float max)
{
    if (value < min) value = min;
    if (value > max) value = max;

    float ratio = (value - min) / (max - min);
    uint8_t fill_w = (uint8_t)(ratio * (float)w);

    SSD1306_DrawRectangle(x, y, w, h, SSD1306_COLOR_WHITE);

    if (fill_w > 0) {
        SSD1306_DrawFilledRectangle(x + 1, y + 1, fill_w, h - 2, SSD1306_COLOR_WHITE);
    }
}
