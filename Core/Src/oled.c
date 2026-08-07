#include "oled.h"
#include "oledfont.h"

/* 显存缓冲区 [128][8] */
uint8_t OLED_GRAM[128][8];

/* ====== 软件 I2C 时序（标准位敲打） ====== */
static void IIC_Delay(void)
{
    volatile uint8_t t = 10;
    while (t--);
}

/* I2C 起始条件：SCL高电平时 SDA 从高变低 */
static void OLED_I2C_Start(void)
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

/* I2C 停止条件：SCL高电平时 SDA 从低变高 */
static void OLED_I2C_Stop(void)
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

/* 发送一个字节，MSB first，不检测 ACK */
static void OLED_Send_Byte(uint8_t dat)
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
        OLED_SCL_Set();   /* 数据在 SCL 上升沿被采样 */
        IIC_Delay();
        OLED_SCL_Clr();   /* 拉低 SCL 准备下一位 */
        IIC_Delay();
        dat <<= 1;
    }
    __enable_irq();
}

/* 发送完字节后产生一个 ACK 时钟（不检测 SDA 状态，软件 I2C 简化处理） */
static void OLED_I2C_Ack(void)
{
    __disable_irq();
    OLED_SDA_Set();   /* 释放 SDA */
    IIC_Delay();
    OLED_SCL_Set();   /* 第9个时钟 */
    IIC_Delay();
    OLED_SCL_Clr();
    IIC_Delay();
    __enable_irq();
}

/* 写一个命令或数据字节（完整的 I2C 帧） */
static void OLED_WR_Byte(uint8_t dat, uint8_t mode)
{
    OLED_I2C_Start();
    OLED_Send_Byte(0x78);   /* 器件地址 + 写 */
    OLED_I2C_Ack();
    if (mode)
        OLED_Send_Byte(0x40);   /* 数据 */
    else
        OLED_Send_Byte(0x00);   /* 命令 */
    OLED_I2C_Ack();
    OLED_Send_Byte(dat);
    OLED_I2C_Ack();
    OLED_I2C_Stop();
}

/* ====== 初始化 GPIO ====== */
static void OLED_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOC_CLK_ENABLE();

    /* PC0(SDA), PC1(SCL) - 开漏输出（I2C 标准模式，需要外部上拉或 STM32 内部上拉） */
    GPIO_InitStruct.Pin = OLED_SDA_PIN | OLED_SCL_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    /* 初始高电平 */
    OLED_SDA_Set();
    OLED_SCL_Set();
}

/* ====== 显示控制 ====== */

/* 反色显示 */
void OLED_ColorTurn(uint8_t i)
{
    if (i == 0)
        OLED_WR_Byte(0xA6, OLED_CMD);  /* 正常显示 */
    else
        OLED_WR_Byte(0xA7, OLED_CMD);  /* 反色显示 */
}

/* 屏幕旋转180度 */
void OLED_DisplayTurn(uint8_t i)
{
    if (i == 0)
    {
        OLED_WR_Byte(0xC8, OLED_CMD);  /* 正常 */
        OLED_WR_Byte(0xA1, OLED_CMD);
    }
    else
    {
        OLED_WR_Byte(0xC0, OLED_CMD);  /* 反转 */
        OLED_WR_Byte(0xA0, OLED_CMD);
    }
}

/* 开启显示 */
void OLED_DisPlay_On(void)
{
    OLED_WR_Byte(0x8D, OLED_CMD);  /* 电荷泵使能 */
    OLED_WR_Byte(0x14, OLED_CMD);  /* 开启电荷泵 */
    OLED_WR_Byte(0xAF, OLED_CMD);  /* 点亮屏幕 */
}

/* 关闭显示 */
void OLED_DisPlay_Off(void)
{
    OLED_WR_Byte(0x8D, OLED_CMD);  /* 电荷泵使能 */
    OLED_WR_Byte(0x10, OLED_CMD);  /* 关闭电荷泵 */
    OLED_WR_Byte(0xAE, OLED_CMD);  /* 关闭屏幕 */
}

/* ====== 初始化 & 显存操作 ====== */

void OLED_Init(void)
{
    OLED_GPIO_Init();

    /* ====== 复位序列：拉低 SCL/SDA 模拟复位 ====== */
    HAL_Delay(50);
    OLED_I2C_Start();
    OLED_I2C_Stop();
    HAL_Delay(100);

    /* ====== 完整初始化序列 ====== */
    OLED_WR_Byte(0xAE, OLED_CMD); /* 关闭显示 */
    OLED_WR_Byte(0x00, OLED_CMD); /* 设置低列地址 */
    OLED_WR_Byte(0x10, OLED_CMD); /* 设置高列地址 */
    OLED_WR_Byte(0x40, OLED_CMD); /* 设置起始行地址 */
    OLED_WR_Byte(0x81, OLED_CMD); /* 对比度设置 */
    OLED_WR_Byte(0xFF, OLED_CMD); /* 亮度最大 */
    OLED_WR_Byte(0xA1, OLED_CMD); /* SEG 映射 正常 */
    OLED_WR_Byte(0xC8, OLED_CMD); /* COM 扫描 正常 */
    OLED_WR_Byte(0xA6, OLED_CMD); /* 正常显示 */
    OLED_WR_Byte(0xA8, OLED_CMD); /* 多路复用比 */
    OLED_WR_Byte(0x3F, OLED_CMD); /* 1/64 duty */
    OLED_WR_Byte(0xD3, OLED_CMD); /* 显示偏移 */
    OLED_WR_Byte(0x00, OLED_CMD);
    OLED_WR_Byte(0xD5, OLED_CMD); /* 时钟分频 */
    OLED_WR_Byte(0x80, OLED_CMD);
    OLED_WR_Byte(0xD9, OLED_CMD); /* 预充电周期 */
    OLED_WR_Byte(0xF1, OLED_CMD);
    OLED_WR_Byte(0xDA, OLED_CMD); /* COM 引脚配置 */
    OLED_WR_Byte(0x12, OLED_CMD);
    OLED_WR_Byte(0xDB, OLED_CMD); /* VCOMH */
    OLED_WR_Byte(0x40, OLED_CMD); /* VCOMH 调高一点 */
    OLED_WR_Byte(0x20, OLED_CMD); /* 页寻址模式 */
    OLED_WR_Byte(0x02, OLED_CMD);
    OLED_WR_Byte(0x8D, OLED_CMD); /* 电荷泵 */
    OLED_WR_Byte(0x14, OLED_CMD); /* 开启电荷泵 */
    HAL_Delay(50);
    OLED_Clear();
    OLED_WR_Byte(0xAF, OLED_CMD); /* 开启显示 */
    HAL_Delay(50);
}

void OLED_Clear(void)
{
    uint8_t i, n;
    for (i = 0; i < 8; i++)
    {
        for (n = 0; n < 128; n++)
        {
            OLED_GRAM[n][i] = 0;
        }
    }
    OLED_Refresh();
}

void OLED_Refresh(void)
{
    uint8_t i, n;
    for (i = 0; i < 8; i++)
    {
        OLED_WR_Byte(0xB0 + i, OLED_CMD);  /* 设置页地址 */
        OLED_WR_Byte(0x00, OLED_CMD);       /* 低列地址 */
        OLED_WR_Byte(0x10, OLED_CMD);       /* 高列地址 */
        IIC_Delay();

        /* 单次 I2C 事务发送整页 128 字节数据 */
        OLED_I2C_Start();
        OLED_Send_Byte(0x78);
        OLED_I2C_Ack();
        OLED_Send_Byte(0x40);  /* 数据模式 */
        OLED_I2C_Ack();
        for (n = 0; n < 128; n++)
        {
            OLED_Send_Byte(OLED_GRAM[n][i]);
        }
        OLED_I2C_Ack();  /* 最后一字节的 ACK */
        OLED_I2C_Stop();
    }
}

/* ====== 绘图函数 ====== */

void OLED_DrawPoint(uint8_t x, uint8_t y, uint8_t t)
{
    uint8_t i, m, n;
    if (x > 127 || y > 63) return;
    i = y / 8;
    m = y % 8;
    n = 1 << m;
    if (t)
        OLED_GRAM[x][i] |= n;
    else
    {
        OLED_GRAM[x][i] = ~OLED_GRAM[x][i];
        OLED_GRAM[x][i] |= n;
        OLED_GRAM[x][i] = ~OLED_GRAM[x][i];
    }
}

void OLED_ClearPoint(uint8_t x, uint8_t y)
{
    OLED_DrawPoint(x, y, 0);
}

/* 画线（Bresenham 算法） */
void OLED_DrawLine(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t mode)
{
    uint16_t t;
    int xerr = 0, yerr = 0, delta_x, delta_y, distance;
    int incx, incy, uRow, uCol;

    delta_x = x2 - x1;
    delta_y = y2 - y1;
    uRow = x1;
    uCol = y1;

    if (delta_x > 0) incx = 1;
    else if (delta_x == 0) incx = 0;
    else { incx = -1; delta_x = -delta_x; }

    if (delta_y > 0) incy = 1;
    else if (delta_y == 0) incy = 0;
    else { incy = -1; delta_y = -delta_y; }

    if (delta_x > delta_y) distance = delta_x;
    else distance = delta_y;

    for (t = 0; t <= (uint16_t)distance; t++)
    {
        OLED_DrawPoint(uRow, uCol, mode);
        xerr += delta_x;
        yerr += delta_y;
        if (xerr > distance)
        {
            xerr -= distance;
            uRow += incx;
        }
        if (yerr > distance)
        {
            yerr -= distance;
            uCol += incy;
        }
    }
}

/* 画圆 */
void OLED_DrawCircle(uint8_t x, uint8_t y, uint8_t r)
{
    int a = 0, b = r, num;
    while (2 * b * b >= r * r)
    {
        OLED_DrawPoint(x + a, y - b, 1);
        OLED_DrawPoint(x - a, y - b, 1);
        OLED_DrawPoint(x - a, y + b, 1);
        OLED_DrawPoint(x + a, y + b, 1);

        OLED_DrawPoint(x + b, y + a, 1);
        OLED_DrawPoint(x + b, y - a, 1);
        OLED_DrawPoint(x - b, y - a, 1);
        OLED_DrawPoint(x - b, y + a, 1);

        a++;
        num = (a * a + b * b) - r * r;
        if (num > 0)
        {
            b--;
            a--;
        }
    }
}

/* ====== 字符显示 ====== */

/* 6x8 字体专用显示（不调用 OLED_Refresh） */
void OLED_ShowChar6x8(uint8_t x, uint8_t y, uint8_t chr, uint8_t mode)
{
    uint8_t i, m, temp, chr1;
    uint8_t x0 = x, y0 = y;

    if (x > 122 || y > 55) return;
    chr1 = chr - ' ';
    for (i = 0; i < 6; i++)
    {
        temp = asc2_0806[chr1][i];
        for (m = 0; m < 8; m++)
        {
            if (temp & 0x01) OLED_DrawPoint(x, y, mode);
            else OLED_DrawPoint(x, y, !mode);
            temp >>= 1;
            y++;
        }
        x++;
        y = y0;
    }
}

/* 显示一个字符（支持 6x8, 12x12, 16x16, 24x24 字体） */
void OLED_ShowChar(uint8_t x, uint8_t y, uint8_t chr, uint8_t size1, uint8_t mode)
{
    uint8_t i, m, temp, size2, chr1;
    uint8_t x0 = x, y0 = y;

    if (size1 == 8) size2 = 6;
    else if (size1 == 12) size2 = 12;
    else size2 = (size1 / 8 + ((size1 % 8) ? 1 : 0)) * (size1 / 2);

    chr1 = chr - ' ';
    for (i = 0; i < size2; i++)
    {
        if (size1 == 8)
            temp = asc2_0806[chr1][i];
        else if (size1 == 12)
            temp = asc2_1206[chr1][i];
        else if (size1 == 16)
            temp = asc2_1608[chr1][i];
        else if (size1 == 24)
            temp = asc2_2412[chr1][i];
        else return;

        for (m = 0; m < 8; m++)
        {
            if (temp & 0x01) OLED_DrawPoint(x, y, mode);
            else OLED_DrawPoint(x, y, !mode);
            temp >>= 1;
            y++;
        }
        x++;
        if ((size1 != 8) && ((x - x0) == size1 / 2))
        {
            x = x0;
            y0 = y0 + 8;
        }
        y = y0;
    }
}

/* 显示字符串 */
void OLED_ShowString(uint8_t x, uint8_t y, uint8_t *chr, uint8_t size1, uint8_t mode)
{
    while ((*chr >= ' ') && (*chr <= '~'))
    {
        OLED_ShowChar(x, y, *chr, size1, mode);
        if (size1 == 8) x += 6;
        else if (size1 == 12) x += 6;
        else x += size1 / 2;
        chr++;
    }
}

/* m^n */
static uint32_t OLED_Pow(uint8_t m, uint8_t n)
{
    uint32_t result = 1;
    while (n--) result *= m;
    return result;
}

/* 显示数字 */
void OLED_ShowNum(uint8_t x, uint8_t y, uint32_t num, uint8_t len, uint8_t size1, uint8_t mode)
{
    uint8_t t, temp, m = 0;
    if (size1 == 8) m = 2;
    for (t = 0; t < len; t++)
    {
        temp = (num / OLED_Pow(10, len - t - 1)) % 10;
        if (temp == 0)
            OLED_ShowChar(x + (size1 / 2 + m) * t, y, '0', size1, mode);
        else
            OLED_ShowChar(x + (size1 / 2 + m) * t, y, temp + '0', size1, mode);
    }
}

/* 显示浮点数 */
void OLED_ShowFloat(uint8_t x, uint8_t y, float num, uint8_t intLen, uint8_t decLen, uint8_t size1, uint8_t mode)
{
    int intPart = (int)num;
    if (intPart < 0) intPart = -intPart;
    uint32_t decPart = (uint32_t)((num > 0 ? num - (int)num : (int)num - num) * OLED_Pow(10, decLen) + 0.5f);
    uint8_t step = (size1 == 8) ? 6 : (size1 / 2);

    /* 负号或空格 */
    if (num < 0) { OLED_ShowChar(x, y, '-', size1, mode); }
    x += step;

    /* 整数部分 */
    OLED_ShowNum(x, y, (uint32_t)intPart, intLen, size1, mode);
    x += step * intLen;

    /* 小数点 */
    OLED_ShowChar(x, y, '.', size1, mode);
    x += step;

    /* 小数部分 */
    OLED_ShowNum(x, y, decPart, decLen, size1, mode);
}

/* ====== 扩展功能 ====== */

/* 显示汉字（依赖 oledfont.h 中的 Hzk1/Hzk2/Hzk3/Hzk4 字库） */
void OLED_ShowChinese(uint8_t x, uint8_t y, uint8_t num, uint8_t size1, uint8_t mode)
{
    uint8_t m, temp;
    uint8_t x0 = x, y0 = y;
    uint16_t i, size3 = (size1 / 8 + ((size1 % 8) ? 1 : 0)) * size1;

    for (i = 0; i < size3; i++)
    {
        if (size1 == 16)
            temp = Hzk1[num][i];
        else if (size1 == 24)
            temp = Hzk2[num][i];
        else if (size1 == 32)
            temp = Hzk3[num][i];
        else if (size1 == 64)
            temp = Hzk4[num][i];
        else return;

        for (m = 0; m < 8; m++)
        {
            if (temp & 0x01) OLED_DrawPoint(x, y, mode);
            else OLED_DrawPoint(x, y, !mode);
            temp >>= 1;
            y++;
        }
        x++;
        if ((x - x0) == size1)
        {
            x = x0;
            y0 = y0 + 8;
        }
        y = y0;
    }
}

/* 显示图片 */
void OLED_ShowPicture(uint8_t x, uint8_t y, uint8_t sizex, uint8_t sizey, uint8_t BMP[], uint8_t mode)
{
    uint16_t j = 0;
    uint8_t i, n, temp, m;
    uint8_t x0 = x, y0 = y;

    sizey = sizey / 8 + ((sizey % 8) ? 1 : 0);
    for (n = 0; n < sizey; n++)
    {
        for (i = 0; i < sizex; i++)
        {
            temp = BMP[j];
            j++;
            for (m = 0; m < 8; m++)
            {
                if (temp & 0x01) OLED_DrawPoint(x, y, mode);
                else OLED_DrawPoint(x, y, !mode);
                temp >>= 1;
                y++;
            }
            x++;
            if ((x - x0) == sizex)
            {
                x = x0;
                y0 = y0 + 8;
            }
            y = y0;
        }
    }
    OLED_Refresh();
}

/* 滚动显示 */
void OLED_ScrollDisplay(uint8_t num, uint8_t space, uint8_t mode)
{
    uint8_t i, n, t = 0, m = 0, r;
    while (1)
    {
        if (m == 0)
        {
            OLED_ShowChinese(128, 24, t, 16, mode);
            t++;
        }
        if (t == num)
        {
            for (r = 0; r < 16 * space; r++)
            {
                for (i = 1; i < 144; i++)
                {
                    for (n = 0; n < 8; n++)
                    {
                        OLED_GRAM[i - 1][n] = OLED_GRAM[i][n];
                    }
                }
                OLED_Refresh();
            }
            t = 0;
        }
        m++;
        if (m == 16) { m = 0; }
        for (i = 1; i < 144; i++)
        {
            for (n = 0; n < 8; n++)
            {
                OLED_GRAM[i - 1][n] = OLED_GRAM[i][n];
            }
        }
        OLED_Refresh();
    }
}
