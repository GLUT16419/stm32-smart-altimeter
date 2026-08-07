#include "bsp_ms5611.h"
#include "stm32f4xx_hal.h"
#include "stdio.h"

uint16_t Cal_C1_6[8];

static void delay_us(uint32_t us)
{
    uint32_t ticks = us * (SystemCoreClock / 1000000) / 5;
    while(ticks--) {
        __NOP();
    }
}

/* ========== 软件 SPI 底层函数 ========== */

/* SPI 读写一个字节（MS5611 是 SPI 模式 0：CPOL=0, CPHA=0） */
static uint8_t MS5611_SPI_ReadWriteByte(uint8_t dat)
{
    uint8_t i, receive = 0;

    for(i = 0; i < 8; i++)
    {
        /* 在 SCK 上升沿之前设置 MOSI */
        MS5611_SPI_SDI((dat & 0x80) >> 7);
        dat <<= 1;
        delay_us(2);

        /* SCK 上升沿：主机发送数据，从机同时发送数据 */
        MS5611_SPI_SCLK(1);
        delay_us(2);

        /* 在 SCK 下降沿采样 MISO */
        receive <<= 1;
        if(MS5611_SPI_SDO_GET()) receive |= 1;

        MS5611_SPI_SCLK(0);
        delay_us(2);
    }

    return receive;
}

/* ========== MS5611 SPI 命令封装 ========== */

/* 发送 ADC 转换命令并等待转换完成 */
static void MS5611_StartConversion(uint8_t cmd)
{
    MS5611_SPI_CSB(0);
    delay_us(2);
    MS5611_SPI_ReadWriteByte(cmd);
    delay_us(2);
    MS5611_SPI_CSB(1);
}

/* 读取 PROM 中的一个字（2 字节） */
static uint16_t MS5611_ReadPROMWord(uint8_t addr)
{
    uint8_t data_H = 0, data_L = 0;
    uint8_t cmd = 0xA0 + addr * 2;

    MS5611_SPI_CSB(0);
    delay_us(2);
    MS5611_SPI_ReadWriteByte(cmd);
    delay_us(2);
    data_H = MS5611_SPI_ReadWriteByte(0x00);
    data_L = MS5611_SPI_ReadWriteByte(0x00);
    delay_us(2);
    MS5611_SPI_CSB(1);

    return (uint16_t)((data_H << 8) | data_L);
}

/* 读取 ADC 结果（先发 0x00 命令，再读 3 字节，CS 全程保持低电平） */
static uint32_t MS5611_ReadADC(void)
{
    uint8_t buff[3] = {0};

    MS5611_SPI_CSB(0);
    delay_us(2);

    /* 发送读 ADC 命令 0x00 */
    MS5611_SPI_ReadWriteByte(0x00);
    delay_us(2);

    /* 连续读 3 字节 */
    buff[0] = MS5611_SPI_ReadWriteByte(0x00);
    buff[1] = MS5611_SPI_ReadWriteByte(0x00);
    buff[2] = MS5611_SPI_ReadWriteByte(0x00);

    delay_us(2);
    MS5611_SPI_CSB(1);

    return (((uint32_t)buff[0] << 16) | ((uint32_t)buff[1] << 8) | buff[2]);
}

/* ========== 初始化 ========== */

void MS5611_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOB_CLK_ENABLE();

    /* SCLK：推挽输出 */
    GPIO_InitStruct.Pin = MS5611_SPI_SCLK_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(MS5611_SPI_PORT, &GPIO_InitStruct);

    /* SDI (MOSI)：推挽输出 */
    GPIO_InitStruct.Pin = MS5611_SPI_SDI_PIN;
    HAL_GPIO_Init(MS5611_SPI_PORT, &GPIO_InitStruct);

    /* CSB：推挽输出 */
    GPIO_InitStruct.Pin = MS5611_SPI_CSB_PIN;
    HAL_GPIO_Init(MS5611_SPI_PORT, &GPIO_InitStruct);

    /* SDO (MISO)：输入 */
    GPIO_InitStruct.Pin = MS5611_SPI_SDO_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(MS5611_SPI_PORT, &GPIO_InitStruct);

    /* 初始状态：CSB 高电平（未选中），SCLK 低电平 */
    MS5611_SPI_CSB(1);
    MS5611_SPI_SCLK(0);
    MS5611_SPI_SDI(0);

    printf("DBG,MS5611,SPI_Init,SCLK=PB6,SDI=PB7,SDO=PB5,CSB=PB4\r\n");
}

char MS5611_Reset(void)
{
    MS5611_SPI_CSB(0);
    delay_us(5);
    MS5611_SPI_ReadWriteByte(0x1E);
    delay_us(5);
    MS5611_SPI_CSB(1);

    /* 等待复位完成（MS5611 复位需要 2.8ms） */
    HAL_Delay(5);

    return 0;
}

void MS5611_Read_PROM(void)
{
    uint8_t i = 0;

    for(i = 0; i < 8; i++)
    {
        Cal_C1_6[i] = MS5611_ReadPROMWord(i);
    }

    printf("DBG,MS5611,PROM,C1=%u,C2=%u,C3=%u,C4=%u,C5=%u,C6=%u\r\n",
           Cal_C1_6[1], Cal_C1_6[2], Cal_C1_6[3],
           Cal_C1_6[4], Cal_C1_6[5], Cal_C1_6[6]);
}

/* ========== 温度/压力读取 ========== */

uint32_t D1 = 0, D2 = 0;
int64_t dT = 0;

float MS5611_Get_Temperature(void)
{
    float dat = 0;
    int64_t TEMP = 0;

    /* 启动 D1 (压力) 转换，OSR=1024（提高采样率至 50Hz 匹配仿真） */
    MS5611_StartConversion(0x44);
    HAL_Delay(3);

    D1 = MS5611_ReadADC();

    /* 启动 D2 (温度) 转换，OSR=1024 */
    MS5611_StartConversion(0x54);
    HAL_Delay(3);

    D2 = MS5611_ReadADC();

    dT = (int64_t)D2 - ((int64_t)Cal_C1_6[5] << 8);
    TEMP = 2000 + ((dT * Cal_C1_6[6]) >> 23);

    dat = (float)TEMP / 100.0f;
    return dat;
}

float MS5611_Get_Pressure(void)
{
    float pressure, temperature;
    MS5611_Read_Data(&pressure, &temperature);
    return pressure;
}

void MS5611_Read_Data(float *pressure_pa, float *temperature_c)
{
    uint32_t d1, d2;
    int64_t dt, TEMP, OFF, SENS, P;

    /* 启动 D1 (压力) 转换，OSR=1024（提高采样率至 50Hz 匹配仿真） */
    MS5611_StartConversion(0x44);
    HAL_Delay(3);
    d1 = MS5611_ReadADC();

    /* 启动 D2 (温度) 转换，OSR=1024 */
    MS5611_StartConversion(0x54);
    HAL_Delay(3);
    d2 = MS5611_ReadADC();

    /* 温度补偿计算 */
    dt = (int64_t)d2 - ((int64_t)Cal_C1_6[5] << 8);
    TEMP = 2000 + ((dt * Cal_C1_6[6]) >> 23);

    OFF = ((int64_t)Cal_C1_6[2] << 16) + ((Cal_C1_6[4] * dt) >> 7);
    SENS = ((int64_t)Cal_C1_6[1] << 15) + ((Cal_C1_6[3] * dt) >> 8);

    P = (((d1 * SENS) >> 21) - OFF) >> 15;
    *pressure_pa = (float)P;           /* MS5611 输出单位是 Pa */
    *temperature_c = (float)TEMP / 100.0f;
}
