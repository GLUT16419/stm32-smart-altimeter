#include "bsp_bmp280.h"
#include "stm32f4xx_hal.h"
#include "stdio.h"
#include <math.h>

#define BMP280_SPI_SCK_PIN GPIO_PIN_8
#define BMP280_SPI_SDI_PIN GPIO_PIN_9
#define BMP280_SPI_SDO_PIN GPIO_PIN_10
#define BMP280_SPI_SDO_PORT GPIOC
#define BMP280_SPI_CSB_PIN GPIO_PIN_8
#define BMP280_SPI_CSB_PORT GPIOC
#define BMP280_SPI_PORT GPIOB

#define BMP280_SPI_SCK(x) HAL_GPIO_WritePin(BMP280_SPI_PORT, BMP280_SPI_SCK_PIN, (x ? GPIO_PIN_SET : GPIO_PIN_RESET))
#define BMP280_SPI_SDI(x) HAL_GPIO_WritePin(BMP280_SPI_PORT, BMP280_SPI_SDI_PIN, (x ? GPIO_PIN_SET : GPIO_PIN_RESET))
#define BMP280_SPI_SDO() HAL_GPIO_ReadPin(BMP280_SPI_SDO_PORT, BMP280_SPI_SDO_PIN)
#define BMP280_SPI_CSB(x) HAL_GPIO_WritePin(BMP280_SPI_CSB_PORT, BMP280_SPI_CSB_PIN, (x ? GPIO_PIN_SET : GPIO_PIN_RESET))

static void bmp280_delay_us(uint32_t us)
{
    uint32_t ticks = us * (SystemCoreClock / 1000000) / 5;
    while(ticks--) { __NOP(); }
}

#define BMP280_PRESSURE_OSR (BMP280_OVERSAMP_8X)
#define BMP280_TEMPERATURE_OSR (BMP280_OVERSAMP_16X)
#define BMP280_MODE_VAL (BMP280_PRESSURE_OSR << 2 | BMP280_TEMPERATURE_OSR << 5 | BMP280_NORMAL_MODE)

/* ===== Sahixi 硬件 IIR 系数（移植自 参考文档/sahixi/code） =====
 * Sahixi 实际写入: BMP280_IIR_FILTER = BMP280_IIR_4(0x04)，再 (<<2) → 0x10
 *   → CONFIG bits[4:2]=0b100=4 → 数据手册系数 16。
 * 注意: sahixi 的宏名 "IIR_4" 与其位移存在歧义，但按其实测寄存器行为，
 *       硬件系数为 16（而非 4）。此处严格复现 sahixi 的实测效果。
 *       如需改为 sahixi 宏名本意的"系数4"，把本值改为 2 即可。 */
#define BMP280_SAHIXI_IIR_COEFF   4   /* CONFIG bits[4:2] 值: 4→系数16, 3→系数8, 2→系数4 */

/* ===== Sahixi 软件气压滤波参数（移植自 参考文档/sahixi/code/Src/BMP280.c） =====
 * FILTER_NUM: 5 点滑动平均窗口; FILTER_A: 突变拒绝阈值(hPa)
 * 与上一"已接受"样本偏差 > FILTER_A 视为野值，拒绝入窗（仅做平均、不更新窗）。 */
#define SAHIXI_FILTER_NUM   5
#define SAHIXI_FILTER_A     0.1f

static BMP280_Calib_TypeDef bmp280Cal;
static bool bmp280_isInit = false;

void BMP280_SPI_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();

    GPIO_InitStruct.Pin = BMP280_SPI_SCK_PIN | BMP280_SPI_SDI_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(BMP280_SPI_PORT, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = BMP280_SPI_CSB_PIN;
    HAL_GPIO_Init(BMP280_SPI_CSB_PORT, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = BMP280_SPI_SDO_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(BMP280_SPI_SDO_PORT, &GPIO_InitStruct);

    BMP280_SPI_CSB(1);
    BMP280_SPI_SCK(0);
    BMP280_SPI_SDI(0);
    
    printf("DBG,BMP280,SPI_Init,SCK=%d,SDI=%d,SDO=%d,CSB=%d\r\n", 
           HAL_GPIO_ReadPin(BMP280_SPI_PORT, BMP280_SPI_SCK_PIN),
           HAL_GPIO_ReadPin(BMP280_SPI_PORT, BMP280_SPI_SDI_PIN),
           HAL_GPIO_ReadPin(BMP280_SPI_SDO_PORT, BMP280_SPI_SDO_PIN),
           HAL_GPIO_ReadPin(BMP280_SPI_CSB_PORT, BMP280_SPI_CSB_PIN));
}

uint8_t BMP280_SPI_ReadWriteByte(uint8_t dat)
{
    uint8_t i, receive = 0;
    
    for(i = 0; i < 8; i++)
    {
        BMP280_SPI_SDI((dat & 0x80) >> 7);
        dat <<= 1;
        bmp280_delay_us(2);
        BMP280_SPI_SCK(1);
        bmp280_delay_us(2);
        receive <<= 1;
        if(BMP280_SPI_SDO()) receive |= 1;
        BMP280_SPI_SCK(0);
        bmp280_delay_us(2);
    }
    
    return receive;
}

void BMP280_WriteReg(uint8_t reg, uint8_t value)
{
    uint8_t spi_reg = reg & 0x7F;
    
    BMP280_SPI_CSB(0);
    bmp280_delay_us(2);
    BMP280_SPI_ReadWriteByte(spi_reg);
    BMP280_SPI_ReadWriteByte(value);
    bmp280_delay_us(2);
    BMP280_SPI_CSB(1);
}

uint8_t BMP280_ReadReg(uint8_t reg)
{
    uint8_t spi_reg = (reg & 0x7F) | 0x80;
    uint8_t value;
    
    BMP280_SPI_CSB(0);
    bmp280_delay_us(2);
    BMP280_SPI_ReadWriteByte(spi_reg);
    value = BMP280_SPI_ReadWriteByte(0x00);
    bmp280_delay_us(2);
    BMP280_SPI_CSB(1);
    
    printf("DBG,BMP280,Reg=0x%02X,SPI_Reg=0x%02X,Value=0x%02X\r\n", reg, spi_reg, value);
    return value;
}

void BMP280_ReadRegs(uint8_t reg, uint8_t *buf, uint8_t len)
{
    uint8_t spi_reg = (reg & 0x7F) | 0x80;
    uint8_t i;
    
    BMP280_SPI_CSB(0);
    bmp280_delay_us(2);
    BMP280_SPI_ReadWriteByte(spi_reg);
    
    for(i = 0; i < len; i++)
    {
        buf[i] = BMP280_SPI_ReadWriteByte(0x00);
    }
    
    bmp280_delay_us(2);
    BMP280_SPI_CSB(1);
}

/* ----------------------------------------------------------------------------
 * Sahixi 软件气压滤波：5点滑动平均 + 突变拒绝（移植自 参考文档/sahixi/code/Src/BMP280.c）
 * 直接在 BMP280_Read_Data 内对气压(mPa→hPa)做与 sahixi 完全一致的处理，
 * 使方案20 的传感器层 conditioning 与 sahixi 原始管线对齐（硬件IIR + 本软件滤波 + EKF/BP）。
 *
 * 注意：输入/输出单位均为 hPa（与 sahixi 原始实现一致），调用方负责 Pa<->hPa 换算。
 * -------------------------------------------------------------------------- */
static void sahixi_pressure_filter(float *in_hpa, float *out_hpa)
{
    static uint8_t i = 0;
    static float   buf[SAHIXI_FILTER_NUM] = {0.0f};
    float          sum = 0.0f;
    uint8_t        c;
    float          delta;

    /* 首次填充：缓冲区尚未初始化时直接写入并返回 */
    if (buf[i] == 0.0f) {
        buf[i] = *in_hpa;
        *out_hpa = *in_hpa;
        if (++i >= SAHIXI_FILTER_NUM) i = 0;
        return;
    }

    /* 突变拒绝：与"上一已接受样本"比较，偏差过大则视为野值，不入窗 */
    delta = (i ? (*in_hpa - buf[i - 1]) : (*in_hpa - buf[SAHIXI_FILTER_NUM - 1]));
    if (fabsf(delta) < SAHIXI_FILTER_A) {
        buf[i] = *in_hpa;
        if (++i >= SAHIXI_FILTER_NUM) i = 0;
    }

    /* 5 点滑动平均 */
    for (c = 0; c < SAHIXI_FILTER_NUM; c++) {
        sum += buf[c];
    }
    *out_hpa = sum / (float)SAHIXI_FILTER_NUM;
}

bool BMP280_Init(void)
{
    uint8_t chip_id;

    BMP280_SPI_Init();
    HAL_Delay(20);

    chip_id = BMP280_ReadReg(BMP280_CHIP_ID_REG);
    printf("DBG,BMP280,ChipID=0x%02X\r\n", chip_id);
    
    if(chip_id != BMP280_DEFAULT_CHIP_ID)
    {
        return false;
    }

    BMP280_WriteReg(BMP280_RST_REG, 0xB6);
    HAL_Delay(2);

    uint8_t calib_data[24];
    BMP280_ReadRegs(BMP280_TEMPERATURE_CALIB_DIG_T1_LSB_REG, calib_data, 24);

    bmp280Cal.dig_T1 = (uint16_t)((calib_data[1] << 8) | calib_data[0]);
    bmp280Cal.dig_T2 = (int16_t)((calib_data[3] << 8) | calib_data[2]);
    bmp280Cal.dig_T3 = (int16_t)((calib_data[5] << 8) | calib_data[4]);
    bmp280Cal.dig_P1 = (uint16_t)((calib_data[7] << 8) | calib_data[6]);
    bmp280Cal.dig_P2 = (int16_t)((calib_data[9] << 8) | calib_data[8]);
    bmp280Cal.dig_P3 = (int16_t)((calib_data[11] << 8) | calib_data[10]);
    bmp280Cal.dig_P4 = (int16_t)((calib_data[13] << 8) | calib_data[12]);
    bmp280Cal.dig_P5 = (int16_t)((calib_data[15] << 8) | calib_data[14]);
    bmp280Cal.dig_P6 = (int16_t)((calib_data[17] << 8) | calib_data[16]);
    bmp280Cal.dig_P7 = (int16_t)((calib_data[19] << 8) | calib_data[18]);
    bmp280Cal.dig_P8 = (int16_t)((calib_data[21] << 8) | calib_data[20]);
    bmp280Cal.dig_P9 = (int16_t)((calib_data[23] << 8) | calib_data[22]);

    BMP280_WriteReg(BMP280_CTRL_MEAS_REG, BMP280_MODE_VAL);
    // CONFIG_REG: [7:5] t_sb=001(62.5ms), [4:2] filter= Sahixi 系数16
    // 移植自 sahixi：硬件 IIR 系数 = 16（CONFIG bits[4:2]=0b100=4），配合下方
    // BMP280_Read_Data 内的 5点滑动平均+突变拒绝软件滤波，构成 sahixi 完整传感器层 conditioning。
    // ⚠ 提示：此前本工程为抑制动态高度滞后曾把硬件系数降到 8（群延迟≈0.6s）。
    //   现按"继续移植 sahixi"的要求改回 16（群延迟≈1.3s）。软件滤波在运动段会
    //   因突变拒绝而冻结输出，动态跟手性依赖上层 EKF/BP 补偿——与 sahixi 原管线一致。
    //   若实测动态滞后不可接受，将 BMP280_SAHIXI_IIR_COEFF 改回 3（系数8）或 2（系数4）即可。
    BMP280_WriteReg(BMP280_CONFIG_REG, (1 << 5) | (BMP280_SAHIXI_IIR_COEFF << 2));   /* filter=4 → 系数16（Sahixi 实测值） */

    bmp280_isInit = true;
    return true;
}

void BMP280_Read_Data(float *pressure_pa, float *temperature_c)
{
    if(!bmp280_isInit) return;

    uint8_t data[6];
    BMP280_ReadRegs(BMP280_PRESSURE_MSB_REG, data, 6);

    uint32_t adc_p = ((((uint32_t)data[0]) << 12) | (((uint32_t)data[1]) << 4) | (((uint32_t)data[2]) >> 4));
    uint32_t adc_t = ((((uint32_t)data[3]) << 12) | (((uint32_t)data[4]) << 4) | (((uint32_t)data[5]) >> 4));

    int64_t var1, var2, T;
    var1 = ((((int64_t)adc_t >> 3) - ((int64_t)bmp280Cal.dig_T1 << 1))) * ((int64_t)bmp280Cal.dig_T2) >> 11;
    var2 = (((((int64_t)adc_t >> 4) - ((int64_t)bmp280Cal.dig_T1)) * (((int64_t)adc_t >> 4) - ((int64_t)bmp280Cal.dig_T1))) >> 12) * ((int64_t)bmp280Cal.dig_T3) >> 14;
    bmp280Cal.t_fine = var1 + var2;
    T = (bmp280Cal.t_fine * 5 + 128) >> 8;
    *temperature_c = (float)T / 100.0f;

    int64_t p_var1, p_var2, p;
    p_var1 = ((int64_t)bmp280Cal.t_fine) - 128000;
    p_var2 = p_var1 * p_var1 * (int64_t)bmp280Cal.dig_P6;
    p_var2 = p_var2 + ((p_var1 * (int64_t)bmp280Cal.dig_P5) << 17);
    p_var2 = p_var2 + (((int64_t)bmp280Cal.dig_P4) << 35);
    p_var1 = ((p_var1 * p_var1 * (int64_t)bmp280Cal.dig_P3) >> 8) + ((p_var1 * (int64_t)bmp280Cal.dig_P2) << 12);
    p_var1 = (((((int64_t)1) << 47) + p_var1)) * ((int64_t)bmp280Cal.dig_P1) >> 33;

    if(p_var1 == 0)
    {
        *pressure_pa = 0.0f;
        return;
    }

    p = 1048576 - adc_p;
    p = (((p << 31) - p_var2) * 3125) / p_var1;
    p_var1 = (((int64_t)bmp280Cal.dig_P9) * (p >> 13) * (p >> 13)) >> 25;
    p_var2 = (((int64_t)bmp280Cal.dig_P8) * p) >> 19;
    p = ((p + p_var1 + p_var2) >> 8) + (((int64_t)bmp280Cal.dig_P7) << 4);

    /* ---- Sahixi 软件气压滤波（移植）：先把 Pa 转 hPa 滤波，再转回 Pa ---- */
    float p_hpa = (float)p / 25600.0f;          /* Q24.8 ÷25600 = hPa */
    sahixi_pressure_filter(&p_hpa, &p_hpa);
    *pressure_pa = p_hpa * 100.0f;
}
