/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "cmsis_os.h"
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "bsp_ms5611.h"
#include "bsp_bmp280.h"
#include "altitude_convert.h"
#if WORK_MODE == 0
#include "altitude_ekf.h"
#include "ssd1306.h"
#include "baro_ekf.h"
#include "bp_denoise.h"
#endif
#include "kalman_filter.h"
#include "stdio.h"
#include <stdbool.h>  
#include <string.h>
#include <math.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* 融合权重：MS5611 占比 80%，BMP280 占比 20% */
#define FUSION_WEIGHT_MS5611_DEFAULT 0.5f
#define FUSION_WEIGHT_BMP280_DEFAULT 0.5f

/* 融合方案选择：
 * 1 = 方案1：单传感器内 NN(50%) + KF(50%) 再加权 → 双传感器融合(80% MS5611 : 20% BMP280)
 * 2 = 方案2：四路直接加权融合（MS5611_NN×w1 + MS5611_KF×w2 + BMP280_NN×w3 + BMP280_KF×w4）
 * 3 = 方案3：KF 对 NN 输出做二次滤波 → 双传感器融合(80% MS5611 : 20% BMP280)
 * 4 = 方案4：BMP280 主力 + MS5611 高频增强通道 — BMP280 定绝对精度，MS5611 补动态响应
 * 5 = 方案5：自适应权重融合 — 静止 BMP280(95%)，运动 MS5611(40%)，自动平滑过渡
 * 6 = 方案6：MS5611 主导（不使用 NN 滤波）— MS5611 KF(85%) + BMP280 KF(15%)，无 NN
 * 7 = 方案7：BMP280 定绝对高度，MS5611 与 BMP280 的高度变化量加权融合后叠加
 * 8 = 方案8：只使用 MS5611 的 KF 值（纯 MS5611，无 BMP280 气压融合）
 * 9 = 方案9：只使用 BMP280 的 NN 和 KF（权重 70% NN + 30% KF）
 * 10= 方案10（创新）：逆方差加权融合 — 实时方差决定权重，MS5611 跳变时自动归零
 * 11= 方案11（Delta 置信度加权累积）：融合帧间气压变化量（Delta），配置信度加权和泄漏锚防漂移
 * 12= 方案12（Hampel 脉冲抑制预处理器）：用中值绝对偏差检测替换离群值，抗偶发脉冲噪声
 * 13= 方案13（二阶互补融合）：气压 + 气压变化率双维度互补，兼顾低噪声和快速响应
 * 14= 方案14（方案4 + BMP280 温漂补偿）：基于方案4，新增实时温度漂移线性补偿，
 *     每帧根据温度偏离校准参考值的幅度修正 BMP280 气压，抑制温循波动
 * 15= 方案15（NN 主导场景门控增量锁定）：联合多任务模型 NN 降噪(MS5611/BMP280) +
 *     NN 场景概率(Schmitt 门控) + 逆方差置信加权 + 门控积分；静止锁死、升降积分。
 * 16= 方案16（KF 主导场景门控增量锁定）：自适应卡尔曼滤波降噪(ht_ms_kf/ht_bmp_kf) +
 *     KF 自身 Δh 幅度(|Δh| EMA + Schmitt)判定场景 + 逆方差加权 + 门控积分；
 *     不依赖任何 NN（纯 KF 主导，适用于算力受限 MCU）。
 *     —— 方案 15/16 结构同构，仅降噪与场景信号来源不同(NN vs KF)，调参参数见
 *        fusion_scheme_15_16_params.h（由 altimeter_tuner/compare_s16.py 自动生成）。
 */

/* 注：FUSION_SCHEME 与各融合/全局参数已由 main.h 引入
 * （含 fusion_scheme_tuned_params.h，方案 1-14 两级调参结果），此处不再重复定义。 */

/* EMA 显示平滑系数：方案15/16/17 用 compare_s16.py 调参值；
 * 其余方案（含 1-14）统一用 tune_all_params.py 两级调参的最优共享 EMA 值。 */
#if FUSION_SCHEME == 15 || FUSION_SCHEME == 16 || FUSION_SCHEME == 17
#define PRESSURE_EMA_ALPHA  S15S16_PRESSURE_EMA_ALPHA
#define HEIGHT_EMA_ALPHA    S15S16_HEIGHT_EMA_ALPHA
#else
#define PRESSURE_EMA_ALPHA  TUNED_PRESSURE_EMA_ALPHA
#define HEIGHT_EMA_ALPHA    TUNED_HEIGHT_EMA_ALPHA
#endif

typedef struct {
    float pressure_pa;
    float temperature_c;
    float height_m;
    float pressure_filtered_pa;
    float height_filtered_m;
    float pressure_filtered_nn;
    float height_filtered_nn;
} SensorData_t;

typedef struct {
    float pressure_fused_pa;    /* NN滤波后融合气压（最终输出） */
    float height_fused_m;       /* 由融合气压通过公式算出的高度（最终输出） */
    float temperature_c;        /* BMP280 温度作为融合温度 */
} FusionData_t;

/* ========== 方案20/21：sahixi 气压域 EKF + BP 去噪（B1 多页对照显示） ========== */
#if FUSION_SCHEME == 20 || FUSION_SCHEME == 21 || FUSION_SCHEME == 22
typedef struct {
    float ms_ekf_pa, ms_ekf_alt, ms_bp_pa, ms_bp_alt;
    float bm_ekf_pa, bm_ekf_alt, bm_bp_pa, bm_bp_alt;
    float ms_temp, bm_temp;
#if FUSION_SCHEME == 21 || FUSION_SCHEME == 22
    float fused_pa, fused_alt, fused_temp;   /* 方案21/22：双 EKF 方差倒数加权融合 */
#endif
} S20_Out_t;
static BaroEKF_TypeDef s20_ms_ekf, s20_bm_ekf;
static BP_Denoise_t     s20_ms_bp, s20_bm_bp;
static S20_Out_t        s20_out;
static int   s20_inited = 0;
static int   g_s20_page = 0;
static uint32_t s20_prev_tick = 0;
#endif

/* ========== 新算法方案：多任务联合模型 (baseline_eqw) ==========
 * 单一模型：双输入(各 10 点相对气压) -> 三输出(MS5611 滤波 / BMP280 滤波 / 场景概率)
 * 替代原"单传感器滤波模型 + 独立场景分类器"两套方案：一次推理同时完成滤波与场景识别，
 * 部署体积更小、调度更简单（细节见 altimeter_tuner/multitask_infer.py 与
 * models/compare_v2/2_baseline_eqw.json）。
 */
#define MT_WINDOW           10
#define MT_INPUT_DIM        20
#define MT_REF_PRESSURE    101325.0f
/* 回归目标反归一化 (StandardScaler，来自 2_baseline_eqw.json) */
#define MT_TARGET_MEAN     (-2741.4666f)
#define MT_TARGET_STD      (26.0931f)
/* 输入特征标准化参数 (StandardScaler, 20 维: 前 10 = MS5611, 后 10 = BMP280) */
static const float MT_FEAT_MEAN[MT_INPUT_DIM] = {
    -2741.7959f, -2741.8234f, -2741.8513f, -2741.8883f, -2741.8743f,
    -2741.8868f, -2741.9126f, -2741.9269f, -2741.9237f, -2741.9403f,
    -2741.1656f, -2741.1770f, -2741.1881f, -2741.1959f, -2741.2013f,
    -2741.2092f, -2741.2185f, -2741.2236f, -2741.2283f, -2741.2276f
};
static const float MT_FEAT_STD[MT_INPUT_DIM] = {
    22.2062f, 22.1954f, 22.1903f, 22.1859f, 22.1823f,
    22.1779f, 22.1691f, 22.1663f, 22.1733f, 22.1719f,
    30.1566f, 30.1561f, 30.1550f, 30.1542f, 30.1537f,
    30.1520f, 30.1499f, 30.1489f, 30.1478f, 30.1484f
};

typedef struct {
    float win_ms[MT_WINDOW];   /* MS5611 相对气压滑动窗口（绝对气压 - 101325） */
    float win_bmp[MT_WINDOW];  /* BMP280 相对气压滑动窗口 */
    int   idx;                 /* 环形写入位置（双窗口共用） */
    int   n_ms;                /* MS5611 已采集样本数 */
    int   n_bmp;               /* BMP280 已采集样本数 */
} MultitaskModel_t;

/* 模型运行态输出（绝对气压 Pa / 场景概率），由 BMP280 任务推理后更新 */
void Multitask_Init(float init_ms, float init_bmp);
void Multitask_PushMS(float ms_pa);
void Multitask_PushBMP(float bmp_pa);
bool Multitask_Run(void);
const char* Multitask_SceneName(int pred);


/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define SAMPLE_PERIOD_MS 20

/* NN 归一化说明（新方案 baseline_eqw，StandardScaler 而非旧的 MinMaxScaler）：
 * 模型输入 = 双路相对气压窗口标准化后拼接：
 *   in[i]   = (MS5611_rel[i]  - MT_FEAT_MEAN[i])   / MT_FEAT_STD[i]      (i=0..9)
 *   in[10+i]= (BMP280_rel[i]  - MT_FEAT_MEAN[10+i]) / MT_FEAT_STD[10+i]  (i=0..9)
 *   其中 rel = raw_pressure - MT_REF_PRESSURE(101325)
 * 模型三头输出：滤波MS / 滤波BMP（归一化相对气压，反归一化 *MT_TARGET_STD+MT_TARGET_MEAN+REF）
 *            + 场景 [static, elevation]（softmax 概率）。
 * 全部参数定义见上方 PTD 区 MT_* 宏与 MT_FEAT_* 数组（来自 2_baseline_eqw.json）。 */

/* ========== FUSION_SCHEME 4, 5, 7, 10, 11, 12, 13 全局静态变量 ========== */
#if FUSION_SCHEME == 4
static float ms5611_hpf_last = 0.0f;  /* 高通滤波器上一帧输出 */
#elif FUSION_SCHEME == 5
static float motion_window[MOTION_WINDOW_SIZE];  /* 运动检测滑动窗口 */
static int motion_window_idx = 0;
static bool motion_detected = false;
static float smooth_weight_ms = FUSION_WEIGHT_MS5611;  /* 平滑后的 MS5611 权重 */
#elif FUSION_SCHEME == 7
static float ms5611_height_prev = 0.0f;   /* MS5611 上一帧高度（KF 滤波后） */
static float bmp280_height_prev = 0.0f;   /* BMP280 上一帧高度（NN 滤波后） */
static float motion_window_7[MOTION_WINDOW_SIZE] = {0}; /* 运动检测滑动窗口 */
static int motion_window_idx_7 = 0;                     /* 窗口索引 */
static float smooth_delta_weight_ms = W_DELTA_MS_STATIC; /* 平滑后的 MS5611 delta 权重 */
static float prev_ms5611_p_7 = 0.0f;                    /* 上一帧 MS5611 KF 气压 */
#elif FUSION_SCHEME == 10
#define IVAR_WINDOW_SIZE 10
#define IVAR_EPSILON     TUNED_IVAR_EPSILON  /* 方案10 调参值（见 fusion_scheme_tuned_params.h） */
static float ivar_ms5611_buf[IVAR_WINDOW_SIZE];  /* 逆方差加权：MS5611 最近10帧气压 */
static float ivar_bmp280_buf[IVAR_WINDOW_SIZE];  /* 逆方差加权：BMP280 最近10帧气压 */
static int ivar_idx = 0;                          /* 循环缓冲区写入位置 */
static int ivar_filled = 0;                       /* 缓冲区已填充数 */
#elif FUSION_SCHEME == 11
static float delta11_prev_ms5611_p = 0.0f;       /* MS5611 KF 上一帧气压 */
static float delta11_prev_bmp280_p = 0.0f;       /* BMP280 NN 上一帧气压 */
static float delta11_ms_window[DELTA_CONF_WINDOW]; /* MS5611 Delta 滑动窗口 */
static float delta11_bmp_window[DELTA_CONF_WINDOW];/* BMP280 Delta 滑动窗口 */
static int delta11_idx = 0;                       /* Delta 窗口写入索引 */
static int delta11_filled = 0;                    /* Delta 窗口已填充数 */
static float delta11_fused_pa = 0.0f;             /* 累积融合气压 */
static bool delta11_first_run = true;             /* 首帧标记 */
#elif FUSION_SCHEME == 12
/* Hampel 滤波器各传感器滑动窗口 */
static float hampel12_ms5611_buf[HAMPEL_WINDOW_SIZE]; /* MS5611 滑动窗口 */
static int    hampel12_ms5611_idx = 0;                 /* 写入索引 */
static float hampel12_bmp280_buf[HAMPEL_WINDOW_SIZE]; /* BMP280 滑动窗口 */
static int    hampel12_bmp280_idx = 0;                 /* 写入索引 */
static bool   hampel12_filled = false;                 /* 窗口已填满 */
#elif FUSION_SCHEME == 13
static float so13_P_fused = 0.0f;                  /* 二阶互补：融合气压 */
static float so13_D_fused = 0.0f;                  /* 二阶互补：融合变化率 */
static float so13_prev_ms5611_p = 0.0f;            /* MS5611 上一帧 KF 气压 */
static bool  so13_first_run = true;                /* 首帧标记 */
#elif FUSION_SCHEME == 14
static float tc14_hpf_last = 0.0f;                 /* 高通滤波器上一帧输出 */
static float tc14_ref_temperature = 0.0f;           /* 温漂补偿：校准参考温度 */
static bool  tc14_ref_initialized = false;          /* 参考温度已初始化 */
#elif FUSION_SCHEME == 15
/* ========== 方案15：原始气压窗口方差门控 + 原始气压 ISA 相对高度 ========== */
static float h15_lock = 0.0f;              /* 相对高度锁定值 (m) */
static bool  gate15_state = false;         /* 门控状态（false=静止锁死, true=升降） */
static bool  first_run_15 = true;          /* 首帧标记 */
static int   s15_warmup = 0;               /* 启动稳定期计数（帧数） */
static bool  s15_locked = false;           /* 数据是否已稳定并锁定 */
/* 运动重锚：起始参考原始气压与高度（static→elevation 跳变时记录），
 * 用于以「运动起点气压 / 当前气压」两点 ISA 算相对高度（无 KF 滞后）。 */
static float s15_motion_ref_pressure = 0.0f;  /* 运动起点原始气压 (Pa) */
static float s15_motion_ref_height = 0.0f;    /* 运动起点锁定相对高度 (m) */
/* 方案15 调优参数（sim_scheme15 / sim_scheme15_v2 用 serial_tool/data/raw 真实数据调优） */
#define S15_WARMUP_FRAMES  64    /* 启动稳定帧数：上电/复位后等待传感器与 KF 稳定（v2 调优自 104，更跟手） */
#define S15_MOT_WIN        20    /* 长窗口帧数：STD 在此窗口内统计（捕捉持续/慢速运动） */
/* 原始气压长窗口（场景方差检测） */
static float s15_pbuf[S15_MOT_WIN];
static uint8_t s15_pbuf_idx = 0;
static uint8_t s15_pbuf_fill = 0;
/* 原始气压短窗口（快速小幅检测：捕捉 25cm 快速变化，对低频漂移免疫） */
static float s15_fbuf[S15_FAST_WIN];
static uint8_t s15_fbuf_idx = 0;
static uint8_t s15_fbuf_fill = 0;
static bool  s15_fast_gate = false;     /* 短窗口 STD 门控状态 */
static uint8_t s15_static_streak = 0;   /* 连续静止帧计数（用于重锚边沿判定） */
static uint8_t s15_settle_cnt = 0;      /* 沉降等待计数：门控转静止后继续积分帧数（延时锁定） */
#define S15_SETTLE_FRAMES  15           /* 判静止后再积分 N 帧（≈1.5s），待气压/门控完全沉降后再冻结，避免提前关门锁在半路值 */
#elif FUSION_SCHEME == 16
/* ========== 方案16：KF 主导场景门控增量锁定（纯 KF，不依赖 NN） ========== */
/* 场景门控改用「气压长/短窗口 STD（OR）」，与方案15 同源（sim_scheme15_v2.py 验证）。
 * 原方案16 用 KF 高度 Δh 幅度 EMA 判场景：慢速小幅运动时每帧偏移低于阈值(0.0118m)
 * → 门控永不打开 → 一直静态、高度不变（实测现象）。气压 STD 对慢速大幅/快速小幅均敏感，
 * 且对 KF 滤波/低频漂移稳健。高度计算仍为 KF 气压 → ISA 相对高度（KF 主导，无 NN）。 */
static float h16_lock = 0.0f;              /* 门控积分锁定的高度 */
static bool  gate16_state = false;         /* 长窗口 STD Schmitt 门控状态 */
static bool  s16_fast_gate = false;        /* 短窗口 STD Schmitt 门控状态（快速小幅） */
static bool  first_run_16 = true;          /* 首帧标记 */
/* 方案16 运动重锚：起始参考气压与高度（静止→升降跳变边沿时记录） */
static float s16_motion_ref_pressure = 0.0f;   /* 运动起始 KF 气压 (Pa) */
static float s16_motion_ref_height = 0.0f;     /* 运动起始锁定高度 (m) */
static uint8_t s16_static_streak = 0;     /* 连续静止帧计数（重锚边沿判定，防抖动重锚） */
#define S16_MOT_WIN  20                     /* 长窗口帧数（复用方案15 调优值，捕捉持续/慢速运动） */
/* 气压长/短窗口（场景方差检测，参数复用方案15 调优值） */
static float s16_pbuf[S16_MOT_WIN];
static uint8_t s16_pbuf_idx = 0;
static uint8_t s16_pbuf_fill = 0;
static float s16_fbuf[S15_FAST_WIN];
static uint8_t s16_fbuf_idx = 0;
static uint8_t s16_fbuf_fill = 0;
static float s16_rest_p = 0.0f;            /* 静息基准气压（仅确认静止时刷新，用于累计偏差判运动） */
static int   s16_stable_cnt = 0;           /* 启动稳定计数器（帧） */
#define S16_STARTUP_STABLE_FRAMES  48      /* 开机等待 48 帧 ≈ 4.8 秒后允许运动跟踪（原80偏长） */
static uint8_t s16_settle_cnt = 0;         /* 沉降等待计数：门控转静止后继续积分帧数（让 KF 追上真值再锁定） */
#define S16_SETTLE_FRAMES  20              /* 判静止后再积分 N 帧（≈2s），待 KF 气压追上真实静止值后再冻结，消除 KF 启动滞后欠读 */
#define S16_DELTA_EXIT_PA  2.0f            /* 原始气压相对静息基线累计偏差 >2Pa 即强制跳出静态（防 STD 门控始终不开门导致一直静态） */
#define S16_BASE_ALPHA     0.01f           /* 静息基线 EMA 系数：极小，仅跟随温漂/极慢漂移，跟不上真实运动（时间常数~100帧≈10s） */
#elif FUSION_SCHEME == 17
/* ===== 方案17：无场景模式 — 无需任何静态变量 ===== */
#endif

/* 全局 BMP280 偏置补偿：使 BMP280 气压向 MS5611 基准对齐，所有融合方案共用 */
static float bmp_bias_pa = 0.0f;                  /* 校准后初始化为 diff_mean (MS5611 - BMP280) */

/* 预设/校零重锚请求：切预设或 SET_ALT 后，将融合积分锁重新钉到新海拔，
 * 使 OLED 立即显示预设海拔，而非停留在旧锁值缓慢收敛（尤其方案15静止时锁冻结）。 */
static volatile bool  g_fusion_reanchor_req = false;
static float g_fusion_reanchor_alt = 0.0f;
static volatile bool  g_height_snap_req = false;   /* 重锚时请求 EMA 立即对齐到目标海拔（避免滑过） */

/* 相对高度基准（相对高度变化 U = height_fused_m - rel_height_ref）：
 * 上电自动重锚 / 手动切换预设时，由 ApplyAltPreset 直接钉到目标海拔，
 * 避免 OLED 在重锚前锁定基准，导致重锚后相对高度出现十几米恒定偏差。
 * 原实现仅在 OLED 首次有效读数时锁基准，而该时刻早于上电自动重锚，
 * 故基准被锁在热瞬态漂移后的高度，重锚后产生 ~15m 恒定相对高度误差。 */
static float rel_height_ref = 0.0f;
static volatile bool  rel_ref_set = false;

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart1;
UART_HandleTypeDef huart2;

/* Definitions for defaultTask */
osThreadId_t defaultTaskHandle;
const osThreadAttr_t defaultTask_attributes = {
  .name = "defaultTask",
  .stack_size = 128 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};
/* USER CODE BEGIN PV */
SensorData_t ms5611_data;
SensorData_t bmp280_data;
FusionData_t fusion_data;        /* 双传感器融合数据 */
float reference_pressure_pa = PRESET_P0_PA;  /* 方案18：默认即代码预设的参考气压 P0（可运行时用 SET_P0 微调） */
bool ms5611_ready = false;
bool bmp280_ready = false;

KalmanFilter_TypeDef ms5611_kf_pressure;
KalmanFilter_TypeDef bmp280_kf_pressure;

/* 增强校准结果 */
CalibResult_t calib_result;
bool calib_use_auto_params = true;  /* 是否使用自动调参结果 */

/* ========== 预设高度切换（PC13 按键循环选择） ==========
 * 每个预设对应一个已知海拔点：按 PC13 循环切换，切换时用当前 MS5611 实测气压
 * 重算参考气压（ISA 公式），使输出高度对齐到该点的真实海拔。OLED 只显示 ID。 */
typedef struct { int id; float altitude_m; } AltPreset_t;
static const AltPreset_t g_alt_presets[] = {
    {    1, 168.000f },   /* 初始预设高度 168 m */
    {  102, 153.430f },
    {  101, 153.846f },
    { 1128, 155.788f },
    { 1136, 158.731f },
    { 1134, 159.006f },
    {  108, 155.773f },
    {  110, 155.973f },
    { 1115, 156.927f },
    { 1121, 159.726f },
    { 1122, 160.730f },
    { 1123, 160.194f },
    { 1124, 160.455f },
    {  120, 157.344f },
    {  121, 159.307f },
    {  122, 158.648f },
    {  123, 154.893f },
};
#define ALT_PRESET_COUNT ((int)(sizeof(g_alt_presets)/sizeof(g_alt_presets[0])))
static int g_alt_preset_idx = 10;     /* 当前预设索引，10 = ID1122(160.730m) */
volatile int g_alt_preset_id = 1122;  /* 当前预设 ID（供 OLED 显示），与 idx=10 对应 */

/* 运行时融合权重（固定 0.5/0.5，不动态调整） */
static float runtime_fusion_weight_ms5611 = FUSION_WEIGHT_MS5611;
static float runtime_fusion_weight_bmp280 = FUSION_WEIGHT_BMP280;

/* 多任务联合模型运行态（替代原 ms5611/bmp280 两个 per-sensor 滤波器）
 * 始终编译，仅在 WORK_MODE==0 时被调用（采集模式不运行 NN） */
static MultitaskModel_t g_mt_model;
float g_mt_filt_ms = 0.0f;     /* MS5611 滤波输出（绝对气压 Pa） */
float g_mt_filt_bmp = 0.0f;    /* BMP280 滤波输出（绝对气压 Pa） */
float g_mt_nn_divergence = 0.0f;  /* NN vs KF 散度（方案16 场景判定用） */
float g_mt_scene[2] = {0.5f, 0.5f};  /* 场景概率 [static, elevation] */
int   g_mt_scene_pred = 0;     /* 0=static 1=elevation */
bool  g_mt_ready = false;

#if WORK_MODE == 0
/* NN 输出二次平滑 KF（方案3 使用，对联合模型输出再做卡尔曼滤波） */
#define NN_KF_Q  0.01f
#define NN_KF_R  0.5f
static KalmanFilter_TypeDef ms5611_kf_nn;
static KalmanFilter_TypeDef bmp280_kf_nn;
#endif

/* output_mode removed - always output full data */

osThreadId_t MS5611TaskHandle;
osThreadId_t BMP280TaskHandle;
osThreadId_t UARTOutputTaskHandle;
osThreadId_t UARTCommandTaskHandle;
osThreadId_t OLEDTaskHandle;
osMutexId_t UARTMutexHandle;
osMutexId_t SensorDataMutexHandle;

const osThreadAttr_t MS5611Task_attributes = {
  .name = "MS5611Task",
  .stack_size = 1024 * 4,
  .priority = (osPriority_t) osPriorityHigh,
};
const osThreadAttr_t BMP280Task_attributes = {
  .name = "BMP280Task",
  .stack_size = 1536 * 4,
  .priority = (osPriority_t) osPriorityHigh,
};
const osThreadAttr_t UARTOutputTask_attributes = {
  .name = "UARTOutputTask",
  .stack_size = 768 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};
const osThreadAttr_t UARTCommandTask_attributes = {
  .name = "UARTCommandTask",
  .stack_size = 512 * 4,
  .priority = (osPriority_t) osPriorityHigh,
};
const osThreadAttr_t OLEDTask_attributes = {
  .name = "OLEDTask",
  .stack_size = 1024 * 4,
  .priority = (osPriority_t) osPriorityHigh,
};
const osMutexAttr_t UARTMutex_attributes = {
  .name = "UARTMutex",
};

uint32_t sample_id = 0;
char uart_rx_buffer[64];
uint8_t uart_rx_len = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_USART2_UART_Init(void);
void StartDefaultTask(void *argument);

/* USER CODE BEGIN PFP */
int fputc(int ch, FILE *f);
void StartMS5611Task(void *argument);
void StartBMP280Task(void *argument);
void StartUARTOutputTask(void *argument);
void StartUARTCommandTask(void *argument);
void StartOLEDTask(void *argument);
void ProcessCommand(char *cmd);

/* ========== 增强校准辅助函数 ========== */
static void CalibStats_Init(CalibStats_t *stats);
static void CalibStats_Update(CalibStats_t *stats, float value);
static void CalibStats_Finalize(CalibStats_t *stats);
static void CalibStats_RobustClean(const float *buf, int count,
                                   float *out_mean, float *out_std,
                                   float *out_min, float *out_max,
                                   int *out_inliers);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* DMA 发送忙标志：必须在 fputc 之前声明（fputc 会等待它） */
#if UART_ENABLE
static volatile uint8_t uart_tx_busy = 0;
#endif

int fputc(int ch, FILE *f)
{
#if UART_ENABLE
    /* 与 DMA 发送互斥：先等 DMA 完成，再在 UART 互斥锁内轮询发送，
       避免轮询(HAL_UART_Transmit)与 DMA(HAL_UART_Transmit_DMA)同时操作
       同一个 huart2，导致 HAL 状态机错乱、后续 DMA 永远无法启动而卡死。
       调度器启动前 UARTMutexHandle 为 NULL，此时直接轮询发送。 */
    if (UARTMutexHandle != NULL)
    {
        osMutexAcquire(UARTMutexHandle, osWaitForever);
    }
    while (uart_tx_busy) { /* 等待上一轮 DMA 发送完成 */ }
    HAL_UART_Transmit(&huart2, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    if (UARTMutexHandle != NULL)
    {
        osMutexRelease(UARTMutexHandle);
    }
#endif
    return ch;
}

#if UART_ENABLE
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2)
    {
        uart_tx_busy = 0;  /* 标记 DMA 传输完成 */
    }
}
#endif /* UART_ENABLE */

/* FreeRTOS 断言失败标志（供调试器观测，不在此处 printf 以免死锁） */
volatile uint32_t g_freertos_assert_failed = 0;

/* 栈溢出钩子：开启 configCHECK_FOR_STACK_OVERFLOW 后由内核调用。
   注意：此处禁止调用 printf/UART（可能在错误上下文且会死锁），
   仅置标志并陷入循环，便于调试器查看是哪个任务栈溢出。 */
void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName)
{
    (void)xTask;
    (void)pcTaskName;
    volatile uint32_t dummy = 0;
    while (1) { dummy++; }
}

/* 内存分配失败钩子：堆不足时由内核调用。 */
void vApplicationMallocFailedHook(void)
{
    volatile uint32_t dummy = 0;
    while (1) { dummy++; }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART1_UART_Init();
  MX_USART2_UART_Init();
  MX_X_CUBE_AI_Init();
  /* USER CODE BEGIN 2 */
#if WORK_MODE == 0
  SSD1306_Init();      /* 初始化 OLED (I2C) */
  /* ====== DEBUG: 全屏填充测试 ====== */
  {
    SSD1306_Fill(SSD1306_COLOR_WHITE);
    SSD1306_UpdateScreen();
    printf("DBG,OLED,Full white written\r\n");
    HAL_Delay(2000);

    SSD1306_Fill(SSD1306_COLOR_BLACK);
    SSD1306_UpdateScreen();
    printf("DBG,OLED,Full black written\r\n");
    HAL_Delay(1000);

    SSD1306_DrawString(0, 0, "OLED Init OK!", SSD1306_FONT_6X8, SSD1306_COLOR_WHITE);
    SSD1306_DrawString(0, 11, "I2C Addr:0x78", SSD1306_FONT_6X8, SSD1306_COLOR_WHITE);
    SSD1306_DrawString(0, 22, "PC0=SDA PC1=SCL", SSD1306_FONT_6X8, SSD1306_COLOR_WHITE);
    SSD1306_DrawString(0, 33, "Wait RTOS...", SSD1306_FONT_6X8, SSD1306_COLOR_WHITE);
    SSD1306_UpdateScreen();
    printf("DBG,OLED,Text pattern written\r\n");
    HAL_Delay(2000);
  }
#endif
  __HAL_RCC_GPIOC_CLK_ENABLE();
  
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  GPIO_InitStruct.Pin = GPIO_PIN_2 | GPIO_PIN_3 | GPIO_PIN_10 | GPIO_PIN_11;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
  
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_2, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_3, GPIO_PIN_SET);
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_10, GPIO_PIN_SET);
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_11, GPIO_PIN_SET);

  /* PC13 按键：输入上拉，按下为低电平（用于循环切换预设高度） */
  GPIO_InitStruct.Pin = GPIO_PIN_13;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
  
  MS5611_GPIO_Init();
  HAL_Delay(100);
  
  char reset_result = MS5611_Reset();
  if(reset_result == 0)
  {
    printf("INFO,MS5611 Reset OK\r\n");
  }
  else
  {
    printf("INFO,MS5611 Reset Failed: %d\r\n", reset_result);
  }
  
  HAL_Delay(300);
  MS5611_Read_PROM();
  printf("INFO,MS5611 PROM Read OK\r\n");
  printf("DBG,PROM,C1=%u,C2=%u,C3=%u,C4=%u,C5=%u,C6=%u\r\n",
         Cal_C1_6[1], Cal_C1_6[2], Cal_C1_6[3],
         Cal_C1_6[4], Cal_C1_6[5], Cal_C1_6[6]);
  
  bool bmp280_init_result = BMP280_Init();
  if(bmp280_init_result)
  {
    printf("INFO,BMP280 Init OK\r\n");
    bmp280_ready = true;
  }
  else
  {
    printf("INFO,BMP280 Init Failed\r\n");
    bmp280_ready = false;
  }
  
  /* ========== 增强校准：更多样本 + 自动分析 + 自动调参 ========== */
  {
    uint32_t calib_start_ms = HAL_GetTick();
    printf("INFO,=== Enhanced Calibration Started ===\r\n");

    /* ---- 热稳定等待：上电后等板载温度达到稳态再开始采样/锁定参考气压 ---- */
    /* 目的：消除上电热瞬态导致的参考气压漂移（日志显示重启前后基准差 ~10 Pa）。
     *       等温度爬升率低于阈值并持续一段时间后，再采集校准样本，保证每次上电
     *       锁定的参考气压处于一致的热状态，从根本上抑制基准漂移。 */
    {
      uint32_t warm_start = HAL_GetTick();
      float t_prev = 0.0f;
      uint32_t t_prev_ms = 0;
      bool have_prev = false;
      uint32_t stable_since = 0;
      printf("INFO,Waiting for thermal stabilization (max %lu s)...\r\n",
             THERMAL_MAX_WAIT_MS / 1000u);
      while((HAL_GetTick() - warm_start) < THERMAL_MAX_WAIT_MS)
      {
        float ms_p = 0.0f, ms_t = 0.0f, bm_p = 0.0f, bm_t = 0.0f;
        MS5611_Read_Data(&ms_p, &ms_t);
        if(bmp280_ready) BMP280_Read_Data(&bm_p, &bm_t);
        float t_now = bmp280_ready ? bm_t : ms_t;   /* 优先用 BMP280 温度 */
        uint32_t now = HAL_GetTick();

        if(have_prev)
        {
          float dt = (float)(now - t_prev_ms) / 1000.0f;
          if(dt > 0.05f)
          {
            float rate = fabsf(t_now - t_prev) / dt;     /* °C/s */
            bool warmed = ((now - warm_start) >= THERMAL_MIN_WARMUP_MS);
            if(warmed && rate < THERMAL_STABLE_RATE)
            {
              if(stable_since == 0u) stable_since = now;
              else if((now - stable_since) >= THERMAL_STABLE_HOLD_MS) break;
            }
            else
            {
              stable_since = 0u;
            }
          }
        }
        t_prev = t_now; t_prev_ms = now; have_prev = true;
        HAL_Delay(500);
      }
      printf("INFO,Thermal stabilization done (waited %lu ms)\r\n",
             HAL_GetTick() - warm_start);
    }

    /* ---- 单阶段校准采样（估计均值/方差，精细野值剔除交由 Phase3 的 RobustClean） ---- */
    printf("INFO,Calibration sampling (%d samples, %.1f seconds)...\r\n",
           CALIB_SAMPLES, CALIB_SAMPLES * 0.05f);

    /* 缓冲区：存原始气压值用于离线分析 */
    float ms5611_buf[CALIB_SAMPLES];
    float bmp280_buf[CALIB_SAMPLES];
    int ms5611_buf_count = 0;
    int bmp280_buf_count = 0;

    /* 初始化校准统计 */
    CalibStats_Init(&calib_result.ms5611_stats);
    CalibStats_Init(&calib_result.bmp280_stats);
    calib_result.state = CALIB_STATE_PHASE1;
    calib_result.phase = 1;
    calib_result.current_altitude = 156.927f;  /* 固定预设海拔 156.927 m */

    /* 临时 KF 用于校准期间的实时显示 */
    KalmanFilter_TypeDef temp_kf_ms, temp_kf_bmp;
    bool temp_kf_init = false;

    for(int i = 0; i < CALIB_SAMPLES; i++)
    {
      float ms5611_p = 0.0f, ms5611_t = 0.0f;
      float bmp280_p = 0.0f, bmp280_t = 0.0f;
      float ms5611_p_filt = 0.0f, bmp280_p_filt = 0.0f;

      /* 读取传感器 */
      MS5611_Read_Data(&ms5611_p, &ms5611_t);
      if(bmp280_ready) BMP280_Read_Data(&bmp280_p, &bmp280_t);

      /* 有效性检查 */
      bool ms5611_ok = (ms5611_p > 0 && ms5611_p < 200000);
      bool bmp280_ok = (bmp280_ready && bmp280_p > 0 && bmp280_p < 200000);

      if(ms5611_ok)
      {
        ms5611_buf[ms5611_buf_count++] = ms5611_p;

        /* 单阶段统计：仅收集原始值，精细野值剔除交由 Phase3 的 RobustClean */
        CalibStats_Update(&calib_result.ms5611_stats, ms5611_p);

        /* 临时 KF 用于实时显示 */
        if(!temp_kf_init)
        {
          KalmanFilter_Init(&temp_kf_ms, ms5611_p, KF_INIT_P, MS5611_KF_Q, MS5611_KF_R);
          if(bmp280_ok) KalmanFilter_Init(&temp_kf_bmp, bmp280_p, KF_INIT_P, BMP280_KF_Q, BMP280_KF_R);
          temp_kf_init = true;
          ms5611_p_filt = ms5611_p;
          bmp280_p_filt = bmp280_p;
        }
        else
        {
          ms5611_p_filt = KalmanFilter_Update_Adaptive(&temp_kf_ms, ms5611_p);
          if(bmp280_ok) bmp280_p_filt = KalmanFilter_Update_Adaptive_BMP280(&temp_kf_bmp, bmp280_p);
        }
      }

      if(bmp280_ok)
      {
        bmp280_buf[bmp280_buf_count++] = bmp280_p;

        /* 单阶段统计：仅收集原始值 */
        CalibStats_Update(&calib_result.bmp280_stats, bmp280_p);
      }

      /* 校准中间结果：每 5 秒输出一次 */
      if((HAL_GetTick() - calib_start_ms) >= 5000 && i > 0 && (i % 50) == 0)
      {
        int pct = (i * 100) / CALIB_SAMPLES;
        printf("CALIB,MS5611,P_raw=%.2f,P_filt=%.2f,T=%.2f,Progress=%d%%\r\n",
               ms5611_p, ms5611_p_filt, ms5611_t, pct);
        if(bmp280_ready)
        {
          printf("CALIB,BMP280,P_raw=%.2f,P_filt=%.2f,T=%.2f,Progress=%d%%\r\n",
                 bmp280_p, bmp280_p_filt, bmp280_t, pct);
        }
      }

      HAL_Delay(50);
    }

    /* ---- 阶段 3：自动分析和参数调优 ---- */
    calib_result.state = CALIB_STATE_ANALYSIS;
    printf("INFO,Phase 3: Analyzing sensor characteristics...\r\n");

    /* 最终化统计 */
    CalibStats_Finalize(&calib_result.ms5611_stats);
    CalibStats_Finalize(&calib_result.bmp280_stats);

    /* ---- 稳健清洗：剔除校准期野值（含采样期未过滤的异常点），覆盖传感器统计 ---- */
    /* 目的：防止偶发坏点（如 BMP280 的 64389 Pa）污染 std/variance，
     *       导致传感器健康度/一致性误判、参考气压锁定偏差。KF 的 Q/R 已由 TUNED_* 固化，不再依赖此处方差。 */
    {
      float m_mean, m_std, m_min, m_max; int m_inl;
      float b_mean, b_std, b_min, b_max; int b_inl;
      CalibStats_RobustClean(ms5611_buf, ms5611_buf_count, &m_mean, &m_std, &m_min, &m_max, &m_inl);
      CalibStats_RobustClean(bmp280_buf, bmp280_buf_count, &b_mean, &b_std, &b_min, &b_max, &b_inl);

      /* 覆盖 MS5611 统计（后续 KF 调参 / 健康度 / 参考气压均使用干净值） */
      calib_result.ms5611_stats.mean = m_mean;
      calib_result.ms5611_stats.std = m_std;
      calib_result.ms5611_stats.variance = m_std * m_std;
      calib_result.ms5611_stats.min = m_min;
      calib_result.ms5611_stats.max = m_max;
      calib_result.ms5611_stats.range = m_max - m_min;
      calib_result.ms5611_stats.valid_count = m_inl;
      calib_result.ms5611_stats.outlier_count = ms5611_buf_count - m_inl;
      calib_result.ms5611_stats.outlier_ratio = (ms5611_buf_count > 0)
            ? (float)(ms5611_buf_count - m_inl) / (float)ms5611_buf_count : 0.0f;

      /* 覆盖 BMP280 统计 */
      calib_result.bmp280_stats.mean = b_mean;
      calib_result.bmp280_stats.std = b_std;
      calib_result.bmp280_stats.variance = b_std * b_std;
      calib_result.bmp280_stats.min = b_min;
      calib_result.bmp280_stats.max = b_max;
      calib_result.bmp280_stats.range = b_max - b_min;
      calib_result.bmp280_stats.valid_count = b_inl;
      calib_result.bmp280_stats.outlier_count = bmp280_buf_count - b_inl;
      calib_result.bmp280_stats.outlier_ratio = (bmp280_buf_count > 0)
            ? (float)(bmp280_buf_count - b_inl) / (float)bmp280_buf_count : 0.0f;
    }

    /* 计算双传感器差值统计 */
    {
      int n = (ms5611_buf_count < bmp280_buf_count) ? ms5611_buf_count : bmp280_buf_count;
      float diff_sum = 0, diff_sq_sum = 0;
      int diff_count = 0;
      for(int i = 0; i < n; i++)
      {
        float d = ms5611_buf[i] - bmp280_buf[i];
        if(fabsf(d) < 500.0f)  /* 合理差值范围过滤 */
        {
          diff_sum += d;
          diff_sq_sum += d * d;
          diff_count++;
        }
      }
      if(diff_count > 0)
      {
        calib_result.diff_mean = diff_sum / diff_count;
        float diff_var = (diff_sq_sum / diff_count) - (calib_result.diff_mean * calib_result.diff_mean);
        calib_result.diff_std = (diff_var > 0) ? sqrtf(diff_var) : 0.1f;
      }
      else
      {
        calib_result.diff_mean = 0;
        calib_result.diff_std = 10.0f;
      }

      /* 一致性评分：差值标准差小 → 一致性好 */
      if(calib_result.diff_std < 2.0f)       calib_result.consistency = 1.0f;
      else if(calib_result.diff_std < 5.0f)  calib_result.consistency = 0.8f;
      else if(calib_result.diff_std < 10.0f) calib_result.consistency = 0.5f;
      else                                    calib_result.consistency = 0.2f;
    }

    /* ---- 融合权重（来自编译期宏，仅用于校准报告展示） ----
     * 注：KF 的 Q/R 与 EKF 噪声参数已不再由本次校准在线估算，
     *     统一在编译期由 fusion_scheme_tuned_params.h 的 TUNED_* 固化（见下方覆盖段）。
     *     在线估算结果会被覆盖段直接丢弃，故此处不再计算。 */
    calib_result.fusion_weight_ms5611 = FUSION_WEIGHT_MS5611;
    calib_result.fusion_weight_bmp280 = FUSION_WEIGHT_BMP280;


    /* 5. 传感器健康状态 */
    {
      float ms_std_val = calib_result.ms5611_stats.std;
      float bm_std_val = calib_result.bmp280_stats.std;

      /* MS5611 健康度 */
      if(ms_std_val < 3.0f && calib_result.ms5611_stats.outlier_ratio < 0.05f)
        calib_result.ms5611_stats.health = SENSOR_HEALTH_GOOD;
      else if(ms_std_val < 8.0f && calib_result.ms5611_stats.outlier_ratio < 0.15f)
        calib_result.ms5611_stats.health = SENSOR_HEALTH_FAIR;
      else
        calib_result.ms5611_stats.health = SENSOR_HEALTH_POOR;

      /* BMP280 健康度 */
      if(bm_std_val < 2.0f && calib_result.bmp280_stats.outlier_ratio < 0.05f)
        calib_result.bmp280_stats.health = SENSOR_HEALTH_GOOD;
      else if(bm_std_val < 5.0f && calib_result.bmp280_stats.outlier_ratio < 0.15f)
        calib_result.bmp280_stats.health = SENSOR_HEALTH_FAIR;
      else
        calib_result.bmp280_stats.health = SENSOR_HEALTH_POOR;
    }

    /* ---- 计算参考气压 ---- */
    {
      float measured_pressure = calib_result.ms5611_stats.mean;
      calib_result.ms5611_pressure_avg = measured_pressure;
      calib_result.bmp280_pressure_avg = calib_result.bmp280_stats.mean;
      float altitude_m = calib_result.current_altitude;
#if ALTITUDE_FORMULA_ISA
      /* ISA 公式反算海平面气压：P0 = P / (1 - L*h/T0)^(g/(L*R)) */
      float isa_factor = 1.0f - ISA_L * altitude_m / ISA_T0;
      if(isa_factor > 0.01f) {
        float exponent = ISA_G / (ISA_L * ISA_R);
        calib_result.reference_pressure_pa = measured_pressure / powf(isa_factor, exponent);
      } else {
        calib_result.reference_pressure_pa = measured_pressure + altitude_m * 11.3f;
      }
#else
      /* 等温模型反解（与前向公式 h = -(R*T/g)*ln(P/P0) 严格互逆）：
       * P0 = P * exp(g*h/(R*T))，温度取 BMP280 实测值，无效时退回标准温度 */
      float temp_k = (bmp280_data.temperature_c > -100.0f && bmp280_data.temperature_c < 100.0f)
                     ? (bmp280_data.temperature_c + 273.15f) : ISA_T0;
      calib_result.reference_pressure_pa = measured_pressure * expf(ISA_G * altitude_m / (ISA_R * temp_k));
#endif
      reference_pressure_pa = calib_result.reference_pressure_pa;
    }

    /* ---- 用 PC 端自动调参得到的最优全局 KF 参数覆盖标定值 ----
     * 降噪对 KF Q/R 极其敏感，必须使用 tune 得到的最优值而非传感器标定的经验值：
     * 方案 1-14：融合参数 + 共享 KF/EMA（fusion_scheme_tuned_params.h，tune_all_params.py 两级调参）
     * 方案 15/16/17：S15S16_* 参数（fusion_scheme_15_16_params.h，compare_s16.py） */
#if FUSION_SCHEME == 15 || FUSION_SCHEME == 16 || FUSION_SCHEME == 17
    calib_result.kf_q_ms5611 = S15S16_MS5611_KF_Q;
    calib_result.kf_r_ms5611 = S15S16_MS5611_KF_R;
    calib_result.kf_q_bmp280 = S15S16_BMP280_KF_Q;
    calib_result.kf_r_bmp280 = S15S16_BMP280_KF_R;
#else
    calib_result.kf_q_ms5611 = TUNED_MS5611_KF_Q;
    calib_result.kf_r_ms5611 = TUNED_MS5611_KF_R;
    calib_result.kf_q_bmp280 = TUNED_BMP280_KF_Q;
    calib_result.kf_r_bmp280 = TUNED_BMP280_KF_R;
#endif

    /* ---- 应用自动调参结果 ---- */
    if(calib_use_auto_params)
    {
      /* 重新初始化 KF（使用自动调参后的参数） */
      KalmanFilter_Init(&ms5611_kf_pressure, calib_result.ms5611_stats.mean, KF_INIT_P,
                        calib_result.kf_q_ms5611, calib_result.kf_r_ms5611);
      KalmanFilter_Init(&bmp280_kf_pressure, calib_result.bmp280_stats.mean, KF_INIT_P,
                        calib_result.kf_q_bmp280, calib_result.kf_r_bmp280);

#if WORK_MODE == 0
      /* 初始化多任务联合模型：用校准均值预填双窗口，采集满 10 帧后即可正式推理 */
      Multitask_Init(calib_result.ms5611_stats.mean, calib_result.bmp280_stats.mean);

      /* 初始化 NN 输出二次平滑 KF（方案3 使用） */
      KalmanFilter_Init(&ms5611_kf_nn, calib_result.ms5611_stats.mean, KF_INIT_P, NN_KF_Q, NN_KF_R);
      KalmanFilter_Init(&bmp280_kf_nn, calib_result.bmp280_stats.mean, KF_INIT_P, NN_KF_Q, NN_KF_R);
#endif

      ms5611_ready = true;

      /* 初始化 BMP280 偏置补偿：使两传感器在绝对气压上对齐 */
      bmp_bias_pa = calib_result.diff_mean;  /* diff_mean = MS5611_mean - BMP280_mean */
    }

    /* ---- 输出详细校准报告 ---- */
    calib_result.state = CALIB_STATE_COMPLETE;
    calib_result.elapsed_ms = HAL_GetTick() - calib_start_ms;

    printf("INFO,=== Enhanced Calibration Complete ===\r\n");
    printf("INFO,Elapsed: %lu ms\r\n", calib_result.elapsed_ms);
    printf("INFO,--- Sensor Statistics ---\r\n");
    printf("INFO,MS5611: mean=%.2f std=%.2f min=%.2f max=%.2f range=%.2f outliers=%d(%.1f%%) health=%d\r\n",
           calib_result.ms5611_stats.mean, calib_result.ms5611_stats.std,
           calib_result.ms5611_stats.min, calib_result.ms5611_stats.max,
           calib_result.ms5611_stats.range,
           calib_result.ms5611_stats.outlier_count, calib_result.ms5611_stats.outlier_ratio * 100.0f,
           calib_result.ms5611_stats.health);
    printf("INFO,BMP280: mean=%.2f std=%.2f min=%.2f max=%.2f range=%.2f outliers=%d(%.1f%%) health=%d\r\n",
           calib_result.bmp280_stats.mean, calib_result.bmp280_stats.std,
           calib_result.bmp280_stats.min, calib_result.bmp280_stats.max,
           calib_result.bmp280_stats.range,
           calib_result.bmp280_stats.outlier_count, calib_result.bmp280_stats.outlier_ratio * 100.0f,
           calib_result.bmp280_stats.health);
    printf("INFO,Dual sensor diff: mean=%.2f std=%.2f consistency=%.2f\r\n",
           calib_result.diff_mean, calib_result.diff_std, calib_result.consistency);
    printf("INFO,--- Auto-tuned Parameters ---\r\n");
    printf("INFO,KF_MS5611: Q=%.4f R=%.2f\r\n", calib_result.kf_q_ms5611, calib_result.kf_r_ms5611);
    printf("INFO,KF_BMP280: Q=%.4f R=%.2f\r\n", calib_result.kf_q_bmp280, calib_result.kf_r_bmp280);
    printf("INFO,Fusion weights: MS5611=%.2f BMP280=%.2f\r\n",
           calib_result.fusion_weight_ms5611, calib_result.fusion_weight_bmp280);
    printf("INFO,Reference pressure: %.2f Pa (altitude=%.1f m)\r\n",
           calib_result.reference_pressure_pa, calib_result.current_altitude);
    printf("INFO,Use STATUS/HELP/SET_ALT commands. Type SET_CALIB to view calibration details.\r\n");
  }  /* 增强校准代码块结束 */
  /* USER CODE END 2 */

  /* Init scheduler */
  osKernelInitialize();

  /* USER CODE BEGIN RTOS_MUTEX */
  UARTMutexHandle = osMutexNew(&UARTMutex_attributes);
  SensorDataMutexHandle = osMutexNew(NULL);
  /* USER CODE END RTOS_MUTEX */

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* add semaphores, ... */
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* start timers, add new ones, ... */
  /* USER CODE END RTOS_TIMERS */

  /* USER CODE BEGIN RTOS_QUEUES */
  /* add queues, ... */
  /* USER CODE END RTOS_QUEUES */

  /* Create the thread(s) */
  /* creation of defaultTask */
  defaultTaskHandle = osThreadNew(StartDefaultTask, NULL, &defaultTask_attributes);

  /* USER CODE BEGIN RTOS_THREADS */
  MS5611TaskHandle = osThreadNew(StartMS5611Task, NULL, &MS5611Task_attributes);
  BMP280TaskHandle = osThreadNew(StartBMP280Task, NULL, &BMP280Task_attributes);
  UARTOutputTaskHandle = osThreadNew(StartUARTOutputTask, NULL, &UARTOutputTask_attributes);
  UARTCommandTaskHandle = osThreadNew(StartUARTCommandTask, NULL, &UARTCommandTask_attributes);
#if WORK_MODE == 0
  OLEDTaskHandle = osThreadNew(StartOLEDTask, NULL, &OLEDTask_attributes);
#endif
  /* USER CODE END RTOS_THREADS */

  /* USER CODE BEGIN RTOS_EVENTS */
  /* add events, ... */
  /* USER CODE END RTOS_EVENTS */

  /* Start scheduler */
  osKernelStart();

  /* We should never get here as control is now taken by the scheduler */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 100;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* ========== 预设高度切换辅助函数 ==========
 * 用指定预设点的已知海拔，基于当前 MS5611 实测气压重算参考气压（ISA 反算海平面气压），
 * 使后续输出高度对齐到该点真实海拔；同时更新供 OLED 显示的当前 ID。 */
#if WORK_MODE == 0
static void ApplyAltPreset(int idx)
{
    if(idx < 0 || idx >= ALT_PRESET_COUNT) idx = 0;
    g_alt_preset_idx = idx;
    g_alt_preset_id  = g_alt_presets[idx].id;
    float new_altitude = g_alt_presets[idx].altitude_m;
    calib_result.current_altitude = new_altitude;

    /* 基于当前 MS5611 KF 滤波气压重算统一参考气压（比原始值更稳定，降低瞬态噪声干扰） */
    float ms_measured = (ms5611_data.pressure_filtered_pa > 0) ? ms5611_data.pressure_filtered_pa : reference_pressure_pa;
#if ALTITUDE_FORMULA_ISA
    float isa_factor = 1.0f - ISA_L * new_altitude / ISA_T0;
    if(isa_factor > 0.01f) {
        float exponent = ISA_G / (ISA_L * ISA_R);
        reference_pressure_pa = ms_measured / powf(isa_factor, exponent);
    } else {
        reference_pressure_pa = ms_measured + new_altitude * 11.3f;
    }
#else
    /* 等温模型反解（与前向公式严格互逆）：P0 = P * exp(g*h/(R*T)) */
    float temp_k = (bmp280_data.temperature_c > -100.0f && bmp280_data.temperature_c < 100.0f)
                   ? (bmp280_data.temperature_c + 273.15f) : ISA_T0;
    reference_pressure_pa = ms_measured * expf(ISA_G * new_altitude / (ISA_R * temp_k));
#endif
    calib_result.reference_pressure_pa = reference_pressure_pa;
    /* 请求融合任务把积分锁重新锚定到新海拔（OLED 立即显示预设值） */
    g_fusion_reanchor_alt = new_altitude;
    g_fusion_reanchor_req = true;
    g_height_snap_req = true;   /* 同时请求 OLED 高度 EMA 立即对齐到目标海拔，避免滑过 */
    /* 相对高度基准直接钉到目标海拔：与重锚在同一时刻完成，消除“基准在重锚前
     * 被 OLED 锁定”的竞态，使上电自动重锚 / 手动切换后相对高度立即为 0。 */
    rel_height_ref = new_altitude;
    rel_ref_set = true;
    printf("INFO,Preset ID=%d altitude=%.3f m ref_pressure=%.2f Pa\r\n",
           g_alt_preset_id, new_altitude, reference_pressure_pa);
}
#endif /* WORK_MODE == 0 */

/* ========== 增强校准辅助函数 ========== */

static void CalibStats_Init(CalibStats_t *stats)
{
    stats->mean = 0;
    stats->variance = 0;
    stats->std = 0;
    stats->min = 1e10f;
    stats->max = -1e10f;
    stats->range = 0;
    stats->valid_count = 0;
    stats->outlier_count = 0;
    stats->outlier_ratio = 0;
    stats->block_count = 0;
    stats->mean_block_std = 0;
    stats->health = SENSOR_HEALTH_GOOD;
    for(int i = 0; i < CALIB_STD_COUNT_MAX; i++)
        stats->block_std[i] = 0;
}

static void CalibStats_Update(CalibStats_t *stats, float value)
{
    /* 增量式均值和方差计算 (Welford's algorithm) */
    int n = stats->valid_count + 1;
    if(n == 1)
    {
        stats->mean = value;
        stats->variance = 0;
    }
    else
    {
        float delta = value - stats->mean;
        stats->mean += delta / n;
        float delta2 = value - stats->mean;
        stats->variance += delta * delta2;
    }

    /* 更新极值 */
    if(value < stats->min) stats->min = value;
    if(value > stats->max) stats->max = value;

    stats->valid_count = n;
}

static void CalibStats_Finalize(CalibStats_t *stats)
{
    if(stats->valid_count > 1)
    {
        stats->variance /= (stats->valid_count - 1);  /* 无偏估计 */
    }
    stats->std = (stats->variance > 0) ? sqrtf(stats->variance) : 0;
    stats->range = stats->max - stats->min;

    if(stats->valid_count > 0)
    {
        stats->outlier_ratio = (float)stats->outlier_count /
                               (float)(stats->valid_count + stats->outlier_count);
    }

    /* 计算分块标准差均值（评估短期噪声） */
    if(stats->block_count > 0)
    {
        float sum = 0;
        for(int i = 0; i < stats->block_count; i++)
            sum += stats->block_std[i];
        stats->mean_block_std = sum / stats->block_count;
    }
    else
    {
        stats->mean_block_std = stats->std;
    }
}

/* 稳健统计清洗：用中位数 + MAD 检测并剔除野值，输出干净均值/标准差/极值/有效数。
 * 用于校准后处理，消除阶段1未过滤的偶发异常点（如 BMP280 的 64389 坏点）对
 * KF 自动调参、传感器健康度及参考气压锁定的污染。 */
static void CalibStats_RobustClean(const float *buf, int count,
                                   float *out_mean, float *out_std,
                                   float *out_min, float *out_max,
                                   int *out_inliers)
{
    *out_mean = 0.0f; *out_std = 0.0f;
    *out_min = 1e10f; *out_max = -1e10f; *out_inliers = 0;
    if(count <= 0) return;

    int n = (count < CALIB_SAMPLES) ? count : CALIB_SAMPLES;
    /* 静态缓冲：避免在调用栈上分配大数组（校准仅启动期调用一次，无重入风险） */
    static float sorted[CALIB_SAMPLES];
    static float dev[CALIB_SAMPLES];

    for(int i = 0; i < n; i++) sorted[i] = buf[i];

    /* 插入排序（样本量小，开销可忽略） */
    for(int i = 1; i < n; i++) {
        float key = sorted[i];
        int j = i - 1;
        while(j >= 0 && sorted[j] > key) { sorted[j+1] = sorted[j]; j--; }
        sorted[j+1] = key;
    }

    float median = sorted[n/2];

    /* MAD（中位绝对偏差）估计离散度 */
    for(int i = 0; i < n; i++) dev[i] = fabsf(sorted[i] - median);
    for(int i = 1; i < n; i++) {
        float key = dev[i];
        int j = i - 1;
        while(j >= 0 && dev[j] > key) { dev[j+1] = dev[j]; j--; }
        dev[j+1] = key;
    }
    float mad = dev[n/2];
    /* 稳健 3σ 阈值（标准差 ≈ 1.4826·MAD），下限 30 Pa 防止正常抖动被误删 */
    float th = 3.0f * 1.4826f * mad;
    if(th < 30.0f) th = 30.0f;

    float sum = 0.0f, sum_sq_dev = 0.0f;
    int nc = 0;
    for(int i = 0; i < n; i++) {
        if(fabsf(buf[i] - median) <= th) {
            float d = buf[i] - median;          /* 相对中位数的偏差，量级很小（几 Pa） */
            sum += buf[i];
            sum_sq_dev += d * d;                /* 围绕中位数累加平方偏差，避免大数相减 */
            if(buf[i] < *out_min) *out_min = buf[i];
            if(buf[i] > *out_max) *out_max = buf[i];
            nc++;
        }
    }
    if(nc > 0) {
        float m = sum / nc;
        /* var = E[(x-median)^2] - (E[x]-median)^2，两项均为小量，无精度损失 */
        float var = sum_sq_dev / nc - (m - median) * (m - median);
        if(var < 0.0f) var = 0.0f;             /* 数值保护 */
        *out_mean = m;
        *out_std  = (var > 0.0f) ? sqrtf(var) : 0.1f;
        *out_inliers = nc;
    }
}
/* ========== 校准辅助函数结束 ========== */

/* ========== 新算法方案：多任务联合模型 (baseline_eqw) ========== */
void Multitask_Init(float init_ms, float init_bmp)
{
    for(int i = 0; i < MT_WINDOW; i++) {
        g_mt_model.win_ms[i] = init_ms;
        g_mt_model.win_bmp[i] = init_bmp;
    }
    g_mt_model.idx  = 0;
    g_mt_model.n_ms = MT_WINDOW;   /* 预填窗口，冷启动首帧即可推理 */
    g_mt_model.n_bmp = MT_WINDOW;
    g_mt_ready = false;
    g_mt_scene[0] = 0.5f; g_mt_scene[1] = 0.5f;
    g_mt_scene_pred = 0;
}

/* 把 MS5611 读数推入联合模型窗口（MS5611 任务调用） */
void Multitask_PushMS(float ms_pa)
{
    g_mt_model.win_ms[g_mt_model.idx] = ms_pa;
    if(g_mt_model.n_ms < MT_WINDOW) g_mt_model.n_ms++;
}

/* 把 BMP280 读数推入联合模型窗口并推进环形索引（BMP280 任务调用） */
void Multitask_PushBMP(float bmp_pa)
{
    g_mt_model.win_bmp[g_mt_model.idx] = bmp_pa;
    if(g_mt_model.n_bmp < MT_WINDOW) g_mt_model.n_bmp++;
    g_mt_model.idx = (g_mt_model.idx + 1) % MT_WINDOW;  /* 双窗口共用同一索引推进，保证时间对齐 */
}

/* 双窗口齐全时执行一次联合推理，输出滤波值与场景概率；返回是否成功更新 */
bool Multitask_Run(void)
{
    if(g_mt_model.n_ms < MT_WINDOW || g_mt_model.n_bmp < MT_WINDOW)
        return false;

    float in[MT_INPUT_DIM];
    for(int i = 0; i < MT_WINDOW; i++) {
        int p = (g_mt_model.idx + i) % MT_WINDOW;       /* 旧 -> 新 时间序 */
        float rel = g_mt_model.win_ms[p] - MT_REF_PRESSURE;
        in[i] = (rel - MT_FEAT_MEAN[i]) / MT_FEAT_STD[i];
    }
    for(int i = 0; i < MT_WINDOW; i++) {
        int p = (g_mt_model.idx + i) % MT_WINDOW;
        float rel = g_mt_model.win_bmp[p] - MT_REF_PRESSURE;
        in[MT_WINDOW + i] = (rel - MT_FEAT_MEAN[MT_WINDOW + i]) / MT_FEAT_STD[MT_WINDOW + i];
    }

    float out[4];   /* out0=MS滤波(归一化相对气压) out1=BMP滤波(归一化相对气压) out2=static out3=elevation */
    if(AI_Run_Inference(in, out) != 0)
        return false;

    /* 拦截推理异常值（NaN/Inf）：模型偶发会输出非有限值（输入瞬态异常、
     * 数值溢出等）。若不拦截，会污染下方偏置 IIR 并传播到融合/OLED，最终
     * 引发 HardFault 死机。故在此直接判定为本次推理失败、回退 KF。 */
    for(int k = 0; k < 4; k++) {
        if(!(out[k] == out[k]) || fabsf(out[k]) > 1.0e6f)
            return false;
    }

    /* 回归头输出为归一化相对气压，反归一化为绝对气压 */
    float rel_ms  = out[0] * MT_TARGET_STD + MT_TARGET_MEAN;
    float rel_bmp = out[1] * MT_TARGET_STD + MT_TARGET_MEAN;
    g_mt_filt_ms  = rel_ms + MT_REF_PRESSURE;
    g_mt_filt_bmp = rel_bmp + MT_REF_PRESSURE;

    /* 分类头：模型仍输出 [static, elevation] 概率，但该头在静止时恒判为
     * elevation（训练于合成数据所致），不可靠。场景判定已改由融合任务的
     * KF 帧间 Δh 幅度完成（见方案15/16），故此处不再用它覆盖全局场景量，
     * 否则会污染 SCENE 显示与 OLED。 */

    /* 双回归头均存在与真值（KF 已与真值对齐）的恒定系统偏置：
     * MS5611 头约 +44 Pa，BMP280 头约 +7 Pa（模型训练于合成数据所致）。
     * 用各自 KF 作基准，慢速估计并扣除该偏置，使 NN 输出与原始/KF 一致。
     *
     * 注意：不能依赖场景分类头（分类头在静止时也会误判为 elevation），
     * 故改用 KF 帧间变化率门限：仅在稳态（变化 < 0.3 Pa/帧）时更新偏置
     * 估计，运动瞬态/快速爬升时沿用上一估计，避免污染。偏置 IIR 做了
     * NaN 保护：一旦估计值非有限就跳过相减，防止把 NaN 写回全局状态。 */
    static float mt_ms_bias  = 0.0f;
    static float mt_bmp_bias = 0.0f;
    static float mt_ms_kf_prev  = 0.0f;
    static float mt_bmp_kf_prev = 0.0f;

    float ms_kf  = ms5611_data.pressure_filtered_pa;
    float bmp_kf = bmp280_data.pressure_filtered_pa;
    float ms_rate  = fabsf(ms_kf  - mt_ms_kf_prev);
    float bmp_rate = fabsf(bmp_kf - mt_bmp_kf_prev);
    mt_ms_kf_prev  = ms_kf;
    mt_bmp_kf_prev = bmp_kf;

    if(ms_rate < 0.3f)
        mt_ms_bias  += 0.02f * ((g_mt_filt_ms  - mt_ms_bias)  - ms_kf);
    if(bmp_rate < 0.3f)
        mt_bmp_bias += 0.02f * ((g_mt_filt_bmp - mt_bmp_bias) - bmp_kf);

    if(mt_ms_bias  == mt_ms_bias)  g_mt_filt_ms  -= mt_ms_bias;   /* 跳过 NaN */
    if(mt_bmp_bias == mt_bmp_bias) g_mt_filt_bmp -= mt_bmp_bias;

    /* NN vs KF 散度（方案16 场景判定用）：静止时偏置校正使 NN→KF → 散度≈0；
     * 运动时偏置冻结，NN 与 KF 跟踪特性差异 → 散度增大。EMA 平滑后输出。 */
    {
        static float nn_div_lp = 0.0f;
        float nn_div = fabsf(g_mt_filt_ms - ms_kf);
        nn_div_lp += NN_DIV_ALPHA * (nn_div - nn_div_lp);
        g_mt_nn_divergence = nn_div_lp;
    }

    g_mt_ready = true;
    return true;
}

const char* Multitask_SceneName(int pred)
{
    return (pred == 1) ? "elevation" : "static";
}

/* ========== 方案12：Hampel 脉冲抑制预处理器 ========== */
#if FUSION_SCHEME == 12
/**
 * @brief Hampel 滤波器：基于中值绝对偏差 (MAD) 的脉冲抑制
 * @param window  滑动窗口缓冲区（长度为 size）
 * @param new_val 当前新样本
 * @param size    窗口大小
 * @param idx     写入索引指针
 * @param filled  窗口是否已填满
 * @retval 滤波输出（正常值直通，离群值替换为中值）
 */
static float HampelFilter_Update(float *window, float new_val, int size, int *idx, bool *filled)
{
    /* 写入窗口 */
    window[*idx] = new_val;
    *idx = (*idx + 1) % size;
    if(!(*filled) && *idx == 0) *filled = true;

    /* 窗口未满时直通 */
    if(!(*filled)) return new_val;

    /* 将窗口数据拷贝到临时数组排序求中值 */
    float buf[5]; /* 固定最大窗口 5 */
    for(int i = 0; i < size; i++) buf[i] = window[i];

    /* 简单排序（冒泡，size 很小） */
    for(int i = 0; i < size-1; i++)
        for(int j = 0; j < size-1-i; j++)
            if(buf[j] > buf[j+1]) {
                float t = buf[j]; buf[j] = buf[j+1]; buf[j+1] = t;
            }

    float median = buf[size / 2];

    /* 计算绝对偏差并排序求 MAD */
    float abs_dev[5];
    for(int i = 0; i < size; i++)
        abs_dev[i] = fabsf(window[i] - median);

    for(int i = 0; i < size-1; i++)
        for(int j = 0; j < size-1-i; j++)
            if(abs_dev[j] > abs_dev[j+1]) {
                float t = abs_dev[j]; abs_dev[j] = abs_dev[j+1]; abs_dev[j+1] = t;
            }

    float mad = abs_dev[size / 2];
    float sigma = 1.4826f * mad;  /* 稳健标准差估计 */
    if(sigma < 0.001f) sigma = 0.001f;  /* 防止完全静止时 sigma=0 */

    /* 检测离群：超过 threshold × sigma 用中值替代 */
    if(fabsf(new_val - median) > HAMPEL_THRESHOLD * sigma)
        return median;
    else
        return new_val;
}
#endif /* FUSION_SCHEME == 12 */

void StartMS5611Task(void *argument)
{
  osDelay(50);  /* 等待调度器稳定 */
  uint32_t prev_tick = osKernelGetTickCount();
  
  for(;;)
  {
    if(ms5611_ready)
    {
      osMutexAcquire(SensorDataMutexHandle, osWaitForever);
      MS5611_Read_Data(&ms5611_data.pressure_pa, &ms5611_data.temperature_c);
      /* 原始高度和滤波高度：由 BMP280 任务统一用 BMP280 温度重算，此处只更新气压和温度 */
      ms5611_data.pressure_filtered_pa = KalmanFilter_Update_Adaptive(&ms5611_kf_pressure, ms5611_data.pressure_pa);
#if WORK_MODE == 0
      /* 新方案：仅把 MS5611 读数推入联合模型窗口；滤波 + 场景由 BMP280 任务统一推理 */
      Multitask_PushMS(ms5611_data.pressure_pa);
#else
      /* NN 滤波暂时使用 KF 数据替代 */
      ms5611_data.pressure_filtered_nn = ms5611_data.pressure_filtered_pa;
#endif
      osMutexRelease(SensorDataMutexHandle);
    }
    
    vTaskDelayUntil(&prev_tick, SAMPLE_PERIOD_MS);
  }
}

/* =====================================================================
 *                  融合引擎 Fusion_Compute()
 * ---------------------------------------------------------------------
 * 双传感器（MS5611 + BMP280）气压/高度融合的全部"方案"集中在此函数。
 * 编译期由 main.h 的 FUSION_SCHEME 选择唯一生效方案，其余不参与编译。
 *
 * 方案号 -> 职责（详见各方案内注释）：
 *   2  : 四路直接加权融合（MS5611_KF + BMP280_NN + BMP280_KF）
 *   4  : BMP280 主导 + MS5611 高频增强通道(HPF)
 *   5  : 自适应权重融合（静止BMP280主导 / 运动增大MS5611权重）
 *   6  : MS5611 主导(85%) + BMP280 KF(15%)，不用NN
 *   7  : BMP280定绝对高度 + 双传感器高度变化量加权累积(带误差抑制)
 *   8  : 仅 MS5611 KF（纯单传感器）
 *   9  : 仅 BMP280 NN+KF
 *   10 : 逆方差加权融合（抗突发噪声）
 *   11 : Delta 置信度加权累积融合（带泄漏锚）
 *   13 : 二阶互补融合（气压 + 气压变化率）
 *   14 : 方案4 + BMP280 温漂补偿
 *   15 : NN 主导场景门控增量锁定
 *   16 : KF 主导 + NN 散度场景门控增量锁定（默认新架构）
 *   17 : 无场景模式 — BMP280 KF 直接算高度
 *  其他: 方案1/3 — 气压域双传感器加权融合（MS5611只用KF, BMP280用NN）
 *
 * 高度计算：除 7/15/16/17 直接积分/直通外，其余由融合气压经 ISA 公式算高度。
 * ===================================================================== */

/* ---------------------------------------------------------------------
 * 逆方差置信加权 Δh 辅助（方案 11 / 15 / 16 共用）
 * 对两传感器帧间变化量窗口做方差估计，以 1/(var+ε) 为权重融合当前帧
 * delta_a / delta_b（方差越小 → 置信度越高 → 权重越大）。
 * 调用前需先把当前帧 delta 推入窗口并更新 filled。
 * ------------------------------------------------------------------- */
static float Fusion_ConfDelta(const float *win_a, const float *win_b,
                              int filled, float eps, float delta_a, float delta_b)
{
    float mean_a = 0.0f, mean_b = 0.0f;
    for (int i = 0; i < filled; i++) {
        mean_a += win_a[i];
        mean_b += win_b[i];
    }
    mean_a /= (float)filled;
    mean_b /= (float)filled;

    float var_a = 0.0f, var_b = 0.0f;
    for (int i = 0; i < filled; i++) {
        float da = win_a[i] - mean_a; var_a += da * da;
        float db = win_b[i] - mean_b; var_b += db * db;
    }
    var_a /= (filled > 1) ? (float)(filled - 1) : 1.0f;
    var_b /= (filled > 1) ? (float)(filled - 1) : 1.0f;

    float w_a = 1.0f / (var_a + eps);   /* 方差小 → 权重大 */
    float w_b = 1.0f / (var_b + eps);
    return (delta_a * w_a + delta_b * w_b) / (w_a + w_b);
}

static void Fusion_Compute(void)
{
#if FUSION_SCHEME == 2
        /* 方案2：四路直接加权融合 — 气压域融合 */
        /* 权重分配：MS5611_KF(0.50) + BMP280_NN(0.25) + BMP280_KF(0.25) */
        fusion_data.pressure_fused_pa = ms5611_data.pressure_filtered_pa * 0.50f
                                      + bmp280_data.pressure_filtered_nn * 0.25f
                                      + bmp280_data.pressure_filtered_pa * 0.25f;
        fusion_data.temperature_c = bmp280_data.temperature_c;
#elif FUSION_SCHEME == 4
        /* ========== 方案4：BMP280 主导 + MS5611 高频增强通道（气压域） ==========
         * BMP280 提供绝对精度（低噪声、低漂移），
         * MS5611 (KF 数据) 提供高频动态增量（24-bit ADC 捕捉快速变化）。
         *
         * 融合公式（气压域）：
         *   pressure_fused = BMP280_pressure + HPF(MS5611_KF_pressure - BMP280_pressure)
         *
         * 高通滤波器 (HPF) 提取 MS5611 的快速变化成分，
         * 缓慢偏差（如温度漂移、参考气压差异）被滤除，不会影响绝对精度。
         */
        {
            float bmp_p = bmp280_data.pressure_filtered_nn;      /* 已含全局偏置补偿 */
            float ms5611_p = ms5611_data.pressure_filtered_pa;   /* MS5611 只用 KF */

            /* MS5611 与 BMP280 之间的差值（偏置已在全局补偿，稳态≈0） */
            float ms5611_diff = ms5611_p - bmp_p;

            /* 一阶低通滤波提取直流偏差 */
            float lpf_output = HPF_ALPHA * ms5611_diff + (1.0f - HPF_ALPHA) * ms5611_hpf_last;
            ms5611_hpf_last = lpf_output;

            /* 高通 = 原始差值 - 低通直流偏差，只保留动态变化 */
            float hpf_output = ms5611_diff - lpf_output;

            /* 最终融合气压：BMP280（已偏置补偿）+ MS5611 高频动态增量 */
            fusion_data.pressure_fused_pa = bmp_p + hpf_output;
            fusion_data.temperature_c = bmp280_data.temperature_c;
        }
#elif FUSION_SCHEME == 5
        /* ========== 方案5：自适应权重融合（气压域） ==========
         * 基于 MS5611 KF 气压滑动窗口残差检测运动状态：
         * - 静止时：BMP280(95%) + MS5611(5%) — 极低噪声
         * - 运动时：BMP280(60%) + MS5611(40%) — 快速跟踪
         * - 权重平滑过渡：防止状态切换导致输出跳变
         * MS5611 只使用 KF 数据。
         */
        {
            /* 1. 计算 MS5611 的短期波动：用 KF 气压滑动窗口残差 */
            float ms5611_p = ms5611_data.pressure_filtered_pa;  /* MS5611 只用 KF */
            static float prev_ms5611_p = 0.0f;
            float residual = fabsf(ms5611_p - prev_ms5611_p);
            prev_ms5611_p = ms5611_p;

            motion_window[motion_window_idx] = residual;
            motion_window_idx = (motion_window_idx + 1) % MOTION_WINDOW_SIZE;

            /* 2. 判定运动状态：窗口内最大残差超过阈值 */
            float max_residual = 0.0f;
            for(int i = 0; i < MOTION_WINDOW_SIZE; i++)
            {
                if(motion_window[i] > max_residual)
                    max_residual = motion_window[i];
            }

            bool is_motion = (max_residual > MOTION_THRESHOLD_PA);

            /* 3. 根据运动状态计算目标权重 */
            float target_weight_ms = is_motion ? WEIGHT_MOTION_MS : WEIGHT_STATIC_MS;

            /* 4. 权重平滑过渡：避免跳变 */
            smooth_weight_ms = smooth_weight_ms
                             + WEIGHT_SMOOTH_ALPHA * (target_weight_ms - smooth_weight_ms);
            if(smooth_weight_ms < 0.0f) smooth_weight_ms = 0.0f;
            if(smooth_weight_ms > 1.0f) smooth_weight_ms = 1.0f;

            float smooth_weight_bmp = 1.0f - smooth_weight_ms;

            /* 5. 气压域加权融合（MS5611 只用 KF） */
            fusion_data.pressure_fused_pa = ms5611_data.pressure_filtered_pa * smooth_weight_ms
                                          + bmp280_data.pressure_filtered_nn * smooth_weight_bmp;
            fusion_data.temperature_c = bmp280_data.temperature_c;
        }
#elif FUSION_SCHEME == 6
        /* ========== 方案6：MS5611 主导（不使用 NN 滤波） ==========
         * 以 MS5611 KF 气压为主导（权重 85%），
         * 辅以 BMP280 KF 气压（权重 15%）提供稳定性，
         * 完全不用 NN 滤波结果。
         * 温度使用 BMP280 的读数。
         */
        fusion_data.pressure_fused_pa = ms5611_data.pressure_filtered_pa * 0.85f
                                      + bmp280_data.pressure_filtered_pa * 0.15f;
        fusion_data.temperature_c = bmp280_data.temperature_c;
#elif FUSION_SCHEME == 7
        /* ========== 方案7：自适应高度变化量加权融合（带错误抑制） ==========
         * 核心思想：融合高度从 BMP280 高度出发，累加加权后的帧间变化量，
         * 但加入"误差抑制"机制防止噪声累积漂移。
         *
         * 关键修正：
         *   - 使用融合高度自身做累加（而非 bmp_h + 单帧 delta），
         *     确保连续帧间的增量正确累积。
         *   - 静止时加入"拉力"：当传感器确认静止时，融合高度缓慢
         *     向 BMP280 高度靠拢，防止噪声累积导致的缓慢漂移。
         *   - 拉力与运动状态联动：运动时不拉，静止时以极慢速率拉回。
         *
         * 自适应逻辑：
         *   - 静止时：MS5611 delta 权重 ≈ 5%，BMP280 delta 权重 ≈ 95%
         *   - 运动时：MS5611 delta 权重 ≈ 50%，BMP280 delta 权重 ≈ 50%
         *
         * 公式：
         *   delta_ms  = MS5611_KF_height - MS5611_KF_height_prev
         *   delta_bmp = BMP280_NN_height - BMP280_NN_height_prev
         *   delta_fused = delta_ms * w_ms + delta_bmp * w_bmp
         *   height_fused += delta_fused
         *   height_fused += PULL_FORCE * (BMP280_height - height_fused)  [静止时]
         */
        {
            float ms_h = ms5611_data.height_filtered_m;     /* MS5611 KF 高度 */
            float bmp_h = bmp280_data.height_filtered_nn;   /* BMP280 NN 高度（绝对基准） */

            /* 预设/校零重锚：把融合高度与基准重新钉到新海拔（避免锁停留在旧值） */
            if(g_fusion_reanchor_req) {
                ms5611_height_prev = ms_h;
                bmp280_height_prev = bmp_h;
                fusion_data.height_fused_m = g_fusion_reanchor_alt;
                g_fusion_reanchor_req = false;
            }

            /* 首次运行初始化：融合高度以 BMP280 为准 */
            static bool first_run_7 = true;
            if(first_run_7) {
                ms5611_height_prev = ms_h;
                bmp280_height_prev = bmp_h;
                fusion_data.height_fused_m = bmp_h;  /* 初始化为 BMP280 高度 */
                fusion_data.pressure_fused_pa = bmp280_data.pressure_filtered_nn;
                fusion_data.temperature_c = bmp280_data.temperature_c;
                first_run_7 = false;
                /* 首帧不执行 delta 融合，直接跳到 #endif 后的代码 */
                goto skip_fusion_7;
            }

            /* 帧间高度变化量 */
            float delta_ms  = ms_h - ms5611_height_prev;
            float delta_bmp = bmp_h - bmp280_height_prev;

            /* 更新上一帧（供下一轮使用） */
            ms5611_height_prev = ms_h;
            bmp280_height_prev = bmp_h;

            /* --- 运动检测（基于 MS5611 KF 气压帧间残差） --- */
            float ms5611_p = ms5611_data.pressure_filtered_pa;
            float residual_7 = fabsf(ms5611_p - prev_ms5611_p_7);
            prev_ms5611_p_7 = ms5611_p;

            motion_window_7[motion_window_idx_7] = residual_7;
            motion_window_idx_7 = (motion_window_idx_7 + 1) % MOTION_WINDOW_SIZE;

            float max_residual_7 = 0.0f;
            for(int i = 0; i < MOTION_WINDOW_SIZE; i++) {
                if(motion_window_7[i] > max_residual_7)
                    max_residual_7 = motion_window_7[i];
            }
            bool is_motion_7 = (max_residual_7 > MOTION_THRESHOLD_PA);

            /* --- 自适应权重 --- */
            float target_weight_ms = is_motion_7 ? W_DELTA_MS_MOTION : W_DELTA_MS_STATIC;

            smooth_delta_weight_ms = smooth_delta_weight_ms
                + DELTA_WEIGHT_SMOOTH_ALPHA * (target_weight_ms - smooth_delta_weight_ms);
            if(smooth_delta_weight_ms < 0.0f) smooth_delta_weight_ms = 0.0f;
            if(smooth_delta_weight_ms > 1.0f) smooth_delta_weight_ms = 1.0f;

            float smooth_weight_bmp = 1.0f - smooth_delta_weight_ms;

            /* --- 变化量加权融合（累加模式） --- */
            float delta_fused = delta_ms * smooth_delta_weight_ms + delta_bmp * smooth_weight_bmp;

            /* 从上一帧融合高度累加 delta */
            fusion_data.height_fused_m += delta_fused;

            /* --- 静止误差抑制：缓慢拉回 BMP280 基准 --- */
            /* 静止时，因噪声累积的微小漂移会被此"拉力"缓慢消除。
             * PULL_COEFF 决定拉回速度：0.003f 相当于约 330 帧（~6.6秒）将误差消除 63%。
             * 运动时不做拉回，以免对抗真实的运动变化。 */
            if(!is_motion_7) {
                float drift = bmp_h - fusion_data.height_fused_m;
                fusion_data.height_fused_m += 0.003f * drift;
            }

            /* 气压和温度以 BMP280 为准 */
            fusion_data.pressure_fused_pa = bmp280_data.pressure_filtered_nn;
            fusion_data.temperature_c = bmp280_data.temperature_c;
        }
        skip_fusion_7: ;  /* 首帧初始化后跳过 delta 融合 */
#elif FUSION_SCHEME == 8
        /* ========== 方案8：只使用 MS5611 的 KF 值 ==========
         * 纯粹使用 MS5611 卡尔曼滤波后的气压值，
         * BMP280 不参与气压融合，仅提供温度参考。
         * 适用于仅依赖 MS5611 单传感器输出的场景。
         */
        fusion_data.pressure_fused_pa = ms5611_data.pressure_filtered_pa;
        fusion_data.temperature_c = bmp280_data.temperature_c;
#elif FUSION_SCHEME == 9
        /* ========== 方案9：只使用 BMP280 的 NN 和 KF（权重 70%） ==========
         * 仅使用 BMP280 传感器数据，MS5611 不参与融合。
         * BMP280 NN(70%) + KF(30%) 已在传感器内部融合为 pressure_filtered_nn。
         * 温度使用 BMP280 自身的读数。
         */
        fusion_data.pressure_fused_pa = bmp280_data.pressure_filtered_nn;
        fusion_data.temperature_c = bmp280_data.temperature_c;
#elif FUSION_SCHEME == 10
        /* ========== 方案10：逆方差加权融合（创新方案 — 抗突发噪声） ==========
         * 利用双传感器实时方差的倒数作为自适应权重：
         *   - MS5611 发生跳变时方差暴增 → 权重自动归零 → 完全信任 BMP280
         *   - 两者都平稳时，低方差传感器占主导
         * 公式（逆方差加权，带正则化 ε 防止极端权重）：
         *   P_fused = (P_BMP * (σ²_MS + ε) + P_MS * (σ²_BMP + ε))
         *            / (σ²_BMP + σ²_MS + 2ε)
         * MS5611 只使用 KF 数据，BMP280 使用 NN 数据。
         */
        {
            float ms5611_p = ms5611_data.pressure_filtered_pa;  /* MS5611 KF */
            float bmp_p = bmp280_data.pressure_filtered_nn;     /* BMP280 NN */

            /* 更新滑动窗口缓冲区 */
            ivar_ms5611_buf[ivar_idx] = ms5611_p;
            ivar_bmp280_buf[ivar_idx] = bmp_p;
            ivar_idx = (ivar_idx + 1) % IVAR_WINDOW_SIZE;
            if(ivar_filled < IVAR_WINDOW_SIZE) ivar_filled++;

            int n = ivar_filled;
            float mean_ms = 0.0f, mean_bmp = 0.0f;
            for(int i = 0; i < n; i++) {
                mean_ms += ivar_ms5611_buf[i];
                mean_bmp += ivar_bmp280_buf[i];
            }
            mean_ms /= n;
            mean_bmp /= n;

            /* 计算方差（无偏估计） */
            float var_ms = 0.0f, var_bmp = 0.0f;
            for(int i = 0; i < n; i++) {
                float d = ivar_ms5611_buf[i] - mean_ms;
                var_ms += d * d;
                d = ivar_bmp280_buf[i] - mean_bmp;
                var_bmp += d * d;
            }
            var_ms /= (n > 1) ? (n - 1) : 1.0f;
            var_bmp /= (n > 1) ? (n - 1) : 1.0f;

            /* 逆方差加权：BMP280 的权重 = MS5611 方差 + ε
             * MS5611 的权重 = BMP280 方差 + ε */
            float w_bmp = var_ms + IVAR_EPSILON;   /* BMP280 权重 */
            float w_ms  = var_bmp + IVAR_EPSILON;  /* MS5611 权重 */
            float w_sum = w_bmp + w_ms;

            fusion_data.pressure_fused_pa = (bmp_p * w_bmp + ms5611_p * w_ms) / w_sum;
            fusion_data.temperature_c = bmp280_data.temperature_c;
        }
#elif FUSION_SCHEME == 11
        /* ========== 方案11：Delta 置信度加权累积融合（气压域） ==========
         * 核心：不融合绝对气压，而是融合帧间气压变化量（Delta），
         * 累加跟踪相对高度，配"泄漏锚"防漂移。
         * 置信度 = 1/(var(delta_window) + ε)，窗口内 Delta 方差越小 → 越稳定 → 置信度越高。
         * 泄漏锚：静止时缓慢拉回 BMP280 绝对气压，消除累积漂移。
         */
        {
            float ms5611_p = ms5611_data.pressure_filtered_pa;  /* MS5611 KF */
            float bmp_p = bmp280_data.pressure_filtered_nn;     /* BMP280 NN */

            /* 首帧初始化 */
            if(delta11_first_run) {
                delta11_prev_ms5611_p = ms5611_p;
                delta11_prev_bmp280_p = bmp_p;
                delta11_fused_pa = bmp_p;
                delta11_first_run = false;
                fusion_data.pressure_fused_pa = bmp_p;
                fusion_data.temperature_c = bmp280_data.temperature_c;
                goto skip_fusion_11;
            }

            /* 帧间 Delta */
            float delta_ms = ms5611_p - delta11_prev_ms5611_p;
            float delta_bmp = bmp_p - delta11_prev_bmp280_p;
            delta11_prev_ms5611_p = ms5611_p;
            delta11_prev_bmp280_p = bmp_p;

            /* 更新 Delta 滑动窗口 */
            delta11_ms_window[delta11_idx] = delta_ms;
            delta11_bmp_window[delta11_idx] = delta_bmp;
            delta11_idx = (delta11_idx + 1) % DELTA_CONF_WINDOW;
            if(delta11_filled < DELTA_CONF_WINDOW) delta11_filled++;

            /* 逆方差置信加权 Δh（窗口方差估计，复用方案11/15/16 同一实现） */
            float delta_fused = Fusion_ConfDelta(delta11_ms_window, delta11_bmp_window,
                                                 delta11_filled, DELTA_CONF_EPS,
                                                 delta_ms, delta_bmp);

            /* 累积到融合气压 */
            delta11_fused_pa += delta_fused;

            /* 泄漏锚：缓慢拉回 BMP280 绝对气压，消除累积漂移 */
            delta11_fused_pa += ANCHOR_ALPHA * (bmp_p - delta11_fused_pa);

            fusion_data.pressure_fused_pa = delta11_fused_pa;
            fusion_data.temperature_c = bmp280_data.temperature_c;
        }
        skip_fusion_11: ;
#elif FUSION_SCHEME == 13
        /* ========== 方案13：二阶互补融合（气压 + 气压变化率） ==========
         * 状态 (P_fused 融合气压, D_fused 融合变化率)：
         *   预测步：MS5611 提供好的短期速度估计（Delta）
         *           D_fused = COMP_β × D_ms + (1-COMP_β) × D_fused
         *   更新步：P_fused 先按 D_fused 速度外推，再被 BMP280 慢速锚定
         * 二阶系统对随机噪声的抑制比一阶互补好一倍，且对 10cm 级台阶响应更灵敏。
         */
        {
            float ms5611_p = ms5611_data.pressure_filtered_pa;  /* MS5611 KF */
            float bmp_p = bmp280_data.pressure_filtered_nn;     /* BMP280 NN */

            /* 首帧初始化 */
            if(so13_first_run) {
                so13_P_fused = bmp_p;
                so13_D_fused = 0.0f;
                so13_prev_ms5611_p = ms5611_p;
                so13_first_run = false;
                fusion_data.pressure_fused_pa = bmp_p;
                fusion_data.temperature_c = bmp280_data.temperature_c;
                goto skip_fusion_13;
            }

            /* 预测步：MS5611 主导速度估算 */
            float D_ms = ms5611_p - so13_prev_ms5611_p;         /* MS5611 的速度估计 */
            so13_prev_ms5611_p = ms5611_p;
            so13_D_fused = COMP_BETA * D_ms + (1.0f - COMP_BETA) * so13_D_fused;

            /* 更新步：用速度外推 + BMP280 锚定修正 */
            so13_P_fused = so13_P_fused + so13_D_fused;          /* 速度外推 */
            so13_P_fused += COMP_ALPHA * (bmp_p - so13_P_fused); /* BMP280 慢速锚定 */

            fusion_data.pressure_fused_pa = so13_P_fused;
            fusion_data.temperature_c = bmp280_data.temperature_c;
        }
        skip_fusion_13: ;
#elif FUSION_SCHEME == 14
        /* ========== 方案14：方案4 + BMP280 温漂补偿增强版 ==========
         * 在方案4的 BMP280 主导 + MS5611 高频增强通道基础上，
         * 新增 BMP280 实时温度漂移补偿（已在 BMP280 预处理阶段完成）。
         *
         * 融合公式（气压域）：
         *   pressure_fused = BMP280_p(温补后) + HPF(MS5611_KF - BMP280_p(温补后))
         *
         * 高通滤波器 (HPF) 提取 MS5611 的快速变化成分，
         * 温度补偿抑制空调温循环导致的 BMP280 缓慢漂移。
         */
        {
            float bmp_p = bmp280_data.pressure_filtered_nn;      /* 已含全局偏置 + 温漂补偿 */
            float ms5611_p = ms5611_data.pressure_filtered_pa;   /* MS5611 只用 KF */

            /* MS5611 与 BMP280 之间的差值 */
            float ms5611_diff = ms5611_p - bmp_p;

            /* 一阶低通滤波提取直流偏差 */
            float lpf_output = HPF_ALPHA * ms5611_diff + (1.0f - HPF_ALPHA) * tc14_hpf_last;
            tc14_hpf_last = lpf_output;

            /* 高通 = 原始差值 - 低通直流偏差，只保留动态变化 */
            float hpf_output = ms5611_diff - lpf_output;

            /* 最终融合气压：BMP280（偏置+温补）+ MS5611 高频动态增量 */
            fusion_data.pressure_fused_pa = bmp_p + hpf_output;
            fusion_data.temperature_c = bmp280_data.temperature_c;
        }
#elif FUSION_SCHEME == 15
        /* ========== 方案15：原始气压窗口方差门控 + 原始气压 ISA 相对高度 ==========
         * 针对原 0.4m 误差的优化：
         *   1) 相对高度改用【原始气压】经 ISA 作差，绕开 KF 自适应滤波的启动滞后
         *      （KF 需 ~5 帧残差窗口才放大 Q 提速，运动初段严重滞后；原方案在静止瞬间
         *      冻结，把滞后偏低的 KF 高度钉死 → 系统性偏低）。原始气压无滞后，即时跟到真值。
         *   2) 场景判定改用【原始气压滑动窗口方差 STD】：静止 STD≈噪声(~0.35Pa)，
         *      运动 STD 达数~数十 Pa，与速度/幅度无关、对瞬态敲击不敏感，比每帧 Δh 阈值稳健。
         * 结构：static→elevation 跳变记录起点气压；运动时相对高度 = 起点高度 + ISA(当前,起点)；
         *       静止完全冻结。路径无关、无累积。 */
        {
            /* 预设/校零重锚：把相对高度钉到设定值，并重置运动锚点与场景窗口 */
            if(g_fusion_reanchor_req) {
                h15_lock = g_fusion_reanchor_alt;
                first_run_15 = false;       /* 用设定值直接锁定，跳过首帧锚定 */
                s15_locked = true;          /* 设定值本身即稳定值，不需再等启动稳定期 */
                s15_warmup = 0;
                s15_motion_ref_pressure = bmp280_data.pressure_pa;
                s15_motion_ref_height   = g_fusion_reanchor_alt;
                s15_fast_gate           = false;
                s15_static_streak       = 0;
                s15_settle_cnt          = 0;           /* 复位沉降等待，切换后直接锁定设定值 */
                s15_pbuf_fill = 0; s15_pbuf_idx = 0;   /* 清空窗口，避免旧数据判场景 */
                s15_fbuf_fill = 0; s15_fbuf_idx = 0;
                g_fusion_reanchor_req = false;
            }

            /* 启动稳定期：上电/复位后先等 S15_WARMUP_FRAMES 帧，避免未稳定首帧被锁死 */
            if(!s15_locked) {
                h15_lock = bmp280_data.height_filtered_m;   /* 暂跟随绝对高度 */
                if(++s15_warmup >= S15_WARMUP_FRAMES) {
                    s15_locked = true;      /* 下一帧起 first_run_15 锚定 */
                }
            } else if(first_run_15) {
                h15_lock = bmp280_data.height_filtered_m;   /* 初始锚定 */
                first_run_15 = false;
                s15_motion_ref_pressure = bmp280_data.pressure_pa;
                s15_motion_ref_height   = h15_lock;
                s15_fast_gate           = false;
                s15_static_streak       = 0;
                s15_settle_cnt          = 0;
                s15_pbuf_fill = 0; s15_pbuf_idx = 0;
                s15_fbuf_fill = 0; s15_fbuf_idx = 0;
                } else {
                /* ---- 场景判定：长窗口方差STD门控 + 短窗口STD门控(OR) ----
                 * 长窗口(20帧)STD：捕捉持续/慢速运动；短窗口(5帧)STD：捕捉快速小幅运动
                 * (25cm仅~3Pa，被长窗口平均稀释后STD低于开门阈值→原方案漏检；短窗口横跨
                 *  跳变、STD陡升~1.2Pa→触发)。短窗口对真实传感器低频漂移免疫。
                 * 各自独立 Schmitt 迟滞，最终门控 = 长 OR 短。 */
                float p = bmp280_data.pressure_pa;
                /* 长窗口（与门控状态无关地推进） */
                float win_oldest = (s15_pbuf_fill >= S15_MOT_WIN)
                                ? s15_pbuf[s15_pbuf_idx] : p;
                s15_pbuf[s15_pbuf_idx] = p;
                s15_pbuf_idx = (s15_pbuf_idx + 1) % S15_MOT_WIN;
                if(s15_pbuf_fill < S15_MOT_WIN) s15_pbuf_fill++;
                float lmean = 0.0f;
                for(int i = 0; i < s15_pbuf_fill; i++) lmean += s15_pbuf[i];
                lmean /= (float)s15_pbuf_fill;
                float lvar = 0.0f;
                for(int i = 0; i < s15_pbuf_fill; i++) {
                    float d = s15_pbuf[i] - lmean; lvar += d * d;
                }
                float lstd = sqrtf(lvar / (float)s15_pbuf_fill);
                /* 短窗口（快速小幅检测） */
                s15_fbuf[s15_fbuf_idx] = p;
                s15_fbuf_idx = (s15_fbuf_idx + 1) % S15_FAST_WIN;
                if(s15_fbuf_fill < S15_FAST_WIN) s15_fbuf_fill++;
                float fmean = 0.0f;
                for(int i = 0; i < s15_fbuf_fill; i++) fmean += s15_fbuf[i];
                fmean /= (float)s15_fbuf_fill;
                float fvar = 0.0f;
                for(int i = 0; i < s15_fbuf_fill; i++) {
                    float d = s15_fbuf[i] - fmean; fvar += d * d;
                }
                float fstd = sqrtf(fvar / (float)s15_fbuf_fill);
                /* 长窗口 Schmitt */
                if(!gate15_state) {
                    if(lstd > S15_MOT_STD_OPEN)  gate15_state = true;
                } else {
                    if(lstd < S15_MOT_STD_CLOSE) gate15_state = false;
                }
                /* 短窗口 Schmitt（快速小幅） */
                if(!s15_fast_gate) {
                    if(fstd > S15_FAST_STD_OPEN)  s15_fast_gate = true;
                } else {
                    if(fstd < S15_FAST_STD_CLOSE) s15_fast_gate = false;
                }
                bool s15_gate_open = gate15_state || s15_fast_gate;

                /* ---- 减滞后延时锁定：判静止后不立即冻结，继续积分 S15_SETTLE_FRAMES 帧 ----
                 * 门控 STD 迟滞可能在平台尚未完全沉降时提前关门，若立即冻结会锁在半路值。
                 * 关门后维持一段沉降等待期继续积分，待气压完全沉降后再冻结（与方案16 同策略）。 */
                if(s15_gate_open) s15_settle_cnt = S15_SETTLE_FRAMES;  /* 运动中：刷新延时窗口 */
                else if(s15_settle_cnt > 0) s15_settle_cnt--;         /* 静止后：延时窗口倒计时 */
                bool s15_integrate = s15_gate_open || (s15_settle_cnt > 0);

                /* 连续静止帧计数（基于真实门控 s15_gate_open，用于重锚边沿判定，防抖动重锚） */
                if(s15_gate_open) s15_static_streak = 0;
                else              s15_static_streak++;

                /* 同步全局场景显示（SCENE 行 / OLED，沿用真实门控，不含沉降等待期） */
                g_mt_scene_pred = s15_gate_open ? 1 : 0;
                g_mt_scene[0] = s15_gate_open ? 0.0f : 1.0f;   /* static 概率 */
                g_mt_scene[1] = s15_gate_open ? 1.0f : 0.0f;   /* elevation 概率 */

                /* ---- 相对高度：原始气压两点 ISA（无 KF 滞后） ----
                 * 仅在「连续静止足够帧后转运动」(真正运动起始边沿)才重锚，回溯到运动前
                 * 静止气压(win_oldest)，消除开门延迟欠读；避免快速运动短窗口抖动(flicker)
                 * 反复重锚导致高度重复计算/卡在半路。 */
                if((s15_static_streak >= S15_REANCHOR_MIN) && s15_gate_open) {
                    s15_motion_ref_pressure = win_oldest;
                    s15_motion_ref_height   = h15_lock;
                }
                /* 门控积分：运动期 + 沉降等待期均积分；沉降结束后冻结（气压已完全沉降） */
                if(s15_integrate) {
                    /* 上升→current<ref→正值→高度增；下降→负值→减；回原位→0，无累积。 */
                    float rel_alt = PressureToAltitudeISA(p, s15_motion_ref_pressure);
                    h15_lock = s15_motion_ref_height + rel_alt;
                }
                /* else: 静止模式——完全冻结 h15_lock，不向任何参考锚定 */
            }
            /* 气压/温度透传（高度已由 h15_lock 直接给出） */
            fusion_data.pressure_fused_pa = bmp280_data.pressure_filtered_nn;
            fusion_data.temperature_c = bmp280_data.temperature_c;
        }
#elif FUSION_SCHEME == 16
        /* ========== 方案16：KF 主导 + NN 散度场景门控增量锁定 ==========
         * 新架构（方案 B）：NN 仅用于场景判断，融合数值全部由 KF 计算。
         *   - 融合入口：自适应卡尔曼滤波高度 ht_ms_kf / ht_bmp_kf
         *   - 场景判定：NN vs KF 散度（|NN_filtered - KF| EMA + Schmitt 门控）
         *   - 结构：逆方差置信加权 Δh + 门控积分（静止锁死 / 升降积分）
         */
        {
            float ht_bmp_kf = bmp280_data.height_filtered_m;   /* KF 降噪 BMP280 高度（已含偏置） */

            /* 预设/校零重锚：把积分锁钉到新海拔，并重置上一帧基准（下一帧 Δ≈0） */
            if(g_fusion_reanchor_req) {
                h16_lock = g_fusion_reanchor_alt;
                /* 同时重置运动起始参考与窗口状态，避免切换后首次运动用旧气压/旧数据 */
                s16_motion_ref_pressure = bmp280_data.pressure_pa;   /* 原始气压：无滞后、升降对称 */
                s16_motion_ref_height    = g_fusion_reanchor_alt;
                gate16_state = false;
                s16_fast_gate = false;
                s16_static_streak = 0;
                s16_settle_cnt = 0;                    /* 复位沉降等待，切换后直接锁定设定值 */
                s16_rest_p = bmp280_data.pressure_pa;  /* 重置静息基准，防切换瞬间误触发 2Pa 跳出 */
                s16_pbuf_fill = 0; s16_pbuf_idx = 0;
                s16_fbuf_fill = 0; s16_fbuf_idx = 0;
                g_fusion_reanchor_req = false;
            }

            if(first_run_16) {
                h16_lock = ht_bmp_kf;        /* 初始锚定 BMP280（KF 降噪后） */
                s16_motion_ref_pressure = bmp280_data.pressure_pa;   /* 原始气压：无滞后、升降对称 */
                s16_motion_ref_height   = h16_lock;
                gate16_state = false;
                s16_fast_gate = false;
                s16_static_streak = 0;
                s16_settle_cnt = 0;
                s16_rest_p = bmp280_data.pressure_pa;  /* 首帧静息基准 */
                s16_pbuf_fill = 0; s16_pbuf_idx = 0;
                s16_fbuf_fill = 0; s16_fbuf_idx = 0;
                s16_stable_cnt = 0;          /* 启动稳定计数器归零 */
                first_run_16 = false;
            } else {
                /* ---- 场景判定：长窗口方差STD门控 + 短窗口STD门控(OR) ----
                 * 原方案16 用「KF 高度 Δh 幅度 EMA」判场景：慢速小幅运动每帧偏移低于
                 * 阈值(0.0118m)→门控永不打开→一直静态（实测现象）。改用语压双窗口 STD
                 * （与方案15 同源，sim_scheme15_v2.py 验证）：长窗口(20帧)STD 捕捉持续/
                 * 慢速运动，短窗口(5帧)STD 捕捉快速小幅(25cm)运动，对 KF 滤波/低频漂移免疫。
                 * 最终门控 = 长 OR 短，各自独立 Schmitt 迟滞。相对高度改用原始气压→ISA
                 * （与已验证方案15 完全一致：门控/重锚/积分同源，无 KF 滞后、升降对称）。 */
                float p = bmp280_data.pressure_pa;   /* 原始气压：门控/重锚/积分同源，与方案15 一致 */
                /* 长窗口（与门控状态无关地推进；窗口存原始气压，供重锚回溯 win_oldest） */
                float win_oldest = (s16_pbuf_fill >= S16_MOT_WIN)
                                ? s16_pbuf[s16_pbuf_idx] : p;
                s16_pbuf[s16_pbuf_idx] = p;
                s16_pbuf_idx = (s16_pbuf_idx + 1) % S16_MOT_WIN;
                if(s16_pbuf_fill < S16_MOT_WIN) s16_pbuf_fill++;
                float lmean = 0.0f;
                for(int i = 0; i < s16_pbuf_fill; i++) lmean += s16_pbuf[i];
                lmean /= (float)s16_pbuf_fill;
                float lvar = 0.0f;
                for(int i = 0; i < s16_pbuf_fill; i++) {
                    float d = s16_pbuf[i] - lmean; lvar += d * d;
                }
                float lstd = sqrtf(lvar / (float)s16_pbuf_fill);
                /* 短窗口（快速小幅检测） */
                s16_fbuf[s16_fbuf_idx] = p;
                s16_fbuf_idx = (s16_fbuf_idx + 1) % S15_FAST_WIN;
                if(s16_fbuf_fill < S15_FAST_WIN) s16_fbuf_fill++;
                float fmean = 0.0f;
                for(int i = 0; i < s16_fbuf_fill; i++) fmean += s16_fbuf[i];
                fmean /= (float)s16_fbuf_fill;
                float fvar = 0.0f;
                for(int i = 0; i < s16_fbuf_fill; i++) {
                    float d = s16_fbuf[i] - fmean; fvar += d * d;
                }
                float fstd = sqrtf(fvar / (float)s16_fbuf_fill);
                /* 长窗口 Schmitt */
                if(!gate16_state) {
                    if(lstd > S15_MOT_STD_OPEN)  gate16_state = true;
                } else {
                    if(lstd < S15_MOT_STD_CLOSE) gate16_state = false;
                }
                /* 短窗口 Schmitt（快速小幅） */
                if(!s16_fast_gate) {
                    if(fstd > S15_FAST_STD_OPEN)  s16_fast_gate = true;
                } else {
                    if(fstd < S15_FAST_STD_CLOSE) s16_fast_gate = false;
                }
                bool s16_gate_open = gate16_state || s16_fast_gate;

                /* 静息基线(慢EMA) + 累计偏差 >2Pa 强制跳出静态：
                 * 上一版用「静止时每帧把基准刷新为当前值」会导致慢速运动时基准追平当前值、
                 * 偏差永远超不过 2Pa → 仍一直静态。改为慢 EMA 基线(alpha 极小，仅跟随温漂/极
                 * 慢漂移，跟不上真实运动)，故任意速度运动都会在基线尚未追上时积累 >2Pa 偏差而
                 * 开门。相对高度已完全改用原始气压积分，此处慢 EMA 仅用于门控辅助判定。 */
                float praw = bmp280_data.pressure_pa;
                s16_rest_p += S16_BASE_ALPHA * (praw - s16_rest_p);
                if(fabsf(praw - s16_rest_p) > S16_DELTA_EXIT_PA) s16_gate_open = true;

                /* 启动稳定期：强制静态，避免上电温漂瞬变误触发 */
                if(s16_stable_cnt < S16_STARTUP_STABLE_FRAMES) {
                    s16_stable_cnt++;
                    s16_gate_open = false;
                }

                /* ---- 减滞后延时锁定：判静止后不立即冻结，继续积分 S16_SETTLE_FRAMES 帧 ----
                 * 门控 STD 迟滞可能在平台尚未完全沉降时提前关门，若立即冻结会锁在半路值。
                 * 关门后维持一段沉降等待期继续积分，待气压完全沉降(原始气压已稳定)后再冻结，
                 * 冻结值即准确值。运动期间持续刷新计数；此策略与方案15 同，仅推迟锁定时刻。 */
                if(s16_gate_open) s16_settle_cnt = S16_SETTLE_FRAMES;  /* 运动中：刷新延时窗口 */
                else if(s16_settle_cnt > 0) s16_settle_cnt--;         /* 静止后：延时窗口倒计时 */
                bool s16_integrate = s16_gate_open || (s16_settle_cnt > 0);

                /* 同步全局场景显示（OLED 用 g_mt_scene_pred，沿用真实门控，不含沉降等待期） */
                g_mt_scene_pred = s16_gate_open ? 1 : 0;
                g_mt_scene[0] = s16_gate_open ? 0.0f : 1.0f;   /* static 概率 */
                g_mt_scene[1] = s16_gate_open ? 1.0f : 0.0f;   /* elevation 概率 */

                /* ---- 运动起始重锚：仅连续静止≥N帧后转运动(真正边沿)才重锚 ----
                 * 关键：边沿判定必须放在 s16_static_streak 重置之前，否则门控开门那帧
                 * streak 已被清零 → 条件永不成立 → 重锚变死代码、参考气压一直停留在
                 * 开机/上次校零值，多段升降累积漂移。
                 * 重锚参考用 win_oldest（长窗口最旧原始气压，即运动开始前约 S16_MOT_WIN
                 * 帧的静止气压），回溯开门延迟，消除运动初段欠读——这是方案15 已验证的关键。
                 * 用原始气压 → 无 KF 滞后、升降完全对称，每段运动从自身起点重新基准化。 */
                if((s16_static_streak >= S15_REANCHOR_MIN) && s16_gate_open) {
                    s16_motion_ref_pressure = win_oldest;
                    s16_motion_ref_height   = h16_lock;
                }

                /* 连续静止帧计数（基于真实门控 s16_gate_open，用于下一帧重锚边沿判定，防抖动重锚） */
                if(s16_gate_open) s16_static_streak = 0;
                else              s16_static_streak++;

                /* 门控积分：运动期 + 沉降等待期均积分；沉降结束后冻结。
                 * 相对高度积分改用原始气压 pressure_pa（与已验证方案15 同源）：无 KF 滞后、
                 * 升降对称，从根上消除「下降欠计数 / 下不到位」。KF 气压仅保留给显示/门控。 */
                if(s16_integrate) {
                    float rel_alt = PressureToAltitudeISA(bmp280_data.pressure_pa,
                                                          s16_motion_ref_pressure);
                    h16_lock = s16_motion_ref_height + rel_alt;
                }
                /* else: 静止模式——完全冻结 h16_lock，不向任何参考锚定 */
            }
            /* 气压以 BMP280 KF 为准（高度已直接积分） */
            fusion_data.pressure_fused_pa = bmp280_data.pressure_filtered_pa;
            fusion_data.temperature_c = bmp280_data.temperature_c;
        }
#elif FUSION_SCHEME == 17
        /* ========== 方案17：无场景模式 — BMP280 KF 直接算高度 ==========
         * 无门控、无运动重锚、无冻结。直接用 BMP280 KF 滤波后的气压
         * 通过 ISA 公式算绝对高度。安静稳定，适合固定海拔参考。 */
        {
            fusion_data.pressure_fused_pa = bmp280_data.pressure_filtered_pa;
            fusion_data.temperature_c = bmp280_data.temperature_c;
        }
#elif FUSION_SCHEME == 20 || FUSION_SCHEME == 21 || FUSION_SCHEME == 22
        /* ========== 方案20/21/22：MS5611/BMP280 各跑气压域 EKF + BP 去噪 → 多页对照 ========== */
        {
            uint32_t now = osKernelGetTickCount();
            float dt = (s20_prev_tick == 0) ? 0.02f : (now - s20_prev_tick) / 1000.0f;
            if (dt <= 0.0f || dt > 1.0f) dt = 0.02f;
            s20_prev_tick = now;

            float ms_p0 = reference_pressure_pa;                  /* MS5611 基准 */
#if FUSION_SCHEME == 21 || FUSION_SCHEME == 22
            /* 方案21/22：两颗 EKF 共用同一 P0。传感器系统性差异由在线偏置估计吸收，
             * 不再用 S20_BM_P0_OFFSET 制造 P0 差，避免两 EKF 对气压变化响应速率
             * 不同 → 偏置追差异 → 融合输出持续漂移。 */
            float bm_p0 = reference_pressure_pa;
#else
            float bm_p0 = reference_pressure_pa + S20_BM_P0_OFFSET; /* BMP280 偏差补偿 */
#endif

            if (!s20_inited) {
                BaroEKF_Init(&s20_ms_ekf, ms_p0, dt, S20_EKF_Q_PRESS, S20_EKF_Q_RATE, S20_EKF_R_MEAS);
                BaroEKF_Init(&s20_bm_ekf, bm_p0, dt, S20_EKF_Q_PRESS, S20_EKF_Q_RATE, S20_EKF_R_MEAS);
                s20_inited = 1;
            } else {
                /* 运行时归零/重校准会改写 reference_pressure_pa，同步到 EKF 基准
                 * （只读共享值，不回写全局，避免多任务隐患） */
                BaroEKF_SetP0(&s20_ms_ekf, ms_p0);
                BaroEKF_SetP0(&s20_bm_ekf, bm_p0);
            }

            /* ---- MS5611：气压域 EKF ---- */
            BaroEKF_Update(&s20_ms_ekf, ms5611_data.pressure_pa, ms5611_data.temperature_c);
            float ms_ekf_pa = BaroEKF_GetPressure(&s20_ms_ekf);
            /* ---- MS5611：BP 去噪（5 点滑窗，输入 hPa） ---- */
            float ms_hpa = ms5611_data.pressure_pa / 100.0f;
            float ms_bp_hpa = BP_Denoise_Update(&s20_ms_bp, BP_MS_W0, BP_MS_B0, BP_MS_W2, BP_MS_B2,
                                                ms_hpa, S20_BP_WIN_SCALE, S20_BP_EMA_ALPHA);
            float ms_bp_pa = s20_ms_bp.inited ? (ms_bp_hpa * 100.0f) : ms5611_data.pressure_pa;

            /* ---- BMP280：气压域 EKF ---- */
            BaroEKF_Update(&s20_bm_ekf, bmp280_data.pressure_pa, bmp280_data.temperature_c);
            float bm_ekf_pa = BaroEKF_GetPressure(&s20_bm_ekf);
            /* ---- BMP280：BP 去噪（5 点滑窗，输入 hPa） ---- */
            float bm_hpa = bmp280_data.pressure_pa / 100.0f;
            float bm_bp_hpa = BP_Denoise_Update(&s20_bm_bp, BP_BM_W0, BP_BM_B0, BP_BM_W2, BP_BM_B2,
                                                bm_hpa, S20_BP_WIN_SCALE, S20_BP_EMA_ALPHA);
            float bm_bp_pa = s20_bm_bp.inited ? (bm_bp_hpa * 100.0f) : bmp280_data.pressure_pa;

            /* ---- 温度补偿 ISA 高度换算 ---- */
            s20_out.ms_ekf_pa = ms_ekf_pa;
            s20_out.ms_ekf_alt = PressureToAltitudeWithTemp(ms_ekf_pa, ms_p0, ms5611_data.temperature_c);
            s20_out.ms_bp_pa = ms_bp_pa;
            s20_out.ms_bp_alt = PressureToAltitudeWithTemp(ms_bp_pa, ms_p0, ms5611_data.temperature_c);
            s20_out.bm_ekf_pa = bm_ekf_pa;
            s20_out.bm_ekf_alt = PressureToAltitudeWithTemp(bm_ekf_pa, bm_p0, bmp280_data.temperature_c);
            s20_out.bm_bp_pa = bm_bp_pa;
            s20_out.bm_bp_alt = PressureToAltitudeWithTemp(bm_bp_pa, bm_p0, bmp280_data.temperature_c);
            s20_out.ms_temp = ms5611_data.temperature_c;
            s20_out.bm_temp = bmp280_data.temperature_c;

#if FUSION_SCHEME == 21 || FUSION_SCHEME == 22
            /* ---- 方案21/22：双 EKF 残差方差倒数加权融合（气压域） ----
             * 实测两传感器“本来差距就大”且融合“波动巨大”，分两步修复：
             * (1) 在线偏置：极慢 EMA 估计 BMP280 相对 MS5611 的真实气压差，
             *     融合前扣除 → 两路落到同一基准，吸收任意恒定/缓变差距。
             * (2) 权重平滑+限幅：残差方差倒数权重先做长 EMA 平滑、再限幅到
             *     [W_MIN,W_MAX]，消除帧间权重抖动（波动巨大的主因）。 */
            {
                static float s21_bias = 0.0f;     /* BMP280 相对 MS5611 偏置（估计收敛后冻结） */
                static float s21_var_ms = 0.0f, s21_var_bm = 0.0f;
                static float s21_w_ms = 0.5f;     /* 平滑后的 MS5611 权重 */
                static float s21_p_lp = 0.0f;     /* 融合输出低通 */
                static int   s21_init = 0;
                static int   s21_bias_cnt = 0;    /* 偏置收敛期计数器 */
#if FUSION_SCHEME == 22
                static float s22_p_fast = 0.0f;   /* 方案22 自适应低通：快低通 */
                static float s22_p_slow = 0.0f;   /* 方案22 自适应低通：慢低通 */
#endif

                /* (1) 在线偏置：两段式
                 *     前 S21_BIAS_CONVERGE_FRAMES 帧：初始值+较快 EMA 收敛
                 *     之后：冻结（不再更新），彻底消除偏置追噪声导致的漂移 */
                if (!s21_init) {
                    s21_bias = bm_ekf_pa - ms_ekf_pa;
                    s21_bias_cnt = 0;
                } else if (s21_bias_cnt < S21_BIAS_CONVERGE_FRAMES) {
                    /* 收敛期：较快的 EMA 逼近真实偏置 */
                    s21_bias = 0.01f * (bm_ekf_pa - ms_ekf_pa)
                             + 0.99f * s21_bias;
                    s21_bias_cnt++;
                }
                /* else: 收敛完成 → 冻结偏置 */
                float p_bm_aligned = bm_ekf_pa - s21_bias;   /* 对齐到 MS5611 基准 */

                /* (2) 慢速 EMA 跟踪各路残差方差（衡量实际噪声/异常水平） */
                float res_ms = ms5611_data.pressure_pa - ms_ekf_pa;
                float res_bm = bmp280_data.pressure_pa - bm_ekf_pa;
                if (!s21_init) {
                    s21_var_ms = res_ms * res_ms;
                    s21_var_bm = res_bm * res_bm;
                    /* NOT setting s21_init here — 等输出低通初始化后再设置 */
                } else {
                    s21_var_ms = S21_VAR_EMA * (res_ms * res_ms) + (1.0f - S21_VAR_EMA) * s21_var_ms;
                    s21_var_bm = S21_VAR_EMA * (res_bm * res_bm) + (1.0f - S21_VAR_EMA) * s21_var_bm;
                }

                float p_fus, t_fus;
#if S21_FUSE_METHOD == 0
                float wm_raw = 1.0f / (s21_var_ms + S21_VAR_EPS);
                float wb_raw = 1.0f / (s21_var_bm + S21_VAR_EPS);
                float w_ms_inst = wm_raw / (wm_raw + wb_raw);
                /* 长 EMA 平滑权重 + 限幅，避免输出帧间振荡 */
                s21_w_ms = S21_W_EMA * w_ms_inst + (1.0f - S21_W_EMA) * s21_w_ms;
                float w_ms = (s21_w_ms < S21_W_MIN) ? S21_W_MIN
                           : ((s21_w_ms > S21_W_MAX) ? S21_W_MAX : s21_w_ms);
                p_fus = w_ms * ms_ekf_pa + (1.0f - w_ms) * p_bm_aligned;
#else
                p_fus = S21_FIXED_W_MS * ms_ekf_pa + (1.0f - S21_FIXED_W_MS) * p_bm_aligned;
#endif
#if S21_FUSE_TEMP_SRC == 0
                t_fus = bmp280_data.temperature_c;
#else
                t_fus = 0.5f * (ms5611_data.temperature_c + bmp280_data.temperature_c);
#endif
                /* 融合气压已对齐到 MS5611 基准，用 ms_p0 + 温度补偿换算高度 */
#if FUSION_SCHEME == 22
                /* 方案22 自适应输出低通（快慢双低通偏差法）：
                 *   p_fast(快,τ≈1s) 与 p_slow(慢,τ≈20s) 同时跟踪融合气压；
                 *   运动指标 m = |p_fast - p_slow|（两者均已平滑，不含高频噪声）；
                 *   norm = sat(m/S22_THR)^2，输出 = norm·p_fast + (1-norm)·p_slow。
                 *   静止 → norm≈0 → 用最平滑 p_slow（精度与方案21一致）；
                 *   运动 → norm≈1 → 切快低通，相对高度变化后几秒到位。 */
                if (!s21_init) {
                    s22_p_fast = p_fus;
                    s22_p_slow = p_fus;
                    s21_p_lp   = p_fus;
                    s21_init   = 1;      /* 低通初始完成后才标记初始化完成 */
                } else {
                    s22_p_fast = S22_A_FAST * p_fus + (1.0f - S22_A_FAST) * s22_p_fast;
                    s22_p_slow = S22_A_SLOW * p_fus + (1.0f - S22_A_SLOW) * s22_p_slow;
                    float s22_m = fabsf(s22_p_fast - s22_p_slow);
                    float s22_n = s22_m / S22_THR;
                    s22_n = (s22_n > 1.0f) ? 1.0f : ((s22_n < 0.0f) ? 0.0f : s22_n);
                    s22_n = s22_n * s22_n;
                    s21_p_lp = s22_n * s22_p_fast + (1.0f - s22_n) * s22_p_slow;
                }
#else
                /* 方案21 固定输出低通（S21_OUTPUT_EMA 级联在权重融合之后） */
                if (!s21_init) {
                    s21_p_lp = p_fus;
                    s21_init = 1;      /* 低通初始完成后才标记初始化完成 */
                } else {
                    s21_p_lp = S21_OUTPUT_EMA * p_fus + (1.0f - S21_OUTPUT_EMA) * s21_p_lp;
                }
#endif
                p_fus = s21_p_lp;
                s20_out.fused_pa   = p_fus;
                s20_out.fused_temp = t_fus;
                s20_out.fused_alt  = PressureToAltitudeWithTemp(p_fus, ms_p0, t_fus);
            }
            fusion_data.pressure_fused_pa = s20_out.fused_pa;   /* 方案21：融合气压为最终输出 */
#else
            fusion_data.pressure_fused_pa = bm_ekf_pa;          /* 方案20：BMP280-EKF 代表值 */
#endif
            fusion_data.temperature_c = bmp280_data.temperature_c;
        }
#else
        /* 方案1/3：气压域双传感器加权融合（MS5611 只用 KF，BMP280 用 NN） */
        fusion_data.pressure_fused_pa = ms5611_data.pressure_filtered_pa * runtime_fusion_weight_ms5611
                                      + bmp280_data.pressure_filtered_nn * runtime_fusion_weight_bmp280;
        fusion_data.temperature_c = bmp280_data.temperature_c;  /* BMP280 温度更稳定 */
#endif
#if FUSION_SCHEME == 7
        /* 方案7 已在上面高度域直接计算，跳过此步 */
#elif FUSION_SCHEME == 15
        fusion_data.height_fused_m = h15_lock;   /* 方案15：直接积分高度（门控锁定） */
#elif FUSION_SCHEME == 16
        fusion_data.height_fused_m = h16_lock;   /* 方案16：直接积分高度（门控锁定） */
#elif FUSION_SCHEME == 17
        fusion_data.height_fused_m = bmp280_data.height_filtered_m;  /* 方案17：KF 高度直通 */
#elif FUSION_SCHEME == 18
        /* 方案18：用预设 P0 + 带温度补偿的 ISA 标准大气公式算高度 */
        fusion_data.height_fused_m = PressureToAltitudeWithTemp(
            fusion_data.pressure_fused_pa,
            reference_pressure_pa,
            fusion_data.temperature_c);
#elif FUSION_SCHEME == 21 || FUSION_SCHEME == 22
        fusion_data.height_fused_m = s20_out.fused_alt;    /* 方案21/22：融合高度供相对高度基准 */
#elif FUSION_SCHEME == 20
        fusion_data.height_fused_m = s20_out.bm_ekf_alt;   /* 方案20：代表值供相对高度基准 */
#else
        /* ★ 方案1-6/8/9/10/11/12/13/14：用融合气压 + 参考气压 + BMP280 温度算高度 ★ */
        fusion_data.height_fused_m = PressureToAltitudeWithTemp(
            fusion_data.pressure_fused_pa,
            reference_pressure_pa,
            bmp280_data.temperature_c);
#endif
}

void StartBMP280Task(void *argument)
{
  osDelay(60);  /* 等待调度器稳定，与 MS5611 错开 */
  uint32_t prev_tick = osKernelGetTickCount();
  
  for(;;)
  {
    if(bmp280_ready)
    {
      osMutexAcquire(SensorDataMutexHandle, osWaitForever);
      BMP280_Read_Data(&bmp280_data.pressure_pa, &bmp280_data.temperature_c);
      /* 原始高度：统一用 MS5611 参考气压（所有高度统一基准） */
      bmp280_data.height_m = PressureToAltitudeWithTemp(bmp280_data.pressure_pa, reference_pressure_pa, bmp280_data.temperature_c);

      /* ---- 场景识别已合并进联合模型：BMP280 任务一次推理同时产出滤波与场景概率 ----



      /* KF 滤波气压 → 公式算高度 */
      bmp280_data.pressure_filtered_pa = KalmanFilter_Update_Adaptive_BMP280(&bmp280_kf_pressure, bmp280_data.pressure_pa);
      bmp280_data.height_filtered_m = PressureToAltitudeWithTemp(bmp280_data.pressure_filtered_pa, reference_pressure_pa, bmp280_data.temperature_c);
#if WORK_MODE == 0
      /* ---- 新架构：NN 只做场景判断，融合数值全部由 KF 计算 ---- */
      /* 运行 NN 推理获取 NN vs KF 散度（用于场景门控），
       * 但数值输出始终回退为 KF（pressure_filtered_nn = pressure_filtered_pa）。 */
      Multitask_PushBMP(bmp280_data.pressure_pa);
      if(ms5611_ready) { Multitask_Run(); }
      /* 所有数值通道统一使用 KF */
      ms5611_data.pressure_filtered_nn = ms5611_data.pressure_filtered_pa;
      bmp280_data.pressure_filtered_nn = bmp280_data.pressure_filtered_pa;
      bmp280_data.height_filtered_nn = bmp280_data.height_filtered_m;
#else
      /* NN 滤波暂时使用 KF 数据替代 */
      bmp280_data.pressure_filtered_nn = bmp280_data.pressure_filtered_pa;
      bmp280_data.height_filtered_nn = bmp280_data.height_filtered_m;
#endif

      /* 用 BMP280 温度重算 MS5611 的高度（统一参考气压） */
      if(ms5611_ready)
      {
        ms5611_data.height_m = PressureToAltitudeWithTemp(ms5611_data.pressure_pa, reference_pressure_pa, bmp280_data.temperature_c);
        ms5611_data.height_filtered_m = PressureToAltitudeWithTemp(ms5611_data.pressure_filtered_pa, reference_pressure_pa, bmp280_data.temperature_c);
        ms5611_data.height_filtered_nn = PressureToAltitudeWithTemp(ms5611_data.pressure_filtered_nn, reference_pressure_pa, bmp280_data.temperature_c);
      }

      /* ---- 全局 BMP280 偏置补偿（校准后生效，所有融合方案共用） ---- */
      /* 用 diff_mean 修正 BMP280 的【原始/KF】绝对气压，使其与 MS5611 基准对齐。
       * 注意：NN 联合模型的两个回归头输出的是【自包含的绝对气压估计】，
       * 其 BMP280 头已内建去偏（输出≈真值），不应再次叠加 bmp_bias，
       * 否则会与原始补偿重复，导致 BMP280 NN / FUSION 偏低约 40 Pa。 */
      bmp280_data.pressure_pa += bmp_bias_pa;
      bmp280_data.height_m = PressureToAltitudeWithTemp(bmp280_data.pressure_pa, reference_pressure_pa, bmp280_data.temperature_c);
      bmp280_data.pressure_filtered_pa += bmp_bias_pa;
      bmp280_data.height_filtered_nn = PressureToAltitudeWithTemp(bmp280_data.pressure_filtered_nn, reference_pressure_pa, bmp280_data.temperature_c);
      bmp280_data.height_filtered_m = PressureToAltitudeWithTemp(bmp280_data.pressure_filtered_pa, reference_pressure_pa, bmp280_data.temperature_c);

#if FUSION_SCHEME == 14
      /* ---- 方案14：BMP280 温漂补偿（实时温度漂移线性校正） ---- */
      /* 校准完成后第一帧记录参考温度，之后每帧根据温度偏差修正 BMP280 气压 */
      {
          if(!tc14_ref_initialized) {
              tc14_ref_temperature = bmp280_data.temperature_c;  /* 记录校准后稳定温度 */
              tc14_ref_initialized = true;
          }
          float temp_drift = bmp280_data.temperature_c - tc14_ref_temperature;
          float tc_correction = TC_COEFF * temp_drift;

          /* 对 BMP280 气压施加温漂补偿（气压偏高→向下修正） */
          bmp280_data.pressure_filtered_nn -= tc_correction;
          bmp280_data.pressure_filtered_pa -= tc_correction;

          /* 同步重算修正后的高度 */
          bmp280_data.height_filtered_nn = PressureToAltitudeWithTemp(
              bmp280_data.pressure_filtered_nn,
              reference_pressure_pa,
              bmp280_data.temperature_c);
          bmp280_data.height_filtered_m = PressureToAltitudeWithTemp(
              bmp280_data.pressure_filtered_pa,
              reference_pressure_pa,
              bmp280_data.temperature_c);
      }
#endif

      /* ---- 融合权重 ---- */
      runtime_fusion_weight_ms5611 = FUSION_WEIGHT_MS5611;
      runtime_fusion_weight_bmp280 = FUSION_WEIGHT_BMP280;

      /* 双传感器融合 — 统一在气压域融合，高度统一用同一参考气压计算 */
      /* MS5611 只使用 KF 数据（pressure_filtered_pa），BMP280 使用 NN 数据（pressure_filtered_nn） */
      if(ms5611_ready)
      {
        Fusion_Compute();
        /* ★ EMA 低通滤波：对最终融合气压做显示平滑，降低串口/OLED 的 Pa 抖动 ★ */
        /*    α=PRESSURE_EMA_ALPHA：方案15/16 用自动调参值(0.25)，其余默认 0.4 */
        {
            static float pressure_smoothed = 0.0f;
            static bool pressure_smoothed_init = false;
            if(!pressure_smoothed_init) {
                pressure_smoothed = fusion_data.pressure_fused_pa;
                pressure_smoothed_init = true;
            } else {
                pressure_smoothed += PRESSURE_EMA_ALPHA * (fusion_data.pressure_fused_pa - pressure_smoothed);
            }
            fusion_data.pressure_fused_pa = pressure_smoothed;
        }
        /* ★ EMA 低通滤波：对最终融合高度做平滑，大幅降低 OLED 显示波动 ★ */
        /*    α=HEIGHT_EMA_ALPHA：方案15/16 用自动调参值(0.71)，其余默认 0.5 */
        /*    方案15/16 静止时冻结 EMA 输出，避免传感器噪声引入缓慢漂移 */
        {
            static float height_smoothed = 0.0f;
            static bool height_smoothed_init = false;
            if(g_height_snap_req) {
                /* 重锚瞬间：直接把平滑值钉到目标海拔，避免 183->168 的滑过过程 */
                height_smoothed = g_fusion_reanchor_alt;
                height_smoothed_init = true;
                g_height_snap_req = false;
            } else if(!height_smoothed_init) {
                height_smoothed = fusion_data.height_fused_m;
                height_smoothed_init = true;
            } else {
#if FUSION_SCHEME == 15 || FUSION_SCHEME == 16
                /* 门控方案：运动时正常更新 EMA，静止时冻结 */
                if(g_mt_scene_pred == 1) {
                    height_smoothed += HEIGHT_EMA_ALPHA * (fusion_data.height_fused_m - height_smoothed);
                }
#else
                /* 方案17(无场景)及其他方案：始终更新 EMA */
                height_smoothed += HEIGHT_EMA_ALPHA * (fusion_data.height_fused_m - height_smoothed);
#endif
            }
            fusion_data.height_fused_m = height_smoothed;
        }
      }
      osMutexRelease(SensorDataMutexHandle);
    }
    
    vTaskDelayUntil(&prev_tick, SAMPLE_PERIOD_MS);
  }
}

void StartUARTOutputTask(void *argument)
{
  osDelay(70);
  uint32_t prev_tick = osKernelGetTickCount();
  char tx_buffer[512];
  uint16_t tx_len;
  
  for(;;)
  {
    SensorData_t ms5611_copy, bmp280_copy;
    bool ms5611_valid, bmp280_valid;
    
    osMutexAcquire(SensorDataMutexHandle, osWaitForever);
    ms5611_copy = ms5611_data;
    bmp280_copy = bmp280_data;
    ms5611_valid = ms5611_ready;
    bmp280_valid = bmp280_ready;
    osMutexRelease(SensorDataMutexHandle);
    
#if UART_ENABLE
    osMutexAcquire(UARTMutexHandle, osWaitForever);
    
    tx_len = 0;
    
#if WORK_MODE == 0
    /* 正常模式：完整输出（原始 + KF + NN + EKF） */
    if(ms5611_valid)
    {
      tx_len += sprintf(tx_buffer + tx_len, "MS5611,%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\r\n", 
        sample_id, 
        ms5611_copy.pressure_pa, ms5611_copy.temperature_c, ms5611_copy.height_m,
        ms5611_copy.pressure_filtered_pa, ms5611_copy.height_filtered_m,
        ms5611_copy.pressure_filtered_nn, ms5611_copy.height_filtered_nn);
    }
    
    if(bmp280_valid)
    {
      tx_len += sprintf(tx_buffer + tx_len, "BMP280,%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\r\n", 
        sample_id, 
        bmp280_copy.pressure_pa, bmp280_copy.temperature_c, bmp280_copy.height_m,
        bmp280_copy.pressure_filtered_pa, bmp280_copy.height_filtered_m,
        bmp280_copy.pressure_filtered_nn, bmp280_copy.height_filtered_nn);
    }

    if(ms5611_valid && bmp280_valid)
    {
      tx_len += sprintf(tx_buffer + tx_len, "FUSION,%lu,%.2f,%.4f,%.2f\r\n",
        sample_id,
        fusion_data.pressure_fused_pa, fusion_data.height_fused_m,
        fusion_data.temperature_c);
    }

    /* 场景识别结果（来自联合模型 OUT_3，与滤波同一次推理） */
    if(ms5611_valid && bmp280_valid)
    {
#if FUSION_SCHEME == 16
      /* 方案16：场景由 KF 衍生判定（gate16_state），不依赖 NN 场景输出 */
      tx_len += sprintf(tx_buffer + tx_len, "SCENE,%lu,%s,%.3f,%.3f\r\n",
        sample_id,
        gate16_state ? "elevation" : "static",
        gate16_state ? 0.0f : 1.0f, gate16_state ? 1.0f : 0.0f);
#elif FUSION_SCHEME == 17
      /* 方案17：无场景模式，一直输出 static */
      tx_len += sprintf(tx_buffer + tx_len, "SCENE,%lu,%s,%.3f,%.3f\r\n",
        sample_id, "static", 1.0f, 0.0f);
#else
      tx_len += sprintf(tx_buffer + tx_len, "SCENE,%lu,%s,%.3f,%.3f\r\n",
        sample_id,
        Multitask_SceneName(g_mt_scene_pred),
        g_mt_scene[0], g_mt_scene[1]);
#endif
    }
#else
    /* 采集模式：简化输出（原始 + KF，不含 EKF/NN），适配 PC 端 24h 采集脚本 */
    if(ms5611_valid)
    {
      tx_len += sprintf(tx_buffer + tx_len, "MS5611,%lu,%.2f,%.2f,%.2f,%.2f,%.2f\r\n", 
        sample_id, 
        ms5611_copy.pressure_pa, ms5611_copy.temperature_c, ms5611_copy.height_m,
        ms5611_copy.pressure_filtered_pa, ms5611_copy.height_filtered_m);
    }
    
    if(bmp280_valid)
    {
      tx_len += sprintf(tx_buffer + tx_len, "BMP280,%lu,%.2f,%.2f,%.2f,%.2f,%.2f\r\n", 
        sample_id, 
        bmp280_copy.pressure_pa, bmp280_copy.temperature_c, bmp280_copy.height_m,
        bmp280_copy.pressure_filtered_pa, bmp280_copy.height_filtered_m);
    }

    /* 采集模式：融合输出简化（不含 EKF） */
    if(ms5611_valid && bmp280_valid)
    {
      tx_len += sprintf(tx_buffer + tx_len, "FUSION,%lu,%.2f,%.4f,%.2f\r\n",
        sample_id,
        fusion_data.pressure_fused_pa, fusion_data.height_fused_m,
        fusion_data.temperature_c);
    }

    /* 采集模式不运行 NN 模型，场景识别结果省略 */
#endif

    if(tx_len > 0)
    {
      /* DMA 方式发送：CPU 不阻塞，发送期间可运行其他任务 */
      while (uart_tx_busy) osDelay(1);  /* 等待上一轮 DMA 完成 */
      uart_tx_busy = 1;
      HAL_UART_Transmit_DMA(&huart2, (uint8_t*)tx_buffer, tx_len);
    }
    
    osMutexRelease(UARTMutexHandle);
#endif /* UART_ENABLE */
    
    sample_id++;
    vTaskDelayUntil(&prev_tick, SAMPLE_PERIOD_MS);
  }
}

void StartUARTCommandTask(void *argument)
{
  uint8_t rx_char;
  
  for(;;)
  {
    if(HAL_UART_Receive(&huart2, &rx_char, 1, 100) == HAL_OK)
    {
      if(rx_char == '\r' || rx_char == '\n')
      {
        if(uart_rx_len > 0)
        {
          uart_rx_buffer[uart_rx_len] = '\0';
          ProcessCommand(uart_rx_buffer);
          uart_rx_len = 0;
        }
      }
      else if(uart_rx_len < 63)
      {
        uart_rx_buffer[uart_rx_len++] = rx_char;
      }
    }
    osDelay(10);
  }
}

void ProcessCommand(char *cmd)
{
  if(strcmp(cmd, "STATUS") == 0)
  {
    printf("INFO,MS5611 Ready: %s\r\n", ms5611_ready ? "YES" : "NO");
    printf("INFO,BMP280 Ready: %s\r\n", bmp280_ready ? "YES" : "NO");
    printf("INFO,Reference Pressure: %.2f Pa (sea level)\r\n", reference_pressure_pa);
#if FUSION_SCHEME == 16
    printf("INFO,Fusion Scheme: 16 (KF-dominant, no NN; scene from KF Δh)\r\n");
#elif FUSION_SCHEME == 15
    printf("INFO,Fusion Scheme: 15 (NN-dominant; scene from joint model)\r\n");
    printf("INFO,NN Model: Multitask baseline_eqw (joint MS5611+BMP280, 3 outputs: filt_ms/filt_bmp/scene)\r\n");
#elif FUSION_SCHEME == 17
    printf("INFO,Fusion Scheme: 17 (no-scene, BMP280 KF direct)\r\n");
    printf("INFO,Altitude fixed at 156.927 m, preset ID cycles on button press\r\n");
#else
    printf("INFO,NN Model: Multitask baseline_eqw (joint MS5611+BMP280, 3 outputs: filt_ms/filt_bmp/scene)\r\n");
#endif
  }
  else if(strncmp(cmd, "SET_ALT", 7) == 0)
  {
    float new_altitude;
    if(sscanf(cmd, "SET_ALT %f", &new_altitude) == 1 && new_altitude >= -500.0f && new_altitude <= 10000.0f)
    {
      /* 用新海拔重新计算统一的参考气压（基于 MS5611 KF 滤波气压，比原始值更稳定） */
      float ms_measured = ms5611_data.pressure_filtered_pa > 0 ? ms5611_data.pressure_filtered_pa : reference_pressure_pa;
#if ALTITUDE_FORMULA_ISA
      /* ISA 公式反算海平面气压 */
      float isa_factor = 1.0f - ISA_L * new_altitude / ISA_T0;
      if(isa_factor > 0.01f) {
        float exponent = ISA_G / (ISA_L * ISA_R);
        reference_pressure_pa = ms_measured / powf(isa_factor, exponent);
      } else {
        reference_pressure_pa = ms_measured + new_altitude * 11.3f;
      }
#else
      /* 等温模型反解（与前向公式严格互逆）：P0 = P * exp(g*h/(R*T)) */
      float temp_k = (bmp280_data.temperature_c > -100.0f && bmp280_data.temperature_c < 100.0f)
                     ? (bmp280_data.temperature_c + 273.15f) : ISA_T0;
      reference_pressure_pa = ms_measured * expf(ISA_G * new_altitude / (ISA_R * temp_k));
#endif
      /* 同步重锚融合积分锁（OLED 立即显示设定值） */
      g_fusion_reanchor_alt = new_altitude;
      g_fusion_reanchor_req = true;
      g_height_snap_req = true;   /* 同时请求 OLED 高度 EMA 立即对齐到目标海拔 */
      printf("INFO,Altitude set to %.1f m, ref_pressure=%.2f Pa\r\n",
             new_altitude, reference_pressure_pa);
    }
    else
    {
      printf("INFO,Usage: SET_ALT <height_m>\r\n");
    }
  }
  else if(strncmp(cmd, "SET_P0", 6) == 0)
  {
    float new_p0;
    if(sscanf(cmd, "SET_P0 %f", &new_p0) == 1 && new_p0 > 50000.0f && new_p0 < 120000.0f)
    {
      /* 方案18：直接预设参考海平面气压 P0，不触发海拔重锚/锁定 */
      reference_pressure_pa = new_p0;
      calib_result.reference_pressure_pa = new_p0;
      printf("INFO,Reference pressure P0 set to %.2f Pa (%.2f hPa)\r\n", new_p0, new_p0/100.0f);
    }
    else
    {
      printf("INFO,Usage: SET_P0 <pressure_pa>  (range 50000~120000 Pa)\r\n");
    }
  }
  else if(strcmp(cmd, "HELP") == 0)
  {
    printf("INFO,Available commands:\r\n");
    printf("INFO,  HELP - Show this help\r\n");
    printf("INFO,  STATUS - Show system status\r\n");
    printf("INFO,  SET_ALT <height_m> - Set current altitude (recalc reference pressure)\r\n");
    printf("INFO,  SET_P0 <pressure_pa> - Set reference sea-level pressure P0 (scheme 18)\r\n");
    printf("INFO,  SET_CALIB - Show detailed calibration report\r\n");
  }
  else if(strcmp(cmd, "SET_CALIB") == 0)
  {
    printf("INFO,=== Calibration Report ===\r\n");
    printf("INFO,State: %d (0=idle 1=phase1 2=phase2 3=analysis 4=complete 5=failed)\r\n", calib_result.state);
    printf("INFO,Elapsed: %lu ms\r\n", calib_result.elapsed_ms);
    printf("INFO,--- Sensor Statistics ---\r\n");
    printf("INFO,MS5611: mean=%.2f std=%.2f min=%.2f max=%.2f range=%.2f outliers=%d(%.1f%%) health=%d\r\n",
           calib_result.ms5611_stats.mean, calib_result.ms5611_stats.std,
           calib_result.ms5611_stats.min, calib_result.ms5611_stats.max,
           calib_result.ms5611_stats.range,
           calib_result.ms5611_stats.outlier_count, calib_result.ms5611_stats.outlier_ratio * 100.0f,
           calib_result.ms5611_stats.health);
    printf("INFO,BMP280: mean=%.2f std=%.2f min=%.2f max=%.2f range=%.2f outliers=%d(%.1f%%) health=%d\r\n",
           calib_result.bmp280_stats.mean, calib_result.bmp280_stats.std,
           calib_result.bmp280_stats.min, calib_result.bmp280_stats.max,
           calib_result.bmp280_stats.range,
           calib_result.bmp280_stats.outlier_count, calib_result.bmp280_stats.outlier_ratio * 100.0f,
           calib_result.bmp280_stats.health);
    printf("INFO,Dual diff: mean=%.2f std=%.2f consistency=%.2f\r\n",
           calib_result.diff_mean, calib_result.diff_std, calib_result.consistency);
    printf("INFO,--- Auto-tuned Parameters ---\r\n");
    printf("INFO,KF_MS5611: Q=%.4f R=%.2f\r\n", calib_result.kf_q_ms5611, calib_result.kf_r_ms5611);
    printf("INFO,KF_BMP280: Q=%.4f R=%.2f\r\n", calib_result.kf_q_bmp280, calib_result.kf_r_bmp280);
    printf("INFO,Fusion weights: MS5611=%.2f BMP280=%.2f\r\n",
           calib_result.fusion_weight_ms5611, calib_result.fusion_weight_bmp280);
    printf("INFO,Reference pressure: %.2f Pa\r\n", calib_result.reference_pressure_pa);
  }
  else
  {
    printf("INFO,Unknown command: %s\r\n", cmd);
    printf("INFO,Type HELP for available commands\r\n");
  }
}
#if WORK_MODE == 0
/* ========== OLED 显示任务 (I2C) ========== */

/* =====================================================================
 *                ★ 屏幕显示怎么改？看这里就够了 ★
 * ---------------------------------------------------------------------
 *  ① 想【换字体大小 / 行间距 / 颜色】：只改下面 4 个 OLEDUI_ 数字。
 *  ② 想【增加一行】：复制下面任意一行 UI_ROW(...)，改引号里的文字和
 *     后面的数值即可（详见下方示例）。
 *  ③ 想【删除一行】：把对应的 UI_ROW(...) 那一行整行删掉。
 *  ④ 想【调整上下顺序】：用鼠标把那一行拖到想要的位置。
 *  每一行会自动排在上一行下面，不用管坐标。
 * ===================================================================== */
#define OLEDUI_Y0    0      /* 第 1 行离屏幕顶部的像素 */
#define OLEDUI_ROW_H 12     /* 每行间隔像素；8X16 字体用 12~16，6X8 用 8 */
#define OLEDUI_FONT  SSD1306_FONT_8X16   /* 字体大小 */
#define OLEDUI_COLOR SSD1306_COLOR_WHITE /* 文字颜色 */

/* ---- 以下为内部辅助，一般不用改 ---- */
static uint8_t oledui_row = 0;
static char    oledui_buf[32];

static void OLEDUI_Begin(void)
{
    SSD1306_Fill(SSD1306_COLOR_BLACK);
    oledui_row = 0;
}
static void OLEDUI_Put(const char *s)
{
    SSD1306_DrawString(0, OLEDUI_Y0 + oledui_row * OLEDUI_ROW_H, s, OLEDUI_FONT, OLEDUI_COLOR);
    oledui_row++;
}
static void OLEDUI_End(void)
{
    SSD1306_UpdateScreen();
}

/* 单行宏：
 *   UI_ROW(条件成立才显示? 条件, "文字格式", 数值...)
 * 例：
 *   UI_ROW(1,            "H:%.4f m", fusion_data.height_fused_m);  // 总是显示高度
 *   UI_ROW(传感器有效,   "P:%.1f Pa", fusion_data.pressure_fused_pa); // 有效才显示气压
 * 引号里 %.4f 表示显示小数(4位)，%d 表示整数，文字可随意改。
 */
#define UI_ROW(cond, fmt, ...)                                       \
    do { if (cond) {                                                 \
            snprintf(oledui_buf, sizeof(oledui_buf), fmt, ##__VA_ARGS__); \
            OLEDUI_Put(oledui_buf);                                  \
         } } while (0)

void StartOLEDTask(void *argument)
{
    osDelay(200);  /* 等待传感器初始化完成 */
    printf("DBG,OLED,Task started\r\n");

    SSD1306_Fill(SSD1306_COLOR_BLACK);
    SSD1306_DrawString(0, 0, "OLED Task OK!", SSD1306_FONT_8X16, SSD1306_COLOR_WHITE);
    SSD1306_DrawString(0, 16, "Sensor wait...", SSD1306_FONT_8X16, SSD1306_COLOR_WHITE);
    SSD1306_UpdateScreen();
    printf("DBG,OLED,Task initial display done\r\n");

    /* 相对高度基准 rel_height_ref / rel_ref_set 已提升为全局变量：
     * 上电自动重锚与手动切换预设时由 ApplyAltPreset 钉到目标海拔；
     * 以下“首次有效读数”仅作为未触发重锚时的兜底锁定。 */

    uint32_t prev_tick = osKernelGetTickCount();
    int display_phase = 0;
    uint32_t loop_count = 0;

    for(;;)
    {
        loop_count++;
        // printf("DBG,OLED,Task loop %lu starting (phase=%d)\r\n", loop_count, display_phase);

        SensorData_t ms5611_copy, bmp280_copy;
        bool ms5611_valid, bmp280_valid;
#if FUSION_SCHEME == 20 || FUSION_SCHEME == 21 || FUSION_SCHEME == 22
        S20_Out_t s20_copy;
#endif

        osMutexAcquire(SensorDataMutexHandle, osWaitForever);
        ms5611_copy = ms5611_data;
        bmp280_copy = bmp280_data;
        ms5611_valid = ms5611_ready;
        bmp280_valid = bmp280_ready;
#if FUSION_SCHEME == 20 || FUSION_SCHEME == 21 || FUSION_SCHEME == 22
        s20_copy = s20_out;
#endif
        osMutexRelease(SensorDataMutexHandle);

        /* 首次有效读数时锁定相对高度基准
         * 注意：刚上电时融合输出不稳定（偏置收敛 + 输出低通需 ~10s 到位）。
         * 延迟 5s（~25 个 OLED 循环 × 200ms）才锁参考，避免启动漂移被固化。 */
        {
            static uint8_t slow_ref_count = 0;
            if(ms5611_valid && bmp280_valid && !rel_ref_set)
            {
                if(++slow_ref_count >= 25)
                {
                    rel_height_ref = fusion_data.height_fused_m;
                    rel_ref_set = true;
                }
            }
        }

        /* PC13 按键检测：短按=循环切页/切 ID；长按 3s=相对高度校准清零 */
        {
            static uint8_t  key_prev = 1;          /* 上拉，默认高电平 */
            static uint32_t key_press_tick = 0;    /* 本次按下起始时刻 */
            static uint8_t  long_press_done = 0;   /* 本次按压是否已触发长按 */
            uint8_t  key_now  = (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_RESET) ? 0 : 1;
            uint32_t now_tick = osKernelGetTickCount();

            if(key_prev == 1 && key_now == 0)
            {
                /* 下降沿=按下：记录起始时刻并清长按标志 */
                key_press_tick  = now_tick;
                long_press_done = 0;
            }
            else if(key_prev == 0 && key_now == 0)
            {
                /* 持续按住达 3s：触发一次相对高度校准清零 */
                if(!long_press_done && (now_tick - key_press_tick) >= 3000u)
                {
                    long_press_done = 1;
#if FUSION_SCHEME == 20 || FUSION_SCHEME == 21 || FUSION_SCHEME == 22
                    /* 相对高度校准：当前融合高度设为新基准 → 相对高度归零 */
                    rel_height_ref = fusion_data.height_fused_m;
                    rel_ref_set    = true;
                    printf("INFO,Relative altitude calibrated to 0 (B1 long press 3s)\r\n");
#endif
                }
            }
            else if(key_prev == 0 && key_now == 1)
            {
                /* 上升沿=松开：仅当未触发长按时，按短按处理 */
                if(!long_press_done)
                {
#if FUSION_SCHEME == 18
                    /* 方案18：只循环 ID 显示，不改变高度/参考气压
                     * （方案18 用预设 P0，不在此处反算海拔） */
                    g_alt_preset_id = g_alt_presets[(++g_alt_preset_idx) % ALT_PRESET_COUNT].id;
                    printf("INFO,Preset ID cycled to %d (scheme 18: P0 unchanged)\r\n", g_alt_preset_id);
#elif FUSION_SCHEME == 20 || FUSION_SCHEME == 21 || FUSION_SCHEME == 22
                    /* 方案20：B1 短按循环切换 EKF/BP 对照页 (0..3)
                     * 方案21：在方案20 四页基础上增加第 5 页"FUSED"融合结果 (0..4) */
#if FUSION_SCHEME == 21 || FUSION_SCHEME == 22
                    g_s20_page = (g_s20_page + 1) % 5;
#else
                    g_s20_page = (g_s20_page + 1) % 4;
#endif
                    printf("INFO,S20 page cycled to %d\r\n", g_s20_page);
#else
                    /* 方案17：上电已用默认预设(Idx=8→ID1122,160.730m)反算 P0 把高度锚定好；
                     * 点击按钮只循环切换屏幕 ID 与参考高度显示，
                     * 不改变实际输出高度 H / 参考气压 P0（高度保持上电预设值）。 */
                    g_alt_preset_id = g_alt_presets[(++g_alt_preset_idx) % ALT_PRESET_COUNT].id;
                    printf("INFO,Preset ID cycled to %d (scheme 17: P0/height unchanged)\r\n", g_alt_preset_id);
#endif
                }
                long_press_done = 0;
            }
            key_prev = key_now;
        }

        // printf("DBG,OLED,Data ms5611_valid=%d bmp280_valid=%d\r\n",
        //        (int)ms5611_valid, (int)bmp280_valid);

        /* ====== 屏幕显示内容（增删一行 = 增删一行 UI_ROW） ====== */
        OLEDUI_Begin();

        /* 开机前 4 次刷新显示校准结果 */
        if (display_phase < 4)
        {
            UI_ROW(1, "Cal:%lu sec",  calib_result.elapsed_ms / 1000);
            UI_ROW(1, "MS:%.1f/%d",   calib_result.ms5611_stats.std,
                                      calib_result.ms5611_stats.health);
            UI_ROW(1, "BM:%.1f/%d",   calib_result.bmp280_stats.std,
                                      calib_result.bmp280_stats.health);
            UI_ROW(1, "Coh:%.2f",     calib_result.consistency);
            display_phase++;
        }
        else
        {
            /* 先算好下面要显示的数值 */
            char  scene_ch = (g_mt_scene_pred == 1) ? 'D' : 'S';
#if FUSION_SCHEME == 17
            scene_ch = 'S';   /* 方案17 无场景，固定显示 S */
#endif
            float rel_h = (ms5611_valid && bmp280_valid && rel_ref_set)
                          ? (fusion_data.height_fused_m - rel_height_ref) : 0.0f;

#if FUSION_SCHEME == 20 || FUSION_SCHEME == 21 || FUSION_SCHEME == 22
            /* ===== 方案20/21：EKF vs BP 多页对照（B1 短按切页） =====
             * 方案20：0=MS5611 EKF  1=BMP280 EKF  2=MS5611 BP  3=BMP280 BP
             * 方案21：额外第 5 页 4=FUSED（双 EKF 方差倒数加权融合结果）
             * 每页：标题 + 温度 + 气压(hPa) + 高度(m)。温度随页刷新。 */
            {
#if FUSION_SCHEME == 21 || FUSION_SCHEME == 22
                static const char *s20_title[5] = {
                    "MS5611 EKF", "BMP280 EKF", "MS5611 BP", "BMP280 BP", "FUSED"};
#else
                static const char *s20_title[4] = {
                    "MS5611 EKF", "BMP280 EKF", "MS5611 BP", "BMP280 BP"};
#endif
                float p_pa = 0.0f, h_m = 0.0f, t_c = 0.0f;
                switch (g_s20_page) {
                    case 0: p_pa = s20_copy.ms_ekf_pa; h_m = s20_copy.ms_ekf_alt; t_c = s20_copy.ms_temp; break;
                    case 1: p_pa = s20_copy.bm_ekf_pa; h_m = s20_copy.bm_ekf_alt; t_c = s20_copy.bm_temp; break;
                    case 2: p_pa = s20_copy.ms_bp_pa;  h_m = s20_copy.ms_bp_alt;  t_c = s20_copy.ms_temp; break;
                    case 3: p_pa = s20_copy.bm_bp_pa;  h_m = s20_copy.bm_bp_alt;  t_c = s20_copy.bm_temp; break;
#if FUSION_SCHEME == 21 || FUSION_SCHEME == 22
                    default:p_pa = s20_copy.fused_pa;  h_m = s20_copy.fused_alt;  t_c = s20_copy.fused_temp; break;
#else
                    default:p_pa = s20_copy.bm_bp_pa;  h_m = s20_copy.bm_bp_alt;  t_c = s20_copy.bm_temp; break;
#endif
                }
                UI_ROW(1, "--- %s ---", s20_title[g_s20_page]);
                UI_ROW(1, "T:+%.2f C",    t_c);
                UI_ROW(1, "P:%.2f hPa",   p_pa / 100.0f);
                UI_ROW(1, "Alt:%.3f m",   h_m);
#if FUSION_SCHEME == 21 || FUSION_SCHEME == 22
                /* FUSED 页(第5页)额外显示相对高度；B1 长按 3s 清零 */
                UI_ROW(g_s20_page == 4, "Rel:%+.3f m", rel_h);
#endif
            }
#else
            /* ↓↓↓ 想加 / 删 / 调整显示行，就改下面这几行 ↓↓↓ */
            UI_ROW(ms5611_valid && bmp280_valid, "H:%.4f m",  fusion_data.height_fused_m);
            UI_ROW(ms5611_valid && bmp280_valid, "P:%.1f Pa", fusion_data.pressure_fused_pa);
            UI_ROW(ms5611_valid && bmp280_valid, "T:%.1f %c",
                   fusion_data.temperature_c, scene_ch);
            //UI_ROW(ms5611_valid && bmp280_valid && rel_ref_set, "U:%+.4f m", rel_h);
           // UI_ROW(1, "Id%d Ref:%.1fm",
                   //g_alt_preset_id, g_alt_presets[g_alt_preset_idx % ALT_PRESET_COUNT].altitude_m);
           // UI_ROW(1, "P0:%.1f Pa", reference_pressure_pa);
            /* ↑↑↑ 想加 / 删 / 调整显示行，就改上面这几行 ↑↑↑ */
#endif

            /* ===== 以下为【暂未启用】的调试显示行：BP滤波 + 各传感器KF原始数据 =====
             * 屏幕空间有限（最多约 5 行），默认用 // 注释关闭。
             * 需要看某项时，把对应行前面的 // 删掉即可显示；不要一次开太多行。 */
            // ---- BP 神经网络滤波结果 ----
             //UI_ROW(ms5611_valid, "MS_BP_P:%.1f Pa", ms5611_copy.pressure_filtered_nn);
             //UI_ROW(ms5611_valid, "MS_BP_H:%.3f m", ms5611_copy.height_filtered_nn);
            // UI_ROW(bmp280_valid, "BM_BP_P:%.1f Pa", bmp280_copy.pressure_filtered_nn);
             //UI_ROW(bmp280_valid, "BM_BP_H:%.3f m", bmp280_copy.height_filtered_nn);
            // ---- MS5611 卡尔曼滤波(KF)原始输出：气压 / 高度 / 温度 ----
            // UI_ROW(ms5611_valid, "MS_KF_P:%.1f Pa", ms5611_copy.pressure_filtered_pa);
            // UI_ROW(ms5611_valid, "MS_KF_H:%.3f m", ms5611_copy.height_filtered_m);
            // UI_ROW(ms5611_valid, "MS_KF_T:%.1f C",  ms5611_copy.temperature_c);
            // ---- BMP280 卡尔曼滤波(KF)原始输出：气压 / 高度 / 温度 ----
            // UI_ROW(bmp280_valid, "BM_KF_P:%.1f Pa", bmp280_copy.pressure_filtered_pa);
            // UI_ROW(bmp280_valid, "BM_KF_H:%.3f m", bmp280_copy.height_filtered_m);
            // UI_ROW(bmp280_valid, "BM_KF_T:%.1f C",  bmp280_copy.temperature_c);
        }

        OLEDUI_End();
        // printf("DBG,OLED,Refresh done (loop=%lu)\r\n", loop_count);

        vTaskDelayUntil(&prev_tick, 200);   /* 0.2s 刷新一次 */
    }
}
#endif
/* USER CODE END 4 */

/* USER CODE BEGIN Header_StartDefaultTask */
/**
  * @brief  Function implementing the defaultTask thread.
  * @param  argument: Not used
  * @retval None
  */
/* USER CODE END Header_StartDefaultTask */
void StartDefaultTask(void *argument)
{
  /* USER CODE BEGIN 5 */
  /* 方案17：上电用默认预设海拔（ID1122）反算 P0，高度锚定到该点；
   * 方案18：上电沿用校准块反算的本地 P0（不覆盖），运行时仍可用 SET_P0 微调。 */
#if WORK_MODE == 0
  for (int i = 0; i < 100 && !ms5611_ready; i++) osDelay(100);  /* 等 MS5611 就绪 */
  osDelay(3000);                          /* 等上电压力漂移稳定（热稳定等待只盯温度） */
  #if FUSION_SCHEME == 17
    /* 方案17：上电即用默认预设（g_alt_preset_idx=10 → ID1122, 160.730m）反算 P0，
     * 使高度锚定到该点海拔；之后按键切换预设会再次重算 P0。 */
    ApplyAltPreset(g_alt_preset_idx);
  #endif
#else
  (void)g_alt_preset_idx;
#endif

#if FUSION_SCHEME != 17
  /* 方案18/其他：沿用启动校准块反算得到的本地参考气压 P0，
   * 不再用硬编码 PRESET_P0_PA 覆盖，避免上电 ~3s 后 P0 被改写导致高度瞬跌。
   * 方案17 已由上方 ApplyAltPreset 设为默认预设对应的 P0，此处不再覆盖。 */
  reference_pressure_pa = calib_result.reference_pressure_pa;
#endif
  /* Infinite loop */
  for(;;)
  {
    osDelay(1);
  }
  /* USER CODE END 5 */
}

/**
  * @brief  Period elapsed callback in non blocking mode
  * @note   This function is called  when TIM10 interrupt took place, inside
  * HAL_TIM_IRQHandler(). It makes a direct call to HAL_IncTick() to increment
  * a global variable "uwTick" used as application time base.
  * @param  htim : TIM handle
  * @retval None
  */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  /* USER CODE BEGIN Callback 0 */

  /* USER CODE END Callback 0 */
  if (htim->Instance == TIM10)
  {
    HAL_IncTick();
  }
  /* USER CODE BEGIN Callback 1 */

  /* USER CODE END Callback 1 */
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
