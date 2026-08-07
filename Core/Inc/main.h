/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdbool.h>  // ????
#include "fusion_scheme_15_16_params.h"  // ???????????15/16???????????(?? compare_s16.py ???)
#include "fusion_scheme_20_params.h"  // 方案20 参数（sahixi 气压域EKF + BP 去噪）
#include "fusion_scheme_21_params.h"  // 方案21 参数（在方案20 基础上做双 EKF 方差倒数加权融合）
#include "fusion_scheme_22_params.h"  // 方案22 参数（在方案21 基础上把固定输出低通换成运动自适应低通）
/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/

/* USER CODE BEGIN Private defines */
#define UART_ENABLE  0  /* 串口输出控制：0=关闭（最快速度），1=开启（DMA方式，不阻塞CPU） */

/* ========== NN 散度场景检测参数（方案16 新架构） ==========
 * NN vs KF 散度 = |NN_filtered_pressure - KF_filtered_pressure|
 * 静止时偏置校正使 NN 趋近 KF → 散度小；运动时偏置冻结 → 散度增大。
 * 实测调参范围建议：GATE_OPEN=3~8, GATE_CLOSE=1~3 (Pa), ALPHA=0.1~0.25 */
#define NN_DIV_ALPHA     0.15f   /* 散度 EMA 系数（越小越平滑，响应越慢） */
#define S16_GATE_OPEN_NN  5.0f   /* NN 散度开门阈值 (Pa)：超过此值认为在运动 */
#define S16_GATE_CLOSE_NN 2.0f   /* NN 散度关门阈值 (Pa)：低于此值认为静止 */


/* 模型类型选择宏
 * 1 = 自监督模型 (self_supervised) — 用滑动平均构造伪标签，模型输出滤波后气压
 * 0 = 有标注模型 (labeled) — 用人工标注真实高度训练，模型直接输出高度
 */
#define USE_SELF_SUPERVISED_MODEL  1

/* 工作模式选择宏
 * 0 = 正常模式（EKF+KF+NN，OLED显示，完整输出）
 * 1 = 数据采集模式（简化输出，仅原始+KF，便于 PC 端采集）
 * 修改后重新编译即可切换模式
 */
#define WORK_MODE  0

/* ========== 融合方案选择（编译期，单固件含一个方案） ==========
 * 1 = 方案1：单传感器内 NN(50%)+KF(50%) 再双传感器加权融合
 * 2 = 方案2：四路直接加权融合（MS5611_NN/KF + BMP280_NN/KF）
 * 3 = 方案3：KF 对 NN 输出做二次滤波后再双传感器加权融合
 * 4 = 方案4：BMP280 主力 + MS5611 高频增强通道（HPF）
 * 5 = 方案5：自适应权重融合（静止 BMP280 主导，运动增大 MS5611 权重）
 * 6 = 方案6：MS5611 主导（不使用 NN）— MS5611 KF(85%)+BMP280 KF(15%)
 * 7 = 方案7：BMP280 定绝对高度 + 双传感器高度变化量加权累积
 * 8 = 方案8：只用 MS5611 KF（纯 MS5611）
 * 9 = 方案9：只用 BMP280 的 NN(70%)+KF(30%)
 * 10= 方案10：逆方差加权融合（实时方差决定权重）
 * 11= 方案11：Delta 置信度加权累积 + 泄漏锚
 * 12= 方案12：Hampel 脉冲抑制预处理 + 加权融合
 * 13= 方案13：二阶互补融合（气压 + 变化率）
 * 14= 方案14：方案4 + BMP280 温漂补偿（默认）
 * 15= 方案15：NN 主导场景门控增量锁定（见 fusion_scheme_15_16_params.h）
 * 16= 方案16：KF 主导 + NN 散度场景门控增量锁定（NN 仅场景，数值全 KF）
 *      （见 fusion_scheme_15_16_params.h）
 * 17= 方案17：无场景模式 — BMP280 KF 直接算高度，按键只循环 ID 不切换海拔
 * 18= 方案18：使用预设 P0 计算高度（手动 SET_P0 设参考海平面气压；
 *      气压域双传感器加权融合 + ISA 公式算绝对高度；上电不自动预设海拔，
 *      不涉及海拔锁定/重锚，按键仅循环 ID 显示，不改变 P0）
 * 19= （预留）
 * 20= 方案20：移植参考项目 sahixi 的"气压域 EKF + 单传感器 BP 去噪"双路线。
 *      对 MS5611/BMP280 各跑气压域 EKF（状态[P,dP/dt]，雅可比∂h/∂P）与
 *      5 点滑窗 BP 网络（5→8(tanh)→1，sahixi 预训练权重），统一用温度补偿
 *      ISA 公式换算高度；B1 短按循环四页对照（MS5611 EKF / BMP280 EKF /
 *      MS5611 BP / BMP280 BP）。详见 fusion_scheme_20_params.h 与 baro_ekf.c / bp_denoise.c。
 * 21= 方案21：在方案20 基础上，对 MS5611-EKF 与 BMP280-EKF 两条主通道做
 *      "方差倒数加权"融合（p_fus=(p_ms/P_ms+p_bm/P_bm)/(1/P_ms+1/P_bm)），
 *      输出比任一单传感器更稳的融合高度；某传感器失效时其 P 增大→权重自动趋零
 *      （天然冗余）。完整保留方案20 的四页对照，并新增第 5 页"FUSED"。
 * 22= 方案22：在方案21 融合链路上，把固定输出低通换成"运动自适应低通"
 *      （快慢双低通偏差法）。相对高度变化后几秒稳定（方案21 需约1分钟），
 *      静止噪声保持厘米级（精度不变）。详见 fusion_scheme_22_params.h。
 *      详见 fusion_scheme_21_params.h。
 *
 * 方案 1-14 的融合参数与共享 KF/EMA 已由 PC 端自动调参固化，
 * 见 fusion_scheme_tuned_params.h（altimeter_tuner/tune_all_params.py 生成）。
 * 下方各融合宏优先取 TUNED_*（当前方案有调参值时），否则回退历史默认值。 */
#define FUSION_SCHEME 21

/* ========== 方案18：代码预设参考海平面气压 P0 ==========
 * 单位 Pa。上电后 StartDefaultTask 会直接用此值计算高度，
 * 不通过海拔反算、也不依赖串口 SET_P0 命令。
 * 改为当地实际海平面气压即可（标准值 101325 = 1013.25 hPa）。
 * 调优（simulation/sim_scheme18.py，基于 serial_tool/data/raw 真实数据）：
 *   采集地实际海平面气压 ≈ 100457 Pa；用标准 101325 会引入约 72 m 系统偏移。
 *   部署到其他地点/时段的海平面气压会随天气漂移，应按实地值重新设定
 *   （或改用串口 SET_P0 命令在线设置）。 */
#define PRESET_P0_PA  100060.0f

/* 引入方案1-14 自动调参参数（必须在 FUSION_SCHEME 定义之后包含） */
#include "fusion_scheme_tuned_params.h"

/* ========== 增强校准参数 ========== */
/* 校准采样：单阶段采集，精细野值剔除交由 Phase3 的 RobustClean 完成。
 * 原两阶段（Phase1=50 + Phase2=300）的在线野值剔除已移除——Phase3 的
 * RobustClean 用中位数+MAD 统一清洗，单阶段 100 样本足以稳健估计均值/参考气压。 */
#define CALIB_SAMPLES   100    /* 100 个样本，~5 秒 */

/* 校准统计参数 */
#define CALIB_STD_COUNT_MAX    10    /* 分块标准差计算块数（每块30个样本） */
#define CALIB_STD_BLOCK_SIZE   30    /* 每块样本数 */

/* 热稳定等待参数：上电后等板载温度达到稳态再开始采样/锁定参考气压，
 * 消除上电热瞬态造成的基准漂移（日志显示重启前后参考气压差 ~10 Pa）。 */
#define THERMAL_MIN_WARMUP_MS   8000    /* 最短预热时间，保证传感器上电稳定 */
#define THERMAL_MAX_WAIT_MS     90000   /* 最长等待上限，防止卡死 */
#define THERMAL_STABLE_RATE     0.02f   /* 温度稳定阈值 (°C/s)：变化率低于此值视为稳定 */
#define THERMAL_STABLE_HOLD_MS  3000    /* 需持续稳定的最短时长 (ms) */

/* 融合方案 1/3/12 双传感器加权权重（有调参值时优先，否则默认 0.10/0.90） */
#ifdef TUNED_FUSION_WEIGHT_MS5611
#define FUSION_WEIGHT_MS5611   TUNED_FUSION_WEIGHT_MS5611
#define FUSION_WEIGHT_BMP280   TUNED_FUSION_WEIGHT_BMP280
#else
#define FUSION_WEIGHT_MS5611   0.10f
#define FUSION_WEIGHT_BMP280   0.90f
#endif

/* ========== BMP280 主导融合方案参数 (FUSION_SCHEME 4 & 5) ========== */
/* 方案 4/14 参数：MS5611 高频增强通道 — 高通滤波器系数 */
#ifdef TUNED_HPF_ALPHA
#define HPF_ALPHA          TUNED_HPF_ALPHA
#else
#define HPF_ALPHA          0.3f    /* 高通滤波系数：越大越灵敏跟踪快速变化 */
                                   /* 推荐范围 0.1~0.5，0.3 在静止噪声和动态响应间平衡 */
#endif

/* 方案 5 参数：运动检测 — 基于 MS5611 滑动窗口残差 */
#define MOTION_WINDOW_SIZE  5       /* 运动检测滑动窗口大小 */
#ifdef TUNED_MOTION_THRESHOLD_PA
#define MOTION_THRESHOLD_PA TUNED_MOTION_THRESHOLD_PA
#else
#define MOTION_THRESHOLD_PA 4.0f    /* 运动判定阈值 (Pa)：超过此值认为在运动 */
                                    /* MS5611 静止噪声 std≈3.05Pa，取 4.0Pa 避免误触发 */
#endif

/* 方案 5 参数：静止 / 运动时的融合权重 */
#ifdef TUNED_WEIGHT_STATIC_MS
#define WEIGHT_STATIC_MS    TUNED_WEIGHT_STATIC_MS
#define WEIGHT_STATIC_BMP   TUNED_WEIGHT_STATIC_BMP
#define WEIGHT_MOTION_MS    TUNED_WEIGHT_MOTION_MS
#define WEIGHT_MOTION_BMP   TUNED_WEIGHT_MOTION_BMP
#else
#define WEIGHT_STATIC_MS    0.05f   /* 静止时 MS5611 占比 5% */
#define WEIGHT_STATIC_BMP   0.95f   /* 静止时 BMP280 占比 95% */
#define WEIGHT_MOTION_MS    0.40f   /* 运动时 MS5611 占比 40% */
#define WEIGHT_MOTION_BMP   0.60f   /* 运动时 BMP280 占比 60% */
#endif

/* 方案 5 参数：权重平滑过渡系数（防止频繁切换导致的输出跳变） */
#ifdef TUNED_WEIGHT_SMOOTH_ALPHA
#define WEIGHT_SMOOTH_ALPHA TUNED_WEIGHT_SMOOTH_ALPHA
#else
#define WEIGHT_SMOOTH_ALPHA 0.2f    /* 权重平滑：越小过渡越慢越平滑 */
#endif

/* ========== 方案 6 参数：自适应高度变化量加权融合 ========== */
/* 原理：BMP280 提供绝对高度基准，MS5611 与 BMP280 的帧间高度变化量做加权融合，
 * 融合后的变化量叠加到 BMP280 的绝对高度上。
 * height_fused = BMP280_height + (delta_ms * w_ms + delta_bmp * w_bmp)
 * delta_ms = MS5611_height - MS5611_height_prev   (帧间变化)
 * delta_bmp = BMP280_height - BMP280_height_prev   (帧间变化)
 *
 * 权重自适应：静止时以 BMP280 变化为主（抑制 MS5611 噪声），
 *           运动时增大 MS5611 权重（快速跟踪动态变化）。 */
/* 静止 / 运动时的 delta 权重（方案7 调参值优先，否则默认） */
#ifdef TUNED_W_DELTA_MS_STATIC
#define W_DELTA_MS_STATIC    TUNED_W_DELTA_MS_STATIC
#define W_DELTA_BMP_STATIC   TUNED_W_DELTA_BMP_STATIC
#define W_DELTA_MS_MOTION    TUNED_W_DELTA_MS_MOTION
#define W_DELTA_BMP_MOTION   TUNED_W_DELTA_BMP_MOTION
#else
#define W_DELTA_MS_STATIC    0.05f   /* 静止时 MS5611 变化量权重（BMP280 主导） */
#define W_DELTA_BMP_STATIC   0.95f   /* 静止时 BMP280 变化量权重 */
#define W_DELTA_MS_MOTION    0.50f   /* 运动时 MS5611 变化量权重 */
#define W_DELTA_BMP_MOTION   0.50f   /* 运动时 BMP280 变化量权重 */
#endif

/* 方案6 运动检测（复用方案5的窗口和阈值参数） */
/* MOTION_WINDOW_SIZE / MOTION_THRESHOLD_PA 已在方案5中定义 */

/* 方案7 权重平滑过渡系数 */
#ifdef TUNED_DELTA_WEIGHT_SMOOTH_ALPHA
#define DELTA_WEIGHT_SMOOTH_ALPHA  TUNED_DELTA_WEIGHT_SMOOTH_ALPHA
#else
#define DELTA_WEIGHT_SMOOTH_ALPHA  0.2f
#endif

/* ========== 方案11 参数：Delta 置信度加权累积融合 ========== */
/* 原理：不融合绝对气压，而是融合帧间气压变化量（Delta），累加跟踪相对高度。
 * 置信度 = 1/(var(delta_window) + ε)，窗口内 Delta 方差越小 → 越稳定 → 置信度越高。
 * 泄漏锚：静止时缓慢拉回 BMP280 绝对气压，消除累积漂移。 */
#define DELTA_CONF_WINDOW    10       /* 置信度估计窗口大小（帧数） */
#ifdef TUNED_DELTA_CONF_EPS
#define DELTA_CONF_EPS       TUNED_DELTA_CONF_EPS
#else
#define DELTA_CONF_EPS       0.1f     /* 置信度正则化因子，防止除零 */
#endif
#ifdef TUNED_ANCHOR_ALPHA
#define ANCHOR_ALPHA         TUNED_ANCHOR_ALPHA
#else
#define ANCHOR_ALPHA         0.05f    /* 泄漏锚系数：静止时拉回 BMP280 的速度 */
#endif

/* ========== 方案12 参数：Hampel 脉冲抑制预处理器 ========== */
/* 原理：对每个传感器维护 5 帧滑动窗口，计算中值和 MAD（中值绝对偏差），
 * 若当前帧偏离中值超过 threshold×sigma，则用中值替代。
 * 只替换离群值，平稳时直通。 */
#define HAMPEL_WINDOW_SIZE   5        /* 滑动窗口大小 */
#define HAMPEL_THRESHOLD     3.0f     /* MAD 倍数阈值：超过 3×sigma 视为离群 */

/* ========== 方案13 参数：二阶互补融合 ========== */
/* 原理：状态 = (P_fused 融合气压, D_fused 气压变化率)
 * 预测步：MS5611 主导速度估算
 * 更新步：BMP280 绝对锚定修正
 * COMP_BETA：速度更新中 MS5611 的权重（0.7=MS5611 主导速度跟踪）
 * COMP_ALPHA：位置锚定中 BMP280 的权重（0.15=慢速锚定，等效 τ≈6 帧） */
#ifdef TUNED_COMP_BETA
#define COMP_BETA            TUNED_COMP_BETA
#else
#define COMP_BETA            0.7f     /* 速度更新权重：MS5611 Delta 占 70% */
#endif
#ifdef TUNED_COMP_ALPHA
#define COMP_ALPHA           TUNED_COMP_ALPHA
#else
#define COMP_ALPHA           0.15f    /* 位置锚定权重：BMP280 修正速度 */
#endif

/* ========== 方案14 参数：方案4 + BMP280 温漂补偿 ========== */
/* 原理：基于方案4的 BMP280 主导 + MS5611 高频增强通道，
 * 新增 BMP280 实时温度漂移补偿，用温度偏差线性校正 BMP280 气压，
 * 抵消空调温循环/环境温度变化引起的缓慢气压漂移。
 * P_compensated = P_raw - TC_COEFF × (T_current - T_reference)
 * 补偿时机：校准完成后第一帧记录参考温度，之后每帧实时校正 */
#ifdef TUNED_TC_COEFF
#define TC_COEFF            TUNED_TC_COEFF
#else
#define TC_COEFF            0.5f     /* 温漂系数 (Pa/°C)：每偏离参考温度 1°C 校正的气压值 */
#endif
                                     /* BMP280 残余温漂约 0.2~1.0 Pa/°C，默认 0.5 兼顾静止和空调环境 */
                                     /* 正值 = 温度升高时气压偏高 → 向下修正（减气压） */

/* 传感器健康状态 */
#define SENSOR_HEALTH_GOOD     2
#define SENSOR_HEALTH_FAIR     1
#define SENSOR_HEALTH_POOR     0

/* 校准状态机 */
typedef enum {
    CALIB_STATE_IDLE = 0,
    CALIB_STATE_PHASE1,
    CALIB_STATE_PHASE2,
    CALIB_STATE_ANALYSIS,
    CALIB_STATE_COMPLETE,
    CALIB_STATE_FAILED
} CalibState_t;

/* 校准统计数据结构 */
typedef struct {
    float mean;                 /* 均值 */
    float variance;             /* 方差 */
    float std;                  /* 标准差 */
    float min;                  /* 最小值 */
    float max;                  /* 最大值 */
    float range;                /* 极差 */
    int   valid_count;          /* 有效样本数 */
    int   outlier_count;        /* 剔除的异常值数 */
    float outlier_ratio;        /* 异常值比例 */
    float block_std[CALIB_STD_COUNT_MAX]; /* 分块标准差 */
    int   block_count;          /* 实际分块数 */
    float mean_block_std;       /* 分块标准差均值（评估短期噪声） */
    int   health;               /* 传感器健康状态 */
} CalibStats_t;

/* 校准结果数据结构 */
typedef struct {
    CalibState_t state;
    int phase;                  /* 当前阶段 1/2 */

    /* MS5611 统计 */
    CalibStats_t ms5611_stats;
    float ms5611_pressure_avg;

    /* BMP280 统计 */
    CalibStats_t bmp280_stats;
    float bmp280_pressure_avg;

    /* 双传感器一致性 */
    float diff_mean;            /* MS5611 - BMP280 均值差 */
    float diff_std;             /* 差值标准差 */
    float consistency;          /* 一致性评分 0.0~1.0 */

    /* 自动调参结果 */
    float kf_r_ms5611;
    float kf_q_ms5611;
    float kf_r_bmp280;
    float kf_q_bmp280;
    float fusion_weight_ms5611;
    float fusion_weight_bmp280;
    float ekf_accel_sigma;
    float ekf_pressure_sigma;

    /* 参考气压 */
    float reference_pressure_pa;      /* 统一参考气压（基于 MS5611 校准，用于所有高度计算） */
    float current_altitude;

    /* 校准耗时 */
    uint32_t elapsed_ms;
} CalibResult_t;

/* 校准进度回调（用于 OLED 显示） */
typedef void (*CalibProgressCb)(int phase, int progress, int total, const char *msg);
/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
