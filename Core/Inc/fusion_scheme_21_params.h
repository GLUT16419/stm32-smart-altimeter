/*
 * fusion_scheme_21_params.h
 * 方案 21：基于方案 20（双传感器各自气压域 EKF + BP 去噪）之上，
 *          对 MS5611-EKF 与 BMP280-EKF 两条主通道做"方差倒数加权"融合，
 *          得到比任一单传感器更稳的融合气压/高度（1+1>2）。
 *
 * 融合数学：
 *   1) 在线偏置对齐（关键，解决"两传感器本来差距就大"）：
 *      实测两传感器存在不可忽视的系统性气压差，且固定扣一个
 *      S20_BM_P0_OFFSET 不够。本方案用极慢 EMA 在线估计真实偏置：
 *          bias      = EMA(bm_ekf_pa - ms_ekf_pa)      （基于已平滑的 EKF 估计）
 *          p_bm_aligned = bm_ekf_pa - bias
 *      对齐后 p_ms 与 p_bm_aligned 落在同一 MS5611 基准上，可稳定加权。
 *   2) 自适应权重：用每路【实测新息(残差)方差】的 EMA 倒数作为权重：
 *          res = 传感器原始气压 - 该路 EKF 估计
 *          var = EMA(res^2)          （在线跟踪每路的实际噪声/异常水平）
 *          w   = 1 / (var + eps)
 *      → MS5611 更吵 → var 大 → 权重自动降低；某传感器突跳/失效 → res 飙升
 *        → var 增大 → 权重趋零（天然冗余容错），BMP280 干净则权重更高。
 *   3) 权重平滑 + 限幅（解决"波动巨大"）：对步骤 2 的瞬时权重先做长 EMA 平滑，
 *      再限幅到 [W_MIN, W_MAX]，任一传感器权重不会瞬间塌到 0 / 飙到 1，
 *      从而让融合输出平滑稳定（这是消除帧间抖动的主手段）。
 *   4) 融合：p_fus = w_ms*p_ms + (1-w_ms)*p_bm_aligned，
 *            再用 MS5611 基准 + 温度补偿 ISA 公式换算融合高度。
 *
 * 说明：EKF / BP 去噪参数直接继承方案 20（S20_EKF_* / S20_BP_* / S20_BM_P0_OFFSET）。
 */
#ifndef FUSION_SCHEME_21_PARAMS_H
#define FUSION_SCHEME_21_PARAMS_H

#include "fusion_scheme_20_params.h"   /* 复用 S20_EKF_* / S20_BP_* / S20_BM_P0_OFFSET */

/* ===== 融合方法 ===== */
/* 0 = 残差方差倒数加权（自适应，推荐）；1 = 固定权重平均（回退/对照用） */
#define S21_FUSE_METHOD      0

/* 固定权重法（S21_FUSE_METHOD==1）下 MS5611 的权重，BMP280 取 (1-w) */
#define S21_FIXED_W_MS       0.5f

/* ---- 在线偏置估计：BMP280 相对 MS5611 的系统性气压差 ----
 * 两段式：前 S21_BIAS_CONVERGE_FRAMES 帧用较快 EMA 快速收敛到真实偏置；
 *         之后冻结（不再更新），彻底消除偏置追噪声导致的融合漂移。
 * 500 帧 @50Hz ≈ 10 秒，确保两路 EKF 都进入稳态后偏置再冻结。 */
#define S21_BIAS_CONVERGE_FRAMES  500   /* 偏置收敛期帧数 */

/* ---- 残差方差 EMA：越小越平滑（对瞬时突跳更钝） ---- */
#define S21_VAR_EMA          0.02f

/* 方差下限，防止 1/var 溢出（Pa^2） */
#define S21_VAR_EPS          1e-3f

/* ---- 权重平滑与限幅：消除"方差倒数加权"的帧间抖动 ---- */
#define S21_W_EMA            0.02f
#define S21_W_MIN            0.10f
#define S21_W_MAX            0.90f

/* ===== 融合输出低通（在权重融合之后再加一级 EMA） =====
 * 用于将静止噪声压到 3-4cm（0.03-0.04m）以内。
 * 0.001 @50Hz ≈ 时间常数 ~20s：静止噪声 ~3cm，75cm 升降约 40-50s 到位。
 * 若需要更跟手可增大（如 0.002 = ~10s），静止噪声升至 ~5cm。 */
#define S21_OUTPUT_EMA       0.001f

/* ===== 融合气压→高度所用温度源 ===== */
/* 0 = 用 BMP280 温度（更稳）；1 = 两传感器温度平均 */
#define S21_FUSE_TEMP_SRC    0

#endif /* FUSION_SCHEME_21_PARAMS_H */
