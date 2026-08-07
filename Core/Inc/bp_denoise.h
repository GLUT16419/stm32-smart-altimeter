#ifndef BP_DENOISE_H
#define BP_DENOISE_H

#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * BP 神经网络气压去噪（移植自参考项目 sahixi 的 ms5611_bp_real / bmp280_bp_real）
 *
 * 结构:  输入 5 点气压滑窗 → Dense(5→8, tanh) → Dense(8→1, linear) → 输出 1
 * 权重:  直接取自 sahixi 的 X-CUBE-AI 生成代码（反解为 float32），
 *        完整保留其预训练权重，不重新训练。
 * 前向:  与 X-CUBE-AI 的 forward_dense 一致：
 *          h[o] = tanh( b0[o] + Σ_i W0[o*5+i] * x[i] )     (o=0..7, i=0..4)
 *          y    = b2[0] + Σ_o W2[o] * h[o]
 *
 * 该模块为纯 C 自包含实现，不依赖 X-CUBE-AI 运行时，避免改动现有多任务网络，
 * 与方案15/16/17 互不干扰。
 * ═══════════════════════════════════════════════════════════════════════════ */

/* 每传感器去噪状态（维护 5 点滑窗与 EMA） */
typedef struct {
    float hist[5];   /* 最近 5 个气压 (hPa), 顺序 [t-4, t-3, t-2, t-1, t] */
    int   cnt;       /* 已填充个数 (满 5 后方可推理) */
    float ema;       /* EMA 后输出 (hPa) */
    int   inited;    /* EMA 是否已初始化 */
} BP_Denoise_t;

/* 单传感器 BP 去噪一步
 *   raw_hpa  : 当前原始气压 (hPa)
 *   scale    : 滑窗归一化尺度 (sahixi: 5.0)
 *   ema_alpha: EMA 系数 (sahixi: 0.1)
 *   返回去噪后气压 (hPa)；滑窗未满时返回 0（调用方应跳过）
 */
float BP_Denoise_Update(BP_Denoise_t *s,
                        const float w0[40], const float b0[8],
                        const float w2[8],  const float b2[1],
                        float raw_hpa, float scale, float ema_alpha);

/* 两套预训练权重（MS5611 / BMP280，来源 sahixi *_bp_real） */
extern const float BP_MS_W0[40], BP_MS_B0[8], BP_MS_W2[8], BP_MS_B2[1];
extern const float BP_BM_W0[40], BP_BM_B0[8], BP_BM_W2[8], BP_BM_B2[1];

#endif /* BP_DENOISE_H */
