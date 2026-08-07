# -*- coding: utf-8 -*-
"""
方案20 参数调优脚本（离线仿真 + 网格扫描）
========================================
复用 sim_scheme20.py 的算法函数，增强以匹配当前固件真实信号链：

1) 建模 BMP280 驱动内的 sahixi 软件预滤波（5点滑动平均 + 突变拒绝，
   移植自 Core/Src/bsp_bmp280.c）。固件中 BMP280_Read_Data 已返回该预滤波
   气压再送 EKF/BP；MS5611 无此预滤波。仿真仅 BMP280 套用，使调参针对
   真实到达 EKF/BP 的信号。

2) 指标同时考察"静止段噪声抑制"与"运动段跟踪误差(RMS)"，并用均衡指标
   balanced = sqrt(静止² + 运动²) 避免只压噪声却牺牲动态跟手。

3) EKF(Q/R) 与 BP(ALPHA) 是固件两条并行去噪支路，相互独立 → 分别寻优后合并。

输出推荐参数，供回写 sim_scheme20.py 与 Core/Inc/fusion_scheme_20_params.h。
"""

import numpy as np
from sim_scheme20 import (
    BaroEKF, bp_denoise, p2a_with_temp, make_trajectory,
    S20_BM_P0_OFFSET, ISA_T0, ISA_L, EXP, DT, FS,
    BP_MS_W0, BP_MS_B0, BP_MS_W2, BP_MS_B2,
    BP_BM_W0, BP_BM_B0, BP_BM_W2, BP_BM_B2,
)

N = 2500
P0 = 101325.0

ms_meas, ms_true, ms_temp = make_trajectory(N, P0, noise_std_pa=20.0, offset_pa=0.0, seed=11)
bm_meas, bm_true, bm_temp = make_trajectory(N, P0, noise_std_pa=4.0, offset_pa=S20_BM_P0_OFFSET, seed=22)

def sahixi_sw_filter(seq_pa, n=5, a_hpa=0.1):
    seq = np.asarray(seq_pa, float) / 100.0
    L = len(seq)
    buf = [0.0] * n
    idx = 0
    out = np.zeros(L)
    for k in range(L):
        x = seq[k]
        if buf[idx] == 0.0:
            buf[idx] = x
            out[k] = x
            idx = (idx + 1) % n
            continue
        delta = (x - buf[idx - 1]) if idx else (x - buf[n - 1])
        if abs(delta) < a_hpa:
            buf[idx] = x
            idx = (idx + 1) % n
        s = 0.0
        for c in range(n):
            s += buf[c]
        out[k] = s / n
    return out * 100.0

bm_pre = sahixi_sw_filter(bm_meas)
ms_pre = ms_meas

nstat = max(20, int(0.30 * N))

def metrics(est, true):
    est = np.asarray(est, float); true = np.asarray(true, float)
    stat = float(np.std((est - true)[:nstat]))
    mov = float(np.sqrt(np.mean((est - true)[nstat:] ** 2)))
    tot = float(np.sqrt(np.mean((est - true) ** 2)))
    return stat, mov, tot

def balanced(ch):
    s, m, _ = ch
    return (s * s + m * m) ** 0.5

def evaluate(q_press, q_rate, r_meas, bp_alpha, bp_scale=5.0):
    ekf_ms = BaroEKF(P0, q_press, q_rate, r_meas, DT)
    ekf_bm = BaroEKF(P0 + S20_BM_P0_OFFSET, q_press, q_rate, r_meas, DT)
    ms_ekf = np.array([ekf_ms.update(ms_pre[i], ms_temp[i]) for i in range(N)])
    bm_ekf = np.array([ekf_bm.update(bm_pre[i], bm_temp[i]) for i in range(N)])
    ms_bp = bp_denoise(ms_pre / 100.0, BP_MS_W0, BP_MS_B0, BP_MS_W2, BP_MS_B2,
                       scale=bp_scale, alpha=bp_alpha) * 100.0
    bm_bp = bp_denoise(bm_pre / 100.0, BP_BM_W0, BP_BM_B0, BP_BM_W2, BP_BM_B2,
                       scale=bp_scale, alpha=bp_alpha) * 100.0
    ch = {
        'MS_EKF': metrics(ms_ekf, ms_true),
        'MS_BP':  metrics(ms_bp,  ms_true),
        'BM_EKF': metrics(bm_ekf, bm_true),
        'BM_BP':  metrics(bm_bp,  bm_true),
    }
    ekf_obj = np.mean([balanced(ch['MS_EKF']), balanced(ch['BM_EKF'])])
    bp_obj  = np.mean([balanced(ch['MS_BP']),  balanced(ch['BM_BP'])])
    return ch, ekf_obj, bp_obj

def report(name, ch):
    print(f"  [{name}]")
    for k, (s, m, t) in ch.items():
        print(f"    {k:7s} 静止std={s:7.3f}  运动RMS={m:7.3f}  整体RMS={t:7.3f}")

# ---- 基线 ----
print("=" * 64)
print("基线（Q_PRESS=5.0 Q_RATE=1.0 R_MEAS=0.15 BP_ALPHA=0.1）")
base_ch, _, _ = evaluate(5.0, 1.0, 0.15, 0.1)
report("基线", base_ch)
base_ekf = np.mean([balanced(base_ch['MS_EKF']), balanced(base_ch['BM_EKF'])])
base_bp  = np.mean([balanced(base_ch['MS_BP']),  balanced(base_ch['BM_BP'])])

# ---- 网格扫描（EKF / BP 独立寻优） ----
print("=" * 64)
print("网格扫描中 ...")
Q_P = [1.0, 2.0, 5.0, 10.0]
Q_R = [0.5, 1.0, 2.0, 5.0]
R_M = [0.1, 0.2, 0.5, 1.0]
A_BP = [0.1, 0.2, 0.3, 0.5, 0.7]

best_ekf = None   # (ekf_obj, qp, qr, rm)
best_bp = None    # (bp_obj, alpha)

for qp in Q_P:
    for qr in Q_R:
        for rm in R_M:
            for al in A_BP:
                ch, ekf_obj, bp_obj = evaluate(qp, qr, rm, al)
                if best_ekf is None or ekf_obj < best_ekf[0]:
                    best_ekf = (ekf_obj, qp, qr, rm)
                if best_bp is None or bp_obj < best_bp[0]:
                    best_bp = (bp_obj, al)

be_obj, bqp, bqr, brm = best_ekf
bp_obj, bal = best_bp

print(f"EKF 支路最优: Q_PRESS={bqp} Q_RATE={bqr} R_MEAS={brm}  (均衡指标 {be_obj:.4f} vs 基线 {base_ekf:.4f})")
print(f"BP  支路最优: BP_EMA_ALPHA={bal}                      (均衡指标 {bp_obj:.4f} vs 基线 {base_bp:.4f})")

# ---- 合并推荐并复盘 ----
print("=" * 64)
print(f"推荐合并: Q_PRESS={bqp} Q_RATE={bqr} R_MEAS={brm} BP_EMA_ALPHA={bal}")
rec_ch, _, _ = evaluate(bqp, bqr, brm, bal)
report("推荐", rec_ch)
print("（EKF 支路用其最优 Q/R；BP 支路用其最优 alpha；两支路独立，互不牺牲）")
