# -*- coding: utf-8 -*-
"""
方案21 离线仿真（在方案20 双 EKF 之上做残差方差倒数加权融合）
================================================================
忠实复现固件 (FUSION_SCHEME=21) 的融合链路，并验证针对两项实测问题的修复：

  [问题1] 两传感器“本来差距就大”  → 修复：在线估计 BMP280 相对 MS5611 的真实偏置，
                                      融合前扣除，吸收任意恒定/缓变差距（不再只扣固定 offset）。
  [问题2] 融合“波动巨大”          → 修复：残差方差倒数权重先做长 EMA 平滑、再限幅到
                                      [W_MIN,W_MAX]，消除帧间权重抖动（波动主因）。

融合链路：
  MS5611_raw ─> 气压域 EKF ─> ms_ekf_pa ─────────────┐
  BMP280_raw ─> 气压域 EKF ─> bm_ekf_pa ─> (减 online bias) ─> p_bm_aligned ┤
      bias = EMA(bm_ekf - ms_ekf)  （极慢，吸收两传感器系统性差距）
      w = 1/EMA((raw-est)^2)  → 长 EMA 平滑 → 限幅[W_MIN,W_MAX]
      p_fus = w*ms_ekf + (1-w)*p_bm_aligned

输出（每张图只含一个坐标系，按场景拆分为多张独立图片）:
  场景1 正常工况:
    sim_scheme21_a_pressure.png   气压域：真值/双 EKF(已对齐)/融合
    sim_scheme21_b_height.png     高度域：真值/单 MS5611/单 BMP280/融合
    sim_scheme21_c_wms.png       权重 w_MS5611
    sim_scheme21_d_wbm.png       权重 w_BMP280 (=1-w_MS5611)
  场景2 BMP280 噪声爆发失效容错:
    sim_scheme21_failover_a_height.png   高度域 + 失效起点竖线
    sim_scheme21_failover_b_weight.png   权重 + 失效起点竖线
  场景3 大差距对照（旧法 vs 新法）:
    sim_scheme21_gapfix_a_pressure.png  气压域 新旧融合对比
    sim_scheme21_gapfix_b_height.png    高度域 新旧融合对比
    sim_scheme21_gapfix_c_weight.png   权重 新旧对比
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# ============== ISA 常量（与 altitude_convert.h 一致） ==============
ISA_T0 = 288.15
ISA_L  = 0.0065
ISA_G  = 9.80665
ISA_R  = 287.05
EXP    = ISA_L * ISA_R / ISA_G          # ≈ 0.190263

# ============== 方案20/21 参数（与头文件一致） ==============
S20_EKF_Q_PRESS = 2.0
S20_EKF_Q_RATE  = 5.0
S20_EKF_R_MEAS  = 1.0                  # 均衡值：静止噪声~0.04m，75cm 升降 ~1s 到位
S20_BM_P0_OFFSET = 100.0               # Pa（BMP280 相对 MS5611 固定系统偏差，固件用）

# 方案21 融合参数（与 fusion_scheme_21_params.h 一致）
S21_FUSE_METHOD = 0                    # 0=残差方差倒数加权, 1=固定权重
S21_FIXED_W_MS  = 0.5
S21_BIAS_CONVERGE_FRAMES = 500        # 偏置收敛期帧数（之后冻结）
S21_VAR_EMA     = 0.02                 # 残差方差 EMA
S21_VAR_EPS     = 1e-3
S21_W_EMA       = 0.02                 # 权重平滑 EMA
S21_W_MIN       = 0.10
S21_W_MAX       = 0.90
S21_OUTPUT_EMA  = 0.005                # 输出低通（~4s @50Hz，静止噪声~5cm）
S21_FUSE_TEMP_SRC = 0

DT = 0.02                               # 50 Hz
FS = 1.0 / DT


# ============== 气压域 EKF（与 baro_ekf.c 一致） ==============
class BaroEKF:
    def __init__(self, p0_pa, q_press, q_rate, r_meas, dt):
        self.p = p0_pa
        self.dp = 0.0
        self.P = [[100.0, 0.0], [0.0, 100.0]]
        self.q_press = q_press
        self.q_rate = q_rate
        self.r_meas = r_meas
        self.p0 = p0_pa
        self.dt = dt

    def get_p00(self):
        return self.P[0][0]

    def update(self, pressure_pa, temp_c):
        dt = self.dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        x0 = self.p + dt * self.dp
        x1 = self.dp
        P00, P01 = self.P[0][0], self.P[0][1]
        P10, P11 = self.P[1][0], self.P[1][1]
        P00 = P00 + dt * (P10 + P01) + dt2 * P11
        P01 = P01 + dt * P11
        P10 = P10 + dt * P11
        P00 += self.q_rate * dt3 / 3.0 + self.q_press * dt
        P01 += self.q_rate * dt2 / 2.0
        P10 += self.q_rate * dt2 / 2.0
        P11 += self.q_rate * dt
        T0 = temp_c + 273.15
        Pabs = x0 if x0 > 100.0 else self.p0
        ratio = (Pabs / self.p0) ** (EXP - 1.0)
        Hjac = -(T0 / ISA_L) * EXP * ratio / self.p0
        h_pred = (T0 / ISA_L) * (1.0 - (x0 / self.p0) ** EXP)
        h_meas = (T0 / ISA_L) * (1.0 - (pressure_pa / self.p0) ** EXP)
        y = h_meas - h_pred
        S = Hjac * P00 * Hjac + self.r_meas
        Si = 1.0 / S if S != 0.0 else 0.0
        K0 = P00 * Hjac * Si
        K1 = P10 * Hjac * Si
        self.p = x0 + K0 * y
        self.dp = x1 + K1 * y
        self.P[0][0] = (1.0 - K0 * Hjac) * P00
        self.P[0][1] = (1.0 - K0 * Hjac) * P01
        self.P[1][0] = self.P[0][1]
        self.P[1][1] = P11 - K1 * Hjac * P01
        return self.p


# ============== 高度换算（与 altitude_convert.c 一致） ==============
def p2a_with_temp(p, p0, t):
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    ratio = p / p0
    ratio = np.clip(ratio, 1e-6, 10.0)
    alt = (ISA_T0 / ISA_L) * (1.0 - np.power(ratio, EXP))
    tk = t + 273.15
    dt = tk - (ISA_T0 - ISA_L * alt)
    alt = alt + dt * 0.035
    return np.where((alt > -1000.0) & (alt < 20000.0), alt, 0.0)


# ============== 方案21 融合：新方法（与固件一致） ==============
def fuse_new(ms_ekf_pa, bm_ekf_pa, ms_raw_pa, bm_raw_pa):
    """在线偏置 + 残差方差倒数权重(平滑+限幅)。返回 (p_fus, w_ms, bias_trace)。"""
    n = len(ms_ekf_pa)
    p_fus = np.zeros(n)
    w_ms = np.zeros(n)
    bias_trace = np.zeros(n)
    var_ms = 0.0
    var_bm = 0.0
    bias = 0.0
    w = 0.5
    fus_lp = 0.0
    inited = False
    bias_cnt = 0
    for i in range(n):
        # (1) 两段式偏置：收敛期快速跟踪，之后冻结
        if not inited:
            bias = bm_ekf_pa[i] - ms_ekf_pa[i]
            bias_cnt = 0
        elif bias_cnt < S21_BIAS_CONVERGE_FRAMES:
            bias = 0.01 * (bm_ekf_pa[i] - ms_ekf_pa[i]) + 0.99 * bias
            bias_cnt += 1
        # else: 冻结偏置
        p_bm_aligned = bm_ekf_pa[i] - bias

        # (2) 慢速 EMA 跟踪各路残差方差
        res_ms = ms_raw_pa[i] - ms_ekf_pa[i]
        res_bm = bm_raw_pa[i] - bm_ekf_pa[i]
        if not inited:
            var_ms = res_ms * res_ms
            var_bm = res_bm * res_bm
            # NOT setting inited here — 等输出低通初始化后再设置
        else:
            var_ms = S21_VAR_EMA * (res_ms * res_ms) + (1.0 - S21_VAR_EMA) * var_ms
            var_bm_res = res_bm * res_bm
            var_bm = S21_VAR_EMA * var_bm_res + (1.0 - S21_VAR_EMA) * var_bm

        # (3) 权重：方差倒数 → 长 EMA 平滑 → 限幅
        if S21_FUSE_METHOD == 0:
            wm_raw = 1.0 / (var_ms + S21_VAR_EPS)
            wb_raw = 1.0 / (var_bm + S21_VAR_EPS)
            w_inst = wm_raw / (wm_raw + wb_raw)
            w = S21_W_EMA * w_inst + (1.0 - S21_W_EMA) * w
        else:
            w = S21_FIXED_W_MS
        w_c = w if S21_W_MIN <= w <= S21_W_MAX else (S21_W_MIN if w < S21_W_MIN else S21_W_MAX)

        # (4) 融合加权
        fus = w_c * ms_ekf_pa[i] + (1.0 - w_c) * p_bm_aligned
        # (5) 输出低通（级联在权重融合之后，与固件 S21_OUTPUT_EMA 一致）
        if not inited:
            fus_lp = fus
            inited = True       # 低通初始完成后才标记初始化完成
        else:
            fus_lp = S21_OUTPUT_EMA * fus + (1.0 - S21_OUTPUT_EMA) * fus_lp
        p_fus[i] = fus_lp
        w_ms[i] = w_c
        bias_trace[i] = bias
    return p_fus, w_ms, bias_trace


# ============== 方案21 融合：旧方法（仅作对照，复现“波动巨大”问题） ==============
def fuse_old(ms_ekf_pa, bm_ekf_pa, ms_raw_pa, bm_raw_pa):
    """旧法：固定 offset + 未平滑的残差方差倒数权重（VAR_EMA=0.05）。"""
    n = len(ms_ekf_pa)
    p_fus = np.zeros(n)
    wm_n = np.zeros(n)
    var_ms = 0.0
    var_bm = 0.0
    inited = False
    for i in range(n):
        p_bm_aligned = bm_ekf_pa[i] - S20_BM_P0_OFFSET
        res_ms = ms_raw_pa[i] - ms_ekf_pa[i]
        res_bm = bm_raw_pa[i] - bm_ekf_pa[i]
        if not inited:
            var_ms = res_ms * res_ms
            var_bm = res_bm * res_bm
            inited = True
        else:
            var_ms = 0.05 * (res_ms * res_ms) + 0.95 * var_ms
            var_bm = 0.05 * (res_bm * res_bm) + 0.95 * var_bm
        wm = 1.0 / (var_ms + S21_VAR_EPS)
        wb = 1.0 / (var_bm + S21_VAR_EPS)
        wsum = wm + wb
        wm_n[i] = wm / wsum
        p_fus[i] = (ms_ekf_pa[i] * wm + p_bm_aligned * wb) / wsum
    return p_fus, wm_n


# ============== 合成数据 ==============
def make_trajectory(n_steps, p0_pa, noise_std_pa, offset_pa=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * DT
    T = n_steps * DT
    base = p0_pa
    ramp = np.where(t < 0.4 * T, 0.0,
                    300.0 * np.sin(2 * np.pi * (t - 0.4 * T) / (0.6 * T)) * 0.5)
    true = base + ramp                       # 共同真值（不含传感器偏差）
    meas = true + offset_pa + rng.normal(0.0, noise_std_pa, n_steps)
    temp = 25.0 + 1.5 * np.sin(2 * np.pi * t / T * 0.5)
    return meas, true, temp


def rmse(x, ref, mask=None):
    d = np.asarray(x) - np.asarray(ref)
    if mask is not None:
        d = d[mask]
    return float(np.sqrt(np.mean(d * d)))


def frame_jump_m(h):
    """帧间最大跳变（米）与 RMS 抖动（米），量化“波动巨大”。"""
    d = np.abs(np.diff(h))
    return float(np.max(d)), float(np.sqrt(np.mean(d * d)))


def run_pipeline(N, P0, ms_noise, bm_noise, seed_ms, seed_bm,
                 bm_offset=S20_BM_P0_OFFSET, bm_fail=None):
    """跑一遍完整链路（双 EKF）。bm_offset 为 BMP280 真实系统性偏差（默认=固件固定值）。"""
    ms_meas, ms_true, ms_temp = make_trajectory(N, P0, ms_noise, offset_pa=0.0, seed=seed_ms)
    bm_meas, bm_true, bm_temp = make_trajectory(N, P0, bm_noise, offset_pa=bm_offset, seed=seed_bm)
    true = ms_true

    if bm_fail is not None:
        fidx, fail_std = bm_fail
        rng = np.random.default_rng(999)
        burst = rng.normal(0.0, fail_std, N - fidx)
        spikes = (rng.random(N - fidx) < 0.05) * rng.normal(0.0, 5 * fail_std, N - fidx)
        bm_meas[fidx:] = bm_meas[fidx:] + burst + spikes

    # BMP280 EKF 与 MS5611 EKF 共用同一 P0（固定 offset 已由在线偏置吸收）
    ekf_ms = BaroEKF(P0, S20_EKF_Q_PRESS, S20_EKF_Q_RATE, S20_EKF_R_MEAS, DT)
    ekf_bm = BaroEKF(P0, S20_EKF_Q_PRESS, S20_EKF_Q_RATE, S20_EKF_R_MEAS, DT)
    ms_ekf_pa = np.array([ekf_ms.update(ms_meas[i], ms_temp[i]) for i in range(N)])
    bm_ekf_pa = np.array([ekf_bm.update(bm_meas[i], bm_temp[i]) for i in range(N)])

    t_fus = bm_temp if S21_FUSE_TEMP_SRC == 0 else 0.5 * (ms_temp + bm_temp)
    ms_h = p2a_with_temp(ms_ekf_pa, P0, ms_temp)
    bm_h = p2a_with_temp(bm_ekf_pa, P0, bm_temp)
    true_h = p2a_with_temp(true, P0, ms_temp)

    return dict(N=N, true=true, true_h=true_h,
                ms_meas=ms_meas, bm_meas=bm_meas,
                ms_ekf_pa=ms_ekf_pa, bm_ekf_pa=bm_ekf_pa,
                ms_h=ms_h, bm_h=bm_h, t_fus=t_fus,
                ms_temp=ms_temp, bm_temp=bm_temp)


def main():
    N = 2500
    P0 = 101325.0
    out_dir = os.path.dirname(os.path.abspath(__file__))
    t_axis = np.arange(N) / FS
    nstat = max(20, int(0.30 * N))
    stat = slice(0, nstat)
    move = slice(int(0.45 * N), N)

    # ========== 通用出图辅助：单坐标系 ==========
    def save_single(fig, fname):
        fig.tight_layout()
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"[完成] 已保存: {fpath}")
        return fpath

    # ===== 场景1：正常工况（MS5611 吵 ~20Pa，BMP280 干净 ~4Pa）=====
    R = run_pipeline(N, P0, ms_noise=20.0, bm_noise=4.0, seed_ms=11, seed_bm=22)
    p_fus, wm_n, bias_tr = fuse_new(R['ms_ekf_pa'], R['bm_ekf_pa'], R['ms_meas'], R['bm_meas'])
    fus_h = p2a_with_temp(p_fus, P0, R['t_fus'])
    R['p_fus'] = p_fus; R['fus_h'] = fus_h; R['wm_n'] = wm_n

    print("========== 方案21 融合仿真（正常工况）==========")
    print("[静止段 高度噪声 std, m]")
    print(f"  单 MS5611 EKF : {np.std((R['ms_h'] - R['true_h'])[stat]):.4f}")
    print(f"  单 BMP280 EKF : {np.std((R['bm_h'] - R['true_h'])[stat]):.4f}")
    print(f"  融合 FUSED    : {np.std((R['fus_h'] - R['true_h'])[stat]):.4f}")
    print("[运动段 高度 RMSE, m]")
    print(f"  单 MS5611 EKF : {rmse(R['ms_h'], R['true_h'], move):.4f}")
    print(f"  单 BMP280 EKF : {rmse(R['bm_h'], R['true_h'], move):.4f}")
    print(f"  融合 FUSED    : {rmse(R['fus_h'], R['true_h'], move):.4f}")
    j_fus = frame_jump_m(R['fus_h'])
    j_bm = frame_jump_m(R['bm_h'])
    print(f"[帧间抖动 最大/RMS, m] 融合={j_fus[0]:.4f}/{j_fus[1]:.4f}  单BMP280={j_bm[0]:.4f}/{j_bm[1]:.4f}")
    print(f"[平均权重] w_MS={np.mean(R['wm_n']):.3f}  w_BM={1-np.mean(R['wm_n']):.3f}")

    # 图 a: 气压域（单坐标系）
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(t_axis, R['true'] / 100.0, 'k-', lw=1.0, alpha=0.6, label='真值')
    ax.plot(t_axis, R['ms_ekf_pa'] / 100.0, 'tab:orange', lw=0.8, alpha=0.7, label='MS5611 EKF')
    ax.plot(t_axis, (R['bm_ekf_pa'] - bias_tr) / 100.0, 'tab:green', lw=0.8, alpha=0.7,
            label='BMP280 EKF (已对齐)')
    ax.plot(t_axis, R['p_fus'] / 100.0, 'b-', lw=1.2, label='融合 FUSED')
    ax.set_ylabel('气压 (hPa)'); ax.set_xlabel('时间 (s)')
    ax.set_title('方案21 正常工况 — 气压域：双 EKF(已对齐) 与融合')
    ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.3)
    save_single(fig, 'sim_scheme21_a_pressure.png')

    # 图 b: 高度域（单坐标系）
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(t_axis, R['true_h'], 'k-', lw=1.0, alpha=0.6, label='真值高度')
    ax.plot(t_axis, R['ms_h'], 'tab:orange', lw=0.8, alpha=0.7, label='单 MS5611')
    ax.plot(t_axis, R['bm_h'], 'tab:green', lw=0.8, alpha=0.7, label='单 BMP280')
    ax.plot(t_axis, R['fus_h'], 'b-', lw=1.2, label='融合 FUSED')
    ax.set_ylabel('高度 (m)'); ax.set_xlabel('时间 (s)')
    ax.set_title('方案21 正常工况 — 高度域：单传感器与融合对照')
    ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.3)
    save_single(fig, 'sim_scheme21_b_height.png')

    # 图 c: 权重 w_MS5611（单坐标系）
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(t_axis, R['wm_n'], 'tab:orange', lw=1.0, label='w_MS5611')
    ax.set_ylabel('权重 w_MS5611'); ax.set_xlabel('时间 (s)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('方案21 正常工况 — 融合权重 w_MS5611（方差倒数加权·平滑+限幅）')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save_single(fig, 'sim_scheme21_c_wms.png')

    # 图 d: 权重 w_BMP280（单坐标系）
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(t_axis, 1 - R['wm_n'], 'tab:green', lw=1.0, label='w_BMP280')
    ax.set_ylabel('权重 w_BMP280'); ax.set_xlabel('时间 (s)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('方案21 正常工况 — 融合权重 w_BMP280 (=1-w_MS5611)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save_single(fig, 'sim_scheme21_d_wbm.png')

    # ===== 场景2：BMP280 中途噪声爆发失效，验证融合自动降权容错 =====
    F = run_pipeline(N, P0, ms_noise=20.0, bm_noise=4.0, seed_ms=11, seed_bm=22,
                     bm_fail=(int(0.6 * N), 120.0))
    fp_fus, fw_n, _ = fuse_new(F['ms_ekf_pa'], F['bm_ekf_pa'], F['ms_meas'], F['bm_meas'])
    f_fus_h = p2a_with_temp(fp_fus, P0, F['t_fus'])
    F['p_fus'] = fp_fus; F['wm_n'] = fw_n; F['fus_h'] = f_fus_h
    fail = slice(int(0.6 * N), N)
    fail_t = 0.6 * N / FS
    print("\n========== 方案21 融合仿真（BMP280 噪声爆发失效）==========")
    print("[失效段 高度 RMSE, m]")
    print(f"  单 MS5611 EKF : {rmse(F['ms_h'], F['true_h'], fail):.4f}")
    print(f"  单 BMP280 EKF : {rmse(F['bm_h'], F['true_h'], fail):.4f}  (已失效)")
    print(f"  融合 FUSED    : {rmse(F['fus_h'], F['true_h'], fail):.4f}  (自动降权后跟随 MS5611)")
    print(f"[失效段平均权重] w_MS={np.mean(F['wm_n'][fail]):.3f}  w_BM={1-np.mean(F['wm_n'][fail]):.3f}")

    # 图 a: 高度域 + 失效起点（单坐标系）
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(t_axis, F['true_h'], 'k-', lw=1.0, alpha=0.6, label='真值高度')
    ax.plot(t_axis, F['ms_h'], 'tab:orange', lw=0.8, alpha=0.7, label='单 MS5611')
    ax.plot(t_axis, F['bm_h'], 'tab:green', lw=0.8, alpha=0.5, label='单 BMP280 (噪声爆发失效)')
    ax.plot(t_axis, F['fus_h'], 'b-', lw=1.4, label='融合 FUSED (容错)')
    ax.axvline(fail_t, color='r', ls='--', lw=1.0, alpha=0.7, label='失效起点')
    ax.set_ylabel('高度 (m)'); ax.set_xlabel('时间 (s)')
    ax.set_title('方案21 失效容错 — 高度域：BMP280 爆发失效时融合自动降权跟随 MS5611')
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
    save_single(fig, 'sim_scheme21_failover_a_height.png')

    # 图 b: 权重 + 失效起点（单坐标系）
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(t_axis, F['wm_n'], 'tab:orange', lw=1.0, label='w_MS5611')
    ax.plot(t_axis, 1 - F['wm_n'], 'tab:green', lw=1.0, label='w_BMP280')
    ax.axvline(fail_t, color='r', ls='--', lw=1.0, alpha=0.7, label='失效起点')
    ax.set_ylabel('权重'); ax.set_xlabel('时间 (s)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('方案21 失效容错 — 融合权重：失效后 w_BMP280 趋零(自动降权)')
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
    save_single(fig, 'sim_scheme21_failover_b_weight.png')

    # ===== 场景3：大差距对照 —— 旧法(波动+偏置) vs 新法(稳定+对齐) =====
    # BMP280 真实系统性偏差 600 Pa（远大于固件固定 100 Pa）→ 模拟“差距本来就大”
    GAP = 600.0
    G = run_pipeline(N, P0, ms_noise=18.0, bm_noise=18.0, seed_ms=7, seed_bm=8, bm_offset=GAP)
    g_new, gw, g_bias = fuse_new(G['ms_ekf_pa'], G['bm_ekf_pa'], G['ms_meas'], G['bm_meas'])
    g_old, g_ow = fuse_old(G['ms_ekf_pa'], G['bm_ekf_pa'], G['ms_meas'], G['bm_meas'])
    g_new_h = p2a_with_temp(g_new, P0, G['t_fus'])
    g_old_h = p2a_with_temp(g_old, P0, G['t_fus'])

    print("\n========== 方案21 大差距场景对照（BMP280 真实偏差 600 Pa）==========")
    print(f"[在线偏置估计最终值] bias = {g_bias[-1]:.1f} Pa  (真值应≈{GAP:.0f})")
    hbias_new = np.mean(g_new_h[stat] - G['true_h'][stat])
    hbias_old = np.mean(g_old_h[stat] - G['true_h'][stat])
    print(f"[静止段高度偏置, m] 新法={hbias_new:+.4f}  旧法={hbias_old:+.4f}")
    jn = frame_jump_m(g_new_h); jo = frame_jump_m(g_old_h)
    print(f"[帧间抖动 最大/RMS, m] 新法={jn[0]:.4f}/{jn[1]:.4f}  旧法={jo[0]:.4f}/{jo[1]:.4f}")
    print(f"[平均权重 新法] w_MS={np.mean(gw):.3f}  w_BM={1-np.mean(gw):.3f}")

    # 图 a: 气压域 新旧对比（单坐标系）
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(t_axis, (G['ms_ekf_pa']) / 100.0, 'tab:orange', lw=0.9, alpha=0.85, label='MS5611 EKF')
    ax.plot(t_axis, (G['bm_ekf_pa'] - GAP) / 100.0, 'tab:green', lw=0.9, alpha=0.85,
            label='BMP280 EKF (真实对齐)')
    ax.plot(t_axis, g_new / 100.0, 'b-', lw=1.4, label='新法 融合(在线偏置)')
    ax.plot(t_axis, g_old / 100.0, 'r--', lw=1.0, alpha=0.85, label='旧法 融合(固定offset)')
    ax.set_ylabel('气压 (hPa)'); ax.set_xlabel('时间 (s)')
    ax.set_title('方案21 大差距(600 Pa) — 气压域：新法(对齐) vs 旧法(固定offset)')
    ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.3)
    save_single(fig, 'sim_scheme21_gapfix_a_pressure.png')

    # 图 b: 高度域 新旧对比（单坐标系）
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(t_axis, G['true_h'], 'k-', lw=1.0, alpha=0.6, label='真值高度')
    ax.plot(t_axis, g_new_h, 'b-', lw=1.4, label='新法 融合(稳定+对齐)')
    ax.plot(t_axis, g_old_h, 'r--', lw=1.0, alpha=0.85, label='旧法 融合(波动+偏置)')
    ax.set_ylabel('高度 (m)'); ax.set_xlabel('时间 (s)')
    ax.set_title('方案21 大差距(600 Pa) — 高度域：蓝=新法(稳定+对齐) 红=旧法(波动+偏置)')
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
    save_single(fig, 'sim_scheme21_gapfix_b_height.png')

    # 图 c: 权重 新旧对比（单坐标系）
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(t_axis, gw, 'b-', lw=1.0, label='新法 w_MS5611')
    ax.plot(t_axis, 1 - gw, 'tab:cyan', lw=1.0, label='新法 w_BMP280')
    ax.plot(t_axis, g_ow, 'r--', lw=0.9, alpha=0.8, label='旧法 w_MS5611(抖动)')
    ax.set_ylabel('权重'); ax.set_xlabel('时间 (s)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('方案21 大差距(600 Pa) — 融合权重：新法平滑 vs 旧法抖动')
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
    save_single(fig, 'sim_scheme21_gapfix_c_weight.png')

    print("\n全部 9 张单坐标系图片已生成。")


if __name__ == '__main__':
    main()
