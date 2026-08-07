# -*- coding: utf-8 -*-
"""
方案22 离线仿真：在方案21（双 EKF 方差倒数加权融合 + 固定输出低通）基础上，
解决“相对高度测试时要等约 1 分钟才稳定”的问题，同时保持静止精度。

问题根因
--------
方案21 输出级固定低通 S21_OUTPUT_EMA = 0.001 @50Hz → 时间常数 τ = dt/α ≈ 20s。
相对高度发生阶跃/斜坡变化后，输出按指数逼近真值，需 3~4τ ≈ 60~80s 才进入 ±5cm，
给人“等1分钟才稳定”的感觉。

方案22 改进：把“固定输出低通”换成“运动自适应输出低通（快慢双低通偏差法）”
  · 同时维护快低通 p_fast(α_fast=0.02, τ≈1s) 与慢低通 p_slow(α_slow=0.001, τ≈20s)；
  · 运动指标 m = |p_fast − p_slow|（两者都已平滑，不含高频噪声；
    运动时快低通跟得上、慢低通滞后，m 达几十 Pa；静止时两者都收敛到真值，m→0）；
  · norm = clip(m/THR,0,1)^2（平方抑制静止微小残差）；
  · 输出 out = norm·p_fast + (1−norm)·p_slow：
      静止 → norm≈0 → 完全用最平滑的 p_slow（静止精度与方案21一致）；
      运动 → norm≈1 → 切到 p_fast（τ≈1s，几秒到位）。
  这样“等待约1分钟”→几秒，且静止噪声不变（精度保持）。

仿真场景 = 相对高度实测：静止(0~20s) → 抬升1m(20~25s) → 静止(25~60s)
          → 降下1m(60~65s) → 静止(65s~结束)。
量化：① 稳定时间（变化结束到读数进入 ±5cm 的等待时长，核心痛点）；
      ② 静止噪声 std；③ 运动段 RMSE。
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
S20_EKF_R_MEAS  = 1.0
S20_BM_P0_OFFSET = 100.0

# 方案21 融合参数（与 fusion_scheme_21_params.h 一致）
S21_FUSE_METHOD = 0
S21_FIXED_W_MS  = 0.5
S21_BIAS_CONVERGE_FRAMES = 500
S21_VAR_EMA     = 0.02
S21_VAR_EPS     = 1e-3
S21_W_EMA       = 0.02
S21_W_MIN       = 0.10
S21_W_MAX       = 0.90
S21_FUSE_TEMP_SRC = 0

# 方案21 输出低通（复现“等1分钟”）：固件 S21_OUTPUT_EMA = 0.001 @50Hz → τ≈20s
S21_OUTPUT_EMA_SLOW = 0.001

# 方案22 自适应输出低通参数
S22_A_FAST = 0.02      # 运动档快低通：τ≈1s
S22_A_SLOW = 0.001     # 静止档慢低通：τ≈20s（=方案21，保证精度一致）
S22_THR    = 3.0       # 运动指标阈值 (Pa)：|p_fast−p_slow| 超过即判运动

DT = 0.02              # 50 Hz
FS = 1.0 / DT


# ============== 气压域 EKF（与 baro_ekf.c 一致，额外保存 dp） ==============
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

    def get_rate(self):
        return self.dp

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


# ============== 方案21 融合核心：在线偏置 + 残差方差倒数权重（与固件一致） ==============
def fuse_core(ms_ekf_pa, bm_ekf_pa, ms_raw_pa, bm_raw_pa):
    n = len(ms_ekf_pa)
    p_fus = np.zeros(n)
    w_ms = np.zeros(n)
    bias_tr = np.zeros(n)
    var_ms = 0.0
    var_bm = 0.0
    bias = 0.0
    w = 0.5
    inited = False
    cnt = 0
    for i in range(n):
        if not inited:
            bias = bm_ekf_pa[i] - ms_ekf_pa[i]
            cnt = 0
        elif cnt < S21_BIAS_CONVERGE_FRAMES:
            bias = 0.01 * (bm_ekf_pa[i] - ms_ekf_pa[i]) + 0.99 * bias
            cnt += 1
        p_bm_aligned = bm_ekf_pa[i] - bias
        res_ms = ms_raw_pa[i] - ms_ekf_pa[i]
        res_bm = bm_raw_pa[i] - bm_ekf_pa[i]
        if not inited:
            var_ms = res_ms * res_ms
            var_bm = res_bm * res_bm
            inited = True
        else:
            var_ms = S21_VAR_EMA * (res_ms * res_ms) + (1.0 - S21_VAR_EMA) * var_ms
            var_bm = S21_VAR_EMA * (res_bm * res_bm) + (1.0 - S21_VAR_EMA) * var_bm
        if S21_FUSE_METHOD == 0:
            wm_raw = 1.0 / (var_ms + S21_VAR_EPS)
            wb_raw = 1.0 / (var_bm + S21_VAR_EPS)
            w_inst = wm_raw / (wm_raw + wb_raw)
            w = S21_W_EMA * w_inst + (1.0 - S21_W_EMA) * w
        else:
            w = S21_FIXED_W_MS
        wc = w if S21_W_MIN <= w <= S21_W_MAX else (S21_W_MIN if w < S21_W_MIN else S21_W_MAX)
        p_fus[i] = wc * ms_ekf_pa[i] + (1.0 - wc) * p_bm_aligned
        w_ms[i] = wc
        bias_tr[i] = bias
    return p_fus, w_ms, bias_tr


# ============== 输出低通 ==============
def lp_fixed(p_fus, alpha):
    """固定系数一阶低通（方案21 输出级）。"""
    n = len(p_fus)
    out = np.zeros(n)
    out[0] = p_fus[0]
    for i in range(1, n):
        out[i] = alpha * p_fus[i] + (1.0 - alpha) * out[i - 1]
    return out


def lp_adaptive(p_fus, a_fast, a_slow, thr):
    """方案22：快慢双低通偏差法自适应输出低通。
    返回 (输出, 运动程度 norm, p_fast, p_slow)。"""
    n = len(p_fus)
    p_fast = np.zeros(n)
    p_slow = np.zeros(n)
    out = np.zeros(n)
    norm = np.zeros(n)
    p_fast[0] = p_fus[0]
    p_slow[0] = p_fus[0]
    out[0] = p_fus[0]
    for i in range(1, n):
        p_fast[i] = a_fast * p_fus[i] + (1.0 - a_fast) * p_fast[i - 1]
        p_slow[i] = a_slow * p_fus[i] + (1.0 - a_slow) * p_slow[i - 1]
        m = abs(p_fast[i] - p_slow[i])               # 运动指标（已平滑，无高频噪声）
        nv = min(max(m / thr, 0.0), 1.0)
        nv = nv * nv                                # 平方：静止残差→≈0
        norm[i] = nv
        out[i] = nv * p_fast[i] + (1.0 - nv) * p_slow[i]
    return out, norm, p_fast, p_slow


# ============== 相对高度测试合成数据 ==============
def make_relheight_trajectory(N, P0, noise_ms, noise_bm, offset_bm, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(N) * DT
    hprof = np.zeros(N)
    for i in range(N):
        ti = t[i]
        if ti < 30.0:
            hprof[i] = 0.0                              # 0~30s 热身（偏置冻结/EKF稳态）
        elif ti < 35.0:
            hprof[i] = (ti - 30.0) / 5.0 * 1.0          # 30~35s 抬升到 1m
        elif ti < 70.0:
            hprof[i] = 1.0
        elif ti < 75.0:
            hprof[i] = 1.0 - (ti - 70.0) / 5.0 * 1.0    # 70~75s 降下到 0
        else:
            hprof[i] = 0.0
    true = P0 - hprof * 12.0                            # Δp ≈ -12 Pa/m
    ms_meas = true + rng.normal(0.0, noise_ms, N)
    bm_meas = true + offset_bm + rng.normal(0.0, noise_bm, N)
    temp = 25.0 + 0.5 * np.sin(2 * np.pi * t / 120.0)
    return ms_meas, bm_meas, true, temp


def rmse(x, ref, mask=None):
    d = np.asarray(x) - np.asarray(ref)
    if mask is not None:
        d = d[mask]
    return float(np.sqrt(np.mean(d * d)))


def settle_time(h, h_true, idx_end, tol=0.05, win=2.0):
    n = len(h)
    win_n = int(win * FS)
    if win_n < 1:
        win_n = 1
    for i in range(idx_end, n - win_n):
        seg = np.abs(h[i:i + win_n] - h_true[i:i + win_n])
        if np.max(seg) < tol:
            return (i - idx_end) / FS
    return (n - idx_end) / FS


def main():
    N = int(130 * FS)
    P0 = 101325.0
    out_dir = os.path.dirname(os.path.abspath(__file__))
    t_axis = np.arange(N) / FS

    ms_meas, bm_meas, true, temp = make_relheight_trajectory(
        N, P0, noise_ms=20.0, noise_bm=4.0, offset_bm=S20_BM_P0_OFFSET, seed=42)

    ekf_ms = BaroEKF(P0, S20_EKF_Q_PRESS, S20_EKF_Q_RATE, S20_EKF_R_MEAS, DT)
    ekf_bm = BaroEKF(P0, S20_EKF_Q_PRESS, S20_EKF_Q_RATE, S20_EKF_R_MEAS, DT)
    ms_ekf_pa = np.zeros(N); bm_ekf_pa = np.zeros(N)
    for i in range(N):
        ms_ekf_pa[i] = ekf_ms.update(ms_meas[i], temp[i])
        bm_ekf_pa[i] = ekf_bm.update(bm_meas[i], temp[i])

    t_fus = temp if S21_FUSE_TEMP_SRC == 0 else 0.5 * (temp + temp)
    p_fus, w_ms, bias_tr = fuse_core(ms_ekf_pa, bm_ekf_pa, ms_meas, bm_meas)

    # ---- 方案21：固定慢档低通 ----
    p21 = lp_fixed(p_fus, S21_OUTPUT_EMA_SLOW)
    h21 = p2a_with_temp(p21, P0, t_fus)
    # ---- 方案22：运动自适应低通（快慢双低通偏差法）----
    p22, norm22, p_fast22, p_slow22 = lp_adaptive(p_fus, S22_A_FAST, S22_A_SLOW, S22_THR)
    h22 = p2a_with_temp(p22, P0, t_fus)

    true_h = p2a_with_temp(true, P0, t_fus)

    # ---- 量化 ----
    stat = slice(int(20 * FS), int(28 * FS))     # 热身后稳态静止段
    move = slice(int(30 * FS), int(40 * FS))
    down = slice(int(70 * FS), int(80 * FS))
    idx_up_end = int(35 * FS)
    idx_down_end = int(75 * FS)

    std21 = np.std((h21 - true_h)[stat])
    std22 = np.std((h22 - true_h)[stat])
    r21 = rmse(h21, true_h, move)
    r22 = rmse(h22, true_h, move)
    r21d = rmse(h21, true_h, down)
    r22d = rmse(h22, true_h, down)
    st21_up = settle_time(h21, true_h, idx_up_end)
    st22_up = settle_time(h22, true_h, idx_up_end)
    st21_dn = settle_time(h21, true_h, idx_down_end)
    st22_dn = settle_time(h22, true_h, idx_down_end)

    # 运动指标诊断（确认 THR 合理）
    m_stat = np.std(np.abs(p_fast22[stat] - p_slow22[stat]))
    m_mv = np.mean(np.abs(p_fast22[move] - p_slow22[move]))

    print("========== 方案21 vs 方案22 融合仿真（相对高度测试）==========")
    print(f"[静止噪声 std, m]  方案21={std21:.4f}   方案22={std22:.4f}   (精度应一致)")
    print(f"[抬升段 RMSE, m]   方案21={r21:.4f}   方案22={r22:.4f}")
    print(f"[降下段 RMSE, m]   方案21={r21d:.4f}   方案22={r22d:.4f}")
    print(f"[稳定时间(抬升结束起, s)]  方案21={st21_up:.1f}   方案22={st22_up:.1f}")
    print(f"[稳定时间(降下结束起, s)]  方案21={st21_dn:.1f}   方案22={st22_dn:.1f}")
    print(f"[运动指标 |p_fast−p_slow|] 静止 RMS≈{m_stat:.3f}Pa   运动均值≈{m_mv:.2f}Pa   (THR={S22_THR}Pa)")

    # ---- 图 ----
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(t_axis, true_h, 'k-', lw=1.2, alpha=0.7, label='真值高度')
    axes[0].plot(t_axis, h21, 'tab:red', lw=1.0, alpha=0.85, label='方案21 固定低通(τ≈20s)')
    axes[0].plot(t_axis, h22, 'tab:blue', lw=1.2, label='方案22 自适应低通')
    for xv in (20, 25, 60, 65):
        axes[0].axvline(xv, color='gray', ls='--', lw=0.8, alpha=0.5)
    axes[0].set_ylabel('高度 (m)')
    axes[0].set_title('方案22：相对高度测试 —— 稳定速度对比（红=方案21 等~1分钟，蓝=方案22 秒级）')
    axes[0].legend(fontsize=8, ncol=3); axes[0].grid(alpha=0.3)

    z0, z1 = int(34 * FS), int(50 * FS)
    axes[1].plot(t_axis[z0:z1], true_h[z0:z1], 'k-', lw=1.2, alpha=0.7, label='真值高度')
    axes[1].plot(t_axis[z0:z1], h21[z0:z1], 'tab:red', lw=1.0, alpha=0.85, label='方案21')
    axes[1].plot(t_axis[z0:z1], h22[z0:z1], 'tab:blue', lw=1.2, label='方案22')
    axes[1].fill_between(t_axis[z0:z1], true_h[z0:z1]-0.05, true_h[z0:z1]+0.05,
                         color='green', alpha=0.12, label='±5cm 稳定带')
    axes[1].axvline(25, color='gray', ls='--', lw=0.8, alpha=0.5)
    axes[1].set_ylabel('高度 (m)')
    axes[1].set_title('局部放大：抬升结束(25s)后，方案22 几秒进入±5cm，方案21 需~60s')
    axes[1].legend(fontsize=8, ncol=4); axes[1].grid(alpha=0.3)

    axes[2].plot(t_axis, norm22, 'tab:purple', lw=1.0, label='方案22 运动程度 norm')
    for xv in (20, 25, 60, 65):
        axes[2].axvline(xv, color='gray', ls='--', lw=0.8, alpha=0.5)
    axes[2].set_ylabel('运动程度 norm (0静/1动)'); axes[2].set_xlabel('时间 (s)')
    axes[2].set_title('方案22 自适应：运动(抬升/降下)→norm≈1 用快低通，静止→norm≈0 用慢低通')
    axes[2].set_ylim(-0.05, 1.1); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    f1 = os.path.join(out_dir, 'sim_scheme22.png')
    plt.savefig(f1, dpi=150); plt.close()
    print(f"[完成] 已保存: {f1}")


if __name__ == '__main__':
    main()
