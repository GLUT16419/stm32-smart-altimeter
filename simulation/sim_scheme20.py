# -*- coding: utf-8 -*-
"""
方案20 离线仿真（对照 sahixi 气压域 EKF vs BP 去噪）
================================================
忠实复现固件 (FUSION_SCHEME=20) 的算法链路，用于验证移植的
baro_ekf.c / bp_denoise.c 行为正确、权重未被破坏：

  MS5611_raw(hPa) ─┬─> 气压域 EKF  ─> ms_ekf_pa ─┐
                   └─> BP 5点去噪  ─> ms_bp_pa  ─┤
  BMP280_raw(hPa) ─┬─> 气压域 EKF  ─> bm_ekf_pa  ─┤
                   └─> BP 5点去噪  ─> bm_bp_pa  ─┤
                                                 │
  各路气压 ──PressureToAltitudeWithTemp──> 对照高度（温度补偿 ISA）

本脚本不依赖真实 CSV，用合成轨迹（静止→上升→下降 + 高斯噪声）演示
两套算法各自的平滑/去噪效果，并量化相对原始噪声的抑制比。

输出: sim_scheme20.png
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

# ============== 方案20 参数（与 fusion_scheme_20_params.h 一致） ==============
# 下述 EKF 三参数经 sim_scheme20_tune.py 离线网格扫描（建模 BMP280 驱动预滤波后）
# 得到的均衡最优：同时降低静止噪声与运动RMS。BP_EMA_ALPHA 保持 0.1 以维持静止精度，
# 如需更强动态跟手性可升到 0.2~0.3（代价是静止噪声上升，见调参脚本）。
S20_EKF_Q_PRESS = 1.0
S20_EKF_Q_RATE  = 5.0
S20_EKF_R_MEAS  = 0.2
S20_BP_EMA_ALPHA = 0.1
S20_BP_WIN_SCALE = 5.0
S20_BM_P0_OFFSET = 100.0                # Pa

DT = 0.02                               # 50 Hz，与固件 SensorTask 一致
FS = 1.0 / DT

# ============== 预训练权重（与 bp_denoise.c 完全一致） ==============
BP_MS_W0 = [0.12129213,-0.43837869,-0.66982877,0.073925033,-0.39800963,
            0.46972299,0.53867275,-0.53806669,-0.12204781,0.23553427,
            0.24751934,-0.79806453,-0.62424195,-0.07045754,-0.28578758,
            -0.1159967,0.25445336,0.39495337,0.45477626,-0.042697523,
            0.77840376,-0.42664009,0.58110178,-0.60451555,0.11663397,
            0.80442661,0.35490465,-0.12372786,0.28214756,0.49488783,
            -0.035429798,0.49410579,-0.4199802,-0.51784551,-0.70666122,
            -0.45513871,0.47054282,0.57356203,-0.030514309,-0.37879375]
BP_MS_B0 = [0.0002038508,0.00026090239,-0.0002271471,-0.00012285044,
            -0.00024494648,-0.0002271266,-0.00027924759,-0.00021629906]
BP_MS_W2 = [0.09858083,0.4061701,-0.19927566,-0.036631376,
            -0.16093147,-0.026557013,-0.31033272,-0.17409751]
BP_MS_B2 = [0.00033927892]

BP_BM_W0 = [-0.51064873,0.15269634,0.094315231,0.27824983,0.35737059,
            -0.64638454,-0.21621756,0.023822606,-0.6067344,-0.65262496,
            -0.53331786,0.10125236,-0.61716789,-0.086650252,-0.64003795,
            -0.51823986,-0.61809033,-0.21011594,0.48388678,0.069006279,
            -0.50922316,-0.20666251,0.11087561,0.28679425,-0.5574069,
            0.62397134,-0.54479349,-0.66166675,-0.27504212,0.4539983,
            0.29315642,0.21384901,-0.14603049,-0.46011981,0.57155293,
            0.23690392,-0.37960237,0.23111033,-0.46760777,-0.060113251]
BP_BM_B0 = [-0.002042291,0.029359214,0.0035425713,-0.013603366,
            -0.00012134975,0.0055174888,0.0046751029,0.0080234539]
BP_BM_W2 = [-0.28211832,-0.53251672,-0.39900565,0.11449191,
            0.58395427,0.11776479,0.51348466,0.25250733]
BP_BM_B2 = [0.014060951]


# ============== BP 前向（与 bp_denoise.c 一致） ==============
def bp_forward(x, w0, b0, w2, b2):
    h = [0.0] * 8
    for o in range(8):
        acc = b0[o]
        for i in range(5):
            acc += w0[o * 5 + i] * x[i]
        h[o] = np.tanh(acc)
    y = b2[0]
    for o in range(8):
        y += w2[o] * h[o]
    return y


def bp_denoise(raw_hpa_seq, w0, b0, w2, b2, scale=S20_BP_WIN_SCALE, alpha=S20_BP_EMA_ALPHA):
    """复用 5 点滑窗 + 归一化 + EMA 逻辑，返回与固件逐帧一致的输出序列。"""
    hist = [0.0] * 5
    cnt = 0
    inited = False
    ema = 0.0
    out = []
    for raw in raw_hpa_seq:
        for j in range(4):
            hist[j] = hist[j + 1]
        hist[4] = raw
        if cnt < 5:
            cnt += 1
            out.append(raw)              # 滑窗未满：回退原始（固件用 raw_pa 兜底）
            continue
        center = hist[2]
        bi = [(hist[j] - center) / scale for j in range(5)]
        bo = bp_forward(bi, w0, b0, w2, b2)
        denoised = bo * scale + center
        if not inited:
            ema = denoised
            inited = True
        else:
            ema = alpha * denoised + (1.0 - alpha) * ema
        out.append(ema)
    return np.array(out, dtype=float)


# ============== 气压域 EKF（与 baro_ekf.c 一致，内部用简单 ISA） ==============
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


# ============== 合成数据 ==============
def make_trajectory(n_steps, p0_pa, noise_std_pa, offset_pa=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * DT
    T = n_steps * DT
    base = p0_pa + offset_pa
    # 真实气压：前 40% 完全静止（用于纯净噪声度量），之后缓慢上升再下降
    ramp = np.where(t < 0.4 * T, 0.0,
                    300.0 * np.sin(2 * np.pi * (t - 0.4 * T) / (0.6 * T)) * 0.5)
    true = base + ramp
    # 传感器噪声（MS5611 更吵，BMP280 更干净）
    meas = true + rng.normal(0.0, noise_std_pa, n_steps)
    # 温度 ~ 25°C 缓慢漂移
    temp = 25.0 + 1.5 * np.sin(2 * np.pi * t / T * 0.5)
    return meas, true, temp


# ============== 主流程 ==============
# ============== Sahixi 驱动预滤波（与 Core/Src/bsp_bmp280.c 一致） ==============
# 固件 BMP280_Read_Data 已对气压套用此 5点滑动平均 + 突变拒绝后才送 EKF/BP；
# MS5611 无此层。为使仿真信号链与固件一致，仅 BMP280 通道套用本函数。
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


def main():
    N = 2500                                   # 50s @ 50Hz
    P0 = 101325.0

    # 噪声量级参照方案18 真实 KF 方差（MS5611 较吵 ~20Pa，BMP280 干净 ~4Pa）
    ms_meas, ms_true, ms_temp = make_trajectory(N, P0, noise_std_pa=20.0, offset_pa=0.0, seed=11)
    bm_meas, bm_true, bm_temp = make_trajectory(N, P0, noise_std_pa=4.0, offset_pa=S20_BM_P0_OFFSET, seed=22)

    ms_hpa = ms_meas / 100.0
    bm_hpa = bm_meas / 100.0

    # BMP280 驱动预滤波（仅 BMP280，与固件一致）；MS5611 保持原始
    bm_pre = sahixi_sw_filter(bm_meas)
    bm_pre_hpa = bm_pre / 100.0

    # ---- 气压域 EKF ----（EKF/BP 均吃预滤波后的 BMP280 气压）
    ekf_ms = BaroEKF(P0, S20_EKF_Q_PRESS, S20_EKF_Q_RATE, S20_EKF_R_MEAS, DT)
    ekf_bm = BaroEKF(P0 + S20_BM_P0_OFFSET, S20_EKF_Q_PRESS, S20_EKF_Q_RATE, S20_EKF_R_MEAS, DT)
    ms_ekf_pa = np.array([ekf_ms.update(ms_meas[i], ms_temp[i]) for i in range(N)])
    bm_ekf_pa = np.array([ekf_bm.update(bm_pre[i], bm_temp[i]) for i in range(N)])

    # ---- BP 去噪 ----
    ms_bp_hpa = bp_denoise(ms_hpa, BP_MS_W0, BP_MS_B0, BP_MS_W2, BP_MS_B2)
    bm_bp_hpa = bp_denoise(bm_pre_hpa, BP_BM_W0, BP_BM_B0, BP_BM_W2, BP_BM_B2)
    ms_bp_pa = ms_bp_hpa * 100.0
    bm_bp_pa = bm_bp_hpa * 100.0

    # ---- 高度换算 ----
    ms_ekf_h = p2a_with_temp(ms_ekf_pa, P0, ms_temp)
    bm_ekf_h = p2a_with_temp(bm_ekf_pa, P0 + S20_BM_P0_OFFSET, bm_temp)
    ms_bp_h  = p2a_with_temp(ms_bp_pa, P0, ms_temp)
    bm_bp_h  = p2a_with_temp(bm_bp_pa, P0 + S20_BM_P0_OFFSET, bm_temp)

    # ---- 指标：相对真实轨迹的噪声抑制（静止段，取前 30%） ----
    nstat = max(20, int(0.30 * N))
    def stat_noise(x, true):
        return float(np.std((x - true)[:nstat]))
    print("方案20 离线仿真结果（静止段噪声 std, Pa）:")
    print(f"  MS5611 原始     : {stat_noise(ms_meas, ms_true):.3f}")
    print(f"  MS5611 EKF      : {stat_noise(ms_ekf_pa, ms_true):.3f}")
    print(f"  MS5611 BP       : {stat_noise(ms_bp_pa, ms_true):.3f}")
    print(f"  BMP280 原始     : {stat_noise(bm_meas, bm_true):.3f}")
    print(f"  BMP280 EKF      : {stat_noise(bm_ekf_pa, bm_true):.3f}")
    print(f"  BMP280 BP       : {stat_noise(bm_bp_pa, bm_true):.3f}")

    # ---- 绘图 ----
    out_dir = os.path.dirname(os.path.abspath(__file__))
    t_axis = np.arange(N) / FS
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

    axes[0].plot(t_axis, ms_true / 100.0, 'k-', lw=1.0, label='MS5611 真值', alpha=0.6)
    axes[0].plot(t_axis, ms_hpa, 'gray', lw=0.4, alpha=0.35, label='MS5611 原始')
    axes[0].plot(t_axis, ms_ekf_pa / 100.0, 'b-', lw=1.0, label='MS5611 EKF')
    axes[0].plot(t_axis, ms_bp_hpa, 'r-', lw=1.0, label='MS5611 BP')
    axes[0].set_ylabel('气压 (hPa)'); axes[0].set_title('方案20 仿真 — MS5611 (EKF vs BP 去噪)')
    axes[0].legend(fontsize=8, ncol=4); axes[0].grid(alpha=0.3)

    axes[1].plot(t_axis, ms_ekf_h, 'b-', lw=1.0, label='MS5611 EKF 高度')
    axes[1].plot(t_axis, ms_bp_h, 'r-', lw=1.0, label='MS5611 BP 高度')
    axes[1].set_ylabel('高度 (m)'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    axes[2].plot(t_axis, bm_true / 100.0, 'k-', lw=1.0, label='BMP280 真值', alpha=0.6)
    axes[2].plot(t_axis, bm_hpa, 'gray', lw=0.4, alpha=0.35, label='BMP280 原始')
    axes[2].plot(t_axis, bm_ekf_pa / 100.0, 'b-', lw=1.0, label='BMP280 EKF')
    axes[2].plot(t_axis, bm_bp_hpa, 'r-', lw=1.0, label='BMP280 BP')
    axes[2].set_ylabel('气压 (hPa)'); axes[2].set_title('方案20 仿真 — BMP280 (EKF vs BP 去噪, 含 +1hPa 偏移)')
    axes[2].legend(fontsize=8, ncol=4); axes[2].grid(alpha=0.3)

    axes[3].plot(t_axis, bm_ekf_h, 'b-', lw=1.0, label='BMP280 EKF 高度')
    axes[3].plot(t_axis, bm_bp_h, 'r-', lw=1.0, label='BMP280 BP 高度')
    axes[3].set_ylabel('高度 (m)'); axes[3].set_xlabel('时间 (s)')
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, 'sim_scheme20.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"\n[完成] 已保存对照图: {fname}")


if __name__ == '__main__':
    main()
