# -*- coding: utf-8 -*-
"""重新仿真 2.9 章 7 张结果图（每个实验单坐标系、图内无图号字样）。
复用 pressure_altitude_simulation.py 的滤波器与数据逻辑，
输出到 sim_ch29_1.png ... sim_ch29_7.png（统一 dpi=150，单图比例）。
"""
import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# ================= 常数（与 altitude_convert.h 一致）=================
SEA_LEVEL_PRESSURE_PA = 101325.0
ISA_T0 = 288.15
ISA_L  = 0.0065
ISA_G  = 9.80665
ISA_R  = 287.05

# BMP280 卡尔曼参数（与 kalman_filter.h 一致）
BMP280_Q = 0.005
BMP280_R = 0.10
BMP280_RESIDUAL_TH = 1.2
BMP280_Q_INC = 1.08
BMP280_Q_DEC = 0.97
BMP280_Q_MAX = 5.0


# ================= 气压-高度转换 =================
def pressure_to_altitude_isa(p, p0=SEA_LEVEL_PRESSURE_PA):
    if p <= 0 or p0 <= 0:
        return 0.0
    ratio = p / p0
    if ratio <= 0:
        return 0.0
    return (ISA_T0 / ISA_L) * (1.0 - ratio ** (ISA_L * ISA_R / ISA_G))

def pressure_to_altitude_iso(p, p0=SEA_LEVEL_PRESSURE_PA):
    if p <= 0 or p0 <= 0:
        return 0.0
    ratio = p / p0
    if ratio <= 0:
        return 0.0
    return -(ISA_R * ISA_T0 / ISA_G) * np.log(ratio)

def pressure_to_altitude_with_temp(p, temp_c, p0=SEA_LEVEL_PRESSURE_PA):
    if p <= 0 or p0 <= 0:
        return 0.0
    ratio = p / p0
    if ratio <= 0 or ratio > 10:
        return 0.0
    h = (ISA_T0 / ISA_L) * (1.0 - ratio ** (ISA_L * ISA_R / ISA_G))
    tk = temp_c + 273.15
    delta_t = tk - (ISA_T0 - ISA_L * h)
    return h + delta_t * 0.035


# ================= 卡尔曼滤波 =================
class KalmanFilter:
    def __init__(self, x0, p0, q, r):
        self.x = x0; self.p = p0; self.q = q; self.r = r
        self.k = 0.0
    def update(self, z):
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (z - self.x)
        self.p = (1 - self.k) * self.p
        return self.x

class AdaptiveKalmanFilter(KalmanFilter):
    def __init__(self, x0, p0, q, r,
                 th=BMP280_RESIDUAL_TH, inc=BMP280_Q_INC, dec=BMP280_Q_DEC, qmax=BMP280_Q_MAX):
        super().__init__(x0, p0, q, r)
        self.th = th; self.inc = inc; self.dec = dec; self.qmax = qmax
        self.q_base = q
        self.win = [0.0] * 5
        self.idx = 0
    def update(self, z):
        res = abs(z - self.x)
        self.win[self.idx] = res
        self.idx = (self.idx + 1) % 5
        m = sum(self.win) / 5.0
        rstd = (sum((w - m) ** 2 for w in self.win) / 5.0) ** 0.5
        if rstd > self.th:
            self.q = min(self.q * self.inc, self.qmax)
        else:
            self.q = max(self.q * self.dec, self.q_base)
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (z - self.x)
        self.p = (1 - self.k) * self.p
        return self.x


def simulate_bmp280(n=500, fs=10.0, noise=0.35, seed=42):
    """运动真值：0-15s静止；15-25s缓升36Pa(≈3m)；25-35s静止；35-42s快降60Pa(≈5m)；42-50s静止"""
    np.random.seed(seed)
    dt = 1.0 / fs
    t = np.arange(n) * dt
    true = np.full(n, 101000.0)
    true[int(15*fs):int(25*fs)] += np.linspace(0, 36, int(25*fs)-int(15*fs))
    true[int(35*fs):int(42*fs)] += np.linspace(0, -60, int(42*fs)-int(35*fs))
    true[int(25*fs):int(35*fs)] = true[int(25*fs)]
    true[int(42*fs):] = true[int(42*fs)-1]
    meas = true + np.random.normal(0, noise, n)
    return t, true, meas


# ================= 通用样式 =================
DPI = 150
FIGSIZE = (9.5, 6.0)
GRID = dict(alpha=0.3)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(name, dpi=DPI)
    plt.close(fig)
    print('saved', name)

out = os.path.join(os.path.dirname(__file__))

# ------------------------------------------------------------------
# 图 1 / 2.9.1 静态降噪对比：气压-时间，原始 / 标准KF / 自适应KF
# ------------------------------------------------------------------
def fig_static():
    n = 300
    np.random.seed(42)
    true = np.full(n, 101000.0)
    meas = true + np.random.normal(0, 0.35, n)
    kf_std = KalmanFilter(101000.0, 0.1, 0.005, 0.10)
    kf_adp = AdaptiveKalmanFilter(101000.0, 0.1, 0.005, 0.10)
    o_std = [kf_std.update(z) for z in meas]
    o_adp = [kf_adp.update(z) for z in meas]
    raw_std = np.std(meas); s_std = np.std(o_std); a_std = np.std(o_adp)
    t = np.arange(n) / 10.0
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.plot(t, meas, 'b-', alpha=0.45, lw=0.8, label=f'原始测量 (σ={raw_std:.3f} Pa)')
    ax.plot(t, o_std, 'r-', lw=1.4, label=f'标准KF (σ={s_std:.3f} Pa)')
    ax.plot(t, o_adp, 'g-', lw=1.4, label=f'自适应KF (σ={a_std:.3f} Pa)')
    ax.axhline(101000.0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('气压 (Pa)')
    ax.set_title(f'静态降噪对比：标准KF与自适应KF抑制噪声约 {raw_std/a_std:.1f} 倍')
    ax.legend(fontsize=9); ax.grid(**GRID)
    save(fig, os.path.join(out, 'sim_ch29_1.png'))

# ------------------------------------------------------------------
# 图 2 / 2.9.2 运动场景跟踪：气压-时间，真值 / 小Q标准KF / 自适应KF
# ------------------------------------------------------------------
def fig_motion():
    t, true, meas = simulate_bmp280(500, 10.0)
    kf_small = KalmanFilter(101000.0, 0.1, 0.005, 0.10)
    kf_adp = AdaptiveKalmanFilter(101000.0, 0.1, 0.005, 0.10)
    o_small = [kf_small.update(z) for z in meas]
    o_adp = [kf_adp.update(z) for z in meas]
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.plot(t, true, 'k-', lw=1.6, label='真实气压')
    ax.plot(t, meas, 'b-', alpha=0.35, lw=0.6, label='原始测量')
    ax.plot(t, o_small, 'r-', lw=1.3, label='标准KF (Q=0.005, 平滑滞后)')
    ax.plot(t, o_adp, 'g-', lw=1.3, label='自适应KF (自动放大Q)')
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('气压 (Pa)')
    ax.set_title('运动场景下自适应KF兼顾平滑与跟踪')
    ax.legend(fontsize=9); ax.grid(**GRID)
    save(fig, os.path.join(out, 'sim_ch29_2.png'))

# ------------------------------------------------------------------
# 图 3 / 2.9.3 Q/R 参数扫描：气压-时间，5 组 Q/R 滤波曲线
# ------------------------------------------------------------------
def fig_qr():
    n = 300
    np.random.seed(42)
    true = np.full(n, 101000.0)
    meas = true + np.random.normal(0, 0.35, n)
    configs = [("Q=0.001, R=0.10", 0.001, 0.10),
               ("Q=0.005, R=0.10", 0.005, 0.10),
               ("Q=0.050, R=0.10", 0.050, 0.10),
               ("Q=0.005, R=0.01", 0.005, 0.01),
               ("Q=0.005, R=1.00", 0.005, 1.00)]
    t = np.arange(n) / 10.0
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.plot(t, meas, 'gray', alpha=0.3, lw=0.5, label='原始数据')
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for (name, q, r), c in zip(configs, colors):
        kf = KalmanFilter(101000.0, 0.1, q, r)
        o = [kf.update(z) for z in meas]
        ax.plot(t, o, color=c, lw=1.2, label=f'{name} (K={q/(q+r):.3f})')
    ax.axhline(101000.0, color='k', ls='--', alpha=0.2)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('气压 (Pa)')
    ax.set_title('Q/R 参数扫描：Q 大则跟踪快但噪声大，R 大则平滑强')
    ax.legend(fontsize=8, ncol=2); ax.grid(**GRID)
    save(fig, os.path.join(out, 'sim_ch29_3.png'))

# ------------------------------------------------------------------
# 图 4 / 2.9.4 气压-高度曲线：气压-高度，ISA / 等温 / 温度补偿
# ------------------------------------------------------------------
def fig_curve():
    ps = np.linspace(95000, 105000, 1000)
    h_isa = np.array([pressure_to_altitude_isa(p) for p in ps])
    h_iso = np.array([pressure_to_altitude_iso(p) for p in ps])
    h_t25 = np.array([pressure_to_altitude_with_temp(p, 25.0) for p in ps])
    h_t35 = np.array([pressure_to_altitude_with_temp(p, 35.0) for p in ps])
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.plot(ps, h_isa, 'b-', lw=2, label='ISA 标准模型')
    ax.plot(ps, h_iso, 'r--', lw=1.5, label='等温模型 (T=288.15K)')
    ax.plot(ps, h_t25, 'g:', lw=1.5, label='ISA+温度补偿 25°C')
    ax.plot(ps, h_t35, color='orange', ls=':', lw=1.5, label='ISA+温度补偿 35°C')
    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    ax.axvline(101325, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel('气压 (Pa)'); ax.set_ylabel('海拔高度 (m)')
    ax.set_title('气压-高度转换曲线（含温度补偿平移）')
    ax.legend(fontsize=9); ax.grid(**GRID)
    save(fig, os.path.join(out, 'sim_ch29_4.png'))

# ------------------------------------------------------------------
# 图 5 / 2.9.5 BMP280 噪声特性：偏差-时间，原始 vs 自适应KF (聚焦 BMP280)
# ------------------------------------------------------------------
def fig_bmp():
    n = 300
    np.random.seed(42)
    data = np.full(n, 101000.0) + np.random.normal(0, 0.35, n)
    kf = AdaptiveKalmanFilter(101000.0, 0.1, 0.005, 0.10)
    filt = [kf.update(z) for z in data]
    t = np.arange(n) / 10.0
    raw = np.std(data - 101000.0); flt = np.std(np.array(filt) - 101000.0)
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.plot(t, data - 101000.0, 'b-', alpha=0.4, lw=0.6,
            label=f'BMP280 原始 (σ={raw:.3f} Pa)')
    ax.plot(t, np.array(filt) - 101000.0, 'g-', lw=1.4,
            label=f'BMP280 自适应KF (σ={flt:.3f} Pa)')
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('相对真值偏差 (Pa)')
    ax.set_title(f'BMP280 噪声特性：自适应KF将噪声标准差由 {raw:.2f} 降至 {flt:.2f} Pa')
    ax.legend(fontsize=9); ax.grid(**GRID)
    save(fig, os.path.join(out, 'sim_ch29_5.png'))

# ------------------------------------------------------------------
# 图 6 / 2.9.6 残差与增益：增益-时间，标准KF vs 自适应KF
# ------------------------------------------------------------------
def fig_gain():
    t, true, meas = simulate_bmp280(500, 10.0)
    kf_std = KalmanFilter(101000.0, 0.1, 0.005, 0.10)
    kf_adp = AdaptiveKalmanFilter(101000.0, 0.1, 0.005, 0.10)
    g_std = []; g_adp = []
    for z in meas:
        kf_std.update(z); g_std.append(kf_std.k)
        kf_adp.update(z); g_adp.append(kf_adp.k)
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.plot(t, g_std, 'r-', lw=1.3, label='标准KF 增益 K (固定)')
    ax.plot(t, g_adp, 'g-', lw=1.3, label='自适应KF 增益 K (运动段上升)')
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('卡尔曼增益 K')
    ax.set_title('残差驱动的自适应增益：运动段增大、静止段回落')
    ax.legend(fontsize=9); ax.grid(**GRID)
    save(fig, os.path.join(out, 'sim_ch29_6.png'))

# ------------------------------------------------------------------
# 图 7 / 2.9.7 气压高度公式验证：高度-气压(ISA)，标出 0.75m 桌面气压差
# ------------------------------------------------------------------
def fig_formula():
    P0 = 101325.0
    p_floor = 100900.0
    desk_h = 0.75
    p_desk = P0 * (1 - ISA_L * desk_h / ISA_T0) ** (ISA_G / (ISA_R * ISA_L))
    dp = p_floor - p_desk
    hs = np.linspace(0, 150, 1000)
    ps = np.array([P0 * (1 - ISA_L * h / ISA_T0) ** (ISA_G / (ISA_R * ISA_L)) for h in hs])
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.plot(hs, ps, 'b-', lw=2, label='ISA 气压-高度曲线')
    ax.plot(desk_h, p_desk, 'ro', ms=8)
    ax.annotate(f'桌面 h=75cm\nΔP≈{dp:.1f}Pa',
                xy=(desk_h, p_desk),
                xytext=(desk_h + 14, p_desk + 15),
                arrowprops=dict(arrowstyle='->'))
    ax.set_xlabel('高度 (m)'); ax.set_ylabel('气压 (Pa)')
    ax.set_title('ISA 气压-高度曲线与 0.75 m 桌面高度对应的气压差')
    ax.legend(fontsize=9); ax.grid(**GRID)
    save(fig, os.path.join(out, 'sim_ch29_7.png'))


if __name__ == '__main__':
    fig_static()
    fig_motion()
    fig_qr()
    fig_curve()
    fig_bmp()
    fig_gain()
    fig_formula()
    print('ALL 7 FIGURES DONE')
