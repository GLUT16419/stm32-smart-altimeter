# -*- coding: utf-8 -*-
"""
气压高度公式仿真验证（配合实习报告）
=====================================
本脚本演示并验证实习报告中两处结论：
  1. 第 2 章推导的 ISA 气压-高度公式，并在曲线上标出
     0.75 m（普通办公桌高度）对应的气压差（ΔP ≈ 9 Pa）；
  2. 第 6 章“室内相对高度测试方法”：地板 ↔ 桌面（约 75 cm）的静态差分测试。
     由于是静态测量，两点气压噪声相互独立、取差分后相互抵消，
     因此即使不做卡尔曼滤波、直接用原始气压经公式换算，
     高度差也能稳定得到约 0.75 m。

运行：python pressure_height_formula_sim.py
依赖：numpy, matplotlib
输出：sim_formula_demo.png
"""

import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# ---- 常数（与 altitude_convert.h 完全一致）----
ISA_T0 = 288.15     # 海平面标准温度 (K)
ISA_L  = 0.0065     # 温度递减率 (K/m)
ISA_G  = 9.80665    # 重力加速度 (m/s^2)
ISA_R  = 287.05     # 干空气气体常数 (J/(kg·K))
P0     = 101325.0   # 海平面标准气压 (Pa)


def isa_altitude(p, p0=P0):
    """ISA 气压-高度公式：h = (T0/L) * (1 - (P/P0)^(L*R/g))"""
    if p <= 0 or p0 <= 0:
        return 0.0
    ratio = p / p0
    exponent = ISA_L * ISA_R / ISA_G   # ≈ 0.19027
    return (ISA_T0 / ISA_L) * (1.0 - ratio ** exponent)


def isa_pressure(h, p0=P0):
    """ISA 高度-气压反算：P = P0 * (1 - L*h/T0)^(g/(R*L))"""
    return p0 * (1 - ISA_L * h / ISA_T0) ** (ISA_G / (ISA_R * ISA_L))


def kf_seq(meas, q=0.005, r=0.10, x0=None):
    """一维标准卡尔曼滤波序列（与 kalman_filter.c 一致）"""
    x = x0 if x0 is not None else meas[0]
    p = 0.1
    out = []
    for z in meas:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        out.append(x)
    return np.array(out)


def main():
    # ===== 场景参数：普通办公桌标称高度 75 cm =====
    desk_h = 0.75
    p_floor = 100900.0          # 地板处近似气压 (Pa)
    p_desk = isa_pressure(desk_h, p_floor)   # 桌面气压（比地板低）
    dp = p_floor - p_desk

    print("=" * 60)
    print("气压高度公式仿真验证")
    print("=" * 60)
    print(f"地板气压 P_floor = {p_floor:.2f} Pa")
    print(f"桌面高度差 Δh     = {desk_h:.2f} m ({desk_h*100:.0f} cm)")
    print(f"桌面气压 P_desk  = {p_desk:.2f} Pa")
    print(f"气压差 ΔP         = {dp:.3f} Pa  (≈ {dp/desk_h:.1f} Pa/m)")
    # 用公式反算高度差，检验还原精度
    h_back = isa_altitude(p_desk, p_floor)
    print(f"公式反算高度差    = {h_back:.4f} m  (误差 {abs(h_back-desk_h)*1000:.3f} mm)")

    # ===== 差分测试仿真：即使不滤波也能得到 ~0.75 m =====
    np.random.seed(2024)
    n = 500
    sigma = 0.35   # BMP280 静态噪声标准差 (Pa)
    p_floor_samples = p_floor + np.random.normal(0, sigma, n)
    p_desk_samples = p_desk + np.random.normal(0, sigma, n)

    # （1）无滤波：原始气压直接换算高度后取差分
    h_floor_raw = np.array([isa_altitude(p, P0) for p in p_floor_samples])
    h_desk_raw = np.array([isa_altitude(p, P0) for p in p_desk_samples])
    delta_raw = h_desk_raw - h_floor_raw
    print()
    print(f"无滤波差分 Δh: 均值={delta_raw.mean():.4f} m, 标准差={delta_raw.std():.4f} m")
    print(f"          -> 约 {delta_raw.mean()*100:.1f} cm，接近 75 cm")

    # （2）有滤波：标准 KF 平滑气压后再换算取差分
    h_floor_f = kf_seq(p_floor_samples)
    h_desk_f = kf_seq(p_desk_samples)
    delta_f = (np.array([isa_altitude(x, P0) for x in h_desk_f])
               - np.array([isa_altitude(x, P0) for x in h_floor_f]))
    print(f"有滤波差分 Δh: 均值={delta_f.mean():.4f} m, 标准差={delta_f.std():.4f} m")

    # ===== 绘图 =====
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    # 上：P-h 曲线（标出 0.75 m 桌面高度的气压差）
    hs = np.linspace(0, 150, 1000)
    ps = np.array([isa_pressure(h, p_floor) for h in hs])
    ax = axes[0]
    ax.plot(hs, ps, 'b-', lw=2, label='ISA 气压-高度曲线')
    ax.plot(desk_h, p_desk, 'ro', ms=8)
    ax.annotate(f'桌面 h={desk_h*100:.0f}cm\nΔP≈{dp:.1f}Pa',
                xy=(desk_h, p_desk), xytext=(desk_h + 12, p_desk + 12),
                arrowprops=dict(arrowstyle='->'))
    ax.set_xlabel('高度 (m)')
    ax.set_ylabel('气压 (Pa)')
    ax.set_title('气压-高度(ISA)曲线与 0.75m 桌面高度对应的气压差')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 下：差分测试直方图（无滤波 vs 有滤波 vs 真实值）
    ax = axes[1]
    ax.hist(delta_raw * 100, bins=30, color='steelblue', alpha=0.7,
            label=f'无滤波 Δh (均值 {delta_raw.mean()*100:.1f}cm)')
    ax.axvline(desk_h * 100, color='r', ls='--', lw=2, label=f'真实 {desk_h*100:.0f}cm')
    ax.axvline(delta_f.mean() * 100, color='g', ls=':', lw=2,
               label=f'有滤波均值 {delta_f.mean()*100:.1f}cm')
    ax.set_xlabel('地板-桌面 高度差 (cm)')
    ax.set_ylabel('样本数')
    ax.set_title('室内相对高度差分测试（静态：无滤波亦可得 ~75cm）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sim_formula_demo.png', dpi=150)
    print("\n→ 已保存 sim_formula_demo.png")


if __name__ == '__main__':
    main()
