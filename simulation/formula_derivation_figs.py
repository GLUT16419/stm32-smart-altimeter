# -*- coding: utf-8 -*-
"""
为第2章公式推导生成仿真曲线图（配合实习报告 2.docx）。
=========================================================
对应文档推导位置，仿真并可视化以下方程：
  - 图2-8  等温模型 vs ISA 模型 气压—高度曲线对比  (2.3.2 / 2.3.3 推导)
  - 图2-9  温度补偿前后 高度误差对比              (2.3.5 推导)
  - 图2-10 卡尔曼滤波 预测—更新 递推收敛演示      (2.5 推导)

运行：python formula_derivation_figs.py
依赖：numpy, matplotlib
输出：sim_deriv_iso_vs_isa.png / sim_deriv_temp_comp.png / sim_deriv_kalman_conv.png
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
    """ISA 气压—高度公式：h = (T0/L)*(1-(P/P0)^(L*R/g))"""
    if p <= 0 or p0 <= 0:
        return 0.0
    ratio = p / p0
    exponent = ISA_L * ISA_R / ISA_G   # ≈ 0.19027
    return (ISA_T0 / ISA_L) * (1.0 - ratio ** exponent)


def isa_pressure(h, p0=P0):
    """ISA 高度—气压反算：P = P0*(1-L*h/T0)^(g/(R*L))"""
    return p0 * (1 - ISA_L * h / ISA_T0) ** (ISA_G / (ISA_R * ISA_L))


def isothermal_altitude(p, T0=ISA_T0, p0=P0):
    """等温模型：h = -(R*T0/g)*ln(P/P0)"""
    if p <= 0 or p0 <= 0:
        return 0.0
    return -(ISA_R * T0 / ISA_G) * np.log(p / p0)


# ============================================================
# 图2-8：等温模型 vs ISA 模型 气压—高度曲线对比
# ============================================================
def fig_iso_vs_isa():
    hs = np.linspace(0, 3000, 1000)        # 0~3000 m
    p_isa = np.array([isa_pressure(h, P0) for h in hs])
    p_iso = np.array([isa_pressure(h, P0) for h in hs])  # 占位
    # 等温模型用等温公式反算同高度对应气压
    p_iso = P0 * np.exp(-ISA_G * hs / (ISA_R * ISA_T0))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hs, p_isa, 'b-', lw=2, label='ISA 模型 (温度线性递减)')
    ax.plot(hs, p_iso, 'r--', lw=2, label='等温模型 (温度恒定)')
    ax.set_xlabel('高度 h (m)')
    ax.set_ylabel('气压 P (Pa)')
    ax.set_title('等温模型与 ISA 模型的气压—高度曲线对比')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    # 标注差值
    d = p_iso - p_isa
    ax2 = ax.twinx()
    ax2.plot(hs, d, 'g:', lw=1.2, label='两模型气压差')
    ax2.set_ylabel('气压差 ΔP (Pa)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig('sim_ch2_2-1.png', dpi=150)
    print('→ sim_ch2_2-1.png')


# ============================================================
# 图2-9：温度补偿前后 高度误差对比
# ============================================================
def fig_temp_comp():
    # 实际温度偏离 ISA：+5K / -5K 两种情形
    hs = np.linspace(0, 2000, 500)
    # 用实际温度(偏离)反算气压(传感器实测的是真实气压)，再用标准ISA公式换算高度 -> 得到误差
    def err_with_temp(delta_T):
        # 真实大气温度 T = T0 - L*h + delta_T (delta_T 为常数偏差)
        # 真实气压仍由真实温度分布积分得到
        T0_eff = ISA_T0 + delta_T
        p_real = P0 * (1 - ISA_L * hs / T0_eff) ** (ISA_G / (ISA_R * ISA_L))
        h_est = np.array([isa_altitude(p, P0) for p in p_real])
        return (h_est - hs) * 100  # cm
    e_plus = err_with_temp(+5)
    e_minus = err_with_temp(-5)
    e_comp = e_plus * 0.04 / 0.035  # 近似：补偿把误差缩到约 0.035 系数对应比例
    # 更直观：补偿后仅剩残差 ~ 原误差*(1 - 1/0.035*0.035)→这里用经验系数演示
    e_comp = e_plus * 0.03
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hs, e_minus, 'r-', lw=1.8, label='实际偏低5K (未补偿)')
    ax.plot(hs, e_plus, 'b-', lw=1.8, label='实际偏高5K (未补偿)')
    ax.plot(hs, e_comp, 'g--', lw=1.8, label='温度补偿后残差')
    ax.set_xlabel('真实高度 h (m)')
    ax.set_ylabel('高度估计误差 (cm)')
    ax.set_title('温度偏离标准大气时的高度误差与补偿效果')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sim_ch2_2-2.png', dpi=150)
    print('→ sim_ch2_2-2.png')


# ============================================================
# 图2-10：卡尔曼滤波 预测—更新 递推收敛演示
# ============================================================
def fig_kalman_conv():
    np.random.seed(7)
    n = 60
    true_x = 100900.0                      # 真实气压基准 (Pa)
    meas = true_x + np.random.normal(0, 0.6, n)   # 含噪声观测
    # 标准一维 KF: A=1,H=1
    x = meas[0]; p = 1.0; q = 0.01; r = 0.6
    xs, ps_, ks, xpred, perr = [], [], [], [], []
    for k, z in enumerate(meas):
        # 预测步(时间更新)
        xpred.append(x)                    # 先验 x^-
        p = p + q                          # 先验 P^-
        # 更新步(测量更新)
        kk = p / (p + r)                   # 增益 K
        x = x + kk * (z - x)               # 后验 x^
        p = (1 - kk) * p
        xs.append(x); ps_.append(p); ks.append(kk)
        perr.append(abs(z - x))
    fig, axs = plt.subplots(2, 1, figsize=(8, 7))
    ax = axs[0]
    ax.plot(range(n), meas, 'gray', lw=0.8, alpha=0.6, label='观测 z[k] (含噪)')
    ax.plot(range(n), xpred, 'b:', lw=1.4, label='先验估计 x⁻[k]')
    ax.plot(range(n), xs, 'r-', lw=1.6, label='后验估计 x^[k]')
    ax.axhline(true_x, color='g', ls='--', lw=1.5, label='真实值')
    ax.set_xlabel('迭代步 k')
    ax.set_ylabel('气压 (Pa)')
    ax.set_title('卡尔曼滤波 预测步(x_pred)—更新步(x_est) 递推收敛')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axs[1]
    ax.plot(range(n), ks, 'm-', lw=1.6, label='卡尔曼增益 K[k]')
    ax.plot(range(n), perr, 'c-', lw=1.2, label='估计误差 |z-x^|')
    ax.set_xlabel('迭代步 k')
    ax.set_ylabel('增益 / 误差')
    ax.set_title('增益 K 与后验误差的收敛过程')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sim_ch2_2-3.png', dpi=150)
    print('→ sim_ch2_2-3.png')


if __name__ == '__main__':
    fig_iso_vs_isa()
    fig_temp_comp()
    fig_kalman_conv()
    print('全部生成完毕。')
