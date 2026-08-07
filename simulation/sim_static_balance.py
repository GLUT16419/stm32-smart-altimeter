# -*- coding: utf-8 -*-
"""图：静止大气压强随高度的变化（对应公式(2-1) dp = -ρ·g·dh 的物理起点）。
- 上：气压 P(h) 随高度单调下降（指数/ISA 曲线）
- 下：变化率 |dP/dh| = ρ·g 随高度递减（空气变稀，低空掉得快、高空掉得慢）
输出 simulation/sim_static_balance.png。
SimHei 中文、dpi150。"""
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# 物理常数（与 altitude_convert.h / 文档一致）
T0 = 288.15      # 海平面标准温度 K
L  = 0.0065      # 温度递减率 K/m
G  = 9.80665     # 重力加速度 m/s^2
R  = 287.05      # 干空气气体常数 J/(kg·K)
P0 = 101325.0     # 海平面标准气压 Pa

def isa_pressure(h, p0=P0):
    return p0 * (1 - L * h / T0) ** (G / (R * L))

def rho(h):
    T = T0 - L * h
    return isa_pressure(h) / (R * T)

h = np.linspace(0, 11000, 800)
P = np.array([isa_pressure(x) for x in h])
dPdh = np.array([rho(x) * G for x in h])   # |dP/dh| = ρ·g

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.2), dpi=150)

# 上：P-h
ax1.plot(h / 1000, P / 1000, 'b-', lw=2.4, label='静止大气压 P(h)')
ax1.set_xlabel('高度 h / km')
ax1.set_ylabel('气压 P / kPa')
ax1.set_title('(a) 静止大气压强随高度单调下降')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# 下：|dP/dh| = ρ·g
ax2.plot(h / 1000, dPdh, 'r-', lw=2.4, label='|dP/dh| = ρ·g')
ax2.set_xlabel('高度 h / km')
ax2.set_ylabel('|dP/dh| / (Pa/m)')
ax2.set_title('(b) 气压变化率 ρ·g 随高度递减（低空掉得快、高空掉得慢）')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# 标注海平面附近灵敏度（~12 Pa/m）
ax2.annotate('海平面附近 ≈ 12 Pa/m', xy=(0, dPdh[0]), xytext=(2.5, dPdh[0] * 0.82),
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10, color='gray')

plt.tight_layout()
OUT = 'sim_static_balance.png'
plt.savefig(OUT, dpi=150)
print('saved', OUT)
