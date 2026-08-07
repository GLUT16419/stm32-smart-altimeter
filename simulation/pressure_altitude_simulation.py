# -*- coding: utf-8 -*-
"""
气压高度测量系统 —— Python 计算机仿真
============================================
对应子课题1：BMP280 气压数据读取、卡尔曼滤波、气压方程推导

本仿真完全复现了嵌入式项目 C 代码的算法逻辑，用于：
  1. 验证卡尔曼滤波算法的正确性
  2. 对比标准KF与自适应KF的性能差异
  3. 展示不同 Q/R 参数对滤波效果的影响
  4. 绘制气压-高度转换的理论曲线
  5. 模拟静止/运动工况下的滤波行为

运行环境：Python 3.8+，需安装 numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ====================================================
# 1. 常量定义（与C代码 altitude_convert.h 一致）
# ====================================================
SEA_LEVEL_PRESSURE_PA = 101325.0   # 海平面标准气压 (Pa)
ISA_T0 = 288.15                     # 海平面标准温度 (K)
ISA_L  = 0.0065                     # 温度递减率 (K/m)
ISA_G  = 9.80665                    # 重力加速度 (m/s^2)
ISA_R  = 287.05                     # 干空气气体常数 (J/(kg·K))

# 卡尔曼滤波参数（BMP280，与 kalman_filter.h 一致）
BMP280_KF_Q = 0.005      # 基础过程噪声 Q_base
BMP280_KF_R = 0.10       # 测量噪声方差 R = σ²
BMP280_RESIDUAL_TH = 1.2 # 残差阈值 (Pa)
BMP280_Q_INCREASE = 1.08 # Q 放大系数
BMP280_Q_DECREASE = 0.97 # Q 衰减系数
BMP280_Q_MAX = 5.0       # Q 上限

# MS5611 参数（对比用）
MS5611_KF_Q = 0.03
MS5611_KF_R = 9.3
KF_RESIDUAL_TH = 5.0
KF_Q_INCREASE = 1.05
KF_Q_DECREASE = 0.98
KF_Q_MAX = 2.0


# ====================================================
# 2. 气压-高度转换（与 altitude_convert.c 一致）
# ====================================================
def pressure_to_altitude_isa(pressure_pa, ref_pressure_pa=SEA_LEVEL_PRESSURE_PA):
    """
    ISA 国际标准大气公式
    h = (T0/L) * (1 - (P/P0)^(L*R/g))
    """
    if pressure_pa <= 0 or ref_pressure_pa <= 0:
        return 0.0
    ratio = pressure_pa / ref_pressure_pa
    if ratio <= 0:
        return 0.0
    exponent = ISA_L * ISA_R / ISA_G   # ≈ 0.1903
    altitude = (ISA_T0 / ISA_L) * (1.0 - ratio ** exponent)
    return altitude

def pressure_to_altitude_isotherm(pressure_pa, ref_pressure_pa=SEA_LEVEL_PRESSURE_PA):
    """
    等温大气模型
    h = -(R*T/g) * ln(P/P0)
    """
    if pressure_pa <= 0 or ref_pressure_pa <= 0:
        return 0.0
    ratio = pressure_pa / ref_pressure_pa
    if ratio <= 0:
        return 0.0
    altitude = -(ISA_R * ISA_T0 / ISA_G) * np.log(ratio)
    return altitude

def pressure_to_altitude_with_temp(pressure_pa, temperature_c,
                                    ref_pressure_pa=SEA_LEVEL_PRESSURE_PA):
    """
    ISA 公式 + 温度补偿
    """
    if pressure_pa <= 0 or ref_pressure_pa <= 0:
        return 0.0
    ratio = pressure_pa / ref_pressure_pa
    if ratio <= 0 or ratio > 10:
        return 0.0
    # ISA 标准高度
    exponent = ISA_L * ISA_R / ISA_G
    altitude = (ISA_T0 / ISA_L) * (1.0 - ratio ** exponent)
    # 温度补偿
    temperature_k = temperature_c + 273.15
    t_isa = ISA_T0 - ISA_L * altitude    # 该高度处的ISA标准温度
    delta_t = temperature_k - t_isa
    altitude += delta_t * 0.035           # 经验系数
    return altitude


# ====================================================
# 3. 卡尔曼滤波器实现（与 kalman_filter.c 一致）
# ====================================================
class KalmanFilter:
    """一维标准卡尔曼滤波器"""
    def __init__(self, init_x, init_p, q, r):
        self.x = init_x          # 状态估计
        self.p = init_p          # 误差协方差
        self.q = q               # 过程噪声协方差
        self.r = r               # 测量噪声协方差
        self.k = 0.0             # 卡尔曼增益
        self.q_base = q          # 基础 Q
        
    def update(self, z):
        """标准卡尔曼滤波更新（5个公式）"""
        # 先验误差协方差
        self.p = self.p + self.q
        # 卡尔曼增益
        self.k = self.p / (self.p + self.r)
        # 后验状态估计
        self.x = self.x + self.k * (z - self.x)
        # 后验误差协方差
        self.p = (1 - self.k) * self.p
        return self.x

class AdaptiveKalmanFilter(KalmanFilter):
    """自适应卡尔曼滤波器（BMP280 版本）"""
    def __init__(self, init_x, init_p, q, r,
                 residual_th=BMP280_RESIDUAL_TH,
                 q_inc=BMP280_Q_INCREASE,
                 q_dec=BMP280_Q_DECREASE,
                 q_max=BMP280_Q_MAX):
        super().__init__(init_x, init_p, q, r)
        self.residual_th = residual_th
        self.q_inc = q_inc
        self.q_dec = q_dec
        self.q_max = q_max
        self.q_history = []    # 记录Q值变化
        self.win = [0.0] * 5  # 最近 5 帧残差绝对值（窗口残差STD判别）
        self.idx = 0
        
    def update(self, z):
        """自适应卡尔曼滤波更新（5帧窗口残差STD判别，与固件一致）"""
        residual = z - self.x
        residual_abs = abs(residual)
        # 维护 5 帧残差滑动窗口，用标准差作运动判据
        self.win[self.idx] = residual_abs
        self.idx = (self.idx + 1) % 5
        m = sum(self.win) / 5.0
        rstd = (sum((w - m) ** 2 for w in self.win) / 5.0) ** 0.5

        # 自适应调整 Q
        if rstd > self.residual_th:
            self.q = self.q * self.q_inc
            if self.q > self.q_max:
                self.q = self.q_max
        else:
            self.q = self.q * self.q_dec
            if self.q < self.q_base:
                self.q = self.q_base

        # 标准卡尔曼五公式
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * residual
        self.p = (1 - self.k) * self.p

        self.q_history.append(self.q)
        return self.x


# ====================================================
# 4. 传感器数据模拟
# ====================================================
def simulate_bmp280_data(n_samples=500, fs=10.0, noise_std=0.35):
    """
    模拟 BMP280 传感器数据（含运动场景）
    
    参数：
        n_samples: 采样点数
        fs: 采样率 (Hz)
        noise_std: 噪声标准差 (Pa)
    
    返回：
        time: 时间轴 (s)
        true_pressure: 真实气压 (Pa)
        measured: 加噪测量值 (Pa)
        temperature: 温度 (℃)
    """
    dt = 1.0 / fs
    t = np.arange(n_samples) * dt
    
    # 真实气压：静止 + 运动
    true_p = np.full(n_samples, 101000.0)
    
    # 阶段1: 静止 (0~15s)
    # 阶段2: 缓慢上升（模拟高度降低）(15~25s)
    # 阶段3: 静止 (25~35s)
    # 阶段4: 快速下降（模拟高度升高）(35~42s)
    # 阶段5: 静止 (42~50s)
    
    # 气压变化对应高度变化：约 -12 Pa/m (海平面附近)
    # 缓慢上升 3m → +36 Pa
    rise_start = int(15 * fs)
    rise_end = int(25 * fs)
    ramp_up = np.linspace(0, 36, rise_end - rise_start)
    true_p[rise_start:rise_end] += ramp_up
    
    # 快速下降 5m → -60 Pa
    fall_start = int(35 * fs)
    fall_end = int(42 * fs)
    ramp_down = np.linspace(0, -60, fall_end - fall_start)
    true_p[fall_start:fall_end] += ramp_down
    
    # 后段保持
    true_p[fall_end:] = true_p[fall_end-1]
    true_p[rise_end:fall_start] = true_p[rise_end-1]
    
    # 叠加噪声
    np.random.seed(42)
    noise = np.random.normal(0, noise_std, n_samples)
    measured = true_p + noise
    
    # 温度基本恒定 (25±0.5℃)
    temperature = 25.0 + np.random.normal(0, 0.2, n_samples)
    
    return t, true_p, measured, temperature


def simulate_ms5611_data(n_samples=500, fs=10.0, noise_std=3.05):
    """模拟 MS5611 传感器数据（噪声更大）"""
    t, true_p, _, temperature = simulate_bmp280_data(n_samples, fs, 0)  # 复用真值
    np.random.seed(123)
    noise = np.random.normal(0, noise_std, n_samples)
    measured = true_p + noise
    return t, true_p, measured, temperature


# ====================================================
# 5. 仿真实验
# ====================================================

def run_stationary_test():
    """实验1: 静态滤波效果对比"""
    print("=" * 60)
    print("实验1: 静态滤波效果对比")
    print("=" * 60)
    
    # 生成静止数据 (仅噪声)
    n = 300
    np.random.seed(42)
    true_p = np.full(n, 101000.0)
    measured = true_p + np.random.normal(0, 0.35, n)
    
    # 标准KF
    kf_std = KalmanFilter(init_x=101000.0, init_p=0.1, q=0.005, r=0.10)
    kf_std_out = [kf_std.update(z) for z in measured]
    
    # 自适应KF (BMP280)
    kf_adp = AdaptiveKalmanFilter(init_x=101000.0, init_p=0.1, q=0.005, r=0.10)
    kf_adp_out = [kf_adp.update(z) for z in measured]
    
    # 统计
    raw_std = np.std(measured)
    kf_std_std = np.std(kf_std_out)
    kf_adp_std = np.std(kf_adp_out)
    
    print(f"原始数据标准差: {raw_std:.4f} Pa")
    print(f"标准 KF 标准差: {kf_std_std:.4f} Pa  (抑制比: {raw_std/kf_std_std:.1f}x)")
    print(f"自适应KF标准差: {kf_adp_std:.4f} Pa  (抑制比: {raw_std/kf_adp_std:.1f}x)")
    
    # 画图
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    t = np.arange(n) / 10.0
    
    ax = axes[0]
    ax.plot(t, measured, 'b-', alpha=0.5, label='原始测量值', linewidth=0.8)
    ax.plot(t, kf_std_out, 'r-', label=f'标准KF (std={kf_std_std:.3f}Pa)', linewidth=1.2)
    ax.plot(t, kf_adp_out, 'g-', label=f'自适应KF (std={kf_adp_std:.3f}Pa)', linewidth=1.2)
    ax.axhline(y=101000.0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylabel('气压 (Pa)')
    ax.set_title('图1: 静态滤波效果对比 (BMP280, σ=0.35Pa)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(t, measured - 101000.0, 'b-', alpha=0.5, label='原始偏差', linewidth=0.8)
    ax.plot(t, np.array(kf_std_out) - 101000.0, 'r-', label='标准KF偏差', linewidth=1.2)
    ax.plot(t, np.array(kf_adp_out) - 101000.0, 'g-', label='自适应KF偏差', linewidth=1.2)
    ax.set_ylabel('偏差 (Pa)')
    ax.set_title('图2: 滤波偏差对比')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(t, np.array(kf_adp.q_history), 'purple', linewidth=1.2)
    ax.axhline(y=BMP280_KF_Q, color='gray', linestyle='--', alpha=0.5, label=f'Q_base={BMP280_KF_Q}')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('Q 值')
    ax.set_title('图3: 自适应Q值变化（静止时应稳定在 Q_base）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sim_stationary_compare.png', dpi=150)
    print("→ 已保存: sim_stationary_compare.png\n")
    plt.show()


def run_motion_test():
    """实验2: 运动场景滤波跟踪效果"""
    print("=" * 60)
    print("实验2: 运动场景滤波跟踪效果")
    print("=" * 60)
    
    t, true_p, measured, temp = simulate_bmp280_data(n_samples=500, fs=10.0)
    
    # 标准KF (小Q)
    kf_std = KalmanFilter(init_x=101000.0, init_p=0.1, q=0.005, r=0.10)
    kf_std_out = [kf_std.update(z) for z in measured]
    
    # 标准KF (大Q)
    kf_std_large_q = KalmanFilter(init_x=101000.0, init_p=0.1, q=1.0, r=0.10)
    kf_std_large_q_out = [kf_std_large_q.update(z) for z in measured]
    
    # 自适应KF
    kf_adp = AdaptiveKalmanFilter(init_x=101000.0, init_p=0.1, q=0.005, r=0.10)
    kf_adp_out = [kf_adp.update(z) for z in measured]
    
    # 计算跟踪延迟
    # 在快速下降段（35~42s），计算滤波值与真值的均方根误差
    fall_start = int(35 * 10)
    fall_end = int(42 * 10)
    
    def rmse(a, b):
        return np.sqrt(np.mean((a - b)**2))
    
    print(f"静止段 (0-15s) RMSE:")
    print(f"  标准KF(Q=0.005): {rmse(kf_std_out[:150], true_p[:150]):.3f} Pa")
    print(f"  自适应KF:        {rmse(kf_adp_out[:150], true_p[:150]):.3f} Pa")
    print(f"运动段 (35-42s) RMSE:")
    print(f"  标准KF(Q=0.005): {rmse(kf_std_out[fall_start:fall_end], true_p[fall_start:fall_end]):.3f} Pa")
    print(f"  标准KF(Q=1.0):   {rmse(kf_std_large_q_out[fall_start:fall_end], true_p[fall_start:fall_end]):.3f} Pa")
    print(f"  自适应KF:        {rmse(kf_adp_out[fall_start:fall_end], true_p[fall_start:fall_end]):.3f} Pa")
    
    # 画图
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    ax = axes[0]
    ax.plot(t, true_p, 'k-', label='真实气压', linewidth=1.5)
    ax.plot(t, measured, 'b-', alpha=0.4, label='原始测量值', linewidth=0.6)
    ax.set_ylabel('气压 (Pa)')
    ax.set_title('图4: 运动场景 — 原始数据')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(t, true_p, 'k-', label='真实气压', linewidth=1.5)
    ax.plot(t, kf_std_out, 'r-', label=f'标准KF (Q=0.005)', linewidth=1.2)
    ax.plot(t, kf_adp_out, 'g-', label=f'自适应KF', linewidth=1.2)
    ax.set_ylabel('气压 (Pa)')
    ax.set_title('图5: 标准KF vs 自适应KF 跟踪效果')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(t, kf_std_out, 'r-', label=f'标准KF (Q=0.005)', linewidth=1.2)
    ax.plot(t, kf_std_large_q_out, 'orange', label=f'标准KF (Q=1.0)', linewidth=1.2)
    ax.plot(t, kf_adp_out, 'g-', label=f'自适应KF', linewidth=1.2)
    ax.plot(t, true_p, 'k--', label='真值', linewidth=1.0, alpha=0.7)
    ax.set_ylabel('气压 (Pa)')
    ax.set_xlim(30, 50)
    ax.set_title('图6: 运动段放大 (30~50s)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[3]
    ax.plot(t, kf_adp.q_history, 'purple', linewidth=1.2)
    ax.axhline(y=BMP280_RESIDUAL_TH, color='red', linestyle='--', alpha=0.5, label='残差阈值')
    ax.axhline(y=BMP280_KF_Q, color='gray', linestyle='--', alpha=0.5, label=f'Q_base={BMP280_KF_Q}')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('Q 值')
    ax.set_title('图7: 自适应Q值变化（运动时Q增大以快速跟踪）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sim_motion_compare.png', dpi=150)
    print("→ 已保存: sim_motion_compare.png\n")
    plt.show()


def run_qr_parameter_study():
    """实验3: Q/R 参数影响分析"""
    print("=" * 60)
    print("实验3: Q/R 参数对滤波效果的影响")
    print("=" * 60)
    
    n = 300
    np.random.seed(42)
    true_p = np.full(n, 101000.0)
    measured = true_p + np.random.normal(0, 0.35, n)
    
    # 不同参数组合
    configs = [
        ("Q=0.001, R=0.10", 0.001, 0.10),
        ("Q=0.005, R=0.10", 0.005, 0.10),  # 默认
        ("Q=0.050, R=0.10", 0.050, 0.10),
        ("Q=0.005, R=0.01", 0.005, 0.01),
        ("Q=0.005, R=1.00", 0.005, 1.00),
    ]
    
    results = {}
    for name, q, r in configs:
        kf = KalmanFilter(init_x=101000.0, init_p=0.1, q=q, r=r)
        out = [kf.update(z) for z in measured]
        results[name] = out
        print(f"  {name}: std={np.std(out):.4f} Pa,  "
              f"K_稳态={q/(q+r):.4f}")
    
    # 画图
    fig, ax = plt.subplots(figsize=(14, 7))
    t = np.arange(n) / 10.0
    
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    ax.plot(t, measured, 'gray', alpha=0.3, label='原始数据', linewidth=0.5)
    for (name, out), color in zip(results.items(), colors):
        ax.plot(t, out, label=name, color=color, linewidth=1.2)
    
    ax.axhline(y=101000.0, color='k', linestyle='--', alpha=0.2)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('气压 (Pa)')
    ax.set_title('图8: 不同 Q/R 参数对滤波效果的影响')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sim_qr_parameter_study.png', dpi=150)
    print("→ 已保存: sim_qr_parameter_study.png\n")
    plt.show()


def run_pressure_altitude_curve():
    """实验4: 气压-高度转换曲线"""
    print("=" * 60)
    print("实验4: 气压-高度转换曲线")
    print("=" * 60)
    
    # 气压范围：95000 ~ 105000 Pa（对应约 -350~+550m）
    pressures = np.linspace(95000, 105000, 1000)
    
    # 高度转换
    h_isa = np.array([pressure_to_altitude_isa(p) for p in pressures])
    h_iso = np.array([pressure_to_altitude_isotherm(p) for p in pressures])
    h_temp25 = np.array([pressure_to_altitude_with_temp(p, 25.0) for p in pressures])
    h_temp35 = np.array([pressure_to_altitude_with_temp(p, 35.0) for p in pressures])
    
    # 局部灵敏度：dP/dh ≈ -12 Pa/m
    def sensitivity(p, delta=0.1):
        """计算气压对高度的灵敏度"""
        h1 = pressure_to_altitude_isa(p - delta)
        h2 = pressure_to_altitude_isa(p + delta)
        return (h2 - h1) / (2 * delta)
    
    sens = np.array([sensitivity(p) for p in pressures])
    
    # 画图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    ax = axes[0]
    ax.plot(pressures, h_isa, 'b-', linewidth=2, label='ISA标准模型')
    ax.plot(pressures, h_iso, 'r--', linewidth=1.5, label='等温模型 (T=288.15K)')
    ax.plot(pressures, h_temp25, 'g:', linewidth=1.5, label='ISA+T=25°C补偿')
    ax.plot(pressures, h_temp35, color='orange', linestyle=':', linewidth=1.5, label='ISA+T=35°C补偿')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=101325, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('气压 (Pa)')
    ax.set_ylabel('海拔高度 (m)')
    ax.set_title('图9: 气压-高度转换曲线')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(pressures, sens, 'purple', linewidth=2)
    ax.axhline(y=-12.0, color='gray', linestyle='--', alpha=0.5, label='约-12 m/Pa')
    ax.set_xlabel('气压 (Pa)')
    ax.set_ylabel('灵敏度 dh/dP (m/Pa)')
    ax.set_title('图10: 气压-高度灵敏度曲线')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sim_pressure_altitude_curve.png', dpi=150)
    print("→ 已保存: sim_pressure_altitude_curve.png")
    
    # 输出典型值
    print("\n典型高度-气压对应值:")
    for h_target in [0, 10, 50, 100, 200, 500]:
        p = 101325 * (1 - h_target * ISA_L / ISA_T0) ** (ISA_G / (ISA_R * ISA_L))
        h_back = pressure_to_altitude_isa(p)
        print(f"  {h_target:4d}m → P={p:.1f}Pa → h={h_back:.2f}m")
    print()


def run_dual_sensor_comparison():
    """实验5: BMP280 vs MS5611 双传感器对比"""
    print("=" * 60)
    print("实验5: BMP280 vs MS5611 双传感器静态对比")
    print("=" * 60)
    
    n = 300
    t = np.arange(n) / 10.0
    
    # BMP280: σ=0.35Pa
    np.random.seed(42)
    bmp_data = np.full(n, 101000.0) + np.random.normal(0, 0.35, n)
    kf_bmp = AdaptiveKalmanFilter(101000.0, 0.1, 0.005, 0.10,
                                   residual_th=1.2, q_inc=1.08, q_dec=0.97, q_max=5.0)
    bmp_filt = [kf_bmp.update(z) for z in bmp_data]
    
    # MS5611: σ=3.05Pa
    np.random.seed(123)
    ms5611_data = np.full(n, 101000.0) + np.random.normal(0, 3.05, n)
    kf_ms = AdaptiveKalmanFilter(101000.0, 0.1, 0.03, 9.3,
                                  residual_th=5.0, q_inc=1.05, q_dec=0.98, q_max=2.0)
    ms_filt = [kf_ms.update(z) for z in ms5611_data]
    
    print(f"BMP280:  原始σ={np.std(bmp_data):.3f}Pa, 滤波后σ={np.std(bmp_filt):.3f}Pa")
    print(f"MS5611: 原始σ={np.std(ms5611_data):.3f}Pa, 滤波后σ={np.std(ms_filt):.3f}Pa")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax = axes[0]
    ax.plot(t, bmp_data - 101000, 'b-', alpha=0.4, label='BMP280 原始', linewidth=0.6)
    ax.plot(t, np.array(bmp_filt) - 101000, 'g-', label='BMP280 自适应KF', linewidth=1.5)
    ax.set_ylabel('偏差 (Pa)')
    ax.set_title('图11: BMP280 滤波效果 (σ=0.35Pa)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(t, ms5611_data - 101000, 'b-', alpha=0.4, label='MS5611 原始', linewidth=0.6)
    ax.plot(t, np.array(ms_filt) - 101000, 'r-', label='MS5611 自适应KF', linewidth=1.5)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('偏差 (Pa)')
    ax.set_title('图12: MS5611 滤波效果 (σ=3.05Pa)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sim_dual_sensor_compare.png', dpi=150)
    print("→ 已保存: sim_dual_sensor_compare.png\n")
    plt.show()


def run_residual_analysis():
    """实验6: 残差分析 — 标准KF vs 自适应KF"""
    print("=" * 60)
    print("实验6: 残差与增益分析")
    print("=" * 60)
    
    t, true_p, measured, temp = simulate_bmp280_data(n_samples=500, fs=10.0)
    
    kf_std = KalmanFilter(101000.0, 0.1, 0.005, 0.10)
    kf_adp = AdaptiveKalmanFilter(101000.0, 0.1, 0.005, 0.10)
    
    residuals_std = []
    residuals_adp = []
    gains_std = []
    gains_adp = []
    
    for i, z in enumerate(measured):
        res_std = z - kf_std.x
        kf_std.update(z)
        residuals_std.append(res_std)
        gains_std.append(kf_std.k)
        
        res_adp = z - kf_adp.x
        kf_adp.update(z)
        residuals_adp.append(res_adp)
        gains_adp.append(kf_adp.k)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    ax = axes[0]
    ax.plot(t, true_p, 'k-', label='真值', linewidth=1.5)
    ax.plot(t, measured, 'b-', alpha=0.3, label='测量值', linewidth=0.5)
    ax.plot(t, kf_std_out := [KalmanFilter(101000.0, 0.1, 0.005, 0.10).update(z) for z in measured], 'r-', linewidth=1.2)
    ax.set_ylabel('气压 (Pa)')
    ax.set_title('图13: 标准KF（固定Q）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(t, residuals_std, 'r-', alpha=0.5, label='标准KF残差', linewidth=0.8)
    ax.plot(t, residuals_adp, 'g-', alpha=0.5, label='自适应KF残差', linewidth=0.8)
    ax.axhline(y=BMP280_RESIDUAL_TH, color='red', linestyle='--', alpha=0.5, label='阈值')
    ax.axhline(y=-BMP280_RESIDUAL_TH, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('残差 (Pa)')
    ax.set_title('图14: 残差对比')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(t, gains_std, 'r-', label='标准KF增益', linewidth=1.2)
    ax.plot(t, gains_adp, 'g-', label='自适应KF增益', linewidth=1.2)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('卡尔曼增益 K')
    ax.set_title('图15: 卡尔曼增益对比')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sim_residual_analysis.png', dpi=150)
    print("→ 已保存: sim_residual_analysis.png\n")
    plt.show()


# ====================================================
# 6. 主程序
# ====================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  气压高度测量系统 — Python 计算机仿真")
    print("  对应项目: 嵌入式系统课程设计 (子课题1)")
    print("=" * 60)
    print()
    
    run_stationary_test()
    run_motion_test()
    run_qr_parameter_study()
    run_pressure_altitude_curve()
    run_dual_sensor_comparison()
    run_residual_analysis()
    
    print("=" * 60)
    print("所有仿真实验完成！")
    print("生成的图片文件：")
    print("  sim_stationary_compare.png    - 静态滤波对比")
    print("  sim_motion_compare.png        - 运动场景跟踪")
    print("  sim_qr_parameter_study.png    - Q/R参数分析")
    print("  sim_pressure_altitude_curve.png - 气压高度曲线")
    print("  sim_dual_sensor_compare.png   - 双传感器对比")
    print("  sim_residual_analysis.png     - 残差与增益分析")
    print("=" * 60)
