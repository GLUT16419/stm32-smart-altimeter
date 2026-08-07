"""
卡尔曼滤波参数优化仿真
对比新旧参数在不同传感器上的效果，并辅助调参
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ============================================================
# 1. 加载数据
# ============================================================
def load_data(filepath, label=""):
    """加载自监督训练数据，返回气压数组（第二列）"""
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    pressure = data[:, 1]  # 第二列是气压(Pa)，第一列是sample_id
    print(f"[{label}] Loaded {len(pressure)} samples: "
          f"min={pressure.min():.2f}, max={pressure.max():.2f}, "
          f"std={pressure.std():.3f}")
    return pressure

# ============================================================
# 2. 卡尔曼滤波实现（C代码的精确对位）
# ============================================================
class KalmanFilter:
    def __init__(self, init_x, init_p, q, r):
        self.x = init_x
        self.p = init_p
        self.q = q
        self.q_base = q  # 自适应恢复基准
        self.r = r
        self.k = 0.0
    
    def update(self, z):
        """标准卡尔曼滤波"""
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (z - self.x)
        self.p = (1 - self.k) * self.p
        return self.x
    
    def update_adaptive(self, z, residual_th=5.0, q_inc=1.05, q_dec=0.995, q_min=0.1, q_max=50.0):
        """自适应卡尔曼滤波（与 C 代码一致）"""
        residual = z - self.x
        residual_abs = abs(residual)
        
        # 自适应调整 Q
        if residual_abs > residual_th:
            self.q = self.q * q_inc
            if self.q > q_max:
                self.q = q_max
        else:
            self.q = self.q * q_dec
            if self.q < self.q_base:
                self.q = self.q_base
        
        # 标准卡尔曼更新
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * residual
        self.p = (1 - self.k) * self.p
        
        return self.x, self.q  # 返回 Q 用于分析

# ============================================================
# 3. 对比实验
# ============================================================
def run_comparison(data, sensor_name, old_q, old_r, new_q, new_r,
                   use_adaptive=True, label=""):
    """
    运行新旧参数对比
    Returns: (old_filtered, new_filtered, adaptive_qs)
    """
    n = len(data)
    
    # 旧参数 (Q=5, R=100)
    kf_old = KalmanFilter(data[0], 1000.0, old_q, old_r)
    old_filt = np.array([kf_old.update(z) for z in data])
    
    # 新参数 + 自适应
    kf_new = KalmanFilter(data[0], 1000.0, new_q, new_r)
    new_filt = np.zeros(n)
    adaptive_qs = np.zeros(n)
    
    for i, z in enumerate(data):
        if use_adaptive:
            val, q = kf_new.update_adaptive(z)
            new_filt[i] = val
            adaptive_qs[i] = q
        else:
            new_filt[i] = kf_new.update(z)
            adaptive_qs[i] = kf_new.q
    
    # 统计
    old_std = np.std(data - old_filt)
    new_std = np.std(data - new_filt)
    
    # Savitzky-Golay 作为"理想"参考
    sg = savgol_filter(data, window_length=7, polyorder=3, mode='mirror')
    sg_std = np.std(data - sg)
    
    print(f"\n{'='*60}")
    print(f"[{sensor_name}] {label}")
    print(f"{'='*60}")
    print(f"  Raw std:           {data.std():.4f}")
    print(f"  Old KF (Q={old_q}, R={old_r}):  std={old_std:.4f}")
    print(f"  New KF (Q={new_q}, R={new_r}):  std={new_std:.4f}")
    print(f"  Savgol ref:        std={sg_std:.4f}")
    print(f"  Improvement:       {(old_std - new_std) / old_std * 100:.1f}%")
    
    return old_filt, new_filt, adaptive_qs, sg


def plot_comparison(data, old_filt, new_filt, adaptive_qs, sg,
                    sensor_name, label=""):
    """绘制对比图"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 子图1：原始数据 + 滤波对比
    ax = axes[0]
    ax.plot(data, 'gray', alpha=0.3, linewidth=0.8, label='Raw')
    ax.plot(old_filt, 'orange', alpha=0.7, linewidth=1.0, label=f'Old KF')
    ax.plot(new_filt, 'green', linewidth=1.2, label=f'New KF (Adaptive)')
    ax.plot(sg, 'red', alpha=0.5, linewidth=1.0, linestyle='--', label='Savgol ref')
    ax.set_ylabel('Pressure (Pa)')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{sensor_name} - {label} - Kalman Filter Comparison')
    
    # 子图2：残差对比
    ax = axes[1]
    ax.plot(data - old_filt, 'orange', alpha=0.6, linewidth=0.8, label='Old KF residual')
    ax.plot(data - new_filt, 'green', alpha=0.6, linewidth=0.8, label='New KF residual')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Residual (Pa)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Filter Residuals (smaller is better)')
    
    # 子图3：自适应 Q 值变化
    ax = axes[2]
    ax.plot(adaptive_qs, 'purple', linewidth=1.0)
    ax.axhline(y=adaptive_qs[0], color='gray', linestyle='--', alpha=0.5,
               label=f'Base Q={adaptive_qs[0]:.1f}')
    ax.set_ylabel('Q value')
    ax.set_xlabel('Sample index')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Adaptive Q value over time')
    
    plt.tight_layout()
    
    # 保存
    output_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'kalman_comparison_{sensor_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def scan_params(data, sensor_name, q_range, r_range):
    """参数扫描：找最佳 (Q, R) 组合"""
    best_std = float('inf')
    best_q, best_r = q_range[0], r_range[0]
    
    results = []
    for q in q_range:
        for r in r_range:
            kf = KalmanFilter(data[0], 1000.0, q, r)
            filt = np.array([kf.update(z) for z in data])
            residual_std = np.std(data - filt)
            results.append((q, r, residual_std))
            
            if residual_std < best_std:
                best_std = residual_std
                best_q, best_r = q, r
    
    print(f"\n[{sensor_name}] Parameter scan results:")
    print(f"  Best: Q={best_q:.1f}, R={best_r:.1f}, residual_std={best_std:.4f}")
    
    # 显示 top 5
    results.sort(key=lambda x: x[2])
    print("  Top 5:")
    for q, r, s in results[:5]:
        print(f"    Q={q:.1f}, R={r:.1f} -> std={s:.4f}")
    
    return best_q, best_r, results


# ============================================================
# 4. 主流程
# ============================================================
def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # 收集所有 MS5611 和 BMP280 的数据文件
    ms5611_files = []
    bmp280_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.csv'):
                path = os.path.join(root, f)
                if 'ms5611' in f.lower():
                    ms5611_files.append(path)
                elif 'bmp280' in f.lower():
                    bmp280_files.append(path)
    
    print(f"Found {len(ms5611_files)} MS5611 files, {len(bmp280_files)} BMP280 files")
    
    # 加载所有数据
    ms5611_all = np.concatenate([load_data(f, f"MS5611-{os.path.basename(f)}") 
                                  for f in ms5611_files]) if ms5611_files else np.array([])
    bmp280_all = np.concatenate([load_data(f, f"BMP280-{os.path.basename(f)}") 
                                  for f in bmp280_files]) if bmp280_files else np.array([])
    
    # ============================================================
    # 4.1 参数扫描（找最优参数）
    # ============================================================
    print("\n" + "="*70)
    print("PARAMETER SCAN")
    print("="*70)
    
    # MS5611: 用更小的 Q 范围（强调平滑），自适应机制负责跟踪变化
    if len(ms5611_all) > 0:
        ms5611_q_vals = np.arange(0.1, 3.1, 0.3)
        ms5611_r_vals = [20, 30, 40, 50, 60, 80, 100]
        best_ms5611_q, best_ms5611_r, _ = scan_params(
            ms5611_all[:2000], "MS5611", ms5611_q_vals, ms5611_r_vals)
    else:
        best_ms5611_q, best_ms5611_r = 1.0, 40.0
    
    # BMP280: 噪声小，R 可以更小，Q 保持适中
    if len(bmp280_all) > 0:
        bmp280_q_vals = np.arange(0.1, 3.1, 0.3)
        bmp280_r_vals = [5, 8, 10, 15, 20, 30, 50]
        best_bmp280_q, best_bmp280_r, _ = scan_params(
            bmp280_all[:2000], "BMP280", bmp280_q_vals, bmp280_r_vals)
    else:
        best_bmp280_q, best_bmp280_r = 2.0, 10.0
    
    # ============================================================
    # 4.2 对比实验 + 可视化
    # ============================================================
    print("\n" + "="*70)
    print("COMPARISON EXPERIMENTS")
    print("="*70)
    
    # 取一段数据做可视化（取中间一段，避开起始瞬态）
    n_plot = 300
    
    if len(ms5611_all) > 0:
        mid = len(ms5611_all) // 2
        ms5611_seg = ms5611_all[mid:mid+n_plot]
        
        old_filt_m, new_filt_m, qs_m, sg_m = run_comparison(
            ms5611_seg, "MS5611",
            old_q=5.0, old_r=100.0,
            new_q=best_ms5611_q, new_r=best_ms5611_r,
            label=f"Old Q=5/R=100 → New Q={best_ms5611_q:.1f}/R={best_ms5611_r:.0f}")
        plot_comparison(ms5611_seg, old_filt_m, new_filt_m, qs_m, sg_m,
                        "MS5611",
                        f"Old Q=5/R=100 → New Q={best_ms5611_q:.1f}/R={best_ms5611_r:.0f}")
    
    if len(bmp280_all) > 0:
        mid = len(bmp280_all) // 2
        bmp280_seg = bmp280_all[mid:mid+n_plot]
        
        old_filt_b, new_filt_b, qs_b, sg_b = run_comparison(
            bmp280_seg, "BMP280",
            old_q=5.0, old_r=100.0,
            new_q=best_bmp280_q, new_r=best_bmp280_r,
            label=f"Old Q=5/R=100 → New Q={best_bmp280_q:.1f}/R={best_bmp280_r:.0f}")
        plot_comparison(bmp280_seg, old_filt_b, new_filt_b, qs_b, sg_b,
                        "BMP280",
                        f"Old Q=5/R=100 → New Q={best_bmp280_q:.1f}/R={best_bmp280_r:.0f}")
    
    # ============================================================
    # 4.3 自适应 vs 非自适应对比
    # ============================================================
    print("\n" + "="*70)
    print("ADAPTIVE vs NON-ADAPTIVE COMPARISON")
    print("="*70)
    
    if len(ms5611_all) > 0:
        kf_non = KalmanFilter(ms5611_seg[0], 1000.0, best_ms5611_q, best_ms5611_r)
        kf_adp = KalmanFilter(ms5611_seg[0], 1000.0, best_ms5611_q, best_ms5611_r)
        
        non_filt = np.array([kf_non.update(z) for z in ms5611_seg])
        adp_filt = np.zeros(n_plot)
        for i, z in enumerate(ms5611_seg):
            adp_filt[i], _ = kf_adp.update_adaptive(z)
        
        non_std = np.std(ms5611_seg - non_filt)
        adp_std = np.std(ms5611_seg - adp_filt)
        print(f"\n  MS5611: Non-adaptive std={non_std:.4f}, Adaptive std={adp_std:.4f}")
    
    print("\n" + "="*70)
    print("RECOMMENDED MCU PARAMETERS")
    print("="*70)
    print(f"""
  /* MS5611: 噪声较大(±4~5Pa)，小 Q 加强平滑 */
  KalmanFilter_Init(&ms5611_kf_pressure, ms5611_p, 1000.0f, {best_ms5611_q:.1f}f, {best_ms5611_r:.0f}.0f);

  /* BMP280: 噪声较小(±1~2Pa)，Q 适中保持响应 */
  KalmanFilter_Init(&bmp280_kf_pressure, bmp280_p, 1000.0f, {best_bmp280_q:.1f}f, {best_bmp280_r:.0f}.0f);
  
  /* 运行阶段使用 KalmanFilter_Update_Adaptive() 实现自适应滤波 */
""")
    print("Done!")


if __name__ == "__main__":
    main()
