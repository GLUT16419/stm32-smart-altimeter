"""
测试现有融合滤波算法和模型在数据集上的波形情况
模拟嵌入式系统的完整处理流程：KF + NN + EMA + 融合
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os
from pathlib import Path

# ============================================================
# 1. 嵌入式卡尔曼滤波 (与 main.c 中完全一致的算法)
# ============================================================
KF_Q_MIN = 0.1
KF_Q_MAX = 50.0
KF_RESIDUAL_TH = 2.5
KF_Q_INCREASE = 1.08
KF_Q_DECREASE = 0.992
MS5611_KF_Q = 2.8
MS5611_KF_R = 20.0
BMP280_KF_Q = 2.8
BMP280_KF_R = 5.0
KF_INIT_P = 1000.0

class KalmanFilter:
    def __init__(self, init_x, q, r, init_p=KF_INIT_P):
        self.x = init_x
        self.p = init_p
        self.q = q
        self.q_base = q
        self.r = r
        self.k = 0.0
    
    def update_adaptive(self, z):
        residual = z - self.x
        residual_abs = abs(residual)
        
        if residual_abs > KF_RESIDUAL_TH:
            self.q = self.q * KF_Q_INCREASE
            if self.q > KF_Q_MAX:
                self.q = KF_Q_MAX
        else:
            self.q = self.q * KF_Q_DECREASE
            if self.q < self.q_base:
                self.q = self.q_base
        
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * residual
        self.p = (1 - self.k) * self.p
        return self.x

# ============================================================
# 2. NN 滤波 (模拟嵌入式端 NN_Filter_Update)
# ============================================================
class NN_Filter:
    def __init__(self, interpreter, scaler, sensor_type, window_size=10, ref_pressure=101325.0):
        self.interpreter = interpreter
        self.scaler = scaler
        self.sensor_type = sensor_type
        self.window_size = window_size
        self.window = np.zeros(window_size)
        self.index = 0
        self.last_output = 0.0
        self.ref_pressure = ref_pressure
        
        self.nn_rel_min = scaler['min']
        self.nn_rel_max = scaler['max']
        self.nn_rel_range = scaler['range']
        
        # 获取 I/O 细节
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
    
    def init(self, init_x):
        self.window.fill(init_x)
        self.index = 0
        self.last_output = init_x
    
    def update(self, z):
        self.window[self.index] = z
        self.index = (self.index + 1) % self.window_size
        
        # 构建输入缓冲区
        in_buf = np.zeros(self.input_details[0]['shape'], dtype=np.float32)
        for i in range(self.window_size):
            raw = self.window[(self.index + i) % self.window_size]
            rel = raw - self.ref_pressure
            norm = (rel - self.nn_rel_min) / self.nn_rel_range
            norm = np.clip(norm, 0.0, 1.0)
            in_buf[0, i] = norm
        
        # 最后一个元素是传感器类型
        in_buf[0, -1] = self.sensor_type
        
        # 推理
        self.interpreter.set_tensor(self.input_details[0]['index'], in_buf)
        self.interpreter.invoke()
        out_buf = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 反归一化
        self.last_output = (out_buf[0, 0] * self.nn_rel_range + self.nn_rel_min) + self.ref_pressure
        return self.last_output

# ============================================================
# 3. 加载模型
# ============================================================
def load_model(model_path, scaler_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    with open(scaler_path) as f:
        scaler = json.load(f)
    return interpreter, scaler

# ============================================================
# 4. 数据加载与对齐
# ============================================================
def load_aligned_data(folder_path, height_label):
    """加载指定文件夹下的 MS5611 和 BMP280 数据，并对其到相同的 sample_id"""
    ms5611_file = None
    bmp280_file = None
    for f in os.listdir(folder_path):
        if 'ms5611' in f.lower() and f.endswith('.csv'):
            ms5611_file = os.path.join(folder_path, f)
        elif 'bmp280' in f.lower() and f.endswith('.csv'):
            bmp280_file = os.path.join(folder_path, f)
    
    if ms5611_file is None or bmp280_file is None:
        print(f"  [!] Missing files in {folder_path}")
        return None, None
    
    ms5611_df = pd.read_csv(ms5611_file)
    bmp280_df = pd.read_csv(bmp280_file)
    
    # 去重（有些数据有重复 sample_id）
    ms5611_df = ms5611_df.drop_duplicates(subset=['sample_id'])
    bmp280_df = bmp280_df.drop_duplicates(subset=['sample_id'])
    
    # 按 sample_id 对齐
    merged = pd.merge(ms5611_df, bmp280_df, on='sample_id', suffixes=('_ms5611', '_bmp280'))
    
    # 标签中的 label_pressure_pa 实际上是 hPa，需要 *100 转成 Pa
    # 但更好的方法是用高度标签推算参考气压
    # 从数据反推: 参考气压 = measured + height * 11.3
    # 然后用 label_height 计算真值: truth = ref_pressure - label_height * 11.3
    
    return merged.sort_values('sample_id').reset_index(drop=True)

# ============================================================
# 5. 主测试流程
# ============================================================
def test_all_algorithms(data_root='data', output_dir='test_results'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print("Loading models...")
    models_dir = 'models'
    
    # 当前固件使用的 v2_deeper_bp 双传感器模型
    nn_dual_interp, nn_dual_scaler = load_model(
        os.path.join(models_dir, 'dual_sensor_self_supervised_filter_v2_deeper_bp.tflite'),
        os.path.join(models_dir, 'dual_sensor_self_supervised_filter_v2_deeper_bp_scaler.json')
    )
    
    # 单传感器模型
    nn_ms5611_interp, nn_ms5611_scaler = load_model(
        os.path.join(models_dir, 'ms5611_self_supervised_filter_bp.tflite'),
        os.path.join(models_dir, 'ms5611_self_supervised_filter_bp_scaler.json')
    )
    
    nn_bmp280_interp, nn_bmp280_scaler = load_model(
        os.path.join(models_dir, 'bmp280_self_supervised_filter_bp.tflite'),
        os.path.join(models_dir, 'bmp280_self_supervised_filter_bp_scaler.json')
    )
    
    # 收集所有高度文件夹
    height_folders = sorted([d for d in os.listdir(data_root) 
                            if os.path.isdir(os.path.join(data_root, d))])
    
    all_results = []
    
    for folder in height_folders:
        folder_path = os.path.join(data_root, folder)
        height_label = float(folder.split('_')[1].replace('m', ''))
        
        print(f"\n{'='*60}")
        print(f"Testing {folder} (label: {height_label:.2f}m)")
        print(f"{'='*60}")
        
        data = load_aligned_data(folder_path, height_label)
        if data is None or len(data) < 20:
            print(f"  [!] Not enough data, skip")
            continue
        
        print(f"  Loaded {len(data)} aligned samples")
        
        ms5611_p = data['pressure_pa_ms5611'].values
        bmp280_p = data['pressure_pa_bmp280'].values
        ms5611_t = data['temperature_c_ms5611'].values
        bmp280_t = data['temperature_c_bmp280'].values
        
        # 参考气压 (取第一个测量值推算)
        ref_pressure = ms5611_p[0] + height_label * 11.3
        
        # ===========================
        # 算法 1: 原始数据 (无滤波)
        # ===========================
        raw_ms5611 = ms5611_p.copy()
        raw_bmp280 = bmp280_p.copy()
        
        # ===========================
        # 算法 2: 自适应卡尔曼滤波 (KF)
        # ===========================
        kf_ms5611 = KalmanFilter(ms5611_p[0], MS5611_KF_Q, MS5611_KF_R)
        kf_bmp280 = KalmanFilter(bmp280_p[0], BMP280_KF_Q, BMP280_KF_R)
        kf_ms5611_out = np.zeros_like(ms5611_p)
        kf_bmp280_out = np.zeros_like(bmp280_p)
        
        for i in range(len(ms5611_p)):
            kf_ms5611_out[i] = kf_ms5611.update_adaptive(ms5611_p[i])
            kf_bmp280_out[i] = kf_bmp280.update_adaptive(bmp280_p[i])
        
        # KF 融合 (20% MS5611 + 80% BMP280)
        kf_fused = kf_ms5611_out * 0.2 + kf_bmp280_out * 0.8
        
        # ===========================
        # 算法 3: NN (双传感器 v2_deeper_bp, 当前固件在用)
        # ===========================
        nn_dual_ms5611_filter = NN_Filter(nn_dual_interp, nn_dual_scaler, 
                                           nn_dual_scaler['sensor_id_ms5611'])
        nn_dual_bmp280_filter = NN_Filter(nn_dual_interp, nn_dual_scaler,
                                           nn_dual_scaler['sensor_id_bmp280'])
        
        nn_dual_ms5611_filter.init(ms5611_p[0])
        nn_dual_bmp280_filter.init(bmp280_p[0])
        
        nn_dual_ms5611_out = np.zeros_like(ms5611_p)
        nn_dual_bmp280_out = np.zeros_like(bmp280_p)
        
        for i in range(len(ms5611_p)):
            nn_dual_ms5611_out[i] = nn_dual_ms5611_filter.update(ms5611_p[i])
            nn_dual_bmp280_out[i] = nn_dual_bmp280_filter.update(bmp280_p[i])
        
        # ===========================
        # 算法 4: 单传感器 NN
        # ===========================
        nn_single_ms5611_filter = NN_Filter(nn_ms5611_interp, nn_ms5611_scaler, 0.0)
        nn_single_bmp280_filter = NN_Filter(nn_bmp280_interp, nn_bmp280_scaler, 0.0)
        
        nn_single_ms5611_filter.init(ms5611_p[0])
        nn_single_bmp280_filter.init(bmp280_p[0])
        
        nn_single_ms5611_out = np.zeros_like(ms5611_p)
        nn_single_bmp280_out = np.zeros_like(bmp280_p)
        
        for i in range(len(ms5611_p)):
            nn_single_ms5611_out[i] = nn_single_ms5611_filter.update(ms5611_p[i])
            nn_single_bmp280_out[i] = nn_single_bmp280_filter.update(bmp280_p[i])
        
        # ===========================
        # 算法 5: NN + EMA 后处理
        # ===========================
        alpha = 0.4  # EMA 系数
        nn_ema_ms5611_out = np.zeros_like(ms5611_p)
        nn_ema_bmp280_out = np.zeros_like(bmp280_p)
        
        for i in range(len(ms5611_p)):
            if i == 0:
                nn_ema_ms5611_out[i] = nn_dual_ms5611_out[i]
                nn_ema_bmp280_out[i] = nn_dual_bmp280_out[i]
            else:
                nn_ema_ms5611_out[i] = alpha * nn_dual_ms5611_out[i] + (1-alpha) * nn_ema_ms5611_out[i-1]
                nn_ema_bmp280_out[i] = alpha * nn_dual_bmp280_out[i] + (1-alpha) * nn_ema_bmp280_out[i-1]
        
        nn_ema_fused = nn_ema_ms5611_out * 0.2 + nn_ema_bmp280_out * 0.8
        
        # ===========================
        # 算法 7: KF 融合 + NN 融合对比 (当前嵌入式方案)
        # ===========================
        nn_dual_fused = nn_dual_ms5611_out * 0.2 + nn_dual_bmp280_out * 0.2 + \
                        0.0  # 占位，实际融合已经做在上面
        # 重新算正确的融合
        nn_dual_fused = nn_dual_ms5611_out * 0.2 + nn_dual_bmp280_out * 0.8
        nn_single_fused = nn_single_ms5611_out * 0.2 + nn_single_bmp280_out * 0.8
        
        # ===========================
        # 计算指标 - 使用相对压力 (去直流分量，只看噪声)
        # ===========================
        # 实际关心的不是绝对压力值，而是滤波后的噪声水平和波形平滑度
        # 使用信号的均值作为"基准"，计算相对误差
        def calc_metrics_rel(estimated, name):
            """计算相对指标：以自身均值为参考，衡量波动和平滑度"""
            # 波动大小: 信号的标准差 (越小表示越稳定)
            noise_std = np.std(estimated)
            # 平滑度: 相邻差分的标准差 (越小越平滑)
            smoothness = np.std(np.diff(estimated))
            # 峰峰值
            p2p = np.max(estimated) - np.min(estimated)
            return {'name': name, 'noise_std': float(noise_std), 'smoothness': smoothness, 'p2p': p2p}
        
        # 使用 KF 融合结果作为"参考真值"(因为 KF 在标定数据上表现最好)
        kf_target = kf_fused.copy()
        
        metrics = []
        metrics.append(calc_metrics_rel(ms5611_p, 'Raw MS5611'))
        metrics.append(calc_metrics_rel(bmp280_p, 'Raw BMP280'))
        metrics.append(calc_metrics_rel(kf_ms5611_out, 'KF MS5611'))
        metrics.append(calc_metrics_rel(kf_bmp280_out, 'KF BMP280'))
        metrics.append(calc_metrics_rel(kf_fused, 'KF Fused'))
        metrics.append(calc_metrics_rel(nn_dual_ms5611_out, 'NN(dual) MS5611'))
        metrics.append(calc_metrics_rel(nn_dual_bmp280_out, 'NN(dual) BMP280'))
        metrics.append(calc_metrics_rel(nn_dual_fused, 'NN(dual) Fused'))
        metrics.append(calc_metrics_rel(nn_single_fused, 'NN(single) Fused'))
        metrics.append(calc_metrics_rel(nn_ema_fused, 'NN+EMA Fused'))
        
        print(f"\n  {'Algorithm':30s} {'波动Std':>10s} {'Smooth':>10s} {'P2P':>10s}")
        print(f"  {'-'*60}")
        for m in metrics:
            print(f"  {m['name']:30s} {m['noise_std']:10.4f} {m['smoothness']:10.4f} {m['p2p']:10.4f}")
        
        all_results.append({
            'folder': folder,
            'height': height_label,
            'metrics': metrics,
            'data': data,
            'kf_ms5611': kf_ms5611_out,
            'kf_bmp280': kf_bmp280_out,
            'kf_fused': kf_fused,
            'nn_dual_ms5611': nn_dual_ms5611_out,
            'nn_dual_bmp280': nn_dual_bmp280_out,
            'nn_dual_fused': nn_dual_fused,
            'nn_single_fused': nn_single_fused,
            'nn_ema_fused': nn_ema_fused,
            'ref_pressure': ref_pressure,
        })
    
    return all_results


# ============================================================
# 6. 可视化
# ============================================================
def plot_results(all_results, output_dir='test_results'):
    # ---- 全局汇总表 ----
    summary_rows = []
    for r in all_results:
        for m in r['metrics']:
            summary_rows.append({
                'height': r['height'],
                'algorithm': m['name'],
                'noise_std': m['noise_std'],
                'smoothness': m['smoothness'],
                'p2p': m['p2p']
            })
    
    summary_df = pd.DataFrame(summary_rows)
    print("\n\n========== 全局汇总 ==========")
    pivot = summary_df.groupby('algorithm').agg({
        'noise_std': ['mean', 'std'],
        'smoothness': ['mean', 'std'],
        'p2p': ['mean', 'std']
    }).round(4)
    print(pivot.to_string())
    pivot.to_csv(os.path.join(output_dir, 'summary_metrics.csv'))
    
    # ---- 各高度对比图 ----
    algorithms_to_plot = [
        ('KF Fused', 'b-', 'KF Fused'),
        ('NN(dual) Fused', 'r-', 'NN(dual) Fused'),
        ('NN+EMA Fused', 'g-', 'NN+EMA Fused'),
        ('NN(single) Fused', 'm-', 'NN(single) Fused'),
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, r in enumerate(all_results):
        if idx >= len(axes):
            break
        ax = axes[idx]
        data = r['data']
        x = np.arange(len(data))
        
        # 原始数据
        ax.plot(x, data['pressure_pa_ms5611'].values, 'gray', alpha=0.3, label='Raw MS5611', linewidth=0.5)
        ax.plot(x, data['pressure_pa_bmp280'].values, 'gray', alpha=0.2, label='Raw BMP280', linewidth=0.5)
        
        # 各算法 - 使用相对压力(去均值)展示波形
        ms5611_mean = np.mean(data['pressure_pa_ms5611'].values)
        bmp280_mean = np.mean(data['pressure_pa_bmp280'].values)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Mean')
        
        for algo_name, style, legend_name in algorithms_to_plot:
            if algo_name == 'KF Fused':
                ax.plot(x, r['kf_fused'] - ms5611_mean, style, label=legend_name, linewidth=1.5)
            elif algo_name == 'NN(dual) Fused':
                ax.plot(x, r['nn_dual_fused'] - ms5611_mean, style, label=legend_name, linewidth=1.5)
            elif algo_name == 'NN+EMA Fused':
                ax.plot(x, r['nn_ema_fused'] - ms5611_mean, style, label=legend_name, linewidth=1.5)
            elif algo_name == 'NN(single) Fused':
                ax.plot(x, r['nn_single_fused'] - ms5611_mean, style, label=legend_name, linewidth=1.5)
        
        ax.set_title(f"Height: {r['height']:.1f}m (relative to mean)")
        ax.set_xlabel('Sample')
        ax.set_ylabel('Pressure Deviation (Pa)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_heights_comparison.png'), dpi=150)
    plt.close()
    
    # ---- 选择 3 个高度做详细对比 ----
    selected = [0, len(all_results)//2, len(all_results)-1]
    if len(all_results) > 0:
        fig, axes = plt.subplots(len(selected), 2, figsize=(16, 4*len(selected)))
        
        for row, idx in enumerate(selected):
            if idx >= len(all_results):
                continue
            r = all_results[idx]
            data = r['data']
            x = np.arange(len(data))
            
            # MS5611 侧 - 相对压力
            ax1 = axes[row][0]
            ms5611_mean = np.mean(data['pressure_pa_ms5611'].values)
            ax1.plot(x, data['pressure_pa_ms5611'].values - ms5611_mean, 'gray', alpha=0.3, label='Raw', linewidth=0.5)
            ax1.plot(x, r['kf_ms5611'] - ms5611_mean, 'b-', label='KF', linewidth=1.5)
            ax1.plot(x, r['nn_dual_ms5611'] - ms5611_mean, 'r-', label='NN(dual)', linewidth=1.5)
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax1.set_title(f"MS5611 @ {r['height']:.1f}m (relative)")
            ax1.set_ylabel('Pressure Deviation (Pa)')
            ax1.legend(fontsize=7)
            ax1.grid(True, alpha=0.3)
            
            # BMP280 侧 - 相对压力
            ax2 = axes[row][1]
            bmp280_mean = np.mean(data['pressure_pa_bmp280'].values)
            ax2.plot(x, data['pressure_pa_bmp280'].values - bmp280_mean, 'gray', alpha=0.3, label='Raw', linewidth=0.5)
            ax2.plot(x, r['kf_bmp280'] - bmp280_mean, 'b-', label='KF', linewidth=1.5)
            ax2.plot(x, r['nn_dual_bmp280'] - bmp280_mean, 'r-', label='NN(dual)', linewidth=1.5)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax2.set_title(f"BMP280 @ {r['height']:.1f}m (relative)")
            ax2.set_ylabel('Pressure Deviation (Pa)')
            ax2.legend(fontsize=7)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_sensor_detail.png'), dpi=150)
        plt.close()
    
    # ---- 波动 vs 平滑度 条形图 ----
    fig, ax = plt.subplots(figsize=(14, 6))
    algo_names = [m['name'] for m in all_results[0]['metrics']]
    algo_波动 = [np.mean([r['metrics'][i]['noise_std'] for r in all_results]) for i in range(len(algo_names))]
    algo_smooth = [np.mean([r['metrics'][i]['smoothness'] for r in all_results]) for i in range(len(algo_names))]
    
    x_pos = np.arange(len(algo_names))
    bars = ax.bar(x_pos - 0.2, algo_波动, 0.35, label='Noise Std (Pa)', color='coral')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + 0.2, algo_smooth, 0.35, label='Smoothness (Pa)', color='skyblue')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algo_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Noise Std (Pa)')
    ax2.set_ylabel('Smoothness (Pa)')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax.set_title('Algorithm Performance Comparison (All Heights Average)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison_bar.png'), dpi=150)
    plt.close()
    
    # ---- 精度 vs 平滑度散点图 ----
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(algo_names)))
    for i, name in enumerate(algo_names):
        noise_vals = [r['metrics'][i]['noise_std'] for r in all_results]
        smooth_vals = [r['metrics'][i]['smoothness'] for r in all_results]
        ax.scatter(noise_vals, smooth_vals, c=[colors[i]], label=name, s=80, alpha=0.7)
    
    ax.set_xlabel('Noise Std (Pa) - 噪声水平（越小越好）')
    ax.set_ylabel('Smoothness (Pa) - 平滑度（越小越好）')
    ax.set_title('Noise vs Smoothness Trade-off')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_vs_smoothness.png'), dpi=150)
    plt.close()
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - summary_metrics.csv")
    print(f"  - all_heights_comparison.png")
    print(f"  - per_sensor_detail.png")
    print(f"  - algorithm_comparison_bar.png")
    print(f"  - precision_vs_smoothness.png")


if __name__ == '__main__':
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else 'data'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'test_results'
    
    print("=" * 70)
    print("嵌入式气压传感器融合滤波算法测试")
    print("模拟嵌入式端完整处理流程")
    print("=" * 70)
    
    results = test_all_algorithms(data_root, output_dir)
    if results:
        plot_results(results, output_dir)
        print("\nDone!")
    else:
        print("\nNo results to plot.")
