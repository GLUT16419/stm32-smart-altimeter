#!/usr/bin/env python
"""
气压传感器校准数据分析工具

功能：
1. 读取 PC 端采集的传感器数据 CSV
2. 统计分析（均值、方差、标准差、极值、异常值检测）
3. 分块统计（评估短期噪声）
4. 双传感器一致性分析
5. 自动生成 KF/EKF/融合权重参数建议
6. 可视化输出

用法：
    python calibration_analyzer.py --data data/raw/160m/ms5611_*.csv --sensor ms5611
    python calibration_analyzer.py --data data/raw/160m/ --all  # 扫描目录下所有 CSV
    python calibration_analyzer.py --ms5611 ms5611.csv --bmp280 bmp280.csv  # 双传感器分析
"""

import os, sys, json, argparse, glob, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== 配置参数（与 MCU 端一致） ====================
CALIB_PHASE1_SAMPLES = 50      # 阶段1快速采样数
CALIB_PHASE2_SAMPLES = 300     # 阶段2精细采样数
CALIB_STD_BLOCK_SIZE = 30      # 分块标准差块大小
CALIB_OUTLIER_STD_TH = 3.0     # 异常值阈值（σ倍数）
KF_R_FACTOR = 1.0              # R = 噪声方差 * 系数
KF_Q_BASE_FACTOR = 0.05        # Q = 噪声标准差 * 系数
FUSION_WEIGHT_MIN = 0.1
FUSION_WEIGHT_MAX = 0.9

SENSOR_HEALTH_GOOD = 2
SENSOR_HEALTH_FAIR = 1
SENSOR_HEALTH_POOR = 0


def read_csv_data(filepath, sensor_name=None):
    """读取 CSV 数据文件，返回 pressure 数组和 metadata"""
    if not os.path.exists(filepath):
        print(f"[ERROR] 文件不存在: {filepath}")
        return None, None

    data = np.genfromtxt(filepath, delimiter=',', dtype=None, names=True, encoding='utf-8', invalid_raise=False)

    if data is None or len(data) == 0:
        # 尝试手动解析
        pressures = []
        temperatures = []
        heights = []
        with open(filepath, 'r') as f:
            header = f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 5:
                    try:
                        if sensor_name and parts[0] != sensor_name:
                            continue
                        p = float(parts[2])
                        t = float(parts[3])
                        h = float(parts[4])
                        if 0 < p < 200000:
                            pressures.append(p)
                            temperatures.append(t)
                            heights.append(h)
                    except ValueError:
                        continue

        if not pressures:
            print(f"[ERROR] 无法解析数据: {filepath}")
            return None, None
        return np.array(pressures), {'temp': np.array(temperatures), 'height': np.array(heights)}

    # 用 genfromtxt 成功读取
    try:
        pressures = data['pressure_pa']
        if pressures is None or len(pressures) == 0:
            return None, None

        # 过滤无效值
        mask = (pressures > 0) & (pressures < 200000)
        pressures = pressures[mask]

        metadata = {}
        try:
            metadata['temp'] = data['temperature_c'][mask]
        except (IndexError, ValueError):
            metadata['temp'] = None
        try:
            metadata['height'] = data['height_m'][mask]
        except (IndexError, ValueError):
            metadata['height'] = None

        return pressures, metadata
    except (ValueError, IndexError, KeyError):
        return None, None


def calc_statistics(data, outlier_th=CALIB_OUTLIER_STD_TH):
    """计算统计量（与 MCU 端 CalibStats 一致）"""
    if data is None or len(data) < 2:
        return None

    n = len(data)
    mean = np.mean(data)
    variance = np.var(data, ddof=1)  # 无偏估计
    std = np.std(data, ddof=1)
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val

    # 异常值检测
    outliers = data[np.abs(data - mean) > outlier_th * std]
    outlier_count = len(outliers)
    outlier_ratio = outlier_count / n

    # 分块标准差（评估短期噪声）
    block_stds = []
    for i in range(0, n, CALIB_STD_BLOCK_SIZE):
        block = data[i:i + CALIB_STD_BLOCK_SIZE]
        if len(block) >= CALIB_STD_BLOCK_SIZE // 2:
            block_stds.append(np.std(block, ddof=1))

    mean_block_std = np.mean(block_stds) if block_stds else std

    # 健康状态
    if std < 3.0 and outlier_ratio < 0.05:
        health = SENSOR_HEALTH_GOOD
    elif std < 8.0 and outlier_ratio < 0.15:
        health = SENSOR_HEALTH_FAIR
    else:
        health = SENSOR_HEALTH_POOR

    return {
        'count': n,
        'mean': mean,
        'variance': variance,
        'std': std,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'outlier_count': outlier_count,
        'outlier_ratio': outlier_ratio,
        'block_stds': block_stds,
        'mean_block_std': mean_block_std,
        'health': health,
        'health_str': ['POOR', 'FAIR', 'GOOD'][health],
    }


def auto_tune_kf(ms5611_stats, bmp280_stats=None):
    """自动调参 KF（与 MCU 端逻辑一致）"""
    result = {}

    # MS5611 调参
    ms_std = ms5611_stats['std']
    ms_var = ms5611_stats['variance']
    result['kf_r_ms5611'] = max(0.5, min(50.0, ms_var * KF_R_FACTOR))
    result['kf_q_ms5611'] = max(0.01, min(5.0, ms_std * KF_Q_BASE_FACTOR))

    if bmp280_stats:
        bm_std = bmp280_stats['std']
        bm_var = bmp280_stats['variance']
        result['kf_r_bmp280'] = max(0.1, min(30.0, bm_var * KF_R_FACTOR))
        result['kf_q_bmp280'] = max(0.005, min(3.0, bm_std * KF_Q_BASE_FACTOR))

        # 融合权重（反比加权）
        if ms_std > 0 and bm_std > 0:
            inv_ms = 1.0 / ms_std
            inv_bm = 1.0 / bm_std
            total_inv = inv_ms + inv_bm
            result['fusion_weight_ms5611'] = max(FUSION_WEIGHT_MIN,
                                                  min(FUSION_WEIGHT_MAX, inv_ms / total_inv))
            result['fusion_weight_bmp280'] = 1.0 - result['fusion_weight_ms5611']
        else:
            result['fusion_weight_ms5611'] = 0.2
            result['fusion_weight_bmp280'] = 0.8

        # EKF 噪声
        result['ekf_accel_sigma'] = max(0.5, min(5.0, ms_std * 0.5))
        result['ekf_pressure_sigma'] = max(1.0, min(10.0, ms_std))
    else:
        result['kf_r_bmp280'] = None
        result['kf_q_bmp280'] = None
        result['fusion_weight_ms5611'] = 1.0
        result['fusion_weight_bmp280'] = 0.0
        result['ekf_accel_sigma'] = max(0.5, min(5.0, ms_std * 0.5))
        result['ekf_pressure_sigma'] = max(1.0, min(10.0, ms_std))

    return result


def simulate_kalman(data, q, r, init_p=1000.0):
    """模拟一维卡尔曼滤波（与 MCU 端一致）"""
    n = len(data)
    x = np.zeros(n)
    p = init_p

    for i in range(n):
        # Predict
        p = p + q
        # Update
        k = p / (p + r)
        x[i] = (x[i-1] if i > 0 else data[i]) + k * (data[i] - (x[i-1] if i > 0 else data[i]))
        p = (1 - k) * p

    return x


def analyze_dual_sensor(ms5611_data, bmp280_data):
    """双传感器一致性分析"""
    n = min(len(ms5611_data), len(bmp280_data))
    diff = ms5611_data[:n] - bmp280_data[:n]

    # 过滤合理差值
    mask = np.abs(diff) < 500
    diff = diff[mask]

    diff_mean = np.mean(diff)
    diff_std = np.std(diff, ddof=1)

    # 一致性评分
    if diff_std < 2.0:
        consistency = 1.0
    elif diff_std < 5.0:
        consistency = 0.8
    elif diff_std < 10.0:
        consistency = 0.5
    else:
        consistency = 0.2

    return {
        'diff_mean': diff_mean,
        'diff_std': diff_std,
        'consistency': consistency,
        'consistency_str': f"{consistency:.2f}",
    }


def generate_report(ms_stats, bm_stats, dual_result, kf_params, filepath_base):
    """生成校准分析报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("气压传感器校准分析报告")
    lines.append("=" * 60)
    lines.append(f"分析时间: {np.datetime64('now')}")
    lines.append(f"文件: {filepath_base}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("一、传感器统计")
    lines.append("-" * 60)

    for name, stats in [("MS5611", ms_stats), ("BMP280", bm_stats)]:
        if stats:
            lines.append(f"  [{name}]")
            lines.append(f"    样本数:    {stats['count']}")
            lines.append(f"    均值:      {stats['mean']:.4f} Pa")
            lines.append(f"    标准差:    {stats['std']:.4f} Pa  (对应高度: {stats['std']/11.3:.4f} m)")
            lines.append(f"    方差:      {stats['variance']:.4f}")
            lines.append(f"    最小值:    {stats['min']:.4f} Pa")
            lines.append(f"    最大值:    {stats['max']:.4f} Pa")
            lines.append(f"    极差:      {stats['range']:.4f} Pa")
            lines.append(f"    异常值:    {stats['outlier_count']} ({stats['outlier_ratio']*100:.1f}%)")
            lines.append(f"    分块标准差均值: {stats['mean_block_std']:.4f} Pa")
            lines.append(f"    健康状态:  {stats['health_str']}")
            lines.append("")

    if dual_result:
        lines.append("-" * 60)
        lines.append("二、双传感器一致性分析")
        lines.append("-" * 60)
        lines.append(f"  差值均值:    {dual_result['diff_mean']:.4f} Pa")
        lines.append(f"  差值标准差:  {dual_result['diff_std']:.4f} Pa")
        lines.append(f"  一致性评分:  {dual_result['consistency_str']}")
        lines.append("")

    lines.append("-" * 60)
    lines.append("三、自动调参建议")
    lines.append("-" * 60)

    lines.append(f"  KF_MS5611:   Q = {kf_params['kf_q_ms5611']:.4f},  R = {kf_params['kf_r_ms5611']:.2f}")
    if kf_params['kf_r_bmp280'] is not None:
        lines.append(f"  KF_BMP280:   Q = {kf_params['kf_q_bmp280']:.4f},  R = {kf_params['kf_r_bmp280']:.2f}")
        lines.append(f"  融合权重:    MS5611 = {kf_params['fusion_weight_ms5611']:.3f},  "
                     f"BMP280 = {kf_params['fusion_weight_bmp280']:.3f}")
    lines.append(f"  EKF:         accel_sigma = {kf_params['ekf_accel_sigma']:.2f},  "
                 f"pressure_sigma = {kf_params['ekf_pressure_sigma']:.2f}")
    lines.append("")

    # 与默认参数对比
    lines.append("-" * 60)
    lines.append("四、与默认参数对比")
    lines.append("-" * 60)
    lines.append("  参数             | 默认值   | 建议值")
    lines.append("  " + "-" * 45)
    lines.append(f"  KF_MS5611 Q      | 0.1000   | {kf_params['kf_q_ms5611']:.4f}")
    lines.append(f"  KF_MS5611 R      | 4.00     | {kf_params['kf_r_ms5611']:.2f}")
    if kf_params['kf_r_bmp280'] is not None:
        lines.append(f"  KF_BMP280 Q      | 0.0500   | {kf_params['kf_q_bmp280']:.4f}")
        lines.append(f"  KF_BMP280 R      | 1.00     | {kf_params['kf_r_bmp280']:.2f}")
        lines.append(f"  Fusion MS5611    | 0.20     | {kf_params['fusion_weight_ms5611']:.2f}")
        lines.append(f"  Fusion BMP280    | 0.80     | {kf_params['fusion_weight_bmp280']:.2f}")
    lines.append("")

    return "\n".join(lines)


def plot_analysis(ms5611_data, bmp280_data, ms_stats, bm_stats, kf_params, output_path):
    """生成分析图表"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('气压传感器校准分析', fontsize=14)

    # 1. MS5611 原始数据
    ax = axes[0, 0]
    ax.plot(ms5611_data, 'b-', alpha=0.7, linewidth=0.5)
    ax.axhline(y=ms_stats['mean'], color='r', linestyle='--', label=f"均值={ms_stats['mean']:.1f}")
    ax.axhline(y=ms_stats['mean'] + ms_stats['std'], color='orange', linestyle=':', label=f"±1σ={ms_stats['std']:.2f}")
    ax.axhline(y=ms_stats['mean'] - ms_stats['std'], color='orange', linestyle=':')
    ax.set_title(f"MS5611 原始气压 (σ={ms_stats['std']:.2f} Pa)")
    ax.set_xlabel('采样点')
    ax.set_ylabel('气压 (Pa)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. BMP280 原始数据
    ax = axes[0, 1]
    if bmp280_data is not None:
        ax.plot(bmp280_data, 'g-', alpha=0.7, linewidth=0.5)
        ax.axhline(y=bm_stats['mean'], color='r', linestyle='--', label=f"均值={bm_stats['mean']:.1f}")
        ax.axhline(y=bm_stats['mean'] + bm_stats['std'], color='orange', linestyle=':', label=f"±1σ={bm_stats['std']:.2f}")
        ax.axhline(y=bm_stats['mean'] - bm_stats['std'], color='orange', linestyle=':')
        ax.set_title(f"BMP280 原始气压 (σ={bm_stats['std']:.2f} Pa)")
    else:
        ax.text(0.5, 0.5, '无 BMP280 数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("BMP280")
    ax.set_xlabel('采样点')
    ax.set_ylabel('气压 (Pa)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. MS5611 直方图
    ax = axes[1, 0]
    ax.hist(ms5611_data, bins=50, color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=ms_stats['mean'], color='r', linestyle='--', label=f"均值={ms_stats['mean']:.1f}")
    ax.axvline(x=ms_stats['mean'] + ms_stats['std'], color='orange', linestyle=':', label=f"±1σ")
    ax.axvline(x=ms_stats['mean'] - ms_stats['std'], color='orange', linestyle=':')
    ax.set_title(f"MS5611 气压分布 (σ={ms_stats['std']:.2f} Pa)")
    ax.set_xlabel('气压 (Pa)')
    ax.set_ylabel('频数')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. BMP280 直方图
    ax = axes[1, 1]
    if bmp280_data is not None:
        ax.hist(bmp280_data, bins=50, color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(x=bm_stats['mean'], color='r', linestyle='--', label=f"均值={bm_stats['mean']:.1f}")
        ax.axvline(x=bm_stats['mean'] + bm_stats['std'], color='orange', linestyle=':', label=f"±1σ")
        ax.axvline(x=bm_stats['mean'] - bm_stats['std'], color='orange', linestyle=':')
        ax.set_title(f"BMP280 气压分布 (σ={bm_stats['std']:.2f} Pa)")
    else:
        ax.text(0.5, 0.5, '无 BMP280 数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("BMP280 分布")
    ax.set_xlabel('气压 (Pa)')
    ax.set_ylabel('频数')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. KF 滤波效果对比 (MS5611)
    ax = axes[2, 0]
    kf_q = kf_params['kf_q_ms5611']
    kf_r = kf_params['kf_r_ms5611']
    kf_filtered = simulate_kalman(ms5611_data, kf_q, kf_r)

    ax.plot(ms5611_data[:500], 'b-', alpha=0.4, linewidth=0.5, label='原始')
    ax.plot(kf_filtered[:500], 'r-', linewidth=1.5, label=f'KF (Q={kf_q:.4f}, R={kf_r:.2f})')
    ax.set_title(f"KF 滤波效果对比 (前500点)")
    ax.set_xlabel('采样点')
    ax.set_ylabel('气压 (Pa)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. 双传感器差值
    ax = axes[2, 1]
    if bmp280_data is not None:
        n = min(len(ms5611_data), len(bmp280_data))
        diff = ms5611_data[:n] - bmp280_data[:n]
        ax.plot(diff, 'purple', alpha=0.5, linewidth=0.5)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_title(f"双传感器差值 (σ={np.std(diff):.2f} Pa)")
    else:
        ax.text(0.5, 0.5, '无双传感器数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("双传感器差值")
    ax.set_xlabel('采样点')
    ax.set_ylabel('气压差 (Pa)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] 图表已保存: {output_path}")
    plt.close()


def generate_mcu_header(kf_params, output_path):
    """生成 MCU 端可直接使用的头文件片段"""
    lines = []
    lines.append("/* ====================================================")
    lines.append(" * 自动生成的校准参数 — 从实际传感器数据分析获得")
    lines.append(" * 将此内容复制到 kalman_filter.h / main.h 中")
    lines.append(" * ====================================================")
    lines.append(" */")
    lines.append("")
    lines.append("#ifndef __CALIB_AUTO_PARAMS_H__")
    lines.append("#define __CALIB_AUTO_PARAMS_H__")
    lines.append("")

    lines.append("/* ========== 卡尔曼滤波参数（自动调参） ========== */")
    lines.append(f"#define MS5611_KF_Q      {kf_params['kf_q_ms5611']:.4f}f")
    lines.append(f"#define MS5611_KF_R      {kf_params['kf_r_ms5611']:.1f}f")
    if kf_params['kf_r_bmp280'] is not None:
        lines.append(f"#define BMP280_KF_Q      {kf_params['kf_q_bmp280']:.4f}f")
        lines.append(f"#define BMP280_KF_R      {kf_params['kf_r_bmp280']:.1f}f")
        lines.append("")
        lines.append("/* ========== 融合权重（自动调参） ========== */")
        lines.append(f"#define FUSION_WEIGHT_MS5611 {kf_params['fusion_weight_ms5611']:.3f}f")
        lines.append(f"#define FUSION_WEIGHT_BMP280 {kf_params['fusion_weight_bmp280']:.3f}f")
        lines.append("")
        lines.append("/* ========== EKF 噪声参数（自动调参） ========== */")
        lines.append(f"#define EKF_ACCEL_SIGMA     {kf_params['ekf_accel_sigma']:.2f}f")
        lines.append(f"#define EKF_PRESSURE_SIGMA  {kf_params['ekf_pressure_sigma']:.2f}f")

    lines.append("")
    lines.append("#endif /* __CALIB_AUTO_PARAMS_H__ */")
    lines.append("")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"[INFO] MCU 参数头文件已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='气压传感器校准分析工具')
    parser.add_argument('--ms5611', type=str, help='MS5611 数据文件路径')
    parser.add_argument('--bmp280', type=str, help='BMP280 数据文件路径')
    parser.add_argument('--data', type=str, help='数据目录或通配符')
    parser.add_argument('--sensor', type=str, choices=['ms5611', 'bmp280', 'all'],
                        default='ms5611', help='传感器类型')
    parser.add_argument('--all', action='store_true', help='扫描目录下所有 CSV')
    parser.add_argument('--output', type=str, default='calibration_report',
                        help='输出文件前缀')
    parser.add_argument('--plot', action='store_true', help='生成分析图表')
    args = parser.parse_args()

    ms5611_data = None
    bmp280_data = None

    # ---- 读取数据 ----
    if args.ms5611:
        data, meta = read_csv_data(args.ms5611, 'MS5611')
        if data is not None:
            ms5611_data = data
            print(f"[INFO] 读取 MS5611: {len(data)} 个样本")

    if args.bmp280:
        data, meta = read_csv_data(args.bmp280, 'BMP280')
        if data is not None:
            bmp280_data = data
            print(f"[INFO] 读取 BMP280: {len(data)} 个样本")

    if args.data:
        files = glob.glob(args.data)
        if not files:
            print(f"[ERROR] 未找到匹配的文件: {args.data}")
            return

        for f in sorted(files):
            if os.path.isdir(f):
                # 扫描目录
                for csv_file in sorted(glob.glob(os.path.join(f, "*.csv"))):
                    for prefix, name in [('ms5611', 'MS5611'), ('bmp280', 'BMP280')]:
                        if prefix in os.path.basename(csv_file).lower():
                            data, meta = read_csv_data(csv_file, name)
                            if data is not None:
                                if name == 'MS5611' and ms5611_data is None:
                                    ms5611_data = data
                                elif name == 'BMP280' and bmp280_data is None:
                                    bmp280_data = data
                                print(f"[INFO] 读取 {name}: {csv_file} ({len(data)} 个样本)")
            else:
                for prefix, name in [('ms5611', 'MS5611'), ('bmp280', 'BMP280')]:
                    if prefix in os.path.basename(f).lower():
                        data, meta = read_csv_data(f, name)
                        if data is not None:
                            if name == 'MS5611' and ms5611_data is None:
                                ms5611_data = data
                            elif name == 'BMP280' and bmp280_data is None:
                                bmp280_data = data
                            print(f"[INFO] 读取 {name}: {f} ({len(data)} 个样本)")

    if ms5611_data is None and bmp280_data is None:
        print("[ERROR] 未读取到任何数据")
        print("用法示例:")
        print("  python calibration_analyzer.py --ms5611 ms5611.csv --bmp280 bmp280.csv")
        print("  python calibration_analyzer.py --data data/raw/160m/ --all")
        return

    # ---- 统计分析 ----
    ms_stats = calc_statistics(ms5611_data) if ms5611_data is not None else None
    bm_stats = calc_statistics(bmp280_data) if bmp280_data is not None else None

    if ms_stats:
        print(f"\n[MS5611]")
        print(f"  样本数: {ms_stats['count']}")
        print(f"  均值: {ms_stats['mean']:.2f} Pa")
        print(f"  标准差: {ms_stats['std']:.4f} Pa ({ms_stats['std']/11.3:.4f} m)")
        print(f"  异常值: {ms_stats['outlier_count']} ({ms_stats['outlier_ratio']*100:.1f}%)")
        print(f"  健康: {ms_stats['health_str']}")

    if bm_stats:
        print(f"\n[BMP280]")
        print(f"  样本数: {bm_stats['count']}")
        print(f"  均值: {bm_stats['mean']:.2f} Pa")
        print(f"  标准差: {bm_stats['std']:.4f} Pa ({bm_stats['std']/11.3:.4f} m)")
        print(f"  异常值: {bm_stats['outlier_count']} ({bm_stats['outlier_ratio']*100:.1f}%)")
        print(f"  健康: {bm_stats['health_str']}")

    # ---- 双传感器分析 ----
    dual_result = None
    if ms5611_data is not None and bmp280_data is not None:
        dual_result = analyze_dual_sensor(ms5611_data, bmp280_data)
        print(f"\n[双传感器一致性]")
        print(f"  差值均值: {dual_result['diff_mean']:.2f} Pa")
        print(f"  差值标准差: {dual_result['diff_std']:.2f} Pa")
        print(f"  一致性: {dual_result['consistency_str']}")

    # ---- 自动调参 ----
    if ms_stats:
        kf_params = auto_tune_kf(ms_stats, bm_stats)
        print(f"\n[自动调参建议]")
        print(f"  KF_MS5611: Q={kf_params['kf_q_ms5611']:.4f}, R={kf_params['kf_r_ms5611']:.2f}")
        if kf_params['kf_r_bmp280'] is not None:
            print(f"  KF_BMP280: Q={kf_params['kf_q_bmp280']:.4f}, R={kf_params['kf_r_bmp280']:.2f}")
            print(f"  融合权重: MS5611={kf_params['fusion_weight_ms5611']:.3f}, "
                  f"BMP280={kf_params['fusion_weight_bmp280']:.3f}")
        print(f"  EKF: accel_sigma={kf_params['ekf_accel_sigma']:.2f}, "
              f"pressure_sigma={kf_params['ekf_pressure_sigma']:.2f}")

        # ---- 生成报告 ----
        report = generate_report(ms_stats, bm_stats, dual_result, kf_params,
                                 args.ms5611 or args.data or "")
        report_path = f"{args.output}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n[INFO] 报告已保存: {report_path}")

        # ---- 生成 MCU 头文件 ----
        header_path = f"{args.output}_params.h"
        generate_mcu_header(kf_params, header_path)

        # ---- 生成图表 ----
        if args.plot:
            plot_path = f"{args.output}.png"
            plot_analysis(ms5611_data, bmp280_data, ms_stats, bm_stats, kf_params, plot_path)

        # ---- 输出 C 代码宏定义（可直接复制使用） ----
        print(f"\n[可直接复制到 main.h / kalman_filter.h]")
        print(f"#define MS5611_KF_Q      {kf_params['kf_q_ms5611']:.4f}f")
        print(f"#define MS5611_KF_R      {kf_params['kf_r_ms5611']:.1f}f")
        if kf_params['kf_r_bmp280'] is not None:
            print(f"#define BMP280_KF_Q      {kf_params['kf_q_bmp280']:.4f}f")
            print(f"#define BMP280_KF_R      {kf_params['kf_r_bmp280']:.1f}f")
            print(f"#define FUSION_WEIGHT_MS5611 {kf_params['fusion_weight_ms5611']:.3f}f")
            print(f"#define FUSION_WEIGHT_BMP280 {kf_params['fusion_weight_bmp280']:.3f}f")


if __name__ == '__main__':
    main()
