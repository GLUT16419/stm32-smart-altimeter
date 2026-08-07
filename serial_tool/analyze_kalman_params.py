#!/usr/bin/env python
"""
卡尔曼滤波参数分析脚本
分析静止/平移运动/升降运动三个场景的气压数据噪声特性，
计算最优 KF 参数（Q, R, 自适应阈值等）。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

sys.path.insert(0, os.path.dirname(__file__))
from train_self_supervised import SMOOTH_WINDOW

data_root = os.path.join(os.path.dirname(__file__), "data")
raw_dir = os.path.join(data_root, "raw")

SCENES = ['静止', '平移运动', '升降运动']
SENSORS = ['ms5611', 'bmp280']
SENSOR_LABELS = {'ms5611': 'MS5611', 'bmp280': 'BMP280'}
COLORS = {'ms5611': 'steelblue', 'bmp280': 'orange'}
COLORS_SCENE = {'静止': '#2196F3', '平移运动': '#FF9800', '升降运动': '#9C27B0'}


def load_scene_data(scene, sensor):
    """加载指定场景和传感器的所有 CSV 数据"""
    scene_path = os.path.join(raw_dir, scene)
    if not os.path.isdir(scene_path):
        return None, None

    all_p = []
    all_t = []
    for f in sorted(os.listdir(scene_path)):
        if f.startswith(f'{sensor}_') and f.endswith('.csv'):
            df = pd.read_csv(os.path.join(scene_path, f))
            all_p.extend(df['pressure_pa'].values.tolist())
            all_t.extend(df['temperature_c'].values.tolist())

    if len(all_p) == 0:
        return None, None
    return np.array(all_p), np.array(all_t)


def compute_noise_metrics(pressure):
    """计算噪声统计指标"""
    # 差分（相邻样本之差）反映噪声水平
    diff = np.diff(pressure)

    # 去趋势：用滑动平均提取趋势，残差为噪声
    smooth = pd.Series(pressure).rolling(window=11, center=True, min_periods=1).mean().values
    noise = pressure - smooth

    metrics = {
        'n_samples': len(pressure),
        'mean': float(np.mean(pressure)),
        'std': float(np.std(pressure)),
        'min': float(np.min(pressure)),
        'max': float(np.max(pressure)),
        'range': float(np.max(pressure) - np.min(pressure)),

        # 差分统计（反映逐点抖动）
        'diff_mean': float(np.mean(diff)),
        'diff_std': float(np.std(diff)),
        'diff_rms': float(np.sqrt(np.mean(diff ** 2))),
        'diff_max_abs': float(np.max(np.abs(diff))),

        # 噪声统计（去趋势后的残差）
        'noise_std': float(np.std(noise)),
        'noise_rms': float(np.sqrt(np.mean(noise ** 2))),
        'noise_mae': float(np.mean(np.abs(noise))),

        # 百分位
        'noise_p5': float(np.percentile(noise, 5)),
        'noise_p95': float(np.percentile(noise, 95)),
        'noise_p1': float(np.percentile(noise, 1)),
        'noise_p99': float(np.percentile(noise, 99)),
    }

    # 自相关分析（判断噪声白度）
    if len(noise) > 100:
        acorr = np.correlate(noise - noise.mean(), noise - noise.mean(), mode='full')
        acorr = acorr / acorr[len(acorr) // 2]
        # lag-1 自相关系数
        metrics['autocorr_lag1'] = float(acorr[len(acorr) // 2 + 1])
    else:
        metrics['autocorr_lag1'] = 0

    return metrics, noise, diff


def compute_optimal_kf_params(metrics, scene_name, sensor_label):
    """
    根据噪声指标计算最优卡尔曼滤波参数

    R = 测量噪声方差 ≈ noise_std²
    Q = 过程噪声方差，控制跟踪速度
    """
    noise_std = metrics['noise_std']
    noise_rms = metrics['noise_rms']
    diff_std = metrics['diff_std']

    # === R 值 ===
    # R 应接近传感器测量噪声方差
    # 使用去趋势后噪声的方差
    R = noise_std ** 2
    R_suggested = round(R, 2)

    # === Q 值 ===
    # Q 控制滤波器对变化的响应速度
    # 静止时 Q 应小（平滑为主）
    # 运动时 Q 应大（跟踪为主）
    # 合理值: Q ≈ (diff_std * 0.1)² ~ (diff_std * 0.5)²
    if scene_name == '静止':
        Q_static = round(0.01 * noise_std, 3)
        Q_motion = round(0.1 * diff_std, 3)
    else:
        Q_static = round(0.05 * diff_std, 3)
        Q_motion = round(0.5 * diff_std, 3)

    # === 自适应阈值 ===
    # 应设置为噪声的 2~3 倍标准差
    th_suggested = round(2.0 * noise_std, 2)
    th_loose = round(3.0 * noise_std, 2)

    return {
        'R': R_suggested,
        'R_sqrt': round(np.sqrt(R_suggested), 3),
        'Q_static': Q_static,
        'Q_motion': Q_motion,
        'adaptive_threshold': th_suggested,
        'adaptive_threshold_loose': th_loose,
    }


def compute_kf_simulation(pressure, Q, R, adaptive=False):
    """用给定的 KF 参数仿真滤波"""
    n = len(pressure)
    x = np.zeros(n)
    p = 1.0
    kf_q = Q
    q_base = Q

    x[0] = pressure[0]
    for i in range(1, n):
        z = pressure[i]

        if adaptive:
            residual = z - x[i - 1]
            residual_abs = abs(residual)
            if residual_abs > 0.8:
                kf_q = min(kf_q * 1.05, 10.0)
            else:
                kf_q = max(kf_q * 0.98, q_base)

        p = p + kf_q
        k = p / (p + R)
        x[i] = x[i - 1] + k * (z - x[i - 1])
        p = (1 - k) * p

    return x


def analyze_scene(scene_name):
    """分析单个场景"""
    print(f"\n{'='*60}")
    print(f"  场景: {scene_name}")
    print(f"{'='*60}")

    results = {}
    for sensor in SENSORS:
        label = SENSOR_LABELS[sensor]
        pressure, temp = load_scene_data(scene_name, sensor)
        if pressure is None:
            print(f"  {label}: 无数据")
            continue

        metrics, noise, diff = compute_noise_metrics(pressure)
        kf_params = compute_optimal_kf_params(metrics, scene_name, label)

        results[sensor] = {
            'metrics': metrics,
            'noise': noise,
            'diff': diff,
            'kf_params': kf_params,
        }

        print(f"\n  --- {label} ---")
        print(f"  样本数: {metrics['n_samples']}")
        print(f"  气压范围: {metrics['min']:.1f} ~ {metrics['max']:.1f} Pa")
        print(f"  气压均值: {metrics['mean']:.2f} Pa")
        print(f"  气压标准差: {metrics['std']:.3f} Pa")
        print(f"  差分 RMS: {metrics['diff_rms']:.3f} Pa")
        print(f"  差分 max|Δ|: {metrics['diff_max_abs']:.2f} Pa")
        print(f"  噪声标准差 (去趋势): {metrics['noise_std']:.3f} Pa")
        print(f"  噪声 MAE: {metrics['noise_mae']:.3f} Pa")
        print(f"  噪声 P1~P99: [{metrics['noise_p1']:.3f}, {metrics['noise_p99']:.3f}]")
        print(f"  自相关 lag-1: {metrics['autocorr_lag1']:.4f}")

        print(f"\n  推荐 KF 参数:")
        print(f"    R (测量噪声方差)  = {kf_params['R']:.3f}  (sqrt ≈ {kf_params['R_sqrt']:.3f} Pa)")
        print(f"    Q_静止 (平滑)     = {kf_params['Q_static']:.4f}")
        print(f"    Q_运动 (跟踪)     = {kf_params['Q_motion']:.4f}")
        print(f"    自适应阈值        = {kf_params['adaptive_threshold']:.2f} Pa")
        print(f"    宽松阈值          = {kf_params['adaptive_threshold_loose']:.2f} Pa")

    return results


def simulate_and_compare(scene_name, all_results):
    """对每个场景仿真不同 KF 参数并对比"""
    for sensor in SENSORS:
        if sensor not in all_results.get(scene_name, {}):
            continue

        pressure = load_scene_data(scene_name, sensor)[0]
        if pressure is None:
            continue

        # 取前 500 点做可视化
        n_show = min(500, len(pressure))
        p_show = pressure[:n_show]

        # 仿真不同参数
        kf_params = all_results[scene_name][sensor]['kf_params']

        # 1. 当前参数 (Q=0.1, R=4.0 for MS5611)
        if sensor == 'ms5611':
            current_q, current_r = 0.1, 4.0
        else:
            current_q, current_r = 0.05, 1.0

        kf_current = compute_kf_simulation(p_show, current_q, current_r, adaptive=True)

        # 2. 推荐参数
        kf_optimal = compute_kf_simulation(
            p_show,
            kf_params['Q_motion'],
            kf_params['R'],
            adaptive=True
        )

        # 3. 推荐参数（静止 Q）
        kf_static = compute_kf_simulation(
            p_show,
            kf_params['Q_static'],
            kf_params['R'],
            adaptive=True
        )

        # 4. 标准 KF（非自适应）
        kf_standard = compute_kf_simulation(
            p_show,
            kf_params['Q_motion'],
            kf_params['R'],
            adaptive=False
        )

        # 计算各方法的噪声抑制效果
        def calc_smoothness(x):
            return np.std(np.diff(x))

        print(f"\n  {SENSOR_LABELS[sensor]} 滤波效果对比 ({scene_name}):")
        print(f"    {'方法':<20s} {'输出噪声std':<15s} {'响应延迟':<10s}")
        print(f"    {'原始信号':<20s} {calc_smoothness(p_show):<15.3f} {'-':<10s}")
        print(f"    {'当前 KF(自适应)':<20s} {calc_smoothness(kf_current):<15.3f} {'-':<10s}")
        print(f"    {'推荐 KF(运动Q)':<20s} {calc_smoothness(kf_optimal):<15.3f} {'-':<10s}")
        print(f"    {'推荐 KF(静止Q)':<20s} {calc_smoothness(kf_static):<15.3f} {'-':<10s}")
        print(f"    {'推荐 KF(非自适应)':<20s} {calc_smoothness(kf_standard):<15.3f} {'-':<10s}")

    # 绘制对比图
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f'卡尔曼滤波参数对比 - {scene_name}', fontsize=14)

    for idx, sensor in enumerate(SENSORS):
        if sensor not in all_results.get(scene_name, {}):
            continue

        pressure = load_scene_data(scene_name, sensor)[0]
        if pressure is None:
            continue

        n_show = min(500, len(pressure))
        p_show = pressure[:n_show]
        kf_params = all_results[scene_name][sensor]['kf_params']

        # 原始信号
        ax = axes[0, idx]
        ax.plot(p_show, alpha=0.5, label='Raw', color='gray', linewidth=0.8)
        ax.set_title(f'{SENSOR_LABELS[sensor]} 原始信号')
        ax.set_ylabel('Pressure (Pa)')
        ax.grid(True, alpha=0.3)

        # 差分（噪声水平）
        diff_show = np.diff(p_show)
        ax2 = axes[1, idx]
        ax2.plot(diff_show, alpha=0.5, color='gray', linewidth=0.6, label=f'Diff (std={np.std(diff_show):.2f})')
        ax2.set_title(f'{SENSOR_LABELS[sensor]} 逐点差分')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Δ Pressure (Pa)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

    # 第三列：滤波效果对比
    sensor = SENSORS[0] if SENSORS[0] in all_results.get(scene_name, {}) else None
    if sensor and sensor in all_results.get(scene_name, {}):
        pressure = load_scene_data(scene_name, sensor)[0]
        if pressure is not None:
            n_show = min(500, len(pressure))
            p_show = pressure[:n_show]
            kf_params = all_results[scene_name][sensor]['kf_params']

            if sensor == 'ms5611':
                current_q, current_r = 0.1, 4.0
            else:
                current_q, current_r = 0.05, 1.0

            kf_current = compute_kf_simulation(p_show, current_q, current_r, adaptive=True)
            kf_optimal = compute_kf_simulation(p_show, kf_params['Q_motion'], kf_params['R'], adaptive=True)
            kf_static = compute_kf_simulation(p_show, kf_params['Q_static'], kf_params['R'], adaptive=True)

            ax = axes[0, 2]
            ax.plot(p_show, alpha=0.3, label='Raw', color='gray', linewidth=0.8)
            ax.plot(kf_current, label=f'Current (Q={current_q}, R={current_r})', alpha=0.7)
            ax.plot(kf_optimal, label=f'Optimal (Q={kf_params["Q_motion"]:.3f}, R={kf_params["R"]:.1f})',
                    alpha=0.7, linestyle='--')
            ax.plot(kf_static, label=f'Static (Q={kf_params["Q_static"]:.4f}, R={kf_params["R"]:.1f})',
                    alpha=0.7, linestyle=':')
            ax.set_title(f'{SENSOR_LABELS[sensor]} 滤波对比')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # 残差分布
            ax = axes[1, 2]
            ax.hist(p_show - kf_current, bins=50, alpha=0.5, label='Current KF', color='steelblue')
            ax.hist(p_show - kf_optimal, bins=50, alpha=0.5, label='Optimal KF', color='orange')
            ax.axvline(x=0, color='r', ls='--', alpha=0.5)
            ax.set_title('滤波残差分布')
            ax.set_xlabel('Residual (Pa)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "models")
    plt.savefig(os.path.join(out_dir, f'kf_analysis_{scene_name}.png'), dpi=150)
    plt.close()


def summary_recommendation(all_results):
    """生成最终的参数推荐总结"""
    print(f"\n\n{'='*60}")
    print(f"  卡尔曼滤波参数优化建议")
    print(f"{'='*60}")

    # 收集各场景各传感器的推荐参数
    for sensor in SENSORS:
        label = SENSOR_LABELS[sensor]
        print(f"\n  --- {label} ---")

        R_vals = []
        Q_static_vals = []
        Q_motion_vals = []
        th_vals = []

        for scene in SCENES:
            if scene in all_results and sensor in all_results[scene]:
                kfp = all_results[scene][sensor]['kf_params']
                R_vals.append(kfp['R'])
                Q_static_vals.append(kfp['Q_static'])
                Q_motion_vals.append(kfp['Q_motion'])
                th_vals.append(kfp['adaptive_threshold'])
                print(f"    {scene}: R={kfp['R']:.3f}, Qs={kfp['Q_static']:.4f}, "
                      f"Qm={kfp['Q_motion']:.4f}, th={kfp['adaptive_threshold']:.2f}")

        if R_vals:
            # R 取最大值（覆盖最坏情况噪声）
            R_final = max(R_vals)
            # Q_static 取最小值（静止时最平滑）
            Qs_final = min(Q_static_vals)
            # Q_motion 取最大值（运动时快速跟踪）
            Qm_final = max(Q_motion_vals)
            # 阈值取各场景的均值
            th_final = np.mean(th_vals)

            print(f"\n    >>> 推荐参数:")
            print(f"        R (测量噪声方差)  = {R_final:.2f}  (噪声 std ≈ {np.sqrt(R_final):.2f} Pa)")
            print(f"        Q_base (静止)     = {Qs_final:.4f}")
            print(f"        Q_max (运动)      = {Qm_final:.4f}")
            print(f"        自适应阈值        = {th_final:.2f} Pa")
            print(f"        自适应阈值(宽松)  = {th_final * 1.5:.2f} Pa")

            # 与当前参数对比
            if sensor == 'ms5611':
                cur_q, cur_r = 0.1, 4.0
                print(f"\n    >>> 当前配置: Q={cur_q}, R={cur_r}")
                print(f"        差异: R {'↑' if R_final > cur_r else '↓'} ({R_final:.1f} vs {cur_r}), "
                      f"Q {'↑' if Qs_final > cur_q else '↓'} ({Qs_final:.4f} vs {cur_q})")
            else:
                cur_q, cur_r = 0.05, 1.0
                print(f"\n    >>> 当前配置: Q={cur_q}, R={cur_r}")
                print(f"        差异: R {'↑' if R_final > cur_r else '↓'} ({R_final:.2f} vs {cur_r}), "
                      f"Q {'↑' if Qs_final > cur_q else '↓'} ({Qs_final:.4f} vs {cur_q})")

            # 给出修改建议
            print(f"\n    >>> 修改建议:")
            print(f"        将 R 改为 {R_final:.2f}")
            print(f"        将 Q_base 改为 {Qs_final:.4f}")
            print(f"        将 KF_RESIDUAL_TH 改为 {th_final:.2f}")
            if sensor == 'ms5611':
                print(f"        将 MS5611_KF_R 改为 {R_final:.1f}f")
                print(f"        将 MS5611_KF_Q 改为 {Qs_final:.4f}f")
            else:
                print(f"        将 BMP280_KF_R 改为 {R_final:.1f}f")
                print(f"        将 BMP280_KF_Q 改为 {Qs_final:.4f}f")


def main():
    print("=" * 60)
    print("   卡尔曼滤波参数分析")
    print("   分析静止/平移运动/升降运动三场景的噪声特性")
    print("   计算最优 KF 参数 (Q, R, 自适应阈值)")
    print("=" * 60)

    all_results = {}

    for scene in SCENES:
        results = analyze_scene(scene)
        all_results[scene] = results

    # 仿真对比
    print(f"\n\n{'='*60}")
    print(f"  滤波效果仿真对比")
    print(f"{'='*60}")
    for scene in SCENES:
        simulate_and_compare(scene, all_results)

    # 最终建议
    summary_recommendation(all_results)

    print(f"\n\n分析完成！图表已保存到 models/ 目录。")


if __name__ == "__main__":
    main()
