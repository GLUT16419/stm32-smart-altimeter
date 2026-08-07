#!/usr/bin/env python
"""BP 变体对比图重绘脚本（加载已训练的 TFLite 模型做预测并绘图）"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from train_self_supervised import load_all_self_supervised, SMOOTH_WINDOW

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_root = os.path.join(os.path.dirname(__file__), "data")
models_dir = os.path.join(os.path.dirname(__file__), "models")

WINDOW_SIZE = 10
INPUT_DIM = WINDOW_SIZE + 1


def create_sequences_with_sensor(data, sensor_id, window_size=WINDOW_SIZE):
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(data, size=SMOOTH_WINDOW, mode='reflect')
    X, y = [], []
    for i in range(0, len(data) - window_size, 1):
        window = data[i:i + window_size]
        feat = np.append(window, sensor_id)
        X.append(feat)
        center_idx = i + window_size - 1
        y.append(smooth[center_idx])
    return np.array(X), np.array(y)


def get_contiguous_segment(indices, max_n=200):
    if len(indices) == 0:
        return np.array([], dtype=int)
    sorted_idx = np.sort(indices)
    gaps = np.diff(sorted_idx)
    split_points = np.where(gaps > 1)[0]
    if len(split_points) > 0:
        seg_end = split_points[0] + 1
        seg = sorted_idx[:seg_end]
    else:
        seg = sorted_idx
    return seg[:max_n]


def replot():
    print("加载数据...")
    ms5611_p, _, _ = load_all_self_supervised(data_root, 'ms5611')
    bmp280_p, _, _ = load_all_self_supervised(data_root, 'bmp280')

    reference_pressure = 101325.0  # 标准海平面气压（1013.25 hPa）
    ms5611_rel = ms5611_p - reference_pressure
    bmp280_rel = bmp280_p - reference_pressure

    combined_rel = np.concatenate([ms5611_rel, bmp280_rel])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(combined_rel.reshape(-1, 1))
    ms5611_scaled = scaler.transform(ms5611_rel.reshape(-1, 1)).flatten()
    bmp280_scaled = scaler.transform(bmp280_rel.reshape(-1, 1)).flatten()

    X_m, y_m = create_sequences_with_sensor(ms5611_scaled, sensor_id=0.0)
    X_b, y_b = create_sequences_with_sensor(bmp280_scaled, sensor_id=1.0)

    X = np.concatenate([X_m, X_b])
    y = np.concatenate([y_m, y_b])
    sensor_labels = np.concatenate([np.zeros(len(X_m)), np.ones(len(X_b))])

    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, sensor_labels, test_size=0.3, random_state=42, stratify=sensor_labels)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=s_temp)

    print(f"测试集: {len(X_test)} 样本")

    # 所有变体名称和对应的 TFLite 文件
    variants = [
        ('baseline',            'v0_baseline'),
        ('residual_bp',         'v1_residual_bp'),
        ('deeper_bp',           'v2_deeper_bp'),
        ('regularized_bp',      'v3_regularized_bp'),
        ('residual_deeper',     'v4_residual_deeper'),
        ('deeper_regularized',  'v5_deeper_regularized'),
        ('full',                'v6_full'),
    ]

    # 加载对比 JSON 获取统计信息
    summary_path = os.path.join(models_dir, "bp_variants_comparison.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary_data = json.load(f)
        stats = {s['variant']: s for s in summary_data}
    else:
        stats = {}

    results = []
    for name, tag in variants:
        tflite_path = os.path.join(models_dir, f"dual_sensor_self_supervised_filter_{tag}.tflite")
        if not os.path.exists(tflite_path):
            print(f"  [跳过] {name} — 找不到 {tflite_path}")
            continue

        print(f"  加载 {name} ({tag})...")
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # 逐样本预测（TFLite 固定 batch=1）
        y_pred = np.zeros(len(X_test))
        for i in range(len(X_test)):
            batch = X_test[i:i + 1].astype(np.float32)  # shape: (1, 11)
            interpreter.set_tensor(input_details[0]['index'], batch)
            interpreter.invoke()
            y_pred[i] = interpreter.get_tensor(output_details[0]['index'])[0, 0]

        y_test_u = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_u = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        overall_rmse = np.sqrt(mean_squared_error(y_test_u, y_pred_u))
        overall_mae = mean_absolute_error(y_test_u, y_pred_u)

        ms5611_mask = s_test == 0
        bmp280_mask = s_test == 1
        rmse_m = mae_m = rmse_b = mae_b = None
        if np.any(ms5611_mask):
            rmse_m = np.sqrt(mean_squared_error(y_test_u[ms5611_mask], y_pred_u[ms5611_mask]))
            mae_m = mean_absolute_error(y_test_u[ms5611_mask], y_pred_u[ms5611_mask])
        if np.any(bmp280_mask):
            rmse_b = np.sqrt(mean_squared_error(y_test_u[bmp280_mask], y_pred_u[bmp280_mask]))
            mae_b = mean_absolute_error(y_test_u[bmp280_mask], y_pred_u[bmp280_mask])

        s = stats.get(tag, {})
        results.append({
            'name': name,
            'variant': tag,
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'rmse_m': rmse_m, 'mae_m': mae_m,
            'rmse_b': rmse_b, 'mae_b': mae_b,
            'total_params': s.get('total_params', 0),
            'tflite_size_kb': s.get('tflite_size_kb', 0),
            'training_time_sec': s.get('training_time_sec', 0),
            'y_pred_u': y_pred_u,
            'y_test_u': y_test_u,
            's_test': s_test,
        })
        print(f"    RMSE={overall_rmse:.2f}Pa MAE={overall_mae:.2f}Pa")

    if not results:
        print("错误：没有可用的模型！")
        return

    # ===== 绘图 =====
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f'BP Variants Comparison (SMOOTH={SMOOTH_WINDOW})', fontsize=16, fontweight='bold')

    names = [r['name'] for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    # 1. RMSE 柱状图
    ax = axes[0, 0]
    overall_rmse = [r['overall_rmse'] for r in results]
    bars = ax.bar(names, overall_rmse, color=colors, edgecolor='gray')
    ax.set_title('Overall RMSE (Pa) - Lower is Better', fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, overall_rmse):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # 2. MAE 柱状图
    ax = axes[0, 1]
    overall_mae = [r['overall_mae'] for r in results]
    bars = ax.bar(names, overall_mae, color=colors, edgecolor='gray')
    ax.set_title('Overall MAE (Pa) - Lower is Better', fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, overall_mae):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # 3. 参数量 vs TFLite 大小
    ax = axes[0, 2]
    params = [r['total_params'] / 1000 for r in results]
    sizes = [r['tflite_size_kb'] for r in results]
    ax2 = ax.twinx()
    ax.bar(np.arange(len(names)) - 0.2, params, 0.35, label='Params (K)', color='steelblue')
    ax2.bar(np.arange(len(names)) + 0.2, sizes, 0.35, label='TFLite (KB)', color='coral')
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title('Model Size', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 4. 各传感器 RMSE 对比
    ax = axes[0, 3]
    x = np.arange(len(names))
    width = 0.35
    rmse_m = [r['rmse_m'] if r['rmse_m'] else 0 for r in results]
    rmse_b = [r['rmse_b'] if r['rmse_b'] else 0 for r in results]
    ax.bar(x - width/2, rmse_m, width, label='MS5611', color='steelblue')
    ax.bar(x + width/2, rmse_b, width, label='BMP280', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title('Per-Sensor RMSE (Pa)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 5. MS5611 滤波结果对比（连续段 + target 曲线）
    ax = axes[1, 0]
    n_show = 200
    ms_indices = np.where(results[0]['s_test'] == 0)[0]
    bmp_indices = np.where(results[0]['s_test'] == 1)[0]
    ms_seg = get_contiguous_segment(ms_indices, n_show)
    bmp_seg = get_contiguous_segment(bmp_indices, n_show)

    ax.plot(results[0]['y_test_u'][ms_seg], label='Filter Target', color='black', linewidth=1.5, alpha=0.8)
    for r in results:
        ax.plot(r['y_pred_u'][ms_seg], alpha=0.7, linewidth=0.8, label=r['name'])
    ax.set_title(f'MS5611 Filter (first {len(ms_seg)} contiguous samples)', fontweight='bold')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)

    # 6. BMP280 滤波结果对比
    ax = axes[1, 1]
    ax.plot(results[0]['y_test_u'][bmp_seg], label='Filter Target', color='black', linewidth=1.5, alpha=0.8)
    for r in results:
        ax.plot(r['y_pred_u'][bmp_seg], alpha=0.7, linewidth=0.8, label=r['name'])
    ax.set_title(f'BMP280 Filter (first {len(bmp_seg)} contiguous samples)', fontweight='bold')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)

    # 7. 训练时间
    ax = axes[1, 2]
    times = [r['training_time_sec'] for r in results]
    bars = ax.bar(names, times, color=colors, edgecolor='gray')
    ax.set_title('Training Time (s)', fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.0f}s', ha='center', va='bottom', fontsize=7)

    # 8. 误差分布对比（MAE 最小的 4 个）
    ax = axes[1, 3]
    sorted_by_mae = sorted(results, key=lambda r: r['overall_mae'])
    top4 = sorted_by_mae[:4]
    for r in top4:
        err = r['y_test_u'] - r['y_pred_u']
        ax.hist(err, bins=60, alpha=0.5, label=f"{r['name']} (MAE={r['overall_mae']:.2f})")
    ax.axvline(x=0, color='r', ls='--', alpha=0.5)
    ax.set_title('Error Distribution (Top 4 Variants)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plot_path = os.path.join(models_dir, "bp_variants_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n对比图已保存: {plot_path}")


if __name__ == "__main__":
    replot()
