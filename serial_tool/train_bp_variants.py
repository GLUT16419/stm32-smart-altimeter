#!/usr/bin/env python
"""
BP 改善版本综合训练对比脚本
========================================
共 7 种变体（含原版），全部训练后汇总对比：

  变体 | 名称                        | 残差 | 更深 | 正则化
  ----|-----------------------------|:----:|:----:|:-----:
  v0  | baseline (原版 64→32→16)    |  ✗   |  ✗   |   ✗
  v1  | residual_bp (残差BP)         |  ✓   |  ✗   |   ✗
  v2  | deeper_bp (更深BP)           |  ✗   |  ✓   |   ✗
  v3  | regularized_bp (加正则化)     |  ✗   |  ✗   |   ✓
  v4  | residual_deeper (残差+更深)  |  ✓   |  ✓   |   ✗
  v5  | deeper_regularized (更深+正则)|  ✗   |  ✓   |   ✓
  v6  | full (残差+更深+正则化)       |  ✓   |  ✓   |   ✓
"""

import os, sys, json, time
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_root = os.path.join(os.path.dirname(__file__), "data")
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

WINDOW_SIZE = 10
INPUT_DIM = WINDOW_SIZE + 1  # 10 气压值 + 1 sensor_id

# ============================================================
# 模型构建函数（7 种变体）
# ============================================================

def build_baseline_bp(input_dim):
    """v0: 原版 64→32→16 (baseline)"""
    model = Sequential(name='bp_baseline')
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_residual_bp(input_dim):
    """v1: 残差 BP — 学习噪声残差，输出 = input_last + residual"""
    inputs = Input(shape=(input_dim,), name='input')
    # 提取最后一个气压值作为"当前值"
    last_pressure = inputs[:, WINDOW_SIZE - 1:WINDOW_SIZE]  # shape: (batch, 1)

    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    residual = Dense(1, name='residual')(x)  # 学习残差

    # 输出 = 当前值 + 残差
    outputs = Add(name='output')([last_pressure, residual])

    model = Model(inputs=inputs, outputs=outputs, name='bp_residual')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_deeper_bp(input_dim):
    """v2: 更深 BP — 128→96→64→48→32→16"""
    model = Sequential(name='bp_deeper')
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_regularized_bp(input_dim):
    """v3: 正则化 BP — 64→32→16 + Dropout + BN + L2"""
    model = Sequential(name='bp_regularized')
    model.add(Dense(64, kernel_regularizer=l2(1e-4), input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(32, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(16, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_residual_deeper_bp(input_dim):
    """v4: 残差 + 更深 — 更深结构但用残差连接"""
    inputs = Input(shape=(input_dim,), name='input')
    last_pressure = inputs[:, WINDOW_SIZE - 1:WINDOW_SIZE]

    x = Dense(128, activation='relu')(inputs)
    x = Dense(96, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(48, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    residual = Dense(1, name='residual')(x)

    outputs = Add(name='output')([last_pressure, residual])
    model = Model(inputs=inputs, outputs=outputs, name='bp_residual_deeper')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_deeper_regularized_bp(input_dim):
    """v5: 更深 + 正则化 — 128→96→64→48→32→16 + Dropout + BN + L2"""
    model = Sequential(name='bp_deeper_regularized')
    model.add(Dense(128, kernel_regularizer=l2(1e-4), input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(96, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(64, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.15))

    model.add(Dense(48, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.15))

    model.add(Dense(32, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(16, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_full_bp(input_dim):
    """v6: 残差 + 更深 + 正则化 — 综合所有技巧"""
    inputs = Input(shape=(input_dim,), name='input')
    last_pressure = inputs[:, WINDOW_SIZE - 1:WINDOW_SIZE]

    x = Dense(128, kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)

    x = Dense(96, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)

    x = Dense(64, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.15)(x)

    x = Dense(48, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.15)(x)

    x = Dense(32, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Dense(16, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    residual = Dense(1, name='residual')(x)
    outputs = Add(name='output')([last_pressure, residual])

    model = Model(inputs=inputs, outputs=outputs, name='bp_full')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ============================================================
# 数据准备（与 train_self_supervised_bp.py 一致）
# ============================================================

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


# ============================================================
# 训练函数
# ============================================================

# 所有变体定义
VARIANTS = [
    ('baseline',            'v0_baseline',                     build_baseline_bp),
    ('residual_bp',         'v1_residual_bp',                  build_residual_bp),
    ('deeper_bp',           'v2_deeper_bp',                    build_deeper_bp),
    ('regularized_bp',      'v3_regularized_bp',               build_regularized_bp),
    ('residual_deeper',     'v4_residual_deeper',              build_residual_deeper_bp),
    ('deeper_regularized',  'v5_deeper_regularized',           build_deeper_regularized_bp),
    ('full',                'v6_full',                         build_full_bp),
]


def get_model_params(model):
    """获取模型参数数量和文件大小估计"""
    trainable = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    return trainable, non_trainable


def train_one_variant(var_name, var_tag, build_fn, X_train, y_train, X_val, y_val,
                      X_test, y_test, s_test, scaler):
    """训练一个变体并返回结果"""
    print(f"\n{'='*60}")
    print(f"  训练: {var_name} ({var_tag})")
    print(f"{'='*60}")

    model = build_fn(INPUT_DIM)
    trainable, non_trainable = get_model_params(model)
    print(f"  参数量: trainable={trainable:,}, non_trainable={non_trainable:,}, "
          f"total={trainable+non_trainable:,}")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
    ]

    t_start = time.time()
    history = model.fit(X_train, y_train, epochs=200, batch_size=32,
                        validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    t_elapsed = time.time() - t_start

    # 评估
    y_pred = model.predict(X_test, verbose=0)
    y_test_u = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_u = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    overall_rmse = np.sqrt(mean_squared_error(y_test_u, y_pred_u))
    overall_mae = mean_absolute_error(y_test_u, y_pred_u)

    # 按传感器分
    ms5611_mask = s_test == 0
    bmp280_mask = s_test == 1
    rmse_m = mae_m = rmse_b = mae_b = None
    if np.any(ms5611_mask):
        rmse_m = np.sqrt(mean_squared_error(y_test_u[ms5611_mask], y_pred_u[ms5611_mask]))
        mae_m = mean_absolute_error(y_test_u[ms5611_mask], y_pred_u[ms5611_mask])
    if np.any(bmp280_mask):
        rmse_b = np.sqrt(mean_squared_error(y_test_u[bmp280_mask], y_pred_u[bmp280_mask]))
        mae_b = mean_absolute_error(y_test_u[bmp280_mask], y_pred_u[bmp280_mask])

    # 保存 TFLite
    model_name = f"dual_sensor_self_supervised_filter_{var_tag}"
    tflite_path = os.path.join(models_dir, f"{model_name}.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    tflite_size = os.path.getsize(tflite_path) / 1024

    # 保存 scaler 信息
    scaler_info = {
        'min': float(scaler.data_min_[0]),
        'max': float(scaler.data_max_[0]),
        'range': float(scaler.data_max_[0] - scaler.data_min_[0]),
        'reference_pressure': 101325.0,
        'window_size': WINDOW_SIZE,
        'input_dim': INPUT_DIM,
        'model_type': f'bp_{var_tag}',
        'variant': var_tag,
        'variant_name': var_name,
        'sensor': 'Dual(MS5611+BMP280)',
        'mode': 'self_supervised_filter_dual',
        'input_is_relative': True,
        'smooth_window': SMOOTH_WINDOW,
        'sensor_id_ms5611': 0.0,
        'sensor_id_bmp280': 1.0,
        'trainable_params': int(trainable),
        'tflite_size_kb': round(tflite_size, 1),
        'training_time_sec': round(t_elapsed, 1),
        'model_mode': 'self_supervised',
    }
    with open(os.path.join(models_dir, f"{model_name}_scaler.json"), 'w') as f:
        json.dump(scaler_info, f, indent=2)

    print(f"\n  {'='*40}")
    print(f"  结果 [{var_name}]:")
    print(f"    总体: RMSE={overall_rmse:.2f}Pa  MAE={overall_mae:.2f}Pa")
    if rmse_m is not None:
        print(f"    MS5611: RMSE={rmse_m:.2f}Pa  MAE={mae_m:.2f}Pa")
    if rmse_b is not None:
        print(f"    BMP280: RMSE={rmse_b:.2f}Pa  MAE={mae_b:.2f}Pa")
    print(f"    参数量: {trainable+non_trainable:,}")
    print(f"    TFLite: {tflite_size:.1f} KB")
    print(f"    训练时间: {t_elapsed:.1f}s")
    print(f"  {'='*40}")

    return {
        'variant': var_tag,
        'name': var_name,
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'rmse_m': rmse_m,
        'mae_m': mae_m,
        'rmse_b': rmse_b,
        'mae_b': mae_b,
        'trainable_params': int(trainable),
        'non_trainable_params': int(non_trainable),
        'total_params': int(trainable + non_trainable),
        'tflite_size_kb': tflite_size,
        'training_time_sec': t_elapsed,
        'epochs_trained': len(history.history['loss']),
        'history': history,
        'model': model,
        'y_pred_u': y_pred_u,
        'y_test_u': y_test_u,
        's_test': s_test,
    }


# ============================================================
# 主函数
# ============================================================

def train_all_variants():
    print(f"\n{'='*60}")
    print(f"   BP 改善版本综合训练对比")
    print(f"   平滑窗口={SMOOTH_WINDOW}")
    print(f"   共 {len(VARIANTS)} 种变体")
    print(f"{'='*60}")

    # ---- 1. 加载数据 ----
    print("\n--- 加载 MS5611 数据 ---")
    ms5611_p, _, _ = load_all_self_supervised(data_root, 'ms5611')
    print("\n--- 加载 BMP280 数据 ---")
    bmp280_p, _, _ = load_all_self_supervised(data_root, 'bmp280')

    # ---- 2. 相对气压 ----
    reference_pressure = 101325.0  # 标准海平面气压（1013.25 hPa）
    ms5611_rel = ms5611_p - reference_pressure
    bmp280_rel = bmp280_p - reference_pressure

    # ---- 3. 合并归一化 ----
    combined_rel = np.concatenate([ms5611_rel, bmp280_rel])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(combined_rel.reshape(-1, 1))
    ms5611_scaled = scaler.transform(ms5611_rel.reshape(-1, 1)).flatten()
    bmp280_scaled = scaler.transform(bmp280_rel.reshape(-1, 1)).flatten()

    print(f"\n合并归一化: range={scaler.data_max_[0] - scaler.data_min_[0]:.2f}")

    # ---- 4. 创建序列 ----
    X_m, y_m = create_sequences_with_sensor(ms5611_scaled, sensor_id=0.0)
    X_b, y_b = create_sequences_with_sensor(bmp280_scaled, sensor_id=1.0)
    print(f"序列: MS5611={X_m.shape}, BMP280={X_b.shape}")

    # ---- 5. 合并 & 切分 ----
    X = np.concatenate([X_m, X_b])
    y = np.concatenate([y_m, y_b])
    sensor_labels = np.concatenate([np.zeros(len(X_m)), np.ones(len(X_b))])

    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, sensor_labels, test_size=0.3, random_state=42, stratify=sensor_labels)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=s_temp)

    print(f"\n数据集: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # ---- 6. 逐个训练所有变体 ----
    all_results = []
    for var_name, var_tag, build_fn in VARIANTS:
        result = train_one_variant(var_name, var_tag, build_fn,
                                    X_train, y_train, X_val, y_val,
                                    X_test, y_test, s_test, scaler)
        all_results.append(result)

    # ---- 7. 汇总对比表 ----
    print(f"\n\n{'='*70}")
    print(f"   BP 变体综合对比汇总")
    print(f"{'='*70}")
    print(f"{'变体':<24} {'参数':>10} {'TFLite':>8} {'总体RMSE':>10} {'总体MAE':>10} "
          f"{'MS5611_RMSE':>12} {'BMP280_RMSE':>12} {'耗时':>8}")
    print(f"{'-'*24} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*8}")

    for r in all_results:
        rmse_m_str = f"{r['rmse_m']:.2f}" if r['rmse_m'] else "N/A"
        rmse_b_str = f"{r['rmse_b']:.2f}" if r['rmse_b'] else "N/A"
        print(f"{r['name']:<24} {r['total_params']:>10,} "
              f"{r['tflite_size_kb']:>7.1f}K "
              f"{r['overall_rmse']:>9.2f}  {r['overall_mae']:>9.2f}  "
              f"{rmse_m_str:>10}  {rmse_b_str:>10}  "
              f"{r['training_time_sec']:>6.1f}s")

    # ---- 8. 可视化对比 ----
    plot_comparison(all_results)
    save_comparison_table(all_results)

    # ---- 9. 保存汇总 JSON ----
    summary = []
    for r in all_results:
        summary.append({
            'variant': r['variant'],
            'name': r['name'],
            'overall_rmse_pa': round(r['overall_rmse'], 2),
            'overall_mae_pa': round(r['overall_mae'], 2),
            'ms5611_rmse_pa': round(r['rmse_m'], 2) if r['rmse_m'] else None,
            'ms5611_mae_pa': round(r['mae_m'], 2) if r['mae_m'] else None,
            'bmp280_rmse_pa': round(r['rmse_b'], 2) if r['rmse_b'] else None,
            'bmp280_mae_pa': round(r['mae_b'], 2) if r['mae_b'] else None,
            'total_params': r['total_params'],
            'tflite_size_kb': round(r['tflite_size_kb'], 1),
            'training_time_sec': round(r['training_time_sec'], 1),
            'epochs': r['epochs_trained'],
        })
    summary_path = os.path.join(models_dir, "bp_variants_comparison.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n汇总 JSON: {summary_path}")
    print(f"\n{'='*70}")
    print(f"   全部训练完成！")
    print(f"{'='*70}")


def plot_comparison(all_results):
    """生成综合对比图"""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f'BP Variants Comparison (SMOOTH={SMOOTH_WINDOW})', fontsize=16, fontweight='bold')

    names = [r['name'] for r in all_results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    # 1. RMSE 柱状图
    ax = axes[0, 0]
    overall_rmse = [r['overall_rmse'] for r in all_results]
    bars = ax.bar(names, overall_rmse, color=colors, edgecolor='gray')
    ax.set_title('Overall RMSE (Pa) - Lower is Better', fontweight='bold')
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, overall_rmse):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # 2. MAE 柱状图
    ax = axes[0, 1]
    overall_mae = [r['overall_mae'] for r in all_results]
    bars = ax.bar(names, overall_mae, color=colors, edgecolor='gray')
    ax.set_title('Overall MAE (Pa) - Lower is Better', fontweight='bold')
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, overall_mae):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # 3. 参数量 vs TFLite 大小
    ax = axes[0, 2]
    params = [r['total_params'] / 1000 for r in all_results]
    sizes = [r['tflite_size_kb'] for r in all_results]
    ax2 = ax.twinx()
    bars1 = ax.bar(np.arange(len(names)) - 0.2, params, 0.35, label='Params (K)', color='steelblue')
    bars2 = ax2.bar(np.arange(len(names)) + 0.2, sizes, 0.35, label='TFLite (KB)', color='coral')
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
    rmse_m = [r['rmse_m'] if r['rmse_m'] else 0 for r in all_results]
    rmse_b = [r['rmse_b'] if r['rmse_b'] else 0 for r in all_results]
    ax.bar(x - width/2, rmse_m, width, label='MS5611', color='steelblue')
    ax.bar(x + width/2, rmse_b, width, label='BMP280', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title('Per-Sensor RMSE (Pa)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 5-7. 滤波结果对比 — 取测试集前 N 个连续样本（默认先按原始顺序，前部分为 MS5611）
    n_show = 200
    # 找 MS5611 和 BMP280 的连续段
    # 测试集是按分层采样的，先找第一个传感器的连续范围
    # 直接用 s_test 找第一个传感器的起始位置
    ref_result = all_results[0]
    s_test = ref_result['s_test']
    y_test_u = ref_result['y_test_u']

    # 找到 MS5611 (0) 和 BMP280 (1) 在 s_test 中的索引
    ms_indices = np.where(s_test == 0)[0]
    bmp_indices = np.where(s_test == 1)[0]

    # 取每个传感器前 n_show 个连续点（按原始索引顺序取连续段）
    def get_contiguous_segment(indices, max_n=200):
        """从排序后的索引中取一个连续段（间隔为1的段）"""
        if len(indices) == 0:
            return np.array([], dtype=int)
        sorted_idx = np.sort(indices)
        # 找第一个连续段
        gaps = np.diff(sorted_idx)
        split_points = np.where(gaps > 1)[0]
        if len(split_points) > 0:
            seg_end = split_points[0] + 1
            seg = sorted_idx[:seg_end]
        else:
            seg = sorted_idx
        return seg[:max_n]

    ms_seg = get_contiguous_segment(ms_indices, n_show)
    bmp_seg = get_contiguous_segment(bmp_indices, n_show)

    # MS5611 滤波结果
    ax = axes[1, 0]
    ax.plot(y_test_u[ms_seg], label='Filter Target', color='black', linewidth=1.5, alpha=0.8)
    for r in all_results:
        ax.plot(r['y_pred_u'][ms_seg], alpha=0.7, linewidth=0.8, label=r['name'])
    ax.set_title(f'MS5611 Filter (first {len(ms_seg)} contiguous samples)', fontweight='bold')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)

    # BMP280 滤波结果
    ax = axes[1, 1]
    ax.plot(y_test_u[bmp_seg], label='Filter Target', color='black', linewidth=1.5, alpha=0.8)
    for r in all_results:
        ax.plot(r['y_pred_u'][bmp_seg], alpha=0.7, linewidth=0.8, label=r['name'])
    ax.set_title(f'BMP280 Filter (first {len(bmp_seg)} contiguous samples)', fontweight='bold')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)

    # 8. 训练时间
    ax = axes[1, 2]
    times = [r['training_time_sec'] for r in all_results]
    bars = ax.bar(names, times, color=colors, edgecolor='gray')
    ax.set_title('Training Time (s)', fontweight='bold')
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.0f}s', ha='center', va='bottom', fontsize=7)

    # 9. 误差分布对比（取 overall MAE 最小的前 4 个变体）
    ax = axes[1, 3]
    sorted_by_mae = sorted(all_results, key=lambda r: r['overall_mae'])
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
    print(f"对比图: {plot_path}")


def save_comparison_table(all_results):
    """保存文本格式对比表"""
    lines = []
    lines.append("=" * 90)
    lines.append("  BP Variants Comparison Report")
    lines.append(f"  Smooth Window = {SMOOTH_WINDOW}, Window Size = {WINDOW_SIZE}")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"{'Variant':<24} {'Params':>10} {'TFLite':>9} {'Overall':>16} "
                 f"{'MS5611':>16} {'BMP280':>16} {'Time':>8}")
    lines.append(f"{'':<24} {'':>10} {'':>9} {'RMSE/MAE(Pa)':>16} "
                 f"{'RMSE/MAE(Pa)':>16} {'RMSE/MAE(Pa)':>16} {'':>8}")
    lines.append("-" * 90)

    for r in all_results:
        overall = f"{r['overall_rmse']:.2f}/{r['overall_mae']:.2f}"
        ms5611 = f"{r['rmse_m']:.2f}/{r['mae_m']:.2f}" if r['rmse_m'] else "N/A"
        bmp280 = f"{r['rmse_b']:.2f}/{r['mae_b']:.2f}" if r['rmse_b'] else "N/A"
        lines.append(f"{r['name']:<24} {r['total_params']:>10,} "
                     f"{r['tflite_size_kb']:>7.1f}KB "
                     f"{overall:>16} {ms5611:>16} {bmp280:>16} "
                     f"{r['training_time_sec']:>6.1f}s")

    lines.append("-" * 90)
    lines.append("")
    lines.append("Legend:")
    lines.append("  v0: baseline       - 64→32→16 (original)")
    lines.append("  v1: residual       - output = current + residual (learn noise)")
    lines.append("  v2: deeper         - 128→96→64→48→32→16")
    lines.append("  v3: regularized    - BN + Dropout + L2 + LeakyReLU")
    lines.append("  v4: residual+deeper- residual + deeper")
    lines.append("  v5: deeper+regular - deeper + BN + Dropout + L2 + LeakyReLU")
    lines.append("  v6: full           - residual + deeper + regularization")
    lines.append("")

    report_path = os.path.join(models_dir, "bp_variants_comparison.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"对比报告: {report_path}")


if __name__ == "__main__":
    train_all_variants()
