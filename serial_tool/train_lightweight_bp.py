#!/usr/bin/env python
"""
轻量级双通道自监督 BP 滤波训练
架构：11→32→16→1（3 层全连接）
目标：比 baseline (64→32→16→1, 3.4K 参数) 更轻量，适合 MCU 快速推理

架构说明：
  - 输入：10 个历史气压值 + 1 个 sensor_id（0=MS5611, 1=BMP280）
  - 隐藏层 1：32 神经元（ReLU）
  - 隐藏层 2：16 神经元（ReLU）
  - 输出层：1 神经元（线性）
  - 总参数量：(11*32 + 32) + (32*16 + 16) + (16*1 + 1) = 352+32 + 512+16 + 16+1 = 929 参数
  - TFLite 大小：约 3.7KB
  - 推理 MACC：11*32 + 32*16 + 16*1 = 352 + 512 + 16 = 880 MACC
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from train_self_supervised import load_all_self_supervised, SMOOTH_WINDOW

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.ndimage import uniform_filter1d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================
# 配置
# =============================================
data_root = os.path.join(os.path.dirname(__file__), "data")
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

WINDOW_SIZE = 10
INPUT_DIM = WINDOW_SIZE + 1  # 10 个气压值 + 1 个 sensor_id
HIDDEN1 = 32
HIDDEN2 = 16
EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 20
REFERENCE_PRESSURE = 101325.0  # 标准海平面气压（1013.25 hPa）


def create_sequences_with_sensor(data, sensor_id, window_size=WINDOW_SIZE):
    """创建双通道输入序列"""
    smooth = uniform_filter1d(data, size=SMOOTH_WINDOW, mode='reflect')

    X, y = [], []
    for i in range(0, len(data) - window_size, 1):
        window = data[i:i + window_size]
        feat = np.append(window, sensor_id)
        X.append(feat)
        center_idx = i + window_size - 1
        y.append(smooth[center_idx])

    return np.array(X), np.array(y)


def build_lightweight_bp(input_dim):
    """构建轻量级 3 层 BP 模型：input_dim → 32 → 16 → 1"""
    model = Sequential(name='lightweight_bp')
    model.add(Dense(HIDDEN1, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(HIDDEN2, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def compute_params(input_dim, h1, h2):
    """计算总参数量"""
    return (input_dim * h1 + h1) + (h1 * h2 + h2) + (h2 * 1 + 1)


def train_lightweight_bp():
    print(f"\n{'='*60}")
    print(f"   轻量级双通道自监督滤波 BP 模型")
    print(f"   架构: {INPUT_DIM} → {HIDDEN1} → {HIDDEN2} → 1")
    total_params = compute_params(INPUT_DIM, HIDDEN1, HIDDEN2)
    print(f"   参数量: {total_params} (baseline 为 3393)")
    print(f"   推理 MACC: {INPUT_DIM * HIDDEN1 + HIDDEN1 * HIDDEN2 + HIDDEN2 * 1}")
    print(f"{'='*60}")

    # =============================================
    # 1. 加载数据
    # =============================================
    print("\n--- 加载 MS5611 数据 ---")
    ms5611_p, ms5611_t, ms5611_meta = load_all_self_supervised(data_root, 'ms5611')
    print(f"\n--- 加载 BMP280 数据 ---")
    bmp280_p, bmp280_t, bmp280_meta = load_all_self_supervised(data_root, 'bmp280')

    # =============================================
    # 2. 相对气压
    # =============================================
    ms5611_rel = ms5611_p - REFERENCE_PRESSURE
    bmp280_rel = bmp280_p - REFERENCE_PRESSURE

    print(f"\nMS5611 相对气压: min={ms5611_rel.min():.2f}, max={ms5611_rel.max():.2f}, "
          f"std={ms5611_rel.std():.3f}")
    print(f"BMP280 相对气压: min={bmp280_rel.min():.2f}, max={bmp280_rel.max():.2f}, "
          f"std={bmp280_rel.std():.3f}")

    # =============================================
    # 3. 合并归一化
    # =============================================
    combined_rel = np.concatenate([ms5611_rel, bmp280_rel])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(combined_rel.reshape(-1, 1))

    ms5611_scaled = scaler.transform(ms5611_rel.reshape(-1, 1)).flatten()
    bmp280_scaled = scaler.transform(bmp280_rel.reshape(-1, 1)).flatten()

    print(f"\n合并归一化:")
    print(f"  整体 min={scaler.data_min_[0]:.2f}, max={scaler.data_max_[0]:.2f}, "
          f"range={scaler.data_max_[0] - scaler.data_min_[0]:.2f}")

    # =============================================
    # 4. 创建序列
    # =============================================
    X_ms5611, y_ms5611 = create_sequences_with_sensor(ms5611_scaled, sensor_id=0.0)
    X_bmp280, y_bmp280 = create_sequences_with_sensor(bmp280_scaled, sensor_id=1.0)

    print(f"\n序列创建完成:")
    print(f"  MS5611: X={X_ms5611.shape}, y={y_ms5611.shape}")
    print(f"  BMP280: X={X_bmp280.shape}, y={y_bmp280.shape}")

    X = np.concatenate([X_ms5611, X_bmp280])
    y = np.concatenate([y_ms5611, y_bmp280])
    sensor_labels = np.concatenate([
        np.zeros(len(X_ms5611)),
        np.ones(len(X_bmp280))
    ])

    print(f"\n合并后总序列: X={X.shape}, y={y.shape}")

    # =============================================
    # 5. 数据集划分（分层）
    # =============================================
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, sensor_labels, test_size=0.3, random_state=42, stratify=sensor_labels)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=s_temp)

    print(f"\n数据集划分:")
    print(f"  Train: {len(X_train)} (MS5611: {np.sum(s_train==0)}, BMP280: {np.sum(s_train==1)})")
    print(f"  Val:   {len(X_val)} (MS5611: {np.sum(s_val==0)}, BMP280: {np.sum(s_val==1)})")
    print(f"  Test:  {len(X_test)} (MS5611: {np.sum(s_test==0)}, BMP280: {np.sum(s_test==1)})")

    # =============================================
    # 6. 构建模型
    # =============================================
    model = build_lightweight_bp(INPUT_DIM)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    # =============================================
    # 7. 训练
    # =============================================
    print("\n开始训练...")
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val), callbacks=callbacks, verbose=2)

    # =============================================
    # 8. 评估
    # =============================================
    y_pred = model.predict(X_test, verbose=0)

    y_test_u = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_u = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    overall_rmse = np.sqrt(mean_squared_error(y_test_u, y_pred_u))
    overall_mae = mean_absolute_error(y_test_u, y_pred_u)
    print(f"\n  总体: RMSE={overall_rmse:.2f}Pa  MAE={overall_mae:.2f}Pa")

    ms5611_mask = s_test == 0
    bmp280_mask = s_test == 1

    if np.any(ms5611_mask):
        rmse_m = np.sqrt(mean_squared_error(y_test_u[ms5611_mask], y_pred_u[ms5611_mask]))
        mae_m = mean_absolute_error(y_test_u[ms5611_mask], y_pred_u[ms5611_mask])
        print(f"  MS5611: RMSE={rmse_m:.2f}Pa  MAE={mae_m:.2f}Pa")

    if np.any(bmp280_mask):
        rmse_b = np.sqrt(mean_squared_error(y_test_u[bmp280_mask], y_pred_u[bmp280_mask]))
        mae_b = mean_absolute_error(y_test_u[bmp280_mask], y_pred_u[bmp280_mask])
        print(f"  BMP280: RMSE={rmse_b:.2f}Pa  MAE={mae_b:.2f}Pa")

    # =============================================
    # 9. 导出模型
    # =============================================
    model_name = "lightweight_bp_11x32x16x1"
    keras_path = os.path.join(models_dir, f"{model_name}.h5")
    tflite_path = os.path.join(models_dir, f"{model_name}.tflite")

    model.save(keras_path)
    print(f"\n  Keras: {keras_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    tflite_kb = len(tflite_model) / 1024
    print(f"  TFLite: {tflite_path} ({tflite_kb:.1f} KB)")

    # Scaler 信息
    scaler_info = {
        'min': float(scaler.data_min_[0]),
        'max': float(scaler.data_max_[0]),
        'range': float(scaler.data_max_[0] - scaler.data_min_[0]),
        'reference_pressure': REFERENCE_PRESSURE,
        'window_size': WINDOW_SIZE,
        'input_dim': INPUT_DIM,
        'architecture': f'{INPUT_DIM}x{HIDDEN1}x{HIDDEN2}x1',
        'total_params': total_params,
        'tflite_size_kb': round(tflite_kb, 1),
        'model_type': 'lightweight_bp',
        'sensor': 'Dual(MS5611+BMP280)',
        'mode': 'self_supervised_filter_dual',
        'input_is_relative': True,
        'smooth_window': SMOOTH_WINDOW,
        'sensor_id_ms5611': 0.0,
        'sensor_id_bmp280': 1.0,
        'model_mode': 'self_supervised'
    }
    with open(os.path.join(models_dir, f"{model_name}_scaler.json"), 'w') as f:
        json.dump(scaler_info, f, indent=2)
    print(f"  Scaler: {os.path.join(models_dir, f'{model_name}_scaler.json')}")

    # =============================================
    # 10. 导出嵌入式 C 头文件（量化 int8 权重）
    # =============================================
    export_c_header(model, scaler_info, models_dir, model_name)

    # =============================================
    # 11. 可视化
    # =============================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f'Lightweight BP ({INPUT_DIM}→{HIDDEN1}→{HIDDEN2}→1, {total_params} params, {tflite_kb:.1f}KB)',
                 fontsize=14)

    axes[0, 0].plot(history.history['loss'], label='Train')
    axes[0, 0].plot(history.history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history.history['mae'], label='Train')
    axes[0, 1].plot(history.history['val_mae'], label='Val')
    axes[0, 1].set_title('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    if np.any(ms5611_mask):
        n_show = min(200, np.sum(ms5611_mask))
        axes[0, 2].plot(y_test_u[ms5611_mask][:n_show], label='Filter Target', alpha=0.7)
        axes[0, 2].plot(y_pred_u[ms5611_mask][:n_show], label='Model Output', alpha=0.7, linestyle='--')
        axes[0, 2].set_title('MS5611 Filter Result')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        err_m = y_test_u[ms5611_mask] - y_pred_u[ms5611_mask]
        axes[1, 0].hist(err_m, bins=50, alpha=0.7, color='steelblue')
        axes[1, 0].axvline(x=0, color='r', ls='--', alpha=0.5)
        axes[1, 0].set_title(f'MS5611 Error (MAE={mae_m:.2f}Pa)')
        axes[1, 0].grid(True)

    if np.any(bmp280_mask):
        n_show = min(200, np.sum(bmp280_mask))
        axes[1, 1].plot(y_test_u[bmp280_mask][:n_show], label='Filter Target', alpha=0.7)
        axes[1, 1].plot(y_pred_u[bmp280_mask][:n_show], label='Model Output', alpha=0.7, linestyle='--')
        axes[1, 1].set_title('BMP280 Filter Result')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        err_b = y_test_u[bmp280_mask] - y_pred_u[bmp280_mask]
        axes[1, 2].hist(err_b, bins=50, alpha=0.7, color='orange')
        axes[1, 2].axvline(x=0, color='r', ls='--', alpha=0.5)
        axes[1, 2].set_title(f'BMP280 Error (MAE={mae_b:.2f}Pa)')
        axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, f"{model_name}_training.png"), dpi=150)
    plt.close()
    print(f"  Plot saved")

    # =============================================
    # 12. 对比汇总
    # =============================================
    print(f"\n{'='*60}")
    print(f"  轻量级 BP 训练完成！")
    print(f"  {INPUT_DIM}→{HIDDEN1}→{HIDDEN2}→1 | {total_params} params | {tflite_kb:.1f} KB")
    print(f"  总体 RMSE={overall_rmse:.2f}Pa  MAE={overall_mae:.2f}Pa")
    print(f"  MS5611 RMSE={rmse_m:.2f}Pa  MAE={mae_m:.2f}Pa")
    print(f"  BMP280 RMSE={rmse_b:.2f}Pa  MAE={mae_b:.2f}Pa")
    print(f"  对比 baseline (64→32→16→1, 3393 params, 15.6KB):")
    print(f"    参数量: {total_params/3393*100:.1f}%")
    print(f"    模型大小: {tflite_kb/15.6*100:.1f}%")
    print(f"{'='*60}")


def export_c_header(model, scaler_info, models_dir, model_name):
    """导出嵌入式 C 头文件（float32 权重，便于 MCU 直接加载）"""
    weights = model.get_weights()

    c_header = f"""/*
 * Lightweight BP Model - {model_name}
 * Architecture: {scaler_info['architecture']}
 * Total params: {scaler_info['total_params']}
 * TFLite size: {scaler_info['tflite_size_kb']} KB
 *
 * Generated automatically - DO NOT EDIT
 */

#ifndef __LIGHTWEIGHT_BP_WEIGHTS_H
#define __LIGHTWEIGHT_BP_WEIGHTS_H

#include <stdint.h>

/* Model parameters */
#define LW_BP_INPUT_DIM    {scaler_info['input_dim']}
#define LW_BP_HIDDEN1      {HIDDEN1}
#define LW_BP_HIDDEN2      {HIDDEN2}
#define LW_BP_WINDOW_SIZE  {WINDOW_SIZE}
#define LW_BP_SMOOTH_WINDOW {SMOOTH_WINDOW}
#define LW_BP_REFERENCE_PRESSURE {REFERENCE_PRESSURE}f
#define LW_BP_INPUT_MIN    {scaler_info['min']}f
#define LW_BP_INPUT_MAX    {scaler_info['max']}f
#define LW_BP_INPUT_RANGE  {scaler_info['range']}f

/* Sensor IDs */
#define LW_BP_SENSOR_MS5611 0.0f
#define LW_BP_SENSOR_BMP280 1.0f

/* Layer 1: {scaler_info['input_dim']} x {HIDDEN1} */
static const float lw_bp_w1[{scaler_info['input_dim']}][{HIDDEN1}] = {{
"""
    # 权重矩阵转置为 [input_dim][hidden1] 格式
    w1 = weights[0]  # shape: [input_dim, hidden1]
    for i in range(w1.shape[0]):
        c_header += "    {" + ", ".join(f"{w1[i,j]:.10f}f" for j in range(w1.shape[1])) + "},\n"
    c_header += """};

static const float lw_bp_b1[32] = {
"""
    b1 = weights[1]
    c_header += "    " + ", ".join(f"{b1[j]:.10f}f" for j in range(b1.shape[0])) + "\n"
    c_header += """};

/* Layer 2: 32 x 16 */
static const float lw_bp_w2[32][16] = {
"""
    w2 = weights[2]  # shape: [32, 16]
    for i in range(w2.shape[0]):
        c_header += "    {" + ", ".join(f"{w2[i,j]:.10f}f" for j in range(w2.shape[1])) + "},\n"
    c_header += """};

static const float lw_bp_b2[16] = {
"""
    b2 = weights[3]
    c_header += "    " + ", ".join(f"{b2[j]:.10f}f" for j in range(b2.shape[0])) + "\n"
    c_header += """};

/* Layer 3: 16 x 1 */
static const float lw_bp_w3[16][1] = {
"""
    w3 = weights[4]  # shape: [16, 1]
    for i in range(w3.shape[0]):
        c_header += "    {" + f"{w3[i,0]:.10f}f" + "},\n"
    c_header += """};

static const float lw_bp_b3[1] = {""" + f"{weights[5][0]:.10f}f" + """};

#endif /* __LIGHTWEIGHT_BP_WEIGHTS_H */
"""

    header_path = os.path.join(models_dir, f"{model_name}_weights.h")
    with open(header_path, 'w') as f:
        f.write(c_header)
    print(f"  C Header: {header_path}")


if __name__ == "__main__":
    train_lightweight_bp()
