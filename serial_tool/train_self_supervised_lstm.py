#!/usr/bin/env python
"""双通道自监督 LSTM 滤波训练：一个模型同时服务 MS5611 和 BMP280"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from train_self_supervised import load_all_self_supervised, create_sequences, SMOOTH_WINDOW

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_root = os.path.join(os.path.dirname(__file__), "data")
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

WINDOW_SIZE = 10
N_FEATURES = 2  # 气压值 + sensor_id


def create_sequences_dual(data, sensor_id, window_size=WINDOW_SIZE):
    """
    创建 LSTM/GRU 双通道输入序列：
    - X: (n_samples, window_size, 2)
        - 每个时间步: [归一化气压值, sensor_id]
        - sensor_id: 0=MS5611, 1=BMP280
    - y: 滑动平均后的当前值（归一化域）
    """
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(data, size=SMOOTH_WINDOW, mode='reflect')

    X, y = [], []
    for i in range(0, len(data) - window_size, 1):
        window_p = data[i:i + window_size]
        # 每个时间步: [气压值, sensor_id]
        feat = np.column_stack([window_p, np.full(window_size, sensor_id)])
        X.append(feat)
        center_idx = i + window_size - 1
        y.append(smooth[center_idx])

    return np.array(X), np.array(y)


def build_lstm_model(window_size, n_features):
    """构建 LSTM 双通道模型"""
    model = Sequential(name='dual_sensor_self_supervised_lstm')
    model.add(LSTM(32, return_sequences=True, input_shape=(window_size, n_features)))
    model.add(LSTM(16, return_sequences=False))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_gru_model(window_size, n_features):
    """构建 GRU 双通道模型"""
    model = Sequential(name='dual_sensor_self_supervised_gru')
    model.add(GRU(32, return_sequences=True, input_shape=(window_size, n_features)))
    model.add(GRU(16, return_sequences=False))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_dual_sensor_lstm(model_type='lstm'):
    """训练 LSTM 或 GRU 双通道模型"""
    build_fn = build_lstm_model if model_type == 'lstm' else build_gru_model
    model_name = f"dual_sensor_self_supervised_filter_{model_type}"

    print(f"\n{'='*60}")
    print(f"   双通道自监督滤波 {model_type.upper()} 模型")
    print(f"   输入：过去{WINDOW_SIZE}个含噪值 + sensor_id → 输出：滤波后当前值")
    print(f"{'='*60}")

    # =============================================
    # 1. 加载两个传感器的数据
    # =============================================
    print("\n--- 加载 MS5611 数据 ---")
    ms5611_p, ms5611_t, ms5611_meta = load_all_self_supervised(data_root, 'ms5611')
    print(f"\n--- 加载 BMP280 数据 ---")
    bmp280_p, bmp280_t, bmp280_meta = load_all_self_supervised(data_root, 'bmp280')

    # =============================================
    # 2. 使用相对气压
    # =============================================
    reference_pressure = 101325.0  # 标准海平面气压（1013.25 hPa）
    ms5611_rel = ms5611_p - reference_pressure
    bmp280_rel = bmp280_p - reference_pressure

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
    # 4. 创建 3D 序列 (samples, timesteps, features)
    # =============================================
    X_ms5611, y_ms5611 = create_sequences_dual(ms5611_scaled, sensor_id=0.0)
    X_bmp280, y_bmp280 = create_sequences_dual(bmp280_scaled, sensor_id=1.0)

    print(f"\n序列创建完成:")
    print(f"  MS5611: X={X_ms5611.shape}, y={y_ms5611.shape}")
    print(f"  BMP280: X={X_bmp280.shape}, y={y_bmp280.shape}")

    # =============================================
    # 5. 合并
    # =============================================
    X = np.concatenate([X_ms5611, X_bmp280])
    y = np.concatenate([y_ms5611, y_bmp280])
    print(f"\n合并后总序列: X={X.shape}, y={y.shape}")

    # =============================================
    # 6. 切分（分层采样）
    # =============================================
    sensor_labels = np.concatenate([
        np.zeros(len(X_ms5611)),
        np.ones(len(X_bmp280))
    ])

    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, sensor_labels, test_size=0.3, random_state=42, stratify=sensor_labels)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=s_temp)

    print(f"\n数据集划分:")
    print(f"  Train: {len(X_train)} (MS5611: {np.sum(s_train==0)}, BMP280: {np.sum(s_train==1)})")
    print(f"  Val:   {len(X_val)} (MS5611: {np.sum(s_val==0)}, BMP280: {np.sum(s_val==1)})")
    print(f"  Test:  {len(X_test)} (MS5611: {np.sum(s_test==0)}, BMP280: {np.sum(s_test==1)})")

    # =============================================
    # 7. 构建模型
    # =============================================
    model = build_fn(WINDOW_SIZE, N_FEATURES)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
    ]

    # =============================================
    # 8. 训练
    # =============================================
    history = model.fit(X_train, y_train, epochs=200, batch_size=32,
                        validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

    # =============================================
    # 9. 评估
    # =============================================
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)

    y_test_u = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_u = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(y_test_u, y_pred_u))
    mae_val = mean_absolute_error(y_test_u, y_pred_u)

    print(f"\n  总体: RMSE={rmse:.2f}Pa  MAE={mae_val:.2f}Pa")

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
    # 10. 导出模型
    # =============================================
    keras_path = os.path.join(models_dir, f"{model_name}.h5")
    tflite_path = os.path.join(models_dir, f"{model_name}.tflite")

    model.save(keras_path)
    print(f"  Keras: {keras_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"  TFLite: {tflite_path}")

    scaler_info = {
        'min': float(scaler.data_min_[0]),
        'max': float(scaler.data_max_[0]),
        'range': float(scaler.data_max_[0] - scaler.data_min_[0]),
        'reference_pressure': reference_pressure,
        'window_size': WINDOW_SIZE,
        'n_features': N_FEATURES,
        'model_type': model_type,
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

    # =============================================
    # 11. 可视化
    # =============================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f'Dual-Sensor Self-Supervised Filter {model_type.upper()}', fontsize=14)

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

    print(f"\n{'='*60}")
    print(f"  双通道 {model_type.upper()} 自监督滤波训练完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['lstm', 'gru'], default='lstm',
                        help='模型类型: lstm 或 gru')
    args = parser.parse_args()

    train_dual_sensor_lstm(args.type)

    print(f"\n{'='*60}")
    print(f"  双通道滤波训练全部完成！")
    print(f"{'='*60}")
    print(f"\n  models/ 目录下的模型：")
    for f in sorted(os.listdir(models_dir)):
        if f'dual_sensor_self_supervised_filter_{args.type}' in f:
            fpath = os.path.join(models_dir, f)
            print(f"    {f:50s} {os.path.getsize(fpath)/1024:>8.1f} KB")
