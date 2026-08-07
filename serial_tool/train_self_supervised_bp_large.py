#!/usr/bin/env python
"""双通道自监督 BP 滤波训练 - 加强版（更大模型）
结构: 128→64→32 (原版: 64→32→16)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from train_self_supervised import load_all_self_supervised, SMOOTH_WINDOW

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_root = os.path.join(os.path.dirname(__file__), "data")
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

WINDOW_SIZE = 10
INPUT_DIM = WINDOW_SIZE + 1  # 10 个气压值 + 1 个传感器标识


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


def build_large_bp(input_dim):
    """加强版 BP：128→64→32→1"""
    model = Sequential(name='self_supervised_bp_large')
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_dual_sensor_bp_large():
    model_name = "dual_sensor_self_supervised_filter_bp_large"
    print(f"\n{'='*60}")
    print(f"   双通道自监督滤波 BP 加强版 (128→64→32)")
    print(f"   平滑窗口={SMOOTH_WINDOW}, 输入：过去{WINDOW_SIZE}个含噪值 + sensor_id")
    print(f"{'='*60}")

    # 1. 加载数据
    print("\n--- 加载 MS5611 数据 ---")
    ms5611_p, _, _ = load_all_self_supervised(data_root, 'ms5611')
    print(f"\n--- 加载 BMP280 数据 ---")
    bmp280_p, _, _ = load_all_self_supervised(data_root, 'bmp280')

    # 2. 相对气压
    reference_pressure = 101325.0  # 标准海平面气压（1013.25 hPa）
    ms5611_rel = ms5611_p - reference_pressure
    bmp280_rel = bmp280_p - reference_pressure

    # 3. 合并归一化
    combined_rel = np.concatenate([ms5611_rel, bmp280_rel])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(combined_rel.reshape(-1, 1))
    ms5611_scaled = scaler.transform(ms5611_rel.reshape(-1, 1)).flatten()
    bmp280_scaled = scaler.transform(bmp280_rel.reshape(-1, 1)).flatten()

    print(f"\n合并归一化: range={scaler.data_max_[0] - scaler.data_min_[0]:.2f}")

    # 4. 创建序列
    X_m, y_m = create_sequences_with_sensor(ms5611_scaled, sensor_id=0.0)
    X_b, y_b = create_sequences_with_sensor(bmp280_scaled, sensor_id=1.0)
    print(f"\n序列: MS5611={X_m.shape}, BMP280={X_b.shape}")

    # 5. 合并
    X = np.concatenate([X_m, X_b])
    y = np.concatenate([y_m, y_b])

    # 6. 切分
    sensor_labels = np.concatenate([np.zeros(len(X_m)), np.ones(len(X_b))])
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, sensor_labels, test_size=0.3, random_state=42, stratify=sensor_labels)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=s_temp)

    # 7. 构建加强版模型
    model = build_large_bp(INPUT_DIM)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
    ]

    # 8. 训练
    history = model.fit(X_train, y_train, epochs=200, batch_size=32,
                        validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

    # 9. 评估
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

    # 10. 导出
    keras_path = os.path.join(models_dir, f"{model_name}.h5")
    tflite_path = os.path.join(models_dir, f"{model_name}.tflite")
    model.save(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"  Keras: {keras_path}")
    print(f"  TFLite: {tflite_path}")

    scaler_info = {
        'min': float(scaler.data_min_[0]),
        'max': float(scaler.data_max_[0]),
        'range': float(scaler.data_max_[0] - scaler.data_min_[0]),
        'reference_pressure': reference_pressure,
        'window_size': WINDOW_SIZE,
        'input_dim': INPUT_DIM,
        'model_type': 'bp_large',
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

    # 11. 可视化
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f'Dual-Sensor Self-Supervised Filter BP Large (SMOOTH={SMOOTH_WINDOW})', fontsize=14)
    axes[0, 0].plot(history.history['loss'], label='Train')
    axes[0, 0].plot(history.history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].plot(history.history['mae'], label='Train')
    axes[0, 1].plot(history.history['val_mae'], label='Val')
    axes[0, 1].set_title('MAE'); axes[0, 1].legend(); axes[0, 1].grid(True)
    if np.any(ms5611_mask):
        n_show = min(200, np.sum(ms5611_mask))
        axes[0, 2].plot(y_test_u[ms5611_mask][:n_show], label='Target', alpha=0.7)
        axes[0, 2].plot(y_pred_u[ms5611_mask][:n_show], label='Output', alpha=0.7, linestyle='--')
        axes[0, 2].set_title('MS5611 Filter Result'); axes[0, 2].legend(); axes[0, 2].grid(True)
        err_m = y_test_u[ms5611_mask] - y_pred_u[ms5611_mask]
        axes[1, 0].hist(err_m, bins=50, alpha=0.7, color='steelblue')
        axes[1, 0].axvline(x=0, color='r', ls='--', alpha=0.5)
        axes[1, 0].set_title(f'MS5611 Error (MAE={mae_m:.2f}Pa)'); axes[1, 0].grid(True)
    if np.any(bmp280_mask):
        n_show = min(200, np.sum(bmp280_mask))
        axes[1, 1].plot(y_test_u[bmp280_mask][:n_show], label='Target', alpha=0.7)
        axes[1, 1].plot(y_pred_u[bmp280_mask][:n_show], label='Output', alpha=0.7, linestyle='--')
        axes[1, 1].set_title('BMP280 Filter Result'); axes[1, 1].legend(); axes[1, 1].grid(True)
        err_b = y_test_u[bmp280_mask] - y_pred_u[bmp280_mask]
        axes[1, 2].hist(err_b, bins=50, alpha=0.7, color='orange')
        axes[1, 2].axvline(x=0, color='r', ls='--', alpha=0.5)
        axes[1, 2].set_title(f'BMP280 Error (MAE={mae_b:.2f}Pa)'); axes[1, 2].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, f"{model_name}_training.png"), dpi=150)
    plt.close()
    print(f"  Plot saved")
    print(f"\n{'='*60}")
    print(f"  双通道 BP 加强版训练完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_dual_sensor_bp_large()
    print(f"\n  models/ 目录下的模型：")
    for f in sorted(os.listdir(models_dir)):
        if 'bp_large' in f:
            fpath = os.path.join(models_dir, f)
            print(f"    {f:50s} {os.path.getsize(fpath)/1024:>8.1f} KB")
