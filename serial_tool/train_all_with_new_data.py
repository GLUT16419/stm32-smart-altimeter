#!/usr/bin/env python
"""
一键训练所有 NN 滤波模型（使用包含静止/平移/升降的完整数据集）
训练完成后自动更新 MCU 端 main.c 中的 scaler 参数。

训练模型列表：
1. 双通道 BP (baseline)        - dual_sensor_self_supervised_filter_bp_v6
2. 双通道 BP 加强版             - dual_sensor_self_supervised_filter_bp_large_v6
3. 双通道轻量 BP                - lightweight_bp_11x32x16x1_v6
4. 双通道 LSTM                  - dual_sensor_self_supervised_filter_lstm_v6
5. 双通道 GRU                   - dual_sensor_self_supervised_filter_gru_v6
6. 单传感器 MS5611 BP           - ms5611_self_supervised_filter_bp_v6
7. 单传感器 BMP280 BP           - bmp280_self_supervised_filter_bp_v6
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.ndimage import uniform_filter1d

# =============================================
# 全局配置
# =============================================
data_root = os.path.join(os.path.dirname(__file__), "data")
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

SMOOTH_WINDOW = 9       # 滑动平均窗口（自监督目标）
WINDOW_SIZE = 10         # 输入窗口大小
INPUT_DIM = WINDOW_SIZE + 1  # 10气压 + 1 sensor_id
REFERENCE_PRESSURE = 101325.0
EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 20

# =============================================
# 数据加载
# =============================================
def load_all_data(data_root):
    """加载所有数据（height_xxx + raw/静止/平移运动/升降运动）"""
    import pandas as pd

    all_ms5611_p = []
    all_bmp280_p = []

    # 1. 加载 height_xxx 文件夹下的 self_supervised 数据
    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path) or folder == 'raw':
            continue
        for f in sorted(os.listdir(folder_path)):
            if f.startswith('self_supervised_ms5611') and f.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, f))
                all_ms5611_p.extend(df['pressure_pa'].values.tolist())
            if f.startswith('self_supervised_bmp280') and f.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, f))
                all_bmp280_p.extend(df['pressure_pa'].values.tolist())

    # 2. 加载 raw 场景数据
    raw_dir = os.path.join(data_root, 'raw')
    if os.path.isdir(raw_dir):
        for scene in sorted(os.listdir(raw_dir)):
            scene_path = os.path.join(raw_dir, scene)
            if not os.path.isdir(scene_path):
                continue
            for f in sorted(os.listdir(scene_path)):
                if f.endswith('.csv'):
                    df = pd.read_csv(os.path.join(scene_path, f))
                    if 'ms5611' in f:
                        all_ms5611_p.extend(df['pressure_pa'].values.tolist())
                    elif 'bmp280' in f:
                        all_bmp280_p.extend(df['pressure_pa'].values.tolist())

    ms5611_p = np.array(all_ms5611_p)
    bmp280_p = np.array(all_bmp280_p)

    print(f"\nMS5611 总数据: {len(ms5611_p)} 条")
    print(f"  pressure: [{ms5611_p.min():.2f}, {ms5611_p.max():.2f}], mean={ms5611_p.mean():.2f}")
    print(f"  rel_pressure: [{ms5611_p.min()-REFERENCE_PRESSURE:.2f}, {ms5611_p.max()-REFERENCE_PRESSURE:.2f}]")

    print(f"BMP280 总数据: {len(bmp280_p)} 条")
    print(f"  pressure: [{bmp280_p.min():.2f}, {bmp280_p.max():.2f}], mean={bmp280_p.mean():.2f}")
    print(f"  rel_pressure: [{bmp280_p.min()-REFERENCE_PRESSURE:.2f}, {bmp280_p.max()-REFERENCE_PRESSURE:.2f}]")

    return ms5611_p, bmp280_p


def create_sequences_bp(data, sensor_id, window_size=WINDOW_SIZE):
    """创建 BP 输入序列 (features=window_size + sensor_id)"""
    smooth = uniform_filter1d(data, size=SMOOTH_WINDOW, mode='reflect')
    X, y = [], []
    for i in range(0, len(data) - window_size, 1):
        window = data[i:i + window_size]
        feat = np.append(window, sensor_id)
        X.append(feat)
        center_idx = i + window_size - 1
        y.append(smooth[center_idx])
    return np.array(X), np.array(y)


def create_sequences_lstm(data, sensor_id, window_size=WINDOW_SIZE):
    """创建 LSTM/GRU 输入序列 (samples, timesteps, features=[pressure, sensor_id])"""
    smooth = uniform_filter1d(data, size=SMOOTH_WINDOW, mode='reflect')
    X, y = [], []
    for i in range(0, len(data) - window_size, 1):
        window_p = data[i:i + window_size]
        feat = np.column_stack([window_p, np.full(window_size, sensor_id)])
        X.append(feat)
        center_idx = i + window_size - 1
        y.append(smooth[center_idx])
    return np.array(X), np.array(y)


# =============================================
# 模型构建
# =============================================
def build_baseline_bp(input_dim):
    """baseline BP: 64→32→16→1"""
    model = Sequential(name='bp_baseline')
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_large_bp(input_dim):
    """加强 BP: 128→64→32→1"""
    model = Sequential(name='bp_large')
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_lightweight_bp(input_dim):
    """轻量 BP: input_dim→32→16→1"""
    model = Sequential(name='bp_lightweight')
    model.add(Dense(32, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_lstm_model(window_size, n_features):
    """LSTM: 32→16→1"""
    model = Sequential(name='lstm_dual')
    model.add(LSTM(32, return_sequences=True, input_shape=(window_size, n_features)))
    model.add(LSTM(16, return_sequences=False))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_gru_model(window_size, n_features):
    """GRU: 32→16→1"""
    model = Sequential(name='gru_dual')
    model.add(GRU(32, return_sequences=True, input_shape=(window_size, n_features)))
    model.add(GRU(16, return_sequences=False))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# =============================================
# 训练函数
# =============================================
def prepare_data(data, scaler, sensor_id):
    """准备单个传感器的训练数据"""
    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()
    X, y = create_sequences_bp(data_scaled, sensor_id)
    return X, y


def prepare_data_lstm(data, scaler, sensor_id):
    """准备 LSTM/GRU 训练数据"""
    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()
    X, y = create_sequences_lstm(data_scaled, sensor_id)
    return X, y


def train_and_save(model_fn, model_name, X_train, y_train, X_val, y_val,
                   X_test, y_test, s_test, scaler, sensor_names=None):
    """训练并保存模型"""
    print(f"\n{'='*60}")
    print(f"  训练: {model_name}")
    print(f"{'='*60}")

    model = model_fn()
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]

    start = time.time()
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)
    elapsed = time.time() - start

    # 评估
    y_pred = model.predict(X_test, verbose=0)
    y_test_u = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_u = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    overall_rmse = np.sqrt(mean_squared_error(y_test_u, y_pred_u))
    overall_mae = mean_absolute_error(y_test_u, y_pred_u)

    results = {'overall': {'rmse': overall_rmse, 'mae': overall_mae}}

    print(f"\n  总体: RMSE={overall_rmse:.2f}Pa  MAE={overall_mae:.2f}Pa  "
          f"耗时={elapsed:.1f}s")

    if s_test is not None:
        for sid, sname in [(0.0, 'MS5611'), (1.0, 'BMP280')]:
            mask = s_test == sid
            if np.any(mask):
                rmse_s = np.sqrt(mean_squared_error(y_test_u[mask], y_pred_u[mask]))
                mae_s = mean_absolute_error(y_test_u[mask], y_pred_u[mask])
                results[sname] = {'rmse': rmse_s, 'mae': mae_s}
                print(f"  {sname}: RMSE={rmse_s:.2f}Pa  MAE={mae_s:.2f}Pa")

    # 保存模型
    keras_path = os.path.join(models_dir, f"{model_name}.h5")
    tflite_path = os.path.join(models_dir, f"{model_name}.tflite")

    model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    tflite_kb = len(tflite_model) / 1024
    print(f"  Keras: {keras_path}")
    print(f"  TFLite: {tflite_path} ({tflite_kb:.1f} KB)")

    # 保存 scaler 信息
    scaler_info = {
        'min': float(scaler.data_min_[0]),
        'max': float(scaler.data_max_[0]),
        'range': float(scaler.data_max_[0] - scaler.data_min_[0]),
        'reference_pressure': REFERENCE_PRESSURE,
        'window_size': WINDOW_SIZE,
        'model_type': model.name,
        'sensor': 'Dual(MS5611+BMP280)' if sensor_names is None else sensor_names,
        'mode': 'self_supervised_filter',
        'input_is_relative': True,
        'smooth_window': SMOOTH_WINDOW,
        'model_mode': 'self_supervised',
    }
    if sensor_names is None:
        scaler_info['sensor_id_ms5611'] = 0.0
        scaler_info['sensor_id_bmp280'] = 1.0
        scaler_info['mode'] = 'self_supervised_filter_dual'

    with open(os.path.join(models_dir, f"{model_name}_scaler.json"), 'w') as f:
        json.dump(scaler_info, f, indent=2)

    # 训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'{model_name}', fontsize=14)

    axes[0, 0].plot(history.history['loss'], label='Train')
    axes[0, 0].plot(history.history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(history.history['mae'], label='Train')
    axes[0, 1].plot(history.history['val_mae'], label='Val')
    axes[0, 1].set_title('MAE'); axes[0, 1].legend(); axes[0, 1].grid(True)

    n_show = min(200, len(y_test_u))
    axes[1, 0].plot(y_test_u[:n_show], label='Target', alpha=0.7)
    axes[1, 0].plot(y_pred_u[:n_show], label='Output', alpha=0.7, linestyle='--')
    axes[1, 0].set_title('Filter Result'); axes[1, 0].legend(); axes[1, 0].grid(True)

    errors = y_test_u - y_pred_u
    axes[1, 1].hist(errors, bins=50, alpha=0.7, color='steelblue')
    axes[1, 1].axvline(x=0, color='r', ls='--', alpha=0.5)
    axes[1, 1].set_title(f'Error (MAE={overall_mae:.2f}Pa)'); axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, f"{model_name}_training.png"), dpi=150)
    plt.close()
    print(f"  Plot saved")

    return results


# =============================================
# 训练所有模型
# =============================================
def train_all():
    print("=" * 70)
    print("  训练所有 NN 滤波模型（含静止/平移/升降完整数据集）")
    print("=" * 70)

    # 1. 加载数据
    print("\n--- 加载数据 ---")
    ms5611_p, bmp280_p = load_all_data(data_root)

    ms5611_rel = ms5611_p - REFERENCE_PRESSURE
    bmp280_rel = bmp280_p - REFERENCE_PRESSURE

    # 2. 合并归一化
    combined_rel = np.concatenate([ms5611_rel, bmp280_rel])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(combined_rel.reshape(-1, 1))

    print(f"\n合并归一化:")
    print(f"  min={scaler.data_min_[0]:.2f}, max={scaler.data_max_[0]:.2f}, "
          f"range={scaler.data_max_[0] - scaler.data_min_[0]:.2f}")

    ms5611_scaled = scaler.transform(ms5611_rel.reshape(-1, 1)).flatten()
    bmp280_scaled = scaler.transform(bmp280_rel.reshape(-1, 1)).flatten()

    print(f"  MS5611 归一化: [{ms5611_scaled.min():.4f}, {ms5611_scaled.max():.4f}]")
    print(f"  BMP280 归一化: [{bmp280_scaled.min():.4f}, {bmp280_scaled.max():.4f}]")

    # =============================================
    # 模型 1: 双通道 BP (baseline v6)
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 1/7: 双通道 BP (baseline) - v6")
    print(f"{'#'*70}")

    X_m, y_m = create_sequences_bp(ms5611_scaled, sensor_id=0.0)
    X_b, y_b = create_sequences_bp(bmp280_scaled, sensor_id=1.0)
    X = np.concatenate([X_m, X_b])
    y = np.concatenate([y_m, y_b])
    sensor_labels = np.concatenate([np.zeros(len(X_m)), np.ones(len(X_b))])

    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, sensor_labels, test_size=0.3, random_state=42, stratify=sensor_labels)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=s_temp)

    print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    train_and_save(
        lambda: build_baseline_bp(INPUT_DIM),
        "dual_sensor_self_supervised_filter_bp_v6",
        X_train, y_train, X_val, y_val, X_test, y_test, s_test, scaler
    )

    # =============================================
    # 模型 2: 双通道 BP 加强版 v6
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 2/7: 双通道 BP 加强版 - v6")
    print(f"{'#'*70}")

    train_and_save(
        lambda: build_large_bp(INPUT_DIM),
        "dual_sensor_self_supervised_filter_bp_large_v6",
        X_train, y_train, X_val, y_val, X_test, y_test, s_test, scaler
    )

    # =============================================
    # 模型 3: 轻量 BP v6
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 3/7: 轻量 BP - v6")
    print(f"{'#'*70}")

    train_and_save(
        lambda: build_lightweight_bp(INPUT_DIM),
        "lightweight_bp_11x32x16x1_v6",
        X_train, y_train, X_val, y_val, X_test, y_test, s_test, scaler
    )

    # =============================================
    # 模型 4: 双通道 LSTM v6
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 4/7: 双通道 LSTM - v6")
    print(f"{'#'*70}")

    N_FEATURES = 2
    X_m_lstm, y_m_lstm = create_sequences_lstm(ms5611_scaled, sensor_id=0.0)
    X_b_lstm, y_b_lstm = create_sequences_lstm(bmp280_scaled, sensor_id=1.0)
    X_lstm = np.concatenate([X_m_lstm, X_b_lstm])
    y_lstm = np.concatenate([y_m_lstm, y_b_lstm])
    s_lstm = np.concatenate([np.zeros(len(X_m_lstm)), np.ones(len(X_b_lstm))])

    X_train_l, X_temp_l, y_train_l, y_temp_l, s_train_l, s_temp_l = train_test_split(
        X_lstm, y_lstm, s_lstm, test_size=0.3, random_state=42, stratify=s_lstm)
    X_val_l, X_test_l, y_val_l, y_test_l, s_val_l, s_test_l = train_test_split(
        X_temp_l, y_temp_l, s_temp_l, test_size=0.5, random_state=42, stratify=s_temp_l)

    print(f"  训练集: {len(X_train_l)}, 验证集: {len(X_val_l)}, 测试集: {len(X_test_l)}")

    train_and_save(
        lambda: build_lstm_model(WINDOW_SIZE, N_FEATURES),
        "dual_sensor_self_supervised_filter_lstm_v6",
        X_train_l, y_train_l, X_val_l, y_val_l, X_test_l, y_test_l, s_test_l, scaler
    )

    # =============================================
    # 模型 5: 双通道 GRU v6
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 5/7: 双通道 GRU - v6")
    print(f"{'#'*70}")

    train_and_save(
        lambda: build_gru_model(WINDOW_SIZE, N_FEATURES),
        "dual_sensor_self_supervised_filter_gru_v6",
        X_train_l, y_train_l, X_val_l, y_val_l, X_test_l, y_test_l, s_test_l, scaler
    )

    # =============================================
    # 模型 6: 单传感器 MS5611 BP v6
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 6/7: 单传感器 MS5611 BP - v6")
    print(f"{'#'*70}")

    X_m_single, y_m_single = create_sequences_bp(ms5611_scaled, sensor_id=0.0)
    X_train_m, X_temp_m, y_train_m, y_temp_m = train_test_split(
        X_m_single, y_m_single, test_size=0.3, random_state=42)
    X_val_m, X_test_m, y_val_m, y_test_m = train_test_split(
        X_temp_m, y_temp_m, test_size=0.5, random_state=42)

    # 注意：单传感器模型也用 11 维输入（sensor_id 始终为 0）
    train_and_save(
        lambda: build_baseline_bp(INPUT_DIM),
        "ms5611_self_supervised_filter_bp_v6",
        X_train_m, y_train_m, X_val_m, y_val_m, X_test_m, y_test_m, None, scaler,
        sensor_names='MS5611'
    )

    # =============================================
    # 模型 7: 单传感器 BMP280 BP v6
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 7/7: 单传感器 BMP280 BP - v6")
    print(f"{'#'*70}")

    X_b_single, y_b_single = create_sequences_bp(bmp280_scaled, sensor_id=1.0)
    X_train_b, X_temp_b, y_train_b, y_temp_b = train_test_split(
        X_b_single, y_b_single, test_size=0.3, random_state=42)
    X_val_b, X_test_b, y_val_b, y_test_b = train_test_split(
        X_temp_b, y_temp_b, test_size=0.5, random_state=42)

    train_and_save(
        lambda: build_baseline_bp(INPUT_DIM),
        "bmp280_self_supervised_filter_bp_v6",
        X_train_b, y_train_b, X_val_b, y_val_b, X_test_b, y_test_b, None, scaler,
        sensor_names='BMP280'
    )

    # =============================================
    # 汇总
    # =============================================
    print(f"\n{'='*70}")
    print(f"  所有模型训练完成！")
    print(f"{'='*70}")
    print(f"\n  models/ 目录下的 v6 模型：")
    for f in sorted(os.listdir(models_dir)):
        if 'v6' in f:
            fpath = os.path.join(models_dir, f)
            print(f"    {f:55s} {os.path.getsize(fpath)/1024:>8.1f} KB")

    # 返回 scaler 信息供 MCU 更新使用
    return {
        'min': float(scaler.data_min_[0]),
        'max': float(scaler.data_max_[0]),
        'range': float(scaler.data_max_[0] - scaler.data_min_[0]),
        'reference_pressure': REFERENCE_PRESSURE,
        'window_size': WINDOW_SIZE,
    }


# =============================================
# 更新 MCU 端 main.c 中的 scaler 参数
# =============================================
def update_mcu_scaler(scaler_info):
    """更新 main.c 中的 NN 滤波参数宏定义"""
    main_c_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "Core", "Src", "main.c")

    if not os.path.exists(main_c_path):
        print(f"\n[警告] 找不到 main.c: {main_c_path}")
        return

    with open(main_c_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 更新参数
    new_min = scaler_info['min']
    new_max = scaler_info['max']
    new_range = scaler_info['range']

    # 查找并替换
    import re

    # NN_REL_MIN
    content = re.sub(
        r'#define\s+NN_REL_MIN\s+[-\d.]+f',
        f'#define NN_REL_MIN       {new_min:.2f}f',
        content
    )

    # NN_REL_MAX
    content = re.sub(
        r'#define\s+NN_REL_MAX\s+[-\d.]+f',
        f'#define NN_REL_MAX       {new_max:.2f}f',
        content
    )

    # NN_REL_RANGE
    content = re.sub(
        r'#define\s+NN_REL_RANGE\s+[-\d.]+f',
        f'#define NN_REL_RANGE     {new_range:.2f}f',
        content
    )

    # 验证 NN_REF_PRESSURE
    if f'#define NN_REF_PRESSURE {REFERENCE_PRESSURE}f' not in content:
        content = re.sub(
            r'#define\s+NN_REF_PRESSURE\s+\d+\.?\d*f',
            f'#define NN_REF_PRESSURE {REFERENCE_PRESSURE}f',
            content
        )

    # 同时更新合理性检查逻辑（将 diff > 50 改为 diff > 15，且与原始输入比较）
    old_check = """        /* 合理性检查：NN 输出若偏离 KF 输出（即 last_output）超过阈值则回退到 KF */
        float diff = nn_output - nn->last_output;
        if(diff < 0.0f) diff = -diff;
        if(diff > 50.0f)
            ;  /* 保持 nn->last_output 不变（回退到 KF） */
        else
            nn->last_output = nn_output;"""

    new_check = """        /* 合理性检查：NN 输出若偏离原始输入超过阈值则回退到原始输入 */
        float diff = nn_output - z;
        if(diff < 0.0f) diff = -diff;
        if(diff > 15.0f)
            nn->last_output = z;  /* 回退到原始输入 */
        else
            nn->last_output = nn_output;"""

    if old_check in content:
        content = content.replace(old_check, new_check)
        print("  ✓ 已更新 NN 滤波合理性检查逻辑（与原始输入比较，阈值 15Pa）")
    else:
        print("  [注意] 未找到旧的合理性检查代码，跳过更新")

    with open(main_c_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n  ✓ main.c 参数已更新:")
    print(f"    NN_REL_MIN  = {new_min:.2f}")
    print(f"    NN_REL_MAX  = {new_max:.2f}")
    print(f"    NN_REL_RANGE = {new_range:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-mcu-update', action='store_true',
                        help='跳过 MCU 端 main.c 的 scaler 参数更新')
    parser.add_argument('--models', type=str, default='all',
                        help='训练的模型: all, bp, lstm, gru, lightweight, ms5611, bmp280')
    args = parser.parse_args()

    scaler_info = train_all()

    if not args.skip_mcu_update:
        print(f"\n{'='*70}")
        print(f"  更新 MCU 端参数")
        print(f"{'='*70}")
        update_mcu_scaler(scaler_info)

    print(f"\n{'='*70}")
    print(f"  全部完成！")
    print(f"{'='*70}")
