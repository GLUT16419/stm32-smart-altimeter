#!/usr/bin/env python
"""
一键训练所有 NN 滤波模型架构（含新推荐的 4 种架构）
训练完成后自动更新 MCU 端 main.c 中的 scaler 参数。

训练模型列表（v6 已有）：
  1. 双通道 BP (baseline)        - dual_sensor_self_supervised_filter_bp_v6
  2. 双通道 BP 加强版             - dual_sensor_self_supervised_filter_bp_large_v6
  3. 双通道轻量 BP                - lightweight_bp_11x32x16x1_v6
  4. 双通道 LSTM                  - dual_sensor_self_supervised_filter_lstm_v6
  5. 双通道 GRU                   - dual_sensor_self_supervised_filter_gru_v6
  6. 单传感器 MS5611 BP           - ms5611_self_supervised_filter_bp_v6
  7. 单传感器 BMP280 BP           - bmp280_self_supervised_filter_bp_v6

新架构（v7）：
  8. 双通道 CNN                   - dual_sensor_self_supervised_filter_cnn_v7
  9. 双通道 BP 残差               - dual_sensor_self_supervised_filter_bp_residual_v7
  10. 双通道 BP 更宽更深          - dual_sensor_self_supervised_filter_bp_wide_v7
  11. 双通道 BP DenseNet 跳跃连接 - dual_sensor_self_supervised_filter_bp_dense_v7
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Concatenate, Add
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
# 数据加载（复用 train_all_with_new_data.py）
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


# =============================================
# 序列创建函数
# =============================================
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
    """创建 LSTM/GRU/CNN 输入序列 (samples, timesteps, features=[pressure, sensor_id])"""
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
# 模型构建（已有架构 v6）
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
# ★ 新架构：CNN
# =============================================
def build_cnn_model(window_size, n_features):
    """
    CNN 一维卷积网络
    架构: Conv1D(16@k3) → MaxPool(2) → Conv1D(8@k3) → Flatten → Dense(16) → Dense(1)
    参数: ~2,000 参数
    """
    model = Sequential(name='cnn_dual')
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu',
                     input_shape=(window_size, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=8, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# =============================================
# ★ 新架构：残差 BP (Residual)
# =============================================
def build_residual_bp(input_dim):
    """
    残差 BP: 学习噪声修正量 Δ
    架构: input → Dense(64) → Dense(32) → Dense(16) → Dense(1) → output = input_pressure + Δ
    """
    inputs = Input(shape=(input_dim,), name='input')
    # 提取最后一个气压值（当前值）用于残差连接
    # input 结构: [p_{t-9}, p_{t-8}, ..., p_t, sensor_id]
    # 取倒数第二个（索引 -2）是当前气压 p_t

    x = Dense(64, activation='relu', name='dense_64')(inputs)
    x = Dense(32, activation='relu', name='dense_32')(x)
    x = Dense(16, activation='relu', name='dense_16')(x)
    delta = Dense(1, name='delta')(x)  # 输出噪声修正量 Δ

    # 残差连接：output = p_t + Δ
    # 用 Lambda 层取输入中的当前气压值 (index -2)
    from tensorflow.keras.layers import Lambda
    p_current = Lambda(lambda t: t[:, -2:-1], name='extract_p_current')(inputs)
    output = Add(name='residual_add')([p_current, delta])

    model = Model(inputs=inputs, outputs=output, name='bp_residual')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# =============================================
# ★ 新架构：更宽更深 BP (Wide & Deep)
# =============================================
def build_wide_bp(input_dim):
    """
    更宽更深 BP: 80→40→20→1
    参数: ~4,500
    """
    model = Sequential(name='bp_wide')
    model.add(Dense(80, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# =============================================
# ★ 新架构：DenseNet 风格跳跃连接 BP
# =============================================
def build_dense_bp(input_dim):
    """
    DenseNet 风格跳跃连接 BP
    架构: input → Dense(32) → concat[input, h1] → Dense(16) → concat[input, h1, h2] → Dense(1)
    """
    inputs = Input(shape=(input_dim,), name='input')
    h1 = Dense(32, activation='relu', name='dense_32')(inputs)

    # 跳跃连接：concat(input, h1)
    concat1 = Concatenate(name='concat_1')([inputs, h1])
    h2 = Dense(16, activation='relu', name='dense_16')(concat1)

    # 跳跃连接：concat(input, h1, h2)
    concat2 = Concatenate(name='concat_2')([inputs, h1, h2])
    output = Dense(1, name='output')(concat2)

    model = Model(inputs=inputs, outputs=output, name='bp_dense')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# =============================================
# 训练函数（统一）
# =============================================
def prepare_data(data, scaler, sensor_id):
    """准备单个传感器的 BP 训练数据"""
    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()
    X, y = create_sequences_bp(data_scaled, sensor_id)
    return X, y


def prepare_data_lstm(data, scaler, sensor_id):
    """准备 LSTM/GRU/CNN 训练数据（3D 格式）"""
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

    # 尝试纯 TFLITE_BUILTINS 导出
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    try:
        tflite_model = converter.convert()
        used_select_ops = False
    except Exception as e:
        print(f"  [info] 纯 TFLITE_BUILTINS 导出失败 ({e})，回退到 SELECT_TF_OPS")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        used_select_ops = True

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    tflite_kb = len(tflite_model) / 1024
    print(f"  Keras: {keras_path}")
    print(f"  TFLite: {tflite_path} ({tflite_kb:.1f} KB, SELECT_TF_OPS={used_select_ops})")

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
        'tflite_size_kb': round(tflite_kb, 1),
        'tflite_select_ops': used_select_ops,
        'total_params': model.count_params(),
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

    return results, scaler_info


# =============================================
# 训练所有模型
# =============================================
def train_all():
    print("=" * 70)
    print("  训练所有 NN 滤波模型架构（含 4 种新架构）")
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

    # 收集所有结果
    all_results = {}

    # =============================================
    # BP 数据准备（共享）
    # =============================================
    X_m, y_m = create_sequences_bp(ms5611_scaled, sensor_id=0.0)
    X_b, y_b = create_sequences_bp(bmp280_scaled, sensor_id=1.0)
    X = np.concatenate([X_m, X_b])
    y = np.concatenate([y_m, y_b])
    sensor_labels = np.concatenate([np.zeros(len(X_m)), np.ones(len(X_b))])

    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, sensor_labels, test_size=0.3, random_state=42, stratify=sensor_labels)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=s_temp)

    print(f"\nBP 数据集: 训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}")

    # =============================================
    # 3D 数据准备（LSTM/GRU/CNN 共享）
    # =============================================
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

    print(f"3D 数据集: 训练={len(X_train_l)}, 验证={len(X_val_l)}, 测试={len(X_test_l)}")

    # =============================================
    # 模型 1: 双通道 BP (baseline v6) - 已有
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 1/11: 双通道 BP (baseline) - v6")
    print(f"{'#'*70}")
    r1, s1 = train_and_save(
        lambda: build_baseline_bp(INPUT_DIM),
        "dual_sensor_self_supervised_filter_bp_v6",
        X_train, y_train, X_val, y_val, X_test, y_test, s_test, scaler
    )
    all_results['bp_baseline_v6'] = r1

    # =============================================
    # 模型 2: 双通道 BP 加强版 v6 - 已有
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 2/11: 双通道 BP 加强版 - v6")
    print(f"{'#'*70}")
    r2, s2 = train_and_save(
        lambda: build_large_bp(INPUT_DIM),
        "dual_sensor_self_supervised_filter_bp_large_v6",
        X_train, y_train, X_val, y_val, X_test, y_test, s_test, scaler
    )
    all_results['bp_large_v6'] = r2

    # =============================================
    # 模型 3: 轻量 BP v6 - 已有
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 3/11: 轻量 BP - v6")
    print(f"{'#'*70}")
    r3, s3 = train_and_save(
        lambda: build_lightweight_bp(INPUT_DIM),
        "lightweight_bp_11x32x16x1_v6",
        X_train, y_train, X_val, y_val, X_test, y_test, s_test, scaler
    )
    all_results['bp_lightweight_v6'] = r3

    # =============================================
    # 模型 4: 双通道 LSTM v6 - 已有
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 4/11: 双通道 LSTM - v6")
    print(f"{'#'*70}")
    r4, s4 = train_and_save(
        lambda: build_lstm_model(WINDOW_SIZE, N_FEATURES),
        "dual_sensor_self_supervised_filter_lstm_v6",
        X_train_l, y_train_l, X_val_l, y_val_l, X_test_l, y_test_l, s_test_l, scaler
    )
    all_results['lstm_v6'] = r4

    # =============================================
    # 模型 5: 双通道 GRU v6 - 已有
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 5/11: 双通道 GRU - v6")
    print(f"{'#'*70}")
    r5, s5 = train_and_save(
        lambda: build_gru_model(WINDOW_SIZE, N_FEATURES),
        "dual_sensor_self_supervised_filter_gru_v6",
        X_train_l, y_train_l, X_val_l, y_val_l, X_test_l, y_test_l, s_test_l, scaler
    )
    all_results['gru_v6'] = r5

    # =============================================
    # 模型 6: 单传感器 MS5611 BP v6 - 已有
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 6/11: 单传感器 MS5611 BP - v6")
    print(f"{'#'*70}")

    X_m_single, y_m_single = create_sequences_bp(ms5611_scaled, sensor_id=0.0)
    X_train_m, X_temp_m, y_train_m, y_temp_m = train_test_split(
        X_m_single, y_m_single, test_size=0.3, random_state=42)
    X_val_m, X_test_m, y_val_m, y_test_m = train_test_split(
        X_temp_m, y_temp_m, test_size=0.5, random_state=42)

    r6, s6 = train_and_save(
        lambda: build_baseline_bp(INPUT_DIM),
        "ms5611_self_supervised_filter_bp_v6",
        X_train_m, y_train_m, X_val_m, y_val_m, X_test_m, y_test_m, None, scaler,
        sensor_names='MS5611'
    )
    all_results['ms5611_bp_v6'] = r6

    # =============================================
    # 模型 7: 单传感器 BMP280 BP v6 - 已有
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # 模型 7/11: 单传感器 BMP280 BP - v6")
    print(f"{'#'*70}")

    X_b_single, y_b_single = create_sequences_bp(bmp280_scaled, sensor_id=1.0)
    X_train_b, X_temp_b, y_train_b, y_temp_b = train_test_split(
        X_b_single, y_b_single, test_size=0.3, random_state=42)
    X_val_b, X_test_b, y_val_b, y_test_b = train_test_split(
        X_temp_b, y_temp_b, test_size=0.5, random_state=42)

    r7, s7 = train_and_save(
        lambda: build_baseline_bp(INPUT_DIM),
        "bmp280_self_supervised_filter_bp_v6",
        X_train_b, y_train_b, X_val_b, y_val_b, X_test_b, y_test_b, None, scaler,
        sensor_names='BMP280'
    )
    all_results['bmp280_bp_v6'] = r7

    # =============================================
    # ★ 模型 8: 双通道 CNN - v7 (新)
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # ★ 模型 8/11: 双通道 CNN - v7 (新)")
    print(f"{'#'*70}")
    r8, s8 = train_and_save(
        lambda: build_cnn_model(WINDOW_SIZE, N_FEATURES),
        "dual_sensor_self_supervised_filter_cnn_v7",
        X_train_l, y_train_l, X_val_l, y_val_l, X_test_l, y_test_l, s_test_l, scaler
    )
    all_results['cnn_v7'] = r8

    # =============================================
    # ★ 模型 9: 双通道 BP 残差 - v7 (新)
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # ★ 模型 9/11: 双通道 BP 残差 - v7 (新)")
    print(f"{'#'*70}")
    r9, s9 = train_and_save(
        lambda: build_residual_bp(INPUT_DIM),
        "dual_sensor_self_supervised_filter_bp_residual_v7",
        X_train, y_train, X_val, y_val, X_test, y_test, s_test, scaler
    )
    all_results['bp_residual_v7'] = r9

    # =============================================
    # ★ 模型 10: 双通道 BP 更宽更深 - v7 (新)
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # ★ 模型 10/11: 双通道 BP 更宽更深 - v7 (新)")
    print(f"{'#'*70}")
    r10, s10 = train_and_save(
        lambda: build_wide_bp(INPUT_DIM),
        "dual_sensor_self_supervised_filter_bp_wide_v7",
        X_train, y_train, X_val, y_val, X_test, y_test, s_test, scaler
    )
    all_results['bp_wide_v7'] = r10

    # =============================================
    # ★ 模型 11: 双通道 BP DenseNet - v7 (新)
    # =============================================
    print(f"\n{'#'*70}")
    print(f"  # ★ 模型 11/11: 双通道 BP DenseNet 跳跃连接 - v7 (新)")
    print(f"{'#'*70}")
    r11, s11 = train_and_save(
        lambda: build_dense_bp(INPUT_DIM),
        "dual_sensor_self_supervised_filter_bp_dense_v7",
        X_train, y_train, X_val, y_val, X_test, y_test, s_test, scaler
    )
    all_results['bp_dense_v7'] = r11

    # =============================================
    # 汇总对比
    # =============================================
    print(f"\n{'='*90}")
    print(f"  ★ 所有模型训练完成！结果汇总")
    print(f"{'='*90}")
    print(f"\n{'模型名称':<55s} {'总体RMSE':>8s} {'总体MAE':>8s} {'M-RMSE':>8s} {'M-MAE':>8s} "
          f"{'B-RMSE':>8s} {'B-MAE':>8s} {'大小KB':>7s} {'参数':>6s}")
    print(f"{'-'*90}")

    # 获取 TFLite 文件大小和参数
    def get_model_info(model_name_key, model_display_name):
        try:
            tflite_fname = ""
            for f in os.listdir(models_dir):
                if f.endswith('.tflite') and model_name_key.replace('_', '') in f.replace('_', '').replace('-', ''):
                    tflite_fname = f
                    break
            if not tflite_fname:
                tflite_fname = f"{model_name_key.replace('_v6', '_v6').replace('_v7', '_v7')}.tflite"
                for f in os.listdir(models_dir):
                    if f == tflite_fname:
                        break
                else:
                    return f"{model_display_name:<55s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>7s} {'N/A':>6s}"

            size_kb = os.path.getsize(os.path.join(models_dir, tflite_fname)) / 1024
        except:
            size_kb = 0

        r = all_results.get(model_name_key, {})
        ov = r.get('overall', {})
        m = r.get('MS5611', {})
        b = r.get('BMP280', {})

        return (f"{model_display_name:<55s} "
                f"{ov.get('rmse', 0):>8.2f} {ov.get('mae', 0):>8.2f} "
                f"{m.get('rmse', 0):>8.2f} {m.get('mae', 0):>8.2f} "
                f"{b.get('rmse', 0):>8.2f} {b.get('mae', 0):>8.2f} "
                f"{size_kb:>7.1f} {'--':>6s}")

    models_to_report = [
        ('bp_baseline_v6', 'BP baseline (64→32→16→1) v6'),
        ('bp_large_v6', 'BP large (128→64→32→1) v6'),
        ('bp_lightweight_v6', 'BP lightweight (32→16→1) v6'),
        ('lstm_v6', 'LSTM (32→16→1) v6'),
        ('gru_v6', 'GRU (32→16→1) v6'),
        ('ms5611_bp_v6', 'MS5611 single BP v6'),
        ('bmp280_bp_v6', 'BMP280 single BP v6'),
        ('cnn_v7', '★ CNN (Conv1D×2→Dense) v7'),
        ('bp_residual_v7', '★ BP Residual (64→32→16 + skip) v7'),
        ('bp_wide_v7', '★ BP Wide (80→40→20→1) v7'),
        ('bp_dense_v7', '★ BP DenseNet (skip-connect) v7'),
    ]

    for key, display in models_to_report:
        print(get_model_info(key, display))

    # 使用 scaler_info 中的 tflite_size_kb 和 total_params
    print(f"\n{'='*90}")
    print(f"  ★ 各模型参数与文件大小详情")
    print(f"{'='*90}")
    print(f"\n{'模型名称':<55s} {'参数量':>8s} {'TFLite(KB)':>10s} {'SELECT_OPS':>10s}")
    print(f"{'-'*90}")

    for key, display in models_to_report:
        scaler_key = key.replace('_v6', '_v6').replace('_v7', '_v7')
        # 尝试读取 scaler json
        scaler_fname = None
        for f in os.listdir(models_dir):
            if f.endswith('_scaler.json'):
                # 尝试匹配
                base = key.replace('bp_baseline_v6', 'dual_sensor_self_supervised_filter_bp_v6')
                base = base.replace('bp_large_v6', 'dual_sensor_self_supervised_filter_bp_large_v6')
                base = base.replace('bp_lightweight_v6', 'lightweight_bp_11x32x16x1_v6')
                base = base.replace('lstm_v6', 'dual_sensor_self_supervised_filter_lstm_v6')
                base = base.replace('gru_v6', 'dual_sensor_self_supervised_filter_gru_v6')
                base = base.replace('ms5611_bp_v6', 'ms5611_self_supervised_filter_bp_v6')
                base = base.replace('bmp280_bp_v6', 'bmp280_self_supervised_filter_bp_v6')
                base = base.replace('cnn_v7', 'dual_sensor_self_supervised_filter_cnn_v7')
                base = base.replace('bp_residual_v7', 'dual_sensor_self_supervised_filter_bp_residual_v7')
                base = base.replace('bp_wide_v7', 'dual_sensor_self_supervised_filter_bp_wide_v7')
                base = base.replace('bp_dense_v7', 'dual_sensor_self_supervised_filter_bp_dense_v7')
                if f == f"{base}_scaler.json":
                    scaler_fname = f
                    break

        if scaler_fname:
            try:
                with open(os.path.join(models_dir, scaler_fname)) as f:
                    si = json.load(f)
                params = si.get('total_params', 'N/A')
                tflite_kb = si.get('tflite_size_kb', 'N/A')
                select_ops = si.get('tflite_select_ops', 'N/A')
                print(f"{display:<55s} {str(params):>8s} {str(tflite_kb):>10s} {str(select_ops):>10s}")
            except:
                pass

    # 返回 scaler 信息供 MCU 更新使用
    return {
        'min': float(scaler.data_min_[0]),
        'max': float(scaler.data_max_[0]),
        'range': float(scaler.data_max_[0] - scaler.data_min_[0]),
        'reference_pressure': REFERENCE_PRESSURE,
        'window_size': WINDOW_SIZE,
    }


# =============================================
# 更新 MCU 端 main.c
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

    new_min = scaler_info['min']
    new_max = scaler_info['max']
    new_range = scaler_info['range']

    import re

    content = re.sub(
        r'#define\s+NN_REL_MIN\s+[-\d.]+f',
        f'#define NN_REL_MIN       {new_min:.2f}f',
        content
    )
    content = re.sub(
        r'#define\s+NN_REL_MAX\s+[-\d.]+f',
        f'#define NN_REL_MAX       {new_max:.2f}f',
        content
    )
    content = re.sub(
        r'#define\s+NN_REL_RANGE\s+[-\d.]+f',
        f'#define NN_REL_RANGE     {new_range:.2f}f',
        content
    )

    if f'#define NN_REF_PRESSURE {REFERENCE_PRESSURE}f' not in content:
        content = re.sub(
            r'#define\s+NN_REF_PRESSURE\s+\d+\.?\d*f',
            f'#define NN_REF_PRESSURE {REFERENCE_PRESSURE}f',
            content
        )

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
