#!/usr/bin/env python
"""
自监督训练：气压信号滤波去噪
目标：输入过去 N 个含噪气压值，输出滤波后的当前值
方法：自监督构造——对含噪信号做滑动平均生成"伪干净"目标
数据来源：self_supervised_*.csv（无标签）
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 抑制 TensorFlow 冗余日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 滤波去噪参数
SMOOTH_WINDOW = 9  # 滑动平均窗口大小，用于生成"伪干净"目标


def load_all_self_supervised(data_root, sensor='ms5611'):
    """加载所有高度文件夹下的自监督数据 + raw 场景数据（静止/平移运动/升降运动）"""
    all_pressure = []
    all_temp = []
    meta = []  # 记录每条数据来自哪个文件夹
    
    # 1. 加载原有的高度文件夹数据 (self_supervised_*.csv)
    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        
        for f in os.listdir(folder_path):
            if f.startswith(f'self_supervised_{sensor}') and f.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, f))
                all_pressure.extend(df['pressure_pa'].values.tolist())
                all_temp.extend(df['temperature_c'].values.tolist())
                meta.extend([folder] * len(df))
                print(f"  Loaded {folder}/{f} ({len(df)} samples)")
    
    # 2. 加载 raw 场景数据 (data/raw/静止/  data/raw/平移运动/  data/raw/升降运动/)
    raw_dir = os.path.join(data_root, 'raw')
    if os.path.isdir(raw_dir):
        for scene_folder in sorted(os.listdir(raw_dir)):
            scene_path = os.path.join(raw_dir, scene_folder)
            if not os.path.isdir(scene_path):
                continue
            for f in os.listdir(scene_path):
                if f.startswith(f'{sensor}_') and f.endswith('.csv'):
                    df = pd.read_csv(os.path.join(scene_path, f))
                    all_pressure.extend(df['pressure_pa'].values.tolist())
                    all_temp.extend(df['temperature_c'].values.tolist())
                    meta.extend([f'raw/{scene_folder}'] * len(df))
                    print(f"  Loaded raw/{scene_folder}/{f} ({len(df)} samples)")
    
    print(f"\nTotal pressure samples: {len(all_pressure)}")
    return np.array(all_pressure), np.array(all_temp), meta


def create_sequences(data, window_size=10, step=1):
    """
    创建自监督滤波序列：
    - X: 过去 window_size 个含噪值（输入）
    - y: 当前值的滑动平均结果（滤波目标，伪干净标签）
    """
    # 先对原始数据做滑动平均得到"伪干净"信号
    # mode='reflect' 避免边界效应
    smooth = uniform_filter1d(data, size=SMOOTH_WINDOW, mode='reflect')
    
    X, y = [], []
    for i in range(0, len(data) - window_size, step):
        X.append(data[i:i + window_size])       # 含噪窗口
        center_idx = i + window_size - 1        # 窗口最后一个是"当前值"
        y.append(smooth[center_idx])            # 目标：当前值的滑动平均结果
    
    return np.array(X), np.array(y)


def build_model(input_dim, model_type='lstm'):
    """构建时序预测模型"""
    model = Sequential(name=f'self_supervised_{model_type}')
    
    if model_type == 'lstm':
        model.add(LSTM(32, return_sequences=True, input_shape=(input_dim, 1)))
        model.add(LSTM(16, return_sequences=False))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1))
    
    elif model_type == 'gru':
        model.add(GRU(32, return_sequences=True, input_shape=(input_dim, 1)))
        model.add(GRU(16, return_sequences=False))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1))
    
    elif model_type == 'cnn':
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_dim, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
    
    elif model_type == 'bp':
        model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_self_supervised():
    """主训练流程"""
    print("=" * 60)
    print("   自监督训练：气压信号滤波去噪")
    print("   输入：过去10个含噪值 → 输出：滤波后当前值")
    print("=" * 60)
    
    data_root = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_root):
        print(f"错误：数据目录不存在 {data_root}")
        return
    
    # 1. 选择传感器
    print("\n选择传感器：")
    print("  1. MS5611")
    print("  2. BMP280")
    choice = input("请输入 (1/2): ").strip()
    sensor = 'ms5611' if choice == '1' else 'bmp280'
    sensor_label = 'MS5611' if choice == '1' else 'BMP280'
    
    # 2. 选择模型架构
    print("\n选择模型架构：")
    print("  1. LSTM  (推荐，适合时序数据)")
    print("  2. GRU")
    print("  3. CNN")
    print("  4. BP 神经网络")
    model_map = {'1': 'lstm', '2': 'gru', '3': 'cnn', '4': 'bp'}
    choice = input("请输入 (1/2/3/4): ").strip()
    model_type = model_map.get(choice, 'lstm')
    
    # 3. 加载数据
    print(f"\n加载 {sensor_label} 自监督数据...")
    pressure, temp, meta = load_all_self_supervised(data_root, sensor)
    
    # 4. 预处理：使用相对气压（减去参考气压），然后 MinMax 归一化
    # 与 MCU 端 NN_Filter_Update 中预处理逻辑保持一致
    reference_pressure = 101325.0  # 标准海平面气压（1013.25 hPa）
    pressure_rel = pressure - reference_pressure
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    pressure_scaled = scaler.fit_transform(pressure_rel.reshape(-1, 1)).flatten()
    
    # 5. 创建滤波序列（输入含噪窗口，输出滑动平均后的当前值）
    window_size = 10
    X, y = create_sequences(pressure_scaled, window_size=window_size, step=1)
    print(f"创建序列: X={X.shape}, y={y.shape}")
    
    # 6. 切分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 7. 调整维度（CNN/LSTM/GRU 需要 3D 输入）
    if model_type in ['lstm', 'gru', 'cnn']:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # 8. 构建模型
    print(f"\n构建 {model_type} 模型...")
    model = build_model(window_size, model_type)
    model.summary()
    
    # 9. 训练
    print("\n开始训练...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 10. 评估
    print("\n评估模型...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    
    # 反归一化
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    mae_val = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    
    print(f"\n{'='*40}")
    print(f"  测试集结果（反归一化后）:")
    print(f"  RMSE: {rmse:.2f} Pa")
    print(f"  MAE:  {mae_val:.2f} Pa")
    print(f"  Loss: {loss:.6f}")
    print(f"{'='*40}")
    
    # 11. 导出模型
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_name = f"{sensor}_self_supervised_filter_{model_type}"
    keras_path = os.path.join(models_dir, f"{model_name}.h5")
    tflite_path = os.path.join(models_dir, f"{model_name}.tflite")
    
    model.save(keras_path)
    print(f"\nKeras model saved: {keras_path}")
    
    # TFLite 导出
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    # 若 LSTM/GRU 转换失败，可尝试启用：
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,
    #     tf.lite.OpsSet.SELECT_TF_OPS
    # ]
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved: {tflite_path}")
    
    # Scaler 信息
    scaler_info = {
        'min': float(scaler.data_min_[0]),
        'max': float(scaler.data_max_[0]),
        'range': float(scaler.data_max_[0] - scaler.data_min_[0]),
        'reference_pressure': reference_pressure,
        'window_size': window_size,
        'model_type': model_type,
        'sensor': sensor_label,
        'mode': 'self_supervised_filter',
        'input_is_relative': True,
        'model_mode': 'self_supervised',
        'smooth_window': SMOOTH_WINDOW
    }
    scaler_path = os.path.join(models_dir, f"{model_name}_scaler.json")
    with open(scaler_path, 'w') as f:
        json.dump(scaler_info, f, indent=2)
    print(f"Scaler info saved: {scaler_path}")
    
    # 12. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'Self-Supervised Filter - {sensor_label} ({model_type})', fontsize=14)
    
    # 训练曲线
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history.history['mae'], label='Train MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Val MAE')
    axes[0, 1].set_title('MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 滤波结果（取前200点，显示原始、滤波目标、模型输出三条曲线）
    n_show = min(200, len(y_test_unscaled))
    axes[1, 0].plot(y_test_unscaled[:n_show], label='Filter Target (Smoothed)', alpha=0.7)
    axes[1, 0].plot(y_pred_unscaled[:n_show], label='Model Output', alpha=0.7, linestyle='--')
    # 也画出原始的含噪信号作为对比
    # 需要从 X_test 还原含噪信号
    axes[1, 0].set_title('Filtering Result (First 200 samples)')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Pressure (Pa)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 误差分布
    errors = y_test_unscaled - y_pred_unscaled
    axes[1, 1].hist(errors, bins=50, alpha=0.7, color='steelblue')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title(f'Error Distribution (MAE={mae_val:.2f} Pa)')
    axes[1, 1].set_xlabel('Error (Pa)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(models_dir, f"{model_name}_training.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"  自监督滤波训练完成！模型已保存到 models/ 目录")
    print(f"  {sensor_label} {model_type}: RMSE={rmse:.2f}Pa  MAE={mae_val:.2f}Pa")
    print(f"  (平滑窗口={SMOOTH_WINDOW}，输入=含噪窗口，输出=滤波后当前值)")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_self_supervised()
