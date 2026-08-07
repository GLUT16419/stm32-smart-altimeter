#!/usr/bin/env python
"""
标注数据训练：使用人工标注的真实高度作为监督信号
目标：输入 pressure + temperature → 预测 height
      或者用过去 N 个窗口的压力序列预测当前高度
数据来源：labeled_*.csv（带 label_height_m 标签）
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_all_labeled(data_root, sensor='ms5611'):
    """加载所有高度文件夹下的标注数据"""
    all_data = []
    
    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        
        for f in os.listdir(folder_path):
            if f.startswith(f'labeled_{sensor}') and f.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, f))
                all_data.append(df)
                print(f"  Loaded {f} ({len(df)} samples, label_height={df['label_height_m'].iloc[0]:.2f}m)")
    
    if not all_data:
        print(f"  未找到 {sensor} 的标注数据")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n  合并后总计: {len(combined)} 条样本")
    return combined


def build_regression_model(input_dim, model_type='bp'):
    """
    监督回归模型：pressure, temperature → height
    支持多种架构
    """
    model = Sequential(name=f'labeled_{model_type}')
    
    if model_type == 'bp':
        model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
    
    elif model_type == 'bp_deep':
        model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
    
    elif model_type == 'lstm':
        model.add(LSTM(32, return_sequences=True, input_shape=(input_dim, 1)))
        model.add(LSTM(16, return_sequences=False))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
    
    elif model_type == 'gru':
        model.add(GRU(32, return_sequences=True, input_shape=(input_dim, 1)))
        model.add(GRU(16, return_sequences=False))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_sliding_window_features(df, window_size=5):
    """用滑动窗口创建特征：过去 window_size 个 pressure + temp"""
    features = []
    targets = []
    
    for i in range(window_size, len(df)):
        # 用过去 window_size 步的 pressure 和 temp
        feat = []
        for j in range(window_size):
            feat.append(df.iloc[i - window_size + j]['pressure_pa'])
            feat.append(df.iloc[i - window_size + j]['temperature_c'])
        features.append(feat)
        targets.append(df.iloc[i]['label_height_m'])
    
    return np.array(features), np.array(targets)


def create_single_sample_features(df):
    """单样本特征：直接用当前 pressure + temperature 预测 height"""
    X = df[['pressure_pa', 'temperature_c']].values
    y = df['label_height_m'].values
    return X, y


def train_labeled():
    """主训练流程"""
    print("=" * 60)
    print("   标注数据训练：气压/温度 → 高度 监督回归")
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
    print("  1. BP 神经网络  (推荐，简单有效)")
    print("  2. BP 深度网络")
    print("  3. LSTM  (带时序记忆)")
    print("  4. GRU")
    model_map = {'1': 'bp', '2': 'bp_deep', '3': 'lstm', '4': 'gru'}
    choice = input("请输入 (1/2/3/4): ").strip()
    model_type = model_map.get(choice, 'bp')
    
    # 3. 选择特征模式
    print("\n特征模式：")
    print("  1. 单样本模式：当前 pressure + temp → height")
    print("  2. 滑动窗口模式：过去 N 步的 pressure + temp → 当前 height")
    mode_choice = input("请输入 (1/2): ").strip()
    use_window = (mode_choice == '2')
    
    # 4. 加载数据
    print(f"\n加载 {sensor_label} 标注数据...")
    df = load_all_labeled(data_root, sensor)
    if df is None:
        return
    
    print(f"\n数据分布：")
    height_groups = df.groupby('label_height_m').size()
    for h, count in height_groups.items():
        print(f"  height={h:.2f}m: {count} samples")
    
    # 5. 创建特征和标签
    if use_window:
        window_size = 5
        X, y = create_sliding_window_features(df, window_size)
        input_dim = window_size * 2  # pressure + temp 各 window_size 个
        print(f"\n滑动窗口特征: window_size={window_size}, 特征维度={input_dim}")
    else:
        X, y = create_single_sample_features(df)
        input_dim = 2  # pressure, temperature
        print(f"\n单样本特征: 特征维度={input_dim}")
    
    print(f"X={X.shape}, y={y.shape}")
    
    # 6. 归一化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 7. 切分数据集（按高度分层抽样，保证各高度在训练/测试中都有）
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # 用 label_height_m 作为分层依据
    height_bins = pd.cut(df['label_height_m'].values if not use_window else 
                          df['label_height_m'].values[window_size:], bins=20, labels=False)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(sss.split(X_scaled, height_bins))
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
    
    # 从训练集中再分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"\n数据集划分：")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    
    # 8. 调整维度（LSTM/GRU 需要 3D）
    if model_type in ['lstm', 'gru']:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # 9. 构建模型
    print(f"\n构建 {model_type} 模型...")
    model = build_regression_model(input_dim, model_type)
    model.summary()
    
    # 10. 训练
    print("\n开始训练...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 11. 评估
    print("\n评估模型...")
    loss, mae_scaled = model.evaluate(X_test, y_test, verbose=0)
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # 反归一化
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    mae_val = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    r2 = r2_score(y_test_unscaled, y_pred_unscaled)
    
    print(f"\n{'='*40}")
    print(f"  测试集结果：")
    print(f"  RMSE: {rmse:.2f} m")
    print(f"  MAE:  {mae_val:.2f} m")
    print(f"  R²:   {r2:.4f}")
    print(f"{'='*40}")
    
    # 12. 导出模型
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_name = f"{sensor}_labeled_{model_type}"
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
    
    # 归一化参数
    scaler_info = {
        'X_mean': scaler_X.mean_.tolist(),
        'X_std': scaler_X.scale_.tolist(),
        'y_mean': float(scaler_y.mean_[0]),
        'y_std': float(scaler_y.scale_[0]),
        'feature_mode': 'window' if use_window else 'single',
        'window_size': window_size if use_window else 1,
        'input_dim': input_dim,
        'model_type': model_type,
        'sensor': sensor_label,
        'mode': 'labeled',
        'test_rmse_m': float(rmse),
        'test_mae_m': float(mae_val),
        'test_r2': float(r2)
    }
    scaler_path = os.path.join(models_dir, f"{model_name}_scaler.json")
    with open(scaler_path, 'w') as f:
        json.dump(scaler_info, f, indent=2)
    print(f"Scaler info saved: {scaler_path}")
    
    # 13. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'Labeled Training - {sensor_label} ({model_type}) - Height Prediction', fontsize=14)
    
    # 训练曲线
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss (MSE)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history.history['mae'], label='Train MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Val MAE')
    axes[0, 1].set_title('MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 预测 vs 真实
    axes[1, 0].scatter(y_test_unscaled, y_pred_unscaled, alpha=0.6, s=20)
    min_val = min(y_test_unscaled.min(), y_pred_unscaled.min())
    max_val = max(y_test_unscaled.max(), y_pred_unscaled.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    axes[1, 0].set_title(f'Predicted vs True Height (R²={r2:.4f})')
    axes[1, 0].set_xlabel('True Height (m)')
    axes[1, 0].set_ylabel('Predicted Height (m)')
    axes[1, 0].grid(True)
    
    # 误差分布
    errors = y_test_unscaled - y_pred_unscaled
    axes[1, 1].hist(errors, bins=30, alpha=0.7, color='steelblue')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title(f'Error Distribution (MAE={mae_val:.2f}m, RMSE={rmse:.2f}m)')
    axes[1, 1].set_xlabel('Error (m)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(models_dir, f"{model_name}_training.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"  标注数据训练完成！模型已保存到 models/ 目录")
    print(f"  {sensor_label} {model_type}: RMSE={rmse:.2f}m  MAE={mae_val:.2f}m  R²={r2:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_labeled()
