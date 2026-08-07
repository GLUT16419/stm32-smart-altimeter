#!/usr/bin/env python
"""快速批量训练 BP 标注模型，跳过交互输入"""

import os
import sys

# 直接调用 train_labeled.py 中的核心函数
sys.path.insert(0, os.path.dirname(__file__))

# 导入训练函数
from train_labeled import load_all_labeled, build_regression_model
from train_labeled_merged import load_and_merge_labeled, build_merged_model

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_root = os.path.join(os.path.dirname(__file__), "data")
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)


def train_sensor(sensor_name, sensor_label):
    """训练单传感器 BP 模型"""
    print(f"\n{'='*60}")
    print(f"   训练 {sensor_label} BP 标注模型")
    print(f"{'='*60}")
    
    df = load_all_labeled(data_root, sensor_name)
    if df is None:
        print(f"  ❌ 加载 {sensor_label} 数据失败")
        return
    
    # 单样本特征
    X = df[['pressure_pa', 'temperature_c']].values
    y = df['label_height_m'].values
    
    print(f"\n  样本数: {len(X)}, 特征维度: {X.shape[1]}")
    
    # 归一化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 分层抽样
    unique_heights = np.unique(y)
    height_bins = pd.cut(y, bins=min(20, len(unique_heights)), labels=False)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(sss.split(X_scaled, height_bins))
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    # 构建 BP 模型
    model = build_regression_model(2, 'bp')
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0),
    ]
    
    history = model.fit(X_train, y_train, epochs=300, batch_size=16,
                        validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    
    # 评估
    loss, mae_scaled = model.evaluate(X_test, y_test, verbose=0)
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_test_u = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_u = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_u, y_pred_u))
    mae = mean_absolute_error(y_test_u, y_pred_u)
    r2 = r2_score(y_test_u, y_pred_u)
    
    print(f"\n  {sensor_label} BP 结果: RMSE={rmse:.2f}m  MAE={mae:.2f}m  R²={r2:.4f}")
    
    # 导出
    model_name = f"{sensor_name}_labeled_bp"
    keras_path = os.path.join(models_dir, f"{model_name}.h5")
    tflite_path = os.path.join(models_dir, f"{model_name}.tflite")
    
    model.save(keras_path)
    print(f"  Keras: {keras_path}")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"  TFLite: {tflite_path}")
    
    scaler_info = {
        'X_mean': scaler_X.mean_.tolist(), 'X_std': scaler_X.scale_.tolist(),
        'y_mean': float(scaler_y.mean_[0]), 'y_std': float(scaler_y.scale_[0]),
        'feature_mode': 'single', 'input_dim': 2, 'model_type': 'bp',
        'mode': 'labeled', 'sensor': sensor_name,
        'feature_names': ['pressure_pa', 'temperature_c'],
        'test_rmse_m': float(rmse), 'test_mae_m': float(mae), 'test_r2': float(r2),
        'model_mode': 'labeled',
        'output_is_pressure': False  # 回归模型直接输出高度
    }
    with open(os.path.join(models_dir, f"{model_name}_scaler.json"), 'w') as f:
        json.dump(scaler_info, f, indent=2)
    
    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(f'{sensor_label} BP - Height Prediction', fontsize=13)
    axes[0, 0].plot(history.history['loss'], label='Train')
    axes[0, 0].plot(history.history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].plot(history.history['mae'], label='Train')
    axes[0, 1].plot(history.history['val_mae'], label='Val')
    axes[0, 1].set_title('MAE'); axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[1, 0].scatter(y_test_u, y_pred_u, alpha=0.6, s=20)
    m = min(y_test_u.min(), y_pred_u.min())
    M = max(y_test_u.max(), y_pred_u.max())
    axes[1, 0].plot([m, M], [m, M], 'r--', alpha=0.5)
    axes[1, 0].set_title(f'Prediction (R²={r2:.4f})')
    axes[1, 0].set_xlabel('True'); axes[1, 0].set_ylabel('Predicted'); axes[1, 0].grid(True)
    axes[1, 1].hist(y_test_u - y_pred_u, bins=30, alpha=0.7, color='steelblue')
    axes[1, 1].axvline(x=0, color='r', ls='--', alpha=0.5)
    axes[1, 1].set_title(f'Error (MAE={mae:.2f}m)'); axes[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, f"{model_name}_training.png"), dpi=150)
    plt.close()
    print(f"  Plot saved")


def train_merged():
    """训练双传感器合并 BP 模型"""
    print(f"\n{'='*60}")
    print(f"   训练 MS5611+BMP280 合并 BP 标注模型")
    print(f"{'='*60}")
    
    X, y = load_and_merge_labeled(data_root)
    if X is None:
        return
    
    print(f"\n  样本数: {len(X)}, 特征维度: {X.shape[1]}")
    
    # 归一化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 分层抽样
    unique_heights = np.unique(y)
    height_bins = pd.cut(y, bins=min(20, len(unique_heights)), labels=False)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(sss.split(X_scaled, height_bins))
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    # 构建 BP 模型
    model = build_merged_model(4, 'bp')
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0),
    ]
    
    history = model.fit(X_train, y_train, epochs=300, batch_size=16,
                        validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    
    # 评估
    loss, mae_scaled = model.evaluate(X_test, y_test, verbose=0)
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_test_u = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_u = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_u, y_pred_u))
    mae = mean_absolute_error(y_test_u, y_pred_u)
    r2 = r2_score(y_test_u, y_pred_u)
    
    print(f"\n  合并 BP 结果: RMSE={rmse:.2f}m  MAE={mae:.2f}m  R²={r2:.4f}")
    
    # 导出
    model_name = "merged_labeled_bp"
    keras_path = os.path.join(models_dir, f"{model_name}.h5")
    tflite_path = os.path.join(models_dir, f"{model_name}.tflite")
    
    model.save(keras_path)
    print(f"  Keras: {keras_path}")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"  TFLite: {tflite_path}")
    
    scaler_info = {
        'X_mean': scaler_X.mean_.tolist(), 'X_std': scaler_X.scale_.tolist(),
        'y_mean': float(scaler_y.mean_[0]), 'y_std': float(scaler_y.scale_[0]),
        'feature_mode': 'single', 'input_dim': 4, 'model_type': 'bp',
        'mode': 'merged_labeled',
        'feature_names': ['ms5611_p', 'ms5611_t', 'bmp280_p', 'bmp280_t'],
        'test_rmse_m': float(rmse), 'test_mae_m': float(mae), 'test_r2': float(r2),
        'model_mode': 'labeled',
        'output_is_pressure': False  # 回归模型直接输出高度
    }
    with open(os.path.join(models_dir, f"{model_name}_scaler.json"), 'w') as f:
        json.dump(scaler_info, f, indent=2)
    
    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle('Merged BP (MS5611+BMP280) - Height Prediction', fontsize=13)
    axes[0, 0].plot(history.history['loss'], label='Train')
    axes[0, 0].plot(history.history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].plot(history.history['mae'], label='Train')
    axes[0, 1].plot(history.history['val_mae'], label='Val')
    axes[0, 1].set_title('MAE'); axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[1, 0].scatter(y_test_u, y_pred_u, alpha=0.6, s=20)
    m = min(y_test_u.min(), y_pred_u.min())
    M = max(y_test_u.max(), y_pred_u.max())
    axes[1, 0].plot([m, M], [m, M], 'r--', alpha=0.5)
    axes[1, 0].set_title(f'Prediction (R²={r2:.4f})')
    axes[1, 0].set_xlabel('True'); axes[1, 0].set_ylabel('Predicted'); axes[1, 0].grid(True)
    axes[1, 1].hist(y_test_u - y_pred_u, bins=30, alpha=0.7, color='steelblue')
    axes[1, 1].axvline(x=0, color='r', ls='--', alpha=0.5)
    axes[1, 1].set_title(f'Error (MAE={mae:.2f}m)'); axes[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, f"{model_name}_training.png"), dpi=150)
    plt.close()
    print(f"  Plot saved")


if __name__ == "__main__":
    # 1. 训练 MS5611 BP
    train_sensor('ms5611', 'MS5611')
    
    # 2. 训练 BMP280 BP
    train_sensor('bmp280', 'BMP280')
    
    # 3. 训练合并 BP
    train_merged()
    
    print(f"\n{'='*60}")
    print(f"  全部训练完成！")
    print(f"{'='*60}")
    print(f"\n  models/ 目录下生成的文件：")
    for f in sorted(os.listdir(models_dir)):
        fpath = os.path.join(models_dir, f)
        size = os.path.getsize(fpath)
        print(f"    {f:45s} {size/1024:>8.1f} KB")
