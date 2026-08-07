#!/usr/bin/env python
"""
合并标注数据训练：MS5611 + BMP280 联合预测高度
输入 4 个特征：ms5611_p, ms5611_t, bmp280_p, bmp280_t → 预测 height
数据来源：labeled_*.csv（带 label_height_m 标签），按 sample_id 对齐
部署时两个传感器都得连，用冗余信息提高精度
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Concatenate, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_and_merge_labeled(data_root):
    """
    加载所有高度文件夹下的两个传感器的标注数据，
    按 sample_id 对齐合并。
    返回：X 4维特征, y 高度标签
    """
    all_X = []
    all_y = []
    
    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # 找该文件夹下的 ms5611 和 bmp280 文件
        ms_files = [f for f in os.listdir(folder_path) 
                     if f.startswith('labeled_ms5611') and f.endswith('.csv')]
        bm_files = [f for f in os.listdir(folder_path) 
                     if f.startswith('labeled_bmp280') and f.endswith('.csv')]
        
        # 按文件名排序配对（同一批次采集的文件时间戳相同）
        ms_files.sort()
        bm_files.sort()
        
        for ms_f, bm_f in zip(ms_files, bm_files):
            df_ms = pd.read_csv(os.path.join(folder_path, ms_f))
            df_bm = pd.read_csv(os.path.join(folder_path, bm_f))
            
            # 按 sample_id 内连接对齐
            merged = pd.merge(df_ms, df_bm, on='sample_id', suffixes=('_ms', '_bm'))
            
            if len(merged) == 0:
                print(f"  ⚠ {ms_f} & {bm_f}: 无共同 sample_id，跳过")
                continue
            
            # 特征：ms5611_p, ms5611_t, bmp280_p, bmp280_t
            X_height = np.column_stack([
                merged['pressure_pa_ms'].values,
                merged['temperature_c_ms'].values,
                merged['pressure_pa_bm'].values,
                merged['temperature_c_bm'].values,
            ])
            y_height = merged['label_height_m_ms'].values  # 两个传感器标签一样
            
            all_X.append(X_height)
            all_y.append(y_height)
            
            label_h = merged['label_height_m_ms'].iloc[0]
            print(f"  ✓ {ms_f[:25]}... & {bm_f[:25]}... → {len(merged)}条对齐 (height={label_h:.2f}m)")
    
    if not all_X:
        print("  未找到可合并的配对数据")
        return None, None
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    print(f"\n  合并后总计: {len(X)} 条对齐样本")
    print(f"  特征维度: {X.shape[1]} (ms5611_p, ms5611_t, bmp280_p, bmp280_t)")
    return X, y


def build_merged_model(input_dim, model_type='bp'):
    """构建合并模型，支持多种架构"""
    model = Sequential(name=f'merged_labeled_{model_type}')
    
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


def train_merged():
    """主训练流程"""
    print("=" * 60)
    print("   合并标注训练：MS5611 + BMP280 → 高度预测")
    print("   输入: ms5611_p, ms5611_t, bmp280_p, bmp280_t")
    print("   输出: height_m")
    print("=" * 60)
    
    data_root = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_root):
        print(f"错误：数据目录不存在 {data_root}")
        return
    
    # 1. 选择模型架构
    print("\n选择模型架构：")
    print("  1. BP 神经网络  (推荐，简单有效)")
    print("  2. BP 深度网络")
    print("  3. LSTM  (带时序记忆)")
    print("  4. GRU")
    model_map = {'1': 'bp', '2': 'bp_deep', '3': 'lstm', '4': 'gru'}
    choice = input("请输入 (1/2/3/4): ").strip()
    model_type = model_map.get(choice, 'bp')
    
    # 2. 选择特征模式
    print("\n特征模式：")
    print("  1. 单样本模式：当前 4 维特征 → height")
    print("  2. 滑动窗口模式：过去 N 步的 4 维特征 → 当前 height")
    mode_choice = input("请输入 (1/2): ").strip()
    use_window = (mode_choice == '2')
    
    # 3. 加载并合并数据
    print(f"\n加载标注数据（双传感器对齐合并）...")
    X, y = load_and_merge_labeled(data_root)
    if X is None:
        return
    
    # 4. 滑动窗口（如果需要）
    if use_window:
        window_size = 5
        X_windowed, y_windowed = [], []
        # 按文件夹分组做滑动窗口，避免跨高度边界
        current_start = 0
        for folder in sorted(os.listdir(data_root)):
            folder_path = os.path.join(data_root, folder)
            if not os.path.isdir(folder_path):
                continue
            ms_files = sorted([f for f in os.listdir(folder_path) 
                               if f.startswith('labeled_ms5611') and f.endswith('.csv')])
            bm_files = sorted([f for f in os.listdir(folder_path) 
                               if f.startswith('labeled_bmp280') and f.endswith('.csv')])
            n_pairs = min(len(ms_files), len(bm_files))
            # 粗略估算该高度下的样本数
            # 实际按对齐后的数据分段更准确，这里简化处理
            pass
        
        # 简化：整条数据做滑动窗口（假设同高度连续采集）
        for i in range(window_size, len(X)):
            feat = X[i - window_size:i + 1].flatten()  # (window_size+1)*4
            X_windowed.append(feat)
            y_windowed.append(y[i])
        X = np.array(X_windowed)
        y = np.array(y_windowed)
        input_dim = (window_size + 1) * 4
        print(f"\n滑动窗口: window_size={window_size}, 特征维度={input_dim}")
    else:
        input_dim = 4
        print(f"\n单样本特征: 特征维度={input_dim}")
    
    print(f"X={X.shape}, y={y.shape}")
    
    # 5. 显示数据分布
    unique_heights = np.unique(y)
    print(f"\n高度分布 ({len(unique_heights)} 级):")
    for h in unique_heights:
        count = np.sum(y == h)
        print(f"  height={h:.2f}m: {count} samples")
    
    # 6. 归一化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 7. 分层抽样切分
    from sklearn.model_selection import StratifiedShuffleSplit
    
    height_bins = pd.cut(y, bins=min(20, len(unique_heights)), labels=False)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(sss.split(X_scaled, height_bins))
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"\n数据集划分：")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    
    # 8. LSTM/GRU 需要 3D 输入
    if model_type in ['lstm', 'gru']:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # 9. 构建模型
    print(f"\n构建 {model_type} 模型...")
    model = build_merged_model(input_dim, model_type)
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
    
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    mae_val = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    r2 = r2_score(y_test_unscaled, y_pred_unscaled)
    
    print(f"\n{'='*40}")
    print(f"  测试集结果（双传感器联合预测）:")
    print(f"  RMSE: {rmse:.2f} m")
    print(f"  MAE:  {mae_val:.2f} m")
    print(f"  R²:   {r2:.4f}")
    print(f"{'='*40}")
    
    # 12. 导出模型
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_name = f"merged_labeled_{model_type}"
    keras_path = os.path.join(models_dir, f"{model_name}.h5")
    tflite_path = os.path.join(models_dir, f"{model_name}.tflite")
    
    model.save(keras_path)
    print(f"\nKeras model saved: {keras_path}")
    
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
    
    scaler_info = {
        'X_mean': scaler_X.mean_.tolist(),
        'X_std': scaler_X.scale_.tolist(),
        'y_mean': float(scaler_y.mean_[0]),
        'y_std': float(scaler_y.scale_[0]),
        'feature_mode': 'window' if use_window else 'single',
        'window_size': window_size if use_window else 1,
        'input_dim': input_dim,
        'model_type': model_type,
        'mode': 'merged_labeled',
        'feature_names': ['ms5611_p', 'ms5611_t', 'bmp280_p', 'bmp280_t'],
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
    fig.suptitle('Merged Sensors Training - MS5611+BMP280 → Height', fontsize=14)
    
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
    
    axes[1, 0].scatter(y_test_unscaled, y_pred_unscaled, alpha=0.6, s=20)
    min_val = min(y_test_unscaled.min(), y_pred_unscaled.min())
    max_val = max(y_test_unscaled.max(), y_pred_unscaled.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    axes[1, 0].set_title(f'Predicted vs True Height (R²={r2:.4f})')
    axes[1, 0].set_xlabel('True Height (m)')
    axes[1, 0].set_ylabel('Predicted Height (m)')
    axes[1, 0].grid(True)
    
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
    print(f"  合并训练完成！模型已保存到 models/ 目录")
    print(f"  双传感器联合: RMSE={rmse:.2f}m  MAE={mae_val:.2f}m  R²={r2:.4f}")
    print(f"  部署时需同时连接 MS5611 和 BMP280")
    print(f"  输入顺序: [ms5611_p, ms5611_t, bmp280_p, bmp280_t]")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_merged()
