#!/usr/bin/env python
"""
标注数据滤波训练：使用人工标注的真实高度训练滤波模型
目标：模型输出滤波后气压，高度统一通过气压公式计算

修改说明：
  - 滤波模式: 输入过去 N 个含噪气压值 → 输出滤波后当前气压
    (不再直接输出高度，用 label_height_m 反推"干净气压"作为训练目标)
  - 回归模式: 保留但同样改为输出气压

数据来源：labeled_*.csv（带 label_height_m 标签）
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免 GUI 阻塞
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 目录
SCRIPT_DIR = os.path.dirname(__file__)
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

WINDOW_SIZE = 10       # 滑动窗口大小
SMOOTH_WINDOW = 5      # 平滑窗口（用于参考曲线）

# 标准大气模型常数 (与 MCU 端 altitude_convert.c 一致)
R = 287.05      # 气体常数 J/(kg·K)
G = 9.80665     # 重力加速度 m/s²


def height_to_pressure(height_m, reference_pressure_pa, temperature_c):
    """
    高度反推气压（与 MCU 端 PressureToAltitudeWithTemp 的逆运算）
    P = P_ref * exp(-g * h / (R * T))
    """
    temperature_k = temperature_c + 273.15
    return reference_pressure_pa * np.exp(-G * height_m / (R * temperature_k))


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


def create_filter_sequences(df, window_size=WINDOW_SIZE):
    """
    创建滤波序列（滤波模式）:
    输入: 过去 window_size 个含噪相对气压值
    输出: 当前"干净"相对气压值（由 label_height_m 通过气压公式反推）

    注意: 每个序列必须在同一个高度文件夹内，避免跨高度边界
    """
    X_list, y_list = [], []
    seq_info = []

    data_root = os.path.join(os.path.dirname(__file__), "data")
    ref_p = 101325.0  # 标准海平面气压

    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue

        for f in os.listdir(folder_path):
            if not (f.startswith('labeled_') and f.endswith('.csv')):
                continue

            df_file = pd.read_csv(os.path.join(folder_path, f))
            pressures = df_file['pressure_pa'].values
            heights = df_file['label_height_m'].values
            temps = df_file['temperature_c'].values

            # 用 label_height_m 反推"干净气压"
            clean_pressures = height_to_pressure(heights, ref_p, temps)
            # 相对气压
            clean_rel = clean_pressures - ref_p
            raw_rel = pressures - ref_p

            for i in range(window_size, len(raw_rel)):
                X_list.append(raw_rel[i - window_size:i])
                y_list.append(clean_rel[i])  # 目标：干净相对气压
                seq_info.append((folder, f))

    return np.array(X_list), np.array(y_list), seq_info


def create_regression_features(df):
    """回归模式：直接用 pressure + temperature → 干净气压"""
    ref_p = 101325.0
    pressures = df['pressure_pa'].values
    temps = df['temperature_c'].values
    heights = df['label_height_m'].values

    # 反推干净气压
    clean_pressures = height_to_pressure(heights, ref_p, temps)
    clean_rel = clean_pressures - ref_p

    X = np.column_stack([pressures - ref_p, temps])  # 相对气压 + 温度
    y = clean_rel  # 目标：干净相对气压
    return X, y


def build_filter_model(input_dim):
    """构建滤波 BP 模型（输入压力窗口 → 输出滤波后气压）"""
    model = Sequential(name='labeled_filter_bp')
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.15))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # 输出滤波后相对气压
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_labeled_filter():
    """主训练流程"""
    print("=" * 60)
    print("   标注数据滤波训练")
    print("   输入：过去%d个含噪气压 → 输出：滤波后当前气压" % WINDOW_SIZE)
    print("   高度通过公式从滤波后气压计算")
    print("=" * 60)

    if not os.path.exists(DATA_ROOT):
        print(f"错误：数据目录不存在 {DATA_ROOT}")
        return

    # 1. 选择传感器
    print("\n选择传感器：")
    print("  1. MS5611")
    print("  2. BMP280")
    print("  3. 双传感器联合（MS5611 + BMP280）")
    choice = input("请输入 (1/2/3): ").strip()

    sensors = []
    sensor_label = ""
    if choice == '1':
        sensors = ['ms5611']
        sensor_label = 'MS5611'
    elif choice == '2':
        sensors = ['bmp280']
        sensor_label = 'BMP280'
    else:
        sensors = ['ms5611', 'bmp280']
        sensor_label = 'MS5611+BMP280'

    # 2. 选择训练模式
    print("\n训练模式：")
    print("  1. 滤波模式 (推荐) — 过去10个气压窗口 → 预测滤波后气压")
    print("  2. 回归模式 — 当前 pressure + temp → 预测滤波后气压")
    mode_choice = input("请输入 (1/2): ").strip()
    use_filter_mode = (mode_choice == '1')

    # 3. 加载数据
    all_X, all_y = [], []

    for sensor in sensors:
        print(f"\n加载 {sensor.upper()} 标注数据...")
        df = load_all_labeled(DATA_ROOT, sensor)
        if df is None:
            continue

        if use_filter_mode:
            X_s, y_s, _ = create_filter_sequences(df)
        else:
            X_s, y_s = create_regression_features(df)

        all_X.append(X_s)
        all_y.append(y_s)

    if not all_X:
        print("没有加载到任何数据！")
        return

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)

    print(f"\n总样本数: {len(X)}")
    print(f"特征维度: {X.shape[1]}")
    print(f"目标(相对气压): min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

    # 4. 归一化（输入和目标都用 MinMax）
    combined_rel = X.flatten()
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(combined_rel.reshape(-1, 1))
    X_scaled = scaler_X.transform(X.reshape(-1, 1)).reshape(X.shape)

    # 目标(相对气压)也做 MinMax 归一化
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(y.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()

    print(f"\n气压归一化: min={scaler_X.data_min_[0]:.2f}, max={scaler_X.data_max_[0]:.2f}")
    print(f"目标归一化: min={scaler_y.data_min_[0]:.2f}, max={scaler_y.data_max_[0]:.2f}")

    # 5. 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    print(f"\n数据集划分：")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    # 6. 构建模型
    print(f"\n构建滤波 BP 模型...")
    model = build_filter_model(X.shape[1])
    model.summary()

    # 7. 训练
    print("\n开始训练...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 8. 评估（反归一化到相对气压域）
    print("\n评估模型...")
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # 反归一化到相对气压
    y_test_rel = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_rel = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    rmse_pa = np.sqrt(mean_squared_error(y_test_rel, y_pred_rel))
    mae_pa = mean_absolute_error(y_test_rel, y_pred_rel)

    print(f"\n{'='*40}")
    print(f"  测试集结果（相对气压域）:")
    print(f"  RMSE: {rmse_pa:.3f} Pa")
    print(f"  MAE:  {mae_pa:.3f} Pa")
    print(f"{'='*40}")

    # 9. 导出模型
    if len(sensors) == 2:
        model_name = f"dual_labeled_filter_bp"
    else:
        model_name = f"{sensors[0]}_labeled_filter_bp"

    keras_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    tflite_path = os.path.join(MODELS_DIR, f"{model_name}.tflite")

    model.save(keras_path)
    print(f"\nKeras model saved: {keras_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved: {tflite_path}")

    # 保存 Scaler 信息
    scaler_info = {
        'min': float(scaler_X.data_min_[0]),
        'max': float(scaler_X.data_max_[0]),
        'range': float(scaler_X.data_max_[0] - scaler_X.data_min_[0]),
        'y_min': float(scaler_y.data_min_[0]),
        'y_max': float(scaler_y.data_max_[0]),
        'y_range': float(scaler_y.data_max_[0] - scaler_y.data_min_[0]),
        'reference_pressure': 101325.0,
        'window_size': WINDOW_SIZE,
        'input_dim': X.shape[1],
        'model_type': 'bp',
        'sensor': sensor_label,
        'mode': 'labeled_filter_pressure',  # 明确标注为气压滤波模式
        'smooth_window': SMOOTH_WINDOW,
        'test_rmse_pa': float(rmse_pa),
        'test_mae_pa': float(mae_pa),
        'model_mode': 'labeled',
        'output_is_pressure': True,  # 明确输出为气压（非高度）
    }
    scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.json")
    with open(scaler_path, 'w') as f:
        json.dump(scaler_info, f, indent=2)
    print(f"Scaler info saved: {scaler_path}")

    # 10. 可视化（气压域）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Labeled Filter Training (Pressure Output) - {sensor_label}', fontsize=14)

    # 训练曲线
    axes[0, 0].plot(history.history['loss'], label='Train Loss', alpha=0.8)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', alpha=0.8)
    axes[0, 0].set_title('Loss (MSE)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history.history['mae'], label='Train MAE', alpha=0.8)
    axes[0, 1].plot(history.history['val_mae'], label='Val MAE', alpha=0.8)
    axes[0, 1].set_title('MAE (normalized)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 预测 vs 真实（气压域）
    axes[0, 2].scatter(y_test_rel, y_pred_rel, alpha=0.5, s=15, c='steelblue')
    min_val = min(y_test_rel.min(), y_pred_rel.min())
    max_val = max(y_test_rel.max(), y_pred_rel.max())
    axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=1)
    axes[0, 2].set_title(f'Predicted vs True Rel Pressure (MAE={mae_pa:.3f}Pa)')
    axes[0, 2].set_xlabel('True Relative Pressure (Pa)')
    axes[0, 2].set_ylabel('Predicted Relative Pressure (Pa)')
    axes[0, 2].grid(True)

    # 误差分布（气压域）
    errors = y_test_rel - y_pred_rel
    axes[1, 0].hist(errors, bins=40, alpha=0.7, color='steelblue', edgecolor='white')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title(f'Error Distribution (MAE={mae_pa:.3f}Pa, RMSE={rmse_pa:.3f}Pa)')
    axes[1, 0].set_xlabel('Error (Pa)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True)

    # 时序对比（取前200个测试样本）
    n_show = min(200, len(y_test_rel))
    axes[1, 1].plot(y_test_rel[:n_show], 'g-', label='Clean Pressure (target)', alpha=0.8, linewidth=1.5)
    axes[1, 1].plot(y_pred_rel[:n_show], 'b--', label='NN Filtered Pressure', alpha=0.8, linewidth=1.5)
    axes[1, 1].set_title(f'Pressure Filter Result (First {n_show} samples)')
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('Relative Pressure (Pa)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True)

    # 显示原始含噪气压 vs 干净气压
    x_test_raw = scaler_X.inverse_transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    raw_show = []
    for i in range(n_show):
        raw_show.append(x_test_raw[i][-1])  # 窗口最后一个含噪相对气压
    axes[1, 2].plot(y_test_rel[:n_show], 'g-', label='Clean (target)', alpha=0.8, linewidth=1.5)
    axes[1, 2].plot(y_pred_rel[:n_show], 'b--', label='NN Filtered', alpha=0.8, linewidth=1.5)
    axes[1, 2].plot(raw_show[:n_show], 'r:', label='Raw (noisy)', alpha=0.5, linewidth=1)
    axes[1, 2].set_title('Raw vs Clean vs NN Filtered Pressure')
    axes[1, 2].set_xlabel('Sample')
    axes[1, 2].set_ylabel('Relative Pressure (Pa)')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, f"{model_name}_training.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")

    # 额外：保存一个详细的误差分析报告
    report = {
        'model': model_name,
        'sensor': sensor_label,
        'window_size': WINDOW_SIZE,
        'smooth_window': SMOOTH_WINDOW,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'input_dim': X.shape[1],
        'test_rmse_pa': float(rmse_pa),
        'test_mae_pa': float(mae_pa),
        'output_is_pressure': True,
    }
    report_path = os.path.join(MODELS_DIR, f"{model_name}_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")

    print(f"\n{'='*60}")
    print(f"  标注数据滤波训练完成！")
    print(f"  模型: {model_name}")
    print(f"  输入: 过去{WINDOW_SIZE}个含噪相对气压 → 输出: 滤波后相对气压")
    print(f"  性能: RMSE={rmse_pa:.3f}Pa  MAE={mae_pa:.3f}Pa")
    print(f"  (高度由公式 PressureToAltitudeWithTemp 从滤波后气压计算)")
    print(f"{'='*60}")

    return model, scaler_info


if __name__ == "__main__":
    train_labeled_filter()
