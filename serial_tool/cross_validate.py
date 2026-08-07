#!/usr/bin/env python
"""
交叉验证：测试 BMP280 数据通过 MS5611 模型的效果
对比三种方案：
  1. BMP280 用自己的模型（基准）
  2. BMP280 桥接到 MS5611 空间再用 MS5611 模型
  3. BMP280 直接过 MS5611 模型（最简方案）
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_root = os.path.join(os.path.dirname(__file__), "data")
models_dir = os.path.join(os.path.dirname(__file__), "models")
WINDOW_SIZE = 10
REF_PRESSURE = 101325.0  # 标准海平面气压（1013.25 hPa）


def load_data(sensor='bmp280'):
    """加载传感器自监督数据"""
    all_pressure = []
    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for f in os.listdir(folder_path):
            if f.startswith(f'self_supervised_{sensor}') and f.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, f))
                all_pressure.extend(df['pressure_pa'].values.tolist())
    return np.array(all_pressure)


def create_sequences(data, window_size=WINDOW_SIZE):
    """创建自监督序列"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def run_tflite(interpreter, in_buf):
    """运行 TFLite 推理"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], in_buf.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index']).flatten()


def evaluate_model(model_path, scaler, X_rel, y_rel, label):
    """评估模型在给定数据上的表现"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # 归一化
    rel_min = scaler['min']
    rel_range = scaler['range']
    
    X_norm = (X_rel - rel_min) / rel_range
    y_norm = (y_rel - rel_min) / rel_range
    
    # 裁剪
    X_norm = np.clip(X_norm, 0, 1)
    y_norm = np.clip(y_norm, 0, 1)
    
    # 推理
    y_pred_norm = np.array([run_tflite(interpreter, x.reshape(1, -1))[0] for x in X_norm])
    
    # 反归一化到相对气压
    y_pred_rel = y_pred_norm * rel_range + rel_min
    
    # 反归一化到绝对气压
    y_pred = y_pred_rel + REF_PRESSURE
    y_true = y_rel + REF_PRESSURE
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    max_err = np.max(np.abs(y_true - y_pred))
    
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  MAE:     {mae:>8.2f} Pa")
    print(f"  RMSE:    {rmse:>8.2f} Pa")
    print(f"  Max Err: {max_err:>8.2f} Pa")
    
    return y_true, y_pred, {'mae': mae, 'rmse': rmse, 'max_err': max_err}


# ============================================================
# 主流程
# ============================================================
print("=" * 60)
print("  交叉验证：BMP280 数据通过 MS5611 模型")
print("=" * 60)

# 1. 加载数据
print("\n--- 加载数据 ---")
bmp280_pressure = load_data('bmp280')
print(f"  BMP280 总样本数: {len(bmp280_pressure)}")

# 使用所有数据（不分 train/test，验证的是"模型能否对 BMP280 数据正确滤波"）
pressure_rel = bmp280_pressure - REF_PRESSURE
X_rel, y_rel = create_sequences(pressure_rel)
print(f"  序列数: {len(X_rel)}")

# 2. 加载 scaler 信息
with open(os.path.join(models_dir, 'ms5611_self_supervised_bp_scaler.json')) as f:
    ms5611_scaler = json.load(f)
with open(os.path.join(models_dir, 'bmp280_self_supervised_bp_scaler.json')) as f:
    bmp280_scaler = json.load(f)

print(f"\n  MS5611 scaler: min={ms5611_scaler['min']:.1f}, max={ms5611_scaler['max']:.1f}, range={ms5611_scaler['range']:.1f}")
print(f"  BMP280 scaler: min={bmp280_scaler['min']:.2f}, max={bmp280_scaler['max']:.2f}, range={bmp280_scaler['range']:.2f}")

# 3. 方案 A: BMP280 用自己的模型（基准）
ms5611_model = os.path.join(models_dir, 'ms5611_self_supervised_bp.tflite')
bmp280_model = os.path.join(models_dir, 'bmp280_self_supervised_bp.tflite')

results = {}

# A: BMP280 用自己的模型
y_true, y_pred_bmp, res = evaluate_model(bmp280_model, bmp280_scaler, X_rel, y_rel, "方案 A: BMP280 用自己的模型（基准）")
results['a_bmp_own'] = y_pred_bmp
results['a_mae'] = res['mae']
results['a_rmse'] = res['rmse']
results['a_max_err'] = res['max_err']

# B: BMP280 桥接到 MS5611 空间
# 先计算 BMP280 归一化值，再反归一化到 MS5611 的相对气压空间
rel_min_bmp = bmp280_scaler['min']
rel_range_bmp = bmp280_scaler['range']
rel_min_ms = ms5611_scaler['min']
rel_range_ms = ms5611_scaler['range']

# BMP280 归一化到 [0,1]
X_norm_bmp = (X_rel - rel_min_bmp) / rel_range_bmp
y_norm_bmp = (y_rel - rel_min_bmp) / rel_range_bmp
X_norm_bmp = np.clip(X_norm_bmp, 0, 1)
y_norm_bmp = np.clip(y_norm_bmp, 0, 1)

# 用 BMP280 的归一化值，送入 MS5611 模型（因为数学上 norm_bmp == 桥接后的 norm）
interpreter_ms = tf.lite.Interpreter(model_path=ms5611_model)
interpreter_ms.allocate_tensors()
y_pred_norm_ms = np.array([run_tflite(interpreter_ms, x.reshape(1, -1))[0] for x in X_norm_bmp])

# 用 MS5611 scaler 反归一化
y_pred_rel_ms = y_pred_norm_ms * rel_range_ms + rel_min_ms
y_pred_bridge = y_pred_rel_ms + REF_PRESSURE
mae_bridge = mean_absolute_error(y_rel + REF_PRESSURE, y_pred_bridge)
rmse_bridge = np.sqrt(mean_squared_error(y_rel + REF_PRESSURE, y_pred_bridge))
max_err_bridge = np.max(np.abs(y_rel + REF_PRESSURE - y_pred_bridge))

print(f"\n{'='*55}")
print(f"  方案 B: BMP280 桥接→MS5611 模型")
print(f"{'='*55}")
print(f"  MAE:     {mae_bridge:>8.2f} Pa")
print(f"  RMSE:    {rmse_bridge:>8.2f} Pa")
print(f"  Max Err: {max_err_bridge:>8.2f} Pa")
results['b_bridge'] = y_pred_bridge

# C: BMP280 直接用 MS5611 scaler（最简方案）
X_norm_direct = (X_rel - rel_min_ms) / rel_range_ms
y_norm_direct = (y_rel - rel_min_ms) / rel_range_ms
X_norm_direct = np.clip(X_norm_direct, 0, 1)
y_norm_direct = np.clip(y_norm_direct, 0, 1)

interpreter_ms2 = tf.lite.Interpreter(model_path=ms5611_model)
interpreter_ms2.allocate_tensors()
y_pred_norm_direct = np.array([run_tflite(interpreter_ms2, x.reshape(1, -1))[0] for x in X_norm_direct])

y_pred_rel_direct = y_pred_norm_direct * rel_range_ms + rel_min_ms
y_pred_direct = y_pred_rel_direct + REF_PRESSURE
mae_direct = mean_absolute_error(y_rel + REF_PRESSURE, y_pred_direct)
rmse_direct = np.sqrt(mean_squared_error(y_rel + REF_PRESSURE, y_pred_direct))
max_err_direct = np.max(np.abs(y_rel + REF_PRESSURE - y_pred_direct))

print(f"\n{'='*55}")
print(f"  方案 C: BMP280 直接过 MS5611 模型（最简方案）")
print(f"{'='*55}")
print(f"  MAE:     {mae_direct:>8.2f} Pa")
print(f"  RMSE:    {rmse_direct:>8.2f} Pa")
print(f"  Max Err: {max_err_direct:>8.2f} Pa")
results['c_direct'] = y_pred_direct

# 4. 方案间差异分析
print(f"\n{'='*55}")
print(f"  方案对比总结")
print(f"{'='*55}")
print(f"  {'方案':<30s} {'MAE(Pa)':<10s} {'RMSE(Pa)':<10s} {'MaxErr(Pa)':<10s}")
print(f"  {'-'*55}")
print(f"  {'A: BMP280 用自己的模型':<30s} {results['a_mae']:<10.2f} {results['a_rmse']:<10.2f} {results['a_max_err']:<10.2f}")

# 5. 可视化
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('BMP280 数据交叉验证：不同模型/方案效果对比', fontsize=14, fontweight='bold')

n_show = min(500, len(y_rel))

# 第一行：原始 vs 滤波
axes[0, 0].plot(y_rel[:n_show] + REF_PRESSURE, label='Raw BMP280', alpha=0.5, color='gray', linewidth=0.8)
axes[0, 0].plot(results['a_bmp_own'][:n_show], label='A: BMP280 自身模型', alpha=0.8, linewidth=1.5)
axes[0, 0].set_title('方案 A vs 原始数据')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(results['a_bmp_own'][:n_show], label='A: BMP280 自身模型', alpha=0.8, linewidth=1.5)
axes[0, 1].plot(results['b_bridge'][:n_show], label='B: 桥接→MS5611', alpha=0.8, linewidth=1.5, linestyle='--')
axes[0, 1].plot(results['c_direct'][:n_show], label='C: 直接过 MS5611', alpha=0.8, linewidth=1.5, linestyle=':')
axes[0, 1].set_title('三种方案对比')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 第二行：方案间差异
diff_b = results['b_bridge'] - results['a_bmp_own']
diff_c = results['c_direct'] - results['a_bmp_own']

axes[1, 0].plot(diff_b[:n_show], label=f'B-A (桥接差异)', color='orange', alpha=0.7)
axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
axes[1, 0].set_title(f'方案 B 与基准的差异 (MAE={mae_bridge-results["a_mae"]:.2f}Pa)')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(diff_c[:n_show], label=f'C-A (直接差异)', color='green', alpha=0.7)
axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
axes[1, 1].set_title(f'方案 C 与基准的差异 (MAE={mae_direct-results["a_mae"]:.2f}Pa)')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

# 第三行：误差分布
axes[2, 0].hist(diff_b, bins=80, alpha=0.6, color='orange', label=f'B-A (σ={np.std(diff_b):.2f}Pa)')
axes[2, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[2, 0].set_title('方案 B 与基准的误差分布')
axes[2, 0].legend(fontsize=9)
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].hist(diff_c, bins=80, alpha=0.6, color='green', label=f'C-A (σ={np.std(diff_c):.2f}Pa)')
axes[2, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[2, 1].set_title('方案 C 与基准的误差分布')
axes[2, 1].legend(fontsize=9)
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(models_dir, 'cross_validation.png')
plt.savefig(plot_path, dpi=150)
print(f"\n对比图已保存: {plot_path}")
plt.close()

# 6. 打印关键数值对比
print(f"\n{'='*60}")
print(f"  最终结论")
print(f"{'='*60}")
print(f"\n  {'方案':<35s} {'MAE(Pa)':<10s} {'RMSE(Pa)':<10s} {'MaxErr(Pa)':<10s}")
print(f"  {'-'*60}")
print(f"  {'A: BMP280 自身模型（基准）':<35s} {results['a_mae']:<10.2f} {results['a_rmse']:<10.2f} {results['a_max_err']:<10.2f}")
print(f"  {'B: 桥接→MS5611':<35s} {mae_bridge:<10.2f} {rmse_bridge:<10.2f} {max_err_bridge:<10.2f}")
print(f"  {'C: 直接过 MS5611（最简）':<35s} {mae_direct:<10.2f} {rmse_direct:<10.2f} {max_err_direct:<10.2f}")
print(f"  {'-'*60}")
print(f"  {'B vs A 差异':<35s} {mae_bridge-results['a_mae']:<+10.2f} {rmse_bridge-results['a_rmse']:<+10.2f} {max_err_bridge-results['a_max_err']:<+10.2f}")
print(f"  {'C vs A 差异':<35s} {mae_direct-results['a_mae']:<+10.2f} {rmse_direct-results['a_rmse']:<+10.2f} {max_err_direct-results['a_max_err']:<+10.2f}")

print(f"\n  结论：")
if abs(mae_bridge - results['a_mae']) < 2.0:
    print(f"  ✓ 方案 B（桥接）与基准方案 A 差异极小 (MAE差={abs(mae_bridge-results['a_mae']):.2f}Pa)，完全可以接受！")
else:
    print(f"  ⚠ 方案 B（桥接）与基准方案 A 差异 {abs(mae_bridge-results['a_mae']):.2f}Pa")

if abs(mae_direct - results['a_mae']) < 2.0:
    print(f"  ✓ 方案 C（直接过 MS5611）与基准差异极小 (MAE差={abs(mae_direct-results['a_mae']):.2f}Pa)，完全可以接受！")
else:
    print(f"  ⚠ 方案 C（直接过 MS5611）与基准差异 {abs(mae_direct-results['a_mae']):.2f}Pa")
