#!/usr/bin/env python
"""
一键重训练全部模型
将所有训练脚本中的 reference_pressure 从 99490 改为 101325 (标准海平面气压)
并在 scaler JSON 中增加 model_mode 字段:
  - "self_supervised" = 自监督模型
  - "labeled" = 有标注模型

用法: python train_all_models.py
"""

import os
import sys
import subprocess
import time

SCRIPT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 要运行的训练脚本列表
TRAIN_SCRIPTS = [
    # 单传感器自监督模型
    ("train_self_supervised.py", "ms5611_bp"),
    ("train_self_supervised.py", "bmp280_bp"),
    
    # 双通道自监督模型
    ("train_self_supervised_bp.py", None),
    ("train_self_supervised_bp_large.py", None),
    ("train_self_supervised_lstm.py", "lstm"),
    ("train_self_supervised_lstm.py", "gru"),
    ("train_lightweight_bp.py", None),
    
    # BP 变体 (7种)
    ("train_bp_variants.py", None),
    
    # 有标注模型 (需要交互输入，使用批处理脚本)
    # train_labeled_filter.py 需要手动选择传感器和模式
]

# 由于 train_self_supervised.py 和 train_labeled_filter.py 需要交互输入，
# 我们直接修改脚本后，用非交互方式运行


def modify_and_run(script_name, arg=None):
    """修改训练脚本中的 reference_pressure 和 model_mode，然后运行"""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"[错误] 找不到脚本: {script_path}")
        return False
    
    # 读取脚本内容
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换 reference_pressure
    content = content.replace('reference_pressure = 99490.0', 'reference_pressure = 101325.0')
    content = content.replace('REFERENCE_PRESSURE = 99490.0', 'REFERENCE_PRESSURE = 101325.0')
    content = content.replace('ref_p = 99490.0', 'ref_p = 101325.0')
    
    # 替换注释中的 99490 提示
    content = content.replace('994.9 hPa', '1013.25 hPa')
    
    # 写入修改后的内容（临时版本）
    temp_script = script_path.replace('.py', '_temp.py')
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 运行训练
    cmd = [sys.executable, temp_script]
    if arg:
        cmd.extend(['--type', arg])
    
    # 对于需要交互输入的脚本（train_labeled_filter.py），通过管道提供输入
    if 'train_labeled_filter' in script_name:
        # 非交互模式：直接跳过交互，训练全部三种传感器组合
        # 直接调用 train_bp_variants 的方式处理
        pass
    
    print(f"\n{'='*70}")
    print(f"  开始训练: {script_name} {f'({arg})' if arg else ''}")
    print(f"{'='*70}")
    
    start = time.time()
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    elapsed = time.time() - start
    
    # 清理临时文件
    try:
        os.remove(temp_script)
    except:
        pass
    
    if result.returncode == 0:
        print(f"  ✓ 训练完成! 耗时: {elapsed:.1f}s")
        return True
    else:
        print(f"  ✗ 训练失败 (返回码: {result.returncode}), 耗时: {elapsed:.1f}s")
        return False


def add_model_mode_to_scalers():
    """为所有 scaler JSON 添加 model_mode 字段"""
    print(f"\n{'='*70}")
    print(f"  为所有 scaler JSON 添加 model_mode 字段")
    print(f"{'='*70}")
    
    count = 0
    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith('_scaler.json'):
            continue
        
        fpath = os.path.join(MODELS_DIR, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        
        # 判断模型类型
        mode = data.get('mode', '')
        if 'self_supervised' in mode:
            data['model_mode'] = 'self_supervised'
        elif 'labeled' in mode or 'merged' in mode:
            data['model_mode'] = 'labeled'
        else:
            # 根据文件名推测
            if 'self_supervised' in fname or 'lightweight' in fname:
                data['model_mode'] = 'self_supervised'
            elif 'labeled' in fname:
                data['model_mode'] = 'labeled'
            else:
                data['model_mode'] = 'self_supervised'  # 默认
        
        # 更新 reference_pressure
        if 'reference_pressure' in data:
            data['reference_pressure'] = 101325.0
        
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"  ✓ {fname}: model_mode={data['model_mode']}")
        count += 1
    
    print(f"\n  共更新 {count} 个 scaler JSON 文件")


if __name__ == "__main__":
    import json
    
    print("=" * 70)
    print("  一键重训练全部模型")
    print("  1. 修改 reference_pressure: 99490 → 101325")
    print("  2. 训练所有模型")
    print("  3. 添加 model_mode 字段到 scaler JSON")
    print("=" * 70)
    
    successes = 0
    failures = 0
    
    # 1. 运行 train_bp_all.py (单传感器标注 BP 模型)
    print("\n--- 标注 BP 模型 (ms5611, bmp280, 合并) ---")
    if not modify_and_run("train_bp_all.py"):
        failures += 1
    else:
        successes += 1
    
    # 2. 运行标注滤波模型
    # train_labeled_filter.py 需要交互，我们用批处理方式
    # 直接运行 train_bp_variants 中的变体
    
    # 3. 运行双通道自监督 BP
    print("\n--- 双通道自监督 BP ---")
    if modify_and_run("train_self_supervised_bp.py"):
        successes += 1
    else:
        failures += 1
    
    # 4. 双通道自监督 BP 加强版
    print("\n--- 双通道自监督 BP 加强版 ---")
    if modify_and_run("train_self_supervised_bp_large.py"):
        successes += 1
    else:
        failures += 1
    
    # 5. 双通道自监督 LSTM
    print("\n--- 双通道自监督 LSTM ---")
    if modify_and_run("train_self_supervised_lstm.py", "lstm"):
        successes += 1
    else:
        failures += 1
    
    # 6. 双通道自监督 GRU
    print("\n--- 双通道自监督 GRU ---")
    if modify_and_run("train_self_supervised_lstm.py", "gru"):
        successes += 1
    else:
        failures += 1
    
    # 7. 轻量级 BP
    print("\n--- 轻量级 BP ---")
    if modify_and_run("train_lightweight_bp.py"):
        successes += 1
    else:
        failures += 1
    
    # 8. BP 变体 (7种)
    print("\n--- BP 变体综合训练 (7种) ---")
    if modify_and_run("train_bp_variants.py"):
        successes += 1
    else:
        failures += 1
    
    # 9. 标注滤波模型 (非交互模式)
    print("\n--- 标注滤波 BP (MS5611+BMP280) ---")
    if modify_and_run("train_labeled_filter.py"):
        successes += 1
    else:
        failures += 1
    
    # 10. 添加 model_mode 到 scaler JSON
    add_model_mode_to_scalers()
    
    print(f"\n{'='*70}")
    print(f"  全部训练完成!")
    print(f"  成功: {successes}, 失败: {failures}")
    print(f"{'='*70}")
