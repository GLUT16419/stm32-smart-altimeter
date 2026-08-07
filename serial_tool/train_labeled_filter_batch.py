#!/usr/bin/env python
"""批处理训练标注滤波模型（非交互模式），自动选择各传感器组合"""
import os, sys, subprocess, time

SCRIPT_DIR = os.path.dirname(__file__)
train_script = os.path.join(SCRIPT_DIR, "train_labeled_filter.py")

# 传感器组合: (输入选择数字, 传感器标签)
sensors = [
    ('1', 'ms5611'),
    ('2', 'bmp280'),
    ('3', 'dual'),
]

for sensor_choice, sensor_label in sensors:
    print(f"\n{'='*60}")
    print(f"  训练标注滤波模型: {sensor_label}")
    print(f"{'='*60}")
    
    # 提供输入: 传感器选择(1/2/3) + 滤波模式(1)
    # train_labeled_filter.py 的 input() 会依次读取
    start = time.time()
    
    proc = subprocess.Popen(
        [sys.executable, train_script],
        cwd=SCRIPT_DIR,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # 依次发送: 传感器选择, 训练模式选择
    out, _ = proc.communicate(input=f"{sensor_choice}\n1\n")
    elapsed = time.time() - start
    
    if proc.returncode == 0:
        print(f"  ✓ {sensor_label} 完成! 耗时: {elapsed:.1f}s")
    else:
        print(f"  ✗ {sensor_label} 失败! 耗时: {elapsed:.1f}s")
        print(f"  输出:\n{out[-500:] if len(out) > 500 else out}")

print(f"\n{'='*60}")
print(f"  标注滤波模型全部训练完成!")
print(f"{'='*60}")
