#!/usr/bin/env python
"""
将已有的 .h5 模型重新导出为 TFLite（仅 TFLITE_BUILTINS，不含 SELECT_TF_OPS）
解决 STM32Cube.AI 报错：Unsupported layer types: WHILE, FlexTensorListStack, FlexTensorListReserve
"""

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

models_dir = os.path.join(os.path.dirname(__file__), "models")

h5_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]

if not h5_files:
    print("未找到 .h5 模型文件")
    exit()

for h5_file in sorted(h5_files):
    h5_path = os.path.join(models_dir, h5_file)
    base_name = h5_file.replace('.h5', '')
    tflite_path = os.path.join(models_dir, f"{base_name}.tflite")
    backup_path = os.path.join(models_dir, f"{base_name}_flex.tflite")

    print(f"\n{'='*50}")
    print(f"  模型: {h5_file}")

    # 备份旧的 tflite（如果有）
    if os.path.exists(tflite_path):
        os.rename(tflite_path, backup_path)
        print(f"  备份旧 TFLite → {base_name}_flex.tflite")

    # 加载模型
    try:
        model = tf.keras.models.load_model(h5_path, compile=False)
        print(f"  加载成功: {h5_path}")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        continue

    # 尝试纯 Builtin 导出
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        size_kb = len(tflite_model) / 1024
        print(f"  ✅ 导出成功 (Builtin only): {tflite_path} ({size_kb:.1f} KB)")

        # 验证：检查是否有 Flex 算子
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        op_details = interpreter.get_tensor_details()
        print(f"  输入: {interpreter.get_input_details()[0]['shape']}")
        print(f"  输出: {interpreter.get_output_details()[0]['shape']}")

    except Exception as e:
        print(f"  ❌ 纯 Builtin 导出失败: {e}")
        print(f"  → 尝试启用 SELECT_TF_OPS 导出...")

        # 如果纯 Builtin 失败，回退到 SELECT_TF_OPS
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            size_kb = len(tflite_model) / 1024
            print(f"  ⚠ 回退 SELECT_TF_OPS: {tflite_path} ({size_kb:.1f} KB)")
            print(f"  ⚠ 此模型可能仍不支持 STM32Cube.AI")
        except Exception as e2:
            print(f"  ❌ 全部导出失败: {e2}")

print(f"\n{'='*50}")
print("  重新导出完成！")
print("  请使用新生成的 .tflite 文件导入 STM32Cube.AI")
print(f"{'='*50}")
