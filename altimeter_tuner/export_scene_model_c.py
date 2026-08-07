#!/usr/bin/env python
"""
将训练好的场景识别模型 (models/scene_classifier.json) 导出为 C 头文件，
供单片机 (Core/Src/scene_classifier.c) 直接前向推理使用（纯浮点 MLP，
与 scene_classifier.py 的 numpy 前向完全一致，不依赖 X-CUBE-AI）。

输出：
  Core/Inc/scene_classifier_model.h

权重布局：W[in][out]（与 numpy 的 a @ W 一致），便于 C 端 out[j] = Σ a[i]*W[i][j]。
"""

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scene_classifier import DEFAULT_MODEL  # noqa: E402

JSON_PATH = os.path.join(SCRIPT_DIR, 'models', 'scene_classifier.json')
OUT_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'Core', 'Inc',
                                          'scene_classifier_model.h'))


def fmt(x):
    """输出可 round-trip 的浮点字面量（C 中作 double 字面量再赋给 float）。"""
    s = repr(float(x))
    if 'e' not in s and 'E' not in s and '.' not in s and 'n' not in s:
        s += '.0'
    return s


def emit_2d(name, W):
    rows = []
    for i, row in enumerate(W):
        inner = ', '.join(fmt(v) for v in row)
        comma = ',' if i < len(W) - 1 else ''
        rows.append(f"  {{{inner}}}{comma}")
    return (f"static const float {name}[{len(W)}][{len(W[0])}] = {{\n"
            + "\n".join(rows) + "\n};")


def emit_1d(name, b):
    inner = ', '.join(fmt(v) for v in b)
    return f"static const float {name}[{len(b)}] = {{ {inner} }};"


def main():
    if not os.path.isfile(JSON_PATH):
        print(f"模型不存在: {JSON_PATH}\n请先运行 train_scene_classifier.py")
        sys.exit(1)
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    input_dim = int(cfg['input_dim'])
    window = int(cfg['window_size'])
    classes = list(cfg['class_names'])
    n_classes = len(classes)
    ref = float(cfg['ref_pressure'])
    fmean = list(cfg['feature_mean'])
    fstd = list(cfg['feature_std'])
    for i in range(len(fstd)):
        if fstd[i] == 0.0:
            fstd[i] = 1.0
    layers = cfg['layers']
    # 隐藏层维度
    h1 = len(layers[0]['b'])
    h2 = len(layers[1]['b'])
    assert len(layers) == 3, "期望 3 层 (2 隐藏 + 输出)"
    assert n_classes == len(layers[2]['b']), "输出维度与类别数不一致"

    W1, b1 = layers[0]['W'], layers[0]['b']
    W2, b2 = layers[1]['W'], layers[1]['b']
    W3, b3 = layers[2]['W'], layers[2]['b']

    # 类别名 -> C 字符串数组
    cls_strs = ', '.join(f'"{c}"' for c in classes)

    lines = []
    lines.append("/* 本文件由 export_scene_model_c.py 自动生成，请勿手工修改 */")
    lines.append("#ifndef SCENE_CLASSIFIER_MODEL_H")
    lines.append("#define SCENE_CLASSIFIER_MODEL_H")
    lines.append("")
    lines.append(f"#define SCENE_INPUT_DIM      {input_dim}")
    lines.append(f"#define SCENE_WINDOW_SIZE    {window}")
    lines.append(f"#define SCENE_HIDDEN1        {h1}")
    lines.append(f"#define SCENE_HIDDEN2        {h2}")
    lines.append(f"#define SCENE_N_CLASSES      {n_classes}")
    lines.append(f"#define SCENE_REF_PRESSURE   {fmt(ref)}f")
    lines.append("")
    lines.append(f"static const char* SCENE_CLASS_NAMES[SCENE_N_CLASSES] = {{{cls_strs}}};")
    lines.append("")
    lines.append(emit_1d("SCENE_FEATURE_MEAN", fmean))
    lines.append(emit_1d("SCENE_FEATURE_STD", fstd))
    lines.append("")
    lines.append(emit_2d("SCENE_W1", W1))
    lines.append(emit_2d("SCENE_W2", W2))
    lines.append(emit_2d("SCENE_W3", W3))
    lines.append(emit_1d("SCENE_B1", b1))
    lines.append(emit_1d("SCENE_B2", b2))
    lines.append(emit_1d("SCENE_B3", b3))
    lines.append("")
    lines.append("#endif /* SCENE_CLASSIFIER_MODEL_H */")
    lines.append("")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"已生成: {OUT_PATH}")
    print(f"  输入={input_dim} 窗口={window} 隐藏={h1}/{h2} 类别={classes} 参考={ref} Pa")


if __name__ == '__main__':
    main()
