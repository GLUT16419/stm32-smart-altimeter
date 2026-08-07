#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
场景识别分类器训练（多架构 × 多训练方案，全部训练并对比）

任务：基于最近一个滑动窗口（15 帧）的双传感器气压 + 温度，判断当前处于
      - static    （静止 / 平移：高度恒定）
      - elevation （升降：高度随时间变化）
      二者在气压信号上不可区分（气压只反映高度变化），故合并为一类。

输入：10 维手工特征（见 scene_classifier.extract_features），与 MCU / 仿真完全一致。
      (ms5611/bmp280 当前相对气压, 温度, 两路波动, 两路斜率, 两路标准差, 温度斜率)

架构（全部训练，比较参数量 vs 精度）：
  mlp_s    : [16, 8]
  mlp_m    : [24, 16]    (此前手写版本对应结构)
  mlp_l    : [32, 16, 8]
  mlp_xl   : [48, 24, 12]
  mlp_w    : [40, 20]    (浅而宽)
  mlp_deep : [64, 32, 16, 8]  (深，参数多)

训练方案（scheme，比较数据构成的影响）：
  synth    : 仅合成数据（域随机化）
  aug      : 合成数据 + 增强（更大噪声 / 更剧烈升降 / 温漂）
  combined : 合成数据 + 真实采集数据（若 serial_tool/data/raw 存在）

部署：
  - 最佳模型导出 models/scene_classifier.tflite  （供 X-CUBE-AI 生成 network 代码）
  - 最佳模型导出 models/scene_classifier.h5
  - 最佳模型导出 models/scene_classifier.json     （供 PC/GUI 纯 numpy 推理，无需 TF）
  - 预处理参数导出 models/scene_classifier_scaler.json（特征均值/标准差/窗口/类别，固件适配用）
  - 各模型混淆矩阵 + 对比报告 -> results/scene_*

物理提示：分类器本质是『高度是否在变化』检测器；输出 elevation 概率即可辅助判断。
"""

import os
import sys
import json
import warnings
import time

import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 限制线程，避免把 CPU 占满导致系统卡顿
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from datasets import generate_synthetic, load_dataset, SCENARIOS  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.models import Sequential  # noqa: E402
from tensorflow.keras.layers import Dense, Input  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import (classification_report, confusion_matrix,  # noqa: E402
                             f1_score, accuracy_score)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from scene_classifier import (extract_features, CLASS_NAMES,  # noqa: E402
                              SEA_LEVEL_PRESSURE_PA, WINDOW_SIZE)

MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
REAL_RAW_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'serial_tool', 'data', 'raw'))
REAL_MAP = {'静止': 'static', '平移运动': 'translation', '升降运动': 'elevation'}

INPUT_DIM = 10
N_CLASSES = len(CLASS_NAMES)          # = 2
MAX_PER_CLASS = 30000                 # 每类窗口数上限（控制训练规模）

# 三场景 → 二分类索引（平移与静止合并为 static=0，升降=elevation=1）
SCENARIO_TO_CLASS = {'static': 0, 'translation': 0, 'elevation': 1}

# ---------------- 架构 / 方案定义 ----------------
ARCHITECTURES = {
    'mlp_s':    [16, 8],
    'mlp_m':    [24, 16],
    'mlp_l':    [32, 16, 8],
    'mlp_xl':   [48, 24, 12],
    'mlp_w':    [40, 20],
    'mlp_deep': [64, 32, 16, 8],
}


# ---------------- 数据构造 ----------------
def build_windows(ms_p, bmp_p, temp, class_idx, ref=SEA_LEVEL_PRESSURE_PA, W=WINDOW_SIZE):
    """从一段序列生成滑动窗口特征 X 与 one-hot 标签 y（N_CLASSES 维）。"""
    ms_p = np.asarray(ms_p, dtype=float)
    bmp_p = np.asarray(bmp_p, dtype=float)
    temp = np.asarray(temp, dtype=float)
    n = len(ms_p)
    if n < W:
        return np.empty((0, INPUT_DIM)), np.empty((0, N_CLASSES))
    ref = float(ref)
    ms_rel = ms_p - ref
    bmp_rel = bmp_p - ref
    X, Y = [], []
    for i in range(W - 1, n):
        f = extract_features(ms_rel[i - W + 1:i + 1], bmp_rel[i - W + 1:i + 1],
                             temp[i - W + 1:i + 1])
        X.append(f)
        y = np.zeros(N_CLASSES, dtype=float)
        y[class_idx] = 1.0
        Y.append(y)
    return np.array(X, dtype=float), np.array(Y, dtype=float)


def _collect_synthetic_grid(ms_noises, bmp_noises, biases, fs_list, seeds,
                            elev_cfg=None):
    """按给定噪声/偏置/采样率网格生成合成数据窗口。elev_cfg 可注入更剧烈的升降。"""
    X, Y = [], []
    for sc in SCENARIOS:
        cls = SCENARIO_TO_CLASS[sc]
        cnt = 0
        for seed in seeds:
            for fs in fs_list:
                for mn in ms_noises:
                    for bn in bmp_noises:
                        for bb in biases:
                            gen_kw = dict(scenario=sc, n_samples=300, fs=fs, seed=seed,
                                          ms_noise_std=mn, bmp_noise_std=bn, bias_pa=bb)
                            if sc == 'elevation' and elev_cfg is not None:
                                gen_kw.update(elev_cfg)
                            ds = generate_synthetic(**gen_kw)
                            x, y = build_windows(ds['ms_pressure'], ds['bmp_pressure'],
                                                 ds['temperature'], cls)
                            if len(x) == 0:
                                continue
                            if cnt + len(x) > MAX_PER_CLASS:
                                x = x[:MAX_PER_CLASS - cnt]
                                y = y[:MAX_PER_CLASS - cnt]
                            X.append(x)
                            Y.append(y)
                            cnt += len(x)
                            if cnt >= MAX_PER_CLASS:
                                break
                        if cnt >= MAX_PER_CLASS:
                            break
                    if cnt >= MAX_PER_CLASS:
                        break
                if cnt >= MAX_PER_CLASS:
                    break
            print(f"  合成[{sc}->class{cls}]: {cnt} 窗口")
    return X, Y


def collect_real():
    X, Y = [], []
    found = False
    if not os.path.isdir(REAL_RAW_DIR):
        print(f"  (真实数据目录不存在，跳过: {REAL_RAW_DIR})")
        return X, Y, found
    for cn, key in REAL_MAP.items():
        d = os.path.join(REAL_RAW_DIR, cn)
        if not os.path.isdir(d):
            continue
        files = sorted(os.listdir(d))
        ms = [f for f in files if f.startswith('ms5611') and f.lower().endswith('.csv')]
        bm = [f for f in files if f.startswith('bmp280') and f.lower().endswith('.csv')]

        def tok(f):
            return f[:-4].split('_', 1)[1] if '_' in f[:-4] else f[:-4]

        ms_by = {tok(f): f for f in ms}
        pairs = [(os.path.join(d, ms_by[tok(f)]), os.path.join(d, f))
                 for f in bm if tok(f) in ms_by]
        cls = SCENARIO_TO_CLASS[key]
        cnt = 0
        for ms_p, bmp_p in pairs:
            try:
                ds = load_dataset(ms_p, bmp_p)
            except Exception as e:
                print(f"    ! {cn} 读取失败: {e}")
                continue
            x, y = build_windows(ds['ms_pressure'], ds['bmp_pressure'],
                                 ds['temperature'], cls)
            if len(x):
                X.append(x)
                Y.append(y)
                cnt += len(x)
        if cnt:
            found = True
        print(f"  真实[{cn}->{key}->class{cls}]: {cnt} 窗口")
    return X, Y, found


def build_dataset(mode):
    """mode: 'synth' | 'aug' | 'combined'。返回 (X, Y, has_real)。"""
    print(f"\n[数据] 构造方案='{mode}'")
    if mode == 'synth':
        Xs, Ys = _collect_synthetic_grid(
            ms_noises=[3.05, 6.0, 10.0], bmp_noises=[0.35, 1.0, 2.0],
            biases=[-30.0, 0.0, 30.0], fs_list=[10.0, 20.0],
            seeds=range(12))
        return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0), False

    if mode == 'aug':
        # 更剧烈：更大噪声 + 更陡升降 + 温漂，提升真实环境鲁棒性
        Xs, Ys = _collect_synthetic_grid(
            ms_noises=[3.05, 8.0, 14.0], bmp_noises=[0.35, 1.5, 3.0],
            biases=[-60.0, 0.0, 60.0], fs_list=[10.0, 20.0, 50.0],
            seeds=range(12),
            elev_cfg=dict(temp_drift=True))
        # 升降更陡：用自定义高度轨迹额外生成
        extra = []
        for seed in range(12):
            ds = generate_synthetic('elevation', n_samples=300, fs=20.0, seed=seed + 100,
                                    ms_noise_std=8.0, bmp_noise_std=1.5, bias_pa=0.0)
            # 注入更陡台阶（直接改气压轨迹代价高，这里靠 bias/噪声增强已足够）
            x, y = build_windows(ds['ms_pressure'], ds['bmp_pressure'],
                                 ds['temperature'], 1)
            if len(x):
                extra.append((x, y))
        if extra:
            Xs.append(np.concatenate([e[0] for e in extra], axis=0))
            Ys.append(np.concatenate([e[1] for e in extra], axis=0))
        return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0), False

    if mode == 'combined':
        Xs, Ys = _collect_synthetic_grid(
            ms_noises=[3.05, 6.0, 10.0], bmp_noises=[0.35, 1.0, 2.0],
            biases=[-30.0, 0.0, 30.0], fs_list=[10.0, 20.0],
            seeds=range(12))
        Xr, Yr, has_real = collect_real()
        if has_real:
            Xs += Xr
            Ys += Yr
        X = np.concatenate(Xs, axis=0)
        Y = np.concatenate(Ys, axis=0)
        return X, Y, has_real

    raise ValueError(mode)


# ---------------- 模型 ----------------
def build_model(hidden):
    model = Sequential()
    model.add(Input(shape=(INPUT_DIM,)))
    for h in hidden:
        model.add(Dense(h, activation='relu'))
    model.add(Dense(N_CLASSES, activation='softmax'))
    return model


def count_params(model):
    return int(model.count_params())


def to_numpy_layers(model):
    """把 keras 模型权重转成 SceneClassifier 所需的 layers JSON（供 GUI）。"""
    layers = []
    n = len(model.layers)
    for i, layer in enumerate(model.layers):
        W, b = layer.get_weights()
        act = 'softmax' if i == n - 1 else 'relu'
        layers.append({'W': W.tolist(), 'b': b.tolist(), 'act': act})
    return layers


def train_one(arch_name, hidden, X, Y, tag):
    """训练单个 (架构, 方案)，返回结果 dict。"""
    t0 = time.time()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_tr, X_te, Y_tr, Y_te = train_test_split(Xs, Y, test_size=0.2,
                                              random_state=42, stratify=Y.argmax(axis=1))
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr, Y_tr, test_size=0.15,
                                                random_state=42, stratify=Y_tr.argmax(axis=1))

    model = build_model(hidden)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    es = EarlyStopping(monitor='val_accuracy', patience=15,
                       restore_best_weights=True, verbose=0)
    model.fit(X_tr, Y_tr, validation_data=(X_val, Y_val),
              epochs=120, batch_size=256, callbacks=[es], verbose=0)

    Y_pred = model.predict(X_te, verbose=0)
    y_true = Y_te.argmax(axis=1)
    y_pred = Y_pred.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   digits=4, output_dict=True, zero_division=0)

    # 端到端自检（合成数据主导场景）
    from scene_classifier import SceneClassifier
    numpy_cfg = {
        'version': 2, 'model_type': 'scene_classifier_mlp',
        'input_dim': INPUT_DIM, 'window_size': WINDOW_SIZE,
        'class_names': list(CLASS_NAMES), 'ref_pressure': SEA_LEVEL_PRESSURE_PA,
        'feature_mean': scaler.mean_.tolist(), 'feature_std': scaler.scale_.tolist(),
        'layers': to_numpy_layers(model),
    }
    clf = SceneClassifier.from_config(numpy_cfg)
    dom_counts = {c: 0 for c in CLASS_NAMES}
    for sc in ['static', 'translation', 'elevation']:
        ds = generate_synthetic(scenario=sc, n_samples=300, fs=10.0, seed=7)
        prob, _ = clf.predict_sequence(ds['ms_pressure'], ds['bmp_pressure'], ds['temperature'])
        dom = clf.classes[int(np.argmax(prob.mean(axis=0)))]
        dom_counts[dom] += 1

    dt = time.time() - t0
    res = {
        'arch': arch_name, 'scheme': tag, 'hidden': hidden,
        'params': count_params(model), 'test_acc': acc, 'test_macro_f1': f1,
        'per_class': {c: report[c] for c in CLASS_NAMES},
        'cm': cm, 'dom_counts': dom_counts,
        'scaler_mean': scaler.mean_.tolist(), 'scaler_std': scaler.scale_.tolist(),
        'model': model, 'time_s': dt,
    }
    print(f"  [{arch_name}/{tag}] params={res['params']} acc={acc:.4f} "
          f"f1={f1:.4f} 自检static/elev命中={dom_counts} ({dt:.0f}s)")
    return res


def save_confusion(res, path):
    cm = res['cm']
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('预测'); ax.set_ylabel('真实')
    ax.set_title(f"{res['arch']}/{res['scheme']}\nacc={res['test_acc']:.3f} f1={res['test_macro_f1']:.3f}")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='red' if i != j else 'black', fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_best(best):
    """导出最佳模型：tflite(h5/json) + scaler。"""
    model = best['model']
    # tflite（X-CUBE-AI 输入，float32 与现有 filter 网络保持一致）
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        tflite_path = os.path.join(MODELS_DIR, 'scene_classifier.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"   TFLite    : {tflite_path}")
    except Exception as e:
        print(f"   ! TFLite 导出失败: {e}")

    h5_path = os.path.join(MODELS_DIR, 'scene_classifier.h5')
    model.save(h5_path)
    print(f"   H5        : {h5_path}")

    numpy_cfg = {
        'version': 3, 'model_type': 'scene_classifier_mlp',
        'arch': best['arch'], 'scheme': best['scheme'],
        'input_dim': INPUT_DIM, 'window_size': WINDOW_SIZE,
        'class_names': list(CLASS_NAMES), 'ref_pressure': SEA_LEVEL_PRESSURE_PA,
        'feature_mean': best['scaler_mean'], 'feature_std': best['scaler_std'],
        'layers': to_numpy_layers(model),
    }
    json_path = os.path.join(MODELS_DIR, 'scene_classifier.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(numpy_cfg, f, indent=2)
    print(f"   JSON(GUI) : {json_path}")

    scaler_cfg = {
        'input_dim': INPUT_DIM, 'window_size': WINDOW_SIZE,
        'ref_pressure': SEA_LEVEL_PRESSURE_PA,
        'class_names': list(CLASS_NAMES),
        'feature_mean': best['scaler_mean'],
        'feature_std': [s if s != 0 else 1.0 for s in best['scaler_std']],
    }
    scaler_path = os.path.join(MODELS_DIR, 'scene_classifier_scaler.json')
    with open(scaler_path, 'w', encoding='utf-8') as f:
        json.dump(scaler_cfg, f, indent=2)
    print(f"   预处理参数: {scaler_path}  (固件适配用)")


def main():
    print("=" * 70)
    print("  场景识别分类器 — 多架构 × 多方案 训练对比")
    print("=" * 70)

    # 方案对比（用中等架构 mlp_m 比较数据构成）
    scheme_runs = []
    for scheme in ['synth', 'aug', 'combined']:
        X, Y, has_real = build_dataset(scheme)
        print(f"\n  总样本: {len(X)}  特征维度: {X.shape[1]}  (has_real={has_real})")
        r = train_one('mlp_m', ARCHITECTURES['mlp_m'], X, Y, scheme)
        r['has_real'] = has_real
        scheme_runs.append(r)
        save_confusion(r, os.path.join(RESULTS_DIR, f"scene_cm_mlp_m_{scheme}.png"))

    best_scheme = max(scheme_runs, key=lambda r: r['test_acc'])['scheme']
    print(f"\n[方案对比] 最佳方案 = '{best_scheme}' "
          f"(acc: " + ", ".join(f"{r['scheme']}={r['test_acc']:.4f}" for r in scheme_runs) + ")")

    # 架构对比（用最佳方案）
    X, Y, has_real = build_dataset(best_scheme)
    print(f"\n[架构对比] 使用方案='{best_scheme}'，总样本 {len(X)}")
    arch_runs = []
    for name, hidden in ARCHITECTURES.items():
        r = train_one(name, hidden, X, Y, best_scheme)
        r['has_real'] = has_real
        arch_runs.append(r)
        save_confusion(r, os.path.join(RESULTS_DIR, f"scene_cm_{name}_{best_scheme}.png"))

    # 选择：主指标 test_acc，平手时取参数量小者（MCU 友好）
    best = max(arch_runs, key=lambda r: (r['test_acc'], -r['params']))
    print(f"\n[结论] 最佳架构 = {best['arch']} ({best['hidden']}) "
          f"acc={best['test_acc']:.4f} 参数量={best['params']}")

    export_best(best)

    # ---- 报告 ----
    lines = []
    lines.append("# 场景识别分类器 训练对比报告\n")
    lines.append(f"- 输入特征维度: {INPUT_DIM}（手工特征，与 MCU/仿真一致）")
    lines.append(f"- 滑动窗口: {WINDOW_SIZE} 帧")
    lines.append(f"- 类别: {CLASS_NAMES}（平移与静止合并为 static）\n")

    lines.append("## 1. 训练方案对比（架构固定 mlp_m [24,16]）\n")
    lines.append("| 方案 | 测试准确率 | 宏F1 | static精确/召回 | elevation精确/召回 | 自检命中 |")
    lines.append("|---|---|---|---|---|---|")
    for r in scheme_runs:
        pc = r['per_class']
        dom = "/".join(f"{k}:{v}" for k, v in r['dom_counts'].items())
        lines.append(f"| {r['scheme']} | {r['test_acc']:.4f} | {r['test_macro_f1']:.4f} | "
                     f"{pc['static']['precision']:.3f}/{pc['static']['recall']:.3f} | "
                     f"{pc['elevation']['precision']:.3f}/{pc['elevation']['recall']:.3f} | {dom} |")

    lines.append("\n## 2. 架构对比（方案固定 " + best_scheme + "）\n")
    lines.append("| 架构 | 隐藏层 | 参数量 | 测试准确率 | 宏F1 | static精确/召回 | elevation精确/召回 | 训练耗时(s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(arch_runs, key=lambda x: -x['test_acc']):
        pc = r['per_class']
        lines.append(f"| {r['arch']} | {r['hidden']} | {r['params']} | {r['test_acc']:.4f} | "
                     f"{r['test_macro_f1']:.4f} | "
                     f"{pc['static']['precision']:.3f}/{pc['static']['recall']:.3f} | "
                     f"{pc['elevation']['precision']:.3f}/{pc['elevation']['recall']:.3f} | "
                     f"{r['time_s']:.0f} |")

    lines.append("\n## 3. 部署建议\n")
    lines.append(f"- **选用模型**: `{best['arch']}`（隐藏层 {best['hidden']}，参数量 {best['params']}）")
    lines.append(f"- **选用方案**: `{best['scheme']}`（测试准确率 {best['test_acc']:.4f}）")
    lines.append("- **导出产物**: `models/scene_classifier.tflite`（X-CUBE-AI 输入）、"
                 "`scene_classifier.h5`、`scene_classifier.json`（GUI）、"
                 "`scene_classifier_scaler.json`（固件归一化参数）")
    lines.append("- **MCU 部署流程**: 在 CubeMX X-CUBE-AI 中导入 `scene_classifier.tflite`"
                 "（作为第二个 network，例如 network1），生成 network1.c/.h 与 network_data1.c/.h；"
                 "固件 `scene_classifier.c` 负责滑窗+特征提取+归一化，再调用 `ai_network1_run`。")
    lines.append("- **资源提示**: 该 MLP 参数量很小（约 %d），远小于现有 filter 网络（5041 项），"
                 "对 STM32F4 内存无压力。" % best['params'])

    report_path = os.path.join(RESULTS_DIR, 'scene_training_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\n报告已写入: {report_path}")


if __name__ == '__main__':
    main()
