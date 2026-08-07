#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多任务高度计模型训练（PC 端，仅产出模型文件，不改动单片机代码）
===============================================================
把现有的两个独立网络（滤波网络 network.* 与场景分类器 scene_classifier）
合并为一个**共享骨干 + 三头**的小型 MLP：

    输入 X : [MS5611 窗口(10), BMP280 窗口(10)] 共 20 维（相对气压，已减 101325）
    共享  : Dense(32, relu) -> Dense(16, relu)
    头1   : reg_ms   -> 1  （MS5611 滤波后相对气压，单位 Pa）
    头2   : reg_bmp  -> 1  （BMP280 滤波后相对气压，单位 Pa）
    头3   : cls      -> 2  （[static, elevation] softmax）

真值定义：
    * 滤波目标 = 该传感器「无噪气压」= base_p (+ 该路系统性偏置)。
      与现有自监督滤波网络语义一致：只去噪、保留各路自身偏置，
      偏置补偿留给上层融合/KF，而不是让网络去消除。
    * 场景标签 = 整段是否处于升降：static/translation -> 0，elevation -> 1。

产出（仅写入 altimeter_tuner/models/，不触碰单片机）：
    - multitask_filter_scene.tflite   （可选：X-CUBE-AI 输入）
    - multitask_filter_scene.json     （纯 numpy 推理配置）
    - multitask_scaler.json           （归一化参数）
    - 控制台打印评估（滤波 MAE + 场景 acc/F1/混淆矩阵）
"""

import os
import sys
import json
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from datasets import _sensor_truth, SEA_LEVEL_PRESSURE_PA, load_dataset  # noqa: E402
from algorithm import altitude_to_pressure                 # noqa: E402
import tensorflow as tf                                     # noqa: E402
from tensorflow.keras.models import Model                  # noqa: E402
from tensorflow.keras.layers import Input, Dense           # noqa: E402
from tensorflow.keras.optimizers import Adam               # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping       # noqa: E402
from sklearn.model_selection import train_test_split       # noqa: E402
from sklearn.preprocessing import StandardScaler           # noqa: E402
from sklearn.metrics import (classification_report,        # noqa: E402
                             confusion_matrix, f1_score,
                             accuracy_score)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                             # noqa: E402

MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------- 配置 ----------------
WINDOW = 10                       # 与现有滤波网络窗口一致
INPUT_DIM = 2 * WINDOW            # 20
HIDDEN1 = 32
HIDDEN2 = 16
N_CLASSES = 2
CLASS_NAMES = ['static', 'elevation']
SCENARIO_TO_CLASS = {'static': 0, 'translation': 0, 'elevation': 1}
SCENARIOS = ['static', 'translation', 'elevation']
SEA = float(SEA_LEVEL_PRESSURE_PA)
# 回归/分类损失权重。真实数据回归目标是绝对气压(约 -2736Pa 量级)，
# MSE 数值本身已含该绝对量级，故回归权重需与分类相当(=1.0)，否则回归头
# 梯度过小、学不会绝对偏移，会在窄分布上外推崩坏(测试 MAE 达数百 Pa)。
W_REG = 1.0
W_CLS = 1.0


# ============================================================
# 数据生成（内联，复用 algorithm 的逆换算，直接产出无噪真值）
# ============================================================
def gen_pair(scenario, n_samples=300, fs=10.0, seed=0,
             ms_noise_std=3.05, bmp_noise_std=0.35,
             temp_mean=25.0, temp_noise_std=0.2,
             temp_drift=False, bias_pa=0.0):
    """返回 (ms_p, bmp_p, temp, ms_clean, bmp_clean) 全为绝对气压(Pa)。

    ms_clean = base_p（无噪），bmp_clean = base_p + bias（无噪，含自身系统偏置）。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    _, true = _sensor_truth(n_samples, fs, scenario, seed)
    if temp_drift:
        temp_ref = temp_mean + 2.0 * np.sin(2 * np.pi * t / 60.0)
    else:
        temp_ref = np.full(n_samples, temp_mean)
    base_p = np.array([altitude_to_pressure(true[i], temp_ref[i], SEA)
                       for i in range(n_samples)], dtype=float)
    bmp_p = base_p + bias_pa + rng.normal(0, bmp_noise_std, n_samples)
    ms_p = base_p + rng.normal(0, ms_noise_std, n_samples)
    temp = temp_mean + rng.normal(0, temp_noise_std, n_samples)
    if temp_drift:
        temp = temp + 2.0 * np.sin(2 * np.pi * t / 60.0)
    ms_clean = base_p.copy()
    bmp_clean = base_p + bias_pa
    return ms_p, bmp_p, temp, ms_clean, bmp_clean


def make_windows(ms_p, bmp_p, ms_clean, bmp_clean, class_idx):
    """滑窗构造 (X[20], y_ms, y_bmp, y_cls)。"""
    n = len(ms_p)
    if n < WINDOW:
        return (np.empty((0, INPUT_DIM)), np.empty(0), np.empty(0),
                np.empty((0, N_CLASSES)))
    ms_rel = ms_p - SEA
    bmp_rel = bmp_p - SEA
    ms_c_rel = ms_clean - SEA
    bmp_c_rel = bmp_clean - SEA
    X, Yms, Ybmp, Yc = [], [], [], []
    for i in range(WINDOW - 1, n):
        x = np.concatenate([ms_rel[i - WINDOW + 1:i + 1],
                            bmp_rel[i - WINDOW + 1:i + 1]])
        X.append(x)
        Yms.append(ms_c_rel[i])
        Ybmp.append(bmp_c_rel[i])
        y = np.zeros(N_CLASSES, dtype=float)
        y[class_idx] = 1.0
        Yc.append(y)
    return (np.array(X, dtype=float), np.array(Yms, dtype=float),
            np.array(Ybmp, dtype=float), np.array(Yc, dtype=float))


def _build_synthetic():
    """原合成数据构造（域随机化），返回 (X, Yms, Ybm, Yc)。"""
    ms_noises = [3.05, 10.0]
    bmp_noises = [0.35, 2.0]
    biases = [-30.0, 0.0, 30.0]
    fs_list = [10.0]
    seeds = range(3)
    Xs, Yms, Ybm, Ycs = [], [], [], []
    for sc in SCENARIOS:
        cls = SCENARIO_TO_CLASS[sc]
        for seed in seeds:
            for fs in fs_list:
                for mn in ms_noises:
                    for bn in bmp_noises:
                        for bb in biases:
                            ds = gen_pair(sc, n_samples=300, fs=fs, seed=seed,
                                          ms_noise_std=mn, bmp_noise_std=bn,
                                          bias_pa=bb)
                            x, yms, ybmp, yc = make_windows(ds[0], ds[1], ds[3], ds[4], cls)
                            if len(x):
                                Xs.append(x); Yms.append(yms)
                                Ybm.append(ybmp); Ycs.append(yc)
    X = np.concatenate(Xs, axis=0)
    Yms = np.concatenate(Yms, axis=0)
    Ybm = np.concatenate(Ybm, axis=0)
    Yc = np.concatenate(Ycs, axis=0)
    return X, Yms, Ybm, Yc


def make_windows_real(ms_p, bmp_p, ms_kf, bmp_kf, class_idx):
    """真实数据滑窗：X=[ms_rel(10),bmp_rel(10)]，回归目标=各路 kf_pressure(去噪伪真值)。"""
    n = len(ms_p)
    if n < WINDOW:
        return (np.empty((0, INPUT_DIM)), np.empty(0), np.empty(0),
                np.empty((0, N_CLASSES)))
    ms_rel = ms_p - SEA
    bmp_rel = bmp_p - SEA
    ms_k_rel = ms_kf - SEA
    bmp_k_rel = bmp_kf - SEA
    X, Yms, Ybm, Yc = [], [], [], []
    for i in range(WINDOW - 1, n):
        x = np.concatenate([ms_rel[i - WINDOW + 1:i + 1],
                            bmp_rel[i - WINDOW + 1:i + 1]])
        X.append(x)
        Yms.append(ms_k_rel[i])
        Ybm.append(bmp_k_rel[i])
        y = np.zeros(N_CLASSES, dtype=float)
        y[class_idx] = 1.0
        Yc.append(y)
    return (np.array(X, dtype=float), np.array(Yms, dtype=float),
            np.array(Ybm, dtype=float), np.array(Yc, dtype=float))


REAL_RAW_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'serial_tool', 'data', 'raw'))
REAL_MAP = {'静止': 'static', '平移运动': 'translation', '升降运动': 'elevation'}


def collect_real_windows():
    """加载 serial_tool/data/raw 下三场景真实双文件，构造多任务窗口。

    回归目标用固件已算好的 kf_pressure_pa（自监督去噪伪真值）：网络学习
    像 KF 一样对各路传感器去噪，同时保留各路自身系统偏置（与合成标签语义一致）。
    """
    Xs, Yms, Ybm, Ycs = [], [], [], []
    found = False
    if not os.path.isdir(REAL_RAW_DIR):
        print(f"  (真实数据目录不存在，跳过: {REAL_RAW_DIR})")
        return Xs, Yms, Ybm, Ycs, found
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
            ms_kf = ds.get('ms_kf_p')
            bmp_kf = ds.get('bmp_kf_p')
            if ms_kf is None or bmp_kf is None:
                print(f"    ! {cn}/{os.path.basename(ms_p)} 缺少 kf_pressure_pa，跳过")
                continue
            x, yms, ybmp, yc = make_windows_real(ds['ms_pressure'], ds['bmp_pressure'],
                                                 ms_kf, bmp_kf, cls)
            if len(x):
                Xs.append(x); Yms.append(yms); Ybm.append(ybmp); Ycs.append(yc)
                cnt += len(x)
        if cnt:
            found = True
        print(f"  真实[{cn}->{key}->class{cls}]: {cnt} 窗口")
    return Xs, Yms, Ybm, Ycs, found


def _concat(Xs, Yms, Ybm, Ycs):
    return (np.concatenate(Xs, axis=0), np.concatenate(Yms, axis=0),
            np.concatenate(Ybm, axis=0), np.concatenate(Ycs, axis=0))


def build_dataset(mode='synth'):
    if mode == 'synth':
        print("[数据] 构造多任务数据集（仅合成数据）")
        return _build_synthetic()
    if mode == 'real':
        print("[数据] 构造多任务数据集（仅真实采集数据，kf_pressure 作伪真值）")
        Xs, Yms, Ybm, Ycs, found = collect_real_windows()
        if not found:
            print("  ! 未找到真实数据，回退到合成数据")
            return _build_synthetic()
        return _concat(Xs, Yms, Ybm, Ycs)
    if mode == 'combined':
        print("[数据] 构造多任务数据集（合成 + 真实；合成按真实基准气压平移对齐分布）")
        Xs, Yms, Ybm, Ycs = _build_synthetic()
        Xr, Ymrs, Ybmr, Ycsr, found = collect_real_windows()
        if not found:
            print("  ! 未找到真实数据，仅用合成")
            return Xs, Yms, Ybm, Ycs
        # 合成数据「地面高度」rel≈0，真实数据 rel≈-2750。把合成整体平移到真实
        # 基准气压附近，使二者落在同一绝对气压区间（避免 StandardScaler 学到双峰）。
        Xr = np.concatenate(Xr, axis=0)
        Ymrs = np.concatenate(Ymrs, axis=0)
        Ybmr = np.concatenate(Ybmr, axis=0)
        Ycsr = np.concatenate(Ycsr, axis=0)
        real_med = np.median(np.concatenate([Xr[:, :WINDOW].ravel(),
                                             Xr[:, WINDOW:].ravel()]))
        synth_med = np.median(np.concatenate([Xs[:, :WINDOW].ravel(),
                                              Xs[:, WINDOW:].ravel()]))
        offset = real_med - synth_med
        Xs = Xs + offset
        Yms = Yms + offset
        Ybm = Ybm + offset
        X = np.concatenate([Xs, Xr], axis=0)
        Ym = np.concatenate([Yms, Ymrs], axis=0)
        Yb = np.concatenate([Ybm, Ybmr], axis=0)
        Yc = np.concatenate([Ycs, Ycsr], axis=0)
        return X, Ym, Yb, Yc
    raise ValueError(mode)


# ============================================================
# 模型
# ============================================================
def build_model():
    inp = Input(shape=(INPUT_DIM,))
    x = Dense(HIDDEN1, activation='relu')(inp)
    x = Dense(HIDDEN2, activation='relu')(x)
    reg_ms = Dense(1, name='reg_ms')(x)
    reg_bmp = Dense(1, name='reg_bmp')(x)
    cls = Dense(N_CLASSES, activation='softmax', name='cls')(x)
    model = Model(inp, [reg_ms, reg_bmp, cls])
    model.compile(
        optimizer=Adam(1e-3),
        loss={'reg_ms': 'mse', 'reg_bmp': 'mse',
              'cls': 'categorical_crossentropy'},
        loss_weights={'reg_ms': W_REG, 'reg_bmp': W_REG, 'cls': W_CLS},
        metrics={'cls': 'accuracy'})
    return model


def count_params(model):
    return int(model.count_params())


# ============================================================
# 主流程
# ============================================================
def main(mode='real'):
    print("=" * 70)
    print("  多任务高度计模型 — 同时滤波 + 判断场景（仅 PC 训练）")
    print(f"  数据模式: {mode}")
    print("=" * 70)

    X, Yms, Ybm, Yc = build_dataset(mode)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 回归目标归一化：真实气压绝对量级(~-2755Pa)，而 reg 头 Dense(1) 偏置初始化为 0。
    # Adam(lr=1e-3) 对持续梯度只能以 ~lr/step 移动参数，有限步内偏置无法从 0 走到
    # -2755，于是网络退化为用大权重放大输入噪声→外推崩坏(测试 MAE 达上千 Pa)。
    # 对回归目标做 StandardScaler，使其~N(0,1)，与 bias=0 初值、Adam 动力学匹配，
    # 收敛稳定。多任务模型仅 PC 端用，导出时存 target_mean/std 反归一化即可。
    tgt = np.concatenate([Yms, Ybm], axis=0).reshape(-1, 1)
    tscaler = StandardScaler().fit(tgt)
    t_mean = float(tscaler.mean_[0])
    t_std = float(tscaler.scale_[0]) if tscaler.scale_[0] != 0 else 1.0
    Yms_n = (Yms - t_mean) / t_std
    Ybm_n = (Ybm - t_mean) / t_std

    (X_tr, X_te, Yms_tr, Yms_te, Ybm_tr, Ybm_te,
     Yc_tr, Yc_te) = train_test_split(
        Xs, Yms_n, Ybm_n, Yc, test_size=0.2, random_state=42,
        stratify=Yc.argmax(axis=1))
    X_tr, X_val, Yms_tr, Yms_val, Ybm_tr, Ybm_val, Yc_tr, Yc_val = \
        train_test_split(X_tr, Yms_tr, Ybm_tr, Yc_tr, test_size=0.15,
                         random_state=42, stratify=Yc_tr.argmax(axis=1))

    model = build_model()
    model.summary()
    es = EarlyStopping(monitor='val_cls_accuracy', mode='max', patience=20,
                       restore_best_weights=True, verbose=0)
    t0 = time.time()
    model.fit(X_tr, {'reg_ms': Yms_tr, 'reg_bmp': Ybm_tr, 'cls': Yc_tr},
              validation_data=(X_val, {'reg_ms': Yms_val, 'reg_bmp': Ybm_val,
                                       'cls': Yc_val}),
              epochs=150, batch_size=256, callbacks=[es], verbose=1)
    dt = time.time() - t0

    # ---- 评估（预测值反归一化为 Pa）----
    pred = model.predict(X_te, verbose=0)
    p_ms = pred[0].flatten() * t_std + t_mean
    p_bmp = pred[1].flatten() * t_std + t_mean
    Yms_te_pa = Yms_te * t_std + t_mean
    Ybm_te_pa = Ybm_te * t_std + t_mean
    ms_mae = float(np.mean(np.abs(p_ms - Yms_te_pa)))
    bmp_mae = float(np.mean(np.abs(p_bmp - Ybm_te_pa)))
    y_true = Yc_te.argmax(axis=1)
    y_pred = pred[2].argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[评估] 滤波 MAE: MS5611={ms_mae:.3f}Pa  BMP280={bmp_mae:.3f}Pa")
    print(f"[评估] 场景 acc={acc:.4f}  macroF1={f1:.4f}")
    print(f"[评估] 混淆矩阵(行=真/列=预测):\n{cm}")
    print(f"[训练] 耗时 {dt:.0f}s  参数量 {count_params(model)}")

    # ---- 导出 tflite ----
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite = converter.convert()
        tp = os.path.join(MODELS_DIR, 'multitask_filter_scene.tflite')
        with open(tp, 'wb') as f:
            f.write(tflite)
        print(f"   TFLite   : {tp}")
    except Exception as e:
        print(f"   ! TFLite 导出失败: {e}")

    # ---- 导出 json（纯 numpy 推理配置）----
    W1, b1 = model.layers[1].get_weights()   # Dense(HIDDEN1)
    W2, b2 = model.layers[2].get_weights()   # Dense(HIDDEN2)
    Wms, bms = model.get_layer('reg_ms').get_weights()
    Wbmp, bbmp = model.get_layer('reg_bmp').get_weights()
    Wcls, bcls = model.get_layer('cls').get_weights()
    cfg = {
        'version': 1, 'model_type': 'multitask_mlp',
        'input_dim': INPUT_DIM, 'window': WINDOW,
        'hidden1': HIDDEN1, 'hidden2': HIDDEN2,
        'class_names': list(CLASS_NAMES), 'ref_pressure': SEA,
        'target_mean': t_mean, 'target_std': t_std,
        'feature_mean': scaler.mean_.tolist(),
        'feature_std': [s if s != 0 else 1.0 for s in scaler.scale_.tolist()],
        'W1': W1.tolist(), 'b1': b1.tolist(),
        'W2': W2.tolist(), 'b2': b2.tolist(),
        'Wms': Wms.tolist(), 'bms': bms.tolist(),
        'Wbmp': Wbmp.tolist(), 'bbmp': bbmp.tolist(),
        'Wcls': Wcls.tolist(), 'bcls': bcls.tolist(),
    }
    jp = os.path.join(MODELS_DIR, 'multitask_filter_scene.json')
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    print(f"   JSON     : {jp}")

    sp = os.path.join(MODELS_DIR, 'multitask_scaler.json')
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump({'input_dim': INPUT_DIM, 'window': WINDOW,
                   'ref_pressure': SEA, 'class_names': list(CLASS_NAMES),
                   'target_mean': t_mean, 'target_std': t_std,
                   'feature_mean': scaler.mean_.tolist(),
                   'feature_std': [s if s != 0 else 1.0 for s in scaler.scale_.tolist()]},
                  f, indent=2)
    print(f"   归一化   : {sp}")

    # ---- 混淆矩阵图 ----
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('预测'); ax.set_ylabel('真实')
    ax.set_title(f"多任务-场景混淆 matrix\nacc={acc:.3f} f1={f1:.3f}")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='red' if i != j else 'black', fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(MODELS_DIR, 'multitask_cm.png'), dpi=150)
    plt.close(fig)
    print(f"   混淆图   : {os.path.join(MODELS_DIR, 'multitask_cm.png')}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['synth', 'real', 'combined'], default='real',
                    help='数据来源: synth=仅合成, real=仅真实采集, combined=合成+真实对齐')
    args = ap.parse_args()
    main(args.mode)
