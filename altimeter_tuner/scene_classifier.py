#!/usr/bin/env python
"""
场景识别器（纯 numpy 实现，无需 TensorFlow 即可在 PC/GUI 运行）。

功能：在滤波的同时，基于最近一个时间窗内的双传感器气压与温度，
      推断当前处于哪种运动场景，用于辅助判断。

      模型为二分类（气压只反映高度变化，『平移』与『静止』气压信号一致，
      故二者合并为同一类）：
          - static    （静止 / 平移：高度恒定）
          - elevation （升降：高度随时间变化）

物理说明（重要）：
      气压计只反映高度变化，因此『静止』与『平移(水平运动)』在气压信号上完全
      不可区分（二者高度均为常量）。本分类器本质是一个『高度是否在变化』的
      检测器：输出 elevation 概率即可辅助判断是否处于升降状态。

模型：小型 MLP（与 X-CUBE-AI / TFLite 可部署架构一致），权重以 JSON 保存，
      推理用纯 numpy 前向传播，避免运行时依赖 tflite_runtime。

特征（窗口内，仅气压 + 温度，不引入其它数据）：
      0  ms5611 当前相对气压
      1  bmp280  当前相对气压
      2  当前温度
      3  ms5611 相邻差绝对值均值（波动）
      4  bmp280  相邻差绝对值均值（波动）
      5  ms5611 窗口斜率（高度变化率主特征）
      6  bmp280 窗口斜率
      7  ms5611 标准差
      8  bmp280 标准差
      9  温度斜率

参考气压固定为 101325 Pa（标准海平面），与训练、与单片机保持一致：
      仅『高度变化率』类特征(5/6)与绝对参考无关，固定参考可保证推理分布
      与训练一致。
"""

import os
import json
import numpy as np

SEA_LEVEL_PRESSURE_PA = 101325.0
WINDOW_SIZE = 15
# 二分类：静止/平移 合并为 static，升降为 elevation
CLASS_NAMES = ['static', 'elevation']
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'models', 'scene_classifier.json')


def extract_features(ms_rel, bmp_rel, temp):
    """从长度为 WINDOW_SIZE 的窗口序列提取 10 维特征。

    输入为相对气压（已减参考气压）与温度数组，长度需等于窗口大小。
    若长度不足，由调用方保证（不足时返回 None 由上层处理）。
    """
    W = len(ms_rel)
    f = np.empty(10, dtype=float)
    f[0] = ms_rel[-1]
    f[1] = bmp_rel[-1]
    f[2] = temp[-1]
    f[3] = float(np.mean(np.abs(np.diff(ms_rel)))) if W > 1 else 0.0
    f[4] = float(np.mean(np.abs(np.diff(bmp_rel)))) if W > 1 else 0.0
    f[5] = float((ms_rel[-1] - ms_rel[0]) / (W - 1)) if W > 1 else 0.0
    f[6] = float((bmp_rel[-1] - bmp_rel[0]) / (W - 1)) if W > 1 else 0.0
    f[7] = float(np.std(ms_rel))
    f[8] = float(np.std(bmp_rel))
    f[9] = float((temp[-1] - temp[0]) / (W - 1)) if W > 1 else 0.0
    return f


class SceneClassifier:
    """基于滑动窗口的场景分类器（纯 numpy 前向）。"""

    def __init__(self, model_path=None, ref_pressure=None):
        self.model_path = model_path or DEFAULT_MODEL
        self.ref_pressure = float(ref_pressure if ref_pressure is not None
                                  else SEA_LEVEL_PRESSURE_PA)
        self.window = int(WINDOW_SIZE)
        self.classes = list(CLASS_NAMES)
        self._load(self.model_path)
        # 环形缓冲
        self._ms = []
        self._bmp = []
        self._temp = []

    @classmethod
    def from_config(cls, cfg):
        """从配置 dict 构造（不读写文件），便于训练脚本内部自测。"""
        inst = cls.__new__(cls)
        inst.model_path = None
        inst.ref_pressure = float(cfg.get('ref_pressure', SEA_LEVEL_PRESSURE_PA))
        inst.window = int(cfg.get('window_size', WINDOW_SIZE))
        inst.classes = list(cfg.get('class_names', CLASS_NAMES))
        inst.input_dim = int(cfg.get('input_dim', len(cfg['feature_mean'])))
        inst._load_cfg(cfg)
        inst._ms = []
        inst._bmp = []
        inst._temp = []
        return inst

    def _load_cfg(self, cfg):
        self.window = int(cfg.get('window_size', self.window))
        self.classes = list(cfg.get('class_names', CLASS_NAMES))
        self.ref_pressure = float(cfg.get('ref_pressure', self.ref_pressure))
        self.feature_mean = np.asarray(cfg['feature_mean'], dtype=float)
        self.feature_std = np.asarray(cfg['feature_std'], dtype=float)
        self.feature_std[self.feature_std == 0] = 1.0
        # 层：顺序 [(W, b, act), ...]
        self.layers = []
        for ly in cfg['layers']:
            W = np.asarray(ly['W'], dtype=float)
            b = np.asarray(ly['b'], dtype=float)
            self.layers.append((W, b, ly.get('act', 'relu')))
        self.input_dim = int(cfg.get('input_dim', self.feature_mean.shape[0]))

    def _load(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"场景分类模型不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self._load_cfg(cfg)

    def reset(self):
        self._ms = []
        self._bmp = []
        self._temp = []

    def _forward(self, x):
        a = (x - self.feature_mean) / self.feature_std
        for W, b, act in self.layers:
            a = a @ W + b
            if act == 'relu':
                a = np.maximum(0.0, a)
            # 最后一层 softmax 在下面统一处理
        # softmax（数值稳定）
        a = a - np.max(a)
        e = np.exp(a)
        return e / e.sum()

    def update(self, ms_pressure, bmp_pressure, temperature):
        """喂入一个时刻的原始气压(Pa)与温度(℃)，返回当前各场景概率(np.array)。

        窗口未填满时返回均匀分布（表示不确定）。
        """
        ms_rel = float(ms_pressure) - self.ref_pressure
        bmp_rel = float(bmp_pressure) - self.ref_pressure
        t = float(temperature)
        self._ms.append(ms_rel)
        self._bmp.append(bmp_rel)
        self._temp.append(t)
        if len(self._ms) > self.window:
            self._ms.pop(0)
            self._bmp.pop(0)
            self._temp.pop(0)
        if len(self._ms) < self.window:
            return np.full(len(self.classes), 1.0 / len(self.classes), dtype=float)
        f = extract_features(np.asarray(self._ms), np.asarray(self._bmp),
                             np.asarray(self._temp))
        return self._forward(f)

    def predict_sequence(self, ms_p, bmp_p, temp):
        """对整段序列推理，返回 (scene_prob[n,3], scene_pred[n])。"""
        self.reset()
        ms_p = np.asarray(ms_p, dtype=float)
        bmp_p = np.asarray(bmp_p, dtype=float)
        temp = np.asarray(temp, dtype=float)
        n = len(ms_p)
        prob = np.zeros((n, len(self.classes)), dtype=float)
        pred = np.zeros(n, dtype=int)
        for i in range(n):
            p = self.update(ms_p[i], bmp_p[i], temp[i])
            prob[i] = p
            pred[i] = int(np.argmax(p))
        return prob, pred


def build_default_model_json(out_path, input_dim=10, hidden=(24, 16),
                             window_size=WINDOW_SIZE):
    """生成一个『未训练』的占位模型 JSON（权重全零 + softmax 输出）。

    仅在缺少训练模型时保证程序不崩溃；实际应使用训练脚本生成真实权重。
    """
    rng = np.random.default_rng(0)
    layers = []
    dims = [input_dim] + list(hidden) + [len(CLASS_NAMES)]
    for i in range(len(dims) - 1):
        scale = np.sqrt(2.0 / dims[i])
        W = rng.normal(0, scale, size=(dims[i], dims[i + 1]))
        b = np.zeros(dims[i + 1])
        act = 'softmax' if i == len(dims) - 2 else 'relu'
        layers.append({'W': W.tolist(), 'b': b.tolist(), 'act': act})
    cfg = {
        'version': 1,
        'model_type': 'scene_classifier_mlp',
        'input_dim': input_dim,
        'window_size': window_size,
        'class_names': list(CLASS_NAMES),
        'ref_pressure': SEA_LEVEL_PRESSURE_PA,
        'feature_mean': [0.0] * input_dim,
        'feature_std': [1.0] * input_dim,
        'layers': layers,
        'note': 'placeholder (untrained) —— 请用 train_scene_classifier.py 训练',
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    return out_path


if __name__ == '__main__':
    # 自测：用合成数据验证推理通路（不依赖训练好的权重）
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets import generate_synthetic
    m = build_default_model_json(os.path.join('models', 'scene_classifier_untrained.json'))
    clf = SceneClassifier(model_path=m)
    for sc in CLASS_NAMES:
        ds = generate_synthetic(scenario=sc, n_samples=200, fs=10.0, seed=1)
        prob, pred = clf.predict_sequence(ds['ms_pressure'], ds['bmp_pressure'], ds['temperature'])
        dom = clf.classes[int(np.argmax(prob.mean(axis=0)))]
        print(f"{sc:12s} -> 主导预测: {dom}  平均概率: {np.round(prob.mean(axis=0),3)}")
