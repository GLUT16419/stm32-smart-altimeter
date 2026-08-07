#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多任务模型通用推理（PC 端验证 / 可视化用）
=========================================
统一接口运行 models/compare_v2/ 下任意架构的 .tflite（滤波 + 场景分类）。
用法：
    from multitask_infer import MultitaskModel
    m = MultitaskModel('models/compare_v2/4.deep_mlp')
    filt_ms, filt_bmp, (p_static, p_elev) = m.predict(ms_rel_win10, bmp_rel_win10)

其中 *_rel_win10 为相对气压窗口(已减海平面参考 101325)，长度 WINDOW=10。
"""
import os
import json
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except Exception:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter


class MultitaskModel:
    def __init__(self, model_dir_or_prefix):
        """model_dir_or_prefix: 目录，或不含后缀的 .tflite/.json 前缀。
        会自动在同一目录寻找 <name>.tflite 与 <name>.json。"""
        if model_dir_or_prefix.endswith('.tflite') or model_dir_or_prefix.endswith('.json'):
            prefix = model_dir_or_prefix[:-len('.tflite')] \
                if model_dir_or_prefix.endswith('.tflite') else model_dir_or_prefix[:-len('.json')]
        else:
            prefix = os.path.join(model_dir_or_prefix,
                                  os.path.basename(model_dir_or_prefix))
        # 若传入的是目录，尝试匹配其中的 tflite
        if os.path.isdir(model_dir_or_prefix):
            tl = [f for f in os.listdir(model_dir_or_prefix) if f.endswith('.tflite')]
            if not tl:
                raise FileNotFoundError('目录中未找到 .tflite: ' + model_dir_or_prefix)
            prefix = os.path.join(model_dir_or_prefix, tl[0][:-len('.tflite')])

        self.tflite_path = prefix + '.tflite'
        self.json_path = prefix + '.json'
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.prep = self.cfg['prep']
        self.window = int(self.cfg['window'])
        self.class_names = self.cfg['class_names']
        self.t_mean = float(self.cfg['target_mean'])
        self.t_std = float(self.cfg['target_std'])
        self.fmean = np.array(self.cfg['feature_mean'], dtype=float)
        self.fstd = np.array(self.cfg['feature_std'], dtype=float)

        self.interp = TFLiteInterpreter(model_path=self.tflite_path)
        self.interp.allocate_tensors()
        self.in_details = self.interp.get_input_details()
        self.out_details = self.interp.get_output_details()
        self.n_inputs = len(self.in_details)

    def _prep(self, ms_rel, bmp_rel):
        ms = np.asarray(ms_rel, dtype=float).reshape(-1)
        bmp = np.asarray(bmp_rel, dtype=float).reshape(-1)
        if len(ms) != self.window or len(bmp) != self.window:
            raise ValueError(f'窗口长度需为 {self.window}')
        X = np.concatenate([ms, bmp])
        Xs = (X - self.fmean) / self.fstd          # 始终用缩放后的特征
        if self.prep == 'raw':
            return Xs.reshape(1, -1).astype(np.float32)
        if self.prep == 'feature_eng':
            a = Xs[:self.window]; b = Xs[self.window:]
            def f(w):
                return np.stack([w.mean(), w.std(), w[-1], w[-1] - w[0]])
            fe = np.concatenate([f(a), f(b)])
            return fe.reshape(1, -1).astype(np.float32)
        if self.prep == 'twotower':
            return [Xs[:self.window].reshape(1, -1).astype(np.float32),
                    Xs[self.window:].reshape(1, -1).astype(np.float32)]
        if self.prep == 'cnn':
            arr = np.stack([Xs[:self.window], Xs[self.window:]], axis=-1)  # (10,2)
            return arr.reshape(1, self.window, 2).astype(np.float32)
        raise ValueError('未知 prep: ' + self.prep)

    def predict(self, ms_rel, bmp_rel):
        x = self._prep(ms_rel, bmp_rel)
        if self.n_inputs == 2:
            self.interp.set_tensor(self.in_details[0]['index'], x[0])
            self.interp.set_tensor(self.in_details[1]['index'], x[1])
        else:
            self.interp.set_tensor(self.in_details[0]['index'], x)
        self.interp.invoke()
        reg_ms = self.interp.get_tensor(self.out_details[0]['index'])
        reg_bmp = self.interp.get_tensor(self.out_details[1]['index'])
        cls = self.interp.get_tensor(self.out_details[2]['index'])
        filt_ms = float(reg_ms.flatten()[0]) * self.t_std + self.t_mean
        filt_bmp = float(reg_bmp.flatten()[0]) * self.t_std + self.t_mean
        probs = cls.flatten().astype(float)
        return filt_ms, filt_bmp, tuple(probs)


def _demo():
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, 'models', 'compare_v2')
    cands = sorted(glob.glob(os.path.join(d, '*.tflite')))
    if not cands:
        print('未找到模型，请先运行 train_multitask_compare.py')
        return
    # 选一个 MLP 架构做演示
    pick = next((c for c in cands if 'deep_mlp' in c or 'baseline_mlp' in c), cands[0])
    m = MultitaskModel(pick[:-len('.tflite')])
    ms_win = np.random.normal(-2755, 3, m.window)
    bmp_win = np.random.normal(-2755, 0.3, m.window)
    fm, fb, p = m.predict(ms_win, bmp_win)
    print(f"模型: {os.path.basename(pick)}")
    print(f"  滤波 MS5611={fm:.2f}Pa  BMP280={fb:.2f}Pa")
    print(f"  场景概率: {dict(zip(m.class_names, [round(v,3) for v in p]))}")


if __name__ == '__main__':
    _demo()
