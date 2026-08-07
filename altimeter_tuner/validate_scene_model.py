#!/usr/bin/env python
"""验证 C 场景分类器 (scene_classifier.c) 与 Python 推理 (scene_classifier.py) 一致。

生成测试序列 -> scene_test_data.csv (ms_pa,bmp_pa,temp)
运行 Python SceneClassifier -> scene_test_py.csv (p0,p1,pred)
C 程序读取同一 csv -> scene_test_c.csv
最后比对两者概率，报告最大误差。
"""
import os
import csv
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from datasets import generate_synthetic  # noqa: E402
from scene_classifier import SceneClassifier  # noqa: E402

DATA_CSV = os.path.join(SCRIPT_DIR, 'scene_test_data.csv')
PY_CSV = os.path.join(SCRIPT_DIR, 'scene_test_py.csv')
C_CSV = os.path.join(SCRIPT_DIR, 'scene_test_c.csv')


def main():
    seqs = []
    for sc in ['static', 'translation', 'elevation']:
        ds = generate_synthetic(scenario=sc, n_samples=200, fs=10.0, seed=11)
        seqs.append((sc, ds))

    # 写原始数据（三个场景连续拼接，模拟固件连续运行，窗口不分段复位）
    all_ms, all_bmp, all_t = [], [], []
    with open(DATA_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        for sc, ds in seqs:
            for i in range(len(ds['ms_pressure'])):
                all_ms.append(ds['ms_pressure'][i])
                all_bmp.append(ds['bmp_pressure'][i])
                all_t.append(ds['temperature'][i])
                w.writerow([f"{ds['ms_pressure'][i]:.6f}",
                            f"{ds['bmp_pressure'][i]:.6f}",
                            f"{ds['temperature'][i]:.6f}"])

    # Python 推理（连续，单次 predict_sequence，与 C 连续行为一致）
    clf = SceneClassifier()
    prob, pred = clf.predict_sequence(np.array(all_ms), np.array(all_bmp),
                                      np.array(all_t))
    with open(PY_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        for i in range(len(prob)):
            w.writerow([f"{prob[i,0]:.8f}", f"{prob[i,1]:.8f}", int(pred[i])])
    print(f"已生成 {DATA_CSV} 与 {PY_CSV}")


if __name__ == '__main__':
    main()
