# -*- coding: utf-8 -*-
"""
数据集生成与加载
================
1) 合成数据 (开箱即用)：按真实传感器噪声特性生成
   - 静止 (static)    : 高度恒定，仅叠加噪声
   - 平移 (translation): 水平运动，高度恒定（验证滤波不引入虚假高度）
   - 升降 (elevation) : 上升 + 下降 + 台阶，验证跟踪与回零
   每个场景同时产出 MS5611(σ≈3.05Pa) 与 BMP280(σ≈0.35Pa) 两路气压。

2) CSV 加载：
   - 合并单文件：列含 t_ms / ms5611* / bmp280* / temp* / true_height* 等
   - 双文件 (真实采集格式)：ms5611_*.csv 与 bmp280_*.csv，按行号对齐合并
"""

import csv
import os
import numpy as np

from algorithm import pressure_to_altitude_with_temp, altitude_to_pressure

SEA_LEVEL_PRESSURE_PA = 101325.0

# 合成数据的『参考段』样本数（与算法校准 calib_samples 默认 50 对齐），
# 用于由数据估计参考气压，使真值高度与算法输出使用同一参考气压。
SYNTH_CALIB_SAMPLES = 50

SCENARIOS = ['static', 'translation', 'elevation']


def _sensor_truth(n_samples, fs, scenario, seed, elevation_range=(0, 3, 6, 9)):
    """根据场景返回 (time_s, true_height_m, base_pressure) 数组。"""
    dt = 1.0 / fs
    t = np.arange(n_samples) * dt
    true = np.zeros(n_samples)
    if scenario == 'static':
        true[:] = 0.0
    elif scenario == 'translation':
        # 纯水平运动，高度恒为 0（用于检测是否产生虚假高度）
        true[:] = 0.0
    elif scenario == 'elevation':
        # 0~6s 静止 0m
        # 6~12s 缓升到 3m
        # 12~16s 静止 3m
        # 16~20s 急升到 6m
        # 20~26s 静止 6m
        # 26~32s 缓降到 0m
        # 32~40s 静止 0m
        sec = t
        for (a, b, h0, h1) in [(6, 12, 0, 3), (16, 20, 3, 6), (26, 32, 6, 0)]:
            m = (sec >= a) & (sec < b)
            frac = (sec[m] - a) / (b - a)
            true[m] = np.linspace(h0, h1, m.sum()) if m.sum() > 0 else true[m]
        # 静止段保持
        true[(sec >= 0) & (sec < 6)] = 0.0
        true[(sec >= 12) & (sec < 16)] = 3.0
        true[(sec >= 20) & (sec < 26)] = 6.0
        true[(sec >= 32)] = 0.0
    else:
        raise ValueError(scenario)
    return t, true


def generate_synthetic(scenario='static', n_samples=400, fs=10.0,
                       seed=42, ms_noise_std=3.05, bmp_noise_std=0.35,
                       temp_mean=25.0, temp_noise_std=0.2,
                       temp_drift=False, bias_pa=0.0):
    """
    生成合成数据集。

    返回 dict:
        time, ms_pressure, bmp_pressure, temperature, true_height,
        scenario
    """
    rng = np.random.default_rng(seed)
    t, true = _sensor_truth(n_samples, fs, scenario, seed)
    temp_ref = temp_mean + (2.0 * np.sin(2 * np.pi * t / 60.0) if temp_drift else np.zeros(n_samples))
    # 用单片机的『高度->气压』逆公式生成真实气压轨迹（不再用线性 12 Pa/m 近似）
    base_p = np.array([altitude_to_pressure(true[i], temp_ref[i], SEA_LEVEL_PRESSURE_PA)
                       for i in range(n_samples)])
    # 给 BMP280 施加一个固定的系统性偏置 (模拟两传感器基准不一致)
    bmp_p = base_p + bias_pa + rng.normal(0, bmp_noise_std, n_samples)
    ms_p = base_p + rng.normal(0, ms_noise_std, n_samples)
    temp = temp_mean + rng.normal(0, temp_noise_std, n_samples)
    if temp_drift:
        # 缓慢温度漂移 (空调温循环)，用于验证温漂补偿
        temp = temp + 2.0 * np.sin(2 * np.pi * t / 60.0)
    # 真值高度：用与固件完全一致的『气压->高度』公式 (含温度补偿) 计算，
    # 参考气压取与算法校准段相同的数据估计值，使真值与算法输出在同一参考下比较
    # （仅差噪声/暂态，RMSE 反映真实平滑与跟踪性能）。
    nc = min(SYNTH_CALIB_SAMPLES, n_samples)
    ref_ds = float(np.mean(base_p[:nc]))
    true_height = np.array([pressure_to_altitude_with_temp(base_p[i], temp_ref[i], ref_ds)
                            for i in range(n_samples)])
    return {
        'time': t,
        'ms_pressure': ms_p,
        'bmp_pressure': bmp_p,
        'temperature': temp,
        'true_height': true_height,
        'scenario': scenario,
    }


def dataset_to_rows(ds):
    """把 dataset dict 转成行列表，便于保存 CSV。"""
    rows = []
    for i in range(len(ds['time'])):
        rows.append({
            't_ms': int(ds['time'][i] * 1000),
            'sample_id': i,
            'ms5611_pressure_pa': round(float(ds['ms_pressure'][i]), 3),
            'bmp280_pressure_pa': round(float(ds['bmp_pressure'][i]), 3),
            'temperature_c': round(float(ds['temperature'][i]), 3),
            'true_height_m': round(float(ds['true_height'][i]), 4),
        })
    return rows


def save_synthetic_csv(scenario, out_dir, **kwargs):
    ds = generate_synthetic(scenario=scenario, **kwargs)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{scenario}.csv")
    rows = dataset_to_rows(ds)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


# ============================================================
# CSV 加载
# ============================================================
def _find_col(header, *names):
    for nm in names:
        for h in header:
            if h.strip().lower() == nm.lower():
                return h
    return None


def load_combined_csv(path):
    """加载合并单文件 CSV。"""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    time_c = _find_col(header, 't', 't_ms', 'time', 'timestamp', 'unix_time', 'sample_id')
    ms_c = _find_col(header, 'ms5611_pressure_pa', 'ms5611_pressure', 'ms_pressure',
                     'ms5611_p', 'ms_p', 'ms5611')
    bmp_c = _find_col(header, 'bmp280_pressure_pa', 'bmp280_pressure', 'bmp_pressure',
                      'bmp280_p', 'bmp_p', 'bmp280')
    temp_c = _find_col(header, 'temperature_c', 'temperature', 'temp', 't_c')
    truth_c = _find_col(header, 'true_height_m', 'true_height', 'height_m', 'truth',
                        'gt_height', 'label')

    def col(name):
        return [float(r[name]) for r in rows] if name else None

    if ms_c is None or bmp_c is None:
        raise ValueError(f"无法在 {path} 中找到 ms5611/bmp280 气压列。列名: {header}")
    time = col(time_c) if time_c else list(range(len(rows)))
    # sample_id 可能非时间，优先用 t_ms/timestamp
    ds = {
        'time': np.array(time, dtype=float),
        'ms_pressure': np.array(col(ms_c), dtype=float),
        'bmp_pressure': np.array(col(bmp_c), dtype=float),
        'temperature': np.array(col(temp_c), dtype=float) if temp_c else np.full(len(rows), 25.0),
        'true_height': np.array(col(truth_c), dtype=float) if truth_c else None,
        'scenario': os.path.splitext(os.path.basename(path))[0],
    }
    return ds


def load_two_files(ms_path, bmp_path):
    """加载真实采集的双文件格式 (ms5611_*.csv + bmp280_*.csv)。

    两路传感器独立记录，行数与 sample_id 序列可能不完全一致，因此按
    sample_id 内连接对齐；并提取固件已计算的 height_m 作为参考数据。
    """
    def _load_one(p):
        with open(p, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = list(reader)
        p_c = _find_col(header, 'pressure_pa', 'pressure', 'p_pa')
        t_c = _find_col(header, 'temperature_c', 'temperature', 'temp')
        sid_c = _find_col(header, 'sample_id', 'id')
        ut_c = _find_col(header, 'unix_time', 'time_s', 'timestamp')
        h_c = _find_col(header, 'height_m', 'kf_height_m', 'alt_m')
        kf_c = _find_col(header, 'kf_pressure_pa', 'kf_pressure', 'kf_p')
        if p_c is None:
            raise ValueError(f"{os.path.basename(p)} 缺少气压列(pressure_pa)")
        def col(name):
            return [float(r[name]) for r in rows] if name else None
        return {
            'sample_id': col(sid_c) if sid_c else list(range(len(rows))),
            'unix_time': col(ut_c) if ut_c else None,
            'pressure': col(p_c),
            'temp': col(t_c),
            'height': col(h_c),
            'kf_p': col(kf_c),
        }
    a = _load_one(ms_path)
    b = _load_one(bmp_path)
    a_by = {int(s): i for i, s in enumerate(a['sample_id'])}
    b_by = {int(s): i for i, s in enumerate(b['sample_id'])}
    common = sorted(set(a_by) & set(b_by))
    if not common:
        raise ValueError(f"{os.path.basename(ms_path)} 与 {os.path.basename(bmp_path)} "
                         f"无共同 sample_id，无法对齐")
    ai = [a_by[c] for c in common]
    bi = [b_by[c] for c in common]
    n = len(common)
    ms_p = np.array([a['pressure'][i] for i in ai], dtype=float)
    bmp_p = np.array([b['pressure'][i] for i in bi], dtype=float)
    # 温度：两路都有则取均值，否则用有的一路
    if a['temp'] is not None and b['temp'] is not None:
        temp = np.array([(a['temp'][ai[k]] + b['temp'][bi[k]]) / 2.0
                         for k in range(n)], dtype=float)
    elif a['temp'] is not None:
        temp = np.array([a['temp'][i] for i in ai], dtype=float)
    elif b['temp'] is not None:
        temp = np.array([b['temp'][i] for i in bi], dtype=float)
    else:
        temp = np.full(n, 25.0)
    ms_fw = (np.array([a['height'][i] for i in ai], dtype=float)
             if a['height'] else None)
    bmp_fw = (np.array([b['height'][i] for i in bi], dtype=float)
              if b['height'] else None)
    ms_kf_p = (np.array([a['kf_p'][ai[k]] for k in range(n)], dtype=float)
               if a['kf_p'] is not None else None)
    bmp_kf_p = (np.array([b['kf_p'][bi[k]] for k in range(n)], dtype=float)
                if b['kf_p'] is not None else None)
    # 时间：优先 unix_time 相对起点(秒)，否则用 sample_id
    if a['unix_time'] is not None:
        ut = np.array([a['unix_time'][i] for i in ai], dtype=float)
        time = ut - ut[0]
    else:
        time = np.array(common, dtype=float) - common[0]
    return {
        'time': time,
        'ms_pressure': ms_p,
        'bmp_pressure': bmp_p,
        'temperature': temp,
        'ms_kf_p': ms_kf_p,      # 固件 KF 滤波气压（真实数据自监督去噪伪真值）
        'bmp_kf_p': bmp_kf_p,
        'ms_fw_height': ms_fw,    # 固件高度(参考数据)，相对起点后在 GUI 中对比
        'bmp_fw_height': bmp_fw,
        'true_height': None,
        'scenario': 'real',
    }


def load_dataset(path_ms_or_combined, path_bmp=None):
    """统一入口：传一个文件按合并 CSV 解析；传两个文件按双文件解析。"""
    if path_bmp is not None:
        return load_two_files(path_ms_or_combined, path_bmp)
    return load_combined_csv(path_ms_or_combined)


if __name__ == '__main__':
    ds = generate_synthetic('elevation', n_samples=400, fs=10.0)
    print("elevation:", ds['ms_pressure'].shape, ds['true_height'][:1], ds['true_height'][-1])
    rows = dataset_to_rows(ds)
    print(rows[0])
