# -*- coding: utf-8 -*-
"""
算法性能指标
============
给定真实高度与融合/滤波高度序列，计算用于调参评估的指标：
  * RMSE         : 整体均方根误差
  * 静态噪声 std : 静止段输出高度标准差（越小越平滑）
  * 跟踪延迟     : 运动段融合高度滞后于真值的峰值时延
  * 回零误差     : 升降结束后回到静止基准的稳态偏差
  * 过冲         : 上升段最大超调
  * 漂移率       : 长时间静止段的线性漂移斜率
"""

import numpy as np


def _contiguous_runs(mask):
    """返回布尔掩码中连续 True 段的 [(start, end), ...]（end 不含）。"""
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return float('nan')
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_metrics(time, fused_h, truth, scenario='elevation'):
    """
    返回指标 dict。truth 可为 None（仅算静态噪声/漂移）。
    """
    res = {}
    n = len(fused_h)
    time = np.asarray(time, dtype=float)
    fused_h = np.asarray(fused_h, dtype=float)

    # 静态噪声：取序列后段（假设已稳定）的标准差；若无 truth 用全段稳态部分
    if truth is not None:
        truth = np.asarray(truth, dtype=float)
        res['rmse'] = rmse(fused_h, truth)
        # 运动检测：真值变化率
        dt = np.gradient(time)
        dt[dt == 0] = 1e-6
        dtruth = np.abs(np.gradient(truth) / dt)
        motion_mask = dtruth > 0.02   # m/s
        static_mask = ~motion_mask
    else:
        # 无真值：假设整段静止噪声用全段（减去线性趋势后）
        static_mask = np.ones(n, dtype=bool)
        motion_mask = np.zeros(n, dtype=bool)
        res['rmse'] = float('nan')

    res['static_noise_std_m'] = float('nan')
    res['drift_rate_m_per_s'] = float('nan')
    if truth is not None and static_mask.sum() > 5:
        residual = fused_h - truth
        # 将静止段切成若干『连续静止片段』，分别统计残差噪声，取中位数，
        # 这样不会被升降场景中段间 DC 偏置（算法固有偏差）污染。
        seg_stds = []
        runs = _contiguous_runs(static_mask)
        for (s, e) in runs:
            if (e - s) < 10:
                continue
            # 取该片段中后 80% 去掉初始收敛
            idx = np.arange(s, e)
            tail = idx[int(len(idx) * 0.2):]
            seg_stds.append(float(np.std(residual[tail])))
        if seg_stds:
            res['static_noise_std_m'] = float(np.median(seg_stds))
        # 漂移率：用首个足够长的静止片段的残差线性拟合斜率 (m/s)
        first = None
        for (s, e) in runs:
            if (e - s) > 10:
                first = (s, e)
                break
        if first is not None:
            idx = np.arange(first[0], first[1])
            tail = idx[int(len(idx) * 0.2):]
            tt = time[tail]
            rr = residual[tail]
            A = np.vstack([tt, np.ones_like(tt)]).T
            slope, _ = np.linalg.lstsq(A, rr, rcond=None)[0]
            res['drift_rate_m_per_s'] = float(slope)

    if truth is not None and motion_mask.sum() > 0:
        # 跟踪延迟：上升段，融合高度达到真值 90% 所需的额外时间
        res['tracking_lag_s'] = _tracking_lag(time, truth, fused_h)
        res['return_error_m'] = float(fused_h[-1] - truth[-1])
        res['overshoot_m'] = _overshoot(truth, fused_h)
    else:
        res['tracking_lag_s'] = float('nan')
        res['return_error_m'] = float('nan')
        res['overshoot_m'] = float('nan')

    return res


def _tracking_lag(time, truth, fused):
    """粗略估计上升段峰值时延（融合与真值的最大 lag）。"""
    diff = fused - truth
    # 仅在运动段寻找最大偏差对应的时间
    dt = np.gradient(time)
    dt[dt == 0] = 1e-6
    dtruth = np.abs(np.gradient(truth) / dt)
    motion = dtruth > 0.02
    if motion.sum() == 0:
        return float('nan')
    idx = np.where(motion)[0]
    k = int(np.argmax(np.abs(diff[idx])))
    return float(abs(diff[idx[k]]) / (np.max(np.abs(np.gradient(truth)/dt)) + 1e-6))


def _overshoot(truth, fused):
    """上升段融合高度相对真值的最大超调量。"""
    dt = np.gradient(np.asarray(truth, dtype=float))
    rising = dt > 0
    if rising.sum() == 0:
        return float('nan')
    over = (fused[rising] - truth[rising])
    return float(np.max(over)) if len(over) else float('nan')


def format_metrics(m):
    lines = []
    for k, v in m.items():
        if isinstance(v, float):
            if v != v:
                lines.append(f"{k}: NaN")
            else:
                lines.append(f"{k}: {v:.4f}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)
