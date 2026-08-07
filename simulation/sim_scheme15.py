# -*- coding: utf-8 -*-
"""
方案15 仿真 + 完全调参（基于 serial_tool/data/raw 真实数据）
============================================================
忠实复现固件 (FUSION_SCHEME=15, 重写版) 的算法链路：
  场景判定：BMP280 原始气压 滑动窗口方差(STD) → Schmitt 门控
            (std > S15_MOT_STD_OPEN 开门=运动； < S15_MOT_STD_CLOSE 关门=静止)
  相对高度：运动段 相对高度 = 锚点锁定高度 + ISA(当前原始气压, 运动起点原始气压)
            静止段完全冻结（路径无关、无累积、无 KF 滞后）
  启动稳定：前 S15_WARMUP_FRAMES 帧 h15_lock 跟随 KF 绝对高度，之后首帧锚定。

注：重写版方案15 只用 BMP280 原始气压（与固件一致），不依赖 MS5611 / KF / NN。
    参考真值取 CSV 的 height_m（设备上报海拔）。

输出：
  - sim_s15_静止.png / 平移.png / 升降.png   三场景波形（默认 vs 调优 vs 参考 + 门控）
  - sim_s15_tuning.png        RMSE 对比 + 场景判定 + 参数表
  - scheme15_tuned_params.json   推荐参数与指标
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# ============== ISA 常量（与 altitude_convert.h 一致） ==============
ISA_T0 = 288.15
ISA_L  = 0.0065
ISA_G  = 9.80665
ISA_R  = 287.05
EXP    = ISA_L * ISA_R / ISA_G      # ≈ 0.1903

def isa(p, p0):
    p = np.asarray(p, dtype=float)
    p0 = float(p0)
    r = p / p0
    r = np.clip(r, 1e-6, 10.0)
    return (ISA_T0 / ISA_L) * (1.0 - r ** EXP)

# ============== 加载真实数据（只用 BMP280） ==============
RAW_DIR = r"e:/ST/QMSX/serial_tool/data/raw"

def load_all():
    """遍历三场景目录，加载所有 bmp280_*.csv，返回数据集列表。"""
    datasets = []
    for scen in ["静止", "平移运动", "升降运动"]:
        files = sorted(glob.glob(os.path.join(RAW_DIR, scen, "bmp280_*.csv")))
        for f in files:
            df = pd.read_csv(f)
            t = df["unix_time"].values.astype(float)
            dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.13
            fs = 1.0 / dt
            datasets.append(dict(
                scen=scen,
                tag=os.path.basename(f)[len("bmp280_"):-4],
                pressure_pa=df["pressure_pa"].values.astype(float),
                kf_height_m=df["kf_height_m"].values.astype(float),
                ref_height=df["height_m"].values.astype(float),
                fs=fs, n=len(df),
            ))
    return datasets

# ============== 预计算滑动窗口 STD（与门控状态无关，可向量化） ==============
def rolling_std(x, win):
    n = len(x)
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - win + 1)
        w = x[lo:i + 1]
        out[i] = float(np.std(w))
    return out

# ============== 方案15 前向计算（忠实复现固件逻辑） ==============
def scheme15(ds, OPEN, CLOSE, WIN, WARMUP, return_gate=False):
    p = ds["pressure_pa"]
    kf_h = ds["kf_height_m"]
    n = ds["n"]

    # 预计算窗口 STD（与固件 fill 递增等价：x[max(0,i-WIN+1):i+1].std()）
    std = rolling_std(p, WIN)

    h15_lock = 0.0
    gate_state = False
    first_run = True
    warmup = 0
    locked = False
    ref_p = 0.0
    ref_h = 0.0
    was_static = True
    pbuf = np.zeros(WIN, dtype=float)
    pidx = 0
    pfill = 0

    out = np.empty(n, dtype=float)
    gate = np.zeros(n, dtype=bool)   # 仅 else 分支（正常运行）有效；启动期记为 False

    for i in range(n):
        if not locked:
            h15_lock = kf_h[i]
            warmup += 1
            if warmup >= WARMUP:
                locked = True
            out[i] = h15_lock
            # 启动期不计门控（固件也未更新 g_mt_scene）
            continue
        elif first_run:
            h15_lock = kf_h[i]
            first_run = False
            ref_p = p[i]
            ref_h = h15_lock
            was_static = True
            pfill = 0
            pidx = 0
            out[i] = h15_lock
            continue
        else:
            s = std[i]
            if not gate_state:
                if s > OPEN:
                    gate_state = True
            else:
                if s < CLOSE:
                    gate_state = False
            g = 1.0 if gate_state else 0.0
            gate[i] = bool(gate_state)

            # win_oldest：将写入位的现有值 = 窗口最旧样本（运动刚起步时≈运动前静止气压）
            win_oldest = pbuf[pidx] if pfill >= WIN else p[i]
            pbuf[pidx] = p[i]
            pidx = (pidx + 1) % WIN
            if pfill < WIN:
                pfill += 1

            gate_open = g > 0.5
            if was_static and gate_open:
                # STATIC→ELEVATION 跳变：冻结当前高度，回溯到运动前静止气压作锚点
                ref_p = win_oldest
                ref_h = h15_lock
            was_static = not gate_open
            if gate_open:
                rel = isa(p[i], ref_p)
                h15_lock = ref_h + rel
            # else：静止段完全冻结
            out[i] = h15_lock

    if return_gate:
        return out, gate
    return out

# ============== 默认（当前固件头文件值） ==============
DEFAULT = dict(OPEN=3.0, CLOSE=1.5, WIN=16, WARMUP=64)

# ============== 指标 ==============
def rmse(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def motion_label(h, fs, thr_disp=0.06, win_s=0.5):
    """由参考 height_m 自动标注运动段（labels）：窗口位移超阈值即运动。"""
    W = max(1, int(round(fs * win_s)))
    disp = np.abs(h - np.roll(h, W))
    disp[:W] = disp[W]
    return disp > thr_disp

def dataset_metrics(ds, params, label=None):
    out = scheme15(ds, **params)
    ref = ds["ref_height"]
    raw = rmse(out, ref)
    dem = rmse(out - out.mean(), ref - ref.mean())
    m = {}
    if label is not None and label.sum() > 10:
        m["rmse_motion"] = rmse(out[label], ref[label])
    else:
        m["rmse_motion"] = float("nan")
    return dict(out=out, ref=ref, rmse_raw=raw, rmse_demean=dem, **m)

def scene_f1(out, ref, gate, label, valid_from):
    """门控 vs 自动标签：只看正常运行帧（i>=valid_from）。
    返回 precision/recall/f1（运动为正类）。"""
    g = gate[valid_from:].astype(bool)
    l = label[valid_from:]
    tp = int(np.sum(g & l))
    fp = int(np.sum(g & ~l))
    fn = int(np.sum(~g & l))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(precision=prec, recall=rec, f1=f1, motion_frac=float(g.mean()))

# ============== 主流程 ==============
def main():
    datasets = load_all()
    print(f"发现 {len(datasets)} 个数据集：")
    for ds in datasets:
        print(f"  [{ds['scen']}] {ds['tag']}: n={ds['n']}, fs={ds['fs']:.1f}Hz, "
              f"参考高度均值={ds['ref_height'].mean():.2f}m, "
              f"气压范围=[{ds['pressure_pa'].min():.1f},{ds['pressure_pa'].max():.1f}]Pa")

    # 自动运动标签（供场景判定评估）
    for ds in datasets:
        ds["label"] = motion_label(ds["ref_height"], ds["fs"])
        frac = ds["label"].mean()
        print(f"    -> 自动标签运动占比 {frac*100:.1f}%")

    # ---------- 1. 默认参数基线 ----------
    print("\n[1] 默认参数基线评估...")
    base_per = [dataset_metrics(ds, DEFAULT, ds["label"]) for ds in datasets]
    base_total_raw = np.sqrt(np.mean([m["rmse_raw"] ** 2 for m in base_per]))
    base_total_dem = np.sqrt(np.mean([m["rmse_demean"] ** 2 for m in base_per]))
    print(f"  默认 总RMSE(含偏移)={base_total_raw:.4f}m  总RMSE(去偏移)={base_total_dem:.4f}m")

    # ---------- 2. 完全调优（网格粗搜 + 局部精修） ----------
    print("\n[2] 完全调优...")
    rng = np.random.default_rng(20260714)

    def obj(params, dss):
        tot = 0.0
        for d in dss:
            m = dataset_metrics(d, params)
            tot += m["rmse_demean"] ** 2
        return np.sqrt(tot / len(dss))

    grid_OPEN  = [1.0, 1.5, 2.0, 3.0, 5.0]
    grid_CLOSE = [0.5, 1.0, 1.5, 2.5]
    grid_WIN   = [8, 12, 16, 24]
    grid_WARM  = [32, 64, 96]
    best = None; best_obj = 1e9
    for O in grid_OPEN:
        for C in grid_CLOSE:
            if C >= O:   # 迟滞必须 OPEN > CLOSE
                continue
            for W in grid_WIN:
                for Wp in grid_WARM:
                    p = dict(OPEN=O, CLOSE=C, WIN=W, WARMUP=Wp)
                    o = obj(p, datasets)
                    if o < best_obj:
                        best_obj = o; best = dict(p)
    print(f"  网格最优(去偏移RMSE)={best_obj:.4f}m  {best}")

    # 局部精修
    for _ in range(400):
        p = dict(best)
        p["OPEN"]  = np.clip(best["OPEN"]  + rng.normal(0, 0.4), 0.6, 8.0)
        p["CLOSE"] = np.clip(best["CLOSE"] + rng.normal(0, 0.3), 0.3, 5.0)
        if p["CLOSE"] >= p["OPEN"]:
            continue
        p["WIN"]   = int(np.clip(best["WIN"] + rng.integers(-4, 5), 6, 40))
        p["WARMUP"]= int(np.clip(best["WARMUP"] + rng.integers(-16, 17), 16, 160))
        o = obj(p, datasets)
        if o < best_obj:
            best_obj = o; best = dict(p)
    TUNED = best
    print(f"  调优后(去偏移)总RMSE={best_obj:.4f}m")
    print(f"  推荐参数: {TUNED}")

    tuned_per = [dataset_metrics(ds, TUNED, ds["label"]) for ds in datasets]
    tuned_total_raw = np.sqrt(np.mean([m["rmse_raw"] ** 2 for m in tuned_per]))
    tuned_total_dem = np.sqrt(np.mean([m["rmse_demean"] ** 2 for m in tuned_per]))

    # 场景判定 F1（默认 vs 调优）
    def f1_of(params):
        res = []
        for ds in datasets:
            out, gate = scheme15(ds, return_gate=True, **params)
            valid_from = params["WARMUP"] + 1
            res.append(scene_f1(out, ds["ref_height"], gate, ds["label"], valid_from))
        return res
    base_f1 = f1_of(DEFAULT)
    tuned_f1 = f1_of(TUNED)

    # ---------- 3. 绘图 ----------
    print("\n[3] 生成波形图...")
    out_dir = os.path.dirname(os.path.abspath(__file__))

    for scen in ["静止", "平移运动", "升降运动"]:
        ds = next(d for d in datasets if d["scen"] == scen)
        base_out = scheme15(ds, **DEFAULT)
        tune_out, tune_gate = scheme15(ds, return_gate=True, **TUNED)
        t = np.arange(ds["n"]) / ds["fs"]
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        # 气压 + 门控
        ax = axes[0]
        ax.plot(t, ds["pressure_pa"], 'gray', lw=0.6, alpha=0.5, label='BMP280 原始气压')
        ax.set_ylabel('气压 (Pa)')
        ax2 = ax.twinx()
        ax2.plot(t, tune_gate.astype(int), 'r-', lw=1.0, alpha=0.7, label='门控(1=运动)')
        ax2.set_ylabel('门控'); ax2.set_ylim(-0.1, 1.3)
        ax.set_title(f'方案15 仿真 — {scen}  [数据集 {ds["tag"]}]')
        ax.legend(fontsize=8, loc='upper left'); ax2.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        # 高度：参考 vs 默认 vs 调优
        axes[1].plot(t, ds["ref_height"], 'k-', lw=1.3, label='参考 height_m')
        axes[1].plot(t, base_out, 'r:', lw=1.0, label='默认')
        axes[1].plot(t, tune_out, 'b-', lw=1.1, label='调优')
        axes[1].set_ylabel('高度 (m)'); axes[1].legend(fontsize=8, ncol=3); axes[1].grid(alpha=0.3)
        # 残差（调优 - 参考）
        axes[2].plot(t, tune_out - ds["ref_height"], 'b-', lw=0.8,
                     label=f'调优残差 std={np.std(tune_out-ds["ref_height"]):.3f}m')
        axes[2].axhline(0, color='k', lw=0.8, alpha=0.4)
        axes[2].set_ylabel('残差 (m)'); axes[2].set_xlabel('时间 (s)')
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
        plt.tight_layout()
        fname = f'sim_s15_{scen}.png'.replace('平移运动', '平移').replace('升降运动', '升降')
        plt.savefig(os.path.join(out_dir, fname), dpi=150); plt.close()
        print(f"  已保存: {fname}")

    # 调优结果图
    fig = plt.figure(figsize=(14, 10))
    labels = [f'{d["scen"]}\n{d["tag"][-6:]}' for d in datasets]
    xx = np.arange(len(datasets)); w = 0.26
    # RMSE 对比（去偏移）
    ax1 = fig.add_subplot(2, 2, 1)
    base_dm = [m["rmse_demean"] for m in base_per]
    tune_dm = [m["rmse_demean"] for m in tuned_per]
    ax1.bar(xx - w, base_dm, w, label='默认', color='orange')
    ax1.bar(xx,     tune_dm, w, label='调优', color='steelblue')
    ax1.set_xticks(xx); ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_ylabel('去偏移 RMSE (m)'); ax1.set_title('相对/动态精度（去偏移）')
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3, axis='y')
    # 运动段 RMSE
    ax1b = fig.add_subplot(2, 2, 2)
    base_m = [m.get("rmse_motion", float('nan')) for m in base_per]
    tune_m = [m.get("rmse_motion", float('nan')) for m in tuned_per]
    ax1b.bar(xx - w, base_m, w, label='默认', color='orange')
    ax1b.bar(xx,     tune_m, w, label='调优', color='steelblue')
    ax1b.set_xticks(xx); ax1b.set_xticklabels(labels, fontsize=6)
    ax1b.set_ylabel('运动段 RMSE (m)'); ax1b.set_title('运动段动态精度')
    ax1b.legend(fontsize=7); ax1b.grid(alpha=0.3, axis='y')
    # 场景判定 F1
    ax2 = fig.add_subplot(2, 2, 3)
    bf1 = [f["f1"] for f in base_f1]; tf1 = [f["f1"] for f in tuned_f1]
    bprec = [f["precision"] for f in base_f1]; tprec = [f["precision"] for f in tuned_f1]
    brec = [f["recall"] for f in base_f1]; trec = [f["recall"] for f in tuned_f1]
    ax2.bar(xx - w, bprec, w, label='默认 precision', color='orange')
    ax2.bar(xx,     tprec, w, label='调优 precision', color='steelblue')
    ax2.bar(xx + w, trec, w, label='调优 recall', color='green')
    ax2.set_xticks(xx); ax2.set_xticklabels(labels, fontsize=6)
    ax2.set_ylabel('分数'); ax2.set_title('场景判定 (F1/精确率/召回率)')
    ax2.set_ylim(0, 1.05); ax2.legend(fontsize=6); ax2.grid(alpha=0.3, axis='y')
    # 参数表
    ax3 = fig.add_subplot(2, 2, 4); ax3.axis('off')
    rows = [["参数", "默认", "调优"]]
    for k in ["OPEN", "CLOSE", "WIN", "WARMUP"]:
        rows.append([k, f"{DEFAULT[k]:g}", f"{TUNED[k]:g}"])
    rows.append(["总RMSE(含偏移)", f"{base_total_raw:.3f}m", f"{tuned_total_raw:.3f}m"])
    rows.append(["总RMSE(去偏移)", f"{base_total_dem:.4f}m", f"{tuned_total_dem:.4f}m"])
    rows.append(["平均F1(默认/调优)", f"{np.mean(bf1):.3f}", f"{np.mean(tf1):.3f}"])
    tbl = ax3.table(cellText=rows, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    ax3.set_title('推荐参数与效果对比', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sim_s15_tuning.png'), dpi=150); plt.close()
    print("  已保存: sim_s15_tuning.png")

    # ---------- 4. 保存推荐参数 ----------
    summary = dict(
        default=DEFAULT, tuned=TUNED,
        metrics=dict(
            base_total_raw_rmse=base_total_raw,
            base_total_demean_rmse=base_total_dem,
            tuned_total_raw_rmse=tuned_total_raw,
            tuned_total_demean_rmse=tuned_total_dem,
            per_dataset=dict(
                scen=[d["scen"] for d in datasets],
                tag=[d["tag"] for d in datasets],
                base_raw=[m["rmse_raw"] for m in base_per],
                tuned_raw=[m["rmse_raw"] for m in tuned_per],
                base_demean=[m["rmse_demean"] for m in base_per],
                tuned_demean=[m["rmse_demean"] for m in tuned_per],
                base_motion=[m.get("rmse_motion", float('nan')) for m in base_per],
                tuned_motion=[m.get("rmse_motion", float('nan')) for m in tuned_per],
                base_f1=[f["f1"] for f in base_f1],
                tuned_f1=[f["f1"] for f in tuned_f1],
                base_prec=[f["precision"] for f in base_f1],
                tuned_prec=[f["precision"] for f in tuned_f1],
                base_rec=[f["recall"] for f in base_f1],
                tuned_rec=[f["recall"] for f in tuned_f1],
            ),
        ),
        notes=dict(
            algo="方案15 重写版：原始气压窗口方差(STD)门控 + 原始气压 ISA 相对高度；只用 BMP280 原始气压。",
            reference="CSV 的 height_m 作为参考真值。",
            target="最小化去偏移 RMSE（相对/动态精度），同时保证场景判定 F1。",
        ),
    )
    with open(os.path.join(out_dir, 'scheme15_tuned_params.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n[完成] 推荐参数已保存至 scheme15_tuned_params.json")

if __name__ == '__main__':
    main()
