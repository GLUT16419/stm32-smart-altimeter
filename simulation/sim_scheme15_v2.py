# -*- coding: utf-8 -*-
"""
方案15 仿真 v2（增强灵敏度：长窗口 STD 门控 + 短窗口 STD 门控 OR 组合）
=================================================================
问题：仅用 20 帧长窗口气压 STD 判场景，对「25cm 小幅度 + 快速」变化不敏感——
25cm 仅约 3Pa 气压偏移，被长窗口平均后 STD 低于开门阈值，门控不触发，场景切不过去。

改进：新增互补检测器 —— 短窗口(FAST_WIN 帧)气压 STD。
  短窗口内低频漂移可忽略，快速移动时窗口横跨跳变、STD 陡升(~1.2Pa)，立刻触发；
  移动结束窗口追上后 STD 回落自动关门。对真实传感器的低频漂移免疫（优于"偏差法"）。
  最终门控 = 长窗口STD门控 OR 短窗口STD门控（各自独立 Schmitt 迟滞）。

验证：
  - 真实数据(静止/平移/升降 6 数据集) 评估 RMSE + 场景 F1 + 静止误触发率，确保不退化；
  - 合成 25cm 快速(0.3s)/中速(1.0s)/慢速(3.0s)/快速来回 运动，验证门控及时开启且最终误差<5cm。
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
ISA_T0 = 288.15; ISA_L = 0.0065; ISA_G = 9.80665; ISA_R = 287.05
EXP = ISA_L * ISA_R / ISA_G

def isa(p, p0):
    p = np.asarray(p, dtype=float); p0 = float(p0)
    r = p / p0; r = np.clip(r, 1e-6, 10.0)
    return (ISA_T0 / ISA_L) * (1.0 - r ** EXP)

def smoothstep(x):
    x = min(1.0, max(0.0, x)); return x*x*(3-2*x)

RAW_DIR = r"e:/ST/QMSX/serial_tool/data/raw"

def load_all():
    datasets = []
    for scen in ["静止", "平移运动", "升降运动"]:
        files = sorted(glob.glob(os.path.join(RAW_DIR, scen, "bmp280_*.csv")))
        for f in files:
            df = pd.read_csv(f)
            t = df["unix_time"].values.astype(float)
            dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.13
            datasets.append(dict(
                scen=scen, tag=os.path.basename(f)[len("bmp280_"):-4],
                pressure_pa=df["pressure_pa"].values.astype(float),
                kf_height_m=df["kf_height_m"].values.astype(float),
                ref_height=df["height_m"].values.astype(float),
                fs=1.0/dt, n=len(df)))
    return datasets

def rolling_std(x, win):
    n = len(x); out = np.empty(n, dtype=float)
    for i in range(n):
        w = x[max(0, i-win+1):i+1]
        out[i] = float(np.std(w))
    return out

# ============== 方案15 v2 前向计算 ==============
def scheme15_v2(ds, OPEN, CLOSE, WIN, WARMUP, FAST_WIN, FAST_OPEN, FAST_CLOSE,
                REANCHOR_MIN=4, return_gate=False):
    p = ds["pressure_pa"]; kf_h = ds["kf_height_m"]; n = ds["n"]
    std = rolling_std(p, WIN); sstd = rolling_std(p, FAST_WIN)
    h15_lock = 0.0; gate_state = False; first_run = True; warmup = 0; locked = False
    ref_p = 0.0; ref_h = 0.0; was_static = True
    std_gate = False; fast_gate = False
    static_streak = 0
    pbuf = np.zeros(WIN, dtype=float); pidx = 0; pfill = 0
    out = np.empty(n, dtype=float); gate = np.zeros(n, dtype=bool)
    for i in range(n):
        if not locked:
            h15_lock = kf_h[i]; warmup += 1
            if warmup >= WARMUP: locked = True
            out[i] = h15_lock; continue
        elif first_run:
            h15_lock = kf_h[i]; first_run = False
            ref_p = p[i]; ref_h = h15_lock; was_static = True
            pfill = 0; pidx = 0; out[i] = h15_lock; continue
        else:
            s = std[i]; fs_ = sstd[i]
            if not std_gate:
                if s > OPEN: std_gate = True
            else:
                if s < CLOSE: std_gate = False
            if not fast_gate:
                if fs_ > FAST_OPEN: fast_gate = True
            else:
                if fs_ < FAST_CLOSE: fast_gate = False
            g = (std_gate or fast_gate)
            gate[i] = bool(g)
            # 连续静止帧计数：仅在"连续静止足够帧后转运动"(真正的运动起始边沿)才重锚，
            # 避免快速运动时短窗口抖动(flicker)反复重锚导致高度重复计算/卡半路。
            if g:
                static_streak = 0
            else:
                static_streak += 1
            win_oldest = pbuf[pidx] if pfill >= WIN else p[i]
            pbuf[pidx] = p[i]; pidx = (pidx + 1) % WIN
            if pfill < WIN: pfill += 1
            gate_open = g
            if (static_streak >= REANCHOR_MIN) and gate_open:
                ref_p = win_oldest; ref_h = h15_lock
            was_static = not gate_open
            if gate_open:
                rel = isa(p[i], ref_p); h15_lock = ref_h + rel
            out[i] = h15_lock
    if return_gate: return out, gate
    return out

# ============== 合成 25cm 运动 ==============
def make_synthetic(fs=7.7, n=320, amp=0.25, dur=1.0, warmup_frames=64, seed=1, double=False):
    rng = np.random.default_rng(seed)
    dt = 1.0/fs; t = np.arange(n)*dt
    p0 = 101325.0
    p = p0 + rng.normal(0, 0.35, n)
    t0 = warmup_frames*dt + 1.5
    dh_up = np.zeros(n)
    for i in range(n):
        tt = t[i]-t0
        if tt >= 0: dh_up[i] = amp*smoothstep(tt/dur)
    ref = dh_up.copy()
    if double:
        t1 = t0 + dur + 2.5
        for i in range(n):
            tt = t[i]-t1
            if tt >= 0: ref[i] = amp*(1 - smoothstep(tt/dur))   # 降回 0
    factor = (1 - ref*ISA_L/ISA_T0)**(1.0/EXP)
    p2 = p * factor
    return dict(scen="合成25cm", tag=f"amp{int(amp*100)}cm_dur{dur:g}{'_dbl' if double else ''}",
                pressure_pa=p2, kf_height_m=ref.copy(), ref_height=ref.copy(), fs=fs, n=n,
                dh=ref, t=t, t0=t0, dur=dur, amp=amp, double=double)

# ============== 指标 ==============
def rmse(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a-b)**2)))

def motion_label(h, fs, thr_disp=0.06, win_s=0.5):
    W = max(1, int(round(fs*win_s)))
    disp = np.abs(h - np.roll(h, W)); disp[:W] = disp[W]
    return disp > thr_disp

def dataset_metrics(ds, params):
    out = scheme15_v2(ds, **params)
    ref = ds["ref_height"]
    return dict(out=out, ref=ref,
                rmse_raw=rmse(out, ref),
                rmse_demean=rmse(out-out.mean(), ref-ref.mean()))

def scene_f1(out, ref, gate, label, valid_from):
    g = gate[valid_from:].astype(bool); l = label[valid_from:]
    tp = int(np.sum(g & l)); fp = int(np.sum(g & ~l)); fn = int(np.sum(~g & l))
    prec = tp/(tp+fp) if (tp+fp)>0 else 1.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 1.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    stat = ~l
    false_trig = float(np.sum(g & stat)/max(1, np.sum(stat)))
    return dict(precision=prec, recall=rec, f1=f1, false_trig=false_trig)

def synth_eval(ds, params):
    out, gate = scheme15_v2(ds, return_gate=True, **params)
    t = ds["t"]; t0 = ds["t0"]; dur = ds["dur"]
    # 检测：上升段或(双向)下降段门控开启
    up_mask = (t >= t0-0.05) & (t <= t0+dur+0.5)
    detected = bool(gate[up_mask].any())
    if ds["double"]:
        t1 = t0 + dur + 2.5
        down_mask = (t >= t1-0.05) & (t <= t1+dur+0.5)
        detected = detected or bool(gate[down_mask].any())
    # 最终稳定段误差（轨迹最后 15% 帧）
    k = int(ds["n"]*0.85)
    err = float(np.mean(np.abs(out[k:] - ds["ref_height"][k:]))) if (ds["n"]-k) > 5 else float("nan")
    return dict(detected=detected, final_err=err)

# ============== 主流程 ==============
def main():
    datasets = load_all()
    for ds in datasets: ds["label"] = motion_label(ds["ref_height"], ds["fs"])
    synths = [
        make_synthetic(dur=0.3, seed=1),
        make_synthetic(dur=1.0, seed=2),
        make_synthetic(dur=3.0, seed=3),
        make_synthetic(dur=0.3, seed=4, double=True),
    ]
    BASE = dict(OPEN=1.19, CLOSE=0.84, WIN=20, WARMUP=64, FAST_WIN=20, FAST_OPEN=1e9, FAST_CLOSE=1e9)

    print("搜索 FAST 短窗口STD门控参数...")
    grid_FAST_WIN  = [4, 5, 6, 8]
    grid_FAST_OPEN = [0.5, 0.6, 0.7, 0.8, 1.0]
    grid_FAST_CLOSE= [0.25, 0.35, 0.45, 0.55]
    grid_WIN = [20]; grid_WARM = [64, 104]
    best = None; best_score = -1e9
    for FW in grid_FAST_WIN:
        for FO in grid_FAST_OPEN:
            for FC in grid_FAST_CLOSE:
                if FC >= FO: continue
                for WARM in grid_WARM:
                    P = dict(OPEN=1.19, CLOSE=0.84, WIN=20, WARMUP=WARM,
                             FAST_WIN=FW, FAST_OPEN=FO, FAST_CLOSE=FC)
                    tot = 0.0; tot_raw = 0.0; max_ft = 0.0
                    for d in datasets:
                        m = dataset_metrics(d, P)
                        tot += m["rmse_demean"]**2
                        tot_raw += m["rmse_raw"]**2
                        out, gate = scheme15_v2(d, return_gate=True, **P)
                        f = scene_f1(m["out"], d["ref_height"], gate, d["label"], P["WARMUP"]+1)
                        max_ft = max(max_ft, f["false_trig"])
                    real_dm = np.sqrt(tot/len(datasets))
                    real_raw = np.sqrt(tot_raw/len(datasets))
                    all_det = True; max_err = 0.0
                    for s in synths:
                        e = synth_eval(s, P)
                        if not e["detected"]: all_det = False
                        max_err = max(max_err, e["final_err"])
                    if not all_det: continue
                    score = -real_dm - 0.4*real_raw - 8.0*max_ft - 1.0*max_err - 0.03*FO - 0.02*FW
                    if score > best_score:
                        best_score = score; best = dict(P)
    if best is None:
        best = dict(OPEN=1.19, CLOSE=0.84, WIN=20, WARMUP=64, FAST_WIN=5, FAST_OPEN=0.7, FAST_CLOSE=0.4)
    print(f"推荐: {best}")
    TUNED = best

    base_per = [dataset_metrics(d, BASE) for d in datasets]
    tune_per = [dataset_metrics(d, TUNED) for d in datasets]
    base_dm = np.sqrt(np.mean([m["rmse_demean"]**2 for m in base_per]))
    tune_dm = np.sqrt(np.mean([m["rmse_demean"]**2 for m in tune_per]))
    base_raw = np.sqrt(np.mean([m["rmse_raw"]**2 for m in base_per]))
    tune_raw = np.sqrt(np.mean([m["rmse_raw"]**2 for m in tune_per]))

    def f1_of(params):
        res = []
        for d in datasets:
            out, gate = scheme15_v2(d, return_gate=True, **params)
            res.append(scene_f1(out, d["ref_height"], gate, d["label"], params["WARMUP"]+1))
        return res
    base_f1 = f1_of(BASE); tune_f1 = f1_of(TUNED)

    print(f"\n真实数据 去偏移RMSE: 基线(仅STD)={base_dm:.4f}  调优(STD+FAST)={tune_dm:.4f}")
    print(f"真实数据 含偏移RMSE: 基线={base_raw:.4f}  调优={tune_raw:.4f}")
    for d, bf, tf in zip(datasets, base_f1, tune_f1):
        print(f"  [{d['scen']}] F1 基线={bf['f1']:.2f}/误触发={bf['false_trig']*100:.1f}%  "
              f"调优={tf['f1']:.2f}/误触发={tf['false_trig']*100:.1f}%")
    print("\n合成 25cm 检测:")
    for s in synths:
        eb = synth_eval(s, BASE); et = synth_eval(s, TUNED)
        print(f"  {s['tag']:20s} 基线检测={eb['detected']} | 调优检测={et['detected']} 最终误差={et['final_err']*100:.1f}cm")

    # 绘图
    out_dir = os.path.dirname(os.path.abspath(__file__))
    s = synths[0]
    out, gate = scheme15_v2(s, return_gate=True, **TUNED)
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    ax[0].plot(s["t"], s["pressure_pa"], 'gray', lw=0.6, alpha=0.6, label='合成原始气压')
    ax[0].set_ylabel('气压 (Pa)')
    ax2 = ax[0].twinx(); ax2.plot(s["t"], gate.astype(int), 'r-', lw=1.0, label='门控')
    ax2.set_ylabel('门控'); ax2.set_ylim(-0.1, 1.3)
    ax[0].axvline(s["t0"], color='g', ls='--', lw=0.8); ax[0].set_title('方案15 v2 — 合成25cm超快速(0.3s)')
    ax[1].plot(s["t"], s["ref_height"], 'k-', lw=1.3, label='参考(25cm)')
    ax[1].plot(s["t"], out, 'b-', lw=1.1, label='方案15-v2')
    ax[1].set_ylabel('高度 (m)'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[2].plot(s["t"], out - s["ref_height"], 'b-', lw=0.8, label='残差')
    ax[2].axhline(0, color='k', lw=0.8, alpha=0.4); ax[2].set_ylabel('残差 (m)'); ax[2].set_xlabel('时间 (s)')
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'sim_s15v2_synth.png'), dpi=150); plt.close()

    ds = next(d for d in datasets if d["scen"] == "升降运动")
    base_out, base_gate = scheme15_v2(ds, return_gate=True, **BASE)
    tune_out, tune_gate = scheme15_v2(ds, return_gate=True, **TUNED)
    t = np.arange(ds["n"])/ds["fs"]
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    ax[0].plot(t, ds["pressure_pa"], 'gray', lw=0.6, alpha=0.5)
    ax[0].set_ylabel('气压 (Pa)')
    ax2 = ax[0].twinx()
    ax2.plot(t, tune_gate.astype(int), 'r-', lw=1.0, alpha=0.7, label='调优门控')
    ax2.plot(t, base_gate.astype(int), 'c:', lw=1.0, alpha=0.7, label='基线门控')
    ax2.set_ylabel('门控'); ax2.set_ylim(-0.1, 1.3); ax2.legend(fontsize=8)
    ax[0].set_title(f'方案15 v2 — 升降运动 [{ds["tag"]}]')
    ax[1].plot(t, ds["ref_height"], 'k-', lw=1.3, label='参考')
    ax[1].plot(t, base_out, 'r:', lw=1.0, label='基线(仅STD)')
    ax[1].plot(t, tune_out, 'b-', lw=1.1, label='调优(STD+FAST)')
    ax[1].set_ylabel('高度 (m)'); ax[1].legend(fontsize=8, ncol=3); ax[1].grid(alpha=0.3)
    ax[2].plot(t, tune_out - ds["ref_height"], 'b-', lw=0.8, label='调优残差')
    ax[2].axhline(0, color='k', lw=0.8, alpha=0.4); ax[2].set_ylabel('残差 (m)'); ax[2].set_xlabel('时间 (s)')
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'sim_s15v2_升降.png'), dpi=150); plt.close()

    fig = plt.figure(figsize=(13, 9))
    labels = [f'{d["scen"]}\n{d["tag"][-6:]}' for d in datasets]
    xx = np.arange(len(datasets)); w = 0.26
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.bar(xx-w, [m["rmse_demean"] for m in base_per], w, label='基线(仅STD)', color='orange')
    ax1.bar(xx,   [m["rmse_demean"] for m in tune_per], w, label='调优(STD+FAST)', color='steelblue')
    ax1.set_xticks(xx); ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_ylabel('去偏移 RMSE (m)'); ax1.set_title('真实数据 相对/动态精度'); ax1.legend(fontsize=7); ax1.grid(alpha=0.3, axis='y')
    ax2 = fig.add_subplot(2, 2, 2)
    tf1=[f["f1"] for f in tune_f1]; tft=[f["false_trig"] for f in tune_f1]; bf1=[f["f1"] for f in base_f1]
    ax2.bar(xx-w, bf1, w, label='基线 F1', color='orange')
    ax2.bar(xx,   tf1, w, label='调优 F1', color='steelblue')
    ax2.bar(xx+w, tft, w, label='调优 误触发率', color='red', alpha=0.6)
    ax2.set_xticks(xx); ax2.set_xticklabels(labels, fontsize=6)
    ax2.set_ylabel('分数'); ax2.set_title('场景判定 F1 / 误触发率'); ax2.set_ylim(0,1.05)
    ax2.legend(fontsize=6); ax2.grid(alpha=0.3, axis='y')
    ax3 = fig.add_subplot(2, 2, 3)
    sc_tags=[s["tag"] for s in synths]
    det=[1 if synth_eval(s, TUNED)["detected"] else 0 for s in synths]
    bdet=[1 if synth_eval(s, BASE)["detected"] else 0 for s in synths]
    ax3.bar(np.arange(len(synths))-0.2, bdet, 0.4, label='基线检测', color='orange')
    ax3.bar(np.arange(len(synths))+0.2, det, 0.4, label='调优检测', color='steelblue')
    ax3.set_xticks(np.arange(len(synths))); ax3.set_xticklabels(sc_tags, fontsize=6, rotation=20)
    ax3.set_ylabel('检测到(1/0)'); ax3.set_title('合成25cm 检测率'); ax3.legend(fontsize=7); ax3.set_ylim(0,1.3); ax3.grid(alpha=0.3, axis='y')
    ax4 = fig.add_subplot(2, 2, 4); ax4.axis('off')
    rows=[["参数","基线","调优"],
          ["OPEN","1.19","1.19"],["CLOSE","0.84","0.84"],
          ["WIN","20","20"],["WARMUP","64","64"],
          ["FAST_WIN","∞",f"{TUNED['FAST_WIN']:g}"],
          ["FAST_OPEN","∞",f"{TUNED['FAST_OPEN']:g}"],
          ["FAST_CLOSE","∞",f"{TUNED['FAST_CLOSE']:g}"],
          ["总RMSE(含)",f"{base_raw:.3f}",f"{tune_raw:.3f}"],
          ["总RMSE(去)",f"{base_dm:.4f}",f"{tune_dm:.4f}"]]
    tbl=ax4.table(cellText=rows, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.4)
    ax4.set_title('推荐参数', fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'sim_s15v2_tuning.png'), dpi=150); plt.close()

    summary = dict(base=BASE, tuned=TUNED,
                   real_demean=dict(base=base_dm, tuned=tune_dm),
                   real_raw=dict(base=base_raw, tuned=tune_raw),
                   synth=dict(base=[synth_eval(s, BASE) for s in synths],
                              tuned=[synth_eval(s, TUNED) for s in synths]))
    with open(os.path.join(out_dir, 'scheme15v2_tuned_params.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n[完成] 参数已保存 scheme15v2_tuned_params.json")

if __name__ == '__main__':
    main()
