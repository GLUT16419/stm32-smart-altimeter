# -*- coding: utf-8 -*-
"""
方案18 仿真 + 完全调优（基于 serial_tool/data/raw 真实数据）
============================================================
忠实复现固件 (FUSION_SCHEME=18, WORK_MODE=0) 的算法链路：

  MS5611_raw --KF(自适应)--> ms_kf
  BMP280_raw --KF(自适应)--> bmp_kf      (CSV 内 BMP280 已含 bmp_bias 校准偏置)
  fused = w_ms*ms_kf + w_bmp*bmp_kf
  raw_h = PressureToAltitudeWithTemp(fused, PRESET_P0_PA, bmp_temp)
  height = EMA(raw_h)                    (仅 HEIGHT_EMA_ALPHA 影响最终高度)

参考轨迹：各 CSV 的 height_m（设备上报海拔，视作最佳可得真值）。

输出：
  - sim_s18_validation.png    KF 实现校验（仿真 KF vs CSV kf_pressure_pa）
  - sim_s18_静止.png / 平移运动.png / 升降运动.png   三场景波形（默认 vs 调优 vs 参考）
  - sim_s18_tuning.png        P0 灵敏度 + 调优前后 RMSE 对比 + 参数表
  - scheme18_tuned_params.json   推荐参数与指标
"""

import os
import json
import glob
import numpy as np
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

# 采样频率（由数据推断 ~7 Hz，仅用于 x 轴标注）
FS_HZ  = 7.0

# ============== 自适应卡尔曼滤波（与 kalman_filter.c 一致） ==============
class AdaKF:
    def __init__(self, q_base, r, th, qinc, qdec, qmax):
        self.q_base = q_base
        self.r = r
        self.th = th
        self.qinc = qinc
        self.qdec = qdec
        self.qmax = qmax

    def run(self, z):
        z = np.asarray(z, dtype=float)
        n = len(z)
        x = z[0]
        p = 1000.0          # KF_INIT_P
        q = self.q_base
        win = np.zeros(5)
        idx = 0
        out = np.empty(n, dtype=float)
        for i in range(n):
            zi = z[i]
            res = zi - x
            ra = abs(res)
            win[idx] = ra
            idx = (idx + 1) % 5
            m = win.mean()
            rstd = np.sqrt(((win - m) ** 2).mean())
            if rstd > self.th:
                q = min(q * self.qinc, self.qmax)
            else:
                q = max(q * self.qdec, self.q_base)
            p = p + q
            k = p / (p + self.r)
            x = x + k * res
            p = (1.0 - k) * p
            out[i] = x
        return out

# MS5611 / BMP280 自适应阈值（与 kalman_filter.h 一致）
MS_PARAMS  = dict(th=5.0, qinc=1.05, qdec=0.98, qmax=2.0)
BMP_PARAMS = dict(th=1.2, qinc=1.08, qdec=0.97, qmax=5.0)

# ============== 气压 -> 高度（ISA + 温度补偿，与 altitude_convert.c 一致） ==============
def p2a_with_temp(p, p0, t):
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    p0 = float(p0)
    ratio = p / p0
    ratio = np.clip(ratio, 1e-6, 10.0)
    alt = (ISA_T0 / ISA_L) * (1.0 - np.power(ratio, EXP))
    tk = t + 273.15
    dt = tk - (ISA_T0 - ISA_L * alt)
    alt = alt + dt * 0.035
    alt = np.where((alt > -1000.0) & (alt < 20000.0), alt, 0.0)
    return alt

# ============== 加载真实数据 ==============
RAW_DIR = r"e:/ST/QMSX/serial_tool/data/raw"

def pair_files():
    """按时间戳配对 ms5611 / bmp280 CSV，返回数据集列表。"""
    datasets = []
    for scen in ["静止", "平移运动", "升降运动"]:
        bmp_files = sorted(glob.glob(os.path.join(RAW_DIR, scen, "bmp280_*.csv")))
        for bf in bmp_files:
            tag = os.path.basename(bf)[len("bmp280_"):-4]   # e.g. 20260709_130522
            mf = os.path.join(RAW_DIR, scen, f"ms5611_{tag}.csv")
            if os.path.exists(mf):
                datasets.append((scen, tag, mf, bf))
    return datasets

def load_dataset(mf, bf):
    import pandas as pd
    dm = pd.read_csv(mf)
    db = pd.read_csv(bf)
    # 两路 CSV 样本数可能差 1，截断到共同长度对齐
    n = min(len(dm), len(db))
    dm = dm.iloc[:n]; db = db.iloc[:n]
    # 参考高度取 BMP280 的 height_m（更准，且固件高度用其温度）
    return dict(
        scen=None, tag=None,
        ms_raw=dm["pressure_pa"].values.astype(float),
        ms_temp=dm["temperature_c"].values.astype(float),
        ms_kf_csv=dm["kf_pressure_pa"].values.astype(float),
        bmp_raw=db["pressure_pa"].values.astype(float),
        bmp_temp=db["temperature_c"].values.astype(float),
        bmp_kf_csv=db["kf_pressure_pa"].values.astype(float),
        ref_height=db["height_m"].values.astype(float),
        n=n,
    )

# ============== 方案18 前向计算 ==============
def scheme18(ds, P0, w_ms, ms_q, ms_r, bmp_q, bmp_r, h_ema,
             p_ema=0.607815, return_extra=False):
    ms_kf  = AdaKF(ms_q, ms_r, **MS_PARAMS).run(ds["ms_raw"])
    bmp_kf = AdaKF(bmp_q, bmp_r, **BMP_PARAMS).run(ds["bmp_raw"])
    fused = w_ms * ms_kf + (1.0 - w_ms) * bmp_kf

    raw_h = p2a_with_temp(fused, P0, ds["bmp_temp"])

    # 气压 EMA（仅影响气压显示，不影响高度）
    ps = fused[0]; pres_ema = np.empty_like(fused)
    for i in range(len(fused)):
        ps = ps + p_ema * (fused[i] - ps)
        pres_ema[i] = ps

    # 高度 EMA（最终高度输出）
    hs = raw_h[0]; h_ema_out = np.empty_like(raw_h)
    for i in range(len(raw_h)):
        hs = hs + h_ema * (raw_h[i] - hs)
        h_ema_out[i] = hs

    if return_extra:
        return dict(height=h_ema_out, raw_h=raw_h, fused=fused,
                    pres_ema=pres_ema, ms_kf=ms_kf, bmp_kf=bmp_kf)
    return h_ema_out

# 默认（当前固件值）
DEFAULT = dict(
    P0=101325.0, w_ms=0.10,
    ms_q=0.25901, ms_r=23.8569,
    bmp_q=0.0879115, bmp_r=2.82916,
    h_ema=0.946719, p_ema=0.607815,
)

# ============== 指标 ==============
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def dataset_metrics(ds, params):
    sim = scheme18(ds, **params)
    ref = ds["ref_height"]
    # 动态（去均值）RMSE，隔离 P0 偏移影响，专评动态/噪声质量
    dm = rmse(sim - sim.mean(), ref - ref.mean())
    # 含偏移 RMSE（绝对误差，受 P0 影响）
    raw = rmse(sim, ref)
    # 静止段噪声：取前 20% 视为静止（参考高度变化极小）
    nstat = max(10, int(0.2 * len(ref)))
    stat_noise = float(np.std((sim - ref)[nstat//2:nstat]))
    return dict(sim=sim, ref=ref, rmse_raw=raw, rmse_demean=dm, stat_noise=stat_noise)

def obj_raw(datasets, params):
    """含偏移总 RMSE —— 用于校准 P0（方案18 真正要修的系统误差）。"""
    tot = 0.0
    for ds in datasets:
        tot += dataset_metrics(ds, params)["rmse_raw"] ** 2
    return np.sqrt(tot / len(datasets))

def obj_demean(datasets, params):
    """去均值总 RMSE —— 用于评估动态/噪声质量（与 P0 无关）。"""
    tot = 0.0
    for ds in datasets:
        tot += dataset_metrics(ds, params)["rmse_demean"] ** 2
    return np.sqrt(tot / len(datasets))

# ============== 主流程 ==============
def main():
    dsinfo = pair_files()
    print(f"发现 {len(dsinfo)} 个数据集：")
    datasets = []
    for scen, tag, mf, bf in dsinfo:
        ds = load_dataset(mf, bf)
        ds["scen"] = scen; ds["tag"] = tag
        datasets.append(ds)
        print(f"  [{scen}] {tag}: {ds['n']} 样本, 参考高度均值={ds['ref_height'].mean():.2f} m")

    # ---------- 1. KF 实现校验 ----------
    print("\n[1] KF 实现校验（仿真 KF vs CSV kf_pressure_pa）...")
    val_ds = datasets[0]
    ms_kf_sim = AdaKF(DEFAULT["ms_q"], DEFAULT["ms_r"], **MS_PARAMS).run(val_ds["ms_raw"])
    bmp_kf_sim = AdaKF(DEFAULT["bmp_q"], DEFAULT["bmp_r"], **BMP_PARAMS).run(val_ds["bmp_raw"])
    ms_err = rmse(ms_kf_sim, val_ds["ms_kf_csv"])
    bmp_err = rmse(bmp_kf_sim, val_ds["bmp_kf_csv"])
    print(f"  MS5611 KF 仿真 vs CSV  RMSE = {ms_err:.3f} Pa")
    print(f"  BMP280 KF 仿真 vs CSV  RMSE = {bmp_err:.3f} Pa")
    print("  （若偏差小，说明固件记录时 KF 参数与当前 TUNED_* 一致；偏大则说明记录时参数不同，但算法结构相同）")

    # ---------- 2. 默认参数基线 ----------
    print("\n[2] 默认参数基线评估...")
    base_total_demean = obj_demean(datasets, DEFAULT)
    base_per = [dataset_metrics(ds, DEFAULT) for ds in datasets]
    base_total_raw = np.sqrt(np.mean([m["rmse_raw"] ** 2 for m in base_per]))
    print(f"  默认(P0=101325,含偏移) 总RMSE = {base_total_raw:.3f} m  ← 系统误差来源")
    print(f"  默认(去P0偏移)         总RMSE = {base_total_demean:.3f} m  ← 动态/噪声质量")

    # ---------- 3. 完全调优 ----------
    print("\n[3] 完全调优（随机搜索 + 局部精修）...")
    rng = np.random.default_rng(20260711)

    def sample_params():
        return dict(
            P0=    rng.uniform(100300, 101500),
            w_ms=  rng.uniform(0.0, 0.40),
            ms_q=  rng.uniform(0.05, 1.0),
            ms_r=  rng.uniform(5.0, 60.0),
            bmp_q= rng.uniform(0.02, 0.50),
            bmp_r= rng.uniform(0.5, 10.0),
            h_ema= rng.uniform(0.30, 0.98),
            p_ema= rng.uniform(0.30, 0.95),
        )

    # 搜索时对数据降采样 2x 加速
    ds_search = []
    for ds in datasets:
        ds2 = dict(ds)
        for k in ["ms_raw", "bmp_raw", "bmp_temp", "ref_height"]:
            ds2[k] = ds[k][::2]
        ds_search.append(ds2)

    best = None
    best_obj = 1e9
    N_ITER = 700
    for it in range(N_ITER):
        p = sample_params()
        obj = obj_raw(ds_search, p)
        if obj < best_obj:
            best_obj = obj; best = dict(p)
        if (it + 1) % 100 == 0:
            print(f"  迭代 {it+1}/{N_ITER}  当前最优(含偏移)RMSE={best_obj:.4f} m")

    # 局部精修
    print("  局部精修...")
    for _ in range(300):
        p = dict(best)
        p["P0"]    += rng.normal(0, 30)
        p["w_ms"]   = np.clip(best["w_ms"]   + rng.normal(0, 0.03), 0, 0.4)
        p["ms_q"]   *= rng.normal(1, 0.15); p["ms_q"] = np.clip(p["ms_q"], 0.05, 1.0)
        p["ms_r"]   *= rng.normal(1, 0.15); p["ms_r"] = np.clip(p["ms_r"], 5, 60)
        p["bmp_q"]  *= rng.normal(1, 0.15); p["bmp_q"] = np.clip(p["bmp_q"], 0.02, 0.5)
        p["bmp_r"]  *= rng.normal(1, 0.15); p["bmp_r"] = np.clip(p["bmp_r"], 0.5, 10)
        p["h_ema"]  = np.clip(best["h_ema"]  + rng.normal(0, 0.03), 0.3, 0.98)
        p["p_ema"]  = np.clip(best["p_ema"]  + rng.normal(0, 0.05), 0.3, 0.95)
        obj = obj_raw(ds_search, p)
        if obj < best_obj:
            best_obj = obj; best = dict(p)

    # P0 精细扫描（用全量数据，固定最佳动态参数，把绝对偏移压到最小）
    print("  P0 精细扫描...")
    dyn = {k: best[k] for k in best if k != "P0"}
    p0_lo, p0_hi = best["P0"] - 150, best["P0"] + 150
    for p0 in np.linspace(p0_lo, p0_hi, 301):
        o = obj_raw(datasets, {**dyn, "P0": p0})
        if o < best_obj:
            best_obj = o; best["P0"] = p0
    TUNED = best

    # 同时给出“保留双传感器冗余”的均衡配置（w_ms 不低于 0.05）
    TUNED_BAL = dict(TUNED); TUNED_BAL["w_ms"] = max(TUNED["w_ms"], 0.05)

    print(f"  调优后(含偏移)总RMSE = {best_obj:.4f} m")
    print("  推荐参数（最优 / 均衡冗余）:")
    for k in TUNED:
        print(f"    {k}: {TUNED[k]:.6g} / {TUNED_BAL[k]:.6g}")

    # 调优后全量评估
    tuned_per = [dataset_metrics(ds, TUNED) for ds in datasets]
    tuned_bal_per = [dataset_metrics(ds, TUNED_BAL) for ds in datasets]
    tuned_total_raw = np.sqrt(np.mean([m["rmse_raw"] ** 2 for m in tuned_per]))
    tuned_total_demean = np.sqrt(np.mean([m["rmse_demean"] ** 2 for m in tuned_per]))
    tuned_bal_total_raw = np.sqrt(np.mean([m["rmse_raw"] ** 2 for m in tuned_bal_per]))
    tuned_bal_total_demean = np.sqrt(np.mean([m["rmse_demean"] ** 2 for m in tuned_bal_per]))

    # per-dataset P0 一致性检查（用均值偏移反推每数据集所需 P0，验证全局 P0 合理性）
    p0_consistency = []
    for ds in datasets:
        sim = scheme18(ds, P0=TUNED["P0"], **{k: TUNED[k] for k in TUNED if k != "P0"})
        # 用均值偏移判断：单 P0 下各数据集剩余高度偏差 = 真实海拔差异/基准误差
        dh = ds["ref_height"].mean() - sim.mean()      # 需要的高度修正
        p0_consistency.append(dh)
    print(f"  各数据集高度均值偏移(调优P0下): {['%.2f'%x for x in p0_consistency]} m")
    print(f"  （偏移范围 {max(p0_consistency)-min(p0_consistency):.2f} m = 各次采集真实海拔差/基准残差）")

    # ---------- 4. 绘图 ----------
    print("\n[4] 生成波形图...")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    t_axis = lambda n: np.arange(n) / FS_HZ

    # 4a. 校验图
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    x = t_axis(val_ds["n"])
    axes[0].plot(x, val_ds["ms_raw"], 'gray', alpha=0.3, label='MS5611 原始', lw=0.5)
    axes[0].plot(x, ms_kf_sim, 'b-', label=f'仿真 KF (RMSE={ms_err:.2f}Pa)', lw=1.2)
    axes[0].plot(x, val_ds["ms_kf_csv"], 'r--', label='CSV kf_pressure_pa', lw=1.0, alpha=0.8)
    axes[0].set_ylabel('气压 (Pa)'); axes[0].set_title('KF 校验 — MS5611')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].plot(x, val_ds["bmp_raw"], 'gray', alpha=0.3, label='BMP280 原始', lw=0.5)
    axes[1].plot(x, bmp_kf_sim, 'b-', label=f'仿真 KF (RMSE={bmp_err:.2f}Pa)', lw=1.2)
    axes[1].plot(x, val_ds["bmp_kf_csv"], 'r--', label='CSV kf_pressure_pa', lw=1.0, alpha=0.8)
    axes[1].set_ylabel('气压 (Pa)'); axes[1].set_xlabel('时间 (s)')
    axes[1].set_title('KF 校验 — BMP280'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'sim_s18_validation.png'), dpi=150); plt.close()

    # 4b. 三场景波形图（默认 vs 调优 vs 参考）
    scen_map = {"静止": 0, "平移运动": 1, "升降运动": 2}
    for scen in ["静止", "平移运动", "升降运动"]:
        # 取该场景下第一个数据集
        ds = next(d for d in datasets if d["scen"] == scen)
        base_extra = scheme18(ds, **DEFAULT, return_extra=True)
        tune_extra = scheme18(ds, **TUNED, return_extra=True)
        tune_bal_extra = scheme18(ds, **TUNED_BAL, return_extra=True)
        # 默认(含72m偏移)的整体偏移量，用于去偏移后展示其“形状”
        base_off = base_extra["height"].mean() - ds["ref_height"].mean()
        base_dm = base_extra["height"] - base_off
        tune = dict(sim=tune_extra["height"], rmse_raw=rmse(tune_extra["height"], ds["ref_height"]))
        tune_bal = dict(sim=tune_bal_extra["height"], rmse_raw=rmse(tune_bal_extra["height"], ds["ref_height"]))
        x = t_axis(ds["n"])
        fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
        # 气压原始 + KF
        axes[0].plot(x, ds["ms_raw"], 'gray', alpha=0.25, label='MS5611 原始', lw=0.5)
        axes[0].plot(x, ds["bmp_raw"], 'silver', alpha=0.4, label='BMP280 原始', lw=0.5)
        axes[0].plot(x, tune_extra["ms_kf"], 'b-', label='MS5611 KF(调优)', lw=0.9)
        axes[0].plot(x, tune_extra["bmp_kf"], 'c-', label='BMP280 KF(调优)', lw=0.9)
        axes[0].set_ylabel('气压 (Pa)'); axes[0].set_title(f'方案18 仿真 — {scen}  [数据集 {ds["tag"]}]')
        axes[0].legend(fontsize=8, ncol=3); axes[0].grid(alpha=0.3)
        # 融合气压
        axes[1].plot(x, tune_extra["fused"], 'g-', lw=1.1, label='融合气压 (调优)')
        axes[1].plot(x, base_extra["fused"], 'g--', lw=0.8, alpha=0.6, label='融合气压 (默认)')
        axes[1].set_ylabel('融合气压 (Pa)'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        # 高度：参考 vs 调优(最优/均衡) vs 默认(去72m偏移后仅看形状)
        axes[2].plot(x, ds["ref_height"], 'k-', lw=1.3, label='参考高度 (CSV height_m)')
        axes[2].plot(x, base_dm, 'r:', lw=1.0, label=f'默认(去72m偏移, 仅看形状)')
        axes[2].plot(x, tune["sim"], 'b-', lw=1.1, label=f'调优最优 (RMSE={tune["rmse_raw"]:.2f}m)')
        axes[2].plot(x, tune_bal["sim"], 'g--', lw=1.0, label=f'调优均衡 (RMSE={tune_bal["rmse_raw"]:.2f}m)')
        axes[2].set_ylabel('高度 (m)'); axes[2].legend(fontsize=8, ncol=2); axes[2].grid(alpha=0.3)
        # 残差（调优 - 参考）
        axes[3].plot(x, tune["sim"] - ds["ref_height"], 'b-', lw=0.8,
                     label=f'最优残差 std={np.std(tune["sim"]-ds["ref_height"]):.3f}m')
        axes[3].plot(x, tune_bal["sim"] - ds["ref_height"], 'g--', lw=0.8,
                     label=f'均衡残差 std={np.std(tune_bal["sim"]-ds["ref_height"]):.3f}m')
        axes[3].axhline(0, color='k', lw=0.8, alpha=0.4)
        axes[3].set_ylabel('残差 (m)'); axes[3].set_xlabel('时间 (s)')
        axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)
        plt.tight_layout()
        fname = f'sim_s18_{scen}.png'.replace('平移运动', '平移').replace('升降运动', '升降')
        plt.savefig(os.path.join(out_dir, fname), dpi=150); plt.close()
        print(f"  已保存: {fname}")

    # 4c. 调优结果图（P0 灵敏度 + RMSE 对比 + 参数表）
    fig = plt.figure(figsize=(14, 10))
    # P0 灵敏度（固定调优的动态参数，仅扫描 P0）
    dyn = {k: TUNED[k] for k in TUNED if k != "P0"}
    p0_grid = np.linspace(100300, 101500, 120)
    p0_err = []
    for p0 in p0_grid:
        e = np.sqrt(np.mean([rmse(scheme18(ds, P0=p0, **dyn), ds["ref_height"]) ** 2 for ds in datasets]))
        p0_err.append(e)
    p0_err = np.array(p0_err)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(p0_grid, p0_err, 'b-', lw=1.5)
    ax1.axvline(TUNED["P0"], color='r', ls='--', label=f'调优 P0={TUNED["P0"]:.1f}')
    ax1.set_xlabel('PRESET_P0_PA (Pa)'); ax1.set_ylabel('总 RMSE (m)')
    ax1.set_title('P0 灵敏度'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # RMSE 对比（去均值，体现动态质量）：默认 / 调优最优 / 调优均衡
    ax2 = fig.add_subplot(2, 2, 2)
    labels = [f'{d["scen"]}\n{d["tag"][-6:]}' for d in datasets]
    base_dm = [dataset_metrics(d, DEFAULT)["rmse_demean"] for d in datasets]
    tune_dm = [dataset_metrics(d, TUNED)["rmse_demean"] for d in datasets]
    tune_bal_dm = [dataset_metrics(d, TUNED_BAL)["rmse_demean"] for d in datasets]
    xx = np.arange(len(datasets))
    w = 0.26
    ax2.bar(xx - w, base_dm, w, label='默认', color='orange')
    ax2.bar(xx,     tune_dm, w, label='调优最优', color='steelblue')
    ax2.bar(xx + w, tune_bal_dm, w, label='调优均衡', color='green')
    ax2.set_xticks(xx); ax2.set_xticklabels(labels, fontsize=6)
    ax2.set_ylabel('去均值 RMSE (m)'); ax2.set_title('动态跟踪误差（去P0偏移）')
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3, axis='y')

    # 静止噪声对比
    ax3 = fig.add_subplot(2, 2, 3)
    base_sn = [dataset_metrics(d, DEFAULT)["stat_noise"] for d in datasets]
    tune_sn = [dataset_metrics(d, TUNED)["stat_noise"] for d in datasets]
    tune_bal_sn = [dataset_metrics(d, TUNED_BAL)["stat_noise"] for d in datasets]
    ax3.bar(xx - w, base_sn, w, label='默认', color='orange')
    ax3.bar(xx,     tune_sn, w, label='调优最优', color='steelblue')
    ax3.bar(xx + w, tune_bal_sn, w, label='调优均衡', color='green')
    ax3.set_xticks(xx); ax3.set_xticklabels(labels, fontsize=6)
    ax3.set_ylabel('静止段残差 std (m)'); ax3.set_title('静止稳定性（越小越好）')
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3, axis='y')

    # 参数表
    ax4 = fig.add_subplot(2, 2, 4); ax4.axis('off')
    rows = [["参数", "默认", "调优最优", "调优均衡"]]
    name_map = [("PRESET_P0_PA","P0"),("w_ms (MS5611权重)","w_ms"),
                ("MS5611 KF Q","ms_q"),("MS5611 KF R","ms_r"),
                ("BMP280 KF Q","bmp_q"),("BMP280 KF R","bmp_r"),
                ("HEIGHT_EMA_ALPHA","h_ema"),("PRESSURE_EMA_ALPHA","p_ema")]
    for cn, key in name_map:
        rows.append([cn, f"{DEFAULT[key]:.5g}", f"{TUNED[key]:.5g}", f"{TUNED_BAL[key]:.5g}"])
    rows.append(["总RMSE(含偏移)", f"{base_total_raw:.3f} m", f"{tuned_total_raw:.3f} m", f"{tuned_bal_total_raw:.3f} m"])
    rows.append(["总RMSE(去均值)", f"{base_total_demean:.4f} m", f"{tuned_total_demean:.4f} m", f"{tuned_bal_total_demean:.4f} m"])
    tbl = ax4.table(cellText=rows, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.35)
    ax4.set_title('推荐参数与效果对比', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sim_s18_tuning.png'), dpi=150); plt.close()
    print("  已保存: sim_s18_tuning.png")

    # ---------- 5. 保存推荐参数 ----------
    summary = dict(
        default=DEFAULT,
        tuned=TUNED,
        tuned_balanced=TUNED_BAL,
        metrics=dict(
            base_total_raw_rmse=base_total_raw,
            base_total_demean_rmse=base_total_demean,
            tuned_total_raw_rmse=tuned_total_raw,
            tuned_total_demean_rmse=tuned_total_demean,
            tuned_bal_total_raw_rmse=tuned_bal_total_raw,
            tuned_bal_total_demean_rmse=tuned_bal_total_demean,
            per_dataset=dict(
                base_raw=[dataset_metrics(d, DEFAULT)["rmse_raw"] for d in datasets],
                tuned_raw=[dataset_metrics(d, TUNED)["rmse_raw"] for d in datasets],
                tuned_bal_raw=[dataset_metrics(d, TUNED_BAL)["rmse_raw"] for d in datasets],
                base_demean=[dataset_metrics(d, DEFAULT)["rmse_demean"] for d in datasets],
                tuned_demean=[dataset_metrics(d, TUNED)["rmse_demean"] for d in datasets],
                tuned_bal_demean=[dataset_metrics(d, TUNED_BAL)["rmse_demean"] for d in datasets],
                stat_noise_base=[dataset_metrics(d, DEFAULT)["stat_noise"] for d in datasets],
                stat_noise_tuned=[dataset_metrics(d, TUNED)["stat_noise"] for d in datasets],
                stat_noise_tuned_bal=[dataset_metrics(d, TUNED_BAL)["stat_noise"] for d in datasets],
            ),
            p0_height_offset_consistency=p0_consistency,
        ),
        notes=dict(
            kf_validation_ms_rmse_pa=ms_err,
            kf_validation_bmp_rmse_pa=bmp_err,
            bmp_bias_note="CSV 中 BMP280 pressure_pa/kf_pressure_pa 已含 bmp_bias 校准偏置；直接对两路 CSV 各跑 KF 再融合即可精确复现固件融合结果。",
            reference="各 CSV 的 height_m 作为参考真值（设备上报海拔）。",
            p0_interpretation="PRESET_P0_PA 应设为采集地实际海平面气压（本次约 %.1f Pa），而非标准 101325 Pa；后者会造成约 72 m 系统偏移。" % TUNED["P0"],
        ),
    )
    with open(os.path.join(out_dir, 'scheme18_tuned_params.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n[完成] 推荐参数已保存至 scheme18_tuned_params.json")

if __name__ == '__main__':
    main()
