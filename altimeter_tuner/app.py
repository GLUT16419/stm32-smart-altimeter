# -*- coding: utf-8 -*-
"""
高度融合算法调参仿真器 (GUI)
============================
用途：在 PC 端复现嵌入式固件的气压高度融合算法，借助
      静止 / 平移 / 升降 数据集进行可视化调参。

依赖：Python 3.8+, numpy, matplotlib   (可选 tflite_runtime 启用 NN 滤波)
运行：python app.py
"""

import os
import sys
import json
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 允许从本目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithm import AlgoParams, simulate, export_header_snippet
from datasets import (
    generate_synthetic, load_dataset, save_synthetic_csv, dataset_to_rows, SCENARIOS
)
from metrics import compute_metrics, format_metrics

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 真实采集数据目录（serial_tool 录制的双文件格式）
RAW_REAL_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'serial_tool', 'data', 'raw'))

# 场景识别模型默认路径（由 train_scene_classifier.py 训练生成）
DEFAULT_SCENE_MODEL = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'models', 'scene_classifier.json'))


# ============================================================
# 参数控件规格：(分组, 标签, 属性名, 最小值, 最大值, 步长)
# ============================================================
PARAM_SPEC = [
    ('校准', '参考气压(Pa, 0=自动)', 'ref_pressure', 90000, 105000, 1),
    ('校准', 'BMP280偏置(Pa, 0=自动)', 'bmp_bias', -50, 50, 0.1),
    ('校准', '校准样本数', 'calib_samples', 5, 300, 1),

    ('MS5611 KF', 'Q_base', 'ms_q', 0.001, 2.0, 0.001),
    ('MS5611 KF', 'R (方差)', 'ms_r', 0.1, 50.0, 0.1),
    ('MS5611 KF', '残差阈值', 'ms_residual_th', 0.5, 20.0, 0.1),
    ('MS5611 KF', 'Q增大系数', 'ms_q_inc', 1.0, 1.5, 0.01),
    ('MS5611 KF', 'Q衰减系数', 'ms_q_dec', 0.9, 1.0, 0.01),
    ('MS5611 KF', 'Q上限', 'ms_q_max', 0.5, 5.0, 0.1),

    ('BMP280 KF', 'Q_base', 'bmp_q', 0.0005, 2.0, 0.0005),
    ('BMP280 KF', 'R (方差)', 'bmp_r', 0.01, 10.0, 0.01),
    ('BMP280 KF', '残差阈值', 'bmp_residual_th', 0.1, 10.0, 0.1),
    ('BMP280 KF', 'Q增大系数', 'bmp_q_inc', 1.0, 1.5, 0.01),
    ('BMP280 KF', 'Q衰减系数', 'bmp_q_dec', 0.9, 1.0, 0.01),
    ('BMP280 KF', 'Q上限', 'bmp_q_max', 0.5, 10.0, 0.1),

    ('融合', '融合方案(1-14)', 'fusion_scheme', 1, 14, 1),
    ('融合', '权重 MS5611', 'w_ms', 0.0, 1.0, 0.01),
    ('融合', '权重 BMP280', 'w_bmp', 0.0, 1.0, 0.01),
    ('融合', 'HPF_ALPHA', 'hpf_alpha', 0.01, 1.0, 0.01),
    ('融合', '运动阈值(Pa)', 'motion_threshold', 0.2, 10.0, 0.1),
    ('融合', '静止权重MS', 'weight_static_ms', 0.0, 1.0, 0.01),
    ('融合', '运动权重MS', 'weight_motion_ms', 0.0, 1.0, 0.01),
    ('融合', '权重平滑α', 'weight_smooth_alpha', 0.01, 0.5, 0.01),
    ('融合', 'Δ静止权重MS', 'w_delta_static_ms', 0.0, 1.0, 0.01),
    ('融合', 'Δ运动权重MS', 'w_delta_motion_ms', 0.0, 1.0, 0.01),
    ('融合', '逆方差ε', 'ivar_epsilon', 0.01, 2.0, 0.01),
    ('融合', '锚定α', 'anchor_alpha', 0.0, 0.5, 0.01),
    ('融合', '二阶α', 'comp_alpha', 0.0, 0.5, 0.01),
    ('融合', '二阶β', 'comp_beta', 0.0, 1.0, 0.01),
    ('融合', '温漂系数(Pa/℃)', 'tc_coeff', 0.0, 5.0, 0.05),

    ('平滑/其它', '气压EMAα', 'pressure_ema_alpha', 0.0, 1.0, 0.01),
    ('平滑/其它', '高度EMAα', 'height_ema_alpha', 0.0, 1.0, 0.01),
]


class TunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title('高度融合算法调参仿真器')
        self.root.geometry('1280x800')

        self.params = AlgoParams()
        self.controls = {}          # attr -> tk.DoubleVar
        self.spinboxes = {}         # attr -> ttk.Spinbox（读取时直接取控件文本，避免 textvariable 同步时机问题）
        self.current_ds = None      # 当前(单/显示)数据集
        self.ghost = None           # 对比用上一次 fused_height
        self.ghost_label = None
        self.t = None               # 当前时间轴
        self._scrolling = False     # 防止 xlim 回调递归
        self._drag = None           # 鼠标拖拽平移状态
        self.window_samples = 100   # 波形显示窗口宽度(样本数)，超出部分用滚动条左右查看
        self.ax1b = None            # MS5611 气压 twin 轴
        self.ax2b = None            # BMP280 气压 twin 轴
        self.ax3b = None            # 融合气压 twin 轴
        self._auto_running = False  # 自动调参进行中

        # 三数据集模式：静止 / 平移 / 升降（输入顺序固定）
        self.three_keys = ['static', 'translation', 'elevation']
        self.three_labels = {'static': '静止', 'translation': '平移', 'elevation': '升降'}
        self.mode = 'single'        # 'single' | 'three'
        # 每个场景 key 下可挂多个『批次』（例如真实数据目录下有多对采集），
        # 故 self.datasets[key] 存为 list[dict]；current_idx[key] 为当前查看批次索引。
        self.datasets = {}          # key -> list[dict]（三数据集模式）
        self.current_idx = {k: 0 for k in self.three_keys}
        self.view_key = 'elevation' # 三数据集模式下当前查看的数据集（默认升降，便于直观看到算法跟踪）
        self.three_sel_vars = {k: tk.StringVar(value=k) for k in self.three_keys}
        self.three_path_vars = {k: tk.StringVar(value='') for k in self.three_keys}
        self.batch_vars = {}        # key -> tk.StringVar（批次下拉框当前选择）
        self.batch_combos = {}      # key -> ttk.Combobox（批次下拉框控件）

        self._build_layout()
        self._build_param_panel()
        self._apply_params_to_controls()
        self._load_scenario('elevation')

    # ---------------- 布局 ----------------
    def _build_layout(self):
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        # ---- 第一排：模式 + 动作按钮 ----
        row1 = ttk.Frame(top)
        row1.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row1, text='模式:').pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value='single')
        ttk.Radiobutton(row1, text='单数据集', variable=self.mode_var, value='single',
                        command=self._on_mode_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(row1, text='三数据集(静止/平移/升降)', variable=self.mode_var,
                        value='three', command=self._on_mode_change).pack(side=tk.LEFT, padx=2)

        # 单数据集模式：数据集选择
        self.single_frame = ttk.Frame(row1)
        self.single_frame.pack(side=tk.LEFT, padx=6)
        ttk.Label(self.single_frame, text='数据集:').pack(side=tk.LEFT)
        self.scenario_var = tk.StringVar(value='elevation')
        self.scenario_combo = ttk.Combobox(self.single_frame, textvariable=self.scenario_var,
                                           values=SCENARIOS + ['从文件(合并CSV)', '从文件(双文件)'],
                                           width=18, state='readonly')
        self.scenario_combo.pack(side=tk.LEFT, padx=2)
        self.scenario_combo.bind('<<ComboboxSelected>>', self._on_scenario)

        ttk.Button(row1, text='运行 ▶', command=self._on_run).pack(side=tk.LEFT, padx=4)
        self.auto_btn = ttk.Button(row1, text='自动调参', command=self._on_auto_tune)
        self.auto_btn.pack(side=tk.LEFT, padx=4)
        self.ghost_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text='对比(保留上次)', variable=self.ghost_var).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text='导出参数数据', command=self._export_params).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text='导出图片', command=self._export_fig).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text='重置参数', command=self._reset_params).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text='生成示例CSV', command=self._gen_samples).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text='加载真实数据(raw)', command=self._load_raw_real).pack(side=tk.LEFT, padx=2)

        # 场景识别（滤波的同时额外输出当前运动场景，辅助判断）
        self.scene_enabled_var = tk.BooleanVar(value=False)
        self.scene_model_var = tk.StringVar(value=DEFAULT_SCENE_MODEL)
        sf = ttk.Frame(row1)
        sf.pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(sf, text='场景识别', variable=self.scene_enabled_var,
                        command=self._on_scene_toggle).pack(side=tk.LEFT)
        ttk.Entry(sf, textvariable=self.scene_model_var, width=22).pack(side=tk.LEFT, padx=2)
        ttk.Button(sf, text='浏览', command=self._on_scene_browse).pack(side=tk.LEFT)

        # ---- 第二排：三数据集选择（输入顺序：静止 -> 平移 -> 升降）----
        self.three_frame = ttk.LabelFrame(top, text='三数据集输入（先静止，后平移，后升降）')
        self.three_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        self.three_view_var = tk.StringVar(value='static')
        for k in self.three_keys:  # 固定顺序：静止 / 平移 / 升降
            f = ttk.Frame(self.three_frame)
            f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=2)
            ttk.Label(f, text=f'{self.three_labels[k]}({k}):',
                      width=12, anchor='w').pack(side=tk.TOP)
            cb = ttk.Combobox(f, textvariable=self.three_sel_vars[k],
                              values=SCENARIOS + ['从文件(合并CSV)', '从文件(双文件)'],
                              width=16, state='readonly')
            cb.pack(side=tk.TOP, fill=tk.X)
            cb.bind('<<ComboboxSelected>>', lambda e, kk=k: self._on_three_select(kk))
            ttk.Button(f, text='选择文件…',
                       command=lambda kk=k: self._on_three_file(kk)).pack(side=tk.TOP, pady=1)
            # 批次下拉框：每个场景可能加载了多批真实数据（目录内多对采集），在此切换
            ttk.Label(f, text='批次:').pack(side=tk.TOP, anchor='w')
            self.batch_vars[k] = tk.StringVar()
            bcb = ttk.Combobox(f, textvariable=self.batch_vars[k], width=16, state='readonly')
            bcb.pack(side=tk.TOP, fill=tk.X)
            bcb.bind('<<ComboboxSelected>>', lambda e, kk=k: self._on_batch_change(kk))
            self.batch_combos[k] = bcb
            ttk.Radiobutton(self.three_frame, text='查看', variable=self.three_view_var,
                            value=k, command=self._on_three_view).pack(side=tk.LEFT, padx=2)
        self.three_frame.pack_forget()  # 默认单数据集模式，隐藏

        # 主区域：左参数 / 右图+指标
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 左：参数面板（带滚动）
        left = ttk.Frame(main, width=320)
        self.param_canvas = tk.Canvas(left)
        self.param_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.param_canvas.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.param_canvas.configure(yscrollcommand=vsb.set)
        self.param_inner = ttk.Frame(self.param_canvas)
        self.param_canvas.create_window((0, 0), window=self.param_inner, anchor='nw')
        self.param_inner.bind('<Configure>', lambda e: self.param_canvas.configure(
            scrollregion=self.param_canvas.bbox('all')))
        main.add(left, weight=1)

        # 右：图 + 时间滚动条 + 指标
        right = ttk.Frame(main)
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 9))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 窗口宽度选择 + 样本滚动条（左右移动波形）
        winrow = ttk.Frame(right)
        winrow.pack(side=tk.TOP, fill=tk.X, padx=4)
        ttk.Label(winrow, text='显示样本数:').pack(side=tk.LEFT)
        self.window_var = tk.StringVar(value=str(int(self.window_samples)))
        win_combo = ttk.Combobox(winrow, textvariable=self.window_var, width=6,
                                 state='readonly',
                                 values=['50', '100', '200', '500', '1000', '全部'])
        win_combo.pack(side=tk.LEFT, padx=4)
        win_combo.bind('<<ComboboxSelected>>', self._on_window_change)
        ttk.Label(winrow, text='  ← 左右拖动下方滚动条查看后续样本').pack(side=tk.LEFT)

        self.scroll_var = tk.DoubleVar(value=0.0)
        sbar = ttk.Scale(right, orient=tk.HORIZONTAL, from_=0, to=1000,
                         variable=self.scroll_var, command=self._on_scroll)
        sbar.pack(side=tk.TOP, fill=tk.X, padx=4)
        ttk.Label(right, text='样本滚动（也可左键直接拖动波形左右滑动；工具栏放大镜可自由缩放）').pack(side=tk.TOP)

        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 鼠标左键拖拽平移波形（直接拖动，无需先点工具栏平移）
        # 用 matplotlib 原生事件，事件携带 event.xdata（数据坐标系的样本序号），
        # 比 Tk 像素坐标 + get_window_extent 换算更可靠
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

        self.metrics_box = scrolledtext.ScrolledText(right, height=9, font=('Consolas', 9))
        self.metrics_box.pack(side=tk.BOTTOM, fill=tk.X)
        main.add(right, weight=3)

        # 三个子图共用同一 x 轴范围（缩放/平移/滚动保持同步）
        for ax in (self.ax1, self.ax2, self.ax3):
            ax.callbacks.connect('xlim_changed', self._on_xlim_changed)

        # 关闭窗口：彻底退出进程，避免任务管理器残留
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # ---------------- 彻底退出 ----------------
    def _on_close(self):
        # 先停止后台自动调参，避免主线程卡在循环里
        self._auto_running = False
        try:
            self.root.quit()
        except Exception:
            pass
        try:
            if getattr(self, 'fig', None) is not None:
                plt.close(self.fig)
            plt.close('all')
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        # 强制结束进程，清理 matplotlib 可能遗留的后台线程
        os._exit(0)

    # ---------------- 参数控件 ----------------
    def _build_param_panel(self):
        groups = {}
        for spec in PARAM_SPEC:
            grp = spec[0]
            groups.setdefault(grp, []).append(spec)
        for grp, specs in groups.items():
            frm = ttk.LabelFrame(self.param_inner, text=grp)
            frm.pack(fill=tk.X, padx=4, pady=3)
            for (_g, label, attr, lo, hi, step) in specs:
                row = ttk.Frame(frm)
                row.pack(fill=tk.X, padx=2, pady=1)
                ttk.Label(row, text=label, width=18, anchor='w').pack(side=tk.LEFT)
                var = tk.DoubleVar()
                self.controls[attr] = var
                # 用 Spinbox 兼顾范围与精度
                decimals = max(0, int(round(-np.log10(step)))) if step < 1 else 0
                fmt = f'%.{decimals}f'
                sb = ttk.Spinbox(row, from_=lo, to=hi, increment=step, textvariable=var,
                                width=10, format=fmt)
                sb.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.spinboxes[attr] = sb      # 保存控件，读取参数时直接取文本

    def _apply_params_to_controls(self):
        for attr, var in self.controls.items():
            val = getattr(self.params, attr)
            if val is None:
                var.set(0.0)
            else:
                var.set(float(val))

    def _read_controls_to_params(self):
        """将参数控件当前值写入 self.params。

        注意：直接读取 ttk.Spinbox 控件的文本（self.spinboxes），而不是依赖
        textvariable 的自动同步。因为 ttk.Spinbox 在用户键盘输入后，textvariable
        往往要等到失焦/回车才更新；若直接读 var.get() 会拿到旧值，导致『改了数字
        点运行没变化』。这里用控件自身的 .get() 取当前显示文本，最可靠。
        """
        for attr, var in self.controls.items():
            sb = self.spinboxes.get(attr)
            raw = sb.get() if sb is not None else var.get()
            try:
                setattr(self.params, attr, float(raw))
            except (ValueError, TypeError):
                # 输入非法（空/非数字）时保留原值
                pass
        # 场景识别开关与模型路径（非浮点参数，单独处理）
        self.params.use_scene_clf = bool(self.scene_enabled_var.get())
        mp = self.scene_model_var.get().strip()
        self.params.scene_model = mp if mp else None

    def _on_scene_toggle(self):
        """勾选/取消场景识别后，立即重跑以刷新场景显示。"""
        if self.current_ds is not None:
            self._run()

    def _on_scene_browse(self):
        path = filedialog.askopenfilename(title='选择场景识别模型',
                                          filetypes=[('JSON', '*.json')])
        if path:
            self.scene_model_var.set(path)
            if self.current_ds is not None:
                self._run()

    def _reset_params(self):
        self.params = AlgoParams()
        self._apply_params_to_controls()
        self.ghost = None
        messagebox.showinfo('提示', '参数已重置为固件默认值。')

    # ---------------- 数据集 ----------------
    def _on_mode_change(self):
        self.mode = self.mode_var.get()
        if self.mode == 'single':
            self.three_frame.pack_forget()
            self.single_frame.pack(side=tk.LEFT, padx=6)
            self._load_scenario(self.scenario_var.get() if self.scenario_var.get() in SCENARIOS
                                else 'static')
        else:
            self.single_frame.pack_forget()
            self.three_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
            self._init_three_datasets()

    def _init_three_datasets(self):
        """三数据集模式：默认载入三个合成场景（每场景 1 个批次），再刷新当前查看。
        已载入的真实数据会被保留（datasets[k] 已是 list 时不重建）。"""
        for k in self.three_keys:
            if k not in self.datasets or not isinstance(self.datasets.get(k), list):
                self.datasets[k] = [generate_synthetic(scenario=k, n_samples=400, fs=10.0)]
                self.three_sel_vars[k].set(k)
                self.three_path_vars[k].set('')
                self._update_batch_combo(k, [f'合成-{self.three_labels[k]}'], 0)
            else:
                # 已有数据（如真实多批次）：仅确保索引合法
                self.current_idx[k] = min(self.current_idx[k], len(self.datasets[k]) - 1)
        self.view_key = 'elevation'
        self.three_view_var.set('elevation')
        self.current_ds = self.datasets[self.view_key][self.current_idx[self.view_key]]
        self._run()

    def _load_raw_real(self):
        """把 serial_tool/data/raw 下的 静止/平移运动/升降运动 三个子目录
        作为真实数据集载入三数据集模式（按 sample_id 配对的双文件）。

        关键改进：每个场景目录内可能有多对采集（如 ms5611_130522 + bmp280_130522、
        ms5611_132144 + bmp280_132144），本函数会**加载目录下全部配对**，每一对作为
        一个独立批次，可通过各场景的『批次』下拉框切换查看。"""
        mapping = {'静止': 'static', '平移运动': 'translation', '升降运动': 'elevation'}
        base = RAW_REAL_DIR
        if not os.path.isdir(base):
            messagebox.showwarning('未找到', f'真实数据目录不存在: {base}')
            return
        loaded = []
        for cn, key in mapping.items():
            d = os.path.join(base, cn)
            if not os.path.isdir(d):
                continue
            pairs = self._find_all_real_pairs(d)
            if not pairs:
                continue
            ds_list, labels = [], []
            for (ms_p, bmp_p) in pairs:
                tok = os.path.basename(ms_p)[len('ms5611_'):-4]  # 时间戳，如 20260709_130522
                try:
                    ds = load_dataset(ms_p, bmp_p)
                except Exception as e:
                    messagebox.showerror('读取失败', f'{cn}/{os.path.basename(ms_p)}: {e}')
                    continue
                ds['scenario'] = key
                ds_list.append(ds)
                labels.append(f'{cn}-{tok}')
            if not ds_list:
                continue
            self.datasets[key] = ds_list
            self.current_idx[key] = 0
            self.three_sel_vars[key].set('真实数据')
            self.three_path_vars[key].set('; '.join(labels))
            self._update_batch_combo(key, labels, 0)
            loaded.append(key)
        if not loaded:
            messagebox.showwarning('未找到', 'raw 目录下未找到可用的真实数据对 (ms5611_*.csv + bmp280_*.csv)')
            return
        # 直接进入三数据集模式（不调用 _on_mode_change，避免被合成数据覆盖）
        self.mode = 'three'
        self.mode_var.set('three')
        self.single_frame.pack_forget()
        self.three_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        self.view_key = 'elevation'
        self.three_view_var.set('elevation')
        self.current_idx['elevation'] = min(self.current_idx.get('elevation', 0),
                                            len(self.datasets['elevation']) - 1)
        self.current_ds = self.datasets['elevation'][self.current_idx['elevation']]
        self._run()

    @staticmethod
    def _find_all_real_pairs(d):
        """返回目录 d 内所有 (ms5611_*.csv, bmp280_*.csv) 配对列表（按时间戳排序）。
        此前只配对第一对会漏掉其余批次，这里返回全部配对。"""
        files = sorted(os.listdir(d))
        ms = [f for f in files if f.startswith('ms5611') and f.lower().endswith('.csv')]
        bm = [f for f in files if f.startswith('bmp280') and f.lower().endswith('.csv')]
        if not ms or not bm:
            return []

        def tok(f):
            name = f[:-4] if f.lower().endswith('.csv') else f
            return name.split('_', 1)[1] if '_' in name else name
        ms_by = {tok(f): f for f in ms}
        pairs = []
        for f in bm:
            t = tok(f)
            if t in ms_by:
                pairs.append((os.path.join(d, ms_by[t]), os.path.join(d, f)))
        return pairs

    def _on_three_select(self, key):
        sel = self.three_sel_vars[key].get()
        if sel in SCENARIOS:
            self.datasets[key] = [generate_synthetic(scenario=sel, n_samples=400, fs=10.0)]
            self.three_path_vars[key].set('')
            self.current_idx[key] = 0
            self._update_batch_combo(key, [f'合成-{self.three_labels[key]}'], 0)
            if self.view_key == key:
                self.current_ds = self.datasets[key][0]
                self._run()

    def _on_three_file(self, key):
        # 合并 CSV
        path = filedialog.askopenfilename(title=f'选择 {self.three_labels[key]} 数据集 CSV',
                                          filetypes=[('CSV', '*.csv')])
        if not path:
            return
        try:
            ds = load_dataset(path)
        except Exception as e:
            messagebox.showerror('读取失败', str(e))
            return
        ds['scenario'] = self.three_sel_vars[key].get()
        self.datasets[key] = [ds]
        self.current_idx[key] = 0
        self.three_sel_vars[key].set('从文件(合并CSV)')
        self.three_path_vars[key].set(os.path.basename(path))
        self._update_batch_combo(key, [os.path.basename(path)], 0)
        if self.view_key == key:
            self.current_ds = ds
            self._run()

    def _on_three_view(self):
        self.view_key = self.three_view_var.get()
        if self.mode == 'three' and self.view_key in self.datasets:
            idx = min(self.current_idx.get(self.view_key, 0), len(self.datasets[self.view_key]) - 1)
            self.current_ds = self.datasets[self.view_key][idx]
            self._run()

    def _update_batch_combo(self, key, labels, idx=0):
        """刷新某场景的批次下拉框（labels 为各批次显示名）。"""
        cb = self.batch_combos.get(key)
        if cb is None:
            return
        cb.configure(values=labels)
        if labels:
            idx = max(0, min(idx, len(labels) - 1))
            self.batch_vars[key].set(labels[idx])
            self.current_idx[key] = idx
        else:
            self.batch_vars[key].set('')
            self.current_idx[key] = 0

    def _on_batch_change(self, key):
        """在批次下拉框切换当前查看的真实数据批次。"""
        cb = self.batch_combos.get(key)
        if cb is None:
            return
        labels = list(cb['values']) if cb['values'] else []
        sel = self.batch_vars[key].get()
        if sel in labels:
            self.current_idx[key] = labels.index(sel)
        if self.mode == 'three' and self.view_key == key:
            idx = min(self.current_idx[key], len(self.datasets[key]) - 1)
            self.current_ds = self.datasets[key][idx]
            self._run()

    def _on_scenario(self, event=None):
        sel = self.scenario_var.get()
        if sel in SCENARIOS:
            self._load_scenario(sel)
        elif sel == '从文件(合并CSV)':
            path = filedialog.askopenfilename(title='选择合并 CSV', filetypes=[('CSV', '*.csv')])
            if path:
                self.current_ds = load_dataset(path)
                self._run()
        elif sel == '从文件(双文件)':
            ms = filedialog.askopenfilename(title='选择 MS5611 CSV', filetypes=[('CSV', '*.csv')])
            if not ms:
                return
            bmp = filedialog.askopenfilename(title='选择 BMP280 CSV', filetypes=[('CSV', '*.csv')])
            if bmp:
                self.current_ds = load_dataset(ms, bmp)
                self._run()

    def _load_scenario(self, scenario):
        self.current_ds = generate_synthetic(scenario=scenario, n_samples=400, fs=10.0)
        self._run()

    # ---------------- 运行 ----------------
    def _on_run(self):
        self._run()

    def _run(self):
        if self.current_ds is None:
            return
        self._read_controls_to_params()
        ds = self.current_ds
        try:
            # 关键：仿真只以『气压(及温度)』为输入，绝不会把高度作为算法输入。
            res = simulate(ds['ms_pressure'], ds['bmp_pressure'], ds['temperature'],
                           self.params)
        except Exception as e:
            messagebox.showerror('运行出错', str(e))
            return

        # true_height 仅用于下方对比显示与指标计算，绝不作为算法输入。
        res['truth'] = ds['true_height']

        # 真实数据：算法高度以 BMP 参考气压为基准(相对)，而固件高度是绝对海拔再
        # 相对各自起点。两者存在约数十米的常值基准差(即传感器偏置)，对跟踪评估无意义。
        # 因此把算法各序列按『前 K 个样本均值』对齐到固件相对起点，使曲线可直接对比。
        fw_ms = ds.get('ms_fw_height')
        fw_bmp = ds.get('bmp_fw_height')
        if fw_ms is not None and fw_bmp is not None:
            fw_ms_ref = fw_ms - fw_ms[0]
            fw_bmp_ref = fw_bmp - fw_bmp[0]
            K = max(1, min(50, len(ds['time']) // 10))
            off_ms = float(np.mean(res['raw_height_ms'][:K]) - np.mean(fw_ms_ref[:K]))
            off_bmp = float(np.mean(res['raw_height_bmp'][:K]) - np.mean(fw_bmp_ref[:K]))
            off_fused = float(np.mean(res['fused_height'][:K]) - np.mean(fw_ms_ref[:K]))
            res['raw_height_ms'] = res['raw_height_ms'] - off_ms
            res['kf_height_ms'] = res['kf_height_ms'] - off_ms
            res['raw_height_bmp'] = res['raw_height_bmp'] - off_bmp
            res['kf_height_bmp'] = res['kf_height_bmp'] - off_bmp
            res['fused_height'] = res['fused_height'] - off_fused
            res['fw_ms_ref'] = fw_ms_ref
            res['fw_bmp_ref'] = fw_bmp_ref
        else:
            res['fw_ms_ref'] = None
            res['fw_bmp_ref'] = None

        self._last_res = res
        self._plot(res, ds)

        # 指标
        m = compute_metrics(ds['time'], res['fused_height'], ds['true_height'], ds['scenario'])
        self.metrics_box.delete('1.0', tk.END)
        if self.mode == 'three':
            self.metrics_box.insert(tk.END,
                f"[三数据集模式] 当前查看: {self.three_labels[self.view_key]}({self.view_key})\n")
        self.metrics_box.insert(tk.END,
            f"场景: {ds['scenario']}   参考气压: {res['ref_pressure']:.2f} Pa   "
            f"BMP280偏置: {res['bmp_bias']:.3f} Pa\n")
        self.metrics_box.insert(tk.END, '-' * 50 + '\n')
        self.metrics_box.insert(tk.END, format_metrics(m))
        if ds['true_height'] is None and res['fw_ms_ref'] is not None:
            # 真实数据无真值：以固件高度(相对)作对比参考
            m_fw = compute_metrics(ds['time'], res['fused_height'],
                                   res['fw_ms_ref'], ds['scenario'])
            self.metrics_box.insert(tk.END,
                f"\n[真实数据] 无真值，已用『固件MS5611高度(相对)』作对比参考\n"
                f"相对固件 RMSE: {m_fw['rmse']:.4f} m   "
                f"静态噪声: {m_fw['static_noise_std_m']:.4f} m\n"
                f"(图中黑/灰虚线为两路固件高度，已减去各自起点，与算法相对高度同基准)")
        if ds['scenario'] in ('static', 'translation'):
            self.metrics_box.insert(tk.END,
                "\n[说明] 静止/平移场景真值高度恒为 0，算法输出接近恒定属正常现象：\n"
                "卡尔曼滤波已抑制传感器噪声。原始高度(绿/蓝细线)的抖动即被滤除的噪声。\n"
                "切到『升降』场景可看到算法对运动信号的跟踪。")
        self.metrics_box.insert(tk.END, f"\n\n融合方案: {self.params.fusion_scheme}")

        # 场景识别结果（滤波的同时额外输出的当前运动场景，用于辅助判断）
        if self.params.use_scene_clf and res.get('scene_label') is not None:
            classes = res['scene_classes']
            prob = np.asarray(res['scene_prob'])
            n = len(prob)
            # 用最后 50 个样本的概率均值作为『当前』场景估计，更稳健
            win = prob[-50:] if n >= 50 else prob
            meanp = win.mean(axis=0)
            dom = classes[int(np.argmax(meanp))]
            prob_str = '  '.join(f"{c}:{meanp[j]:.2f}" for j, c in enumerate(classes))
            self.metrics_box.insert(
                tk.END,
                f"\n[场景识别(辅助判断)] 当前场景: {dom}    各场景概率: {prob_str}\n"
                f"  (说明: 气压只反映高度变化，故『平移』并入『静止』；"
                f"有效信号为是否处于『升降』(高度在变))")

        if self.mode == 'three':
            # 汇总三个数据集的评价（与自动调参一致）；每场景可能有多个批次，全部纳入
            parts = []
            total = 0.0
            for k in self.three_keys:
                dlist = self.datasets.get(k) or []
                if not dlist:
                    continue
                sc = 0.0
                for d in dlist:
                    r = simulate(d['ms_pressure'], d['bmp_pressure'], d['temperature'], self.params)
                    sc += self._cost(r, d)
                parts.append(f"{self.three_labels[k]}={sc:.4f}({len(dlist)}批)")
                total += sc
            self.metrics_box.insert(tk.END, f"\n三者代价: {'  '.join(parts)}  "
                                            f"(合计={total:.4f})")

        if self.ghost_var.get():
            self.ghost = res['fused_height'].copy()
            self.ghost_label = (f"{ds['scenario']} 方案{self.params.fusion_scheme} "
                                f"Q{self.params.bmp_q:.3f} R{self.params.bmp_r:.2f}")
        else:
            self.ghost = None

    # ---------------- 自动调参 ----------------
    def _cost(self, res, ds):
        """评价当前仿真结果的『代价』，越小越好。供自动调参做目标函数。"""
        truth = ds['true_height']
        m = compute_metrics(ds['time'], res['fused_height'], truth, ds['scenario'])

        def n(x):
            return 0.0 if (x != x) else float(x)

        if truth is not None:
            # 有真值：综合 RMSE / 静态噪声 / 回零误差 / 过冲 / 跟踪延迟
            return (n(m.get('rmse')) + 5.0 * n(m.get('static_noise_std_m'))
                    + 1.0 * abs(n(m.get('return_error_m')))
                    + 1.0 * n(m.get('overshoot_m'))
                    + 0.5 * n(m.get('tracking_lag_s')))
        # 无真值：用融合高度的平滑度(二阶差分均值)作代理，越小越平滑
        h = np.asarray(res['fused_height'], dtype=float)
        if len(h) > 2:
            d2 = np.abs(h[2:] - 2 * h[1:-1] + h[:-2])
            return float(np.mean(d2))
        return 0.0

    def _evaluate(self, params):
        """评价给定参数下『当前模式』的综合代价：
        单数据集 -> 该数据集代价；三数据集 -> 三者代价之和。"""
        if self.mode == 'three':
            total = 0.0
            for k in self.three_keys:
                dlist = self.datasets.get(k) or []
                for d in dlist:
                    res = simulate(d['ms_pressure'], d['bmp_pressure'], d['temperature'], params)
                    total += self._cost(res, d)
            return total
        ds = self.current_ds
        res = simulate(ds['ms_pressure'], ds['bmp_pressure'], ds['temperature'], params)
        return self._cost(res, ds)

    def _on_auto_tune(self):
        if self.current_ds is None:
            messagebox.showinfo('提示', '请先选择数据集再自动调参。')
            return
        if self._auto_running:          # 再次点击 = 停止
            self._auto_running = False
            return
        self._auto_running = True
        self.auto_btn.config(text='停止')
        try:
            self._auto_tune()
        finally:
            self._auto_running = False
            self.auto_btn.config(text='自动调参')

    def _auto_tune(self):
        """坐标上升(爬山)自动调参：逐参数沿改善方向搜索，多轮迭代，
        直到一轮内无改善或相对改善小于阈值。"""
        self._read_controls_to_params()
        params = self.params

        bounds = {s[2]: (s[3], s[4], s[5]) for s in PARAM_SPEC}
        tune_attrs = ['ms_q', 'ms_r', 'ms_residual_th', 'ms_q_inc', 'ms_q_dec', 'ms_q_max',
                      'bmp_q', 'bmp_r', 'bmp_residual_th', 'bmp_q_inc', 'bmp_q_dec', 'bmp_q_max',
                      'pressure_ema_alpha', 'height_ema_alpha', 'w_ms', 'w_bmp', 'hpf_alpha',
                      'motion_threshold', 'weight_static_ms', 'weight_motion_ms']
        tune_attrs = [a for a in tune_attrs if a in bounds]

        cur_cost = self._evaluate(params)
        start_cost = cur_cost

        max_passes = 12
        pass_idx = 0
        while pass_idx < max_passes and self._auto_running:
            improved = False
            for attr in tune_attrs:
                if not self._auto_running:
                    break
                lo, hi, _ = bounds[attr]
                span = hi - lo
                if span <= 0:
                    continue
                step = span * 0.15
                for direction in (1, -1):
                    val = getattr(params, attr)
                    while True:
                        cand = val + direction * step
                        if cand < lo or cand > hi:
                            break
                        setattr(params, attr, cand)
                        c = self._evaluate(params)
                        if c < cur_cost - 1e-9:
                            cur_cost = c
                            val = cand
                            improved = True
                        else:
                            break
                    setattr(params, attr, val)
                self.root.update_idletasks()
            pass_idx += 1
            self._apply_params_to_controls()
            self._run()
            self.metrics_box.insert(tk.END,
                f"\n[自动调参] 第{pass_idx}轮  cost={cur_cost:.5f}")
            self.metrics_box.see(tk.END)
            self.root.update()
            if not improved:
                break
            if start_cost > 0 and (start_cost - cur_cost) / start_cost < 1e-3:
                break

        self._apply_params_to_controls()
        self._run()
        self.metrics_box.insert(
            tk.END, f"\n[自动调参] 完成：cost {start_cost:.5f} -> {cur_cost:.5f}"
                    f"（共 {pass_idx} 轮）\n")
        self.metrics_box.see(tk.END)

    # ---------------- 滚动 / x 轴同步 ----------------
    def _on_scroll(self, _val):
        """样本滚动条：左右移动波形查看窗口。"""
        if self.t is None or self._scrolling:
            return
        T = float(self.t[-1] - self.t[0])
        if T <= 0:
            return
        self._scrolling = True
        try:
            frac = float(self.scroll_var.get()) / 1000.0
            x0, x1 = self.ax1.get_xlim()
            w = x1 - x0
            if w >= T:                      # 当前已是全览，无需滚动
                lo, hi = self.t[0], self.t[-1]
            else:
                center = self.t[0] + frac * T
                lo = max(self.t[0], center - w / 2.0)
                hi = min(self.t[-1], center + w / 2.0)
            for ax in (self.ax1, self.ax2, self.ax3):
                ax.set_xlim(lo, hi)
        finally:
            self._scrolling = False
        self.canvas.draw_idle()

    def _on_xlim_changed(self, ax):
        """任一子图缩放/平移后，同步另外两个子图与滚动条位置。"""
        if self.t is None or self._scrolling:
            return
        self._scrolling = True
        x0, x1 = ax.get_xlim()
        for o in (self.ax1, self.ax2, self.ax3):
            if o is not ax:
                o.set_xlim(x0, x1)
        T = float(self.t[-1] - self.t[0])
        if T > 0 and (x1 - x0) < T:
            center = (x0 + x1) / 2.0
            frac = max(0.0, min(1.0, (center - self.t[0]) / T))
            self.scroll_var.set(frac * 1000.0)
        self._scrolling = False          # 关键：必须复位，否则后续滚动条/拖拽被永久锁死
        self.canvas.draw_idle()

    def _on_press(self, event):
        """左键在画布上按下 -> 进入拖拽平移（用 matplotlib 数据坐标 xdata，避免像素换算）。"""
        if getattr(self.toolbar, 'mode', ''):   # 工具栏放大/平移模式激活时不抢事件
            return
        if self.t is None or getattr(event, 'button', 0) != 1 or event.xdata is None:
            return
        x0, x1 = self.ax1.get_xlim()
        self._drag = {'x': event.xdata, 'x0': x0, 'x1': x1, 'w': x1 - x0}

    def _on_motion(self, event):
        """拖动时按数据坐标位移平移视口（抓拽式：鼠标右移则显示后续样本）。"""
        if self._drag is None or self.t is None or event.xdata is None:
            return
        delta = event.xdata - self._drag['x']
        x0 = self._drag['x0'] + delta
        x1 = self._drag['x1'] + delta
        T0, T1 = float(self.t[0]), float(self.t[-1])
        if x0 < T0:
            x0, x1 = T0, T0 + self._drag['w']
        if x1 > T1:
            x1, x0 = T1, T1 - self._drag['w']
        self.ax1.set_xlim(x0, x1)       # 触发 _on_xlim_changed 同步 ax2/ax3 与滚动条
        self.canvas.draw_idle()

    def _on_release(self, event):
        self._drag = None

    # ---------------- 绘图 ----------------
    def _plot(self, res, ds):
        for ax in (self.ax1, self.ax2, self.ax3):
            ax.clear()
        # 清除上一轮的 twin 轴（气压右轴）
        for attr in ('ax1b', 'ax2b', 'ax3b'):
            twin = getattr(self, attr, None)
            if twin is not None:
                twin.remove()
                setattr(self, attr, None)
        # x 轴用样本序号（而非时间），按固定样本数窗口显示
        t = np.arange(len(ds['time']), dtype=float)
        self.t = t
        truth = res['truth']
        fw_ms_ref = res.get('fw_ms_ref')
        fw_bmp_ref = res.get('fw_bmp_ref')

        # ---------- 子窗口1：MS5611（输入=气压，输出=高度）----------
        self.ax1.plot(t, res['raw_height_ms'], 'g-', lw=0.5, alpha=0.4,
                      label='MS5611原始高度(由气压换算)')
        self.ax1.plot(t, res['kf_height_ms'], 'g-', lw=1.2, label='MS5611 KF高度')
        if fw_ms_ref is not None:
            self.ax1.plot(t, fw_ms_ref, 'k--', lw=1.0, label='固件MS5611(参考,相对)')
        ax1b = self.ax1.twinx()
        ax1b.plot(t, ds['ms_pressure'], color='#9acd32', lw=0.5, alpha=0.5,
                  label='MS5611气压(输入,Pa)')
        ax1b.set_ylabel('气压 (Pa)')
        self.ax1b = ax1b
        self.ax1.set_ylabel('高度 (m)')
        self.ax1.set_title('MS5611')
        l1, p1 = self.ax1.get_legend_handles_labels()
        l2, p2 = ax1b.get_legend_handles_labels()
        self.ax1.legend(l1 + l2, p1 + p2, fontsize=6, loc='upper left')
        self.ax1.grid(alpha=0.3)

        # ---------- 子窗口2：BMP280 ----------
        self.ax2.plot(t, res['raw_height_bmp'], 'b-', lw=0.5, alpha=0.4,
                      label='BMP280原始高度(由气压换算)')
        self.ax2.plot(t, res['kf_height_bmp'], 'b-', lw=1.2, label='BMP280 KF高度')
        if fw_bmp_ref is not None:
            self.ax2.plot(t, fw_bmp_ref, 'k--', lw=1.0, label='固件BMP280(参考,相对)')
        ax2b = self.ax2.twinx()
        ax2b.plot(t, ds['bmp_pressure'], color='#87cefa', lw=0.5, alpha=0.5,
                  label='BMP280气压(输入,Pa)')
        ax2b.set_ylabel('气压 (Pa)')
        self.ax2b = ax2b
        self.ax2.set_ylabel('高度 (m)')
        self.ax2.set_title('BMP280')
        l1, p1 = self.ax2.get_legend_handles_labels()
        l2, p2 = ax2b.get_legend_handles_labels()
        self.ax2.legend(l1 + l2, p1 + p2, fontsize=6, loc='upper left')
        self.ax2.grid(alpha=0.3)

        # ---------- 子窗口3：融合 ----------
        self.ax3.plot(t, res['fused_height'], 'r-', lw=1.4, label='融合高度')
        if truth is not None:
            self.ax3.plot(t, truth, 'k--', lw=1.2, label='参考真值(仅对比,非算法输入)')
        if fw_ms_ref is not None:
            self.ax3.plot(t, fw_ms_ref, 'k--', lw=1.0, label='固件MS5611(参考,相对)')
        if fw_bmp_ref is not None:
            self.ax3.plot(t, fw_bmp_ref, 'gray', lw=0.8, ls=':', label='固件BMP280(参考,相对)')
        if self.ghost is not None:
            self.ax3.plot(t, self.ghost, 'gray', lw=1.0, ls=':', label='上次(对比)')
        ax3b = self.ax3.twinx()
        ax3b.plot(t, res['fused_pressure'], color='#ff9999', lw=0.6, alpha=0.5,
                  label='融合气压(Pa)')
        ax3b.set_ylabel('气压 (Pa)')
        self.ax3b = ax3b
        self.ax3.set_ylabel('高度 (m)')
        self.ax3.set_xlabel('样本序号')
        self.ax3.set_title('融合')
        l1, p1 = self.ax3.get_legend_handles_labels()
        l2, p2 = ax3b.get_legend_handles_labels()
        self.ax3.legend(l1 + l2, p1 + p2, fontsize=6, loc='upper left')
        self.ax3.grid(alpha=0.3)

        self.fig.tight_layout()
        # 初始按固定窗口显示开头一段，超出部分用滚动条查看
        self._apply_window(reset=True)
        self.canvas.draw()

    def _apply_window(self, reset=False):
        """按 self.window_samples 设置 x 轴显示宽度(样本数)，reset=True 时回到起点。"""
        if self.t is None or len(self.t) == 0:
            return
        t0, t1 = float(self.t[0]), float(self.t[-1])
        T = t1 - t0
        w = self.window_samples
        self._scrolling = True
        if w is None or w >= (T + 1):     # 全部 / 窗口覆盖全部样本 -> 全览
            lo, hi = t0, t1
            self.scroll_var.set(0.0)
        else:
            if reset:
                lo, hi = t0, t0 + w
                self.scroll_var.set(0.0)
            else:
                # 保持当前中心，仅调整宽度
                x0, x1 = self.ax1.get_xlim()
                center = (x0 + x1) / 2.0
                lo = max(t0, center - w / 2.0)
                hi = min(t1, lo + w)
                lo = max(t0, hi - w)
                self.scroll_var.set((( (lo + hi) / 2.0 - t0) / T) * 1000.0)
        for ax in (self.ax1, self.ax2, self.ax3):
            ax.set_xlim(lo, hi)
        self._scrolling = False
        self.canvas.draw_idle()

    def _on_window_change(self, event=None):
        val = self.window_var.get()
        if val == '全部':
            self.window_samples = None
        else:
            try:
                self.window_samples = float(val)
            except ValueError:
                return
        self._apply_window(reset=True)

    # ---------------- 导出 ----------------
    def _collect_params(self):
        """收集当前所有参数名为值的 dict（含模式信息）。"""
        p = self.params
        ignore = {'use_nn', 'nn_ms_model', 'nn_bmp_model',
                  'use_scene_clf', 'scene_model'}
        data = {}
        for attr in vars(p):
            if attr in ignore:
                continue
            val = getattr(p, attr)
            if isinstance(val, (np.floating, np.integer)):
                val = float(val)
            elif isinstance(val, bool):
                val = bool(val)
            elif val is not None:
                val = float(val)
            data[attr] = val
        data['_mode'] = self.mode
        if self.mode == 'three':
            data['_three_datasets'] = {
                k: {'scenario': (self.datasets.get(k) or [{}])[0].get('scenario', k),
                    'source': self.three_path_vars[k].get() or 'synthetic',
                    'n_batches': len(self.datasets.get(k) or [])}
                for k in self.three_keys
            }
        else:
            data['_single_dataset'] = self.scenario_var.get()
        data['_c_header_snippet'] = export_header_snippet(p)
        return data

    def _export_params(self):
        self._read_controls_to_params()
        data = self._collect_params()
        path = filedialog.asksaveasfilename(title='导出参数数据',
                                            defaultextension='.json',
                                            filetypes=[('JSON', '*.json'), ('CSV', '*.csv')])
        if not path:
            return
        if path.lower().endswith('.csv'):
            import csv as _csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = _csv.writer(f)
                w.writerow(['param', 'value'])
                for k, v in data.items():
                    if k.startswith('_'):
                        continue
                    w.writerow([k, '' if v is None else v])
            messagebox.showinfo('已导出', f'参数数据(CSV)已写入:\n{path}')
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo('已导出', f'参数数据(JSON)已写入:\n{path}')

    def _export_fig(self):
        path = filedialog.asksaveasfilename(title='保存图片',
                                            defaultextension='.png',
                                            filetypes=[('PNG', '*.png')])
        if path:
            self.fig.savefig(path, dpi=150)
            messagebox.showinfo('已导出', f'图片已保存:\n{path}')

    def _gen_samples(self):
        out_dir = filedialog.askdirectory(title='选择示例CSV输出目录')
        if not out_dir:
            return
        for s in SCENARIOS:
            p = save_synthetic_csv(s, out_dir, n_samples=400, fs=10.0)
            print('saved', p)
        messagebox.showinfo('完成', f'已生成 {", ".join(SCENARIOS)}.csv 到:\n{out_dir}')


def main():
    root = tk.Tk()
    app = TunerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
