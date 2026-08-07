#!/usr/bin/env python
"""STM32 气压传感器 24 小时数据采集 GUI

功能:
  - 一键启动/停止 24h 数据采集
  - 每 15 分钟自动获取桂林雁山天气气压标签
  - 实时显示传感器数据和波形
  - 每 15 分钟输出时段统计报告
  - 24h 结束后生成全天气压分析报告

用法:
    python data_collector_gui.py
"""

import sys
import os
import time
import csv
import json
import threading
try:
    import urllib2
except ImportError:
    import urllib.request as urllib2
from datetime import datetime, timedelta
from collections import deque

import serial
import serial.tools.list_ports

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QProgressBar, QStatusBar, QTabWidget, QGroupBox, QCheckBox,
    QSplitter, QMessageBox, QLineEdit, QDoubleSpinBox, QFrame, QGridLayout,
    QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QTextCursor, QColor, QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# ================================================================
#  常量
# ================================================================
GUILIN_YANSHAN_ADCODE = "450311"
WEATHER_API_URL = "https://uapis.cn/api/v1/misc/weather?adcode={adcode}&extended=true"
PERIOD_MINUTES = 15
DEFAULT_DURATION_HOURS = 24


# ================================================================
#  工具函数
# ================================================================
def fetch_guilin_pressure():
    """获取桂林雁山当前气压（Pa）"""
    try:
        url = WEATHER_API_URL.format(adcode=GUILIN_YANSHAN_ADCODE)
        req = urllib2.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib2.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        if data.get('pressure') is not None:
            pressure_hpa = data['pressure']
            return pressure_hpa * 100.0, data.get('report_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return None, None
    except Exception as e:
        return None, None


# ================================================================
#  串口读取线程
# ================================================================
class SerialReader(QThread):
    data_received = pyqtSignal(str)

    def __init__(self, port, baud_rate):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.running = True
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            while self.running:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self.data_received.emit(line)
                except Exception:
                    pass
                time.sleep(0.01)
        except Exception as e:
            self.data_received.emit("ERROR,{}".format(str(e)))
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()

    def send_command(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write((cmd + '\r\n').encode('utf-8'))


# ================================================================
#  波形图组件
# ================================================================
class WaveformPlot(FigureCanvas):
    def __init__(self, parent=None, title='', xlabel='Sample #', ylabel=''):
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title, fontsize=9)
        self.ax.set_xlabel(xlabel, fontsize=8)
        if ylabel:
            self.ax.set_ylabel(ylabel, fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.ax.tick_params(labelsize=7)
        super().__init__(self.fig)
        self.setParent(parent)

        self.window_points = 100
        self.data_lines = {}
        self._counter = 0

        # Y 轴范围管理：使用百分位数固定范围，不受毛刺影响
        self.y_min = None
        self.y_max = None
        self.y_margin_ratio = 0.15

        self.fig.tight_layout(pad=1.5)

    def add_line(self, name, color, width=1.0, alpha=1.0, style='-'):
        data = np.full(self.window_points, np.nan)
        line, = self.ax.plot(np.arange(self.window_points), data,
                             color=color, linewidth=width, alpha=alpha,
                             linestyle=style, label=name)
        self.data_lines[name] = {'data': data, 'line': line}
        self.ax.legend(fontsize=7, loc='upper right')
        return self

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if name in self.data_lines:
                d = self.data_lines[name]
                d['data'] = np.roll(d['data'], -1)
                d['data'][-1] = value

        self._counter += 1
        x = np.arange(self._counter - self.window_points + 1, self._counter + 1)

        # 每 5 次更新才重新计算 Y 轴范围，降低 CPU 开销
        if self._counter % 5 == 0:
            all_valid = np.concatenate([
                d['data'][~np.isnan(d['data'])]
                for d in self.data_lines.values()
            ])
            if len(all_valid) > 5:
                median = np.median(all_valid)
                std = np.std(all_valid)
                clean = all_valid[np.abs(all_valid - median) < 5.0 * std]
                if len(clean) > 2:
                    d_min, d_max = np.min(clean), np.max(clean)
                else:
                    d_min, d_max = np.min(all_valid), np.max(all_valid)

                margin = max((d_max - d_min) * self.y_margin_ratio, 2.0)
                new_ymin = d_min - margin
                new_ymax = d_max + margin

                if self.y_min is None:
                    self.y_min, self.y_max = new_ymin, new_ymax
                else:
                    self.y_min = self.y_min * 0.8 + new_ymin * 0.2
                    self.y_max = self.y_max * 0.8 + new_ymax * 0.2

                self.ax.set_ylim(self.y_min, self.y_max)

        for d in self.data_lines.values():
            d['line'].set_xdata(x)
            d['line'].set_ydata(d['data'])
        self.ax.set_xlim(x[0] - 1, x[-1] + 1)

        # 降低刷新频率：每 2 次更新才真正绘制一次
        if self._counter % 2 == 0:
            self.draw_idle()


# ================================================================
#  时段统计
# ================================================================
class PeriodStats(object):
    def __init__(self, label):
        self.period_label = label
        self.ms5611_p = []
        self.bmp280_p = []
        self.ms5611_t = []
        self.bmp280_t = []
        self.ms5611_h = []
        self.bmp280_h = []
        self.count = 0

    def add_ms5611(self, p, t, h):
        self.ms5611_p.append(p); self.ms5611_t.append(t)
        self.ms5611_h.append(h); self.count += 1

    def add_bmp280(self, p, t, h):
        self.bmp280_p.append(p); self.bmp280_t.append(t)
        self.bmp280_h.append(h)

    def _s(self, arr):
        if not arr:
            return (0, 0, 0)
        return (min(arr), max(arr), sum(arr) / len(arr))

    def to_text(self, label_pa=None):
        lines = []
        lines.append("=" * 60)
        lines.append("  时段: {}".format(self.period_label))
        lines.append("  样本数: {}".format(self.count))
        for name, p, t, h in [("MS5611", self.ms5611_p, self.ms5611_t, self.ms5611_h),
                               ("BMP280", self.bmp280_p, self.bmp280_t, self.bmp280_h)]:
            if not p:
                continue
            ps = self._s(p); ts = self._s(t); hs = self._s(h)
            lines.append("  {}:  P={:.1f}~{:.1f}(avg={:.1f}) Pa  T={:.1f}~{:.1f}(avg={:.1f}) C  H={:.1f}~{:.1f}(avg={:.1f}) m".format(
                name, ps[0], ps[1], ps[2], ts[0], ts[1], ts[2], hs[0], hs[1], hs[2]))
        if self.ms5611_p and self.bmp280_p:
            dp = self._s(self.ms5611_p)[2] - self._s(self.bmp280_p)[2]
            dh = self._s(self.ms5611_h)[2] - self._s(self.bmp280_h)[2]
            lines.append("  偏差:  P={:.1f} Pa  H={:.1f} m".format(dp, dh))
        if label_pa:
            lines.append("  标签气压: {:.1f} Pa".format(label_pa))
        lines.append("=" * 60)
        return '\n'.join(lines)


# ================================================================
#  主窗口
# ================================================================
class DataCollectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('气压传感器 24h 数据采集系统 v4.0')
        self.setGeometry(80, 40, 1500, 900)

        # --- 状态 ---
        self.serial_reader = None
        self.collecting = False
        self.altitude_m = 160.0
        self.start_time = None

        # 场景采集状态
        self.scene_collecting = False
        self.scene_scene_name = ''
        self.scene_start_time = 0
        self.scene_file = None
        self.scene_writer = None
        self.scene_ms5611_count = 0
        self.scene_bmp280_count = 0
        self.scene_auto_stop_timer = None

        # 数据统计
        self.ms5611_count = 0
        self.bmp280_count = 0
        self.current_period = None
        self.all_periods = []
        self.next_period_time = 0

        # 标签
        self.label_pa = None
        self.label_time = None

        # 文件
        self.labeled_file = None
        self.labeled_writer = None

        # 数据缓冲区（用于波形）
        self.ms5611_buf = deque(maxlen=100)
        self.bmp280_buf = deque(maxlen=100)
        self.ms5611_kf_buf = deque(maxlen=100)
        self.bmp280_kf_buf = deque(maxlen=100)
        self.fusion_p_buf = deque(maxlen=100)
        self.fusion_h_buf = deque(maxlen=100)

        # 滑动平均缓冲区（显示用，压制噪声到 0.2m 以内）
        self.smooth_window = 10  # 10 点滑动平均 ≈ 1 秒
        self.ms5611_smooth = deque(maxlen=self.smooth_window)
        self.bmp280_smooth = deque(maxlen=self.smooth_window)
        self.fusion_smooth = deque(maxlen=self.smooth_window)

        self._build_ui()
        self._refresh_ports()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)  # 100ms 刷新一次，波形更流畅

        self.status_bar.showMessage('就绪 — 请连接串口')

    # ============================================================
    #  UI 构建
    # ============================================================
    def _build_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)

        # --- 左侧: 标签页 ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 顶栏
        left_layout.addWidget(self._build_top_bar())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_monitor_tab(), '📊 实时监控')
        self.tabs.addTab(self._build_report_tab(), '📋 报告')
        self.tabs.addTab(self._build_raw_tab(), '📝 原始数据')
        left_layout.addWidget(self.tabs)

        # --- 右侧: 控制面板 ---
        right = QWidget()
        right.setMaximumWidth(340)
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._build_control_panel())
        right_layout.addWidget(self._build_status_panel())
        right_layout.addStretch()

        layout.addWidget(left, stretch=3)
        layout.addWidget(right, stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _build_top_bar(self):
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 4, 4, 4)

        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(150)
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['115200', '9600', '19200', '38400', '57600'])
        self.baud_combo.setCurrentText('115200')

        self.refresh_btn = QPushButton('🔄 刷新')
        self.refresh_btn.clicked.connect(self._refresh_ports)
        self.connect_btn = QPushButton('🔌 连接')
        self.connect_btn.clicked.connect(self._toggle_connection)

        self.alt_spin = QDoubleSpinBox()
        self.alt_spin.setRange(-500, 10000)
        self.alt_spin.setDecimals(1)
        self.alt_spin.setValue(self.altitude_m)
        self.alt_spin.setSuffix(' m')
        self.set_alt_btn = QPushButton('设置海拔')
        self.set_alt_btn.clicked.connect(self._send_set_alt)

        layout.addWidget(QLabel('串口:'))
        layout.addWidget(self.port_combo)
        layout.addWidget(QLabel('波特率:'))
        layout.addWidget(self.baud_combo)
        layout.addWidget(self.refresh_btn)
        layout.addWidget(self.connect_btn)
        layout.addWidget(QLabel('海拔:'))
        layout.addWidget(self.alt_spin)
        layout.addWidget(self.set_alt_btn)
        return bar

    def _build_monitor_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- 传感器实时值 ---
        info = QHBoxLayout()

        g1 = QGroupBox('MS5611')
        g1l = QVBoxLayout(g1)
        self.ms5611_p_label = QLabel('气压: -- Pa')
        self.ms5611_t_label = QLabel('温度: -- °C')
        self.ms5611_h_label = QLabel('高度: -- m')
        self.ms5611_kfp_label = QLabel('KF气压: -- Pa')
        self.ms5611_kfh_label = QLabel('KF高度: -- m')
        self.ms5611_nnp_label = QLabel('NN气压: -- Pa')
        self.ms5611_nnh_label = QLabel('NN高度: -- m')
        for w in [self.ms5611_p_label, self.ms5611_t_label, self.ms5611_h_label,
                  self.ms5611_kfp_label, self.ms5611_kfh_label,
                  self.ms5611_nnp_label, self.ms5611_nnh_label]:
            g1l.addWidget(w)
        info.addWidget(g1)

        g2 = QGroupBox('BMP280')
        g2l = QVBoxLayout(g2)
        self.bmp280_p_label = QLabel('气压: -- Pa')
        self.bmp280_t_label = QLabel('温度: -- °C')
        self.bmp280_h_label = QLabel('高度: -- m')
        self.bmp280_kfp_label = QLabel('KF气压: -- Pa')
        self.bmp280_kfh_label = QLabel('KF高度: -- m')
        self.bmp280_nnp_label = QLabel('NN气压: -- Pa')
        self.bmp280_nnh_label = QLabel('NN高度: -- m')
        for w in [self.bmp280_p_label, self.bmp280_t_label, self.bmp280_h_label,
                  self.bmp280_kfp_label, self.bmp280_kfh_label,
                  self.bmp280_nnp_label, self.bmp280_nnh_label]:
            g2l.addWidget(w)
        info.addWidget(g2)

        g3 = QGroupBox('融合 (FUSION)')
        g3l = QVBoxLayout(g3)
        self.fusion_p_label = QLabel('融合气压: -- Pa')
        self.fusion_t_label = QLabel('温度: -- °C')
        self.fusion_h_label = QLabel('融合高度: -- m')
        for w in [self.fusion_p_label, self.fusion_t_label, self.fusion_h_label]:
            g3l.addWidget(w)
        info.addWidget(g3)

        g4 = QGroupBox('标签')
        g4l = QVBoxLayout(g4)
        self.label_val_label = QLabel('API气压: --')
        self.label_time_label = QLabel('更新时间: --')
        self.period_label = QLabel('当前时段: --')
        for w in [self.label_val_label, self.label_time_label, self.period_label]:
            g4l.addWidget(w)
        info.addWidget(g4)

        layout.addLayout(info)

        # --- 波形图：三个独立窗口，用子标签页切换 ---
        self.wave_tabs = QTabWidget()
        self.ms5611_plot = WaveformPlot(title='MS5611 - 气压滤波对比 (Raw/KF/NN)', ylabel='Pa')
        self.ms5611_plot.add_line('Raw', 'red', 0.8, 0.6).add_line('KF', 'green', 1.2).add_line('NN', 'blue', 1.0)
        self.wave_tabs.addTab(self.ms5611_plot, 'MS5611 波形')

        self.bmp280_plot = WaveformPlot(title='BMP280 - 气压滤波对比 (Raw/KF/NN)', ylabel='Pa')
        self.bmp280_plot.add_line('Raw', 'red', 0.8, 0.6).add_line('KF', 'green', 1.2).add_line('NN', 'blue', 1.0)
        self.wave_tabs.addTab(self.bmp280_plot, 'BMP280 波形')

        self.fusion_plot = WaveformPlot(title='融合气压 (NN融合)', ylabel='Pa')
        self.fusion_plot.add_line('KF', 'blue', 1.2)
        self.wave_tabs.addTab(self.fusion_plot, '融合 波形')

        layout.addWidget(self.wave_tabs)

        return tab

    def _build_report_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        btn_layout = QHBoxLayout()
        self.clear_report_btn = QPushButton('🗑 清空报告')
        self.clear_report_btn.clicked.connect(lambda: self.report_text.clear())
        btn_layout.addStretch()
        btn_layout.addWidget(self.clear_report_btn)
        layout.addLayout(btn_layout)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont('Consolas', 10))
        layout.addWidget(self.report_text)
        return tab

    def _build_raw_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.raw_text = QTextEdit()
        self.raw_text.setReadOnly(True)
        self.raw_text.setFont(QFont('Consolas', 9))
        layout.addWidget(self.raw_text)
        return tab

    def _build_control_panel(self):
        group = QGroupBox('🎯 采集控制')
        layout = QVBoxLayout(group)

        # 采集参数
        grid = QGridLayout()
        grid.addWidget(QLabel('采集时长:'), 0, 0)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1, 72)
        self.duration_spin.setValue(DEFAULT_DURATION_HOURS)
        self.duration_spin.setSuffix(' 小时')
        grid.addWidget(self.duration_spin, 0, 1)

        grid.addWidget(QLabel('标签间隔:'), 1, 0)
        self.period_label_display = QLabel('15 分钟 (自动)')
        grid.addWidget(self.period_label_display, 1, 1)
        layout.addLayout(grid)

        # 按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton('▶ 开始 24h 采集')
        self.start_btn.setStyleSheet(
            'QPushButton { background-color: #4CAF50; color: white; '
            'font-weight: bold; padding: 10px; font-size: 14px; }'
            'QPushButton:disabled { background-color: #ccc; }')
        self.stop_btn = QPushButton('⏹ 停止')
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            'QPushButton { background-color: #f44336; color: white; '
            'font-weight: bold; padding: 10px; font-size: 14px; }'
            'QPushButton:disabled { background-color: #ccc; }')
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.start_btn.clicked.connect(self._start_collection)
        self.stop_btn.clicked.connect(self._stop_collection)

        # --- 场景采集按钮 ---
        scene_layout = QVBoxLayout()
        scene_layout.setSpacing(4)

        scene_title = QLabel('📥 场景数据采集')
        scene_title.setStyleSheet('font-weight: bold; font-size: 12px; margin-top: 8px;')
        scene_layout.addWidget(scene_title)

        scene_btn_layout = QHBoxLayout()

        self.static_btn = QPushButton('🟢 静止采集')
        self.static_btn.setStyleSheet(
            'QPushButton { background-color: #2196F3; color: white; '
            'font-weight: bold; padding: 8px; font-size: 12px; }'
            'QPushButton:disabled { background-color: #ccc; }')
        self.static_btn.setToolTip('设备静止不动，采集基线噪声数据\n保存到 data/raw/静止/')

        self.move_btn = QPushButton('🔵 平移运动采集')
        self.move_btn.setStyleSheet(
            'QPushButton { background-color: #FF9800; color: white; '
            'font-weight: bold; padding: 8px; font-size: 12px; }'
            'QPushButton:disabled { background-color: #ccc; }')
        self.move_btn.setToolTip('设备在水平方向平移运动，采集运动噪声数据\n保存到 data/raw/平移运动/')

        self.lift_btn = QPushButton('🟣 升降运动采集')
        self.lift_btn.setStyleSheet(
            'QPushButton { background-color: #9C27B0; color: white; '
            'font-weight: bold; padding: 8px; font-size: 12px; }'
            'QPushButton:disabled { background-color: #ccc; }')
        self.lift_btn.setToolTip('设备上下升降运动（0.7~1m），采集高度变化数据\n保存到 data/raw/升降运动/')

        self.scene_stop_btn = QPushButton('⏹ 停止场景采集')
        self.scene_stop_btn.setEnabled(False)
        self.scene_stop_btn.setStyleSheet(
            'QPushButton { background-color: #f44336; color: white; '
            'font-weight: bold; padding: 8px; font-size: 12px; }'
            'QPushButton:disabled { background-color: #ccc; }')

        scene_btn_layout.addWidget(self.static_btn)
        scene_btn_layout.addWidget(self.move_btn)
        scene_btn_layout.addWidget(self.lift_btn)
        scene_layout.addLayout(scene_btn_layout)
        scene_layout.addWidget(self.scene_stop_btn)

        self.static_btn.clicked.connect(lambda: self._start_scene_collection('静止'))
        self.move_btn.clicked.connect(lambda: self._start_scene_collection('平移运动'))
        self.lift_btn.clicked.connect(lambda: self._start_scene_collection('升降运动'))
        self.scene_stop_btn.clicked.connect(self._stop_scene_collection)

        layout.addLayout(scene_layout)

        # 命令
        cmd_layout = QHBoxLayout()
        self.status_cmd_btn = QPushButton('STATUS')
        self.status_cmd_btn.clicked.connect(lambda: self._send_cmd('STATUS'))
        self.help_cmd_btn = QPushButton('HELP')
        self.help_cmd_btn.clicked.connect(lambda: self._send_cmd('HELP'))
        cmd_layout.addWidget(self.status_cmd_btn)
        cmd_layout.addWidget(self.help_cmd_btn)
        layout.addLayout(cmd_layout)

        return group

    def _build_status_panel(self):
        group = QGroupBox('📈 状态')
        layout = QVBoxLayout(group)

        self.status_info = QLabel('未启动')
        self.status_info.setWordWrap(True)
        self.status_info.setStyleSheet('font-size: 12px; padding: 4px;')
        layout.addWidget(self.status_info)

        self.ms5611_progress = QProgressBar()
        self.ms5611_progress.setFormat('MS5611: %v')
        self.bmp280_progress = QProgressBar()
        self.bmp280_progress.setFormat('BMP280: %v')
        layout.addWidget(self.ms5611_progress)
        layout.addWidget(self.bmp280_progress)

        self.time_label = QLabel('已采集: 00:00:00 / 剩余: 24:00:00')
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet('font-weight: bold; font-size: 13px; padding: 4px;')
        layout.addWidget(self.time_label)

        self.sample_rate_label = QLabel('采样率: --/s')
        self.sample_rate_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.sample_rate_label)

        return group

    # ============================================================
    #  串口操作
    # ============================================================
    def _refresh_ports(self):
        self.port_combo.clear()
        for p in serial.tools.list_ports.comports():
            self.port_combo.addItem('{} - {}'.format(p.device, p.description))

    def _toggle_connection(self):
        if self.serial_reader and self.serial_reader.isRunning():
            if self.collecting:
                QMessageBox.warning(self, '警告', '请先停止采集再断开')
                return
            self.serial_reader.stop()
            self.serial_reader.wait()
            self.serial_reader = None
            self.connect_btn.setText('🔌 连接')
            self.status_bar.showMessage('已断开')
        else:
            txt = self.port_combo.currentText()
            if not txt:
                return
            port = txt.split(' - ')[0]
            baud = int(self.baud_combo.currentText())
            self.serial_reader = SerialReader(port, baud)
            self.serial_reader.data_received.connect(self._on_data)
            self.serial_reader.start()
            self.connect_btn.setText('🔌 断开')
            self.status_bar.showMessage('已连接 {} @ {}'.format(port, baud))

    def _send_cmd(self, cmd):
        if self.serial_reader and self.serial_reader.isRunning():
            self.serial_reader.send_command(cmd)

    def _send_set_alt(self):
        self.altitude_m = self.alt_spin.value()
        self._send_cmd('SET_ALT {:.1f}'.format(self.altitude_m))

    # ============================================================
    #  采集控制
    # ============================================================
    def _start_collection(self):
        if not self.serial_reader or not self.serial_reader.isRunning():
            QMessageBox.warning(self, '警告', '请先连接串口')
            return

        # 先发 SET_ALT
        self._send_cmd('SET_ALT {:.1f}'.format(self.altitude_m))

        # 创建数据集文件
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = os.path.join(os.path.dirname(__file__), "data", "labeled")
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, "dataset_{}.csv".format(ts))
        self.labeled_file = open(path, 'w', newline='')
        self.labeled_writer = csv.writer(self.labeled_file)
        self.labeled_writer.writerow([
            'unix_time', 'sample_id', 'sensor', 'pressure_pa', 'temperature_c',
            'height_m', 'kf_pressure_pa', 'kf_height_m',
            'altitude_m', 'label_pressure_pa', 'label_source', 'label_time'
        ])

        # 重置状态
        self.ms5611_count = 0
        self.bmp280_count = 0
        self.all_periods = []
        self.current_period = None
        self.ms5611_buf.clear()
        self.bmp280_buf.clear()
        self.ms5611_kf_buf.clear()
        self.bmp280_kf_buf.clear()
        self.fusion_p_buf.clear()
        self.fusion_h_buf.clear()

        self.collecting = True
        self.start_time = time.time()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # 采集模式：停止非必要活动，全力采集数据
        self.timer.stop()                     # 停止 UI 定时刷新
        self.raw_text.setVisible(False)       # 隐藏原始数据显示
        self.tabs.setTabEnabled(0, False)     # 禁用监控标签页
        self.tabs.setTabEnabled(2, False)     # 禁用原始数据标签页
        self.tabs.setCurrentIndex(1)          # 切换到报告标签页
        self.port_combo.setEnabled(False)
        self.baud_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.connect_btn.setEnabled(False)

        # 对齐到下一个 15 分钟
        now = datetime.now()
        next_q = ((now.minute // 15) + 1) * 15
        if next_q >= 60:
            delay = ((60 - now.minute) * 60 - now.second) + 10
        else:
            delay = ((next_q - now.minute) * 60 - now.second) + 5
        self.next_period_time = time.time() + max(delay, 5)

        self.status_info.setText('采集中...\n数据文件: dataset_{}.csv'.format(ts))
        self.status_bar.showMessage('24h 采集已启动 → data/labeled/dataset_{}.csv'.format(ts))

        # 输出到报告
        self._append_report("=" * 60)
        self._append_report("  24h 数据采集启动")
        self._append_report("  时间: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self._append_report("  海拔: {:.1f} m".format(self.altitude_m))
        self._append_report("  时长: {} 小时".format(self.duration_spin.value()))
        self._append_report("=" * 60)

    def _stop_collection(self):
        self.collecting = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # 恢复所有 UI 功能
        self.timer.start(500)
        self.raw_text.setVisible(True)
        self.tabs.setTabEnabled(0, True)
        self.tabs.setTabEnabled(2, True)
        self.port_combo.setEnabled(True)
        self.baud_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.connect_btn.setEnabled(True)

        # 保存最后一个时段
        if self.current_period and self.current_period.count > 0:
            self.all_periods.append(self.current_period)

        # 关闭文件
        if self.labeled_file:
            self.labeled_file.close()
            self.labeled_file = None

        elapsed = time.time() - self.start_time if self.start_time else 0
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        self.status_info.setText('已停止\n采集时长: {}\nMS5611: {} | BMP280: {}'.format(
            elapsed_str, self.ms5611_count, self.bmp280_count))

        # 输出 24h 报告
        self._append_report("")
        self._append_report("=" * 60)
        self._append_report("  采集结束")
        self._append_report("  时长: {}".format(elapsed_str))
        self._append_report("  MS5611: {} 条 | BMP280: {} 条".format(
            self.ms5611_count, self.bmp280_count))
        self._append_report("=" * 60)

        if self.all_periods:
            self._append_report(self._generate_day_report())

        self.status_bar.showMessage('采集已停止')

    # ============================================================
    #  场景数据采集（静止/平移运动）
    # ============================================================
    def _start_scene_collection(self, scene_name):
        """启动场景数据采集（静止或平移运动）"""
        if not self.serial_reader or not self.serial_reader.isRunning():
            QMessageBox.warning(self, '警告', '请先连接串口')
            return

        if self.collecting:
            QMessageBox.warning(self, '警告', '24h 采集中，请先停止')
            return

        if self.scene_collecting:
            QMessageBox.warning(self, '警告', '场景采集中，请先停止')
            return

        # 创建场景目录和文件
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = os.path.join(os.path.dirname(__file__), "data", "raw", scene_name)
        os.makedirs(data_dir, exist_ok=True)
        ms5611_path = os.path.join(data_dir, 'ms5611_{}.csv'.format(ts))
        bmp280_path = os.path.join(data_dir, 'bmp280_{}.csv'.format(ts))

        try:
            self.scene_ms5611_file = open(ms5611_path, 'w', newline='')
            self.scene_ms5611_writer = csv.writer(self.scene_ms5611_file)
            self.scene_bmp280_file = open(bmp280_path, 'w', newline='')
            self.scene_bmp280_writer = csv.writer(self.scene_bmp280_file)

            header = ['unix_time', 'sample_id', 'pressure_pa', 'temperature_c',
                      'height_m', 'kf_pressure_pa', 'kf_height_m']
            self.scene_ms5611_writer.writerow(header)
            self.scene_bmp280_writer.writerow(header)
        except IOError as e:
            QMessageBox.critical(self, '错误', '创建文件失败: {}'.format(e))
            return

        self.scene_collecting = True
        self.scene_scene_name = scene_name
        self.scene_start_time = time.time()
        self.scene_ms5611_count = 0
        self.scene_bmp280_count = 0

        # 更新按钮状态
        self.static_btn.setEnabled(False)
        self.move_btn.setEnabled(False)
        self.lift_btn.setEnabled(False)
        self.scene_stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        # 发送海拔设置
        self._send_cmd('SET_ALT {:.1f}'.format(self.altitude_m))

        msg = '场景采集已启动: {} → data/raw/{}/\nMS5611: {}\nBMP280: {}'.format(
            scene_name, scene_name,
            os.path.basename(ms5611_path), os.path.basename(bmp280_path))
        self.status_info.setText(msg)
        self.status_bar.showMessage('场景采集: {} ({})'.format(scene_name, ts))

        # 输出到报告
        self._append_report("")
        self._append_report("-" * 50)
        self._append_report("  场景采集启动: {}".format(scene_name))
        self._append_report("  时间: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self._append_report("  文件: data/raw/{}/".format(scene_name))
        self._append_report("-" * 50)

    def _stop_scene_collection(self):
        """停止场景数据采集"""
        if not self.scene_collecting:
            return

        self.scene_collecting = False

        # 关闭文件
        if hasattr(self, 'scene_ms5611_file') and self.scene_ms5611_file:
            self.scene_ms5611_file.close()
            self.scene_ms5611_file = None
        if hasattr(self, 'scene_bmp280_file') and self.scene_bmp280_file:
            self.scene_bmp280_file.close()
            self.scene_bmp280_file = None

        # 恢复按钮
        self.static_btn.setEnabled(True)
        self.move_btn.setEnabled(True)
        self.lift_btn.setEnabled(True)
        self.scene_stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        elapsed = time.time() - self.scene_start_time if self.scene_start_time else 0
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        msg = '场景采集已停止: {}\n时长: {}\nMS5611: {} | BMP280: {}'.format(
            self.scene_scene_name, elapsed_str,
            self.scene_ms5611_count, self.scene_bmp280_count)
        self.status_info.setText(msg)
        self.status_bar.showMessage('场景采集已停止')

        # 输出到报告
        self._append_report("")
        self._append_report("-" * 50)
        self._append_report("  场景采集结束: {}".format(self.scene_scene_name))
        self._append_report("  时长: {}".format(elapsed_str))
        self._append_report("  MS5611: {} | BMP280: {}".format(
            self.scene_ms5611_count, self.scene_bmp280_count))
        self._append_report("-" * 50)

        self.scene_scene_name = ''
        self.scene_ms5611_count = 0
        self.scene_bmp280_count = 0

    def _write_scene_data(self, sensor, sid, p, t, h, kfp, kfh):
        """将传感器数据写入场景采集文件"""
        if not self.scene_collecting:
            return
        now_ts = time.time()
        row = [int(now_ts), sid, p, t, h, kfp, kfh]

        if sensor == 'MS5611' and hasattr(self, 'scene_ms5611_writer') and self.scene_ms5611_writer:
            self.scene_ms5611_writer.writerow(row)
            self.scene_ms5611_file.flush()
            self.scene_ms5611_count += 1
        elif sensor == 'BMP280' and hasattr(self, 'scene_bmp280_writer') and self.scene_bmp280_writer:
            self.scene_bmp280_writer.writerow(row)
            self.scene_bmp280_file.flush()
            self.scene_bmp280_count += 1

    # ============================================================
    #  数据接收
    # ============================================================
    def _on_data(self, line):
        # 原始数据显示
        self.raw_text.append(line)
        self.raw_text.moveCursor(QTextCursor.End)

        if line.startswith('INFO,'):
            msg = line[5:]
            if 'Altitude set to' in msg:
                try:
                    self.altitude_m = float(msg.split('to')[1].split('m')[0].strip())
                except Exception:
                    pass
            return

        parts = line.split(',')
        if len(parts) < 5:
            return

        sensor = parts[0]
        try:
            sid = int(parts[1])
            if sensor == 'FUSION':
                self._handle_fusion(parts)
                return
            pressure = float(parts[2])
            temp = float(parts[3])
            height = float(parts[4])
            kf_p = float(parts[5]) if len(parts) >= 6 else pressure
            kf_h = float(parts[6]) if len(parts) >= 7 else height
            nn_p = float(parts[7]) if len(parts) >= 8 else kf_p
            nn_h = float(parts[8]) if len(parts) >= 9 else kf_h

            if sensor == 'MS5611':
                self._handle_ms5611(sid, pressure, temp, height, kf_p, kf_h, nn_p, nn_h)
            elif sensor == 'BMP280':
                self._handle_bmp280(sid, pressure, temp, height, kf_p, kf_h, nn_p, nn_h)
        except ValueError:
            pass

    def _handle_ms5611(self, sid, p, t, h, kfp, kfh, nnp, nnh):
        self.ms5611_p_label.setText('气压: {:.2f} Pa'.format(p))
        self.ms5611_t_label.setText('温度: {:.2f} °C'.format(t))
        self.ms5611_h_label.setText('高度: {:.2f} m'.format(h))
        self.ms5611_kfp_label.setText('KF气压: {:.2f} Pa'.format(kfp))
        self.ms5611_kfh_label.setText('KF高度: {:.2f} m'.format(kfh))
        self.ms5611_nnp_label.setText('NN气压: {:.2f} Pa'.format(nnp))
        self.ms5611_nnh_label.setText('NN高度: {:.2f} m'.format(nnh))

        self.ms5611_buf.append(p)
        self.ms5611_kf_buf.append(kfp)
        self.ms5611_smooth.append(p)
        self.ms5611_count += 1
        self.ms5611_progress.setValue(self.ms5611_count)

        # 波形更新：收到数据立即刷新
        if self.wave_tabs.currentIndex() == 0:
            self.ms5611_plot.update(Raw=p, KF=kfp, NN=nnp)

        if self.current_period:
            self.current_period.add_ms5611(p, t, h)

        self._write_labeled('MS5611', sid, p, t, h, kfp, kfh)
        self._write_scene_data('MS5611', sid, p, t, h, kfp, kfh)

    def _handle_bmp280(self, sid, p, t, h, kfp, kfh, nnp, nnh):
        self.bmp280_p_label.setText('气压: {:.2f} Pa'.format(p))
        self.bmp280_t_label.setText('温度: {:.2f} °C'.format(t))
        self.bmp280_h_label.setText('高度: {:.2f} m'.format(h))
        self.bmp280_kfp_label.setText('KF气压: {:.2f} Pa'.format(kfp))
        self.bmp280_kfh_label.setText('KF高度: {:.2f} m'.format(kfh))
        self.bmp280_nnp_label.setText('NN气压: {:.2f} Pa'.format(nnp))
        self.bmp280_nnh_label.setText('NN高度: {:.2f} m'.format(nnh))

        self.bmp280_buf.append(p)
        self.bmp280_kf_buf.append(kfp)
        self.bmp280_smooth.append(p)
        self.bmp280_count += 1
        self.bmp280_progress.setValue(self.bmp280_count)

        # 波形更新：收到数据立即刷新
        if self.wave_tabs.currentIndex() == 1:
            self.bmp280_plot.update(Raw=p, KF=kfp, NN=nnp)

        if self.current_period:
            self.current_period.add_bmp280(p, t, h)

        self._write_labeled('BMP280', sid, p, t, h, kfp, kfh)
        self._write_scene_data('BMP280', sid, p, t, h, kfp, kfh)

    def _handle_fusion(self, parts):
        try:
            fp = float(parts[2])       # 融合气压（NN滤波后加权融合）
            fh = float(parts[3])       # 融合高度（由气压公式计算）
            t = float(parts[4])        # 温度
            self.fusion_p_label.setText('融合气压: {:.2f} Pa'.format(fp))
            self.fusion_t_label.setText('温度: {:.2f} °C'.format(t))
            self.fusion_h_label.setText('融合高度: {:.2f} m'.format(fh))
            self.fusion_p_buf.append(fp)
            self.fusion_h_buf.append(fh)
            self.fusion_smooth.append(fp)

            # 波形更新
            if self.wave_tabs.currentIndex() == 2:
                self.fusion_plot.update(KF=fp)
        except ValueError:
            pass

    def _write_labeled(self, sensor, sid, p, t, h, kfp, kfh):
        if not self.collecting or not self.labeled_writer:
            return
        now_ts = time.time()
        self.labeled_writer.writerow([
            int(now_ts), sid, sensor, p, t, h, kfp, kfh,
            self.altitude_m,
            self.label_pa or 0, 'weather_api' if self.label_pa else 'none',
            self.label_time or ''
        ])
        self.labeled_file.flush()

    # ============================================================
    #  时段管理
    # ============================================================
    def _start_new_period(self):
        # 保存上一个时段
        if self.current_period and self.current_period.count > 0:
            self.all_periods.append(self.current_period)
            # 输出时段报告
            report = self.current_period.to_text(self.label_pa)
            self._append_report(report)

        # 新时段
        dt = datetime.now()
        aligned = (dt.minute // 15) * 15
        start = dt.replace(minute=aligned, second=0, microsecond=0)
        label = start.strftime("%H:%M") + "-" + \
                (start + timedelta(minutes=15)).strftime("%H:%M")
        self.current_period = PeriodStats(label)
        self.next_period_time = time.time() + PERIOD_MINUTES * 60

        # 获取标签
        self.period_label.setText('当前时段: {}'.format(label))
        pa, ut = fetch_guilin_pressure()
        if pa:
            self.label_pa = pa
            self.label_time = ut or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.label_val_label.setText('API气压: {:.1f} Pa'.format(pa))
            self.label_time_label.setText('更新时间: {}'.format(self.label_time))
        else:
            self.label_val_label.setText('API气压: 获取失败')

    def _append_report(self, text):
        self.report_text.append(text)
        self.report_text.moveCursor(QTextCursor.End)

    # ============================================================
    #  24h 报告
    # ============================================================
    def _generate_day_report(self):
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  24-HOUR DATA COLLECTION REPORT")
        lines.append("  采集: {} - {}".format(
            datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M"),
            datetime.now().strftime("%Y-%m-%d %H:%M")))
        lines.append("  海拔: {:.1f} m".format(self.altitude_m))
        lines.append("=" * 70)

        # 总体
        all_mp, all_bp, all_mt, all_bt = [], [], [], []
        for p in self.all_periods:
            all_mp.extend(p.ms5611_p); all_bp.extend(p.bmp280_p)
            all_mt.extend(p.ms5611_t); all_bt.extend(p.bmp280_t)

        lines.append("")
        lines.append("【1. 总体统计】")
        lines.append("  总采样: MS5611={}, BMP280={}".format(len(all_mp), len(all_bp)))

        def sta(arr, label):
            if not arr:
                return
            mn, mx, avg = min(arr), max(arr), sum(arr) / len(arr)
            std = (sum((x - avg)**2 for x in arr) / len(arr))**0.5
            lines.append("  {}: min={:.1f} max={:.1f} avg={:.1f} std={:.3f}".format(
                label, mn, mx, avg, std))

        sta(all_mp, "MS5611 气压(Pa)")
        sta(all_bp, "BMP280 气压(Pa)")
        sta(all_mt, "MS5611 温度(C)")
        sta(all_bt, "BMP280 温度(C)")

        # 偏差
        lines.append("")
        lines.append("【2. 双传感器偏差】")
        diffs = []
        for p in self.all_periods:
            if p.ms5611_p and p.bmp280_p:
                diffs.append(sum(p.ms5611_p)/len(p.ms5611_p) -
                             sum(p.bmp280_p)/len(p.bmp280_p))
        if diffs:
            lines.append("  气压偏差(MS-BM): avg={:.2f} Pa ({:.2f} m)".format(
                sum(diffs)/len(diffs), (sum(diffs)/len(diffs))/11.3))
            lines.append("                  max={:.2f} min={:.2f}".format(
                max(diffs), min(diffs)))

        # 时段趋势
        lines.append("")
        lines.append("【3. 时段趋势】")
        lines.append("  {:<12s} {:>12s} {:>12s} {:>10s} {:>8s} {:>8s}".format(
            "时段", "MS5611 avg", "BMP280 avg", "偏差", "T(MS)", "T(BM)"))
        lines.append("  " + "-" * 66)
        for p in self.all_periods:
            if not p.ms5611_p:
                continue
            mp = sum(p.ms5611_p) / len(p.ms5611_p)
            bp = sum(p.bmp280_p) / len(p.bmp280_p) if p.bmp280_p else 0
            mt = sum(p.ms5611_t) / len(p.ms5611_t) if p.ms5611_t else 0
            bt = sum(p.bmp280_t) / len(p.bmp280_t) if p.bmp280_t else 0
            lines.append("  {:<12s} {:>10.1f}Pa {:>10.1f}Pa {:>+8.1f}Pa {:>6.1f}C {:>6.1f}C".format(
                p.period_label, mp, bp, mp - bp, mt, bt))

        # 变化幅度
        lines.append("")
        lines.append("【4. 气压变化幅度】")
        if all_mp:
            r = max(all_mp) - min(all_mp)
            lines.append("  MS5611: {:.1f} Pa ({:.1f} m)".format(r, r / 11.3))
        if all_bp:
            r = max(all_bp) - min(all_bp)
            lines.append("  BMP280: {:.1f} Pa ({:.1f} m)".format(r, r / 11.3))

        lines.append("=" * 70)
        return '\n'.join(lines)

    # ============================================================
    #  定时器
    # ============================================================
    def _tick(self):
        now = time.time()

        # 检查时段切换
        if self.collecting and now >= self.next_period_time:
            self._start_new_period()

        # 波形更新由 _handle_ms5611/_handle_bmp280/_handle_fusion 中直接调用，
        # 定时器不再重复绘制，避免空转

        # 更新时间/速率
        if self.collecting and self.start_time:
            elapsed = now - self.start_time
            duration_h = self.duration_spin.value()
            remaining = duration_h * 3600 - elapsed
            if remaining < 0:
                remaining = 0
                self._stop_collection()
            e_str = str(timedelta(seconds=int(elapsed)))
            r_str = str(timedelta(seconds=int(remaining)))
            self.time_label.setText('已采集: {} / 剩余: {}'.format(e_str, r_str))

            total = self.ms5611_count + self.bmp280_count
            if elapsed > 0:
                rate = total / elapsed
                self.sample_rate_label.setText('采样率: {:.0f} 条/秒 ({:.0f} 条/分)'.format(
                    rate, rate * 60))

        # 场景采集状态更新
        if self.scene_collecting and self.scene_start_time:
            elapsed = now - self.scene_start_time
            e_str = str(timedelta(seconds=int(elapsed)))
            total = self.scene_ms5611_count + self.scene_bmp280_count
            rate = total / elapsed if elapsed > 0 else 0
            self.time_label.setText('场景[{}]: {} / {:.0f}条/秒'.format(
                self.scene_scene_name, e_str, rate))
            self.sample_rate_label.setText('MS:{} BM:{}'.format(
                self.scene_ms5611_count, self.scene_bmp280_count))

    # ============================================================
    #  关闭
    # ============================================================
    def closeEvent(self, event):
        if self.collecting:
            self._stop_collection()
        if self.scene_collecting:
            self._stop_scene_collection()
        if self.serial_reader and self.serial_reader.isRunning():
            self.serial_reader.stop()
            self.serial_reader.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = DataCollectorGUI()
    w.show()
    sys.exit(app.exec_())
