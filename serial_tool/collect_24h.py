#!/usr/bin/env python
"""24 小时气压传感器数据采集 + 15 分钟自动标签 + 数据分析报告

功能:
  1. 连接 STM32 串口，持续采集 MS5611/BMP280 数据（每 100ms 一条）
  2. 每 15 分钟自动从免费天气 API 获取桂林雁山地区气压作为标签
  3. 每 15 分钟更新标签时，自动总结该时段传感器数据情况
  4. 24 小时采集结束后，输出全天气压变化分析报告
  5. 按传感器和时间自动分类生成带标签的数据集

用法:
    python collect_24h.py                              # 交互选择串口
    python collect_24h.py --port COM3                   # 指定串口
    python collect_24h.py --port COM3 --altitude 160    # 指定海拔高度
    python collect_24h.py --port COM3 --duration 12     # 采集 12 小时

数据目录结构:
    data/
    ├── labeled/                           # 带标签数据集（用于训练）
    │   ├── ms5611_labeled.csv
    │   └── bmp280_labeled.csv
    └── self_supervised/                   # 原始数据备份
        ├── ms5611_20260705_214400.csv
        └── bmp280_20260705_214400.csv
"""

from __future__ import absolute_import

import os
import sys
import time
import csv
import json
import argparse
import threading
try:
    import urllib2
except ImportError:
    import urllib.request as urllib2
from datetime import datetime, timedelta
from collections import deque

import serial
from serial.tools.list_ports import comports

SAMPLE_INTERVAL = 0.1   # 100ms
DEFAULT_DURATION_HOURS = 24

# 桂林雁山行政区划代码（用于天气 API）
GUILIN_YANSHAN_ADCODE = "450311"

# 免费天气 API（无需 key）
WEATHER_API_URL = "https://uapis.cn/api/v1/misc/weather?adcode={adcode}&extended=true"


def ask_for_port():
    sys.stderr.write('\n--- Available ports:\n')
    ports = []
    for n, (port, desc, hwid) in enumerate(sorted(comports()), 1):
        sys.stderr.write('--- {:2}: {:20} {!r}\n'.format(n, port, desc))
        ports.append(port)
    if not ports:
        sys.stderr.write('--- No serial ports found!\n')
        sys.exit(1)
    while True:
        try:
            choice = int(raw_input('--- Enter port number (1-{}): '.format(len(ports))))
            if 1 <= choice <= len(ports):
                return ports[choice - 1]
        except ValueError:
            pass
        print('Invalid choice.')


def fetch_guilin_pressure():
    """从免费天气 API 获取桂林雁山当前气压（单位：Pa）"""
    try:
        url = WEATHER_API_URL.format(adcode=GUILIN_YANSHAN_ADCODE)
        req = urllib2.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib2.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        if data.get('pressure') is not None:
            pressure_hpa = data['pressure']  # 单位 hPa
            pressure_pa = pressure_hpa * 100.0  # 转换为 Pa
            return pressure_pa, data.get('report_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return None, None
    except Exception as e:
        sys.stderr.write('\n[WARN] Weather API error: {}\n'.format(e))
        return None, None


class PeriodStats(object):
    """存储一个时段内的传感器数据统计"""
    def __init__(self, period_label):
        self.period_label = period_label
        self.ms5611_pressures = []
        self.bmp280_pressures = []
        self.ms5611_temps = []
        self.bmp280_temps = []
        self.ms5611_heights = []
        self.bmp280_heights = []
        self.sample_count = 0

    def add_ms5611(self, pressure, temp, height):
        self.ms5611_pressures.append(pressure)
        self.ms5611_temps.append(temp)
        self.ms5611_heights.append(height)
        self.sample_count += 1

    def add_bmp280(self, pressure, temp, height):
        self.bmp280_pressures.append(pressure)
        self.bmp280_temps.append(temp)
        self.bmp280_heights.append(height)

    @staticmethod
    def _stat(arr):
        if not arr:
            return None, None, None
        return min(arr), max(arr), sum(arr) / len(arr)

    def summary(self, title=None):
        lines = []
        if title:
            lines.append(title)
        lines.append("  Samples: {}".format(self.sample_count))

        ms_p = self._stat(self.ms5611_pressures)
        ms_t = self._stat(self.ms5611_temps)
        ms_h = self._stat(self.ms5611_heights)
        bm_p = self._stat(self.bmp280_pressures)
        bm_t = self._stat(self.bmp280_temps)
        bm_h = self._stat(self.bmp280_heights)

        if ms_p[0] is not None:
            lines.append("  MS5611:  P={:.1f}~{:.1f}(avg={:.1f}) Pa  T={:.1f}~{:.1f}(avg={:.1f}) C  H={:.1f}~{:.1f}(avg={:.1f}) m".format(
                ms_p[0], ms_p[1], ms_p[2], ms_t[0], ms_t[1], ms_t[2], ms_h[0], ms_h[1], ms_h[2]))
        if bm_p[0] is not None:
            lines.append("  BMP280:  P={:.1f}~{:.1f}(avg={:.1f}) Pa  T={:.1f}~{:.1f}(avg={:.1f}) C  H={:.1f}~{:.1f}(avg={:.1f}) m".format(
                bm_p[0], bm_p[1], bm_p[2], bm_t[0], bm_t[1], bm_t[2], bm_h[0], bm_h[1], bm_h[2]))

        # 双传感器偏差
        if ms_p[2] is not None and bm_p[2] is not None:
            diff_p = ms_p[2] - bm_p[2]
            diff_h = ms_h[2] - bm_h[2]
            lines.append("  MS5611-BMP280偏差:  P={:.1f} Pa  H={:.1f} m".format(diff_p, diff_h))

        return '\n'.join(lines)


class DataCollector24H:
    def __init__(self, serial_port, altitude_m, data_dir=None):
        self.ser = serial_port
        self.altitude_m = altitude_m
        self.alive = False
        self.base_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(os.path.join(self.base_dir, "labeled"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "self_supervised"), exist_ok=True)

        # 原始数据文件（每 6 小时轮换）
        self.raw_ms5611_file = None
        self.raw_bmp280_file = None
        self.raw_ms5611_writer = None
        self.raw_bmp280_writer = None
        self.raw_ms5611_msgs = 0

        # 带标签数据集文件
        self.labeled_ms5611_file = None
        self.labeled_bmp280_file = None
        self.labeled_ms5611_writer = None
        self.labeled_bmp280_writer = None

        # 统计
        self.ms5611_samples = 0
        self.bmp280_samples = 0
        self.start_time = None

        # 标签气压（每 15 分钟更新一次）
        self.current_label_pressure_pa = None
        self.current_label_time = None

        # 15 分钟时段统计
        self.period_stats = None       # 当前时段统计
        self.all_periods = []          # 所有历史时段统计
        self.period_duration = 15 * 60  # 15 分钟
        self.next_period_time = None   # 下一个时段结束时间
        self.label_interval = 15 * 60  # 15 分钟

        # 线程锁
        self.lock = threading.Lock()

    def _open_raw_files(self):
        if self.raw_ms5611_file:
            self.raw_ms5611_file.close()
        if self.raw_bmp280_file:
            self.raw_bmp280_file.close()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_dir = os.path.join(self.base_dir, "self_supervised")
        ms5611_path = os.path.join(raw_dir, "ms5611_{}.csv".format(timestamp))
        bmp280_path = os.path.join(raw_dir, "bmp280_{}.csv".format(timestamp))

        self.raw_ms5611_file = open(ms5611_path, 'w', newline='')
        self.raw_bmp280_file = open(bmp280_path, 'w', newline='')
        self.raw_ms5611_writer = csv.writer(self.raw_ms5611_file)
        self.raw_bmp280_writer = csv.writer(self.raw_bmp280_file)

        header = ['unix_time', 'sample_id', 'pressure_pa', 'temperature_c', 'height_m',
                  'kf_pressure_pa', 'kf_height_m']
        self.raw_ms5611_writer.writerow(header)
        self.raw_bmp280_writer.writerow(header)

        print("\n[Raw file] MS5611: {}".format(ms5611_path))
        print("[Raw file] BMP280: {}".format(bmp280_path))
        self.raw_ms5611_msgs = 0

    def _open_labeled_files(self):
        label_dir = os.path.join(self.base_dir, "labeled")
        ms5611_path = os.path.join(label_dir, "ms5611_labeled.csv")
        bmp280_path = os.path.join(label_dir, "bmp280_labeled.csv")

        self.labeled_ms5611_file = open(ms5611_path, 'a', newline='')
        self.labeled_bmp280_file = open(bmp280_path, 'a', newline='')
        self.labeled_ms5611_writer = csv.writer(self.labeled_ms5611_file)
        self.labeled_bmp280_writer = csv.writer(self.labeled_bmp280_file)

        if os.path.getsize(ms5611_path) == 0:
            self.labeled_ms5611_writer.writerow([
                'unix_time', 'sample_id', 'sensor', 'pressure_pa', 'temperature_c',
                'height_m', 'kf_pressure_pa', 'kf_height_m',
                'altitude_m', 'label_pressure_pa', 'label_source', 'label_time'
            ])
        if os.path.getsize(bmp280_path) == 0:
            self.labeled_bmp280_writer.writerow([
                'unix_time', 'sample_id', 'sensor', 'pressure_pa', 'temperature_c',
                'height_m', 'kf_pressure_pa', 'kf_height_m',
                'altitude_m', 'label_pressure_pa', 'label_source', 'label_time'
            ])

        print("\n[Labeled dataset] MS5611: {}".format(ms5611_path))
        print("[Labeled dataset] BMP280: {}".format(bmp280_path))

    def _fetch_label(self):
        """获取桂林雁山气压作为标签"""
        pressure_pa, update_time = fetch_guilin_pressure()
        if pressure_pa:
            self.current_label_pressure_pa = pressure_pa
            self.current_label_time = update_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return True
        return False

    def _start_new_period(self):
        """开始一个新的 15 分钟统计时段"""
        if self.period_stats and self.period_stats.sample_count > 0:
            # 保存上一个时段
            self.all_periods.append(self.period_stats)
            # 输出上一个时段的统计报告
            period_time = datetime.fromtimestamp(self.period_stats.start_ts)
            print("\n" + "=" * 60)
            print(self.period_stats.summary(
                title="[Period Report] {} ({})".format(
                    period_time.strftime("%H:%M"),
                    self.period_stats.period_label)))
            if self.current_label_pressure_pa:
                print("  Weather API label: {:.1f} Pa".format(self.current_label_pressure_pa))
            print("=" * 60)

        # 开始新时段
        now_ts = time.time()
        dt = datetime.now()
        # 对齐到最近的 15 分钟
        aligned_min = (dt.minute // 15) * 15
        period_start = dt.replace(minute=aligned_min, second=0, microsecond=0)
        label = period_start.strftime("%H:%M") + "-" + \
                (period_start + timedelta(minutes=15)).strftime("%H:%M")

        self.period_stats = PeriodStats(label)
        self.period_stats.start_ts = now_ts
        self.next_period_time = now_ts + self.period_duration

        # 获取一次标签气压
        print("\n[Label] Fetching Guilin Yanshan pressure for period {}...".format(label))
        if self._fetch_label():
            print("[Label] Got: {:.1f} Pa".format(self.current_label_pressure_pa))
        else:
            print("[Label] Failed, will retry")

    def _write_labeled(self, sensor, unix_time, sample_id, pressure, temp, height,
                       kf_p, kf_h):
        if self.current_label_pressure_pa is None:
            return

        writer = (self.labeled_ms5611_writer if sensor == "MS5611"
                  else self.labeled_bmp280_writer)
        file_obj = (self.labeled_ms5611_file if sensor == "MS5611"
                    else self.labeled_bmp280_file)
        if not writer or not file_obj:
            return

        with self.lock:
            writer.writerow([
                int(unix_time), sample_id, sensor, pressure, temp, height,
                kf_p, kf_h,
                self.altitude_m, self.current_label_pressure_pa, "weather_api",
                self.current_label_time
            ])
            file_obj.flush()

    def _process_line(self, line, current_unix_time):
        line = line.strip()
        if not line:
            return

        parts = line.split(',')
        if len(parts) < 2:
            return

        sensor_type = parts[0].strip()

        if sensor_type == "INFO":
            msg = ','.join(parts[1:])
            if "Altitude set to" in msg:
                try:
                    self.altitude_m = float(msg.split("to")[1].split("m")[0].strip())
                    print("\n[Config] Altitude updated to {:.1f} m".format(self.altitude_m))
                except Exception:
                    pass
            return

        if sensor_type == "FUSION":
            return

        if sensor_type in ("MS5611", "BMP280"):
            if len(parts) < 5:
                return
            try:
                sample_id = int(parts[1])
                pressure = float(parts[2])
                temp = float(parts[3])
                height = float(parts[4])
                kf_p = float(parts[5]) if len(parts) >= 6 else pressure
                kf_h = float(parts[6]) if len(parts) >= 7 else height
            except ValueError:
                return

            # 原始数据备份
            raw_row = [int(current_unix_time), sample_id, pressure, temp, height, kf_p, kf_h]
            if sensor_type == "MS5611" and self.raw_ms5611_writer:
                self.raw_ms5611_writer.writerow(raw_row)
                self.raw_ms5611_file.flush()
                self.ms5611_samples += 1
                self.raw_ms5611_msgs += 1
                # 加入时段统计
                if self.period_stats:
                    self.period_stats.add_ms5611(pressure, temp, height)
            elif sensor_type == "BMP280" and self.raw_bmp280_writer:
                self.raw_bmp280_writer.writerow(raw_row)
                self.raw_bmp280_file.flush()
                self.bmp280_samples += 1
                if self.period_stats:
                    self.period_stats.add_bmp280(pressure, temp, height)

            # 带标签数据集
            self._write_labeled(sensor_type, current_unix_time, sample_id,
                                pressure, temp, height, kf_p, kf_h)

    def _generate_day_report(self):
        """生成 24 小时全天数据分析报告"""
        if not self.all_periods:
            return "No data collected."

        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  24-HOUR DATA COLLECTION REPORT")
        lines.append(" 采集时间: {} - {}".format(
            datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M"),
            datetime.now().strftime("%Y-%m-%d %H:%M")))
        lines.append(" 海拔高度: {:.1f} m".format(self.altitude_m))
        lines.append("=" * 70)

        # 1. 总体统计
        all_ms_p = []
        all_bm_p = []
        all_ms_t = []
        all_bm_t = []
        all_ms_h = []
        all_bm_h = []
        labels = []

        for p in self.all_periods:
            all_ms_p.extend(p.ms5611_pressures)
            all_bm_p.extend(p.bmp280_pressures)
            all_ms_t.extend(p.ms5611_temps)
            all_bm_t.extend(p.bmp280_temps)
            all_ms_h.extend(p.ms5611_heights)
            all_bm_h.extend(p.bmp280_heights)

        lines.append("")
        lines.append("【1. 总体统计】")
        lines.append("  总采样数: {} (MS5611: {}, BMP280: {})".format(
            len(all_ms_p) + len(all_bm_p), len(all_ms_p), len(all_bm_p)))

        def stats(arr, label):
            if not arr:
                return
            mn, mx, avg = min(arr), max(arr), sum(arr) / len(arr)
            std = (sum((x - avg) ** 2 for x in arr) / len(arr)) ** 0.5
            lines.append("  {}: min={:.1f} max={:.1f} avg={:.1f} std={:.3f}".format(
                label, mn, mx, avg, std))

        stats(all_ms_p, "MS5611 气压(Pa)")
        stats(all_bm_p, "BMP280 气压(Pa)")
        stats(all_ms_t, "MS5611 温度(C)")
        stats(all_bm_t, "BMP280 温度(C)")
        stats(all_ms_h, "MS5611 高度(m)")
        stats(all_bm_h, "BMP280 高度(m)")

        # 2. 双传感器偏差
        lines.append("")
        lines.append("【2. 双传感器偏差分析】")
        if all_ms_p and all_bm_p:
            # 按时段计算平均偏差
            period_diffs = []
            for p in self.all_periods:
                if p.ms5611_pressures and p.bmp280_pressures:
                    ms_avg = sum(p.ms5611_pressures) / len(p.ms5611_pressures)
                    bm_avg = sum(p.bmp280_pressures) / len(p.bmp280_pressures)
                    period_diffs.append(ms_avg - bm_avg)

            if period_diffs:
                avg_diff = sum(period_diffs) / len(period_diffs)
                max_diff = max(period_diffs)
                min_diff = min(period_diffs)
                lines.append("  气压偏差(MS5611-BMP280):")
                lines.append("    平均: {:.2f} Pa  ({:.2f} m)".format(avg_diff, avg_diff / 11.3))
                lines.append("    最大: {:.2f} Pa  ({:.2f} m)".format(max_diff, max_diff / 11.3))
                lines.append("    最小: {:.2f} Pa  ({:.2f} m)".format(min_diff, min_diff / 11.3))

        # 3. 各时段趋势
        lines.append("")
        lines.append("【3. 各时段气压趋势】")
        lines.append("  {:<12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
            "时段", "MS5611 avg", "BMP280 avg", "偏差", "温度(MS)", "温度(BM)"))
        lines.append("  " + "-" * 72)
        for p in self.all_periods:
            ms_p_avg = (sum(p.ms5611_pressures) / len(p.ms5611_pressures)
                        if p.ms5611_pressures else 0)
            bm_p_avg = (sum(p.bmp280_pressures) / len(p.bmp280_pressures)
                        if p.bmp280_pressures else 0)
            ms_t_avg = (sum(p.ms5611_temps) / len(p.ms5611_temps)
                        if p.ms5611_temps else 0)
            bm_t_avg = (sum(p.bmp280_temps) / len(p.bmp280_temps)
                        if p.bmp280_temps else 0)
            diff = ms_p_avg - bm_p_avg
            lines.append("  {:<12s} {:>10.1f}Pa {:>10.1f}Pa {:>+8.1f}Pa {:>8.1f}C {:>8.1f}C".format(
                p.period_label, ms_p_avg, bm_p_avg, diff, ms_t_avg, bm_t_avg))

        # 4. 最大变化
        lines.append("")
        lines.append("【4. 24 小时气压变化幅度】")
        if all_ms_p and all_bm_p:
            ms_range = max(all_ms_p) - min(all_ms_p)
            bm_range = max(all_bm_p) - min(all_bm_p)
            ms_range_h = ms_range / 11.3
            bm_range_h = bm_range / 11.3
            lines.append("  MS5611: 变化 {:.1f} Pa ({:.1f} m)".format(ms_range, ms_range_h))
            lines.append("  BMP280: 变化 {:.1f} Pa ({:.1f} m)".format(bm_range, bm_range_h))
            lines.append("  MS5611 最低气压: {:.1f} Pa @ {}".format(
                min(all_ms_p),
                self._find_min_time(all_ms_p)))
            lines.append("  MS5611 最高气压: {:.1f} Pa @ {}".format(
                max(all_ms_p),
                self._find_max_time(all_ms_p)))

        lines.append("=" * 70)
        lines.append("")

        return '\n'.join(lines)

    def _find_min_time(self, arr):
        """找到最小值对应的时间（粗略）"""
        if not self.all_periods:
            return "--"
        # 简化：遍历所有时段找最小
        min_val = min(arr)
        cum = 0
        for p in self.all_periods:
            for v in p.ms5611_pressures:
                if v == min_val:
                    return datetime.fromtimestamp(p.start_ts + cum * 0.1).strftime("%H:%M")
                cum += 1
        return "--"

    def _find_max_time(self, arr):
        max_val = max(arr)
        cum = 0
        for p in self.all_periods:
            for v in p.ms5611_pressures:
                if v == max_val:
                    return datetime.fromtimestamp(p.start_ts + cum * 0.1).strftime("%H:%M")
                cum += 1
        return "--"

    def run(self, duration_hours):
        self.alive = True
        self.start_time = time.time()
        end_time = self.start_time + duration_hours * 3600
        line_buffer = ''

        self._open_raw_files()
        self._open_labeled_files()

        # 对齐到下一个 15 分钟边界
        now = datetime.now()
        next_quarter = (now.minute // 15 + 1) * 15
        if next_quarter >= 60:
            next_quarter = 0
            first_period_delay = ((60 - now.minute) * 60 - now.second)
        else:
            first_period_delay = ((next_quarter - now.minute) * 60 - now.second)

        self.next_period_time = time.time() + max(first_period_delay, 10)

        # 清空串口
        self.ser.reset_input_buffer()
        time.sleep(0.5)
        self.ser.reset_input_buffer()

        print("\n" + "=" * 60)
        print("Starting 24-hour data collection with 15-min auto-labeling")
        print("  Port: {}".format(self.ser.port))
        print("  Altitude: {:.1f} m".format(self.altitude_m))
        print("  Duration: {} hours".format(duration_hours))
        print("  Period: 15 minutes (label + report every 15 min)")
        print("  Data dir: {}".format(self.base_dir))
        print("  Press Ctrl+C to stop early")
        print("=" * 60)

        next_status_time = time.time() + 60
        fetch_retry_count = 0

        try:
            while time.time() < end_time:
                now_ts = time.time()
                current_unix_time = now_ts

                # 检查是否该开始/结束一个时段
                if self.period_stats is None or now_ts >= self.next_period_time:
                    self._start_new_period()
                    fetch_retry_count = 0
                elif self.current_label_pressure_pa is None:
                    # 当前时段还没有标签，尝试获取
                    if fetch_retry_count < 5:  # 最多重试 5 次
                        if self._fetch_label():
                            print("[Label] Got: {:.1f} Pa (retry)".format(
                                self.current_label_pressure_pa))
                            fetch_retry_count = 0
                        else:
                            fetch_retry_count += 1

                # 读取串口数据
                data = self.ser.read(1)
                if not data:
                    time.sleep(0.001)
                    continue

                try:
                    char = data.decode('utf-8', errors='ignore')
                except Exception:
                    continue

                line_buffer += char
                if '\n' in line_buffer:
                    lines = line_buffer.split('\n')
                    for line in lines[:-1]:
                        self._process_line(line, current_unix_time)
                    line_buffer = lines[-1]

                # 每 6 小时轮换文件
                if self.raw_ms5611_msgs >= 6 * 3600 * 10:
                    self._open_raw_files()

                # 每分钟状态
                if now_ts >= next_status_time:
                    elapsed = now_ts - self.start_time
                    remaining = end_time - now_ts
                    elapsed_str = str(timedelta(seconds=int(elapsed)))
                    remaining_str = str(timedelta(seconds=int(remaining)))
                    rate = self.ms5611_samples / (elapsed / 3600) if elapsed > 0 else 0

                    period_info = (self.period_stats.period_label
                                   if self.period_stats else "init")
                    label_info = ("{:.1f} Pa".format(self.current_label_pressure_pa)
                                  if self.current_label_pressure_pa else "wait")

                    print("\r[{elapsed}] MS:{ms} BM:{bm} ~{rate:.0f}/hr "
                          "[{period}] label={label} rem:{rem}".format(
                        elapsed=elapsed_str, ms=self.ms5611_samples,
                        bm=self.bmp280_samples, rate=rate,
                        period=period_info, label=label_info,
                        rem=remaining_str), end='')
                    sys.stdout.flush()
                    next_status_time = now_ts + 60

        except KeyboardInterrupt:
            print("\n\n[Stopped by user]")

        self.alive = False

        # 保存最后一个时段
        if self.period_stats and self.period_stats.sample_count > 0:
            self.all_periods.append(self.period_stats)

        # 关闭文件
        if self.raw_ms5611_file:
            self.raw_ms5611_file.close()
        if self.raw_bmp280_file:
            self.raw_bmp280_file.close()
        if self.labeled_ms5611_file:
            self.labeled_ms5611_file.close()
        if self.labeled_bmp280_file:
            self.labeled_bmp280_file.close()

        # 输出采集完成总结
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        print("\n" + "=" * 60)
        print("Collection finished")
        print("  Duration: {}".format(elapsed_str))
        print("  MS5611 samples: {}".format(self.ms5611_samples))
        print("  BMP280 samples: {}".format(self.bmp280_samples))
        print("=" * 60)

        # 输出 24 小时分析报告
        report = self._generate_day_report()
        print(report)

        # 同时保存报告到文件
        report_path = os.path.join(self.base_dir, "24h_report_{}.txt".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")))
        with open(report_path, 'w') as f:
            f.write(report)
        print("[Report] Saved to: {}".format(report_path))


def main():
    parser = argparse.ArgumentParser(
        description='24h pressure sensor data collector with 15-min auto-labeling')
    parser.add_argument('--port', type=str, default=None,
                        help='Serial port (e.g. COM3)')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION_HOURS,
                        help='Collection duration in hours (default: {})'.format(
                            DEFAULT_DURATION_HOURS))
    parser.add_argument('--altitude', type=float, default=160.0,
                        help='Current altitude in meters (default: 160.0)')
    parser.add_argument('--dir', type=str, default=None,
                        help='Data output directory (default: data/)')
    args = parser.parse_args()

    port = args.port or ask_for_port()

    try:
        ser = serial.Serial(port, 115200, timeout=0.01)
        print("Connected to {} at 115200 baud".format(port))
    except serial.SerialException as e:
        sys.stderr.write('Error opening serial port: {}\n'.format(e))
        sys.exit(1)

    collector = DataCollector24H(ser, altitude_m=args.altitude, data_dir=args.dir)
    collector.run(args.duration)
    ser.close()


if __name__ == '__main__':
    main()
