#!/usr/bin/env python
#
# Data Collector for Pressure Sensors (MS5611 & BMP280)
# 支持两种采集模式：
#   1. labeled      — 人工标注模式，数据带 label_height / label_pressure
#   2. self_supervised — 自监督模式，纯原始数据
#
# 用法:
#   python data_collector.py                        # 交互式选择模式
#   python data_collector.py --mode labeled          # 直接进入标注模式
#   python data_collector.py --mode self_supervised  # 直接进入自监督模式
#

from __future__ import absolute_import

import codecs
import os
import sys
import threading
import csv
import argparse
from datetime import datetime

import serial
from serial.tools.list_ports import comports

try:
    raw_input
except NameError:
    raw_input = input


# ================================================================
#  Serial port helper
# ================================================================
def ask_for_port():
    sys.stderr.write('\n--- Available ports:\n')
    ports = []
    for n, (port, desc, hwid) in enumerate(sorted(comports()), 1):
        sys.stderr.write('--- {:2}: {:20} {!r}\n'.format(n, port, desc))
        ports.append(port)
    while True:
        sys.stderr.write('--- Enter port index or full name: ')
        port = raw_input('')
        try:
            index = int(port) - 1
            if not 0 <= index < len(ports):
                sys.stderr.write('--- Invalid index!\n')
                continue
        except ValueError:
            pass
        else:
            port = ports[index]
        return port


# ================================================================
#  DataCollector
# ================================================================
class DataCollector(object):
    def __init__(self, serial_instance, mode='self_supervised',
                 label_height=0.0, label_pressure=0.0):
        """
        mode: 'labeled' | 'self_supervised'
        label_height / label_pressure: 仅 labeled 模式生效
        """
        self.serial = serial_instance
        self.alive = False
        self._reader_alive = False
        self.receiver_thread = None
        
        self.mode = mode
        self.label_height = label_height
        self.label_pressure = label_pressure
        
        self.ms5611_writer = None
        self.bmp280_writer = None
        self.ms5611_file = None
        self.bmp280_file = None
        self.lock = threading.Lock()
        
        self.sample_count = 0
        self.ms5611_count = 0
        self.bmp280_count = 0

    def _create_data_files(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.mode == 'labeled':
            data_dir = os.path.join(os.path.dirname(__file__), "data", "labeled")
        else:
            data_dir = os.path.join(os.path.dirname(__file__), "data", "self_supervised")
        
        os.makedirs(data_dir, exist_ok=True)
        
        ms5611_path = os.path.join(data_dir, f"ms5611_{timestamp}.csv")
        bmp280_path = os.path.join(data_dir, f"bmp280_{timestamp}.csv")
        
        self.ms5611_file = open(ms5611_path, 'w', newline='')
        self.bmp280_file = open(bmp280_path, 'w', newline='')
        
        self.ms5611_writer = csv.writer(self.ms5611_file)
        self.bmp280_writer = csv.writer(self.bmp280_file)
        
        if self.mode == 'labeled':
            self.ms5611_writer.writerow([
                'sample_id', 'pressure_pa', 'temperature_c', 'height_m',
                'label_height_m', 'label_pressure_pa'
            ])
            self.bmp280_writer.writerow([
                'sample_id', 'pressure_pa', 'temperature_c', 'height_m',
                'label_height_m', 'label_pressure_pa'
            ])
        else:
            self.ms5611_writer.writerow([
                'sample_id', 'pressure_pa', 'temperature_c', 'height_m'
            ])
            self.bmp280_writer.writerow([
                'sample_id', 'pressure_pa', 'temperature_c', 'height_m'
            ])
        
        print(f"\nData files created [{self.mode} mode]:")
        print(f"  MS5611: {ms5611_path}")
        print(f"  BMP280: {bmp280_path}")

    def _close_data_files(self):
        if self.ms5611_file:
            self.ms5611_file.close()
            self.ms5611_file = None
        if self.bmp280_file:
            self.bmp280_file.close()
            self.bmp280_file = None

    def _start_reader(self):
        self._reader_alive = True
        self.receiver_thread = threading.Thread(target=self.reader, name='rx')
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

    def _stop_reader(self):
        self._reader_alive = False
        if hasattr(self.serial, 'cancel_read'):
            self.serial.cancel_read()
        self.receiver_thread.join()

    def start(self):
        self.alive = True
        self._create_data_files()
        self._start_reader()

    def stop(self):
        self.alive = False

    def join(self):
        if hasattr(self.serial, 'cancel_read'):
            self.serial.cancel_read()
        self.receiver_thread.join()
        self._close_data_files()

    def close(self):
        self.serial.close()

    def reader(self):
        try:
            line_buffer = ''
            while self._reader_alive:
                data = self.serial.read(1)
                if data:
                    try:
                        char = data.decode('utf-8', errors='ignore')
                        line_buffer += char
                        if '\n' in line_buffer:
                            lines = line_buffer.split('\n')
                            for line in lines[:-1]:
                                self._process_line(line.strip())
                            line_buffer = lines[-1]
                    except Exception:
                        pass
        except serial.SerialException:
            if self.alive:
                raise

    def _process_line(self, line):
        try:
            if not line:
                return
            
            parts = line.split(',')
            if len(parts) < 2:
                print(f"[RAW] {line}")
                return
            
            sensor_type = parts[0].strip()
            
            if sensor_type == "INFO":
                print(f"[INFO] {','.join(parts[1:])}")
                return
            
            if sensor_type == "FUSION":
                return  # data_collector 只采集原始传感器数据
            
            # 新格式: MS5611/BMP280,id,P_raw,T,H_raw,P_KF,H_KF (7字段)
            # 只取前5个字段写入CSV
            sample_id = int(parts[1])
            pressure_pa = float(parts[2])
            temperature_c = float(parts[3])
            height_m = float(parts[4])
            
            with self.lock:
                if sensor_type == "MS5611" and self.ms5611_writer:
                    if self.mode == 'labeled':
                        self.ms5611_writer.writerow([
                            sample_id, pressure_pa, temperature_c, height_m,
                            self.label_height, self.label_pressure
                        ])
                    else:
                        self.ms5611_writer.writerow([
                            sample_id, pressure_pa, temperature_c, height_m
                        ])
                    self.ms5611_file.flush()
                    self.ms5611_count += 1
                elif sensor_type == "BMP280" and self.bmp280_writer:
                    if self.mode == 'labeled':
                        self.bmp280_writer.writerow([
                            sample_id, pressure_pa, temperature_c, height_m,
                            self.label_height, self.label_pressure
                        ])
                    else:
                        self.bmp280_writer.writerow([
                            sample_id, pressure_pa, temperature_c, height_m
                        ])
                    self.bmp280_file.flush()
                    self.bmp280_count += 1
            
            if self.sample_count % 10 == 0:
                print(f"\rMS5611: {self.ms5611_count} | BMP280: {self.bmp280_count}", end='')
                sys.stdout.flush()
            
            self.sample_count += 1
                
        except Exception as e:
            print(f"\n[ERROR] Parse error: {str(e)}")


# ================================================================
#  main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Pressure Sensor Data Collector'
    )
    parser.add_argument(
        '--mode', choices=['labeled', 'self_supervised'],
        default=None,
        help='采集模式: labeled (人工标注) / self_supervised (自监督)'
    )
    parser.add_argument(
        '--height', type=float, default=None,
        help='标注模式下的参考高度 (m)'
    )
    parser.add_argument(
        '--pressure', type=float, default=None,
        help='标注模式下的参考气压 (Pa)'
    )
    args = parser.parse_args()
    
    # ---------- 确定模式 ----------
    mode = args.mode
    if mode is None:
        print("=== Pressure Sensor Data Collector ===")
        print()
        print("Select collection mode:")
        print("  1) Labeled (人工标注) — data saved to data/labeled/")
        print("  2) Self-Supervised (自监督) — data saved to data/self_supervised/")
        while True:
            choice = raw_input("Enter 1 or 2: ").strip()
            if choice == '1':
                mode = 'labeled'
                break
            elif choice == '2':
                mode = 'self_supervised'
                break
            print("Invalid choice, enter 1 or 2.")
    
    # ---------- 标注参数 ----------
    label_height = 0.0
    label_pressure = 0.0
    if mode == 'labeled':
        if args.height is not None:
            label_height = args.height
        else:
            while True:
                try:
                    label_height = float(raw_input("Enter reference height (m): "))
                    break
                except ValueError:
                    print("Invalid number.")
        
        if args.pressure is not None:
            label_pressure = args.pressure
        else:
            while True:
                try:
                    label_pressure = float(raw_input("Enter reference pressure (Pa) [0=skip]: "))
                    break
                except ValueError:
                    print("Invalid number.")
        
        print(f"\nLabeled mode:")
        print(f"  Reference height  = {label_height:.2f} m")
        print(f"  Reference pressure = {label_pressure:.1f} Pa")
    else:
        print(f"\nSelf-Supervised mode (no labels)")
    
    # ---------- 连接串口 ----------
    port = ask_for_port()
    
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        print(f"\nConnected to {port} at 115200 baud")
        
        collector = DataCollector(ser, mode=mode,
                                  label_height=label_height,
                                  label_pressure=label_pressure)
        collector.start()
        
        print("Collecting data... (Press Ctrl+C to stop)")
        print("=" * 60)
        
        while True:
            try:
                raw_input()
                break
            except KeyboardInterrupt:
                print("\n\nStopping...")
                break
        
        collector.stop()
        collector.join()
        collector.close()
        
        print("\nData collection stopped")
        print(f"MS5611 samples: {collector.ms5611_count}")
        print(f"BMP280 samples: {collector.bmp280_count}")
        if mode == 'labeled':
            print(f"All samples labeled with: height={label_height:.2f}m, pressure={label_pressure:.1f}Pa")
        
    except serial.SerialException as e:
        sys.stderr.write('Error opening serial port: {}\n'.format(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
