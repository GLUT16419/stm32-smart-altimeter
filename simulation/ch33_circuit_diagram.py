# -*- coding: utf-8 -*-
"""生成 3.3 节系统电路原理示意图 (图3-2)。图内标题不带图号。"""
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 8.5))
ax.set_xlim(0, 120); ax.set_ylim(0, 92); ax.axis('off')

def box(x, y, w, h, title, subtitle=None, fc='#dbeafe', ec='#1e40af', title_fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1,rounding_size=2.5",
                                linewidth=1.8, edgecolor=ec, facecolor=fc))
    ax.text(x+w/2, y+h-4, title, ha='center', va='top', fontsize=title_fs, weight='bold', color='#1e3a8a')
    if subtitle:
        ax.text(x+w/2, y+h-13, subtitle, ha='center', va='top', fontsize=8, color='#334155')
    return (x, y, w, h)

def pin(x, y, label, dx, dy, color='#b91c1c', ha='left', fs=8.5):
    ax.plot(x, y, 'ko', ms=4.5)
    ax.text(x+dx, y+dy, label, fontsize=fs, color=color, ha=ha, va='center')

# ---- 中央 MCU ----
mcu = box(46, 30, 28, 26, "STM32F411RET6", "Cortex-M4 100MHz 硬件FPU\nLQFP64", fc='#e0f2fe', ec='#0369a1')
mcx, mcy, mcw, mch = mcu

# 左侧引脚 (USART / SWD)
pin(mcx, mcy+20, "PA2 TX", -2, 0.6, color='#b91c1c', ha='right')
pin(mcx, mcy+15, "PA3 RX", -2, 0, color='#b91c1c', ha='right')
pin(mcx, mcy+10, "SWDIO", -2, 0.6, color='#64748b', ha='right')
pin(mcx, mcy+5,  "SWCLK", -2, 0, color='#64748b', ha='right')
# 右侧引脚 (SPI)
pin(mcx+mcw, mcy+21, "PB8 SCK", 2, 0.6, color='#15803d', ha='left')
pin(mcx+mcw, mcy+16, "PB9 SDI", 2, 0, color='#15803d', ha='left')
pin(mcx+mcw, mcy+11, "PC10 SDO", 2, -0.5, color='#15803d', ha='left')
pin(mcx+mcw, mcy+6,  "PC8 CSB", 2, 0, color='#15803d', ha='left')
# 上侧引脚 (I2C)
pin(mcx+8, mcy+mch, "PC0 SCL", 0, 1.5, color='#a16207', ha='center')
pin(mcx+16, mcy+mch, "PC1 SDA", 0, 1.5, color='#a16207', ha='center')
# 下侧电源
pin(mcx+8, mcy, "VDD", 0, -1.5, color='#3730a3', ha='center')
pin(mcx+18, mcy, "VSS", 0, -1.5, color='#3730a3', ha='center')

# ---- 外围 ----
bmp = box(84, 34, 22, 18, "BMP280", "数字 MEMS 气压传感器\n软件 SPI", fc='#dcfce7', ec='#15803d')
oled = box(50, 64, 20, 13, "OLED SSD1306", "0.96\" 128×64 单色\n软件模拟 I²C", fc='#fef9c3', ec='#a16207')
ch = box(4, 34, 20, 18, "CH340", "USB 转串口", fc='#fee2e2', ec='#b91c1c')
ldo = box(50, 4, 20, 12, "AMS1117-3.3", "5V → 3.3V", fc='#e0e7ff', ec='#3730a3')

# 连线 (尽量直角, 避免交叉)
# MCU 右 -> BMP280 左 (4线)
for k, yy in enumerate([21, 16, 11, 6]):
    yb = bmp[1]+bmp[3]-5 - k*3
    ax.plot([mcx+mcw, (mcx+mcw+bmp[0])/2, bmp[0]], [mcy+yy, mcy+yy, yb], color='#15803d', lw=1.2)
    ax.plot([(mcx+mcw+bmp[0])/2, bmp[0]], [yb, yb], color='#15803d', lw=1.2)
# MCU 上 -> OLED 下
x_os = [oled[0]+5, oled[0]+13]
for k, xo in enumerate(x_os):
    xm = mcx+8 if k==0 else mcx+16
    ax.plot([xm, xo], [mcy+mch, oled[1]], color='#a16207', lw=1.2)
# MCU 左 -> CH340 右 (交叉, USART)
ax.plot([mcx, ch[0]+ch[2]], [mcy+20, ch[1]+12], color='#b91c1c', lw=1.2)
ax.plot([mcx, ch[0]+ch[2]], [mcy+15, ch[1]+8], color='#b91c1c', lw=1.2)
# MCU 下 -> LDO 上
x_lm = [ldo[0]+6, ldo[0]+14]
for k, xl in enumerate(x_lm):
    xm = mcx+8 if k==0 else mcx+18
    ax.plot([xm, xl], [mcy, ldo[1]+ldo[3]], color='#3730a3', lw=1.2)

# 上拉 / 交叉说明

# 接口协议说明框 (四角, 不重叠)
def proto_box(x, y, text, color, fc):
    ax.text(x, y, text, fontsize=9, color=color, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc=fc, ec=color, lw=0.9, alpha=0.95))

# 右上: BMP280 SPI
proto_box(88, 10,
          "BMP280 接口协议\n软件 SPI 4 线\n• CPOL=0, CPHA=0\n• 读命令: 寄存器地址 | 0x80\n• 寄存器地址自动递增",
          '#15803d', '#f0fdf4')
# 左上: OLED 软件 I2C
proto_box(4, 75,
          "OLED 接口协议\n软件模拟 I²C\n• GPIO 模拟 SCL/SDA\n• 起始/停止/ACK 全软件实现\n• 设备地址 0x78, 速率约 400kHz\n• SCL/SDA 4.7kΩ 上拉至 3.3V",
          '#a16207', '#fefce8')
# 左下: USART 调试
proto_box(4, 10,
          "调试接口协议\n硬件 USART2\n• 115200-8-N-1\n• 输出原始/滤波气压、高度",
          '#b91c1c', '#fef2f2')
# 右下: 电源
proto_box(88, 72,
          "电源与抗干扰\n• 5V→3.3V LDO\n• 0.1μF + 1μF 退耦\n• 磁珠单点地",
          '#3730a3', '#eef2ff')

ax.set_title('系统硬件电路原理示意图（接口与引脚标注）', fontsize=12, weight='bold', pad=10)
plt.tight_layout()
plt.savefig('sim_ch3_3-2.png', dpi=150, bbox_inches='tight')
print('→ sim_ch3_3-2.png')
