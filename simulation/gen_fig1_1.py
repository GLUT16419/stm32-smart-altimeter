# -*- coding: utf-8 -*-
"""重绘图1-1：绝对高度(海拔)与相对高度(高差)概念对比示意图。
改进点(相对原 1.py)：
  - 字号整体放大(标题15 / 基准标签13~14 / 注释11~12 / 数值12)，不再“字太小”
  - 画布紧凑、左右分区(左=室外海拔、右=室内高差)，充分占满空间，减少无谓留白
  - 两处粗双向箭头(红色海拔、蓝色高差) + 数值标注，概念一目了然
  - dpi 提到 150，插入论文仍清晰
输出：simulation/fig1_1.png（仅出图，不改文档）
"""
import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__), 'fig1_1.png')

fig, ax = plt.subplots(figsize=(13.0, 6.3), dpi=150)
ax.set_xlim(0, 13); ax.set_ylim(0, 7.0); ax.axis('off')

# ---------- 公共：海平面(蓝色) ----------
sea_y = 1.6
ax.plot([0, 13], [sea_y, sea_y], color='#2980b9', lw=2.5, zorder=1)
ax.text(0.15, sea_y + 0.12, '平均海平面（海拔基准 0 m）', color='#2980b9',
        fontsize=13, fontweight='bold')
ax.fill_between([0, 13], [0, 0], [sea_y, sea_y], color='#d6eaf8', alpha=0.55, zorder=0)

# ===================== 左区：绝对高度(海拔) =====================
# 山
mx0, mx1, peak_x, peak_h = 0.8, 3.4, 2.1, 5.4
ax.plot([mx0, peak_x], [sea_y, peak_h], color='#7f8c8d', lw=2.2, zorder=2)
ax.plot([peak_x, mx1], [peak_h, sea_y], color='#7f8c8d', lw=2.2, zorder=2)
ax.scatter([peak_x], [peak_h], color='#27ae60', s=70, zorder=4)
# 绝对高度：海平面 -> 山顶 粗红双向箭头
arrow_abs = FancyArrowPatch((peak_x, sea_y), (peak_x, peak_h),
                              arrowstyle='<->', mutation_scale=18,
                              color='#e74c3c', lw=2.8, zorder=5)
ax.add_patch(arrow_abs)
ax.text(peak_x + 0.18, (sea_y + peak_h) / 2, '绝对高度\n(海拔)\n≈ 3.8 m',
        fontsize=12, color='#e74c3c', va='center', fontweight='bold')
ax.text(peak_x, peak_h + 0.18, '山顶', fontsize=11, color='#1e7a34', ha='center')

# 左标题块
ax.text(2.0, 6.55, '（一）绝对高度 —— 以平均海平面为统一基准',
        fontsize=13.5, fontweight='bold', ha='center', color='#c0392b')
ax.text(2.0, 0.35, '室外/野外适用：需获知海平面基准气压',
        fontsize=11, ha='center', color='#7f8c8d', style='italic')

# ===================== 右区：相对高度(高差) =====================
# 楼 + 室内地面
bx0, by0, bw, bh = 8.2, sea_y, 3.4, 3.4
ax.add_patch(mpatches.Rectangle((bx0, by0), bw, bh, facecolor='#fdf2e9',
                                          edgecolor='#b9770e', lw=1.8, zorder=2))
ax.plot([bx0, bx0 + bw], [by0, by0], color='#b9770e', lw=2.5, zorder=3)  # 地板线
desk_y = by0 + 1.5
ax.plot([bx0 + 0.6, bx0 + bw - 0.6], [desk_y, desk_y], color='#8e44ad', lw=2.0, ls='--', zorder=3)  # 桌面线
ax.scatter([bx0 + bw / 2], [desk_y], color='#8e44ad', s=70, zorder=4)
# 相对高度：室内地面 -> 桌面 粗蓝双向箭头
arrow_rel = FancyArrowPatch((bx0 + bw / 2, by0), (bx0 + bw / 2, desk_y),
                              arrowstyle='<->', mutation_scale=18,
                              color='#2980b9', lw=2.8, zorder=5)
ax.add_patch(arrow_rel)
ax.text(bx0 + bw / 2 + 0.22, (by0 + desk_y) / 2, '相对高度\n(高差)\nΔh ≈ 0.75 m',
        fontsize=12, color='#2471a3', va='center', fontweight='bold')
ax.text(bx0 + bw / 2, desk_y + 0.18, '桌面', fontsize=11, color='#2471a3', ha='center')

# 右标题块
ax.text(9.9, 6.55, '（二）相对高度 —— 以本地地面/桌面为参考基准',
        fontsize=13.5, fontweight='bold', ha='center', color='#2471a3')
ax.text(9.9, 0.35, '室内适用：只比两处气压差，抗全局气压波动',
        fontsize=11, ha='center', color='#7f8c8d', style='italic')

# 中间对勾说明
ax.annotate('', xy=(4.6, 3.5), xytext=(7.4, 3.5),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))
ax.text(6.0, 3.95, '同一只气压计\n换基准即换视角', fontsize=11,
        ha='center', color='#555555')

plt.title('图 1-1  绝对高度与相对高度概念图', fontsize=15, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print('已写出:', OUT)
