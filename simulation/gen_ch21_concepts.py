# -*- coding: utf-8 -*-
"""生成 2.1 节两张面向初学者的概念图（单坐标系，全 py 绘制）。
图A sim_ch21_concept1.png：海拔(绝对高度) vs 室内相对高度 的基准对比示意图。
图B sim_ch21_concept2.png：传感器测压强 p -> 气压-高度模型 -> 输出高度 h 的因果示意图。
输出到 simulation/。"""
import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__))

# ============ 图A：高度基准概念 ============
def fig_concept_a():
    fig, ax = plt.subplots(figsize=(9.6, 6.0), dpi=150)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')

    # 海平面（蓝色线）
    ax.plot([0, 10], [2.0, 2.0], color='#2980b9', lw=2.5)
    ax.text(0.2, 2.1, '平均海平面（海拔基准 0 m）', color='#2980b9', fontsize=11, fontweight='bold')
    # 海底/地下
    ax.fill_between([0, 10], [0, 0], [2.0, 2.0], color='#d6eaf8', alpha=0.6)

    # 左：山（海拔示例）
    ax.plot([0.8, 1.6], [2.0, 5.5], color='#7f8c8d', lw=2)   # 山坡
    ax.plot([1.6, 2.4], [5.5, 2.0], color='#7f8c8d', lw=2)
    ax.scatter([1.6], [5.5], color='#27ae60', s=60, zorder=3)
    ax.annotate('山顶海拔 h≈3.5 m\n(从平均海平面量起)',
                xy=(1.6, 5.5), xytext=(3.0, 5.6),
                fontsize=10, color='#1e7a34',
                arrowprops=dict(arrowstyle='->', color='#27ae60'))

    # 右：楼（室内相对高度）
    bx0, by0, bw, bh = 5.2, 2.0, 2.6, 3.4
    ax.add_patch(plt.Rectangle((bx0, by0), bw, bh, fill=True, facecolor='#fdf2e9', edgecolor='#b9770e', lw=1.5))
    # 地板线 / 桌面线
    ax.plot([bx0, bx0+bw], [by0, by0], color='#b9770e', lw=2)
    desk_y = by0 + 1.4
    ax.plot([bx0+0.2, bx0+bw-0.2], [desk_y, desk_y], color='#8e44ad', lw=3)  # 桌面
    ax.text(bx0+bw+0.05, by0+0.1, '地板', fontsize=10, color='#b9770e')
    ax.text(bx0+bw+0.05, desk_y+0.05, '桌面', fontsize=10, color='#8e44ad')
    # 相对高度 Δh 标注
    ax.annotate('', xy=(bx0+bw+1.0, desk_y), xytext=(bx0+bw+1.0, by0),
                arrowprops=dict(arrowstyle='<->', color='#8e44ad', lw=1.8))
    ax.text(bx0+bw+1.05, (by0+desk_y)/2, '相对高度\nΔh≈0.75 m', fontsize=10, color='#8e44ad',
            va='center')

    ax.set_title('图A 绝对高度（海拔）与相对高度的区别', fontsize=12.5, fontweight='bold', color='#2c3e50', pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'sim_ch21_concept1.png'), dpi=150)
    plt.close(fig)
    print('saved concept1')

# ============ 图B：压强 -> 高度 ============
def fig_concept_b():
    fig, ax = plt.subplots(figsize=(9.6, 6.0), dpi=150)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')

    # 大气柱：越往上越稀（用渐变圆点表示）
    import numpy as np
    np.random.seed(1)
    xs, ys = [], []
    for h in np.linspace(1.0, 6.0, 12):
        n = int(14 - (h-1.0)*2)   # 越高点越少
        for _ in range(max(n, 3)):
            xs.append(np.random.uniform(1.0, 3.0))
            ys.append(h + np.random.uniform(-0.15, 0.15))
    ax.scatter(xs, ys, s=70, color='#aed6f1', alpha=0.9, zorder=2,
               label='空气分子（越往高处越稀疏）')
    ax.text(1.4, 6.3, '大气（空气柱）', fontsize=11, color='#21618c')
    # 高度箭头
    ax.annotate('', xy=(3.6, 6.0), xytext=(3.6, 1.0),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    ax.text(3.7, 3.5, '高度 h ↑', fontsize=11, color='#2c3e50')

    # 传感器
    ax.add_patch(FancyBboxPatch((4.4, 1.6), 1.6, 1.2,
                 boxstyle='round,pad=0.05', facecolor='#fdebd0', edgecolor='#ca6f1e', lw=1.5))
    ax.text(5.2, 2.2, 'BMP280\n传感器', ha='center', va='center', fontsize=10, color='#9c640c')
    ax.text(5.2, 1.4, '测到的是压强 p', ha='center', fontsize=9.5, color='#9c640c')

    # 模型框
    ax.add_patch(FancyBboxPatch((6.6, 3.2), 2.2, 1.4,
                 boxstyle='round,pad=0.05', facecolor='#e8f8f5', edgecolor='#117a65', lw=1.5))
    ax.text(7.7, 4.2, '气压—高度模型', ha='center', va='center', fontsize=10.5, color='#0e6655')
    ax.text(7.7, 3.7, 'h = f(p)', ha='center', va='center', fontsize=10, color='#0e6655')

    # 箭头：传感器->模型, 模型->高度输出
    arr1 = FancyArrowPatch((6.0, 2.2), (6.6, 3.9), arrowstyle='->', mutation_scale=14, color='#117a65', lw=1.8)
    ax.add_patch(arr1)
    arr2 = FancyArrowPatch((8.8, 3.9), (9.0, 5.6), arrowstyle='->', mutation_scale=14, color='#117a65', lw=1.8)
    ax.add_patch(arr2)
    ax.text(8.2, 5.9, '输出高度 h', fontsize=11, color='#0e6655', fontweight='bold')

    ax.set_title('图B 传感器测的是压强，高度由模型反算得到', fontsize=12.5, fontweight='bold', color='#2c3e50', pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'sim_ch21_concept2.png'), dpi=150)
    plt.close(fig)
    print('saved concept2')

if __name__ == '__main__':
    fig_concept_a()
    fig_concept_b()
    print('ALL CONCEPT FIGS DONE')
