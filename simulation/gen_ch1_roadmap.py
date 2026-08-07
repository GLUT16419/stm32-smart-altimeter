# -*- coding: utf-8 -*-
"""气压测高技术演进路线图 —— “己”字回折走向（无图号标题）。
走向：上排主轴向右横走 → 末端折下（束）→ 下排向左横走回来，
形如“己”字横-折-横笔顺。9 节点分上下两排，上排5(右向)、下排4(左向)。
SimHei 中文、dpi160。"""
import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'sim_ch1_roadmap.png')

# (年份, 里程碑上句, 里程碑下句)
nodes = [
    ('1643', '托里拆利', '水银柱实验'),
    ('1648', '帕斯卡', '登山验高关系'),
    ('19C', '水银气压', '高度计工程化'),
    ('1900s', '机械式', '气压高度表'),
    ('1950s', '伺服振筒', '高度表'),
    ('1990s', '硅压阻MEMS', '数字化'),
    ('2000s', 'BMP数字', '传感器商用'),
    ('2016', 'BMP280', '亚帕级'),
    ('2020s', 'MEMS+自适应', '滤波融合'),
]

top = nodes[:5]     # 上排：从左到右（向右横走）
bot = nodes[5:]     # 下排：从右到左（向左横走）

MARGIN = 0.06
def xs_lr(count):   # 左→右
    return [MARGIN + i * (1 - 2 * MARGIN) / (count - 1) for i in range(count)]
def xs_rl(count):   # 右→左
    return [1 - MARGIN - i * (1 - 2 * MARGIN) / (count - 1) for i in range(count)]

xt = xs_lr(len(top))
xb = xs_rl(len(bot))

fig, ax = plt.subplots(figsize=(12, 9), dpi=160)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

Y_TOP = 0.74
Y_BOT = 0.26

# 己字主轴：上排向右横走
ax.plot([xt[0], xt[-1]], [Y_TOP, Y_TOP], color='#2c3e50', lw=4, zorder=2)
# 下排向左横走
ax.plot([xb[0], xb[-1]], [Y_BOT, Y_BOT], color='#2c3e50', lw=4, zorder=2)
# 末端折笔（束）：上排最右 -> 下排最右
ax.plot([xt[-1], xb[0]], [Y_TOP, Y_BOT], color='#2c3e50', lw=4, zorder=2)

def draw(x, y_line, yr, l1, l2, above):
    ax.scatter([x], [y_line], s=210, color='#c0392b', zorder=4,
               edgecolors='white', linewidths=2.5)
    if above:
        ax.text(x, y_line + 0.075, l1, ha='center', va='bottom',
                fontsize=15.5, color='#1a1a1a', fontweight='bold', zorder=4)
        ax.text(x, y_line + 0.043, l2, ha='center', va='top',
                fontsize=13.5, color='#34495e', zorder=4)
        ax.text(x, y_line - 0.052, yr, ha='center', va='top',
                fontsize=16, color='#c0392b', fontweight='bold', zorder=4)
    else:
        ax.text(x, y_line - 0.075, l1, ha='center', va='top',
                fontsize=15.5, color='#1a1a1a', fontweight='bold', zorder=4)
        ax.text(x, y_line - 0.043, l2, ha='center', va='bottom',
                fontsize=13.5, color='#34495e', zorder=4)
        ax.text(x, y_line + 0.052, yr, ha='center', va='bottom',
                fontsize=16, color='#c0392b', fontweight='bold', zorder=4)

for x, (yr, l1, l2) in zip(xt, top):
    draw(x, Y_TOP, yr, l1, l2, above=True)
for x, (yr, l1, l2) in zip(xb, bot):
    draw(x, Y_BOT, yr, l1, l2, above=False)

fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches='tight')
plt.close(fig)
print('saved', OUT)
