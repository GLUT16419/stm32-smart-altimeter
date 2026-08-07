import matplotlib.pyplot as plt
import numpy as np

# Windows中文设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
ax.set_xlim(0, 10)
ax.set_ylim(-1, 12)
ax.set_axis_off()

# ---------------------- 两条基准线（仅保留关键基准） ----------------------
# 平均海平面（绝对高度基准）
sea_y = 3
ax.axhline(y=sea_y, c='k', lw=1)
ax.text(9.3, sea_y, '平均海平面', va='center', fontsize=11)
ax.plot([7.5, 9.8], [sea_y-0.2, sea_y-0.2], 'k--', lw=0.7)

# ---------------------- 简化地形 ----------------------
x = np.linspace(0, 10, 250)
terrain = np.zeros_like(x)
# 左侧平地：本地参考基准面（室内地面）
terrain[x < 2] = 2
# 简化小山
peak_x, peak_h = 4.2, 9
mask_mount = (x >= 2) & (x <= 7)
terrain[mask_mount] = 2 + 7 * np.exp(-((x[mask_mount]-peak_x)/1.8)**2)
# 下坡入海
idx7 = np.argmin(np.abs(x - 7))
terrain[(x>7)] = np.linspace(terrain[idx7], sea_y, len(terrain[x>7]))
ax.plot(x, terrain, 'k', lw=1.2)

# ---------------------- 点位 ----------------------
ref_x, ref_h = 1.0, 2.0    # 本地参考地面
sensor_x, sensor_h = 4.2, 11# 气压测量点

# ====================== 1. 绝对高度（粗双向箭头，重点突出） ======================
ax.annotate(
    "",
    xy=(sensor_x, sea_y),
    xytext=(sensor_x, sensor_h),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5)
)
# 文字靠右放置，不遮挡线条
ax.text(sensor_x + 0.8, (sea_y + sensor_h)/2,
        "绝对高度\n基准：海平面",
        rotation=90, va="center", fontsize=12)

# ====================== 2. 相对高度（细双向箭头 + 斜虚线空间连线） ======================
# 斜虚线：两点空间连线
ax.plot([ref_x, sensor_x], [ref_h, sensor_h], 'k--', lw=1)
# 双向箭头竖线：竖直高差范围
ax.annotate(
    "",
    xy=(ref_x, ref_h),
    xytext=(ref_x, sensor_h),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1)
)
ax.text(ref_x + 0.3, (ref_h + sensor_h)/2,
        "相对高度\n基准：本地地面",
        rotation=90, va="center", fontsize=12)

# 标题简洁无编号
plt.title("高度概念示意图", fontsize=13, pad=15)
plt.tight_layout()
plt.show()

# 配套论文文字说明
doc = """
# 示意图概念说明
1. 绝对高度
以全球统一平均海平面为基准，双向箭头线段代表测量点到海平面的垂直距离，即海拔；测量需要海平面基准气压，适用于室外海拔检测。

2. 相对高度（高差）
以本地固定地面为参考基准，斜虚线表示测量点与参考点的空间位置关系，带双向箭头的竖线代表两点竖直高度差值；仅需对比两处气压差计算，不受全局气压波动干扰，适合室内高度测量。
"""
print(doc)