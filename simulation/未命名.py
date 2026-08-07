import matplotlib.pyplot as plt
import numpy as np

# 解决中文显示（更换为系统预装的兼容中文字体）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
ax.set_xlim(0, 10)
ax.set_ylim(-2, 12)
ax.set_axis_off()  # 隐藏坐标轴

# ---------------------- 基准水平线 ----------------------
# 标准气压平面 y=1
std_press_y = 1
ax.axhline(y=std_press_y, color='black', linewidth=1.2)
ax.text(5, std_press_y - 0.4, '标准气压平面\n(760mmHg，288K,101.325kPa)', ha='center', fontsize=10)

# 海平面 y=3
sea_y = 3
ax.axhline(y=sea_y, color='black', linewidth=1)
ax.text(9.2, sea_y, '海平面', va='center', fontsize=11)
# 海底虚线
ax.plot([7.2, 9.8], [sea_y-0.3, sea_y-0.3], color='black', linestyle='--', linewidth=0.8)

# ---------------------- 绘制地形轮廓 ----------------------
x_terrain = np.linspace(0, 10, 300)
terrain_y = np.zeros_like(x_terrain)

# 1. 机场平地 0~2
mask_airport = x_terrain < 2
terrain_y[mask_airport] = 2

# 2. 山体隆起 2~7
mask_mountain = (x_terrain >= 2) & (x_terrain <= 7)
peak_x, peak_y = 4.2, 9
terrain_y[mask_mountain] = 2 + 7 * np.exp(-((x_terrain[mask_mountain]-peak_x)/1.8)**2)

# 3. 山坡入海 7~8.2
idx_7 = np.argmin(np.abs(x_terrain - 7))
mask_slope = (x_terrain > 7) & (x_terrain <= 8.2)
x_slope = x_terrain[mask_slope]
terrain_y[mask_slope] = np.linspace(terrain_y[idx_7], sea_y, len(x_slope))

# 4. 海底区域 8.2~10
mask_sea = x_terrain > 8.2
x_sea = x_terrain[mask_sea]
terrain_y[mask_sea] = sea_y - np.linspace(0, 1.6, len(x_sea))

ax.plot(x_terrain, terrain_y, color='black', linewidth=1.3)

# ---------------------- 关键点位 ----------------------
airport_x, airport_y = 1.0, 2.0   # 机场地面
plane_x, plane_y = 4.2, 11.0       # 飞机位置（只画线，不画飞机图形）
mtn_x, mtn_y = peak_x, peak_y      # 飞机正下方山顶

# ---------------------- 绘制各类高度标注竖线 ----------------------
# 1. 标准气压高度：飞机 ↔ 标准气压平面
ax.plot([plane_x, plane_x], [std_press_y, plane_y], 'k', lw=1)
ax.text(plane_x-0.7, (std_press_y+plane_y)/2, '标准气压高度', rotation=90, va='center', fontsize=11)

# 2. 相对气压高度（相对高度）：飞机 ↔ 机场地面
ax.plot([plane_x, airport_x], [plane_y, airport_y], 'k--', lw=1)
ax.plot([airport_x, airport_x], [airport_y, plane_y], 'k', lw=1)
ax.text(airport_x+0.3, (airport_y+plane_y)/2, '相对气压高度', rotation=90, va='center', fontsize=11)

# 3. 真实高度：飞机 ↔ 正下方山顶地面
ax.plot([plane_x, plane_x], [mtn_y, plane_y], 'k', lw=1)
ax.text(plane_x+0.2, (mtn_y+plane_y)/2, '真实高度', rotation=90, va='center', fontsize=11)
# 地点标高：山顶 ↔ 标准气压平面
ax.plot([plane_x, plane_x], [std_press_y, mtn_y], 'k', lw=1)
ax.text(plane_x+0.2, (std_press_y+mtn_y)/2, '地点标高', rotation=90, va='center', fontsize=11)

# 4. 绝对气压高度（绝对高度）：飞机 ↔ 海平面
ax.plot([plane_x, plane_x], [sea_y, plane_y], 'k', lw=1)
ax.text(plane_x+1.2, (sea_y+plane_y)/2, '绝对气压高度', rotation=90, va='center', fontsize=11)

# 机场标高：机场地面 ↔ 标准气压平面
ax.plot([airport_x, airport_x], [std_press_y, airport_y], 'k', lw=1)
ax.text(airport_x-0.6, (std_press_y+airport_y)/2, '机场标高', rotation=90, va='center', fontsize=11)

# 机场标准气压高度
ax.plot([0.3, 0.3], [std_press_y, airport_y], 'k', lw=1)
ax.text(0.3-0.4, (std_press_y+airport_y)/2, '机场标准气压高度', rotation=90, va='center', fontsize=10)

# 图标题
plt.title('图2-3 飞行高度概念图', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# ===================== 文字概念讲解 =====================
doc = """
# 飞行高度核心概念：相对高度 vs 绝对高度
## 1. 绝对气压高度（简称：绝对高度）
- **基准面**：平均海平面
- **定义**：飞行器垂直距离海平面的高度，图中「绝对气压高度」竖线代表该值
- **使用场景**：高空巡航、航线飞行。所有飞机统一以海平面为基准划分高度层，避免空中相撞。

## 2. 相对气压高度（简称：相对高度 / 场压高度）
- **基准面**：本场机场跑道地面
- **定义**：飞行器垂直距离机场地面的高度，图中虚线连接飞机与机场、竖线为相对气压高度
- **使用场景**：起飞、五边进近、着陆阶段。飞行员直观判断距离跑道还有多高，保障落地安全。

## 补充3个辅助高度概念（图中全部标出）
1. **标准气压高度**
    基准：国际标准海平面101.325kPa，不受当地天气气压影响；用于飞机性能计算、跨区域高空飞行。
2. **真实高度（离地高度）**
    飞机垂直向下到正下方地面/山体的真实高度，用来判断是否会撞山、障碍物。
3. **标高（机场/地点标高）**
    地面某点到标准气压平面的垂直高度，代表机场、山峰自身的海拔基准。
"""
print(doc)
