import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, init_x, init_p, q, r):
        self.x = init_x
        self.p = init_p
        self.q = q
        self.q_base = q
        self.r = r
        self.k = 0.0
    
    def update(self, z):
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (z - self.x)
        self.p = (1 - self.k) * self.p
        return self.x
    
    def update_adaptive(self, z, residual_th=2.5, q_inc=1.08, q_dec=0.992, q_min=0.1, q_max=50.0):
        residual = z - self.x
        residual_abs = abs(residual)
        if residual_abs > residual_th:
            self.q = self.q * q_inc
            if self.q > q_max:
                self.q = q_max
        else:
            self.q = self.q * q_dec
            if self.q < self.q_base:
                self.q = self.q_base
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * residual
        self.p = (1 - self.k) * self.p
        return self.x

# 生成含阶跃的合成数据
np.random.seed(42)
base = 98700.0
n = 300
t = np.arange(n)
signal = np.zeros(n)
signal[:100] = base
signal[100:200] = base - 20  # 阶跃下降
signal[200:] = base - 5      # 部分恢复
noise = np.random.normal(0, 2.5, n)  # 模拟 MS5611 噪声
z = signal + noise

# 测试三种配置
configs = [
    ("Old: Q=5, R=100", 5.0, 100.0, False),
    ("New fixed: Q=2.8, R=20", 2.8, 20.0, False),
    ("New adaptive: Q=2.8, R=20, th=2.5", 2.8, 20.0, True),
]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(z, 'gray', alpha=0.3, label='Raw noisy')
axes[0].plot(signal, 'k--', alpha=0.7, label='True signal')

for label, q, r, adaptive in configs:
    kf = KalmanFilter(z[0], 1000.0, q, r)
    out = np.zeros(n)
    for i, val in enumerate(z):
        if adaptive:
            out[i] = kf.update_adaptive(val)
        else:
            out[i] = kf.update(val)
    axes[0].plot(out, label=label)
    
    rmse = np.sqrt(np.mean((out - signal)**2))
    print(f"{label}: RMSE={rmse:.3f} Pa")

axes[0].set_ylabel('Pressure (Pa)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Step Response Test')

# 放大看阶跃响应
axes[1].plot(signal, 'k--', alpha=0.7, label='True signal')
for label, q, r, adaptive in configs:
    kf = KalmanFilter(z[0], 1000.0, q, r)
    out = np.zeros(n)
    for i, val in enumerate(z):
        if adaptive:
            out[i] = kf.update_adaptive(val)
        else:
            out[i] = kf.update(val)
    axes[1].plot(out, label=label)
axes[1].set_xlim(95, 130)
axes[1].set_ylabel('Pressure (Pa)')
axes[1].set_xlabel('Sample index')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Step Response Detail (samples 95-130)')

plt.tight_layout()
plt.savefig('kalman_step_response_test.png', dpi=150, bbox_inches='tight')
print("Saved: kalman_step_response_test.png")
plt.close()
