# -*- coding: utf-8 -*-
"""
高度融合算法 —— 与嵌入式固件 (Core/Src/main.c + kalman_filter.c) 一致的 Python 移植
=================================================================================
本模块忠实复现固件的传感器处理与双传感器融合流程，供 PC 端 GUI 调参使用。

复现内容：
  * 气压 -> 高度转换 (PressureToAltitudeWithTemp，含温度补偿)
  * 一维标准卡尔曼滤波 (KalmanFilter)
  * 自适应卡尔曼滤波 (MS5611 版 / BMP280 版，基于 5 帧残差 STD 自适应调 Q)
  * NN 滤波 (可选；启用时需 tflite_runtime，否则自动退化为 KF)
  * 校准 (参考气压锁定 + BMP280 绝对偏置 bmp_bias)
  * 14 种融合方案 (FUSION_SCHEME 1..14)，含 HPF / 自适应权重 / Delta 累积 / 二阶互补 / 逆方差
  * EMA 显示平滑 (气压 alpha + 高度 alpha)

所有默认值取自 kalman_filter.h / main.c 中的宏定义。
"""

import os
import math
import numpy as np

from scene_classifier import SceneClassifier, CLASS_NAMES

# ============================================================
# 1. 常量 (与固件 altitude_convert / main.c 一致)
# ============================================================
SEA_LEVEL_PRESSURE_PA = 101325.0
ISA_T0 = 288.15
ISA_L = 0.0065
ISA_G = 9.80665
ISA_R = 287.05

# NN 滤波归一化参数 (NN_Filter_Update 使用)
NN_REF_PRESSURE = 101325.0
NN_REL_MIN = -3426.0
NN_REL_MAX = -2534.72
NN_REL_RANGE = NN_REL_MAX - NN_REL_MIN
NN_SENSOR_MS5611 = 0.0
NN_SENSOR_BMP280 = 1.0

# KF 默认参数 (kalman_filter.h)
KF_Q_MIN = 0.01
KF_Q_MAX = 2.0
KF_RESIDUAL_TH = 5.0
KF_Q_INCREASE = 1.05
KF_Q_DECREASE = 0.98

BMP280_RESIDUAL_TH = 1.2
BMP280_Q_INCREASE = 1.08
BMP280_Q_DECREASE = 0.97
BMP280_Q_MAX = 5.0

MS5611_KF_Q = 0.03
MS5611_KF_R = 9.3
BMP280_KF_Q = 0.005
BMP280_KF_R = 1.0

KF_INIT_P = 1000.0

# NN 二次滤波 KF 默认参数
NN_KF_Q = 0.01
NN_KF_R = 0.5

# 默认融合相关宏
FUSION_WEIGHT_MS5611_DEFAULT = 0.5
FUSION_WEIGHT_BMP280_DEFAULT = 0.5
HPF_ALPHA_DEFAULT = 0.2
MOTION_WINDOW_SIZE = 8
MOTION_THRESHOLD_PA_DEFAULT = 2.0
WEIGHT_STATIC_MS_DEFAULT = 0.05
WEIGHT_MOTION_MS_DEFAULT = 0.40
WEIGHT_SMOOTH_ALPHA_DEFAULT = 0.1
W_DELTA_MS_STATIC_DEFAULT = 0.05
W_DELTA_MS_MOTION_DEFAULT = 0.50
DELTA_WEIGHT_SMOOTH_ALPHA_DEFAULT = 0.1
IVAR_WINDOW_SIZE = 10
IVAR_EPSILON_DEFAULT = 0.1
DELTA_CONF_WINDOW = 10
DELTA_CONF_EPS_DEFAULT = 0.05
ANCHOR_ALPHA_DEFAULT = 0.02
COMP_ALPHA_DEFAULT = 0.02
COMP_BETA_DEFAULT = 0.5
TC_COEFF_DEFAULT = 0.0        # 方案14 温漂补偿系数 (Pa/℃)，0 表示关闭
PRESSURE_EMA_ALPHA_DEFAULT = 0.4
HEIGHT_EMA_ALPHA_DEFAULT = 0.5
PULL_COEFF = 0.003            # 方案7 静止拉力 (固定)


# ============================================================
# 2. 气压 -> 高度 (与 altitude_convert.c 的 PressureToAltitudeWithTemp 一致)
# ============================================================
def pressure_to_altitude_with_temp(pressure_pa, temperature_c,
                                    ref_pressure_pa=SEA_LEVEL_PRESSURE_PA):
    if pressure_pa <= 0 or ref_pressure_pa <= 0:
        return 0.0
    ratio = pressure_pa / ref_pressure_pa
    if ratio <= 0 or ratio > 10:
        return 0.0
    exponent = ISA_L * ISA_R / ISA_G
    altitude = (ISA_T0 / ISA_L) * (1.0 - ratio ** exponent)
    temperature_k = temperature_c + 273.15
    t_isa = ISA_T0 - ISA_L * altitude
    delta_t = temperature_k - t_isa
    altitude += delta_t * 0.035
    return altitude


def altitude_to_pressure(altitude_m, temperature_c, ref_pressure_pa=SEA_LEVEL_PRESSURE_PA):
    """PressureToAltitudeWithTemp 的逆运算（同样含温度补偿项）。

    给定高度(米)与温度(℃)，返回对应气压(Pa)。用于合成数据集生成，
    保证『高度↔气压』全程只使用单片机的气压-高度公式，而不再依赖
    线性近似 (PA_PER_M)。
    """
    exponent = ISA_L * ISA_R / ISA_G
    t_k = temperature_c + 273.15
    # 由 altitude = (T0/L)*(1 - ratio**exponent) + (t_k - (T0 - L*alt_nc))*0.035
    # 反解未加温度补偿的高度 alt_nc
    denom = 1.0 + 0.035 * ISA_L
    alt_nc = (altitude_m - 0.035 * t_k + 0.035 * ISA_T0) / denom
    arg = 1.0 - alt_nc * ISA_L / ISA_T0
    if arg <= 0:
        return 0.0
    ratio = arg ** (1.0 / exponent)
    return ref_pressure_pa * ratio


# ============================================================
# 3. 卡尔曼滤波器
# ============================================================
class KalmanFilter:
    """标准一维卡尔曼滤波 (kalman_filter.c: KalmanFilter_Update)"""
    def __init__(self, init_x, init_p, q, r):
        self.x = init_x
        self.p = init_p
        self.q = q
        self.r = r
        self.k = 0.0
        self.q_base = q

    def update(self, z):
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (z - self.x)
        self.p = (1 - self.k) * self.p
        return self.x


class AdaptiveKalmanFilter(KalmanFilter):
    """自适应卡尔曼滤波 (kalman_filter.c: KalmanFilter_Update_Adaptive / _BMP280)

    参数：
        sensor: 'ms5611' 或 'bmp280'，决定默认阈值与 Q 变化速率
        residual_th / q_inc / q_dec / q_max 可由 GUI 覆盖
    """
    def __init__(self, init_x, init_p, q, r, sensor='bmp280',
                 residual_th=None, q_inc=None, q_dec=None, q_max=None):
        super().__init__(init_x, init_p, q, r)
        if sensor == 'ms5611':
            self.residual_th = residual_th if residual_th is not None else KF_RESIDUAL_TH
            self.q_inc = q_inc if q_inc is not None else KF_Q_INCREASE
            self.q_dec = q_dec if q_dec is not None else KF_Q_DECREASE
            self.q_max = q_max if q_max is not None else KF_Q_MAX
        else:
            self.residual_th = residual_th if residual_th is not None else BMP280_RESIDUAL_TH
            self.q_inc = q_inc if q_inc is not None else BMP280_Q_INCREASE
            self.q_dec = q_dec if q_dec is not None else BMP280_Q_DECREASE
            self.q_max = q_max if q_max is not None else BMP280_Q_MAX
        self.residual_window = [0.0] * 5
        self.residual_idx = 0

    def update(self, z):
        residual = z - self.x
        residual_abs = abs(residual)
        self.residual_window[self.residual_idx] = residual_abs
        self.residual_idx = (self.residual_idx + 1) % 5
        mean = sum(self.residual_window) / 5.0
        var = sum((w - mean) ** 2 for w in self.residual_window) / 5.0
        residual_std = math.sqrt(var)
        if residual_std > self.residual_th:
            self.q = self.q * self.q_inc
            if self.q > self.q_max:
                self.q = self.q_max
        else:
            self.q = self.q * self.q_dec
            if self.q < self.q_base:
                self.q = self.q_base
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * residual
        self.p = (1 - self.k) * self.p
        return self.x


# ============================================================
# 4. NN 滤波 (可选)
# ============================================================
class NNFilter:
    """NN 自监督气压滤波 (main.c: NN_Filter_Update)

    启用真实 TFLM 模型需要 tflite_runtime；若不可用则 is_available=False，
    调用 update() 直接回退为原始输入（等价于固件 WORK_MODE!=0 的退化路径）。
    """
    def __init__(self, sensor_type='bmp280', window_size=10, model_path=None):
        self.sensor_type = NN_SENSOR_MS5611 if sensor_type == 'ms5611' else NN_SENSOR_BMP280
        self.window_size = window_size
        self.window = [0.0] * window_size
        self.index = 0
        self.filled = 1
        self.last_output = 0.0
        self.interpreter = None
        self.is_available = False
        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.is_available = True
        except Exception as e:
            print(f"[NN] 无法加载模型 {model_path}: {e}，将退化为原始输入。")
            self.is_available = False

    def update(self, z):
        self.window[self.index] = z
        self.index = (self.index + 1) % self.window_size
        if not self.is_available:
            self.last_output = z
            return z
        # 构造 10 窗 + sensor_type 的输入
        in_buf = []
        for i in range(self.window_size):
            raw = self.window[(self.index + i) % self.window_size]
            rel = raw - NN_REF_PRESSURE
            norm = (rel - NN_REL_MIN) / NN_REL_RANGE
            in_buf.append(norm)
        in_buf.append(self.sensor_type)
        inp = np.array([in_buf], dtype=np.float32)
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])
        nn_output = (float(out[0][0]) * NN_REL_RANGE + NN_REL_MIN) + NN_REF_PRESSURE
        # 合理性检查：偏离原始输入过大则回退
        if abs(nn_output - z) > 50.0:
            self.last_output = z
        else:
            self.last_output = nn_output
        return self.last_output


# ============================================================
# 4b. 联合多任务模型滤波 (复刻固件 Multitask_Run + baseline_eqw)
# ============================================================
MT_WINDOW_DEFAULT = 10


class MultitaskFilter:
    """复刻固件 Multitask_Run：维护 MS5611 / BMP280 两个 10 点绝对气压窗口（共用索引，
    时间对齐），标准化后一次性联合推理，输出：
        * base_ms  —— 联合模型 OUT_1 滤波后的 MS5611 绝对气压 (Pa)
        * base_bmp —— 联合模型 OUT_2 滤波后的 BMP280 绝对气压 (Pa)
        * scene    —— OUT_3 场景概率 [static, elevation] (softmax)
    推理特征与固件完全一致：
        in[i]   = (MS5611_win[i]  - 101325) - FEAT_MEAN[i]    / FEAT_STD[i]   (i=0..9)
        in[10+i]= (BMP280_win[i]  - 101325) - FEAT_MEAN[10+i] / FEAT_STD[10+i] (i=0..9)
    反归一化：rel = out*target_std + target_mean；绝对气压 = rel + 101325。
    """

    def __init__(self, model_prefix, window=MT_WINDOW_DEFAULT, ref_pressure=101325.0):
        from multitask_infer import MultitaskModel
        self.model = MultitaskModel(model_prefix)   # 传入 .tflite 完整路径
        self.window = window
        self.ref = ref_pressure
        self.fmean = np.asarray(self.model.fmean, dtype=float)
        self.fstd = np.asarray(self.model.fstd, dtype=float)
        self.t_mean = float(self.model.t_mean)
        self.t_std = float(self.model.t_std)
        self.class_names = list(self.model.class_names)
        self.reset(0.0, 0.0)

    def reset(self, init_ms, init_bmp):
        self.win_ms = [float(init_ms)] * self.window
        self.win_bmp = [float(init_bmp)] * self.window
        self.idx = 0
        self.n_ms = self.window     # 冷启动预填，首帧即可推理（与固件 Multitask_Init 一致）
        self.n_bmp = self.window
        self.ready = False
        self.scene = np.array([0.5, 0.5], dtype=float)

    def push(self, ms_pa, bmp_pa):
        self.win_ms[self.idx] = float(ms_pa)
        self.win_bmp[self.idx] = float(bmp_pa)
        if self.n_ms < self.window:
            self.n_ms += 1
        if self.n_bmp < self.window:
            self.n_bmp += 1
        self.idx = (self.idx + 1) % self.window   # 双窗口共用同一索引推进，保证时间对齐

    def run(self):
        if self.n_ms < self.window or self.n_bmp < self.window:
            return None
        X = np.zeros(self.window * 2, dtype=np.float32)
        for i in range(self.window):
            p = (self.idx + i) % self.window        # 旧 -> 新 时间序
            rel = self.win_ms[p] - self.ref
            X[i] = (rel - self.fmean[i]) / self.fstd[i]
            rel = self.win_bmp[p] - self.ref
            X[self.window + i] = (rel - self.fmean[self.window + i]) / self.fstd[self.window + i]
        x = X.reshape(1, -1).astype(np.float32)     # prep == 'raw'
        self.model.interp.set_tensor(self.model.in_details[0]['index'], x)
        self.model.interp.invoke()
        out = [self.model.interp.get_tensor(d['index']) for d in self.model.out_details]
        rel_ms = float(out[0].flatten()[0]) * self.t_std + self.t_mean
        rel_bmp = float(out[1].flatten()[0]) * self.t_std + self.t_mean
        self.ready = True
        self.scene = np.array(out[2].flatten(), dtype=float)
        return rel_ms + self.ref, rel_bmp + self.ref, self.scene


def run_joint_model(ms_p, bmp_p, model_prefix, window=MT_WINDOW_DEFAULT, ref=101325.0):
    """对整段序列预计算联合模型输出（base_ms / base_bmp / scene_prob）。

    返回的数组与输入等长；自动调参时可复用，避免每个候选参数都重新跑模型。
    """
    ms_p = np.asarray(ms_p, dtype=float)
    bmp_p = np.asarray(bmp_p, dtype=float)
    n = len(ms_p)
    f = MultitaskFilter(model_prefix, window, ref)
    f.reset(ms_p[0], bmp_p[0])
    base_ms = np.zeros(n)
    base_bmp = np.zeros(n)
    scene = np.zeros((n, 2), dtype=float)
    for i in range(n):
        f.push(ms_p[i], bmp_p[i])
        r = f.run()
        if r is None:
            base_ms[i], base_bmp[i] = ms_p[i], bmp_p[i]
            scene[i] = [0.5, 0.5]
        else:
            base_ms[i], base_bmp[i], scene[i] = r
    return base_ms, base_bmp, scene


# ============================================================
# 5. 参数集合 (GUI 填充后传入 simulate)
# ============================================================
class AlgoParams:
    """集中存放所有可调参数，默认值与固件宏一致。"""
    def __init__(self):
        # MS5611 KF
        self.ms_q = MS5611_KF_Q
        self.ms_r = MS5611_KF_R
        self.ms_residual_th = KF_RESIDUAL_TH
        self.ms_q_inc = KF_Q_INCREASE
        self.ms_q_dec = KF_Q_DECREASE
        self.ms_q_max = KF_Q_MAX
        # BMP280 KF
        self.bmp_q = BMP280_KF_Q
        self.bmp_r = BMP280_KF_R
        self.bmp_residual_th = BMP280_RESIDUAL_TH
        self.bmp_q_inc = BMP280_Q_INCREASE
        self.bmp_q_dec = BMP280_Q_DECREASE
        self.bmp_q_max = BMP280_Q_MAX
        # 校准
        self.ref_pressure = None      # None -> 自动由数据估计
        self.bmp_bias = None          # None -> 自动由数据估计
        self.calib_samples = 50       # 用于估计参考气压/偏置的前 N 个样本
        # 融合
        self.fusion_scheme = 14
        self.w_ms = FUSION_WEIGHT_MS5611_DEFAULT
        self.w_bmp = FUSION_WEIGHT_BMP280_DEFAULT
        self.hpf_alpha = HPF_ALPHA_DEFAULT
        self.motion_threshold = MOTION_THRESHOLD_PA_DEFAULT
        self.weight_static_ms = WEIGHT_STATIC_MS_DEFAULT
        self.weight_motion_ms = WEIGHT_MOTION_MS_DEFAULT
        self.weight_smooth_alpha = WEIGHT_SMOOTH_ALPHA_DEFAULT
        self.w_delta_static_ms = W_DELTA_MS_STATIC_DEFAULT
        self.w_delta_motion_ms = W_DELTA_MS_MOTION_DEFAULT
        self.delta_weight_smooth_alpha = DELTA_WEIGHT_SMOOTH_ALPHA_DEFAULT
        self.ivar_epsilon = IVAR_EPSILON_DEFAULT
        self.delta_conf_eps = DELTA_CONF_EPS_DEFAULT
        self.anchor_alpha = ANCHOR_ALPHA_DEFAULT
        self.comp_alpha = COMP_ALPHA_DEFAULT
        self.comp_beta = COMP_BETA_DEFAULT
        self.tc_coeff = TC_COEFF_DEFAULT
        # 方案 15 场景门控增量锁定（Schmitt 门控 + 逆方差置信加权 + 门控积分）
        self.gate_open = 0.5
        self.gate_close = 0.3
        self.lock_integ = 1.0
        self.hold_anchor = 0.0
        self.delta_lp_alpha = 0.1    # 方案15：Δh 低通系数（平滑单步噪声尖峰，保留持续升降）
        self.motion_lp_alpha = 0.4    # 方案15：升降段 Δh 低通系数（高于锁定时用的 delta_lp_alpha，提升动态响应）
        # 方案 16 KF 主导场景门控增量锁定（与 S15 同构，但降噪与场景判定全部来自 KF，不依赖 NN）
        self.gate_open_kf = 0.03     # KF 衍生场景：进入升降(开)阈值（|Δh| 平滑幅度，m/样本）
        self.gate_close_kf = 0.015   # 退出升降(关)阈值（迟滞，防抖动）
        self.scene16_lp_alpha = 0.15 # KF 衍生场景 |Δh| 的 EMA 系数
        self.scene16_delta_alpha = 0.2  # 场景信号用的固定低通 α（与门控解耦，避免运动段高 α 把噪声带入门控判定）
        # 以下与 S15 共用：lock_integ / hold_anchor / delta_lp_alpha / motion_lp_alpha / delta_conf_eps
        # EMA 平滑
        self.pressure_ema_alpha = PRESSURE_EMA_ALPHA_DEFAULT
        self.height_ema_alpha = HEIGHT_EMA_ALPHA_DEFAULT
        # NN 是否启用（联合多任务模型 baseline_eqw：一次推理同时给出
        # 滤波后的 MS5611/BMP280 气压与场景概率，复刻固件）
        self.use_nn = True
        self.nn_model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'models', 'compare_v2', '2_baseline_eqw.tflite')
        self.nn_ms_model = None   # 向后兼容保留
        self.nn_bmp_model = None
        # 场景识别（滤波的同时额外输出当前运动场景，辅助判断）
        self.use_scene_clf = False
        self.scene_model = None


# ============================================================
# 6. 主模拟函数
# ============================================================
def simulate(ms_p, bmp_p, temp, params, truth=None,
              base_ms=None, base_bmp=None, scene_prob=None):
    """
    对一组输入序列运行完整固件算法。

    参数：
        ms_p : list/array  MS5611 原始气压 (Pa)
        bmp_p: list/array  BMP280 原始气压 (Pa)
        temp : list/array  BMP280 温度 (℃)（高度统一用 BMP280 温度重算）
        params: AlgoParams
        truth : list/array  真实高度 (m)，可选，用于指标计算

    返回：
        dict，包含各阶段气压/高度序列与统计字段，供绘图与指标使用。
    """
    n = len(ms_p)
    ms_p = np.asarray(ms_p, dtype=float)
    bmp_p = np.asarray(bmp_p, dtype=float)
    temp = np.asarray(temp, dtype=float)

    # ---- 校准 (估计参考气压与 bmp_bias) ----
    # 注意：GUI 从 Spinbox 读来的 calib_samples / fusion_scheme 是 float，
    # 切片与方案判定需转为 int，否则会触发 "slice indices must be integers" 错误。
    nc = int(min(int(params.calib_samples), n))
    if params.ref_pressure is None:
        ref_pressure = float(np.mean(bmp_p[:nc]))
        # 保护：相对/差分气压数据其均值可能接近 0，会导致气压->高度公式
        # 直接返回 0（整段变直线）。此时回退到标准海平面气压做参考。
        if ref_pressure <= 1000.0:
            ref_pressure = SEA_LEVEL_PRESSURE_PA
    else:
        ref_pressure = float(params.ref_pressure)
        if ref_pressure <= 1000.0:
            ref_pressure = SEA_LEVEL_PRESSURE_PA
    if params.bmp_bias is None:
        # 注意：MS5611 原始气压在固件里未加 bmp_bias，这里用原始均值差估计
        bias = float(np.mean(ms_p[:nc]) - np.mean(bmp_p[:nc]))
    else:
        bias = float(params.bmp_bias)

    # ---- 滤波器与融合状态初始化 ----
    ms_kf = AdaptiveKalmanFilter(np.mean(ms_p[:nc]) if nc > 0 else ms_p[0],
                                 KF_INIT_P, params.ms_q, params.ms_r, sensor='ms5611',
                                 residual_th=params.ms_residual_th, q_inc=params.ms_q_inc,
                                 q_dec=params.ms_q_dec, q_max=params.ms_q_max)
    bmp_kf = AdaptiveKalmanFilter(np.mean(bmp_p[:nc]) if nc > 0 else bmp_p[0],
                                  KF_INIT_P, params.bmp_q, params.bmp_r, sensor='bmp280',
                                  residual_th=params.bmp_residual_th, q_inc=params.bmp_q_inc,
                                  q_dec=params.bmp_q_dec, q_max=params.bmp_q_max)
    # ---- 联合多任务模型：一次推理得到 base_ms / base_bmp（绝对气压）与场景概率 ----
    # 若调用方已预计算（自动调参时复用，避免每个候选都重跑模型），则直接采用快速路径。
    base_ms_seq = base_bmp_seq = scene_seq = None
    if params.use_nn:
        if base_ms is not None and base_bmp is not None:
            base_ms_seq = np.asarray(base_ms, dtype=float)
            base_bmp_seq = np.asarray(base_bmp, dtype=float)
            scene_seq = np.asarray(scene_prob) if scene_prob is not None else None
        else:
            try:
                base_ms_seq, base_bmp_seq, scene_seq = run_joint_model(
                    ms_p, bmp_p, params.nn_model, window=MT_WINDOW_DEFAULT)
            except Exception as e:
                print(f"[联合模型] 加载/推理失败，已回退 KF: {e}")
                base_ms_seq = base_bmp_seq = scene_seq = None

    # 场景识别器：与滤波并行运行，额外输出当前运动场景（辅助判断）
    # 参考气压固定为 101325 Pa（标准海平面），与训练、与单片机保持一致，
    # 保证推理特征分布与训练一致（识别主要靠斜率/方差类特征，与绝对参考无关）。
    scene_clf = None
    if params.use_scene_clf:
        try:
            scene_clf = SceneClassifier(model_path=params.scene_model,
                                        ref_pressure=SEA_LEVEL_PRESSURE_PA)
        except Exception as e:
            print(f"[场景识别] 模型加载失败，已跳过: {e}")
            scene_clf = None
    ms_kf_nn = KalmanFilter(np.mean(ms_p[:nc]) if nc > 0 else ms_p[0], KF_INIT_P, NN_KF_Q, NN_KF_R)
    bmp_kf_nn = KalmanFilter(np.mean(bmp_p[:nc]) if nc > 0 else bmp_p[0], KF_INIT_P, NN_KF_Q, NN_KF_R)

    scheme = int(params.fusion_scheme)

    # 融合方案 4/14 的 HPF 状态
    hpf_last = 0.0
    # 方案 5 自适权重状态
    smooth_weight_ms = params.w_ms
    prev_ms5611_p = 0.0
    motion_window = [0.0] * MOTION_WINDOW_SIZE
    motion_window_idx = 0
    # 方案 7 状态
    first_run_7 = True
    ms5611_height_prev = 0.0
    bmp280_height_prev = 0.0
    smooth_delta_weight_ms = params.w_delta_static_ms
    prev_ms5611_p_7 = 0.0
    motion_window_7 = [0.0] * MOTION_WINDOW_SIZE
    motion_window_idx_7 = 0
    # 方案 10/11 状态
    ivar_ms5611_buf = [0.0] * IVAR_WINDOW_SIZE
    ivar_bmp280_buf = [0.0] * IVAR_WINDOW_SIZE
    ivar_idx = 0
    ivar_filled = 0
    delta11_first_run = True
    delta11_prev_ms5611_p = 0.0
    delta11_prev_bmp280_p = 0.0
    delta11_fused_pa = 0.0
    delta11_ms_window = [0.0] * DELTA_CONF_WINDOW
    delta11_bmp_window = [0.0] * DELTA_CONF_WINDOW
    delta11_idx = 0
    delta11_filled = 0
    # 方案 13 状态
    so13_first_run = True
    so13_P_fused = 0.0
    so13_D_fused = 0.0
    so13_prev_ms5611_p = 0.0
    # 方案 14 温漂补偿参考温度
    tc14_ref_initialized = False
    tc14_ref_temperature = 0.0
    # 方案 15 状态（场景门控增量锁定）
    gate15_state = False
    first_run_15 = True
    h15_prev_ms = 0.0
    h15_prev_bmp = 0.0
    h15_lock = 0.0
    delta15_ms_window = [0.0] * DELTA_CONF_WINDOW
    delta15_bmp_window = [0.0] * DELTA_CONF_WINDOW
    delta15_idx = 0
    delta15_filled = 0
    delta15_lp = 0.0
    # 方案 16 状态（KF 主导场景门控增量锁定）
    gate16_state = False
    first_run_16 = True
    h16_prev_ms = 0.0
    h16_prev_bmp = 0.0
    h16_lock = 0.0
    delta16_ms_window = [0.0] * DELTA_CONF_WINDOW
    delta16_bmp_window = [0.0] * DELTA_CONF_WINDOW
    delta16_idx = 0
    delta16_filled = 0
    delta16_lp = 0.0
    scene16_lp = 0.0
    scene_delta_lp = 0.0
    # EMA 状态
    pressure_smoothed = None
    height_smoothed = None

    # 输出缓冲
    out = {
        'ref_pressure': ref_pressure,
        'bmp_bias': bias,
        'raw_height_bmp': np.zeros(n),
        'raw_height_ms': np.zeros(n),
        'kf_height_bmp': np.zeros(n),
        'kf_height_ms': np.zeros(n),
        'nn_height_bmp': np.zeros(n),
        'nn_height_ms': np.zeros(n),
        'fused_pressure': np.zeros(n),
        'fused_height': np.zeros(n),
        'q_ms': np.zeros(n),
        'q_bmp': np.zeros(n),
        'k_ms': np.zeros(n),
        'k_bmp': np.zeros(n),
        # 场景识别输出（仅在 use_scene_clf 时填充；否则保持空/零）
        'scene_prob': np.zeros((n, len(CLASS_NAMES)), dtype=float),
        'scene_pred': np.zeros(n, dtype=int),
        'scene_label': None,
        'scene_classes': list(CLASS_NAMES),
        's15_lock': np.zeros(n),
        's15_integ': np.zeros(n),
        's16_lock': np.zeros(n),
        's16_gate': np.zeros(n),
        's16_scene_lp': np.zeros(n),
    }

    for i in range(n):
        # ---- MS5611 处理 ----
        ms_raw = ms_p[i]
        ms_kf_out = ms_kf.update(ms_raw)
        if params.use_nn and base_ms_seq is not None:
            base_ms_i = base_ms_seq[i]
            if scheme == 1:
                ms_nn_out = base_ms_i * 0.5 + ms_kf_out * 0.5
            elif scheme == 3:
                ms_nn_out = ms_kf_nn.update(base_ms_i)
            else:
                ms_nn_out = base_ms_i   # 联合模型 OUT_1（绝对气压）
        else:
            ms_nn_out = ms_kf_out

        # ---- BMP280 处理 ----
        bmp_raw = bmp_p[i]
        t = temp[i]
        bmp_kf_out = bmp_kf.update(bmp_raw)
        if params.use_nn and base_bmp_seq is not None:
            base_bmp_i = base_bmp_seq[i]
            if scheme == 1:
                bmp_nn_out = base_bmp_i * 0.5 + bmp_kf_out * 0.5
            elif scheme == 3:
                bmp_nn_out = bmp_kf_nn.update(base_bmp_i)
            elif scheme == 9:
                bmp_nn_out = base_bmp_i * 0.7 + bmp_kf_out * 0.3
            else:
                bmp_nn_out = base_bmp_i   # 联合模型 OUT_2（绝对气压）
        else:
            bmp_nn_out = bmp_kf_out

        # ---- 全局 BMP280 偏置补偿 (bmp_bias) ----
        bmp_raw_b = bmp_raw + bias
        bmp_kf_out_b = bmp_kf_out + bias
        bmp_nn_out_b = bmp_nn_out + bias

        # 高度：统一用 BMP280 温度 t 与参考气压
        ht_bmp_raw = pressure_to_altitude_with_temp(bmp_raw_b, t, ref_pressure)
        ht_ms_raw = pressure_to_altitude_with_temp(ms_raw, t, ref_pressure)
        ht_bmp_kf = pressure_to_altitude_with_temp(bmp_kf_out_b, t, ref_pressure)
        ht_ms_kf = pressure_to_altitude_with_temp(ms_kf_out, t, ref_pressure)
        ht_bmp_nn = pressure_to_altitude_with_temp(bmp_nn_out_b, t, ref_pressure)
        ht_ms_nn = pressure_to_altitude_with_temp(ms_nn_out, t, ref_pressure)

        # ---- 方案14 温漂补偿 ----
        tc_correction = 0.0
        if scheme == 14 and params.tc_coeff != 0.0:
            if not tc14_ref_initialized:
                tc14_ref_temperature = t
                tc14_ref_initialized = True
            temp_drift = t - tc14_ref_temperature
            tc_correction = params.tc_coeff * temp_drift
            bmp_kf_out_b -= tc_correction
            bmp_nn_out_b -= tc_correction
            ht_bmp_kf = pressure_to_altitude_with_temp(bmp_kf_out_b, t, ref_pressure)
            ht_bmp_nn = pressure_to_altitude_with_temp(bmp_nn_out_b, t, ref_pressure)

        # ---- 融合 ----
        if scheme in (2,):
            fused_pa = ms_kf_out * 0.50 + bmp_nn_out_b * 0.25 + bmp_kf_out_b * 0.25
        elif scheme in (4, 14):
            ms5611_diff = ms_kf_out - bmp_nn_out_b
            lpf = params.hpf_alpha * ms5611_diff + (1 - params.hpf_alpha) * hpf_last
            hpf_last = lpf
            hpf = ms5611_diff - lpf
            fused_pa = bmp_nn_out_b + hpf
        elif scheme == 5:
            residual = abs(ms_kf_out - prev_ms5611_p)
            prev_ms5611_p = ms_kf_out
            motion_window[motion_window_idx] = residual
            motion_window_idx = (motion_window_idx + 1) % MOTION_WINDOW_SIZE
            max_res = max(motion_window)
            is_motion = max_res > params.motion_threshold
            target = params.weight_motion_ms if is_motion else params.weight_static_ms
            smooth_weight_ms += params.weight_smooth_alpha * (target - smooth_weight_ms)
            fused_pa = ms_kf_out * smooth_weight_ms + bmp_nn_out_b * (1 - smooth_weight_ms)
        elif scheme == 6:
            fused_pa = ms_kf_out * 0.85 + bmp_kf_out_b * 0.15
        elif scheme == 7:
            ms_h = ht_ms_kf
            bmp_h = ht_bmp_nn
            if first_run_7:
                ms5611_height_prev = ms_h
                bmp280_height_prev = bmp_h
                fused_h_prev = bmp_h
                first_run_7 = False
            else:
                delta_ms = ms_h - ms5611_height_prev
                delta_bmp = bmp_h - bmp280_height_prev
                ms5611_height_prev = ms_h
                bmp280_height_prev = bmp_h
                res7 = abs(ms_kf_out - prev_ms5611_p_7)
                prev_ms5611_p_7 = ms_kf_out
                motion_window_7[motion_window_idx_7] = res7
                motion_window_idx_7 = (motion_window_idx_7 + 1) % MOTION_WINDOW_SIZE
                is_motion_7 = max(motion_window_7) > params.motion_threshold
                target = params.w_delta_motion_ms if is_motion_7 else params.w_delta_static_ms
                smooth_delta_weight_ms += params.delta_weight_smooth_alpha * (target - smooth_delta_weight_ms)
                delta_fused = delta_ms * smooth_delta_weight_ms + delta_bmp * (1 - smooth_delta_weight_ms)
                fused_h_prev = fused_h_prev + delta_fused
                if not is_motion_7:
                    drift = bmp_h - fused_h_prev
                    fused_h_prev += PULL_COEFF * drift
            fused_pa = bmp_nn_out_b  # 气压以 BMP280 为准（高度已直接积分）
        elif scheme == 8:
            fused_pa = ms_kf_out
        elif scheme == 9:
            fused_pa = bmp_nn_out_b
        elif scheme == 10:
            ivar_ms5611_buf[ivar_idx] = ms_kf_out
            ivar_bmp280_buf[ivar_idx] = bmp_nn_out_b
            ivar_idx = (ivar_idx + 1) % IVAR_WINDOW_SIZE
            if ivar_filled < IVAR_WINDOW_SIZE:
                ivar_filled += 1
            nn_ = ivar_filled
            mean_ms = sum(ivar_ms5611_buf) / nn_
            mean_bmp = sum(ivar_bmp280_buf) / nn_
            var_ms = sum((x - mean_ms) ** 2 for x in ivar_ms5611_buf) / (nn_ - 1 if nn_ > 1 else 1)
            var_bmp = sum((x - mean_bmp) ** 2 for x in ivar_bmp280_buf) / (nn_ - 1 if nn_ > 1 else 1)
            w_bmp = var_ms + params.ivar_epsilon
            w_ms = var_bmp + params.ivar_epsilon
            fused_pa = (bmp_nn_out_b * w_bmp + ms_kf_out * w_ms) / (w_bmp + w_ms)
        elif scheme == 11:
            if delta11_first_run:
                delta11_prev_ms5611_p = ms_kf_out
                delta11_prev_bmp280_p = bmp_nn_out_b
                delta11_fused_pa = bmp_nn_out_b
                delta11_first_run = False
            else:
                d_ms = ms_kf_out - delta11_prev_ms5611_p
                d_bmp = bmp_nn_out_b - delta11_prev_bmp280_p
                delta11_prev_ms5611_p = ms_kf_out
                delta11_prev_bmp280_p = bmp_nn_out_b
                delta11_ms_window[delta11_idx] = d_ms
                delta11_bmp_window[delta11_idx] = d_bmp
                delta11_idx = (delta11_idx + 1) % DELTA_CONF_WINDOW
                if delta11_filled < DELTA_CONF_WINDOW:
                    delta11_filled += 1
                nn_ = delta11_filled
                mean_ms = sum(delta11_ms_window) / nn_
                mean_bmp = sum(delta11_bmp_window) / nn_
                var_ms = sum((x - mean_ms) ** 2 for x in delta11_ms_window) / (nn_ - 1 if nn_ > 1 else 1)
                var_bmp = sum((x - mean_bmp) ** 2 for x in delta11_bmp_window) / (nn_ - 1 if nn_ > 1 else 1)
                conf_ms = 1.0 / (var_ms + params.delta_conf_eps)
                conf_bmp = 1.0 / (var_bmp + params.delta_conf_eps)
                delta_fused = (d_ms * conf_ms + d_bmp * conf_bmp) / (conf_ms + conf_bmp)
                delta11_fused_pa += delta_fused
                delta11_fused_pa += params.anchor_alpha * (bmp_nn_out_b - delta11_fused_pa)
            fused_pa = delta11_fused_pa
        elif scheme == 13:
            if so13_first_run:
                so13_P_fused = bmp_nn_out_b
                so13_D_fused = 0.0
                so13_prev_ms5611_p = ms_kf_out
                so13_first_run = False
            else:
                D_ms = ms_kf_out - so13_prev_ms5611_p
                so13_prev_ms5611_p = ms_kf_out
                so13_D_fused = params.comp_beta * D_ms + (1 - params.comp_beta) * so13_D_fused
                so13_P_fused = so13_P_fused + so13_D_fused
                so13_P_fused += params.comp_alpha * (bmp_nn_out_b - so13_P_fused)
            fused_pa = so13_P_fused
        elif scheme == 15:
            # ---- 场景门控增量锁定 ----
            # 场景概率（联合模型 OUT_3[1] = p_elevation）；无场景则恒锁（gate=0）
            if scene_seq is not None:
                p_elev = float(scene_seq[i][1])
            else:
                p_elev = 0.0
            # 1) Schmitt 触发器（迟滞，防止门控抖动）
            if not gate15_state:
                if p_elev > params.gate_open:
                    gate15_state = True
            else:
                if p_elev < params.gate_close:
                    gate15_state = False
            gate = 1.0 if gate15_state else 0.0
            did_integrate_15 = False
            # 2) 两传感器高度增量（ht_*_nn 已含温度补偿）
            if first_run_15:
                h15_prev_ms = ht_ms_nn
                h15_prev_bmp = ht_bmp_nn
                h15_lock = ht_bmp_nn      # 初始锚定 BMP280（NN 降噪后）
                first_run_15 = False
            else:
                d_ms = ht_ms_nn - h15_prev_ms
                d_bmp = ht_bmp_nn - h15_prev_bmp
                h15_prev_ms = ht_ms_nn
                h15_prev_bmp = ht_bmp_nn
                # 3) 逆方差置信加权（短窗估计 Δ 噪声，复用方案11思路）
                delta15_ms_window[delta15_idx] = d_ms
                delta15_bmp_window[delta15_idx] = d_bmp
                delta15_idx = (delta15_idx + 1) % DELTA_CONF_WINDOW
                if delta15_filled < DELTA_CONF_WINDOW:
                    delta15_filled += 1
                nn_ = delta15_filled
                mean_ms = sum(delta15_ms_window) / nn_
                mean_bmp = sum(delta15_bmp_window) / nn_
                var_ms = sum((x - mean_ms) ** 2 for x in delta15_ms_window) / (nn_ - 1 if nn_ > 1 else 1)
                var_bmp = sum((x - mean_bmp) ** 2 for x in delta15_bmp_window) / (nn_ - 1 if nn_ > 1 else 1)
                conf_ms = 1.0 / (var_ms + params.delta_conf_eps)
                conf_bmp = 1.0 / (var_bmp + params.delta_conf_eps)
                delta_fused = (d_ms * conf_ms + d_bmp * conf_bmp) / (conf_ms + conf_bmp)
                # Δh 低通：升降段用更高 alpha 快速响应真实变化，静止段用低 alpha 锁死噪声。
                # 这样静止(gate=0)仍纯锁定（不积分），升降(gate=1)才快速跟踪，动静解耦且
                # 不引入绝对高度慢漂（比"绝对高度跟随"更干净，避免误开门控把漂移带进静止段）。
                _lp = params.motion_lp_alpha if gate > 0.5 else params.delta_lp_alpha
                delta15_lp = (_lp * delta_fused + (1 - _lp) * delta15_lp)
                # 门控积分：静止(gate=0) 高度冻结；升降(gate=1) 按低通 Δh 积分。
                did_integrate_15 = False
                if gate > 0.5:
                    h15_lock += params.lock_integ * delta15_lp
                    did_integrate_15 = True
                elif gate < 0.5:
                    # 静止时仅极弱锚定（hold_anchor=0 即纯锁定）
                    h15_lock += params.hold_anchor * (ht_bmp_nn - h15_lock)
            out['s15_lock'][i] = h15_lock
            out['s15_integ'][i] = 1.0 if did_integrate_15 else 0.0
            fused_pa = bmp_nn_out_b   # 气压以 BMP280 为准（高度已直接积分）
        elif scheme == 16:
            # ---- KF 主导场景门控增量锁定 ----
            # 与 S15 同构，但：① 融合入口用 KF 平滑高度 ht_ms_kf / ht_bmp_kf
            # （不引用任何 NN 输出，降噪完全来自自适应卡尔曼滤波）；
            # ② 场景判定由 KF 自身增量推导（BMP280 KF Δh 经逆方差加权后的
            # |Δh| 幅度），Schmitt 迟滞门控，零 NN 依赖。
            if first_run_16:
                h16_prev_ms = ht_ms_kf
                h16_prev_bmp = ht_bmp_kf
                h16_lock = ht_bmp_kf      # 初始锚定 BMP280（KF 降噪后）
                scene16_lp = 0.0
                first_run_16 = False
            else:
                d_ms = ht_ms_kf - h16_prev_ms
                d_bmp = ht_bmp_kf - h16_prev_bmp
                h16_prev_ms = ht_ms_kf
                h16_prev_bmp = ht_bmp_kf
                # 逆方差置信加权（短窗估计 Δ 噪声，复用方案11/15 思路）
                delta16_ms_window[delta16_idx] = d_ms
                delta16_bmp_window[delta16_idx] = d_bmp
                delta16_idx = (delta16_idx + 1) % DELTA_CONF_WINDOW
                if delta16_filled < DELTA_CONF_WINDOW:
                    delta16_filled += 1
                nn_ = delta16_filled
                mean_ms = sum(delta16_ms_window) / nn_
                mean_bmp = sum(delta16_bmp_window) / nn_
                var_ms = sum((x - mean_ms) ** 2 for x in delta16_ms_window) / (nn_ - 1 if nn_ > 1 else 1)
                var_bmp = sum((x - mean_bmp) ** 2 for x in delta16_bmp_window) / (nn_ - 1 if nn_ > 1 else 1)
                conf_ms = 1.0 / (var_ms + params.delta_conf_eps)
                conf_bmp = 1.0 / (var_bmp + params.delta_conf_eps)
                delta_fused = (d_ms * conf_ms + d_bmp * conf_bmp) / (conf_ms + conf_bmp)
                # KF 衍生场景：用「与门控解耦」的固定低通 Δh 幅度（scene16_delta_alpha，
                # 不随 motion_lp_alpha 变化），避免运动段高 α 把单步噪声带进门控判定，
                # 再用 |·| EMA + Schmitt 迟滞判定 static/elevation。全程无 NN。
                scene_delta_lp += params.scene16_delta_alpha * (delta_fused - scene_delta_lp)
                scene16_lp += params.scene16_lp_alpha * (abs(scene_delta_lp) - scene16_lp)
                if not gate16_state:
                    if scene16_lp > params.gate_open_kf:
                        gate16_state = True
                else:
                    if scene16_lp < params.gate_close_kf:
                        gate16_state = False
                gate = 1.0 if gate16_state else 0.0
                # Δh 低通：升降段用更高 alpha 快速响应，静止段用低 alpha 锁死噪声
                _lp = params.motion_lp_alpha if gate > 0.5 else params.delta_lp_alpha
                delta16_lp = (_lp * delta_fused + (1 - _lp) * delta16_lp)
                # 门控积分：静止(gate=0) 高度冻结；升降(gate=1) 按低通 Δh 积分
                if gate > 0.5:
                    h16_lock += params.lock_integ * delta16_lp
                else:
                    h16_lock += params.hold_anchor * (ht_bmp_kf - h16_lock)
            out['s16_lock'][i] = h16_lock
            out['s16_gate'][i] = 1.0 if gate16_state else 0.0
            out['s16_scene_lp'][i] = scene16_lp
            # KF 衍生场景填充（供 scene_accuracy 评估，区别于 S15 的 NN 场景）
            out['scene_pred'][i] = 1 if gate16_state else 0
            out['scene_prob'][i] = [0.0, 1.0] if gate16_state else [1.0, 0.0]
            fused_pa = bmp_kf_out_b   # 气压以 BMP280 KF 为准（高度已直接积分）
        else:  # scheme 1 / 3 及其它：简单加权
            fused_pa = ms_kf_out * params.w_ms + bmp_nn_out_b * params.w_bmp

        # ---- 由融合气压算高度（方案 7/15 已在上面直接积分）----
        if scheme not in (7, 15, 16):
            fused_h = pressure_to_altitude_with_temp(fused_pa, t, ref_pressure)
        elif scheme == 7:
            fused_h = fused_h_prev
        elif scheme == 15:
            fused_h = h15_lock
        else:  # scheme == 16
            fused_h = h16_lock

        # ---- EMA 平滑 ----
        if pressure_smoothed is None:
            pressure_smoothed = fused_pa
            height_smoothed = fused_h
        else:
            pressure_smoothed += params.pressure_ema_alpha * (fused_pa - pressure_smoothed)
            height_smoothed += params.height_ema_alpha * (fused_h - height_smoothed)
        fused_pa_s = pressure_smoothed
        fused_h_s = height_smoothed

        # ---- 记录 ----
        out['raw_height_bmp'][i] = ht_bmp_raw
        out['raw_height_ms'][i] = ht_ms_raw
        out['kf_height_bmp'][i] = ht_bmp_kf
        out['kf_height_ms'][i] = ht_ms_kf
        out['nn_height_bmp'][i] = ht_bmp_nn
        out['nn_height_ms'][i] = ht_ms_nn
        out['fused_pressure'][i] = fused_pa_s
        out['fused_height'][i] = fused_h_s
        out['q_ms'][i] = ms_kf.q
        out['q_bmp'][i] = bmp_kf.q
        out['k_ms'][i] = ms_kf.k
        out['k_bmp'][i] = bmp_kf.k

        # 场景识别：优先采用联合模型 OUT_3（与滤波同一次推理）；若启用旧 scene_clf 也并行记录
        if scene_seq is not None:
            out['scene_prob'][i] = scene_seq[i]
            out['scene_pred'][i] = int(np.argmax(scene_seq[i]))
        elif scene_clf is not None:
            p = scene_clf.update(ms_p[i], bmp_p[i], temp[i])
            out['scene_prob'][i] = p
            out['scene_pred'][i] = int(np.argmax(p))

    # 整段主导场景：取平均概率最高的类别
    if scene_seq is not None:
        mean_prob = out['scene_prob'].mean(axis=0)
        out['scene_label'] = CLASS_NAMES[int(np.argmax(mean_prob))]
    elif scene_clf is not None:
        mean_prob = out['scene_prob'].mean(axis=0)
        out['scene_label'] = scene_clf.classes[int(np.argmax(mean_prob))]

    out['truth'] = np.asarray(truth, dtype=float) if truth is not None else None
    return out


# ============================================================
# 7. 由 AlgoParams 生成固件头片段 (便于把调好的参数写回 C)
# ============================================================
def export_header_snippet(params):
    lines = [
        "/* 由 altimeter_tuner 导出 —— 请人工核对后替换 kalman_filter.h / main.c 宏 */",
        f"#define MS5611_KF_Q      {params.ms_q:.4f}f",
        f"#define MS5611_KF_R      {params.ms_r:.4f}f",
        f"#define BMP280_KF_Q      {params.bmp_q:.4f}f",
        f"#define BMP280_KF_R      {params.bmp_r:.4f}f",
        f"#define FUSION_SCHEME    {params.fusion_scheme}",
        f"#define HPF_ALPHA        {params.hpf_alpha:.3f}f",
        f"#define MOTION_THRESHOLD_PA  {params.motion_threshold:.3f}f",
        f"#define PRESSURE_EMA_ALPHA   {params.pressure_ema_alpha:.3f}f",
        f"#define HEIGHT_EMA_ALPHA     {params.height_ema_alpha:.3f}f",
    ]
    if params.ref_pressure is not None:
        lines.append(f"/* 参考气压(手动): {params.ref_pressure:.2f} Pa */")
    if params.bmp_bias is not None:
        lines.append(f"/* BMP280 偏置(手动): {params.bmp_bias:.3f} Pa */")
    return "\n".join(lines)


if __name__ == '__main__':
    # 简单自测
    t = np.arange(300) / 10.0
    true = np.concatenate([np.full(100, 100), np.linspace(100, 103, 100), np.full(100, 103)])
    true_p = 101325.0 - true * 12.0
    ms = true_p + np.random.normal(0, 3.05, 300)
    bmp = true_p + np.random.normal(0, 0.35, 300)
    p = AlgoParams()
    res = simulate(ms, bmp, np.full(300, 25.0), p, truth=true)
    print("ref_pressure=", res['ref_pressure'])
    print("bmp_bias=", res['bmp_bias'])
    print("fused end=", res['fused_height'][-1])
