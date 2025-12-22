import config
import numpy as np

"""
空对地信道模型 (Air-to-Ground Channel Model)
================================================
实现基于 ITU-R / 3GPP TR 36.777 的视距(LoS)/非视距(NLoS)概率信道模型。

模型特点：
1. LoS概率模型：基于仰角计算视距链路概率
   P_LoS = 1 / (1 + a * exp(-b * (θ - a)))
   其中 θ 是仰角（度），a、b 是环境相关参数

2. 平均路径损耗：结合LoS和NLoS的加权平均
   PL_avg = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS

位置坐标格式：[x, y, z] (meters)
- UE: [x, y, 0.0] - 地面用户
- UAV: [x, y, UAV_ALTITUDE] - 空中基站
- MBS: [500.0, 500.0, 30.0] - 宏基站
"""


def _calculate_elevation_angle(pos_ground: np.ndarray, pos_aerial: np.ndarray) -> float:
    """
    计算从地面点到空中点的仰角（elevation angle）。
    
    Args:
        pos_ground: 地面位置 [x, y, z_ground]
        pos_aerial: 空中位置 [x, y, z_aerial]
    
    Returns:
        仰角（度），范围 [0, 90]
    """
    horizontal_dist = np.sqrt((pos_aerial[0] - pos_ground[0])**2 + (pos_aerial[1] - pos_ground[1])**2)
    vertical_dist = abs(pos_aerial[2] - pos_ground[2])
    
    if horizontal_dist < config.EPSILON:
        return 90.0  # 正上方
    
    # 仰角 = arctan(垂直距离 / 水平距离)
    elevation_rad = np.arctan(vertical_dist / horizontal_dist)
    return np.degrees(elevation_rad)


def _calculate_los_probability(elevation_angle: float) -> float:
    """
    计算视距(LoS)链路概率。
    基于 ITU-R / 3GPP 模型: P_LoS = 1 / (1 + a * exp(-b * (θ - a)))
    Args:
        elevation_angle: 仰角（度）
    
    Returns:
        LoS概率，范围 [0, 1]
    """
    a, b = config.LOS_PARAMS.get(config.ENVIRONMENT_TYPE, config.LOS_PARAMS['urban'])

    p_los = 1.0 / (1.0 + a * np.exp(-b * (elevation_angle - a)))
    return np.clip(p_los, 0.0, 1.0)

def _calculate_path_loss(distance: float) -> float:
    """
    计算简化路径损耗 (距离平方模型)。
    与原模型保持一致，频率相关项已合并到 G_CONSTS_PRODUCT 中。
    
    Args:
        distance: 传播距离（米）
    
    Returns:
        路径损耗（线性值）
    """
    return distance ** 2


def calculate_channel_gain(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    计算两点之间的信道增益，考虑LoS/NLoS概率。
    
    对于空对地链路(UE-UAV)，使用概率信道模型：
    - 计算仰角和LoS概率
    - 综合LoS和NLoS路径损耗
    
    对于空对空链路(UAV-UAV)和UAV-MBS链路，假设为LoS。
    
    Args:
        pos1: 第一个位置 [x, y, z]
        pos2: 第二个位置 [x, y, z]
    
    Returns:
        信道增益（线性值）
    """
    distance = np.sqrt(np.sum((pos1 - pos2) ** 2))
    
    # 判断链路类型：如果其中一个在地面(z≈0)，则为空对地链路
    is_air_to_ground = (pos1[2] < 1.0) or (pos2[2] < 1.0)
    
    if is_air_to_ground:
        # 确定地面点和空中点
        if pos1[2] < pos2[2]:
            pos_ground, pos_aerial = pos1, pos2
        else:
            pos_ground, pos_aerial = pos2, pos1
        
        # 计算仰角和LoS概率
        elevation_angle = _calculate_elevation_angle(pos_ground, pos_aerial)
        p_los = _calculate_los_probability(elevation_angle)
        
        # 计算路径损耗
        path_loss = _calculate_path_loss(distance)
        pl_los = path_loss  # LoS路径损耗
        
        # NLoS额外损耗（从dB转换为线性）
        nlos_factor = 10 ** (config.NLOS_ADDITIONAL_LOSS_DB / 10)
        pl_nlos = path_loss * nlos_factor  # NLoS路径损耗
        
        # 平均路径损耗 (概率加权)
        avg_path_loss = p_los * pl_los + (1 - p_los) * pl_nlos
    else:
        # 空对空链路或UAV-MBS：假设为LoS
        avg_path_loss = _calculate_path_loss(distance)
    
    # 信道增益 = 天线增益 / 路径损耗
    channel_gain = config.G_CONSTS_PRODUCT / (avg_path_loss + config.EPSILON)
    return channel_gain


def calculate_ue_uav_rate(channel_gain: float, num_associated_ues: int) -> float:
    """Calculates data rate between a UE and a UAV."""
    assert num_associated_ues != 0
    bandwidth_per_ue: float = config.BANDWIDTH_EDGE / num_associated_ues# 每个UE分配的带宽
    # 计算信噪比（发射功率/噪声功率）
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    # 香农公式计算数据速率
    return bandwidth_per_ue * np.log2(1 + snr)


def calculate_uav_mbs_rate(channel_gain: float) -> float:
    """Calculates data rate between a UAV and the MBS."""
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return config.BANDWIDTH_BACKHAUL * np.log2(1 + snr)


def calculate_uav_uav_rate(channel_gain: float) -> float:
    """Calculates data rate between two UAVs."""
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return config.BANDWIDTH_INTER * np.log2(1 + snr)
