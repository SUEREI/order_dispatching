# -*- coding: utf-8 -*-
"""
地理计算工具模块

该模块提供地理坐标相关的计算功能，包括：
- 两点之间的球面距离计算（Haversine公式）
- 沿方向移动后的位置计算
"""

import math
from typing import Tuple, List, Union

import numpy as np

# 地球半径（米）
EARTH_RADIUS = 6371000
# 路网系数 - 直线距离乘以这个系数得到实际骑行距离估计
# 城市路网多为网格状，实际骑行距离约为直线距离的1.3~1.5倍
ROAD_FACTOR = 1.4


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    使用Haversine公式计算两点之间的球面距离
    Haversine公式考虑了地球的曲率，适用于计算地球表面两点之间的距离。
    该公式假设地球是一个完美的球体，半径为6371公里。
    """
    # 将经纬度从度转换为弧度
    lon1, lat1 = math.radians(coord1[0]), math.radians(coord1[1])
    lon2, lat2 = math.radians(coord2[0]), math.radians(coord2[1])
    # 计算经纬度差值
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine公式
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    # 计算距离(考虑路网系数)
    distance = EARTH_RADIUS * c * ROAD_FACTOR

    return distance


def haversine_distance_vectorized(
    coord1: Tuple[float, float],
    coord2_list: Union[List[Tuple[float, float]], np.ndarray],
) -> np.ndarray:
    """
    向量化Haversine距离计算：一个点与一组点之间的距离

    Parameters
    ----------
    coord1 : Tuple[float, float]
        单个位置 (lng, lat)
    coord2_list : List[Tuple[float, float]] | np.ndarray
        一组位置，shape=(N,2)，每行为 (lng, lat)

    Returns
    -------
    np.ndarray
        shape=(N,)，每个元素为 coord1 到对应点的距离（米），含路网系数
    """
    coord2_arr = np.asarray(coord2_list)

    if coord2_arr.ndim == 0 or coord2_arr.shape[0] == 0:
        return np.array([], dtype=float)

    lng1, lat1 = coord1
    lng1_rad, lat1_rad = np.radians(lng1), np.radians(lat1)
    lng2_rad = np.radians(coord2_arr[:, 0])
    lat2_rad = np.radians(coord2_arr[:, 1])

    dlon = lng2_rad - lng1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS * c * ROAD_FACTOR


def calculate_move_position(
    current_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
    distance_to_target: float,
    move_distance: float
) -> Tuple[float, float]:
    """
    计算从当前位置向目标位置方向移动指定距离后的位置
    该函数用于模拟骑手沿路径移动，计算移动后的新位置。
    """
    # # 如果移动距离大于等于总距离，直接返回目标位置
    # if move_distance >= distance_to_target:
    #     return target_pos
    # # 如果两点重合，返回当前位置
    # if distance_to_target == 0:
    #     return current_pos

    # 计算移动比例
    ratio = move_distance / distance_to_target
    # 将经纬度从度转换为弧度
    lon1, lat1 = math.radians(current_pos[0]), math.radians(current_pos[1])
    lon2, lat2 = math.radians(target_pos[0]), math.radians(target_pos[1])
    # 使用球面线性插值计算新位置
    # 这种方法在小距离移动时近似为直线插值
    new_lon = lon1 + ratio * (lon2 - lon1)
    new_lat = lat1 + ratio * (lat2 - lat1)
    # 将弧度转换回度
    new_pos = (math.degrees(new_lon), math.degrees(new_lat))

    return new_pos
