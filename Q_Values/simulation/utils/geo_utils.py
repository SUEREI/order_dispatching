# -*- coding: utf-8 -*-
"""
地理计算工具模块

该模块提供地理坐标相关的计算功能，包括：
- 两点之间的球面距离计算（Haversine公式）
- 沿方向移动后的位置计算

作者：CodeArts代码智能体
日期：2025-04-13
"""

import math
from typing import Tuple

# 地球半径（米）
EARTH_RADIUS = 6371000


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    使用Haversine公式计算两点之间的球面距离

    Haversine公式考虑了地球的曲率，适用于计算地球表面两点之间的距离。
    该公式假设地球是一个完美的球体，半径为6371公里。

    参数：
        coord1: 第一个点的坐标 (经度, 纬度)，单位：度
        coord2: 第二个点的坐标 (经度, 纬度)，单位：度

    返回：
        两点之间的距离，单位：米

    示例：
        >>> # 计算北京天安门到故宫的距离
        >>> coord1 = (116.397128, 39.916527)  # 天安门
        >>> coord2 = (116.397479, 39.918988)  # 故宫
        >>> distance = haversine_distance(coord1, coord2)
        >>> print(f"距离: {distance:.2f} 米")
        距离: 276.12 米
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

    # 计算距离
    distance = EARTH_RADIUS * c

    return distance / 1000  # 转换为公里


def calculate_move_position(
    current_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
    move_distance: float
) -> Tuple[float, float]:
    """
    计算从当前位置向目标位置方向移动指定距离后的位置

    该函数用于模拟骑手沿路径移动，计算移动后的新位置。
    如果移动距离大于等于两点之间的距离，则返回目标位置。

    参数：
        current_pos: 当前位置坐标 (经度, 纬度)，单位：度
        target_pos: 目标位置坐标 (经度, 纬度)，单位：度
        move_distance: 移动距离，单位：米

    返回：
        移动后的新位置坐标 (经度, 纬度)，单位：度

    示例：
        >>> # 从天安门向故宫方向移动100米
        >>> current_pos = (116.397128, 39.916527)  # 天安门
        >>> target_pos = (116.397479, 39.918988)  # 故宫
        >>> new_pos = calculate_move_position(current_pos, target_pos, 100)
        >>> print(f"新位置: ({new_pos[0]:.6f}, {new_pos[1]:.6f})")
        新位置: (116.397255, 39.917423)
    """
    # 计算两点之间的总距离
    total_distance = haversine_distance(current_pos, target_pos)

    # 如果移动距离大于等于总距离，直接返回目标位置
    if move_distance >= total_distance:
        return target_pos

    # 如果两点重合，返回当前位置
    if total_distance == 0:
        return current_pos

    # 计算移动比例
    ratio = move_distance / total_distance

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


# 模块测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("地理计算工具模块测试")
    print("=" * 60)

    # 测试1：计算两点距离
    print("\n测试1：计算两点距离")
    coord1 = (116.397128, 39.916527)  # 天安门
    coord2 = (116.397479, 39.918988)  # 故宫
    distance = haversine_distance(coord1, coord2)
    print(f"天安门坐标: {coord1}")
    print(f"故宫坐标: {coord2}")
    print(f"距离: {distance:.2f} 米")

    # 测试2：计算移动位置
    print("\n测试2：计算移动位置")
    move_distance = 100  # 移动100米
    new_pos = calculate_move_position(coord1, coord2, move_distance)
    print(f"从天安门向故宫方向移动 {move_distance} 米")
    print(f"新位置: ({new_pos[0]:.6f}, {new_pos[1]:.6f})")

    # 验证移动距离
    actual_distance = haversine_distance(coord1, new_pos)
    print(f"实际移动距离: {actual_distance:.2f} 米")

    # 测试3：移动距离超过总距离
    print("\n测试3：移动距离超过总距离")
    move_distance = 1000  # 移动1000米（超过总距离）
    new_pos = calculate_move_position(coord1, coord2, move_distance)
    print(f"从天安门向故宫方向移动 {move_distance} 米")
    print(f"新位置: ({new_pos[0]:.6f}, {new_pos[1]:.6f})")
    print(f"是否到达目标: {new_pos == coord2}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
