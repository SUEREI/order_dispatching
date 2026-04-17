#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
骑手与订单状态刷新器模块

该模块实现骑手和订单状态的刷新功能，包括：
- 根据路径规划结果计算骑手移动
- 记录取餐和送餐时间
- 处理特殊状态（等待备餐、等待送餐）
- 刷新骑手和订单状态
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, List

from utils import haversine_distance, calculate_move_position

# 配置日志
logger = logging.getLogger(__name__)


class StateRefresher:
    """骑手与订单状态刷新器"""
    # 骑手速度（KM/H）
    COURIER_SPEED = 30.0
    # 时间步长（秒）
    TIME_STEP = 30
    # 送餐等待时间（秒）= 3分钟
    DELIVERY_WAIT_TIME = 180
    def __init__(self):
        """初始化状态刷新器"""
        logger.info(f"状态刷新器初始化完成，骑手速度: {self.COURIER_SPEED} KM/H, "
                    f"时间步长: {self.TIME_STEP} 秒")

    def record_pickup_time(self,
        order: Dict[str, Any],
        arrival_time: datetime,
        estimate_prepare_time: datetime
    ) -> Tuple[datetime, str]:
        """
        记录取餐时间
        如果骑手到达时间早于预计备餐完成时间，需要等待备餐完成。
        否则可以直接取餐。
        """
        if arrival_time < estimate_prepare_time:
            # 需要等待备餐完成
            pickup_time = estimate_prepare_time
            special_status = 'waiting_meal'
            wait_duration = (estimate_prepare_time - arrival_time).total_seconds()
            logger.debug(f"订单 {order['order_id']} 需等待备餐 {wait_duration:.0f} 秒")
        else:
            # 可以直接取餐
            pickup_time = arrival_time
            special_status = 'none'
            logger.debug(f"订单 {order['order_id']} 直接取餐")

        return pickup_time, special_status

    def record_delivery_time(
        self,
        order: Dict[str, Any],
        arrival_time: datetime
    ) -> Tuple[datetime, str]:
        """
        记录送餐时间
        骑手到达送餐点后，需要等待3分钟才能将餐送到客户手中。
        """
        delivery_time = arrival_time + timedelta(seconds=self.DELIVERY_WAIT_TIME)
        special_status = 'waiting_delivery'
        logger.debug(f"订单 {order['order_id']} 送餐时间: {delivery_time}")

        return delivery_time, special_status

    def handle_special_status(
        self,
        courier: Dict[str, Any],
        current_time: datetime
    ) -> Dict[str, Any]:
        """
        处理骑手的特殊状态
        检查骑手是否有特殊状态（等待备餐或等待送餐），
        如果特殊状态未结束，则延续状态；否则清除状态。
        """
        special_status = courier.get('special_status', 'none')
        special_status_end_time = courier.get('special_status_end_time')
        # 无特殊状态
        if special_status == 'none' or special_status is None:

            return {'status': 'normal'}

        # 检查特殊状态是否结束
        if current_time < special_status_end_time:
            # 特殊状态未结束，继续等待
            remaining_time = (special_status_end_time - current_time).total_seconds()
            logger.info(f"骑手 {courier['courier_id']} 处于 {special_status} 状态，"
                        f"剩余 {remaining_time:.0f} 秒")

            return {
                'status': 'waiting',
                'remaining_time': remaining_time
            }

        else:
            # 特殊状态已结束
            # logger.debug(f"骑手 {courier['courier_id']} 的 {special_status} 状态已结束")
            return {
                'status': 'completed',
                'type': special_status
            }

    def refresh_courier(
        self,
        courier: Dict[str, Any],
        current_time: datetime,
    ) -> Dict[str, Any]:
        """
        刷新单个骑手的状态
        根据路径规划结果，计算骑手在时间步长内的移动，
        判断是否到达路径点，记录取餐送餐时间，更新骑手位置和状态。
        """
        time_step = self.TIME_STEP
        # 初始化更新结果
        updates = {
            'courier_updates': None,
            'order_updates': []
        }

        # 处理特殊状态
        special_status_result = self.handle_special_status(courier, current_time)
        if special_status_result['status'] == 'waiting':
            # 处于等待状态，不移动
            # logger.debug(f"骑手 {courier['courier_id']} 处于等待状态，不移动")
            updates['courier_updates'] = courier.copy()

            return updates

        # 如果特殊状态刚结束，清除状态
        if special_status_result['status'] == 'completed':
            courier['special_status'] = 'none'
            courier['special_status_end_time'] = None
            # logger.debug(f"骑手 {courier['courier_id']} 清除特殊状态")

        # # 如果没有路径，不移动
        # if not route:
        #     logger.debug(f"骑手 {courier['courier_id']} 无路径，不移动")
        #     updates['courier_updates'] = courier.copy()
        #     return updates

        # 计算移动距离
        move_distance = self.COURIER_SPEED * 1000 / 3600 * time_step
        # 当前位置和剩余移动距离
        current_pos = courier['position']
        remaining_distance = move_distance

        # 遍历路径点
        for route_point in courier['route'].copy():
            target_pos = route_point['position']
            distance_to_target = haversine_distance(current_pos, target_pos)
            if distance_to_target <= remaining_distance:
                # 可以到达该点，从路径中移除
                courier['route'] = [p for p in courier['route'] if not (p['order_id'] == route_point['order_id'] and p['point_type'] == route_point['point_type'])]
                # 可以到达该点
                logger.info(f"骑手 {courier['courier_id']} 到达 "
                             f"Order{route_point['order_id']}-{route_point['point_type']}")
                # 更新位置和剩余距离
                current_pos = target_pos
                remaining_distance -= distance_to_target
    
                # 处理取餐或送餐
                orders_dict = {order['order_id']: order for order in courier['current_orders']}
                order_id = route_point['order_id']
                order = orders_dict.get(order_id)
                if route_point['point_type'] == 'pickup':
                    # 取餐
                    arrival_time = current_time + timedelta(
                        seconds=(move_distance - remaining_distance) / (self.COURIER_SPEED * 1000 / 3600)
                    )
                    estimate_prepare_time = order.get('estimate_meal_prepare_time', arrival_time)
                    pickup_time, special_status = self.record_pickup_time(
                        order, arrival_time, estimate_prepare_time
                    )

                    # 更新订单状态
                    order_update = {
                        'order_id': order_id,
                        'status': 'picked_up',
                        'actual_pickup_time': pickup_time
                    }
                    updates['order_updates'].append(order_update)
                    # 同步更新骑手订单列表中的订单状态
                    for courier_order in courier['current_orders']:
                        if courier_order['order_id'] == order_id:
                            courier_order.update(order_update)

                            break

                    # 如果需要等待备餐，设置骑手特殊状态并返回
                    if special_status == 'waiting_meal':
                        courier['special_status'] = 'waiting_meal'
                        courier['special_status_end_time'] = pickup_time
                        courier['position'] = current_pos
                        updates['courier_updates'] = courier.copy()

                        return updates

                else:
                    # 送餐
                    arrival_time = current_time + timedelta(
                        seconds=(move_distance - remaining_distance) / (self.COURIER_SPEED * 1000 / 3600)
                    )
                    delivery_time, special_status = self.record_delivery_time(
                        order, arrival_time
                    )

                    # 更新订单状态
                    order_update = {
                        'order_id': order_id,
                        'status': 'delivered',
                        'actual_delivery_time': delivery_time
                    }
                    updates['order_updates'].append(order_update)
                    # 同步更新骑手订单列表中的订单状态
                    for courier_order in courier['current_orders']:
                        if courier_order['order_id'] == order_id:
                            courier_order.update(order_update)

                            break

                    # 设置骑手特殊状态（等待送餐）并返回
                    courier['special_status'] = 'waiting_delivery'
                    courier['special_status_end_time'] = delivery_time
                    courier['position'] = current_pos
                    updates['courier_updates'] = courier.copy()
                    
                    return updates

            else:
                # 不能到达该点，沿方向移动
                new_pos = calculate_move_position(current_pos, target_pos, distance_to_target, remaining_distance)
                current_pos = new_pos
                remaining_distance = 0
                logger.info(f"骑手 {courier['courier_id']} 移动到 {new_pos}")

                break

        # 更新骑手位置
        courier['position'] = current_pos
        updates['courier_updates'] = courier.copy()

        return updates


