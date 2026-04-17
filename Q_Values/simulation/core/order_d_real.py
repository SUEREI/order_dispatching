#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单分派器 - 基于真实订单-骑手匹配数据
"""
import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


class OrderDispatcherByRealMatch:
    """基于真实订单-骑手匹配数据的订单分派器"""
    def __init__(self, order_data: pd.DataFrame):
        """初始化分派器，根据真实数据构建订单-骑手映射"""
        logger.info("真实匹配订单分派器初始化完成")
        self.order_courier_map: Dict[str, str] = {}
        self._build_order_courier_map(order_data)

    def _build_order_courier_map(self, order_data: pd.DataFrame) -> None:
        """从原始数据构建订单ID到骑手ID的映射"""
        if 'order_id' in order_data.columns and 'courier_id' in order_data.columns:
            valid_data = order_data[['order_id', 'courier_id']].dropna()
            self.order_courier_map = dict(zip(
                valid_data['order_id'].astype(str),
                valid_data['courier_id'].astype(str)
            ))
            logger.info(f"已构建 {len(self.order_courier_map)} 条订单-骑手映射关系")

    def dispatch_orders_by_real(self, orders, couriers) -> Dict[str, str]:
        """根据真实数据批量处理订单分派"""
        dispatch_results = {}
        for order in orders:
            order_id = str(order['order_id'])
            if order_id in self.order_courier_map:
                courier_id = self.order_courier_map[order_id]
                if courier_id in couriers:
                    dispatch_results[order_id] = courier_id
                    logger.debug(f"订单 {order_id} 匹配到真实骑手 {courier_id}")
                else:
                    logger.error(f"订单 {order_id} 匹配的骑手 {courier_id} 不在可用骑手列表中")
            else:
                logger.error(f"订单 {order_id} 在真实匹配数据中未找到")

        return dispatch_results