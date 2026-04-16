# -*- coding: utf-8 -*-
"""
数据表生成器模块

该模块实现数据表的生成功能，将订单和骑手信息整合成一张数据表，
包含订单详细信息和对应的骑手ID。

作者：CodeArts代码智能体
日期：2025-04-13
"""

import logging
import pandas as pd
from typing import Dict, List, Any

# 配置日志
logger = logging.getLogger(__name__)


class DataTableGenerator:
    """
    数据表生成器

    该类负责将订单和骑手信息整合成一张数据表，
    包含订单详细信息和对应的骑手ID。
    """

    def __init__(self):
        """初始化数据表生成器"""
        logger.info("数据表生成器初始化完成")

    def get_order_row(
        self,
        order: Dict[str, Any],
        courier_id: str
    ) -> Dict[str, Any]:
        """
        获取单条订单记录

        参数：
            order: 订单信息，包含以下字段：
                - order_id: 订单ID
                - sender_lng, sender_lat: 取餐点坐标
                - recipient_lng, recipient_lat: 送餐点坐标
                - dispatch_time: 派单时间
                - estimate_meal_prepare_time: 预计备餐完成时间
                - estimate_arrived_time: 预计到达时间
                - actual_pickup_time: 实际取餐时间（可选）
                - actual_delivery_time: 实际送餐时间（可选）
            courier_id: 骑手ID

        返回：
            单条记录字典，包含以下字段：
                - order_id: 订单ID
                - courier_id: 骑手ID
                - sender_lng: 取餐点经度
                - sender_lat: 取餐点纬度
                - recipient_lng: 送餐点经度
                - recipient_lat: 送餐点纬度
                - dispatch_time: 派单时间
                - estimate_meal_prepare_time: 预计备餐完成时间
                - estimate_arrived_time: 预计到达时间
                - actual_pickup_time: 实际取餐时间
                - actual_delivery_time: 实际送餐时间

        示例：
            >>> order = {
            ...     'order_id': 'order1',
            ...     'sender_lng': 116.0, 'sender_lat': 39.0,
            ...     'recipient_lng': 116.1, 'recipient_lat': 39.1,
            ...     'dispatch_time': datetime(2022, 10, 17, 10, 0, 0),
            ...     'estimate_meal_prepare_time': datetime(2022, 10, 17, 10, 5, 0),
            ...     'estimate_arrived_time': datetime(2022, 10, 17, 10, 30, 0),
            ...     'actual_pickup_time': datetime(2022, 10, 17, 10, 5, 0),
            ...     'actual_delivery_time': datetime(2022, 10, 17, 10, 32, 0)
            ... }
            >>> row = generator.get_order_row(order, 'courier1')
            >>> print(row['order_id'])
            'order1'
        """
        row = {
            'order_id': order['order_id'],
            'courier_id': courier_id,
            'sender_lng': order['sender_lng'],
            'sender_lat': order['sender_lat'],
            'recipient_lng': order['recipient_lng'],
            'recipient_lat': order['recipient_lat'],
            'dispatch_time': order.get('dispatch_time'),
            'estimate_meal_prepare_time': order.get('estimate_meal_prepare_time'),
            'estimate_arrived_time': order.get('estimate_arrived_time'),
            'actual_pickup_time': order.get('actual_pickup_time'),
            'actual_delivery_time': order.get('actual_delivery_time')
        }

        return row

    def generate(
        self,
        orders: Dict[str, Dict[str, Any]],
        couriers: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        生成数据表

        将所有订单和对应的骑手信息整合成一张数据表。

        参数：
            orders: 订单字典 {order_id: order_info}
            couriers: 骑手字典 {courier_id: courier_info}
                每个骑手信息包含 'orders' 字段，为订单列表

        返回：
            pandas DataFrame，包含所有订单的详细信息

        示例：
            >>> orders = {
            ...     'order1': {'order_id': 'order1', ...},
            ...     'order2': {'order_id': 'order2', ...}
            ... }
            >>> couriers = {
            ...     'courier1': {
            ...         'courier_id': 'courier1',
            ...         'orders': [{'order_id': 'order1', ...}]
            ...     }
            ... }
            >>> df = generator.generate(orders, couriers)
            >>> print(len(df))
            1
        """
        # 构建订单到骑手的映射
        order_to_courier = {}

        for courier_id, courier in couriers.items():
            courier_orders = courier.get('orders', [])
            for order in courier_orders:
                order_id = order['order_id']
                order_to_courier[order_id] = courier_id

        # 构建数据行列表
        rows = []

        for order_id, order in orders.items():
            # 获取对应的骑手ID
            courier_id = order_to_courier.get(order_id, None)

            # 如果订单已分配给骑手，添加到数据表
            if courier_id:
                row = self.get_order_row(order, courier_id)
                rows.append(row)

        # 创建DataFrame
        df = pd.DataFrame(rows)

        # 设置列顺序
        columns = [
            'order_id',
            'courier_id',
            'sender_lng',
            'sender_lat',
            'recipient_lng',
            'recipient_lat',
            'dispatch_time',
            'estimate_meal_prepare_time',
            'estimate_arrived_time',
            'actual_pickup_time',
            'actual_delivery_time'
        ]

        if not df.empty:
            df = df[columns]

        logger.info(f"数据表生成完成，共 {len(df)} 条记录")

        return df

    def save_to_csv(
        self,
        df: pd.DataFrame,
        file_path: str
    ) -> None:
        """
        将数据表保存为CSV文件

        参数：
            df: 数据表
            file_path: 文件路径

        示例：
            >>> generator.save_to_csv(df, 'output/order_table.csv')
        """
        df.to_csv(file_path, index=False, encoding='utf-8')
        logger.info(f"数据表已保存到 {file_path}")


# 模块测试代码
if __name__ == '__main__':
    import logging
    from datetime import datetime
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("=" * 60)
    print("数据表生成器测试")
    print("=" * 60)

    # 创建数据表生成器
    generator = DataTableGenerator()

    # 测试数据
    orders = {
        'order1': {
            'order_id': 'order1',
            'sender_lng': 116.0,
            'sender_lat': 39.0,
            'recipient_lng': 116.1,
            'recipient_lat': 39.1,
            'dispatch_time': datetime(2022, 10, 17, 10, 0, 0),
            'estimate_meal_prepare_time': datetime(2022, 10, 17, 10, 5, 0),
            'estimate_arrived_time': datetime(2022, 10, 17, 10, 30, 0),
            'actual_pickup_time': datetime(2022, 10, 17, 10, 5, 0),
            'actual_delivery_time': datetime(2022, 10, 17, 10, 32, 0)
        },
        'order2': {
            'order_id': 'order2',
            'sender_lng': 116.2,
            'sender_lat': 39.2,
            'recipient_lng': 116.3,
            'recipient_lat': 39.3,
            'dispatch_time': datetime(2022, 10, 17, 10, 10, 0),
            'estimate_meal_prepare_time': datetime(2022, 10, 17, 10, 15, 0),
            'estimate_arrived_time': datetime(2022, 10, 17, 10, 40, 0),
            'actual_pickup_time': datetime(2022, 10, 17, 10, 15, 0),
            'actual_delivery_time': datetime(2022, 10, 17, 10, 42, 0)
        }
    }

    couriers = {
        'courier1': {
            'courier_id': 'courier1',
            'orders': [
                {
                    'order_id': 'order1',
                    'status': 'delivered'
                }
            ]
        },
        'courier2': {
            'courier_id': 'courier2',
            'orders': [
                {
                    'order_id': 'order2',
                    'status': 'delivered'
                }
            ]
        }
    }

    # 测试1：生成单条记录
    print("\n测试1：生成单条记录")
    row = generator.get_order_row(orders['order1'], 'courier1')
    print(f"订单ID: {row['order_id']}")
    print(f"骑手ID: {row['courier_id']}")
    print(f"取餐点: ({row['sender_lng']}, {row['sender_lat']})")
    print(f"送餐点: ({row['recipient_lng']}, {row['recipient_lat']})")
    print(f"派单时间: {row['dispatch_time']}")
    print(f"实际取餐时间: {row['actual_pickup_time']}")
    print(f"实际送餐时间: {row['actual_delivery_time']}")

    # 测试2：生成数据表
    print("\n测试2：生成数据表")
    df = generator.generate(orders, couriers)
    print(f"数据表行数: {len(df)}")
    print(f"数据表列数: {len(df.columns)}")
    print("\n数据表内容:")
    print(df)

    # 测试3：保存数据表
    print("\n测试3：保存数据表")
    output_path = 'Q-Values/Data/test_order_table.csv'
    generator.save_to_csv(df, output_path)
    print(f"数据表已保存到 {output_path}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
