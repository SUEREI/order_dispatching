# -*- coding: utf-8 -*-
"""
骑手路径规划与状态刷新系统测试模块

该模块包含所有模块的单元测试和集成测试。

作者：CodeArts代码智能体
日期：2025-04-13
"""

import unittest
from datetime import datetime, timedelta
import sys
import os


from Simulation.core.route import RoutePlanner
from Simulation.core.state import StateRefresher
from Simulation.core.data import DataTableGenerator
from Simulation.utils.geo_utils import haversine_distance, calculate_move_position


class TestGeoUtils(unittest.TestCase):
    """地理计算工具测试"""

    def test_haversine_distance(self):
        """测试距离计算"""
        # 北京天安门到故宫的距离（约276米）
        coord1 = (116.397128, 39.916527)  # 天安门
        coord2 = (116.397479, 39.918988)  # 故宫
        distance = haversine_distance(coord1, coord2)
        # 验证距离在合理范围内（250-300米）
        self.assertGreater(distance, 250)
        self.assertLess(distance, 300)

    def test_haversine_distance_same_point(self):
        """测试同一点距离为0"""
        coord = (116.0, 39.0)
        distance = haversine_distance(coord, coord)
        self.assertEqual(distance, 0)

    def test_calculate_move_position(self):
        """测试移动位置计算"""
        current_pos = (116.0, 39.0)
        target_pos = (116.1, 39.1)
        move_distance = 1000  # 移动1000米
        new_pos = calculate_move_position(current_pos, target_pos, move_distance)

        # 验证新位置在连线上
        # 计算实际移动距离
        actual_distance = haversine_distance(current_pos, new_pos)
        # 允许误差在5%以内
        self.assertAlmostEqual(actual_distance, move_distance, delta=50)

    def test_calculate_move_position_exceed_distance(self):
        """测试移动距离超过总距离"""
        current_pos = (116.0, 39.0)
        target_pos = (116.1, 39.1)
        move_distance = 100000  # 移动100公里（超过总距离）
        new_pos = calculate_move_position(current_pos, target_pos, move_distance)

        # 应该返回目标位置
        self.assertAlmostEqual(new_pos[0], target_pos[0], places=5)
        self.assertAlmostEqual(new_pos[1], target_pos[1], places=5)


class TestRoutePlanner(unittest.TestCase):
    """路径规划器测试"""

    def setUp(self):
        """测试前准备"""
        self.planner = RoutePlanner()

    def test_get_accessible_points_pending(self):
        """测试pending状态订单的可达点"""
        orders = [
            {
                'order_id': 'order1',
                'status': 'pending',
                'sender_lng': 116.0,
                'sender_lat': 39.0,
                'recipient_lng': 116.1,
                'recipient_lat': 39.1
            }
        ]
        points = self.planner.get_accessible_points(orders)

        # 应该只有取餐点
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]['point_type'], 'pickup')
        self.assertEqual(points[0]['order_id'], 'order1')

    def test_get_accessible_points_picked_up(self):
        """测试picked_up状态订单的可达点"""
        orders = [
            {
                'order_id': 'order1',
                'status': 'picked_up',
                'sender_lng': 116.0,
                'sender_lat': 39.0,
                'recipient_lng': 116.1,
                'recipient_lat': 39.1
            }
        ]
        points = self.planner.get_accessible_points(orders)

        # 应该只有送餐点
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]['point_type'], 'delivery')
        self.assertEqual(points[0]['order_id'], 'order1')

    def test_get_accessible_points_delivered(self):
        """测试delivered状态订单的可达点"""
        orders = [
            {
                'order_id': 'order1',
                'status': 'delivered',
                'sender_lng': 116.0,
                'sender_lat': 39.0,
                'recipient_lng': 116.1,
                'recipient_lat': 39.1
            }
        ]
        points = self.planner.get_accessible_points(orders)

        # 应该没有可达点
        self.assertEqual(len(points), 0)

    def test_select_nearest_point(self):
        """测试最近点选择"""
        current_pos = (116.0, 39.0)
        points = [
            {'point_type': 'pickup', 'order_id': 'order1', 'position': (116.1, 39.1)},
            {'point_type': 'pickup', 'order_id': 'order2', 'position': (116.05, 39.05)},
            {'point_type': 'pickup', 'order_id': 'order3', 'position': (116.2, 39.2)}
        ]
        nearest = self.planner.select_nearest_point(current_pos, points)

        # 应该选择order2（最近）
        self.assertEqual(nearest['order_id'], 'order2')

    def test_plan_route_single_order(self):
        """测试单订单路径规划"""
        courier = {
            'courier_id': 'courier1',
            'position': (116.0, 39.0),
            'orders': [
                {
                    'order_id': 'order1',
                    'status': 'pending',
                    'sender_lng': 116.1,
                    'sender_lat': 39.1,
                    'recipient_lng': 116.2,
                    'recipient_lat': 39.2
                }
            ]
        }
        route = self.planner.plan_route(courier)

        # 应该有2个点（取餐点 + 送餐点）
        self.assertEqual(len(route), 2)
        self.assertEqual(route[0]['point_type'], 'pickup')
        self.assertEqual(route[1]['point_type'], 'delivery')

    def test_plan_route_multiple_orders(self):
        """测试多订单路径规划"""
        courier = {
            'courier_id': 'courier1',
            'position': (116.0, 39.0),
            'orders': [
                {
                    'order_id': 'order1',
                    'status': 'pending',
                    'sender_lng': 116.1,
                    'sender_lat': 39.1,
                    'recipient_lng': 116.3,
                    'recipient_lat': 39.3
                },
                {
                    'order_id': 'order2',
                    'status': 'pending',
                    'sender_lng': 116.05,
                    'sender_lat': 39.05,
                    'recipient_lng': 116.15,
                    'recipient_lat': 39.15
                }
            ]
        }
        route = self.planner.plan_route(courier)

        # 应该有4个点（2个取餐点 + 2个送餐点）
        self.assertEqual(len(route), 4)

        # 验证先取后送约束
        is_valid = self.planner.validate_route(route)
        self.assertTrue(is_valid)

    def test_plan_route_empty_orders(self):
        """测试空订单路径规划"""
        courier = {
            'courier_id': 'courier1',
            'position': (116.0, 39.0),
            'orders': []
        }
        route = self.planner.plan_route(courier)

        # 应该返回空路径
        self.assertEqual(len(route), 0)

    def test_validate_route(self):
        """测试路径验证"""
        # 满足约束的路径
        route1 = [
            {'sequence': 1, 'point_type': 'pickup', 'order_id': 'order1', 'position': (116.1, 39.1)},
            {'sequence': 2, 'point_type': 'pickup', 'order_id': 'order2', 'position': (116.15, 39.15)},
            {'sequence': 3, 'point_type': 'delivery', 'order_id': 'order1', 'position': (116.2, 39.2)},
            {'sequence': 4, 'point_type': 'delivery', 'order_id': 'order2', 'position': (116.25, 39.25)}
        ]
        is_valid = self.planner.validate_route(route1)
        self.assertTrue(is_valid)

        # 不满足约束的路径（先送后取）
        route2 = [
            {'sequence': 1, 'point_type': 'delivery', 'order_id': 'order1', 'position': (116.2, 39.2)},
            {'sequence': 2, 'point_type': 'pickup', 'order_id': 'order1', 'position': (116.1, 39.1)}
        ]
        is_valid = self.planner.validate_route(route2)
        self.assertFalse(is_valid)


class TestStateRefresher(unittest.TestCase):
    """状态刷新器测试"""

    def setUp(self):
        """测试前准备"""
        self.planner = RoutePlanner()
        self.refresher = StateRefresher(self.planner)

    def test_record_pickup_time_wait(self):
        """测试取餐时间记录（需要等待）"""
        order = {'order_id': 'order1'}
        arrival_time = datetime(2022, 10, 17, 10, 0, 0)
        estimate_prepare_time = datetime(2022, 10, 17, 10, 5, 0)
        pickup_time, status = self.refresher.record_pickup_time(
            order, arrival_time, estimate_prepare_time
        )

        # 应该等待备餐完成
        self.assertEqual(pickup_time, estimate_prepare_time)
        self.assertEqual(status, 'waiting_meal')

    def test_record_pickup_time_no_wait(self):
        """测试取餐时间记录（不需要等待）"""
        order = {'order_id': 'order1'}
        arrival_time = datetime(2022, 10, 17, 10, 10, 0)
        estimate_prepare_time = datetime(2022, 10, 17, 10, 5, 0)
        pickup_time, status = self.refresher.record_pickup_time(
            order, arrival_time, estimate_prepare_time
        )

        # 应该直接取餐
        self.assertEqual(pickup_time, arrival_time)
        self.assertEqual(status, 'none')

    def test_record_delivery_time(self):
        """测试送餐时间记录"""
        order = {'order_id': 'order1'}
        arrival_time = datetime(2022, 10, 17, 10, 30, 0)
        delivery_time, status = self.refresher.record_delivery_time(order, arrival_time)

        # 应该是到达时间 + 2分钟
        expected_delivery_time = arrival_time + timedelta(seconds=120)
        self.assertEqual(delivery_time, expected_delivery_time)
        self.assertEqual(status, 'waiting_delivery')

    def test_handle_special_status_waiting(self):
        """测试特殊状态处理（等待中）"""
        courier = {
            'courier_id': 'courier1',
            'special_status': 'waiting_meal',
            'special_status_end_time': datetime(2022, 10, 17, 10, 5, 0)
        }
        current_time = datetime(2022, 10, 17, 10, 3, 0)
        result = self.refresher.handle_special_status(courier, current_time)

        # 应该处于等待状态
        self.assertEqual(result['status'], 'waiting')
        self.assertEqual(result['remaining_time'], 120)

    def test_handle_special_status_completed(self):
        """测试特殊状态处理（已完成）"""
        courier = {
            'courier_id': 'courier1',
            'special_status': 'waiting_meal',
            'special_status_end_time': datetime(2022, 10, 17, 10, 5, 0)
        }
        current_time = datetime(2022, 10, 17, 10, 6, 0)
        result = self.refresher.handle_special_status(courier, current_time)

        # 应该已完成
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['type'], 'waiting_meal')

    def test_handle_special_status_normal(self):
        """测试特殊状态处理（无特殊状态）"""
        courier = {
            'courier_id': 'courier1',
            'special_status': 'none',
            'special_status_end_time': None
        }
        current_time = datetime(2022, 10, 17, 10, 0, 0)
        result = self.refresher.handle_special_status(courier, current_time)

        # 应该是正常状态
        self.assertEqual(result['status'], 'normal')


class TestDataTableGenerator(unittest.TestCase):
    """数据表生成器测试"""

    def setUp(self):
        """测试前准备"""
        self.generator = DataTableGenerator()

    def test_get_order_row(self):
        """测试单条记录生成"""
        order = {
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
        }
        row = self.generator.get_order_row(order, 'courier1')

        # 验证字段
        self.assertEqual(row['order_id'], 'order1')
        self.assertEqual(row['courier_id'], 'courier1')
        self.assertEqual(row['sender_lng'], 116.0)
        self.assertEqual(row['sender_lat'], 39.0)
        self.assertEqual(row['recipient_lng'], 116.1)
        self.assertEqual(row['recipient_lat'], 39.1)

    def test_generate(self):
        """测试数据表生成"""
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
                'orders': [{'order_id': 'order1', 'status': 'delivered'}]
            },
            'courier2': {
                'courier_id': 'courier2',
                'orders': [{'order_id': 'order2', 'status': 'delivered'}]
            }
        }

        df = self.generator.generate(orders, couriers)

        # 验证数据表
        self.assertEqual(len(df), 2)
        self.assertIn('order_id', df.columns)
        self.assertIn('courier_id', df.columns)
        self.assertIn('sender_lng', df.columns)
        self.assertIn('recipient_lng', df.columns)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """测试前准备"""
        self.planner = RoutePlanner()
        self.refresher = StateRefresher(self.planner)
        self.generator = DataTableGenerator()

    def test_full_delivery_flow(self):
        """测试完整配送流程"""
        # 初始化骑手和订单
        courier = {
            'courier_id': 'courier1',
            'position': (116.0, 39.0),
            'status': 'online',
            'orders': [
                {
                    'order_id': 'order1',
                    'status': 'pending',
                    'sender_lng': 116.1,
                    'sender_lat': 39.1,
                    'recipient_lng': 116.2,
                    'recipient_lat': 39.2,
                    'estimate_meal_prepare_time': datetime(2022, 10, 17, 10, 5, 0)
                }
            ],
            'special_status': 'none',
            'special_status_end_time': None
        }

        orders_dict = {
            'order1': {
                'order_id': 'order1',
                'status': 'pending',
                'sender_lng': 116.1,
                'sender_lat': 39.1,
                'recipient_lng': 116.2,
                'recipient_lat': 39.2,
                'estimate_meal_prepare_time': datetime(2022, 10, 17, 10, 5, 0)
            }
        }

        current_time = datetime(2022, 10, 17, 10, 0, 0)

        # 多次刷新状态，模拟配送过程
        for i in range(200):
            result = self.refresher.refresh_courier(courier, orders_dict, current_time, 30)

            # 更新骑手状态
            if result['courier_updates']:
                courier = result['courier_updates']

            # 更新订单状态
            for order_update in result['order_updates']:
                order_id = order_update['order_id']
                orders_dict[order_id].update(order_update)
                # 同步更新骑手订单列表中的订单状态
                for order in courier['orders']:
                    if order['order_id'] == order_id:
                        order.update(order_update)

            current_time += timedelta(seconds=30)

            # 如果订单已送达，结束循环
            if orders_dict['order1']['status'] == 'delivered':
                break

        # 验证订单最终状态
        self.assertEqual(orders_dict['order1']['status'], 'delivered')
        self.assertIsNotNone(orders_dict['order1'].get('actual_pickup_time'))
        self.assertIsNotNone(orders_dict['order1'].get('actual_delivery_time'))

    def test_multiple_orders_delivery(self):
        """测试多订单配送场景"""
        # 初始化骑手和多个订单
        courier = {
            'courier_id': 'courier1',
            'position': (116.0, 39.0),
            'status': 'online',
            'orders': [
                {
                    'order_id': 'order1',
                    'status': 'pending',
                    'sender_lng': 116.1,
                    'sender_lat': 39.1,
                    'recipient_lng': 116.3,
                    'recipient_lat': 39.3,
                    'estimate_meal_prepare_time': datetime(2022, 10, 17, 10, 5, 0)
                },
                {
                    'order_id': 'order2',
                    'status': 'pending',
                    'sender_lng': 116.05,
                    'sender_lat': 39.05,
                    'recipient_lng': 116.15,
                    'recipient_lat': 39.15,
                    'estimate_meal_prepare_time': datetime(2022, 10, 17, 10, 5, 0)
                }
            ],
            'special_status': 'none',
            'special_status_end_time': None
        }

        orders_dict = {
            'order1': {
                'order_id': 'order1',
                'status': 'pending',
                'sender_lng': 116.1,
                'sender_lat': 39.1,
                'recipient_lng': 116.3,
                'recipient_lat': 39.3,
                'estimate_meal_prepare_time': datetime(2022, 10, 17, 10, 5, 0)
            },
            'order2': {
                'order_id': 'order2',
                'status': 'pending',
                'sender_lng': 116.05,
                'sender_lat': 39.05,
                'recipient_lng': 116.15,
                'recipient_lat': 39.15,
                'estimate_meal_prepare_time': datetime(2022, 10, 17, 10, 5, 0)
            }
        }

        # 规划路径
        route = self.planner.plan_route(courier)

        # 验证路径满足先取后送约束
        is_valid = self.planner.validate_route(route)
        self.assertTrue(is_valid)

        # 验证路径长度（2个取餐点 + 2个送餐点）
        self.assertEqual(len(route), 4)


# 运行测试
if __name__ == '__main__':
    print("=" * 60)
    print("骑手路径规划与状态刷新系统测试")
    print("=" * 60)
    print()

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestGeoUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestRoutePlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestStateRefresher))
    suite.addTests(loader.loadTestsFromTestCase(TestDataTableGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 60)
    print(f"测试完成: {result.testsRun} 个测试")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 60)
