import pandas as pd
import logging
from Courier_Simulator import CourierSimulator
from Order_Generator import OrderGenerator
from Order_Dispatcher import OrderDispatcher
from Route_Planer import RoutePlanner
from Courier_State_Refresh import CourierStateRefresh
from Order_State_Refresh import OrderStateRefresh
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeliverySimulation:
    def __init__(self, order_data_path, idle_timeout_minutes=30):
        # 加载数据
        self.df = pd.read_csv(order_data_path)
        self.df['dispatch_time'] = pd.to_datetime(self.df['dispatch_time'])
        self.df['arrive_time'] = pd.to_datetime(self.df['arrive_time'])
        
        # 初始化生成器、派单器和状态刷新器
        self.order_gen = OrderGenerator(self.df)  # 复用你的类
        self.courier_sim = CourierSimulator(self.df, idle_timeout_minutes)
        self.dispatcher = OrderDispatcher()  # 初始化订单派单器
        self.route_planner = RoutePlanner()  # 初始化路径规划器
        self.courier_refresher = CourierStateRefresh()  # 初始化骑手状态刷新器
        self.order_refresher = OrderStateRefresh()  # 初始化订单状态刷新器
        
        # 骑手状态字典： courier_id -> {'status': 'online'/'offline', 'position': (lng, lat), ...}
        self.couriers = {}
        self._init_couriers()
        
        # 订单状态字典： order_id -> order_info
        self.orders = {}
        
        # 时间轴管理
        self.current_time = self.order_gen.get_time_range()[0]  # 从最早订单时间开始
        self.order_timestamps = self.order_gen.get_all_timestamps()
        self.order_index = 0  # 指向下一个要处理的订单时间
        
        # 待分配订单池（用于DQN决策）
        self.order_pool = 0
        self.pending_orders = []
        
        logger.info(f"仿真初始化完成，起始时间: {self.current_time}")
    
    def _init_couriers(self):
        """初始化所有骑手状态，默认全部离线"""
        all_couriers = self.courier_sim.get_all_courier_ids()
        for cid in all_couriers:
            self.couriers[cid] = {
                'status': 'offline',
                'position': None,  # 上线时会设置初始位置
                'current_orders': []
            }
    
    def step(self, action=None):
        """
        执行一步仿真：处理当前时间点所有事件，先返回当前状态，然后推进到下一个事件时间
        action: DQN的动作（暂未使用，预留接口）
        返回：state, reward, done, current_time（返回当前处理的时间点）
        """
        # 保存当前时间（用于返回）
        current_time = self.current_time
        
        # 重置当前时间步的新订单数量
        self.order_pool = 0
        
        # 1. 处理当前时间点的骑手上下线事件
        courier_events = self.courier_sim.get_events_until(self.current_time)
        for cid, etype, pos, orders in courier_events:
            self.couriers[cid]['status'] = etype
            self.couriers[cid]['position'] = pos
            self.couriers[cid]['current_orders'] = orders
        online_couriers = {cid: info for cid, info in self.couriers.items() if info['status'] == 'online'}
        logger.debug(f"当前时间点 {self.current_time} 的在线骑手数量: {len(online_couriers)}")
        
        # 2. 处理当前时间点的订单生成事件
        if self.order_index < len(self.order_timestamps):
            if self.order_timestamps[self.order_index] == self.current_time:
                new_orders = self.order_gen.get_orders_at_time(self.current_time)
                self.pending_orders.extend(new_orders)
                self.order_pool = len(new_orders)
                self.order_index += 1
        
        # 3. 如果 pending_orders 不为空，使用派单器将订单派给最近的骑手
        if self.pending_orders:
            # 派单给最近的骑手
            dispatch_results = self.dispatcher.dispatch_orders(self.pending_orders, online_couriers)
            logger.debug(f"当前时间点 {self.current_time} 的订单派单结果: {dispatch_results}")
            
            # 处理派单结果
            dispatched_orders = []
            for order in self.pending_orders:
                if order['order_id'] in dispatch_results:
                    courier_id = dispatch_results[order['order_id']]
                    # 将订单添加到骑手的当前订单列表
                    self.couriers[courier_id]['current_orders'].append(order)
                    dispatched_orders.append(order)
            
            # 从待分配订单池中移除已派单的订单
            for order in dispatched_orders:
                self.pending_orders.remove(order)
        
        # # 4. 刷新骑手状态
        # self.couriers = self.courier_refresher.refresh_all_couriers(self.couriers, current_time)
        
        # # 5. 刷新订单状态
        # self.orders = self.order_refresher.refresh_orders_from_couriers(self.couriers, current_time)
        
        # # 显示分配了订单的骑手信息
        # active_couriers = {cid: info for cid, info in self.couriers.items() 
        #                 if info['status'] == 'online' and len(info['current_orders']) > 0}
        # logger.info(f"当前时间: {current_time}, 有订单的骑手数: {len(active_couriers)}, 有订单的骑手: {active_couriers}")
        
        # 4. 构造状态向量（根据你的DQN需求实现）- 先返回当前状态
        state = self._get_state()
        reward = self._calculate_reward()  # 需实现奖励逻辑
        done = (self.order_index >= len(self.order_timestamps) and 
                self.courier_sim.get_next_event_time() is None and 
                len(self.pending_orders) == 0)
        
        # 5. 确定下一个事件时间，推进 current_time（在返回状态后再推进）
        if not done:
            next_order_time = self.order_timestamps[self.order_index] if self.order_index < len(self.order_timestamps) else None
            next_courier_time = self.courier_sim.get_next_event_time()
            
            candidate_times = [t for t in [next_order_time, next_courier_time] if t is not None]
            if candidate_times:
                self.current_time = min(candidate_times)
        
        return state, reward, done, current_time
    
    def _courier_online(self, courier_id, pos=None):
        """骑手上线时的处理：设置状态，初始化位置"""
        self.couriers[courier_id]['status'] = 'online'
        if pos is None:
            # 初始位置可以设为历史数据中该骑手第一单的商家位置，或者城市中心随机
            # 这里简化为从历史订单中找该骑手最早订单的商家坐标
            logger.warning(f"骑手 {courier_id} 上线时位置缺失")
        else:
            self.couriers[courier_id]['position'] = pos
        # 这里简化为从历史订单中找该骑手最早订单的商家坐标
        logger.debug(f"骑手 {courier_id} 上线，位置: {self.couriers[courier_id]['position']}")
    
    def _courier_offline(self, courier_id, pos=None):
        """骑手下线处理：设置状态，初始化位置"""
        self.couriers[courier_id]['status'] = 'offline'
        if pos is None:
            # 初始位置可以设为历史数据中该骑手第一单的商家位置，或者城市中心随机
            # 这里简化为从历史订单中找该骑手最早订单的商家坐标
            logger.warning(f"骑手 {courier_id} 下线时位置缺失")
        else:
            self.couriers[courier_id]['position'] = pos
        # 这里简化为从历史订单中找该骑手最早订单的商家坐标
        logger.debug(f"骑手 {courier_id} 下线，位置: {self.couriers[courier_id]['position']}")
    
    def _get_state(self):
        """构造状态向量，此处需根据你的DQN定制"""
        # 示例：返回当前待分配订单数、在线骑手数、平均已接单数等
        online_count = sum(1 for c in self.couriers.values() if c['status'] == 'online')
        return np.array([len(self.pending_orders), online_count, self.current_time.hour])
    
    def _calculate_reward(self):
        # 暂时返回0，实际训练时需结合订单完成/超时计算
        return 0


if __name__ == '__main__':
    # 假设你的数据路径正确
    sim = DeliverySimulation('Q-values/Data/Data_Processed/cleaned_waybill_info.csv')
    
    # 运行几步观察事件
    for i in range(1, 11):
        state, reward, done, current_time = sim.step()
        print(f"Step {i}: time={current_time}, order_pool={sim.order_pool}, pending_orders={len(sim.pending_orders)}, online_couriers={sum(1 for c in sim.couriers.values() if c['status']=='online')}")    
        if done:
            break