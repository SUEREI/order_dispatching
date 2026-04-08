class DeliverySimulation:
    def __init__(self, order_data_path, idle_timeout_minutes=10):
        # 加载数据
        self.df = pd.read_csv(order_data_path)
        self.df['dispatch_time'] = pd.to_datetime(self.df['dispatch_time'])
        self.df['finish_time'] = pd.to_datetime(self.df['finish_time'])
        
        # 初始化两个生成器
        self.order_gen = OrderGenerator(order_data_path)  # 复用你的类
        self.rider_sim = RiderSimulator(self.df, idle_timeout_minutes)
        
        # 骑手状态字典： rider_id -> {'status': 'online'/'offline', 'position': (lng, lat), ...}
        self.riders = {}
        self._init_riders()
        
        # 时间轴管理
        self.current_time = self.order_gen.get_time_range()[0]  # 从最早订单时间开始
        self.order_timestamps = self.order_gen.get_all_timestamps()
        self.order_index = 0  # 指向下一个要处理的订单时间
        
        # 待分配订单池（用于DQN决策）
        self.pending_orders = []
        
        logger.info(f"仿真初始化完成，起始时间: {self.current_time}")
    
    def _init_riders(self):
        """初始化所有骑手状态，默认全部离线"""
        all_riders = self.rider_sim.get_all_rider_ids()
        for rid in all_riders:
            self.riders[rid] = {
                'status': 'offline',
                'position': None,  # 上线时会设置初始位置
                'current_orders': []
            }
    
    def step(self, action=None):
        """
        执行一步仿真：处理当前时间点所有事件，然后推进到下一个事件时间
        action: DQN的动作（暂未使用，预留接口）
        返回：state, reward, done
        """
        # 1. 处理当前时间点的骑手上下线事件
        rider_events = self.rider_sim.get_events_until(self.current_time)
        for rid, etype in rider_events:
            if etype == 'online':
                self._rider_online(rid)
            else:
                self._rider_offline(rid)
        
        # 2. 处理当前时间点的订单生成事件
        if self.order_index < len(self.order_timestamps):
            if self.order_timestamps[self.order_index] == self.current_time:
                new_orders = self.order_gen.get_orders_at_time(self.current_time)
                self.pending_orders.extend(new_orders)
                self.order_index += 1
                # 跳过同一时间戳可能还有的其他订单时间（因为你按 dispatch_time 分组，可能多个订单同一秒）
                while (self.order_index < len(self.order_timestamps) and 
                       self.order_timestamps[self.order_index] == self.current_time):
                    new_orders = self.order_gen.get_orders_at_time(self.current_time)
                    self.pending_orders.extend(new_orders)
                    self.order_index += 1
        
        # 3. 如果 pending_orders 不为空，DQN 需要在此决策
        #    （这里只是框架，实际 DQN 会在 step 前调用 act，step 传入 action）
        
        # 4. 确定下一个事件时间，推进 current_time
        next_order_time = self.order_timestamps[self.order_index] if self.order_index < len(self.order_timestamps) else None
        next_rider_time = self.rider_sim.get_next_event_time()
        
        candidate_times = [t for t in [next_order_time, next_rider_time] if t is not None]
        if not candidate_times:
            return None, 0, True  # 仿真结束
        
        self.current_time = min(candidate_times)
        
        # 5. 构造状态向量（根据你的DQN需求实现）
        state = self._get_state()
        reward = self._calculate_reward()  # 需实现奖励逻辑
        done = (self.order_index >= len(self.order_timestamps) and 
                self.rider_sim.get_next_event_time() is None and 
                len(self.pending_orders) == 0)
        
        return state, reward, done
    
    def _rider_online(self, rider_id):
        """骑手上线时的处理：设置状态，初始化位置"""
        self.riders[rider_id]['status'] = 'online'
        # 初始位置可以设为历史数据中该骑手第一单的商家位置，或者城市中心随机
        # 这里简化为从历史订单中找该骑手最早订单的商家坐标
        first_order = self.df[self.df['rider_id'] == rider_id].iloc[0]
        self.riders[rider_id]['position'] = (first_order['sender_lng'], first_order['sender_lat'])
        logger.debug(f"骑手 {rider_id} 上线，位置: {self.riders[rider_id]['position']}")
    
    def _rider_offline(self, rider_id):
        """骑手下线处理"""
        self.riders[rider_id]['status'] = 'offline'
        # 实际中如果骑手还有未完成订单应拒绝下线，但我们的推断基于完成时间之后，所以不会有未完成单
        logger.debug(f"骑手 {rider_id} 下线")
    
    def _get_state(self):
        """构造状态向量，此处需根据你的DQN定制"""
        # 示例：返回当前待分配订单数、在线骑手数、平均已接单数等
        online_count = sum(1 for r in self.riders.values() if r['status'] == 'online')
        return np.array([len(self.pending_orders), online_count, self.current_time.hour])
    
    def _calculate_reward(self):
        # 暂时返回0，实际训练时需结合订单完成/超时计算
        return 0