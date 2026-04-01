import pandas as pd
import numpy as np
import pickle
# 设置pandas显示选项，确保列字段显示完整
pd.set_option('display.max_columns', None)


class Sample_SetProcessor:
    """
    Sample_Set 数据处理类
    用于处理订单数据，进行特征重命名、预测和计算惩罚比例
    """
    
    def __init__(self, data_path, model_path, scaler_path, feature_columns_path):
        """
        初始化 Sample_SetProcessor 类
        
        :param data_path: 数据文件路径
        :param model_path: 模型文件路径
        :param scaler_path: 标准化器文件路径
        :param feature_columns_path: 特征列文件路径
        """
        self.data_path = data_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_columns_path = feature_columns_path
        self.df = None
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.df_output = None  # 用于存储最终输出
    
    def load_data(self):
        """
        加载数据文件
        """
        # print(f"加载数据文件: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        # print("当前字段名称：")
        # print(self.df.columns.tolist())
    
    def load_model(self):
        """
        加载模型和标准化器
        """
        # print("加载模型和标准化器...")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(self.feature_columns_path, 'r', encoding='utf-8') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]
        # print(f'加载的模型特征 ({len(self.feature_columns)}个)：{self.feature_columns}')
    
    def rename_and_reorder_columns(self):
        """
        重命名和重新排列DataFrame的列
        """
        if self.df is None:
            raise ValueError("数据未加载，请先调用 load_data() 方法")
        
        old_names = [
            # 距离相关特征
            'order_distance_km', 'rider_to_order_km', 
            # 工作量相关特征
            'ongoing_order_count', 'ongoing_cross_area_count', 'ongoing_max_distance_km', 'ongoing_area_km2', 
            # 新订单与现有订单适配性相关特征
            'area_change_ratio', 'direction_similarity', 
            # 疲惫程度相关特征
            'courier_daily_completed_distance_km', 'courier_session_completed_distance_km',
            # 区域熟悉度相关特征
            'area_familiarity', 
            # 时间相关特征
            'is_weekend', 'dispatch_hour']  # 旧字段名列表

        new_names = [
            # 距离相关特征
            'sender_to_recipient_km', 'dispatch_to_sender_km', 
            # 工作量相关特征
            'ongoing_other_count', 'ongoing_cross_area_count', 'ongoing_max_distance_km', 'ongoing_area_km2', 
            # 新订单与现有订单适配性相关特征
            'area_increase_ratio', 'direction_similarity', 
            # 疲惫程度相关特征
            'courier_daily_completed_before_dispatch_distance', 'courier_daily_completed_before_dispatch_with_reset_distance',
            # 区域熟悉度相关特征
            'area_familiarity', 
            # 时间相关特征
            'is_weekend', 'dispatch_hour']  # 新字段名列表

        # 将两个列表组合成字典
        new_columns = dict(zip(old_names, new_names))

        # 按照指定顺序排列字段
        column_order = new_names  # 按需要的顺序填写

        # 重命名字段
        self.df = self.df.rename(columns=new_columns)

        # 重新排列字段顺序
        # 保留 order_id 和 dispatch_to_estimateArrived_tm 字段
        columns_to_keep = ['order_id'] if 'order_id' in self.df.columns else []
        if 'dispatch_to_estimateArrived_tm' in self.df.columns:
            columns_to_keep.append('dispatch_to_estimateArrived_tm')
        
        self.df = self.df[column_order + columns_to_keep]
    
    def predict(self, threshold=0.5):
        """
        使用模型进行预测
        
        :param threshold: 分类阈值，默认为0.5
        """
        if self.df is None:
            raise ValueError("数据未加载，请先调用 load_data() 方法")
        if self.model is None or self.scaler is None:
            raise ValueError("模型未加载，请先调用 load_model() 方法")
        
        # 准备预测数据
        available_features = [col for col in self.feature_columns if col in self.df.columns]
        # print(f'\n可用特征 ({len(available_features)}个)：{available_features}')
        
        X = self.df[available_features].copy()
        X = X.fillna(X.median())
        
        X_scaled = self.scaler.transform(X)
        
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        self.df['xgboost_prediction'] = y_pred
        self.df['xgboost_probability'] = y_pred_proba
    
    def calculate_reject_penalty(self):
        """
        计算 reject_penalty_ratio 字段
        """
        if self.df is None:
            raise ValueError("数据未加载，请先调用 load_data() 方法")
        
        # 计算 reject_penalty_ratio 字段
        if 'dispatch_to_estimateArrived_tm' in self.df.columns and 'xgboost_prediction' in self.df.columns:
            self.df['reject_penalty_ratio'] = np.where(
                self.df['xgboost_prediction'] == 0,
                1 / self.df['dispatch_to_estimateArrived_tm'],
                0
            )
            # print("\nreject_penalty_ratio 计算完成")
        else:
            print("\n警告：缺少必要字段，无法计算 reject_penalty_ratio")
    
    def process(self, threshold=0.5):
        """
        执行完整的处理流程
        
        :param threshold: 分类阈值，默认为0.5
        :return: 处理后的数据框
        """
        self.load_data()
        self.load_model()
        self.rename_and_reorder_columns()
        self.predict(threshold)
        self.calculate_reject_penalty()
        
        # 只保留特征和 reject_penalty_ratio 字段（保留 order_id 用于随机选择）
        feature_columns = [
            'sender_to_recipient_km', 'dispatch_to_sender_km', 
            'ongoing_other_count', 'ongoing_cross_area_count', 'ongoing_max_distance_km', 'ongoing_area_km2', 
            'area_increase_ratio', 'direction_similarity', 
            'courier_daily_completed_before_dispatch_distance', 'courier_daily_completed_before_dispatch_with_reset_distance',
            'area_familiarity', 
            'is_weekend', 'dispatch_hour'
        ]
        
        # 添加 order_id 和 reject_penalty_ratio 字段
        columns_to_keep = ['order_id'] if 'order_id' in self.df.columns else []
        columns_to_keep = columns_to_keep + feature_columns + ['reject_penalty_ratio']
        
        # 过滤字段
        self.df_output = self.df[columns_to_keep].copy()
        # print(f"\n字段过滤完成，保留 {len(self.df_output.columns)} 个字段")
        
        return self.df_output
    
    def get_order_data(self, order_id=None, sample_size=10):
        """
        获取指定 order_id 的数据
        
        :param order_id: 指定的订单ID，如果为None则随机选择
        :param sample_size: 采样数量，默认为10
        :return: 不包含 order_id 的数据框
        """
        self.process()
        if self.df_output is None:
            raise ValueError("数据未处理，请先调用 process() 方法")
        
        if 'order_id' not in self.df_output.columns:
            print("警告：数据中不包含 order_id 字段")
            return self.df_output.drop(columns=['order_id'], errors='ignore')
        
        # 如果未指定 order_id，则随机选择一个
        if order_id is None:
            unique_order_ids = self.df_output['order_id'].unique()
            if len(unique_order_ids) == 0:
                print("警告：没有可用的 order_id")
                return pd.DataFrame()
            order_id = np.random.choice(unique_order_ids)
            print(f"随机选择的 order_id: {order_id}")
        
        # 获取该 order_id 的所有数据
        order_data = self.df_output[self.df_output['order_id'] == order_id].copy()
        
        if len(order_data) == 0:
            print(f"警告：未找到 order_id 为 {order_id} 的数据")
            return pd.DataFrame()
        
        # 如果数据超过 sample_size 条，随机选择 sample_size 条
        if len(order_data) > sample_size:
            order_data = order_data.sample(n=sample_size, random_state=42)
        
        # 移除 order_id 字段
        order_data = order_data.drop(columns=['order_id'], errors='ignore')
        
        return order_data
    
    def display_results(self, order_id=None, sample_size=10):
        """
        显示处理结果
        
        :param order_id: 指定的订单ID，如果为None则随机选择
        :param sample_size: 采样数量，默认为10
        """
        order_data = self.get_order_data(order_id, sample_size)
        
        if order_data.empty:
            return
        
        print(f"\n该订单共有 {len(order_data)} 条数据")
        print("\n输出数据：")
        print(order_data.to_string())
        
        # 保存到文件
        output_file = f'sample_order_{order_id if order_id else "random"}.csv'
        order_data.to_csv(output_file, index=False)
        print(f"\n数据已保存到: {output_file}")


if __name__ == '__main__':
    # 初始化处理器
    processor = Sample_SetProcessor(
        data_path='virtual_dispatch_result_v2.csv',
        model_path=r'Models\pkl_s\xgboost\xgboost_model.pkl',
        scaler_path=r'Models\pkl_s\xgboost\scaler.pkl',
        feature_columns_path=r'Models\pkl_s\xgboost\feature_columns.txt'
    )
    
    # 执行处理流程
    df = processor.process()
    
    # 显示结果（可以传入指定的 order_id）
    # processor.display_results(order_id=12345)  # 指定 order_id
    processor.display_results()  # 随机选择
