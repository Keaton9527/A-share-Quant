"""
LSTM策略 - 基于LSTM模型预测涨跌的交易策略
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from .base_strategy import BaseStrategy
from ..operators import KDJOperator, MACDOperator
from ..models import LSTMModel


class LSTMStrategy(BaseStrategy):
    """基于LSTM模型的交易策略"""
    
    def __init__(self, sequence_length=10, model_path=None, train_before_predict=True, name="LSTM"):
        """
        初始化LSTM策略
        
        Args:
            sequence_length: 输入序列长度（使用多少天的历史数据）
            model_path: 模型保存路径，如果指定且文件存在，则加载模型
            train_before_predict: 在预测前是否需要训练模型
            name: 策略名称
        """
        super().__init__(name)
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.train_before_predict = train_before_predict
        
        # 创建LSTM模型
        self.model = LSTMModel(sequence_length=sequence_length)
        
        # 添加依赖的算子
        self.kdj_operator = KDJOperator()
        self.macd_operator = MACDOperator()
        self.add_operator(self.kdj_operator)
        self.add_operator(self.macd_operator)
        
        # 如果指定了模型路径，尝试加载模型
        if model_path and os.path.exists(model_path):
            self.model.load(model_path)
    
    def prepare_features(self, df):
        """
        准备特征数据
        
        Args:
            df: 原始股票数据
        
        Returns:
            添加了技术指标的DataFrame
        """
        # 计算KDJ指标
        df = self.kdj_operator(df)
        
        # 计算MACD指标
        df = self.macd_operator(df)
        
        return df
    
    def train_model(self, stock_data_dict, test_size=0.2):
        """
        训练LSTM模型
        
        Args:
            stock_data_dict: 股票数据字典
            test_size: 测试集比例
        
        Returns:
            训练结果
        """
        print(f"开始训练LSTM模型，使用 {len(stock_data_dict)} 只股票的数据...")
        
        # 准备训练数据
        X_all, y_all = [], []
        
        for stock_code, df in stock_data_dict.items():
            # 添加技术指标
            df_with_features = self.prepare_features(df)
            
            # 预处理数据
            X, y = self.model.preprocess(df_with_features)
            
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
        
        if not X_all:
            raise ValueError("没有足够的数据用于训练")
        
        # 合并所有股票的数据
        X_train = np.vstack(X_all)
        y_train = np.hstack(y_all)
        
        # 训练模型
        history = self.model.train(
            X_train, y_train,
            validation_split=test_size,
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # 保存模型
        if self.model_path:
            self.model.save(self.model_path)
        
        return history
    
    def generate_signals(self, stock_data_dict, date=None):
        """
        生成交易信号
        
        Args:
            stock_data_dict: 股票数据字典，键为股票代码，值为DataFrame
            date: 指定日期，如果为None则使用每只股票的最新日期
            
        Returns:
            dict: 包含买入信号和卖出信号的股票代码
        """
        # 如果模型未训练且需要训练，先训练模型
        if not self.model.is_trained and self.train_before_predict:
            self.train_model(stock_data_dict)
        
        buy_signals = []
        sell_signals = []
        
        print(f"正在为 {len(stock_data_dict)} 只股票生成LSTM预测信号...")
        
        for stock_code, df in stock_data_dict.items():
            # 添加技术指标
            df_with_features = self.prepare_features(df)
            
            # 获取指定日期或最新日期的数据
            if date is None:
                # 使用最新可用数据
                prediction = self.model.predict_for_stock(df_with_features)
            else:
                # 获取截至指定日期的数据
                if date in df_with_features['trade_date'].values:
                    date_idx = df_with_features[df_with_features['trade_date'] == date].index[0]
                    df_until_date = df_with_features.loc[df_with_features.index >= date_idx]
                    prediction = self.model.predict_for_stock(df_until_date)
                else:
                    print(f"警告：股票 {stock_code} 在 {date} 没有数据")
                    continue
            
            # 根据预测结果生成信号
            if prediction == 1:  # 买入信号
                buy_signals.append(stock_code)
            elif prediction == -1:  # 卖出信号
                sell_signals.append(stock_code)
        
        return {
            'buy': buy_signals,
            'sell': sell_signals
        } 