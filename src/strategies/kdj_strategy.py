"""
KDJ策略 - 基于KDJ指标的交易策略
"""
import pandas as pd
from .base_strategy import BaseStrategy
from ..operators import KDJOperator


class KDJStrategy(BaseStrategy):
    """基于KDJ指标的交易策略"""
    
    def __init__(self, buy_threshold=0, sell_threshold=70, n=9, m1=3, m2=3, name="KDJ"):
        """
        初始化KDJ策略
        
        Args:
            buy_threshold: 买入阈值，J值低于此值视为买入信号
            sell_threshold: 卖出阈值，J值高于此值视为卖出信号
            n: KDJ指标的窗口大小
            m1: K值的平滑因子
            m2: D值的平滑因子
            name: 策略名称
        """
        super().__init__(name)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.n = n
        self.m1 = m1
        self.m2 = m2
        
        # 添加依赖的KDJ算子
        self.kdj_operator = KDJOperator()
        self.add_operator(self.kdj_operator)
    
    def get_buy_signals(self, df, date=None):
        """
        获取买入信号
        
        Args:
            df: 包含KDJ指标的DataFrame
            date: 指定日期，如果为None则使用最新日期
            
        Returns:
            布尔值，表示是否有买入信号
        """
        if 'J' not in df.columns:
            df = self.kdj_operator(df, n=self.n, m1=self.m1, m2=self.m2)
        
        if date is None:
            # 使用最新日期（第一行，因为数据是降序的）
            date = df['trade_date'].iloc[0]
        
        # 获取指定日期的数据
        current_row = df[df['trade_date'] == date]
        if current_row.empty:
            return False
        
        current_j = current_row['J'].iloc[0]
        
        # 获取前一日数据（在降序排列中是下一行）
        next_date_idx = df[df['trade_date'] < date].index.min()
        if pd.isna(next_date_idx):
            return current_j <= self.buy_threshold
        
        prev_j = df.loc[next_date_idx, 'J']
        
        # J值穿越买入阈值（从上方向下）
        return current_j <= self.buy_threshold and prev_j > self.buy_threshold
    
    def get_sell_signals(self, df, date=None):
        """
        获取卖出信号
        
        Args:
            df: 包含KDJ指标的DataFrame
            date: 指定日期，如果为None则使用最新日期
            
        Returns:
            布尔值，表示是否有卖出信号
        """
        if 'J' not in df.columns:
            df = self.kdj_operator(df, n=self.n, m1=self.m1, m2=self.m2)
        
        if date is None:
            # 使用最新日期（第一行，因为数据是降序的）
            date = df['trade_date'].iloc[0]
        
        # 获取指定日期的数据
        current_row = df[df['trade_date'] == date]
        if current_row.empty:
            return False
        
        current_j = current_row['J'].iloc[0]
        
        # 获取前一日数据（在降序排列中是下一行）
        next_date_idx = df[df['trade_date'] < date].index.min()
        if pd.isna(next_date_idx):
            return current_j >= self.sell_threshold
        
        prev_j = df.loc[next_date_idx, 'J']
        
        # J值穿越卖出阈值（从下方向上）
        return current_j >= self.sell_threshold and prev_j < self.sell_threshold
    
    def generate_signals(self, stock_data_dict, date=None):
        """
        生成交易信号
        
        Args:
            stock_data_dict: 股票数据字典，键为股票代码，值为DataFrame
            date: 指定日期，如果为None则使用每只股票的最新日期
            
        Returns:
            dict: 包含买入信号和卖出信号的股票代码
        """
        buy_signals = []
        sell_signals = []
        
        for stock_code, df in stock_data_dict.items():
            # 计算KDJ指标
            df_kdj = self.kdj_operator(df, n=self.n, m1=self.m1, m2=self.m2)
            
            # 获取指定日期或最新日期
            if date is None:
                stock_date = df_kdj['trade_date'].iloc[0]
            else:
                stock_date = date
                if stock_date not in df_kdj['trade_date'].values:
                    print(f"警告：股票 {stock_code} 在 {stock_date} 没有数据")
                    continue
            
            # 检查买入信号
            if self.get_buy_signals(df_kdj, stock_date):
                buy_signals.append(stock_code)
            
            # 检查卖出信号
            if self.get_sell_signals(df_kdj, stock_date):
                sell_signals.append(stock_code)
        
        return {
            'buy': buy_signals,
            'sell': sell_signals
        } 