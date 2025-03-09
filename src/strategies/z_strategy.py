"""
Z策略 - 基于z哥投资逻辑的交易策略
结合KDJ指标和量价关系的B1入场法则
"""
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from ..operators import (
    KDJOperator, 
    MACDOperator, 
    RelativeVolumeOperator,
    VolumeChangeOperator, 
    VolumeBreakoutOperator
)


class ZStrategy(BaseStrategy):
    """基于z哥投资逻辑的交易策略，结合量价关系"""
    
    def __init__(
        self, 
        kdj_buy_threshold=20,    # J值低于此值视为超卖
        kdj_sell_threshold=80,   # J值高于此值视为超买
        vol_breakout_threshold=1.8,  # 成交量突破阈值（相对于20日均量）
        consecutive_vol_days=5,   # 连续成交量放大的天数要求
        stop_loss_pct=0.03,      # 止损百分比（相对于买入点）
        vol_change_threshold=0.3, # 量能变化率阈值
        name="Z战法"
    ):
        """
        初始化Z策略
        
        Args:
            kdj_buy_threshold: KDJ的J值低于此值视为超卖区
            kdj_sell_threshold: KDJ的J值高于此值视为超买区
            vol_breakout_threshold: 成交量突破阈值
            consecutive_vol_days: 连续成交量放大的天数要求
            stop_loss_pct: 止损百分比
            vol_change_threshold: 量能变化率阈值
            name: 策略名称
        """
        super().__init__(name)
        self.kdj_buy_threshold = kdj_buy_threshold
        self.kdj_sell_threshold = kdj_sell_threshold
        self.vol_breakout_threshold = vol_breakout_threshold
        self.consecutive_vol_days = consecutive_vol_days
        self.stop_loss_pct = stop_loss_pct
        self.vol_change_threshold = vol_change_threshold
        
        # 添加依赖的算子
        self.kdj_operator = KDJOperator()
        self.macd_operator = MACDOperator()
        self.rel_volume_operator = RelativeVolumeOperator()
        self.vol_change_operator = VolumeChangeOperator()
        self.vol_breakout_operator = VolumeBreakoutOperator()
        
        self.add_operator(self.kdj_operator)
        self.add_operator(self.macd_operator)
        self.add_operator(self.rel_volume_operator)
        self.add_operator(self.vol_change_operator)
        self.add_operator(self.vol_breakout_operator)
    
    def check_b1_signal(self, df, date=None):
        """
        检查B1入场信号
        B1信号：J值从超卖区向上突破，并且成交量显著放大
        
        Args:
            df: 包含指标的DataFrame
            date: 指定日期，如果为None则使用最新日期
            
        Returns:
            布尔值，表示是否有B1入场信号
        """
        if 'J' not in df.columns:
            df = self.kdj_operator.compute(df)
        
        if 'rel_volume' not in df.columns:
            df = self.rel_volume_operator.compute(df)
            
        if 'vol_breakout' not in df.columns:
            df = self.vol_breakout_operator.compute(df, threshold=self.vol_breakout_threshold)
        
        if date is None:
            # 使用最新日期（第一行，因为数据是降序的）
            date = df['trade_date'].iloc[0]
        
        # 获取当前日期和之前的数据
        date_idx = df[df['trade_date'] == date].index
        if len(date_idx) == 0:
            return False
        
        date_idx = date_idx[0]
        
        # 确保有足够的历史数据
        if date_idx >= len(df) - 3:
            return False
        
        # 获取当前和前一个交易日的J值
        current_j = df.loc[date_idx, 'J']
        prev_j = df.loc[date_idx+1, 'J']
        
        # 检查J值是否从低于阈值向上突破
        j_breakout = prev_j < self.kdj_buy_threshold and current_j >= self.kdj_buy_threshold
        
        # 检查是否出现了金叉信号（K线上穿D线）
        current_k = df.loc[date_idx, 'K']
        current_d = df.loc[date_idx, 'D']
        prev_k = df.loc[date_idx+1, 'K']
        prev_d = df.loc[date_idx+1, 'D']
        
        golden_cross = prev_k < prev_d and current_k >= current_d
        
        # 检查当前成交量是否放大
        volume_increased = df.loc[date_idx, 'rel_volume'] > 1.0 and df.loc[date_idx, 'vol_breakout'] == 1
        
        # B1信号：J值突破超卖区域 且 出现金叉 且 成交量放大
        return j_breakout and golden_cross and volume_increased
    
    def check_continuous_volume_increase(self, df, date=None, days=5):
        """
        检查连续成交量放大
        
        Args:
            df: 包含成交量数据的DataFrame
            date: 指定日期，如果为None则使用最新日期
            days: 连续检查的天数
            
        Returns:
            布尔值，表示是否有连续成交量放大
        """
        if 'vol_change' not in df.columns:
            df = self.vol_change_operator.compute(df)
        
        if date is None:
            # 使用最新日期（第一行，因为数据是降序的）
            date = df['trade_date'].iloc[0]
        
        # 获取当前日期索引
        date_idx = df[df['trade_date'] == date].index
        if len(date_idx) == 0:
            return False
        
        date_idx = date_idx[0]
        
        # 确保有足够的历史数据
        if date_idx >= len(df) - days:
            return False
        
        # 检查是否有连续days天成交量增加
        vol_increasing_days = 0
        for i in range(days):
            if date_idx + i >= len(df):
                break
                
            if df.loc[date_idx + i, 'vol_change'] > self.vol_change_threshold:
                vol_increasing_days += 1
        
        return vol_increasing_days >= self.consecutive_vol_days // 2
    
    def check_exit_signal(self, df, entry_price=None, date=None):
        """
        检查卖出信号
        
        Args:
            df: 包含指标的DataFrame
            entry_price: 买入价格，用于计算止损
            date: 指定日期，如果为None则使用最新日期
            
        Returns:
            布尔值，表示是否有卖出信号
        """
        if 'J' not in df.columns:
            df = self.kdj_operator.compute(df)
        
        if date is None:
            # 使用最新日期（第一行，因为数据是降序的）
            date = df['trade_date'].iloc[0]
        
        # 获取当前日期索引
        date_idx = df[df['trade_date'] == date].index
        if len(date_idx) == 0:
            return False
        
        date_idx = date_idx[0]
        
        # 确保有足够的历史数据
        if date_idx >= len(df) - 3:
            return False
        
        # 获取当前和前一个交易日的J值
        current_j = df.loc[date_idx, 'J']
        prev_j = df.loc[date_idx+1, 'J']
        
        # 确保使用前复权价格
        close_col = 'close_pre_adj' if 'close_pre_adj' in df.columns else 'close'
        
        # 检查J值是否高于卖出阈值
        j_high = current_j > self.kdj_sell_threshold
        
        # 检查是否出现了死叉信号（K线下穿D线）
        current_k = df.loc[date_idx, 'K']
        current_d = df.loc[date_idx, 'D']
        prev_k = df.loc[date_idx+1, 'K']
        prev_d = df.loc[date_idx+1, 'D']
        
        death_cross = prev_k > prev_d and current_k <= current_d
        
        # 检查止损条件
        stop_loss = False
        if entry_price is not None:
            current_price = df.loc[date_idx, close_col]
            stop_loss = current_price < entry_price * (1 - self.stop_loss_pct)
        
        # 检查高位横盘（下跌中继）
        high_consolidation = False
        if date_idx < len(df) - 5:  # 确保有足够的历史数据
            # 计算最近5天的最高价和最低价之间的波动率
            high_prices = df.loc[date_idx:date_idx+4, 'high'].values
            low_prices = df.loc[date_idx:date_idx+4, 'low'].values
            price_range = (max(high_prices) - min(low_prices)) / min(low_prices)
            
            # 如果波动率小于3%并且J值高于70，视为高位横盘
            high_consolidation = price_range < 0.03 and current_j > 70
        
        # 任何一个条件满足即可退出
        return j_high or death_cross or stop_loss or high_consolidation
    
    def get_buy_signals(self, df, date=None):
        """
        获取买入信号
        
        Args:
            df: 包含指标的DataFrame
            date: 指定日期，如果为None则使用最新日期
            
        Returns:
            布尔值，表示是否有买入信号
        """
        if 'J' not in df.columns or 'K' not in df.columns or 'D' not in df.columns:
            df = self.kdj_operator.compute(df)
        
        if 'rel_volume' not in df.columns:
            df = self.rel_volume_operator.compute(df)
        
        if 'vol_breakout' not in df.columns:
            df = self.vol_breakout_operator.compute(df, threshold=self.vol_breakout_threshold)
        
        # 检查B1信号和连续成交量增加
        b1_signal = self.check_b1_signal(df, date)
        volume_increasing = self.check_continuous_volume_increase(df, date)
        
        # 同时满足B1信号和连续成交量增加
        return b1_signal and volume_increasing
    
    def get_sell_signals(self, df, date=None):
        """
        获取卖出信号
        
        Args:
            df: 包含指标的DataFrame
            date: 指定日期，如果为None则使用最新日期
            
        Returns:
            布尔值，表示是否有卖出信号
        """
        return self.check_exit_signal(df, date=date)
    
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
            # 计算各项指标
            df_processed = self.kdj_operator.compute(df)
            df_processed = self.rel_volume_operator.compute(df_processed)
            df_processed = self.vol_change_operator.compute(df_processed)
            df_processed = self.vol_breakout_operator.compute(df_processed, threshold=self.vol_breakout_threshold)
            
            # 获取指定日期或最新日期
            if date is None:
                stock_date = df_processed['trade_date'].iloc[0]
            else:
                stock_date = date
                if stock_date not in df_processed['trade_date'].values:
                    print(f"警告：股票 {stock_code} 在 {stock_date} 没有数据")
                    continue
            
            # 检查买入信号
            if self.get_buy_signals(df_processed, stock_date):
                buy_signals.append(stock_code)
            
            # 检查卖出信号
            if self.get_sell_signals(df_processed, stock_date):
                sell_signals.append(stock_code)
        
        return {
            'buy': buy_signals,
            'sell': sell_signals
        } 