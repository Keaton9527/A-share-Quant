"""
选股策略模块 - 实现各种选股策略
"""
import pandas as pd
import numpy as np


class KDJStockSelector:
    """基于KDJ指标的选股策略"""
    
    def __init__(self, n=9, m1=3, m2=3, buy_threshold=0, sell_threshold=70):
        """
        初始化KDJ选股策略
        
        Args:
            n: KDJ指标的窗口大小
            m1: K值的平滑因子
            m2: D值的平滑因子
            buy_threshold: 买入阈值，J值低于此值视为买入信号
            sell_threshold: 卖出阈值，J值高于此值视为卖出信号
        """
        self.n = n
        self.m1 = m1
        self.m2 = m2
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
    
    def calculate_kdj(self, df):
        """
        计算KDJ指标
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            添加了KDJ指标的DataFrame
        """
        # 确保使用前复权价格计算指标
        high_col = 'high_pre_adj' if 'high_pre_adj' in df.columns else 'high'
        low_col = 'low_pre_adj' if 'low_pre_adj' in df.columns else 'low'
        close_col = 'close_pre_adj' if 'close_pre_adj' in df.columns else 'close'
        
        df = df.copy()
        
        # 注意：数据是按日期降序排列的，最新的数据在前面
        # 为了正确计算，我们需要先按日期升序排列
        df_sorted = df.sort_values('trade_date', ascending=True)
        
        # 计算N日内的最高价和最低价
        df_sorted['HHV'] = df_sorted[high_col].rolling(self.n).max()
        df_sorted['LLV'] = df_sorted[low_col].rolling(self.n).min()
        
        # 计算RSV
        df_sorted['RSV'] = 100 * (df_sorted[close_col] - df_sorted['LLV']) / (df_sorted['HHV'] - df_sorted['LLV'] + 1e-10)
        
        # 计算K、D、J值（使用加权计算方法）
        # 初始化K和D，设为50
        df_sorted['K'] = 50.0
        df_sorted['D'] = 50.0
        
        # 使用循环计算K和D值，因为每个值都依赖于前一个值
        for i in range(1, len(df_sorted)):
            if pd.notna(df_sorted['RSV'].iloc[i]):
                df_sorted.loc[df_sorted.index[i], 'K'] = (2/3) * df_sorted['K'].iloc[i-1] + (1/3) * df_sorted['RSV'].iloc[i]
                df_sorted.loc[df_sorted.index[i], 'D'] = (2/3) * df_sorted['D'].iloc[i-1] + (1/3) * df_sorted['K'].iloc[i]
        
        # 计算J值
        df_sorted['J'] = 3 * df_sorted['K'] - 2 * df_sorted['D']
        
        # 清理临时列
        df_sorted = df_sorted.drop(['HHV', 'LLV', 'RSV'], axis=1)
        
        # 恢复原来的日期排序（降序）
        df_result = df_sorted.sort_values('trade_date', ascending=False)
        
        return df_result
    
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
            df = self.calculate_kdj(df)
        
        if date is None:
            # 使用最新日期
            date = df['trade_date'].iloc[0]
        
        # 获取指定日期的数据
        current_row = df[df['trade_date'] == date]
        if current_row.empty:
            return False
        
        current_j = current_row['J'].iloc[0]
        
        # 获取前一日数据
        prev_date_idx = df[df['trade_date'] < date].index.min()
        if pd.isna(prev_date_idx):
            return current_j <= self.buy_threshold
        
        prev_j = df.loc[prev_date_idx, 'J']
        
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
            df = self.calculate_kdj(df)
        
        if date is None:
            # 使用最新日期
            date = df['trade_date'].iloc[0]
        
        # 获取指定日期的数据
        current_row = df[df['trade_date'] == date]
        if current_row.empty:
            return False
        
        current_j = current_row['J'].iloc[0]
        
        # 获取前一日数据
        prev_date_idx = df[df['trade_date'] < date].index.min()
        if pd.isna(prev_date_idx):
            return current_j >= self.sell_threshold
        
        prev_j = df.loc[prev_date_idx, 'J']
        
        # J值穿越卖出阈值（从下方向上）
        return current_j >= self.sell_threshold and prev_j < self.sell_threshold
    
    def screen_stocks(self, stock_data_dict, date=None):
        """
        筛选股票
        
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
            df_kdj = self.calculate_kdj(df)
            
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


# 示例使用
if __name__ == "__main__":
    # 假设我们已经有了股票数据
    import os
    import pandas as pd
    from data_fetcher import DataFetcher
    
    data_dir = 'dataset/stocktrading'
    stock_codes = ['000001.SZ', '002594.SZ']
    
    # 加载股票数据
    stock_data = {}
    for stock_code in stock_codes:
        file_path = os.path.join(data_dir, f"{stock_code}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            stock_data[stock_code] = df
    
    # 创建选股器
    selector = KDJStockSelector(n=9, m1=3, m2=3, buy_threshold=0, sell_threshold=70)
    
    # 筛选股票
    if stock_data:
        signals = selector.screen_stocks(stock_data)
        print("买入信号股票:", signals['buy'])
        print("卖出信号股票:", signals['sell'])
    else:
        print("没有找到股票数据") 