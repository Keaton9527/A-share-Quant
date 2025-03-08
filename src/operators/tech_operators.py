"""
技术指标算子 - 提供各种技术指标的计算
"""
import pandas as pd
import numpy as np
from .base_operator import BaseOperator


class TechOperator(BaseOperator):
    """技术指标算子基类"""
    
    def __init__(self, name=None):
        """
        初始化技术指标算子
        
        Args:
            name: 算子名称
        """
        super().__init__(name)
    
    def compute(self, df, **kwargs):
        """
        计算技术指标
        
        Args:
            df: 包含OHLC数据的DataFrame
            **kwargs: 指标参数
        
        Returns:
            添加了指标值的DataFrame
        """
        # 在子类中实现具体的计算逻辑
        raise NotImplementedError("必须在子类中实现compute方法")


class KDJOperator(TechOperator):
    """KDJ指标算子"""
    
    def __init__(self, name="KDJ"):
        """
        初始化KDJ指标算子
        
        Args:
            name: 算子名称
        """
        super().__init__(name)
    
    def compute(self, df, n=9, m1=3, m2=3):
        """
        计算KDJ指标
        
        Args:
            df: 包含OHLC数据的DataFrame
            n: KDJ指标的窗口大小，默认为9
            m1: K值的平滑因子，默认为3
            m2: D值的平滑因子，默认为3
        
        Returns:
            添加了KDJ指标的DataFrame
        """
        # 确保使用前复权价格计算指标
        high_col = 'high_pre_adj' if 'high_pre_adj' in df.columns else 'high'
        low_col = 'low_pre_adj' if 'low_pre_adj' in df.columns else 'low'
        close_col = 'close_pre_adj' if 'close_pre_adj' in df.columns else 'close'
        
        df_result = df.copy()
        
        # 注意：数据是按日期降序排列的，最新的数据在前面
        # 为了正确计算，我们需要先按日期升序排列
        df_sorted = df_result.sort_values('trade_date', ascending=True)
        
        # 计算N日内的最高价和最低价
        df_sorted['HHV'] = df_sorted[high_col].rolling(n).max()
        df_sorted['LLV'] = df_sorted[low_col].rolling(n).min()
        
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


class MACDOperator(TechOperator):
    """MACD指标算子"""
    
    def __init__(self, name="MACD"):
        """
        初始化MACD指标算子
        
        Args:
            name: 算子名称
        """
        super().__init__(name)
    
    def compute(self, df, fast_period=12, slow_period=26, signal_period=9):
        """
        计算MACD指标
        
        Args:
            df: 包含OHLC数据的DataFrame
            fast_period: 快线周期，默认为12
            slow_period: 慢线周期，默认为26
            signal_period: 信号线周期，默认为9
        
        Returns:
            添加了MACD指标的DataFrame
        """
        # 确保使用前复权价格计算指标
        close_col = 'close_pre_adj' if 'close_pre_adj' in df.columns else 'close'
        
        df_result = df.copy()
        
        # 注意：数据是按日期降序排列的，最新的数据在前面
        # 为了正确计算，我们需要先按日期升序排列
        df_sorted = df_result.sort_values('trade_date', ascending=True)
        
        # 计算快线和慢线的EMA
        df_sorted['EMA_fast'] = df_sorted[close_col].ewm(span=fast_period, adjust=False).mean()
        df_sorted['EMA_slow'] = df_sorted[close_col].ewm(span=slow_period, adjust=False).mean()
        
        # 计算DIF（快线 - 慢线）
        df_sorted['DIF'] = df_sorted['EMA_fast'] - df_sorted['EMA_slow']
        
        # 计算DEA（DIF的信号线）
        df_sorted['DEA'] = df_sorted['DIF'].ewm(span=signal_period, adjust=False).mean()
        
        # 计算MACD柱状图（DIF - DEA）* 2
        df_sorted['MACD'] = (df_sorted['DIF'] - df_sorted['DEA']) * 2
        
        # 清理临时列
        df_sorted = df_sorted.drop(['EMA_fast', 'EMA_slow'], axis=1)
        
        # 恢复原来的日期排序（降序）
        df_result = df_sorted.sort_values('trade_date', ascending=False)
        
        return df_result 