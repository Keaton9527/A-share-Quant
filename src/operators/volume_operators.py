"""
量价关系算子 - 提供成交量与价格关系的计算指标
"""
import pandas as pd
import numpy as np
from .base_operator import BaseOperator


class VolumeOperator(BaseOperator):
    """成交量指标算子基类"""
    
    def __init__(self, name=None):
        """
        初始化成交量指标算子
        
        Args:
            name: 算子名称
        """
        super().__init__(name)
    
    def compute(self, df, **kwargs):
        """
        计算成交量指标
        
        Args:
            df: 包含成交量数据的DataFrame
            **kwargs: 指标参数
        
        Returns:
            添加了指标值的DataFrame
        """
        # 在子类中实现具体的计算逻辑
        raise NotImplementedError("必须在子类中实现compute方法")


class RelativeVolumeOperator(VolumeOperator):
    """相对成交量指标算子，计算当前成交量相对于N日平均成交量的比率"""
    
    def __init__(self, name="RelativeVolume"):
        """
        初始化相对成交量指标算子
        
        Args:
            name: 算子名称
        """
        super().__init__(name)
    
    def compute(self, df, n=20):
        """
        计算相对成交量
        
        Args:
            df: 包含成交量数据的DataFrame
            n: 平均成交量的窗口大小，默认为20
        
        Returns:
            添加了相对成交量指标的DataFrame
        """
        df_result = df.copy()
        
        # 确保数据按日期升序排列
        df_sorted = df_result.sort_values('trade_date', ascending=True)
        
        # 计算N日平均成交量
        df_sorted['volume_ma'] = df_sorted['vol'].rolling(n).mean()
        
        # 计算相对成交量比率 (当前成交量/N日平均成交量)
        df_sorted['rel_volume'] = df_sorted['vol'] / df_sorted['volume_ma']
        
        # 恢复原来的日期排序（降序）
        df_result = df_sorted.sort_values('trade_date', ascending=False)
        
        return df_result


class VolumeChangeOperator(VolumeOperator):
    """成交量变化率算子，计算当前成交量相对于前N日的变化率"""
    
    def __init__(self, name="VolumeChange"):
        """
        初始化成交量变化率算子
        
        Args:
            name: 算子名称
        """
        super().__init__(name)
    
    def compute(self, df, n=5):
        """
        计算成交量变化率
        
        Args:
            df: 包含成交量数据的DataFrame
            n: 变化率的窗口大小，默认为5
        
        Returns:
            添加了成交量变化率指标的DataFrame
        """
        df_result = df.copy()
        
        # 确保数据按日期升序排列
        df_sorted = df_result.sort_values('trade_date', ascending=True)
        
        # 计算N日前的成交量
        df_sorted['vol_prev'] = df_sorted['vol'].shift(n)
        
        # 计算成交量变化率 ((当前成交量-N日前成交量)/N日前成交量)
        df_sorted['vol_change'] = (df_sorted['vol'] - df_sorted['vol_prev']) / df_sorted['vol_prev']
        
        # 移除临时列
        df_sorted = df_sorted.drop(['vol_prev'], axis=1)
        
        # 恢复原来的日期排序（降序）
        df_result = df_sorted.sort_values('trade_date', ascending=False)
        
        return df_result


class VolumePriceCorrelationOperator(VolumeOperator):
    """量价相关性算子，计算成交量与价格变动的相关性"""
    
    def __init__(self, name="VolumePriceCorrelation"):
        """
        初始化量价相关性算子
        
        Args:
            name: 算子名称
        """
        super().__init__(name)
    
    def compute(self, df, n=10):
        """
        计算量价相关性
        
        Args:
            df: 包含价格和成交量数据的DataFrame
            n: 相关性的窗口大小，默认为10
        
        Returns:
            添加了量价相关性指标的DataFrame
        """
        df_result = df.copy()
        
        # 确保使用前复权价格计算指标
        close_col = 'close_pre_adj' if 'close_pre_adj' in df.columns else 'close'
        
        # 确保数据按日期升序排列
        df_sorted = df_result.sort_values('trade_date', ascending=True)
        
        # 计算价格变动率
        df_sorted['price_change'] = df_sorted[close_col].pct_change()
        
        # 计算成交量变动率
        df_sorted['vol_change_1d'] = df_sorted['vol'].pct_change()
        
        # 使用滚动窗口计算相关性
        df_sorted['vol_price_corr'] = df_sorted['price_change'].rolling(n).corr(df_sorted['vol_change_1d'])
        
        # 移除临时列
        df_sorted = df_sorted.drop(['price_change', 'vol_change_1d'], axis=1)
        
        # 恢复原来的日期排序（降序）
        df_result = df_sorted.sort_values('trade_date', ascending=False)
        
        return df_result


class VolumeBreakoutOperator(VolumeOperator):
    """成交量突破算子，识别成交量异常放大的情况"""
    
    def __init__(self, name="VolumeBreakout"):
        """
        初始化成交量突破算子
        
        Args:
            name: 算子名称
        """
        super().__init__(name)
    
    def compute(self, df, n=20, threshold=2.0):
        """
        计算成交量突破
        
        Args:
            df: 包含成交量数据的DataFrame
            n: 平均成交量的窗口大小，默认为20
            threshold: 突破阈值，默认为2.0（即当前成交量是N日平均成交量的2倍以上）
        
        Returns:
            添加了成交量突破指标的DataFrame
        """
        df_result = df.copy()
        
        # 确保数据按日期升序排列
        df_sorted = df_result.sort_values('trade_date', ascending=True)
        
        # 计算N日平均成交量
        df_sorted['volume_ma'] = df_sorted['vol'].rolling(n).mean()
        
        # 标记成交量突破
        df_sorted['vol_breakout'] = (df_sorted['vol'] > df_sorted['volume_ma'] * threshold).astype(int)
        
        # 恢复原来的日期排序（降序）
        df_result = df_sorted.sort_values('trade_date', ascending=False)
        
        return df_result 