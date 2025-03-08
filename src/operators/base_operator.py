"""
算子基类 - 所有因子计算器的抽象基类
"""
import pandas as pd
from abc import ABC, abstractmethod


class BaseOperator(ABC):
    """算子基类，提供统一的接口"""
    
    def __init__(self, name=None):
        """
        初始化算子
        
        Args:
            name: 算子名称
        """
        self.name = name or self.__class__.__name__
        self._cache = {}  # 缓存，用于存储计算结果，避免重复计算
    
    @abstractmethod
    def compute(self, df, **kwargs):
        """
        计算因子值
        
        Args:
            df: 包含股票数据的DataFrame
            **kwargs: 其他计算参数
        
        Returns:
            包含因子值的DataFrame
        """
        pass
    
    def _get_cache_key(self, df, **kwargs):
        """
        生成缓存键
        
        Args:
            df: 输入数据
            **kwargs: 计算参数
        
        Returns:
            缓存键
        """
        # 使用DataFrame的id和参数的哈希值作为缓存键
        df_id = id(df)
        params_hash = hash(frozenset(kwargs.items()))
        return f"{df_id}_{params_hash}"
    
    def __call__(self, df, use_cache=True, **kwargs):
        """
        调用算子计算因子
        
        Args:
            df: 输入数据
            use_cache: 是否使用缓存
            **kwargs: 计算参数
        
        Returns:
            计算结果
        """
        if use_cache:
            cache_key = self._get_cache_key(df, **kwargs)
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            result = self.compute(df, **kwargs)
            self._cache[cache_key] = result
            return result
        
        return self.compute(df, **kwargs)
    
    def clear_cache(self):
        """清除缓存"""
        self._cache = {}
    
    def __str__(self):
        return f"{self.name} Operator"
    
    def __repr__(self):
        return self.__str__() 