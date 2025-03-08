"""
策略基类 - 所有交易策略的抽象基类
"""
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """交易策略基类，提供统一的接口"""
    
    def __init__(self, name=None):
        """
        初始化策略
        
        Args:
            name: 策略名称
        """
        self.name = name or self.__class__.__name__
        self.operators = []  # 策略依赖的算子列表
    
    def add_operator(self, operator):
        """
        添加算子
        
        Args:
            operator: 算子实例
        """
        if operator not in self.operators:
            self.operators.append(operator)
    
    def get_operators(self):
        """
        获取策略依赖的算子列表
        
        Returns:
            算子列表
        """
        return self.operators
    
    @abstractmethod
    def generate_signals(self, stock_data_dict, date=None):
        """
        生成交易信号
        
        Args:
            stock_data_dict: 股票数据字典，键为股票代码，值为DataFrame
            date: 指定日期，如果为None则使用每只股票的最新日期
            
        Returns:
            dict: 包含买入信号和卖出信号的股票代码
        """
        pass
    
    def __str__(self):
        return f"{self.name} Strategy"
    
    def __repr__(self):
        return self.__str__() 