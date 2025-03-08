"""
模型基类 - 所有机器学习模型的抽象基类
"""
import os
import pickle
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """模型基类，提供统一的接口"""
    
    def __init__(self, name=None):
        """
        初始化模型
        
        Args:
            name: 模型名称
        """
        self.name = name or self.__class__.__name__
        self.model = None  # 具体模型实例
        self.is_trained = False  # 模型是否已训练
    
    @abstractmethod
    def preprocess(self, data):
        """
        数据预处理
        
        Args:
            data: 原始数据
            
        Returns:
            处理后的数据
        """
        pass
    
    @abstractmethod
    def train(self, X, y, **kwargs):
        """
        训练模型
        
        Args:
            X: 特征数据
            y: 标签数据
            **kwargs: 训练参数
            
        Returns:
            训练结果
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        模型预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        pass
    
    def save(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
            
        Returns:
            是否保存成功
        """
        if self.model is None:
            print(f"错误：模型 {self.name} 未初始化，无法保存")
            return False
        
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存模型
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            
            print(f"模型已保存到 {path}")
            return True
        except Exception as e:
            print(f"保存模型时出错: {e}")
            return False
    
    def load(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            是否加载成功
        """
        if not os.path.exists(path):
            print(f"错误：模型文件 {path} 不存在")
            return False
        
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.is_trained = True
            print(f"已加载模型 {path}")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False
    
    def __str__(self):
        return f"{self.name} Model"
    
    def __repr__(self):
        return self.__str__() 