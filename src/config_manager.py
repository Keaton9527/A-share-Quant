"""
配置管理模块 - 读取和管理YAML配置文件
"""
import os
import yaml
from datetime import datetime, timedelta


class ConfigManager:
    """配置管理类，负责读取和处理配置信息"""
    
    def __init__(self, config_path='config.yaml'):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._process_config()
        
    def _load_config(self):
        """加载YAML配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _process_config(self):
        """处理配置，设置默认值和计算派生值"""
        # 处理日期
        if not self.config.get('end_date'):
            self.config['end_date'] = datetime.now().strftime('%Y%m%d')
            
        # 处理回测日期
        if 'backtest' in self.config:
            # 如果回测开始日期为空，默认为一年前
            if not self.config['backtest'].get('start_date'):
                one_year_ago = datetime.now() - timedelta(days=365)
                self.config['backtest']['start_date'] = one_year_ago.strftime('%Y%m%d')
            
            # 如果回测结束日期为空，默认为当前日期
            if not self.config['backtest'].get('end_date'):
                self.config['backtest']['end_date'] = datetime.now().strftime('%Y%m%d')
    
    def get_stock_codes(self):
        """获取股票代码列表"""
        return self.config.get('stock_code', [])
    
    def should_update_data(self):
        """是否需要更新数据"""
        return self.config.get('update_data', True)
    
    def get_data_date_range(self):
        """获取数据日期范围"""
        return {
            'start_date': self.config.get('start_date'),
            'end_date': self.config.get('end_date')
        }
    
    def get_backtest_config(self):
        """获取回测配置"""
        return self.config.get('backtest', {})
    
    def get_kdj_params(self):
        """获取KDJ参数"""
        return self.config.get('kdj', {
            'n': 9,
            'm1': 3,
            'm2': 3,
            'buy_threshold': 0,
            'sell_threshold': 70
        })
    
    def get_tushare_token(self):
        """获取Tushare API Token"""
        token = self.config.get('tushare_token', '')
        if not token or token == "在此处填写您的Tushare API Token":
            print("警告：Tushare API Token未设置或使用了默认值")
            print("请在config.yaml中设置有效的tushare_token")
        return token
    
    def get_strategy_config(self):
        """
        获取策略配置
        
        Returns:
            dict: 策略配置，包含策略名称和描述
        """
        default_strategy = {
            'name': 'KDJ',
            'description': '基于KDJ指标的超买超卖策略'
        }
        return self.config.get('strategy', default_strategy)
    
    def get_lstm_params(self):
        """
        获取LSTM策略参数
        
        Returns:
            dict: LSTM策略参数
        """
        default_params = {
            'sequence_length': 10,
            'hidden_dim': 50,
            'num_layers': 2,
            'train_before_predict': True,
            'model_dir': 'models'
        }
        return self.config.get('lstm', default_params)


# 示例使用
if __name__ == "__main__":
    config_manager = ConfigManager()
    print("股票代码列表:", config_manager.get_stock_codes())
    print("是否更新数据:", config_manager.should_update_data())
    print("数据日期范围:", config_manager.get_data_date_range())
    print("回测配置:", config_manager.get_backtest_config())
    print("KDJ参数:", config_manager.get_kdj_params()) 