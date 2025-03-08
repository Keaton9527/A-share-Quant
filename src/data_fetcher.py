"""
数据获取模块 - 使用Tushare API拉取股票数据
"""
import os
import time
import pandas as pd
import tushare as ts
from datetime import datetime


class DataFetcher:
    """数据获取类，负责从Tushare获取股票数据并保存到本地"""
    
    def __init__(self, api_token=None, data_dir='dataset/stocktrading'):
        """
        初始化数据获取器
        
        Args:
            api_token: Tushare的API Token
            data_dir: 数据保存目录
        """
        self.api_token = api_token
        self.data_dir = data_dir
        self.pro = self._init_tushare()
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _init_tushare(self):
        """初始化Tushare API"""
        if self.api_token:
            ts.set_token(self.api_token)
        return ts.pro_api()
    
    def set_token(self, token):
        """设置Tushare API Token"""
        self.api_token = token
        ts.set_token(token)
        self.pro = ts.pro_api()
    
    def update_stock_data(self, stock_codes, start_date='20130101', end_date=None):
        """
        更新股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期，格式：YYYYMMDD
            end_date: 结束日期，格式：YYYYMMDD，默认为当前日期
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 记录API调用次数
        api_calls = 0
        
        for stock_code in stock_codes:
            file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
            
            # 确定需要请求的日期范围
            if os.path.exists(file_path):
                # 已有文件，只更新新数据
                existing_data = pd.read_csv(file_path)
                if not existing_data.empty:
                    # 假设数据按日期从新到旧排序
                    last_date = existing_data['trade_date'].iloc[0]
                    request_start_date = (pd.to_datetime(last_date) + 
                                         pd.Timedelta(days=1)).strftime('%Y%m%d')
                    
                    # 如果最新数据已经是最新日期，则跳过
                    if pd.to_datetime(request_start_date) > pd.to_datetime(end_date):
                        print(f"股票 {stock_code} 数据已是最新，无需更新")
                        continue
                else:
                    request_start_date = start_date
            else:
                request_start_date = start_date
            
            print(f"正在获取股票 {stock_code} 的数据，日期范围：{request_start_date} 至 {end_date}")
            
            try:
                # 获取日线行情数据
                df = self.pro.daily(ts_code=stock_code, 
                                   start_date=request_start_date, 
                                   end_date=end_date)
                
                # 获取复权因子
                adj_factor = self.pro.adj_factor(ts_code=stock_code, 
                                               start_date=request_start_date, 
                                               end_date=end_date)
                
                # API调用次数增加
                api_calls += 2
                
                # 如果超出API限制，等待一段时间
                if api_calls >= 490:  # 接近限制，预留一些余量
                    print("接近API调用限制，暂停60秒...")
                    time.sleep(60)
                    api_calls = 0
                
                # 数据合并
                if not df.empty and not adj_factor.empty:
                    # 合并复权因子
                    df = pd.merge(df, adj_factor, on=['ts_code', 'trade_date'], how='left')
                    
                    # 按日期从新到旧排序
                    df = df.sort_values('trade_date', ascending=False)
                    
                    # 保存或追加数据
                    if os.path.exists(file_path):
                        existing_data = pd.read_csv(file_path)
                        # 删除可能重复的日期
                        existing_data = existing_data[~existing_data['trade_date'].isin(df['trade_date'])]
                        # 合并并排序
                        combined_data = pd.concat([df, existing_data])
                        combined_data = combined_data.sort_values('trade_date', ascending=False)
                        combined_data.to_csv(file_path, index=False)
                        print(f"股票 {stock_code} 数据已更新，新增 {len(df)} 条记录")
                    else:
                        df.to_csv(file_path, index=False)
                        print(f"股票 {stock_code} 数据已保存，共 {len(df)} 条记录")
                else:
                    print(f"股票 {stock_code} 在指定日期范围内无数据")
                
                # 避免频繁调用API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"获取股票 {stock_code} 数据时出错: {e}")
    
    def load_stock_data(self, stock_code):
        """
        加载股票数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            pandas.DataFrame: 股票数据
        """
        file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"股票 {stock_code} 的数据文件不存在")
        
        df = pd.read_csv(file_path)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        
        # 计算前复权价格
        if 'adj_factor' in df.columns:
            latest_factor = df['adj_factor'].iloc[0]
            df['close_pre_adj'] = df['close'] * df['adj_factor'] / latest_factor
            df['open_pre_adj'] = df['open'] * df['adj_factor'] / latest_factor
            df['high_pre_adj'] = df['high'] * df['adj_factor'] / latest_factor
            df['low_pre_adj'] = df['low'] * df['adj_factor'] / latest_factor
            
            # 计算后复权价格
            df['close_post_adj'] = df['close'] * df['adj_factor']
            df['open_post_adj'] = df['open'] * df['adj_factor']
            df['high_post_adj'] = df['high'] * df['adj_factor']
            df['low_post_adj'] = df['low'] * df['adj_factor']
        
        return df


# 示例使用
if __name__ == "__main__":
    # 注意：需要替换为您自己的Tushare API Token
    API_TOKEN = "eaf33a386c2131d066dce3c7b777d519fdd054d59dd2a5f038d4623b"
    
    fetcher = DataFetcher(API_TOKEN)
    # 更新几只股票的数据作为示例
    fetcher.update_stock_data(['000001.SZ', '002594.SZ'], start_date='20130101')
    
    # 加载数据示例
    data = fetcher.load_stock_data('000001.SZ')
    print(data.head()) 