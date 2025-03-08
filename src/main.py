"""
股票池筛选框架主程序
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

from config_manager import ConfigManager
from data_fetcher import DataFetcher
from stock_selector import KDJStockSelector
from backtest import Backtest


def update_log_file(equal_weight_result, market_cap_result, latest_date):
    """更新开发日志文件，记录回测结果"""
    log_file = 'development_log.md'
    
    if not os.path.exists(log_file):
        print(f"警告：找不到日志文件 {log_file}")
        return
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找最新版本的运行结果部分
        run_result_marker = "### 运行结果"
        result_placeholder = "*待回测完成后补充*"
        
        if run_result_marker in content and result_placeholder in content:
            # 获取当前日期
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 格式化回测结果
            result_text = f"""*{current_date} 回测结果*

#### 回测信息
- 回测日期：{latest_date.strftime('%Y-%m-%d')}
- 回测策略：KDJ策略（J值低于0买入，J值高于70卖出）

#### 等权重策略结果
```
{str(equal_weight_result.summary()).replace('{', '').replace('}', '').replace("'", "")}
```

#### 市值加权策略结果
```
{str(market_cap_result.summary()).replace('{', '').replace('}', '').replace("'", "")}
```

*回测结果图表已保存为 backtest_result.png*
"""
            # 替换占位符
            updated_content = content.replace(result_placeholder, result_text)
            
            # 写回文件
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
                
            print(f"已更新回测结果到日志文件 {log_file}")
        else:
            print(f"警告：未找到可更新的结果占位符")
    except Exception as e:
        print(f"更新日志文件时出错: {e}")


def main():
    """主程序入口"""
    # 加载配置
    config_manager = ConfigManager('config.yaml')
    
    # 获取股票代码列表
    stock_codes = config_manager.get_stock_codes()
    if not stock_codes:
        print("错误：股票池为空，请在配置文件中添加股票代码")
        return
    
    print(f"股票池: {stock_codes}")
    
    # 设置Tushare API Token
    api_token = config_manager.get_tushare_token()
    if not api_token:
        api_token = input("请输入您的Tushare API Token: ")
    
    data_fetcher = DataFetcher(api_token)
    
    # 检查是否需要更新数据
    if config_manager.should_update_data():
        print("正在更新股票数据...")
        date_range = config_manager.get_data_date_range()
        data_fetcher.update_stock_data(
            stock_codes, 
            start_date=date_range['start_date'],
            end_date=date_range['end_date']
        )
    
    # 加载股票数据
    print("正在加载股票数据...")
    stock_data = {}
    for stock_code in tqdm(stock_codes, desc="加载股票数据"):
        try:
            df = data_fetcher.load_stock_data(stock_code)
            stock_data[stock_code] = df
            print(f"已加载股票 {stock_code} 的数据，共 {len(df)} 条记录")
        except Exception as e:
            print(f"加载股票 {stock_code} 数据时出错: {e}")
    
    if not stock_data:
        print("错误：未能加载任何股票数据")
        return
    
    # 创建KDJ选股器
    kdj_params = config_manager.get_kdj_params()
    selector = KDJStockSelector(
        n=kdj_params.get('n', 9),
        m1=kdj_params.get('m1', 3),
        m2=kdj_params.get('m2', 3),
        buy_threshold=kdj_params.get('buy_threshold', 0),
        sell_threshold=kdj_params.get('sell_threshold', 70)
    )
    
    # 获取最新日期的选股结果
    print("正在进行KDJ选股...")
    latest_date = max(df['trade_date'].max() for df in stock_data.values())
    signals = selector.screen_stocks(stock_data, latest_date)
    
    print(f"选股日期: {latest_date.strftime('%Y-%m-%d')}")
    print(f"买入信号: {signals['buy']}")
    print(f"卖出信号: {signals['sell']}")
    
    # 获取回测配置
    backtest_config = config_manager.get_backtest_config()
    
    # 设置回测日期范围
    end_date = latest_date
    if backtest_config.get('start_date'):
        start_date = pd.to_datetime(backtest_config['start_date'])
    else:
        # 默认回测一年
        start_date = end_date - timedelta(days=365)
    
    print(f"回测时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    
    # 创建回测器
    backtest = Backtest(
        stock_data=stock_data,
        start_date=start_date,
        end_date=end_date,
        initial_capital=backtest_config.get('start_capital', 100000),
        commission_rate=backtest_config.get('commission_rate', 0.001),
        slippage=backtest_config.get('slippage', 0.0005)
    )
    
    # 运行等权重回测
    print("正在进行等权重回测...")
    equal_weight_result = backtest.run(selector, weighting_method='equal')
    print("等权重回测结果:")
    for key, value in equal_weight_result.summary().items():
        print(f"  {key}: {value}")
    
    # 运行市值加权回测
    print("正在进行市值加权回测...")
    market_cap_result = backtest.run(selector, weighting_method='market_cap')
    print("市值加权回测结果:")
    for key, value in market_cap_result.summary().items():
        print(f"  {key}: {value}")
    
    # 绘制回测结果图表
    plt.figure(figsize=(12, 8))
    
    # 绘制等权重和市值加权的净值曲线
    equity_curve_equal = (1 + equal_weight_result.cumulative_returns) * equal_weight_result.initial_capital
    equity_curve_market = (1 + market_cap_result.cumulative_returns) * market_cap_result.initial_capital
    
    plt.plot(equity_curve_equal.index, equity_curve_equal, label='等权重')
    plt.plot(equity_curve_market.index, equity_curve_market, label='市值加权')
    
    plt.title('KDJ策略回测结果')
    plt.xlabel('日期')
    plt.ylabel('净值')
    plt.grid(True)
    plt.legend()
    
    # 保存图表
    plt.savefig('backtest_result.png')
    plt.close()
    
    print("回测结果图表已保存到 backtest_result.png")
    
    # 更新日志文件
    update_log_file(equal_weight_result, market_cap_result, latest_date)


if __name__ == "__main__":
    main() 