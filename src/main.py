"""
股票池筛选框架主程序
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

from .config_manager import ConfigManager
from .data_fetcher import DataFetcher
from .backtest import Backtest
from .strategies import KDJStrategy, LSTMStrategy, ZStrategy


def update_log_file(equal_weight_result, market_cap_result, latest_date, strategy_name, save_path):
    """
    更新开发日志文件，记录回测结果
    
    Args:
        equal_weight_result: 等权重回测结果
        market_cap_result: 市值加权回测结果
        latest_date: 最新回测日期
        strategy_name: 策略名称
        save_path: 保存图表的路径
    """
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
- 回测策略：{strategy_name}策略

#### 等权重策略结果
```
{str(equal_weight_result.summary()).replace('{', '').replace('}', '').replace("'", "")}
```

#### 市值加权策略结果
```
{str(market_cap_result.summary()).replace('{', '').replace('}', '').replace("'", "")}
```

*回测结果图表已保存为 {save_path}*
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
    for stock_code in stock_codes:
        try:
            df = data_fetcher.load_stock_data(stock_code)
            stock_data[stock_code] = df
            print(f"已加载股票 {stock_code} 的数据，共 {len(df)} 条记录")
        except Exception as e:
            print(f"加载股票 {stock_code} 数据时出错: {e}")
    
    if not stock_data:
        print("错误：未能加载任何股票数据")
        return
    
    # 根据配置使用指定策略
    strategy_config = config_manager.get_strategy_config()
    strategy_name = strategy_config.get('name', 'KDJ').upper()
    strategy_desc = strategy_config.get('description', '')
    
    print(f"\n使用策略: {strategy_name} - {strategy_desc}")
    
    # 创建策略实例
    if strategy_name == 'KDJ':
        # 创建KDJ策略
        kdj_params = config_manager.get_kdj_params()
        strategy = KDJStrategy(
            buy_threshold=kdj_params.get('buy_threshold', 0),
            sell_threshold=kdj_params.get('sell_threshold', 70),
            n=kdj_params.get('n', 9),
            m1=kdj_params.get('m1', 3),
            m2=kdj_params.get('m2', 3)
        )
    elif strategy_name == 'LSTM':
        # 创建LSTM策略
        lstm_params = config_manager.get_lstm_params()
        model_dir = lstm_params.get('model_dir', 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'lstm_model.pkl')
        
        strategy = LSTMStrategy(
            sequence_length=lstm_params.get('sequence_length', 10),
            model_path=model_path,
            train_before_predict=lstm_params.get('train_before_predict', True)
        )
    elif strategy_name == 'Z':
        # 创建Z策略
        z_params = config_manager.get_z_strategy_params()
        strategy = ZStrategy(
            kdj_buy_threshold=z_params.get('kdj_buy_threshold', 20),
            kdj_sell_threshold=z_params.get('kdj_sell_threshold', 80),
            vol_breakout_threshold=z_params.get('vol_breakout_threshold', 1.8),
            consecutive_vol_days=z_params.get('consecutive_vol_days', 5),
            stop_loss_pct=z_params.get('stop_loss_pct', 0.03),
            vol_change_threshold=z_params.get('vol_change_threshold', 0.3)
        )
    else:
        print(f"未知策略: {strategy_name}，将使用默认的KDJ策略")
        strategy = KDJStrategy()
    
    # 获取最新日期的选股结果
    print(f"正在使用{strategy.name}策略进行选股...")
    latest_date = max(df['trade_date'].max() for df in stock_data.values())
    signals = strategy.generate_signals(stock_data, latest_date)
    
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
    equal_weight_result = backtest.run(strategy, weighting_method='equal')
    print("等权重回测结果:")
    for key, value in equal_weight_result.summary().items():
        print(f"  {key}: {value}")
    
    # 运行市值加权回测
    print("正在进行市值加权回测...")
    market_cap_result = backtest.run(strategy, weighting_method='market_cap')
    print("市值加权回测结果:")
    for key, value in market_cap_result.summary().items():
        print(f"  {key}: {value}")
    
    # 绘制回测结果图表
    plt.figure(figsize=(12, 8))
    
    # 绘制等权重和市值加权的净值曲线
    equity_curve_equal = (1 + equal_weight_result.cumulative_returns) * equal_weight_result.initial_capital
    equity_curve_market = (1 + market_cap_result.cumulative_returns) * market_cap_result.initial_capital
    
    plt.plot(equity_curve_equal.index, equity_curve_equal, label='Equal Weight')
    plt.plot(equity_curve_market.index, equity_curve_market, label='Market Cap Weighted')
    
    plt.title(f'{strategy.name} Strategy Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    
    # 确保pictures文件夹存在
    os.makedirs('pictures', exist_ok=True)
    
    # 使用新的命名格式: <date>_<strategy_name>_<backtest_start_date>_<otherinfo>.png
    date_str = latest_date.strftime('%Y%m%d')
    
    # 获取回测开始日期
    backtest_start_date = start_date.strftime('%Y%m%d')
    
    # 构建图片名称，添加额外信息（如收益率）
    other_info = f"EW{equal_weight_result.summary().get('总收益率', 0)}_MW{market_cap_result.summary().get('总收益率', 0)}"
    
    save_path = f'pictures/{date_str}_{strategy.name}_{backtest_start_date}_{other_info}.png'
    plt.savefig(save_path)
    plt.close()
    
    print(f"回测结果图表已保存到 {save_path}")
    
    # 更新日志文件
    update_log_file(equal_weight_result, market_cap_result, latest_date, strategy.name, save_path)


if __name__ == "__main__":
    main() 