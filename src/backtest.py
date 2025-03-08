"""
回测模块 - 实现股票交易策略的回测
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from tqdm import tqdm

from stock_selector import KDJStockSelector


class BacktestResult:
    """回测结果类"""
    
    def __init__(self, daily_returns, positions, trades, initial_capital):
        """
        初始化回测结果
        
        Args:
            daily_returns: 每日回报率
            positions: 每日持仓
            trades: 交易记录
            initial_capital: 初始资金
        """
        self.daily_returns = daily_returns
        self.positions = positions
        self.trades = trades
        self.initial_capital = initial_capital
        
        # 计算收益和回撤
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """计算回测指标"""
        # 累积收益率
        self.cumulative_returns = (1 + self.daily_returns).cumprod() - 1
        
        # 总收益率
        self.total_return = self.cumulative_returns.iloc[-1]
        
        # 年化收益率
        days = (self.daily_returns.index[-1] - self.daily_returns.index[0]).days
        self.annual_return = (1 + self.total_return) ** (365 / days) - 1
        
        # 最大回撤
        cumulative_max = self.cumulative_returns.cummax()
        drawdown = (self.cumulative_returns - cumulative_max) / (1 + cumulative_max)
        self.max_drawdown = drawdown.min()
        
        # 夏普比率
        risk_free_rate = 0.03  # 假设无风险利率为3%
        excess_returns = self.daily_returns - risk_free_rate / 365
        self.sharpe_ratio = np.sqrt(365) * excess_returns.mean() / excess_returns.std()
    
    def summary(self):
        """输出回测结果摘要"""
        return {
            "初始资金": self.initial_capital,
            "总收益率": f"{self.total_return:.2%}",
            "年化收益率": f"{self.annual_return:.2%}",
            "最大回撤": f"{self.max_drawdown:.2%}",
            "夏普比率": f"{self.sharpe_ratio:.2f}",
            "交易次数": len(self.trades)
        }
    
    def plot(self, figsize=(12, 8)):
        """绘制回测结果图表"""
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # 绘制净值曲线
        equity_curve = (1 + self.cumulative_returns) * self.initial_capital
        equity_curve.plot(ax=axes[0], title='策略净值曲线')
        axes[0].set_ylabel('净值')
        axes[0].grid(True)
        
        # 绘制回撤曲线
        cumulative_max = self.cumulative_returns.cummax()
        drawdown = (self.cumulative_returns - cumulative_max) / (1 + cumulative_max)
        drawdown.plot(ax=axes[1], title='回撤曲线')
        axes[1].set_ylabel('回撤')
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig


class Backtest:
    """回测类，负责执行回测逻辑"""
    
    def __init__(self, stock_data, start_date, end_date, initial_capital=100000, 
                 commission_rate=0.001, slippage=0.0005):
        """
        初始化回测
        
        Args:
            stock_data: 股票数据字典，键为股票代码，值为DataFrame
            start_date: 回测开始日期
            end_date: 回测结束日期
            initial_capital: 初始资金
            commission_rate: 交易佣金率
            slippage: 滑点
        """
        self.stock_data = stock_data
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # 准备回测数据
        self._prepare_data()
    
    def _prepare_data(self):
        """准备回测数据"""
        # 生成所有交易日
        all_dates = set()
        for df in self.stock_data.values():
            # 过滤日期范围
            mask = (df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)
            all_dates.update(df.loc[mask, 'trade_date'].tolist())
        
        # 按照升序排列交易日期，确保回测按时间顺序进行
        self.trading_dates = sorted(all_dates)
    
    def run(self, selector, weighting_method='equal'):
        """
        执行回测
        
        Args:
            selector: 选股器实例
            weighting_method: 权重分配方法，可选值为'equal'（等权重）或'market_cap'（市值加权）
            
        Returns:
            BacktestResult: 回测结果
        """
        # 初始化回测变量
        positions = {}  # 持仓，{股票代码: 持仓股数}
        cash = self.initial_capital  # 现金
        trades = []  # 交易记录
        daily_total_values = []  # 每日总价值
        
        # 添加进度条
        print(f"开始回测，共 {len(self.trading_dates)} 个交易日...")
        for date in tqdm(self.trading_dates, desc="回测进度"):
            # 当日数据
            current_data = {}
            for code, df in self.stock_data.items():
                curr_row = df[df['trade_date'] == date]
                if not curr_row.empty:
                    current_data[code] = curr_row
            
            # 获取当日市值数据，用于市值加权（如果需要）
            if weighting_method == 'market_cap':
                market_caps = {}
                for code, df_curr in current_data.items():
                    if 'total_mv' in df_curr.columns:
                        market_caps[code] = df_curr['total_mv'].iloc[0]
                    else:
                        # 如果没有市值数据，使用收盘价 × 流通股数作为近似
                        market_caps[code] = df_curr['close'].iloc[0] * 1000000  # 假设每个股票有一百万流通股
            
            # 计算当日持仓价值
            portfolio_value = cash
            for code, shares in positions.items():
                if code in current_data:
                    close_price = current_data[code]['close'].iloc[0]
                    portfolio_value += shares * close_price
            
            # 计算KDJ指标并获取信号
            available_stock_data = {code: df for code, df in self.stock_data.items() 
                                   if code in current_data}
            signals = selector.screen_stocks(available_stock_data, date)
            
            # 执行卖出操作（先卖出后买入）
            sell_codes = []
            for code in list(positions.keys()):
                if code in signals['sell'] and positions[code] > 0:
                    sell_codes.append(code)
            
            # 执行卖出，释放资金
            for code in sell_codes:
                shares = positions[code]
                close_price = current_data[code]['close'].iloc[0]
                
                # 考虑滑点和交易成本
                actual_price = close_price * (1 - self.slippage)
                sell_value = shares * actual_price
                commission = sell_value * self.commission_rate
                
                cash += sell_value - commission
                
                # 记录交易
                trades.append({
                    'date': date,
                    'code': code,
                    'action': 'sell',
                    'shares': shares,
                    'price': actual_price,
                    'value': sell_value,
                    'commission': commission
                })
                
                # 更新持仓
                positions[code] = 0
            
            # 执行买入操作
            buy_codes = signals['buy']
            if buy_codes and cash > 0:
                # 计算每只股票的权重
                if weighting_method == 'equal':
                    # 等权重
                    weights = {code: 1/len(buy_codes) for code in buy_codes}
                elif weighting_method == 'market_cap':
                    # 市值加权
                    total_market_cap = sum(market_caps.get(code, 0) for code in buy_codes)
                    weights = {code: market_caps.get(code, 0)/total_market_cap for code in buy_codes}
                    
                # 计算每只股票的目标价值
                for code in buy_codes:
                    if code in current_data:
                        close_price = current_data[code]['close'].iloc[0]
                        weight = weights.get(code, 0)
                        
                        # 考虑滑点
                        actual_price = close_price * (1 + self.slippage)
                        
                        # 计算可买入股数
                        target_value = portfolio_value * weight
                        max_shares = int(target_value / actual_price)
                        
                        # 检查现金是否足够
                        cost = max_shares * actual_price
                        commission = cost * self.commission_rate
                        total_cost = cost + commission
                        
                        if total_cost <= cash and max_shares > 0:
                            # 更新持仓和现金
                            positions[code] = positions.get(code, 0) + max_shares
                            cash -= total_cost
                            
                            # 记录交易
                            trades.append({
                                'date': date,
                                'code': code,
                                'action': 'buy',
                                'shares': max_shares,
                                'price': actual_price,
                                'value': cost,
                                'commission': commission
                            })
            
            # 记录当日总资产价值
            daily_total_values.append({
                'date': date,
                'cash': cash,
                'positions_value': portfolio_value - cash,
                'total_value': portfolio_value
            })
        
        # 计算每日回报率
        values_df = pd.DataFrame(daily_total_values)
        values_df['date'] = pd.to_datetime(values_df['date'])
        values_df = values_df.set_index('date')
        values_df['daily_return'] = values_df['total_value'].pct_change()
        values_df.loc[values_df.index[0], 'daily_return'] = (values_df.loc[values_df.index[0], 'total_value'] / 
                                                             self.initial_capital - 1)
        
        # 创建回测结果
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df = trades_df.set_index('date')
        
        result = BacktestResult(
            daily_returns=values_df['daily_return'],
            positions=values_df[['cash', 'positions_value', 'total_value']],
            trades=trades_df,
            initial_capital=self.initial_capital
        )
        
        return result


# 示例使用
if __name__ == "__main__":
    # 假设我们已经有了股票数据
    import os
    import pandas as pd
    from datetime import datetime, timedelta
    from data_fetcher import DataFetcher
    from stock_selector import KDJStockSelector
    
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
    
    if stock_data:
        # 设置回测参数
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # 创建选股器
        selector = KDJStockSelector(n=9, m1=3, m2=3, buy_threshold=0, sell_threshold=70)
        
        # 创建回测器
        backtest = Backtest(
            stock_data=stock_data,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,
            commission_rate=0.001,
            slippage=0.0005
        )
        
        # 运行等权重回测
        equal_weight_result = backtest.run(selector, weighting_method='equal')
        print("等权重回测结果:")
        print(equal_weight_result.summary())
        
        # 运行市值加权回测
        market_cap_result = backtest.run(selector, weighting_method='market_cap')
        print("市值加权回测结果:")
        print(market_cap_result.summary())
    else:
        print("没有找到股票数据") 