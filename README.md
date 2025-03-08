# 股票池筛选框架

基于KDJ指标的股票池筛选和回测框架。

## 功能特点

- **配置管理**：使用YAML文件存储和管理配置信息
- **数据获取**：自动从Tushare API获取股票数据并保存到本地
- **选股策略**：基于KDJ指标J值的选股策略
- **交易逻辑**：J值低于0买入，首次超过70卖出
- **回测功能**：支持等权重和市值加权两种回测方式
- **性能评估**：包括收益率、最大回撤等指标

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 首先配置`config.yaml`文件，设置股票池和其他参数，包括API Token

```yaml
# API配置
tushare_token: "您的Tushare API Token"  # 请替换为您的Tushare API Token

# 策略选择
strategy:
  name: "KDJ"  # 可选策略: "KDJ" 或 "LSTM"
  description: "基于KDJ指标的超买超卖策略"

# 股票代码列表
stock_code: 
  - '002594.SZ'  # 比亚迪
  - '000001.SZ'  # 平安银行
  # 可添加更多股票...

# 数据配置
update_data: True  # 是否更新本地数据库中的数据
```

2. 运行主程序

从项目根目录运行：
```bash
python run.py
```

或者直接运行主模块：
```bash
python -m src.main
```

3. 程序会自动使用配置文件中的API Token和策略设置

## 配置说明

- **tushare_token**: Tushare API令牌，用于获取股票数据
- **stock_code**: 股票池代码列表
- **update_data**: 是否更新本地数据库中的数据
- **start_date**: 数据开始日期，默认为'20130101'
- **end_date**: 数据结束日期，默认为当前日期
- **backtest**: 回测配置
  - **start_capital**: 初始资金
  - **commission_rate**: 交易成本
  - **slippage**: 滑点
- **kdj**: KDJ指标参数
  - **n**: KDJ的N参数，默认为9
  - **m1**: KDJ的M1参数，默认为3
  - **m2**: KDJ的M2参数，默认为3
  - **buy_threshold**: 买入阈值，默认为0
  - **sell_threshold**: 卖出阈值，默认为70

## 项目结构

```
├── run.py                     # 主运行脚本
├── config.yaml                # 配置文件
├── requirements.txt           # 依赖列表
├── dataset/
│   └── stocktrading/         # 股票数据存储文件夹
├── pictures/                  # 回测结果图表保存文件夹
├── src/                       # 源代码包
│   ├── __init__.py           # 包初始化文件
│   ├── main.py               # 主程序入口
│   ├── config_manager.py     # 配置管理模块
│   ├── data_fetcher.py       # 数据获取模块
│   ├── stock_selector.py     # 选股策略模块（旧实现）
│   ├── backtest.py           # 回测模块
│   ├── operators/            # 算子文件夹
│   │   ├── __init__.py
│   │   ├── base_operator.py  # 算子基类
│   │   └── tech_operators.py # 技术指标算子
│   ├── strategies/           # 策略文件夹
│   │   ├── __init__.py
│   │   ├── base_strategy.py  # 策略基类
│   │   ├── kdj_strategy.py   # KDJ策略
│   │   └── lstm_strategy.py  # LSTM预测策略
│   └── models/               # 模型文件夹
│       ├── __init__.py
│       ├── base_model.py     # 模型基类
│       └── lstm_model.py     # LSTM模型实现
├── development_log.md        # 开发日志
└── README.md                 # 说明文档
```

## 开发日志

项目包含一个开发日志文件（development_log.md），用于记录项目的修改历史和回测结果。

### 日志功能

1. **自动记录回测结果**：每次成功运行回测后，程序会自动将回测结果记录到日志文件中
2. **版本管理**：使用不同级别的标题区分不同版本和不同的修改内容
3. **直接对话更新**：通过与AI助手对话，直接更新日志文件中的版本和修改记录

## 注意事项

- 首次运行时需要提供Tushare API Token
- 数据拉取可能受API调用次数限制，请合理配置股票池大小
- 回测结果仅供参考，实际交易还需结合更多因素考虑 