# ETH AI Trading R03：中线 Swing 多周期监督学习

## 一行运行命令

```bat
python research\eth_ai_trading\03_swing_multiframe_supervised_baseline.py
```

## 依赖

R03默认包含LightGBM模型，程序会在读取数据和构建缓存之前立即检查依赖。缺少时先执行：

```bat
python -m pip install lightgbm
```

## 数据边界

- 只使用 `src.data_feed.OKXTradeBarLoader(timeframe="1m")`。
- `build_missing=False`，不会下载或重新构建数据。
- 不读取Raw Trades、ZIP或SQLite实现细节。
- 1m数据按年读取；每年只增加180天warmup和最多120小时未来标签窗口。
- 缓存使用NumPy memory-map文件，避免反复读取与一次性长期驻留多年DataFrame。

## 模型与目标

默认架构：

```text
high_logistic
high_lightgbm
full_lightgbm
hierarchical_lightgbm
```

目标：

```text
72小时内3%潜在涨跌幅，MAE≤1.25%
120小时内5%潜在涨跌幅，MAE≤1.75%
```

## 退出

- 4H结构止损。
- 1H/4H趋势失效。
- 反向模型候选。
- 盈亏平衡和波动率追踪。
- 最长120小时只作安全上限。

## 缓存

```text
data\cache\eth_ai_trading\r03_swing
```

正常重跑会复用缓存。只有特征、标签或因果Schema变化时才自动失效。不要随意使用：

```bat
--force-rebuild-cache
```

## 报告

```text
data\reports\research\eth_ai_trading\03_swing_baseline
```

关键文件：

- `03_label_balance.csv`
- `04_prediction_metrics.csv`
- `05_trade_stress_matrix.csv`
- `06_trades.csv`
- `07_feature_importance.csv`
- `08_champion.json`
- `99_decision.md`

## 可选诊断运行

只跑线性高周期模型：

```bat
python research\eth_ai_trading\03_swing_multiframe_supervised_baseline.py --architectures high_logistic
```

只跑LightGBM主候选：

```bat
python research\eth_ai_trading\03_swing_multiframe_supervised_baseline.py --architectures high_lightgbm,full_lightgbm,hierarchical_lightgbm
```

## R03结果后的纠偏

R03完整报告显示多数交易被15分钟趋势失效快速退出，中位持仓约15分钟，因此该版本不能代表3%–5% Swing开仓能力。后续请运行R03.1；R03保留为诊断基线，不再通过调整原退出参数继续救结果。

R03.1说明：`docs/ETH_AI_TRADING_R03_1_SWING_ENTRY_MVP_RUNBOOK.md`。
