# Binance USD-M Futures Metrics 5m 数据接口

## 目的

为 CoinBacktest 提供 2022 年以来的 Binance `ETHUSDT` 永续合约 5 分钟持仓与情绪指标，作为 OKX 策略研究的跨交易所仓位环境，不冒充 OKX 本地 OI。

数据来自 Binance 官方公开归档：

```text
https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}/
```

每日文件包含：

```text
sum_open_interest
sum_open_interest_value
count_toptrader_long_short_ratio
sum_toptrader_long_short_ratio
count_long_short_ratio
sum_taker_long_short_vol_ratio
```

## 下载

Windows 一行命令：

```bat
python tools\prebuild_binance_futures_metrics.py --symbol ETHUSDT --start-date 2022-01-01 --end-date 2026-06-30 --workers 6
```

日期表示 Binance 官方 UTC 归档日。默认行为：

- 下载并校验官方 SHA256；
- 原始 ZIP 保存到 `data/binance/raw/futures_metrics/ETHUSDT/YYYY/MM/`；
- 规范化数据写入 `data/binance_futures_metrics.db`；
- 已完成日期自动跳过；
- 下载中断后重复相同命令即可续传；
- 网络下载并发，SQLite 单线程落盘。

先检查一天：

```bat
python tools\prebuild_binance_futures_metrics.py --symbol ETHUSDT --start-date 2022-01-01 --end-date 2022-01-01 --inspect-day 2022-01-01
```

## Python 读取

```python
from src.data_feed.binance_futures_metrics_loader import BinanceFuturesMetricsLoader

loader = BinanceFuturesMetricsLoader("ETHUSDT")
raw = loader.load_metrics(
    "2025-10-01 08:00:00",
    "2025-10-02 08:00:00",
    publication_lag="1min",
    index_mode="available_time",
)

features = loader.load_relative_features(
    "2025-10-01 08:00:00",
    "2025-10-02 08:00:00",
    windows=("5m", "15m", "30m", "1h", "4h", "1d"),
    publication_lag="1min",
)
```

相对 OI 特征是小数收益，例如 `0.01` 表示 OI 增加 1%。基线使用目标时间之前最后一个可见样本，并对数据缺口设置 5 分钟容忍度，禁止用过旧数据伪造短周期变化。

## 因果规则

Binance `create_time` 代表 5 分钟指标周期结束时间。默认：

```text
available_time = create_time + 1 minute
```

研究对齐时必须使用 `available_time`，不能按未来行回填。可额外测试 0、1、5 分钟发布延迟压力，但不能根据 Holdout 选择最有利延迟。

## 数据语义

该数据是 Binance `ETHUSDT` 永续市场的跨交易所指标：

- `sum_open_interest_value`：Binance OI 的 USDT 价值；
- `oi_usd_change_*`：Binance OI 相对变化；
- `sum_taker_long_short_vol_ratio`：Binance 主动买量/主动卖量比；
- 多空比字段描述 Binance 账户或大户持仓结构。

它可以辅助判断全市场追空、减仓和拥挤度，但不能直接声称是 OKX OI，也不能单独识别某一笔成交是开仓还是平仓。
