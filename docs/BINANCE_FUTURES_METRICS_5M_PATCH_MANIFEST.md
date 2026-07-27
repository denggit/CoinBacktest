# Binance Futures Metrics 5m Patch Manifest

## 目标

新增 Binance USD-M 永续合约 5 分钟 metrics 数据接口，用于 CoinBacktest 中的跨交易所 OI、持仓拥挤度和主动买卖情绪研究。

## 新增文件

```text
src/data_feed/binance_futures_metrics/
    __init__.py
    archive.py
    features.py
    loader.py
    models.py
    store.py
src/data_feed/binance_futures_metrics_loader.py
tools/prebuild_binance_futures_metrics.py
tests/data_feed/test_binance_futures_metrics_loader.py
docs/BINANCE_FUTURES_METRICS_5M_RUNBOOK.md
docs/BINANCE_FUTURES_METRICS_5M_PATCH_MANIFEST.md
```

## 架构边界

- `archive.py`：官方 ZIP 下载、重试、SHA256、CSV 解析、原始文件落盘。
- `store.py`：SQLite schema、逐日事务写入、coverage 和本地读取。
- `features.py`：因果相对 OI、比率标准化和 index 模式。
- `loader.py`：对外 facade、并发编排和策略侧读取。
- 顶层 `binance_futures_metrics_loader.py`：兼容导出，不承载实现。
- 未修改 OKX Loader、策略逻辑、回测执行或 TP/SL。

## 数据产物

```text
data/binance/raw/futures_metrics/ETHUSDT/YYYY/MM/*.zip
data/binance_futures_metrics.db
```

## 验证

```text
专项与相关回归：20 passed
compileall：passed
离线真实 CLI 端到端：288/288 rows
重复命令断点续传：downloaded=0, skipped=1
新增文件 import-boundary 违规：0
```

项目全局 import-boundary 测试仍被仓库原有的 research→research 违规阻塞，本次没有新增违规。
