# ETH AI Trading R01：Trades-only 监督学习基线运行说明

## 定位

R01 不是单独的数据审计。它只做一次轻量的公共 Loader 预检，然后立刻建立样本、训练模型和回测完整交易。

数据访问固定使用：

```python
src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader
```

禁止研究代码直接读取 raw trades ZIP 或 SQLite。

## 默认运行

Windows 一行命令：

```bat
python research\eth_ai_trading\01_trades_only_supervised_baseline.py
```

默认：

- 数据：ETH-USDT-SWAP 1s Trade Bar。
- 研究：2023-01-01 至 2026-06-30。
- 模型：Ridge + LightGBM。
- 训练样本上限：每个模型/折叠/周期最多 2,000,000 行；Ridge 最多 750,000 行。
- 分日读取、分月 NumPy 内存映射缓存、可断点续跑。
- 不下载、不自动重建缺失数据。

## 重新构建特征缓存

```bat
python research\eth_ai_trading\01_trades_only_supervised_baseline.py --force-rebuild-cache
```

普通重跑不要加此参数，已有月份缓存会直接复用。

## 降低训练样本用于工程 smoke

```bat
python research\eth_ai_trading\01_trades_only_supervised_baseline.py --max-train-rows 200000
```

该结果只用于确认代码能跑，不可作为最终研究结论。

## 输出

```text
data/reports/research/eth_ai_trading/01_trades_only_baseline/
```

主要文件：

- `00_runtime_and_config.json`
- `01_public_loader_preflight.json`
- `02_sample_cache_manifest.json`
- `03_walk_forward_folds.json`
- `04_prediction_metrics.csv`
- `05_trade_stress_matrix.csv`
- `06_base_scenario_trades.csv.gz`
- `07_validation_champion.json`
- `08_sealed_result.json`
- `09_champion_period_breakdown.csv`
- `99_decision.md`
- `models/`
- `full_reports/`

样本缓存：

```text
data/cache/eth_ai_trading/r01_trades_only/
```

## 决策值

- `PASS_TRADES_ONLY_EDGE`：验证期选出的冠军在 2026 封存期和压力测试仍通过。
- `FAIL_NO_VALIDATION_EDGE`：2025 验证期没有候选通过，停止用复杂模型掩盖失败。
- `FAIL_SEALED_HOLDOUT`：验证期看起来有效，但 2026 封存期失败。
- `BLOCKED_PUBLIC_LOADER`：公共 1s Loader 在少量抽样窗口无法正常提供必要字段。

## 性能约束

- 不一次性读取多年 1s 数据。
- 每天仅加载前 5 分钟历史上下文和后 15 分钟标签上下文。
- 每月落盘一个 `NumPy .npy` 内存映射样本分片，避免后续模型反复解压。
- 训练使用确定性时间哈希抽样，不改变时序切分。
- 预测和交易评估按月流式执行。
- 训练、校准和测试边界自动加入最大标签周期 embargo。


## R01.1 时间精度兼容修复

部分 Pandas / SQLite 组合会把 Loader 返回的索引保存为 `datetime64[us]`。
旧版 R01 使用 `.view("int64")` 后错误地把微秒整数当成纳秒，导致未来标签搜索全部失效，最终报：

```text
RuntimeError: no R01 samples produced for 2023-01
```

R01.1 会在所有标签、MFE/MAE、缓存时间戳路径中显式统一为 epoch nanoseconds，并将缓存 schema 升级到 v3。覆盖补丁后直接执行默认命令即可；旧 schema 缓存不会被误复用。

如果再次出现零样本，程序会在第一个异常日期立即输出 `feature_valid`、`label_valid` 和索引 dtype，不再无意义地跑完整个月。

## 启动依赖检查

默认研究同时运行 Ridge 和 LightGBM。运行前当前 Python 环境必须安装 LightGBM：

```text
python -m pip install lightgbm
```

脚本会在访问数据和构建月度缓存之前检查依赖。若依赖缺失会立即退出，不再先运行多年缓存。已经生成的兼容月度缓存会自动复用，重新运行时不要传 `--force-rebuild-cache`。

只做 Ridge 诊断时可以显式运行：

```text
python research\eth_ai_trading\01_trades_only_supervised_baseline.py --models ridge
```

Ridge-only 不能替代完整的 Ridge + LightGBM R01 结论。
