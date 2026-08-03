# R01 Data Audit Runbook（已废止）

原独立数据审计阶段已经被取消，原因是 CoinBacktest 的 `src.data_feed` 已经过大量研究验证，不应在每个新方向重复平台级法证审计。

现在请运行：

```bat
python research\eth_ai_trading\01_trades_only_supervised_baseline.py
```

新的 R01 只做三段公共 Loader smoke check，然后立即构建因果样本、训练 Ridge/LightGBM 并执行真实成本回测。

详细说明：`docs/ETH_AI_TRADING_R01_TRADES_ONLY_BASELINE_RUNBOOK.md`。
