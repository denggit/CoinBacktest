> **Historical R01 manifest.** R01.1已替代初版的微观Swing与Swing中心化解释。当前交付清单见 `R01_1_PATCH_MANIFEST.md`。

# R01 Patch Manifest

## 新增入口

```text
research/eth_ai_trading/eth_latent_liquidity_path_v1/01_release_reversal_path_atlas.py
```

历史兼容提示入口：

```text
research/eth_ai_trading/01_audit_trade_data.py
```

该兼容入口不会恢复、运行或修改已封存的Q70模型，只用于明确旧入口已被替代并保持历史测试兼容。

## 新增模块

```text
src/ai_research/latent_liquidity_path_atlas/config.py
src/ai_research/latent_liquidity_path_atlas/candidates.py
src/ai_research/latent_liquidity_path_atlas/features.py
src/ai_research/latent_liquidity_path_atlas/macro.py
src/ai_research/latent_liquidity_path_atlas/outcomes.py
src/ai_research/latent_liquidity_path_atlas/clustering.py
src/ai_research/latent_liquidity_path_atlas/reports.py
src/ai_research/latent_liquidity_path_atlas/pipeline.py
src/ai_research/latent_liquidity_path_atlas/time_axis.py
```

## 测试

```text
tests/ai_research/test_latent_liquidity_path_atlas.py
```

覆盖：

- 无Swing也能由流动性释放进入；
- 普通成交下的边界越界也能进入；
- 当前爆量不污染自己的历史基线；
- 浅扫立即反转标签；
- 延伸稳定后反转标签；
- Swing为特征而非门槛；
- 聚类只在冻结发现期拟合；
- 高周期Bar按真实可用时间对齐；
- 事件当秒不进入潜在池路径特征；
- 聚类不使用事件爆量、绝对ETH价格或绝对成交规模；
- SQLite/Arrow `datetime64[us]` 与事件 `datetime64[ns]` 可安全as-of对齐；
- 1秒和1分钟内部轴统一为`datetime64[ns]`；
- 空候选/空标签生成明确质量失败而不是缺列崩溃；
- 1分钟宏观上下文覆盖不足时质量门禁失败。

## 已执行验证

```text
R01专项：14 passed
全部AI Research：256 passed
Data Feed + Research Common：23 passed
混合us/ns合成端到端报告：passed
compileall：passed
CLI --help：passed
```

完整Import Boundary仍报告仓库既有155项研究脚本耦合问题；新模型目录新增0项。
研究逻辑、候选、标签和聚类参数未修改。
