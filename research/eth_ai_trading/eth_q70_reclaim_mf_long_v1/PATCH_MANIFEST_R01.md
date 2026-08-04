# ETH AI Trading R01 Replacement Patch

## 变更目的

废止“只做数据审计”的 R01，改为一次性完成必要前置并立刻运行 Trades-only 监督学习与完整交易基线。

## 核心变化

1. 所有数据统一通过 `OKXTradeBarLoader` 公共接口。
2. 不直接读取 raw ZIP，不直接访问 SQLite。
3. 三个抽样窗口 smoke check 后立即研究。
4. 分日构建因果特征，分月缓存，断点续跑。
5. Ridge 与 LightGBM 同时作为简单基线。
6. 2025 选择冠军，2026H1 完全封存。
7. 真实手续费、滑点、1x/2x/3x 成本和多延迟压力。
8. 输出完整报告和明确 PASS/FAIL。

## 不做

- 不修改任何既有策略。
- 不修改 TP/SL 或 Portfolio。
- 不修改 AetherEdge。
- 不自动下载或重建数据。
- 不执行 Git commit。
