# ETH LF Portfolio V10B All-Engine Swing Structural Stop

V10B 是 V10A 的正式回测候选，不是实盘版本。

## 变更

在 V10A 基线基础上加入：

```text
struct_stop_all_swing_n21_buf0p0_trig0p0_h0
```

普通语言解释：

- 多单持仓后，看最近 21 根已完成 4H K 线的最低点。
- 空单持仓后，看最近 21 根已完成 4H K 线的最高点。
- 如果这个结构位比原止损更有利，并且没有穿过当前收盘价，就把后续止损收紧到这个结构位。
- 当前 4H bar 内是否触发止损，仍然只能用上一根已经生效的 stop；当前 bar close 计算出来的新结构止损，只能影响后面的 bar。

## 为什么不是 future function

- entry 仍然是 V10A 的 4H close 信号、下一根 4H open 执行。
- swing high/low 只用当前已经收完的 4H bar 和之前的 bar。
- 新结构 stop 不会用于当前 bar 的 intrabar stop touch。
- 不用全样本分位数、不用日期过滤、不用未来 MFE/MAE 生成规则。

## Windows 运行

```powershell
$env:PYTHONPATH="."
python backtest/lf/eth_lf_portfolio_v10b_all_swing_structural_stop_backtest.py --end-date 2026-06-15 --out-dir data/reports/lf/eth_lf_portfolio_v10b_all_swing_structural_stop/turbo
```

## Linux / macOS 运行

```bash
PYTHONPATH=. python backtest/lf/eth_lf_portfolio_v10b_all_swing_structural_stop_backtest.py --end-date 2026-06-15 --out-dir data/reports/lf/eth_lf_portfolio_v10b_all_swing_structural_stop/turbo
```

## 输出

```text
data/reports/lf/eth_lf_portfolio_v10b_all_swing_structural_stop/turbo/
  eth_lf_portfolio_v10b_all_swing_structural_stop_trades.csv
  eth_lf_portfolio_v10b_all_swing_structural_stop_equity.csv
  eth_lf_portfolio_v10b_all_swing_structural_stop_signal_audit.csv
  eth_lf_portfolio_v10b_all_swing_structural_stop_summary.json
```

## 后续验证

V10B 如果回测结果复现 research 结论，再继续做：

1. V10A vs V10B 逐笔 trade diff 修复版。
2. 参数邻域验证：n=13/21/34，buffer=0/0.1/0.25/0.5。
3. 手续费 / 滑点 stress。
4. top winner dependency。
5. AetherEdge 实盘止损同步可实现性评估。

不能直接迁移实盘。
