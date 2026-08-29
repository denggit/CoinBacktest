# Human Trader Replay Lab V1.8

## 本次目标

1. Replay Lab 不再把 Symbol 写死为 `SOXL-USDT-SWAP`。
2. 自动扫描 `data/crypto_history.db` 中已有的 `*_1m` 表，并在前端 Symbol 下拉框中列出。
3. 支持 `SOXL-USDT-SWAP`、`ETH-USDT-SWAP` 以及其他已经存在本地 OKX 1m 表的标的。
4. 大型 ETH 1m 数据不再为了 Replay 启动而整表加载；新增 bounded range loader，仅加载当前 Episode 所需窗口并缓存。
5. 增加 `research/human_trader_replay/01_manual_setup_execution_audit.py`，用 Replay Lab 的人工交易记录 + OKX 1m 路径做执行失败审计。

## Replay 时钟

V1.8 暂时保留现有 Blind Replay 时钟：

- 只创建工作日 Episode；
- 07:30 ET 开始；
- 16:00 ET 结束；
- 图表上下文不裁剪夜盘、盘后或周末 OKX bars；
- UI 继续显示北京时间。

这是为了让 SOXL 与 ETH 的第一批 Human Clone 数据保持可比。ETH 独立 24h Episode 起点可以在后续版本单独设计，不在本补丁中擅自改交易流程。

## Manual Setup Execution Audit

默认命令：

```bash
python research/human_trader_replay/01_manual_setup_execution_audit.py
```

输入：

- `data/human_replay_lab/replay.sqlite3`
- `data/crypto_history.db` 中对应 Symbol 的 OKX 1m 表

输出：

```text
data/reports/research/human_trader_replay/manual_setup_execution_audit/
    trade_path_audit.csv
    horizon_audit.csv
    failure_taxonomy.csv
    episode_audit.csv
    legacy_unclosed_trade_candidates.csv
    manual_setup_execution_audit.md
    manifest.json
```

核心只回答：

- 止损后原 TP 是否后来兑现；
- 为了活到原 TP 实际需要承受多少 MAE；
- 是否与人工 Bias 冲突；
- 同 Episode 后续同方向重进是否成功；
- 失败更像方向错误还是执行过早。

不会做 stop 参数网格搜索，也不会看到小样本亏损后调参。
