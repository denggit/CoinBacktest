# Human Trader Replay Lab V0 — Runbook / Progress Log

## 目标

把手动实盘 setup 的量化过程从“先写 ICT 规则”改成“先采集真实人工决策轨迹”。V0 只负责历史 causal replay、人工 Liquidity/方向/目标标注、模拟交易和 Episode 存储；暂不训练模型，不自动判断 MSS/FVG，不优化收益。

## 本阶段完成

### 1. 独立本地 Web 应用

新增 `human_replay_lab/`，不侵入现有 `analyze_tool`、backtest 或 research 逻辑。

启动：

```bash
python human_replay_lab/server.py --host 127.0.0.1 --port 8775
```

### 2. Replay 数据读取

- 只读现有 `data/crypto_history.db`。
- 支持 1m/5m/15m/30m/1H/4H/1D。
- 查询采用 bounded SQLite slice，避免每一步 replay 把几百万根 1m 全表加载到内存。
- 若某高周期本地表缺失，可从已有 1m 本地数据做 bounded causal resample；不联网补数据。

### 3. 强制因果规则

后端统一使用：

```text
bar_available_time = bar_start_time + timeframe
```

仅当：

```text
bar_available_time <= replay_cursor
```

才把 bar 返回浏览器。

例如 cursor=10:05：

```text
5m 10:00 bar: 可见
15m 10:00 bar: 不可见
```

模拟市价开/平仓只读取当前 cursor 对应 1m bar 的 `open`；不会读取该分钟的 high/low/close。

### 4. 人工行为采集

已支持：

- 在过去 K 线上点击具体价格和 anchor candle。
- 标 BSL / SSL / Other Liquidity。
- Liquidity importance: Normal / High / A+。
- 记录 anchor_time + anchor_timeframe，用于以后学习“为什么这个过去结构被交易员认作 liquidity”。
- Bullish / Bearish / Neutral / Unsure Bias。
- Expected Delivery Target。
- Watch / Wait / Skip / Invalidate。
- Long / Short / Flat 模拟市价执行。
- SL / TP。
- 自然语言 Note。
- Decision Timeline。
- JSON Episode export。

### 5. Episode 数据结构

默认持久化：

```text
data/human_replay_lab/replay.sqlite3
```

表：

```text
episodes
events
```

每个 event 至少保存：

```text
episode_id
event_time
event_type
timeframe
price
payload_json
created_at
```

不会只保存最终交易；WAIT、WATCH、SKIP、BIAS、TARGET、LIQUIDITY 等决策过程都会保存。

## 自动测试

新增：

```text
tests/human_replay_lab/test_data_service.py
tests/human_replay_lab/test_store.py
```

覆盖：

- Episode/event SQLite round-trip。
- cursor update/close。
- 5m bar 在 available_time 前不可见、关闭后才可见。
- 市价成交只使用 cursor 1m open。
- 缺少 native 15m 时，1m causal resample fallback 仍不泄漏未关闭 bar。

当前专属测试：

```text
5 passed
```

另外运行项目现有 `tests/test_import_boundaries.py` 时失败；失败项来自上传基线中已有的 `backtest/mf/trend_following/*` import coupling，与本 V0 新文件无关。本次没有修改这些文件。

## V0 明确未做

- 不训练行为克隆模型。
- 不训练 RL。
- 不自动识别 MSS/FVG/Sweep。
- 不根据最终盈亏给人工标注打标签。
- 不显示未来结果。
- 不做策略参数优化。

## 下一阶段建议 V0.1

优先让用户实际用 V0 replay 一批 Episode，再根据真实使用摩擦修 UI。之后再加：

1. 图形化拖动 Entry / SL / TP。
2. 多图联屏（15m + 5m + 1m 同时显示，而非 tab 切换）。
3. Liquidity 线可编辑、删除、消费/未消费状态。
4. Position / PnL 仅基于已 replay 到的数据动态更新。
5. Setup Watch 从开始到 invalid/entry 的区间化 Episode 子状态。
6. A/B setup preference capture。
7. 训练数据 exporter，而后才进入 Human Clone V0。
