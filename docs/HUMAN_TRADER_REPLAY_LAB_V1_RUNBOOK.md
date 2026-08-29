# Human Trader Replay Lab V1 Runbook

## 本版目的

让人工 Replay 更接近实际手动交易流程：30m/15m 做前置 setup，2m/1m 做执行，同时允许任意 pane 切周期并共享人工标记。

## 默认布局

- Setup A: 30m
- Setup B: 15m
- Execution A: 2m
- Execution B: 1m

四张图共享同一 `episode.cursor_time`，但拥有独立的缩放、拖动和 timeframe 设置。

## 标记共享语义

任意 pane 点击价格后创建的 LIQUIDITY / TARGET / MARKER / SL / TP 都写入 Episode event stream，因此所有 pane 都绘制该事件。

同时事件保存：

- `event.timeframe`: 创建标记时的 source timeframe
- `payload.anchor_time`: 用户点击的历史 K 线起点
- `payload.anchor_timeframe`: source timeframe
- `payload.source_pane`: 创建它的 pane

这样共享显示不会抹掉“这条 liquidity 是我从哪个周期/哪根结构识别出来”的训练语义。

## 因果规则

2m 由 1m resample 时同样使用：

```text
available_time = bar_start + 2 minutes
```

只有 `available_time <= cursor_time` 的 bar 才返回前端。

## 启动

```bash
python human_replay_lab/server.py --host 127.0.0.1 --port 8775
```
