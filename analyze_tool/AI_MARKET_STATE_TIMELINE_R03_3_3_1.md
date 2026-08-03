# AI多周期市场状态时序可视化 R03.3.3.1

## 目的

把 R03.3.3.1 的真实历史输出按时间对齐到 Analyze Tool K 线，人工检查：

- 战略状态是否符合长期结构；
- 战术状态是否对应推动、回调和整理；
- 入场层是否过度频繁翻转；
- 活跃度状态是否与肉眼可见波动一致；
- 活跃状态持续概率高低是否与未来3小时真实持续结果一致。

## 数据合同

状态色带读取：

`data/cache/eth_ai_trading/r03_3_3_1_universal_state/state_YYYY/`

完整 OOS 活跃持续概率读取：

`data/cache/analyze_tool/ai_market_state_r03_3_3_1/activity_persist_oos_YYYY/`

Analyze Tool HTTP 请求不会训练模型，也不会重建多年特征。

## 运行

```text
python tools\prebuild_ai_market_state_timeline.py
```

```text
python analyze_tool\server.py --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765`，选择：

```text
AI多周期市场状态 R03.3.3.1
```

建议：

- 数据类型使用普通K线或Trade Bar均可；插件读取的是独立模型缓存；
- 周期优先1m或15m；
- 日期先看2024、2025；
- 研究模式显示完整分数、概率、风险和方向一致性；
- 点击K线锁定后，右侧显示该时刻状态年龄、边界距离和OOS概率。

## 因果对齐

状态缓存的 `decision_time` 表示当时已经可用的模型状态。插件只执行向后对齐：

```text
state.decision_time <= candle.timestamp
```

最大容忍30分钟，不会把后续状态提前画到更早K线上。

## 未来结果审计色带

`历史结果审计（未来3h）` 使用未来3小时真实状态结果：

- 绿色：未来3小时状态保持；
- 红色：未来3小时内发生转换。

它只用于历史可视化审核。该字段没有进入当前特征，也不能用于实盘判断。可在插件参数中关闭。

## 覆盖范围

- 状态：2021—2025；
- 完整OOS持续概率：2024、2025；
- 2026：封存，不展示研究预测。
