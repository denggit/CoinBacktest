# Human Trader Replay Lab V1.12 — 6 个可持久化快捷周期槽位

## 目标

解决单图模式下快捷周期被写死的问题。用户可以把任意一个快捷槽位从 30m 改成 4H、1H、5m 等，之后切换到其它槽位再回来，仍然保留该设置。

## 默认槽位

`30m / 15m / 5m / 2m / 1m / 4H`

六个槽位均可选择：`1m / 2m / 5m / 15m / 30m / 1H / 4H / 1D`。

## 持久化语义

周期槽位属于用户界面偏好，不属于单个 Episode 的市场决策数据，因此保存在浏览器 localStorage：

- `humanReplayLab.timeframeSlots.v1`
- `humanReplayLab.activeTimeframeSlot.v1`

Fit 只恢复当前图表的缩放/平移，不修改周期槽位。新建 Episode、切换 Symbol 和页面刷新也不重置槽位。

## 因果性

本版本只修改周期选择 UI，不修改服务端 K 线生成、HTF forming candle 或 `available_time` 规则。
