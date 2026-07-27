# Flow–Impact R03 Performance Hotfix

## 问题

R03 在以下日志后长时间无进展：

```text
[pa] accumulated pressure -> sweep/reclaim or break/retest
```

原因是旧实现对每一个压力事件，都重新对完整历史特征列执行
`pd.to_numeric`。正式历史约 236 万行，导致复杂度接近：

```text
事件数 × 全历史行数 × 多个特征列
```

因此不是正常慢，而是性能实现错误。

## 修复

1. 所有特征列只在每个 accumulation window 开始时转换一次。
2. Pivot 查询改成 NumPy + `searchsorted` 的因果局部索引。
3. PA检测增加 `[pa-detect] pressure events` 进度条。
4. 结构TP/SL First-Touch 改为局部NumPy扫描。
5. 结构退出增加 `[pa-exit] structural first-touch` 进度条。
6. 新增回归测试，禁止以后重新出现“每事件全列转换”。

## 逻辑边界

没有修改：

- 累计压力定义；
- 1m closed bar 信号；
- next bar open 入场；
- causal pivot available time；
- sweep/reclaim 与 break/retest 条件；
- Price Action止损和止盈；
- 同bar TP/SL按止损优先；
- 手续费、滑点与候选门槛。

在 5,000 行确定性样本上，修复前后产生的 127 个 setup 在核心字段上逐笔一致。

## 验证

```text
Python compileall: PASS
Flow Impact + R03 tests: 10 passed
R03 self-test: PASS
5,000行等价性检查: 127/127 setups逐笔一致
100,000行性能烟雾测试:
  features 约1.29s
  pivots   约0.02s
  PA detect约0.05s
```

## 使用

停止旧进程后，将本压缩包覆盖到 CoinBacktest 根目录，再运行原命令：

```bat
python research\mhf\flow_impact_state\03_accumulated_pressure_pa.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30
```
