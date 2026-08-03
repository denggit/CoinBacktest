# ETH AI Trading R03.3.1 独立预警与剩余空间审核

## 目标

R03.3按每15分钟判断点统计命中，不能回答一次真实预警是否有交易价值。R03.3.1不增加新特征、不打开2026、不改变事件定义，只审核：

1. 连续高分合并为一个独立预警段；
2. 每段只看第一次预警；
3. 允许状态刚开始后的短暂早期确认；
4. 预警后必须仍有足够目标或双向区间；
5. 统计独立预警成功率和独立事件覆盖率。

## 固定定义

- 信号间隔不超过1小时：视为同一预警段。
- 早期确认：启动后不超过2小时，且过程完成不超过25%。
- 单边扩张剩余机会：至少2.5%。
- 高波动震荡剩余双向区间：至少3%。
- 仅审核2024与2025；2026H1继续封存。

## 运行

```text
python research\eth_ai_trading\03_3_1_process_alert_value_audit.py
```

正常运行复用R03.2/R03.3缓存，不加任何force参数。

## 报告

目录：

```text
data\reports\research\eth_ai_trading\03_3_1_process_alert_value_audit
```

重点文件：

- `03_independent_alert_episode_metrics.csv`
- `04_event_level_coverage_metrics.csv`
- `05_first_alert_episodes.csv`
- `06_event_first_alert_coverage.csv`
- `07_actionability_candidates.csv`
- `99_decision.md`
- `gpt_review_pack.zip`
