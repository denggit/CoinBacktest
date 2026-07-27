# Prebuild 安全提速与断点恢复

本补丁只优化数据预构建路径，不改变市场时间、订单簿重放顺序、特征可用时间、价格分桶、成交归因或回测口径。

## 1. Offline Liquidity Map

`tools/prebuild_okx_offline_liquidity_map.py` 的主要提速：

- 同一订单簿版本只计算一次深度窗口、墙候选基础量和保留价格格，多个固定时钟采样复用该只读结果。
- 最优买卖价增量维护，不再在每次采样时反复对完整字典做 `max/min`。
- Raw Trades 只读取并保留 `ts_ms/price/size/side`，不再生成本任务不用的时间对象、symbol、trade_id 和逐行 `raw_json`。
- 支持按 UTC 日进程级并行；5000 档自动模式最多 2 个 worker，并根据当前可用内存降到 1 个。
- 任一日失败会在全新 worker 中重试；重试时自动降低并发，减少内存和磁盘压力。
- NPZ 默认使用 DEFLATE level 1；数组内容和 dtype 不变，只降低压缩 CPU 成本。
- 默认关闭重复 schema probe；需要排查输入格式时再加 `--schema-probe` 或 `--inspect-only`。

容灾保持不变并加强：

- 每个 UTC 日独立构建。
- 先写 `.part`，同盘 `os.replace` 原子发布。
- Features 与 Heatmap 发布后，Metadata 最后发布；Metadata 是完成标记。
- 中断后原命令重跑，完整日直接跳过，不完整日自动重建。
- 已有缓存的配置若与本次命令不同，会明确报错，防止静默复用错误口径；确认后才使用 `--force-rebuild`。

当前 5000 档补建命令：

```bat
python tools\prebuild_okx_offline_liquidity_map.py --symbol ETH-USDT-SWAP --start-date 2025-11-01 --end-date 2025-12-31 --books-depth 5000 --map-profile broad --price-step 1 --feature-seconds 1 --heatmap-seconds 5 --large-depth-ratio 0.5
```

默认 `--workers 0` 会自动选择安全并发。内存很紧时可显式使用：

```bat
python tools\prebuild_okx_offline_liquidity_map.py --symbol ETH-USDT-SWAP --start-date 2025-11-01 --end-date 2025-12-31 --books-depth 5000 --map-profile broad --price-step 1 --feature-seconds 1 --heatmap-seconds 5 --large-depth-ratio 0.5 --workers 1 --trade-chunksize 300000
```

不要为了速度使用 `--allow-books-only`；该参数会让成交消耗、撤单和补单归因无效。

## 2. Liquidity Primitives

`tools/prebuild_okx_liquidity_primitives.py` 必须保持逐日因果顺序，因此没有盲目并行。安全提速来自：

- 直接读取 canonical 数值 Heatmap，不生成 UI 才需要的 side 文本、价格列和时间列。
- 断点续跑遇到已有日时，只从 NPZ 解压 `bucket_end_ms/snapshot_q95/snapshot_q99` 三个因果参考数组；不再加载数百万个 cell 数组。
- 默认 NPZ 压缩级别改为 1，仍然原子写盘、Metadata 最后发布。
- 缓存参数不一致时拒绝静默跳过。

Offline Map 完成后运行：

```bat
python tools\prebuild_okx_liquidity_primitives.py --symbol ETH-USDT-SWAP --start-date 2025-11-01 --end-date 2025-12-31 --books-depth 5000 --cache-version v1
```

由于不存在 2025-10-31 的 5000 档数据，`[warmup-missing] 2025-10-31` 是预期行为。研究统计排除 2025-11-01 的因果热身区即可。

## 3. Liquidity Period-End Cache

`tools/prebuild_okx_liquidity_period_end_cache.py` 现在支持独立按日进程并行、失败重试和进度 ETA。底层缓存仍按日原子发布。

```bat
python tools\prebuild_okx_liquidity_period_end_cache.py --symbol ETH-USDT-SWAP --start-date 2025-11-01 --end-date 2026-06-30 --books-depth 5000 --timeframe 15m --price-step 1
```

## 4. Trade Bars

`tools/prebuild_okx_trade_bars.py` 从“每个 timeframe 重新扫描一次同一日 Raw Trades”改为“每个 UTC 日只扫描和标准化一次，再同时分发给全部 timeframes”。

- 3 个周期由 3 次 ZIP/CSV 扫描降为 1 次。
- 每个 timeframe 仍有独立 DB 表和 coverage 完成标记。
- coverage 只在整个 UTC 日全部读取完成后写入。
- 中途中断留下的行是幂等 upsert；没有 coverage 的日重跑后会覆盖为完整结果。
- SQLite 使用 WAL、NORMAL、busy timeout、内存临时表和受控 mmap/cache。

```bat
python tools\prebuild_okx_trade_bars.py --symbol ETH-USDT-SWAP --start-date 2025-11-01 --end-date 2026-06-30 --timeframes 1s 5s 1m 5m 15m 30m 1H 4H --chunksize 500000
```

## 5. Range Bars / Range Footprints

所有 Range 预构建共享的 Raw Trades CSV reader 现在只解压必要列，避免读取无关 payload。

新任务优先使用 `tools/prebuild_okx_range_all.py`，因为它已经具备：

- 一次扫描同时生成多个 range pct；
- 精确 checkpoint；
- 中断续跑；
- 内存失败自动缩小 chunksize；
- staged flush 和 purge/replay 修复。

旧的 `prebuild_okx_range_bars.py` 与 `prebuild_okx_range_footprints.py` 仍兼容，但分别运行会重复扫描 Raw Trades。

## 6. Liquidation Inputs

`prebuild_okx_liquidation_inputs.py` 未强行并行。它调用多个公共 API 并写共享持久层，盲目并发会增加限频、HTTP 重试和 SQLite 写冲突风险。该路径保持顺序执行，是有意的稳定性选择。

## 7. 验证结果

- Liquidity Map、Replay、Store、Loader、墙研究相关测试：83 passed。
- Range checkpoint/resume 与 OKX 衍生数据测试：19 passed。
- 新增 primitives 轻量参考恢复与 minimal trades 路径测试：6 passed。
- Trade Bars 旧/新实现对同一合成 Raw Trades 的 1m/5m/15m 表逐字段完全一致；物理 Raw Trades chunks 从 15 次读取降为 5 次读取。
- 两日 spawn 进程 Offline Map 集成测试成功，并验证原命令重跑全量 skip。
- 两日 Period-End 并行集成测试成功。
- 全仓可收集部分另有 146 passed；剩余失败来自上传工程本身缺少 panic 插件/脚本及已有 import-boundary 违规，与本补丁无关。
