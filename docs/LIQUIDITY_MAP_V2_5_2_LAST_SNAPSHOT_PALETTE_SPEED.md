# Liquidity Map V2.5.2 — Last-Snapshot Full Palette & Speed

## 目标

V2.5.2 保留 V2.5 的正确时间语义，同时修复周期末快照图只剩深色块、浅色挂单大面积消失的问题，并把重复运行从“每次重扫全部 5 秒热力数据”改为读取按日的轻量周期末缓存。

## 不变的时间语义

以 15m 为例：

- 历史 K 线只使用该 K 线结束前最后一个有效 Books 快照；
- 当前尚未结束的 K 线使用最新 Books 快照并持续覆盖更新；
- 5 秒数据只负责提供离线回放和寻找最后快照，不直接形成墙切片；
- 不做 15m 内平均、最大值、并集或未来回填。

## 浅色色块消失的根因

V2.5 的“最后快照”本身没有删除浅挂单。真正的问题来自显示链路：

1. 后端默认 `min_intensity_pct=0.5`，在前端色阶显示 0% 时仍会预先丢弃很浅的正值单元格；
2. 超过 `max_render_cells=300000` 后，旧 reducer 主要保留最深单元格，浅色背景被系统性淘汰；
3. 浅正值颜色过于接近暖白背景，即使保留下来也难以辨认。

V2.5.2 的修复：

- 后端默认阈值改为 0%，只要最后快照深度为正就有资格显示；
- 默认浏览器预算提高到 800,000 格；
- 超预算时按弱/中/强三层分配容量，并保证每个时间点、每一侧至少保留一个最强格；
- 最浅正值使用单调的显示提升，仍保持挂单强弱排序，不影响墙检测或回测数据；
- 周期末缓存明确保留所有 `end_depth_base > 0` 的价格格。

## 性能优化

### 周期末日缓存

查询缓存 schema 升级为 V2，默认目录为：

```text
data/okx/derived/liquidity_map/<SYMBOL>/books_<DEPTH>/_query_cache/period_end_v2/
```

缓存只保存每根目标 K 线最后快照中的正深度价格格。它使用未压缩 NPZ，目的是加快反复读取，而不是节约磁盘。

原始 Liquidity Map 日数据无需重新构建。第一次为某个 `timeframe + price_step` 建缓存时仍需扫描对应日文件一次；之后 Analyze Tool 直接读取轻量缓存。

### 一次计算 24h 因果量尺

周期末显示和墙检测共用同一份：

```text
depth_ratio
reference_depth
```

不再对同一矩阵重复计算 24h 因果深度参考。

### NumPy 因果归一化

24h 高位参考改为 NumPy 分组与单调队列滚动最大值，保持原有因果语义，不依赖未来可见窗口。

## 墙检测补充

墙仍使用每根 K 线最后/最新快照，但历史连续性允许在相邻价格格内漂移。默认 `history_price_tolerance_bins=2`：

- 过去某根 K 线在 2030，下一根在 2031 或 2032，仍可视为同一视觉价格带；
- 当前候选必须在真实当前价格格存在，邻域容忍只用于历史覆盖率，不会凭空扩大当前墙；
- 整个过程只读取当时及之前的数据。

## 一次性预构建示例

```text
python tools\prebuild_okx_liquidity_period_end_cache.py --symbol ETH-USDT-SWAP --start-date 2026-01-01 --end-date 2026-01-31 --books-depth 5000 --timeframe 15m --price-step 1
```

旧 `period_end_v1` 缓存可以保留；V2 使用独立目录，不会误读旧语义。

## 验证

- `PYTHONPATH=. pytest tests/liquidity_map -q`：69 passed
- `node --check analyze_tool/static/app.js`：通过
- `PYTHONPATH=. python analyze_tool/selftest.py`：通过

项目全量测试仍被两个既有 panic 模块导入错误阻塞，与本补丁无关。
