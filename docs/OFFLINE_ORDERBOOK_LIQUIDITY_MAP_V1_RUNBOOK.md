# Offline Order Book Liquidity Map V1.2

## 目标

V1 只处理离线历史数据，不创建 WebSocket 采集器：

```text
OKX Books 原始文件
+ 同日 OKX Raw Trades
→ 重建盘口
→ 1 秒因果流动性特征
→ 1 分钟默认热力缓存
→ analyze_tool 展示 / 后续事件研究与回测
```

Books 是挂单流动性主体；Raw Trades 只用于估算挂单减少中有多少属于真实成交消耗，以及成交后的补单。热力图不是交易信号。

## 数据边界与时间

原始 Books 和 Raw Trades 按 UTC 自然日组织。项目 K 线默认按 `config.loader.TIMEZONE`（当前 `+8`）显示，因此：

```text
UTC 原始日 2026-06-01
→ analyze_tool 大致显示为 2026-06-01 08:00 至 2026-06-02 08:00
```

工具不会用未来快照补盘口缺口。`seqId/prevSeqId` 断档后，增量盘口会失效，直到源数据中的下一份完整 snapshot。

## 第一步：检查 Books 文件格式

```bash
python tools\prebuild_okx_offline_liquidity_map.py --symbol ETH-USDT-SWAP --start-date 2026-06-01 --end-date 2026-06-01 --inspect-only
```

它只打印本地文件和前几行，不构建、不下载、不复制原始文件。

支持的输入形态：

- JSON/JSONL：`action + data + asks/bids`
- CSV：`timestamp,snapshot,asks,bids,seq_id,prev_seq_id`
- CSV 价格档行：`timestamp,side,price,size`，没有 action 时按完整 snapshot 处理

如果检测失败，保留完整 `[schema]` 输出用于适配真实文件，不能猜测字段后继续构建。

## 第二步：构建 2026-06-01

```bash
python tools\prebuild_okx_offline_liquidity_map.py --symbol ETH-USDT-SWAP --start-date 2026-06-01 --end-date 2026-06-01 --price-step 1 --feature-seconds 1 --heatmap-seconds 60 --large-depth-ratio 0.5
```

默认参数含义：

- `feature-seconds=1`：回测读取的盘口状态时钟
- `heatmap-seconds=60`：仅用于展示与快速查询，降低磁盘和浏览器压力
- `price-step=1`：每 1 USD 一个价格格
- `large-depth-ratio=0.5`：当时同侧最厚价格格的 50% 以上属于“大流动性候选”
- `decision-delay-ms=1000`：1 秒特征在桶结束后再延迟 1 秒开放
- `max-book-staleness-seconds=30`：长时间没有任何盘口事件时停止沿用旧盘口

Raw Trades 缺失时默认失败。`--allow-books-only` 仅用于检查画面，此时 `trade_attribution_valid/flow_valid=0`，撤单、消耗、补单估算保持 0，回测不得把它们解释为真实零值。

## 衍生数据位置

```text
data/okx/derived/liquidity_map/ETH-USDT-SWAP/books_400/YYYY/MM/
├── YYYY-MM-DD.features.npz
├── YYYY-MM-DD.heatmap.npz
└── YYYY-MM-DD.metadata.json
```

原始 Books 和 Raw Trades 不会被复制。NPZ 是可删除、可重新构建的派生缓存。

`features.npz` 保存策略/研究字段：

- `available_time_ms`
- 5/10/25/50 bps 买卖深度
- 25 bps 深度不平衡
- 上下方最厚挂单价格与数量
- 大流动性价格格数量与合计深度
- `trade_attribution_valid`（缺 Raw Trades 时为 0）
- 主动买卖成交量
- 盘口新增/减少
- 估算撤单、成交消耗、补单

`heatmap.npz` 保存时间 × 价格格：

- 时间加权平均挂单深度
- 同时刻同侧相对最厚比例
- `flow_valid`
- 新增、减少、成交、撤单、消耗、补单估算

热力深度按 1 秒固定时钟采样后在整个热力时间桶内平均；只存在一个采样点的短暂挂单，不会与持续整分钟的挂单显示成同样深。V1 不声称还原 100ms 内的精确驻留时长。


## 单一基础缓存与动态周期

`--heatmap-seconds 60` 只构建一份 1 分钟基础热力图，不按 K 线周期重复保存文件。

```text
1分钟基础 heatmap.npz
→ 查询时动态聚合 3m / 5m / 15m / 30m / 1H / 4H
```

Analyze Tool 默认选择“跟随 K 线周期”。切换 5m K 线时直接把连续 5 个 1 分钟格聚合成一个 5 分钟格；切换 15m 时聚合连续 15 个。无需重新运行 prebuild。

聚合语义固定为：

- `depth_base/depth_usd/order_count`：时间加权平均，不能直接累加。
- `added/removed/executed/cancelled/consumed/replenished`：区间总量求和。
- 颜色归一化：完成周期聚合后再计算，不能先归一化每分钟颜色再合并。
- 1 分钟基础缓存不能反推 15 秒或 30 秒，禁止伪造更细粒度。

回测与研究也通过同一个 `src.data_feed` 接口动态聚合：

```python
heatmap_5m = loader.load_heatmap_aggregated(
    "2026-06-01 08:00:00",
    "2026-06-02 08:00:00",
    timeframe="5m",
    price_step=1.0,
)
```

因此当前已经构建完成的 `2026-06-01.heatmap.npz` 可以直接复用，不需要重新构建。

## 第三步：analyze_tool

```bash
python analyze_tool/server.py --host 127.0.0.1 --port 8765
```

建议第一次选择：

```text
数据：Trade Bar 或普通 K 线
周期：1m
区间：2026-06-01 08:00:00 至 2026-06-02 07:59:59
插件：离线订单簿流动性热力图 V1.2
```

颜色模式：

1. `当前请求区间 Q99 封顶`：最接近 CoinGlass 观察方式，超过 Q99 的格子全部最深。
2. `手动最大值`：例如选择 ETH 单位并设为 100，则 100 ETH 及以上全部最深。
3. `每个时刻相对最厚挂单`：同一时刻、同一侧最厚格为 100%。

“大流动性阈值 50%”只用于描述当时相对厚度。回测不能因为颜色超过 50% 就直接交易，还要加入持续时间、价格靠近、撤单、成交消耗、补单和后续路径。

## 回测读取接口

所有研究和回测通过 `src.data_feed`：

```python
from src.data_feed.okx_liquidity_map_loader import OKXLiquidityMapLoader

loader = OKXLiquidityMapLoader(symbol="ETH-USDT-SWAP", books_depth=400)

features = loader.load_features(
    "2026-06-01 08:00:00",
    "2026-06-02 07:59:59",
    index_mode="available_time",
    valid_only=True,
)

heatmap = loader.load_heatmap(
    "2026-06-01 08:00:00",
    "2026-06-02 07:59:59",
)
```

策略默认只能按 `available_time` 使用特征。`bucket_start/bucket_end` 只用于追踪特征来自哪个已完成区间。

## V1 验收顺序

1. `--inspect-only` 确认真实文件字段。
2. 构建 2026-06-01。
3. 检查事件数、snapshot 数、sequence gap 和有效 feature 行数。
4. 在 analyze_tool 对照 K 线观察挂单带出现、增强、撤走和被穿越。
5. 点击 K 线检查最厚买卖墙、主动成交和撤单/消耗/补单归因。
6. 单日链路正确后，再构建 2026-06-01 至 2026-06-30。

## V1 局限

- 外部只能用盘口变化和逐笔成交近似区分成交、撤单、补单，无法知道订单级队列身份。
- 默认 `ETH-USDT-SWAP` 每张按 `0.1 ETH` 转换；合约规格变化时必须同步调整 `--contract-value-base`。
- 热力图颜色是展示归一化；策略条件必须使用固定的历史因果定义，不能依赖浏览器缩放后的颜色。
- V1 尚未定义交易策略，只完成可信数据底座、观察界面和回测读取接口。

## V1.3 Broad 5000-level mode

V1.3 separates the two OKX historical feeds explicitly:

- `--books-depth 400`: near-book map, default canonical heatmap 60 seconds, compact filtering.
- `--books-depth 5000`: broad map, default canonical heatmap 5 seconds, no depth-ratio thinning and no price-bin count cap.

Official 400-level and 5000-level archives may coexist under the same symbol directory. The loader now matches the requested `<depth>lv` filename exactly and never replays both files for the same day.

Inspect a 5000-level archive before building:

```text
python tools\prebuild_okx_offline_liquidity_map.py --symbol ETH-USDT-SWAP --start-date 2026-06-01 --end-date 2026-06-01 --books-depth 5000 --inspect-only
```

The inspection output includes a bounded normalized-event probe: sampled action counts, median timestamp interval, and bid/ask level counts. A smaller compressed 5000-level file is not automatically suspicious: it may consist of slower repeated full snapshots, while the 400-level file is a high-frequency incremental event stream. Trust the probe rather than file size alone.

Build the broad map:

```text
python tools\prebuild_okx_offline_liquidity_map.py --symbol ETH-USDT-SWAP --start-date 2026-06-01 --end-date 2026-06-01 --books-depth 5000 --price-step 1 --feature-seconds 1 --large-depth-ratio 0.5
```

The automatic broad profile resolves to:

- canonical strategy features: 1 second;
- canonical heatmap: 5 seconds;
- maximum distance from mid: 10%;
- no per-side price-bin cap;
- no prebuild depth-percentile thinning.

Analyze Tool defaults to the 5000-level artifact. The heatmap resolution is independent of the candlestick timeframe. `auto detail` chooses the finest multiple of the 5-second source that fits the visible range, so a full day is display-downsampled while a zoomed interval returns to 5-second detail. No extra per-timeframe artifact is created.

## V1.7 Analyze Tool：周期末列与位置对齐

Analyze Tool 默认使用 `period_end`：

- 一根时间K线对应一列热力格；
- 该列使用K线结束前最后一份已经完成的基础热力快照；
- 当前尚未结束的K线可被后续快照覆盖；
- 历史K线结束后冻结；
- 回测与未来实盘仍使用基础5秒因果状态，不因显示聚合而降低频率。

价格格采用半开区间 `[price_low, price_high)`。例如基础/显示价格格为 `$1` 时，
`2006.40` 会显示在 `[2006.00, 2007.00)`，允许的视觉离散误差不超过一个价格格。
鼠标悬停会显示该区间、方向、ETH/USD深度、订单数、源快照时间和滚动阈值。

插件运行后，工具栏会显示“对齐通过 / 对齐警告”徽章。审计同时检查：

1. 每根K线使用的源快照不晚于K线结束时间；
2. 源快照距离K线结束不超过一个基础热力周期；
3. `best_bid` 和 `best_ask` 是否落入页面对应的最高买盘格/最低卖盘格；
4. K线收盘价和同期订单簿中间价的偏差分布。

也可独立运行：

```text
python tools\audit_okx_liquidity_map_alignment.py --symbol ETH-USDT-SWAP --start 2026-06-01 --end 2026-06-02 --timeframe 15m --data-type trade_bar --books-depth 5000 --display-price-step 1
```

返回 `status=pass` 才表示本次本地数据范围通过时间与价格格对齐检查。
颜色双滑杆只改变浏览器显示，不会改变派生数据、滚动阈值或策略字段。

## V1.9：二维长期流动性墙人工审计

Analyze Tool 的离线订单簿热力图新增“墙”叠加层。运行插件后，图表工具栏会出现一个默认未勾选的 `墙` 复选框；勾选后以黄色价格—时间框显示机器识别的长期流动性簇。

检测逻辑不是逐列寻找最厚单格，而是对 5m（可调 1m/15m）热力状态进行因果二维追踪：

- 深度必须同时显著高于过去同方向、同距离带的滚动分位数，以及当前周围价格格的局部背景；
- 允许相邻深色块之间缺少少量价格格；
- 允许墙体短时间变浅或消失后在原位置附近恢复；
- 默认累计持续 120 分钟、时间覆盖率达到 65% 后才确认；
- 黄色框从确认时刻开始，而不是从首次出现时刻ย้อนหลัง补画，避免视觉结果暗示未来可知；
- 多价格簇标记为主墙，长期单格/窄价格带标记为窄墙。

该功能目前只用于墙识别人工审计，不是交易信号。先由人工核对范围、持续时间和误报，再固定检测定义并进入 5s Trade Bar / Raw Trades 的反应研究。
