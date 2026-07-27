# CoinBacktest Analyze Tool

一个放在项目根目录下的轻量 K 线分析工具，用来读取 CoinBacktest 本地数据，并通过插件把事件和连续市场状态显示到图上。

## 启动

```bash
python analyze_tool/server.py --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。`/favicon.ico` 返回 `204`，不影响功能。

## 数据源

所有数据继续通过现有 `src.data_feed` 接口读取：

- 普通 K 线：`src.data_feed.okx_loader.OKXDataLoader`
- Trade Bar：`src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader`
- Range Bar：`src.data_feed.okx_range_bar_loader.OKXRangeBarLoader`

默认勾选“只读本地缓存”，不会因为看图自动下载或重建重数据。

## 市场状态地图 V3.1
V3.1 关键纠偏：

- Sweep/Reclaim 后的恢复必须来自后续新出现的反向订单流、正向价格响应和进一步价格收复。
- 波动压缩必须先持续成熟，再由新的单向冲击真正突破已知价位。
- 突破完成不能由同一根冲击 bar 宣告，必须等待后续回踩守住或连续停留接受。
- V3 旧语义仍可通过 `ProcessMapConfig(semantic_version="v3")` 复现。


计算核心仍位于 `src/market_state`，可视化插件只负责展示。V3.1 保留有先后顺序和有效期的市场过程，并纠正 V3 中过宽的恢复与突破完成语义：

- 多头反转：卖压阶段 → 卖压吸收 → 下扫收回 → 严格买盘恢复
- 空头反转：买压阶段 → 买压吸收 → 上扫拒绝 → 严格卖盘恢复
- 向上突破：压缩成熟 → 新买方突破冲击 → 回踩/停留接受
- 向下突破：压缩成熟 → 新卖方跌破冲击 → 回抽/停留接受

每一步必须出现在前一步之后的已关闭 Bar，并且必须在固定有效期内出现；超时后过程自动失效。压缩阶段在方向选择前显示为中性，不会伪造多空冲突。

### 默认交易视图

默认只保留三层：

- 历史结构：上涨结构已确认 / 平衡结构 / 下跌结构已确认。它只是后验结构，不是未来方向许可。
- 当前阶段：延续、回撤/反弹、整理、衰减和冲击。
- 多阶段过程：显示当前过程走到哪一步，以及是否已经过期或完成。

点击 K 线可以看到过程完成度、剩余有效期、历史方向概率、相对基线增量和样本支持。概率不足 `30` 个已结算历史样本时不显示。

### 概率因果性

- 阶段完成概率只使用已经完成或已经超时的旧过程。
- 方向概率只有等旧过程的未来观察窗口完整结束后，才加入历史统计。
- 当前过程永远不能用自己的未来结果训练当前显示概率。
- 概率是条件统计，不是开仓、平仓或仓位建议。

### 研究视图

研究视图保留价格结构、波动、订单流、冲击吸收、关键位置、过程完成度和历史概率增量等完整诊断轨道。普通 OHLCV 缺少主动成交字段时，订单流过程会关闭，不会根据 K 线颜色伪造主动买卖方向。

### 因果规则

- 所有滚动特征只使用当前及历史已关闭数据。
- 滚动支撑/阻力使用 `high/low.shift(1)`。
- 普通 K 线和 Trade Bar 的状态在 `timestamp + timeframe` 才可见。
- Range Bar 使用 `end_ts` 作为可用时间。
- 多阶段过程必须严格按后续已关闭 Bar 顺序推进，禁止同 Bar 连跳多个阶段。

## 验证

```bash
python analyze_tool/selftest.py
```

```bash
python -m pytest tests/market_state tests/test_analyze_tool_market_state.py tests/test_analyze_tool_panic_episode.py -q
```

```bash
node --check analyze_tool/static/app.js
```

```bash
python tools/check_import_boundaries.py
```

## 推定清算热力图 V1

先准备公开衍生品数据：

```bash
python tools\prebuild_okx_liquidation_inputs.py --symbol ETH-USDT-SWAP --start-date 2026-06-01 --end-date "2026-06-30 23:59:59" --oi-period 5m --mark-timeframe 1m
```

然后选择 `Trade Bar / 1m / 推定清算热力图 V1`。默认只显示红色多头潜在清算区、青色空头潜在清算区和简洁状态卡，不显示额外指标线。

该图是公开 OI、Funding、Mark、清算事件与订单流生成的透明估算，不是交易所账户仓位，也不是 CoinGlass 数据。

## 离线订单簿流动性热力图 V1.5 Matrix

只使用本地历史 Books 与同日 Raw Trades，不启动实时采集。宽范围地图使用 5000 档：

```bash
python tools\prebuild_okx_offline_liquidity_map.py --symbol ETH-USDT-SWAP --start-date 2026-06-01 --end-date 2026-06-01 --books-depth 5000 --inspect-only
```

确认字段后构建：

```bash
python tools\prebuild_okx_offline_liquidity_map.py --symbol ETH-USDT-SWAP --start-date 2026-06-01 --end-date 2026-06-01 --books-depth 5000 --price-step 1 --feature-seconds 1 --large-depth-ratio 0.5
```

选择 `离线订单簿流动性热力图 V1.5 Matrix`。热力图使用真正的“价格行 × 时间列”矩阵：热力格按自身开始/结束时间绘制，不再把 5 秒格压到 1m/15m K 线索引上，因此不会因为多个热力格重叠而染成整片色块。热力时间精度与 K 线周期相互独立；显示层支持厚墙显著性、原始深度 Q99、手动封顶和同刻相对厚度。

图表交互：

- 图内滚轮：横向时间缩放；
- 图内普通拖动：仅左右平移时间；
- `Shift +` 图内拖动：同时平移时间与价格；
- 右侧价格轴滚轮/拖动：以鼠标落点对应价格为固定锚点进行纵向缩放；
- `Shift +` 拖动价格轴：上下平移价格视窗；
- 双击价格轴或点击“价格自适应”：恢复当前可视区自动价格范围；
- 工具栏支持全部复位、跳到最早、跳到最新和按日期时间居中跳转。

Raw Books 日期是 UTC 日；项目默认 `+8` 显示，所以 2026-06-01 原始日主要显示在 2026-06-01 08:00 至 2026-06-02 08:00。完整说明见 `docs/OFFLINE_ORDERBOOK_LIQUIDITY_MAP_V1_RUNBOOK.md`。
