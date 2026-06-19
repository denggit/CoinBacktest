# OKX 本地数据预构建使用说明

本文记录 CoinBacktest 里三个预构建工具的用途、推荐顺序和常用命令，避免以后忘记怎么跑。

三个工具分别是：

1. `tools/prebuild_okx_trade_bars.py`：把 OKX trades 聚合成普通时间 bar，例如 1m、5m。
2. `tools/prebuild_okx_range_bars.py`：把 OKX trades 聚合成 Range Bar，例如 0.15%、0.20%、0.25%。
3. `tools/prebuild_okx_range_footprints.py`：把每根 Range Bar 内部成交按价格桶聚合成 Footprint。

默认标的：`ETH-USDT-SWAP`。

默认 raw trades 目录：

```text
data/okx/raw/trades/ETH-USDT-SWAP/
```

默认要求 raw trades 文件名类似：

```text
ETH-USDT-SWAP-trades-2022-01-01.zip
ETH-USDT-SWAP-trades-2022-01-02.zip
...
```

如果本地 raw trades 缺失，loader 会尝试按 OKX 官方 URL template 下载；但正式批量预构建前，建议先用 `tools/download_okx_historical_data.py` 把 raw trades 下载好，避免预构建时一边下载一边聚合导致速度慢、网络不稳定或触发限流。

---

## 一、总原则

### 1. 不要并行跑

不要同时开多个窗口跑这些 prebuild。

错误方式：

```text
窗口 1 跑 prebuild_okx_trade_bars.py
窗口 2 跑 prebuild_okx_range_bars.py
窗口 3 跑 prebuild_okx_range_footprints.py
```

原因：

- 会同时读取同一批大 zip，磁盘 IO 互相抢。
- 如果某天 raw trades 缺失，可能同时触发 OKX 下载。
- Windows 下可能出现 socket/端口问题。
- SQLite 多写入也没有必要。

推荐方式：

```text
一个跑完，再跑下一个。
```

---

### 2. 推荐执行顺序

推荐顺序：

```text
1. 先预构建普通 trade bars，例如 1m、5m
2. 再预构建 Range Bars
3. 最后预构建 Range Footprints
```

原因：

- trade bars 是常规时间 K，很多中低频/高频初筛都会用。
- range bars 是基础事件 bar。
- footprint 是基于 range bar 内部成交结构的二级数据，数据量最大，最后跑。

---

### 3. 断点续跑

三个工具默认都有 coverage 机制。

已经完整构建过的日期，下次会跳过。

所以如果中途 Ctrl+C 或报错，可以直接用同一条命令重跑。

示例：

```bat
python tools\prebuild_okx_range_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.002 0.0025
```

如果前面已经构建过一部分，重跑时会自动跳过已缓存日期，只补缺失日期。

---

### 4. 什么时候用 `--force-rebuild`

如果只是中断后继续跑，不需要 `--force-rebuild`。

以下情况建议加：

- 之前用旧版本工具构建过，担心 coverage 状态不干净。
- 改了字段逻辑、range bar 逻辑、large trade 阈值、contract value。
- 想完整重建一张表。

示例：

```bat
python tools\prebuild_okx_range_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.002 0.0025 --force-rebuild
```

注意：`--force-rebuild` 会重建目标日期范围内的缓存，时间会更久。

---

## 二、预构建普通 Trade Bars

工具：

```text
tools/prebuild_okx_trade_bars.py
```

默认 DB：

```text
data/okx_trade_bars.db
```

用途：

```text
从 OKX trades zip 聚合出 1m、5m、15m、30m、1H、4H、1D 等普通时间 bar。
```

常用命令：

```bat
python tools\prebuild_okx_trade_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --timeframes 1m 5m
```

如果只想预构建 1m：

```bat
python tools\prebuild_okx_trade_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --timeframes 1m
```

如果要重建：

```bat
python tools\prebuild_okx_trade_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --timeframes 1m 5m --force-rebuild
```

常用参数：

```text
--timeframes 1m 5m       要构建的时间周期
--chunksize 300000       每次读取 raw trades 的行数
--force-rebuild          强制重建
--utc-timestamps         使用 UTC-naive 时间戳保存，不跟随 config.loader.TIMEZONE 偏移
--sleep-sec 1            每天之间暂停 1 秒，降低 IO/API 压力
```

---

## 三、预构建 Range Bars

工具：

```text
tools/prebuild_okx_range_bars.py
```

默认 DB：

```text
data/okx_range_bars.db
```

默认三组 range：

```text
0.15% -> r0015
0.20% -> r0020
0.25% -> r0025
```

表名：

```text
ETH_USDT_SWAP_range_bars_r0015
ETH_USDT_SWAP_range_bars_r0020
ETH_USDT_SWAP_range_bars_r0025
```

优先推荐先跑 0.20%：

```bat
python tools\prebuild_okx_range_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.002
```

三组一起跑：

```bat
python tools\prebuild_okx_range_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.002 0.0025
```

强制重建三组：

```bat
python tools\prebuild_okx_range_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.002 0.0025 --force-rebuild
```

### one-pass multi-range 说明

`prebuild_okx_range_bars.py` 已经支持 one-pass multi-range。

也就是说，这条命令：

```bat
python tools\prebuild_okx_range_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.002 0.0025
```

不是把 tick 读三遍。

实际流程是：

```text
读一个 raw trades chunk
    -> 同时喂给 r0015 RangeBarBuilder
    -> 同时喂给 r0020 RangeBarBuilder
    -> 同时喂给 r0025 RangeBarBuilder
    -> 分别写入三张 SQLite 表
```

日志里应该看到：

```text
[RANGE-BAR-PREBUILD-START] ... ranges=r0015,r0020,r0025 ... mode=one_pass_multi_range
[RANGE-BAR-MULTI-DAY-START] ... utc_day=2022-01-01 ...
[RANGE-BAR-MULTI-DAY-DONE] ... rows_by_range=r0015:xxx,r0020:xxx,r0025:xxx ...
```

如果某个 range 的 rows 是 0，不一定代表没有生成 bar，也可能是那一天已经缓存过，本次没有重复写入。

---

## 四、预构建 Range Footprints

工具：

```text
tools/prebuild_okx_range_footprints.py
```

默认 DB：

```text
data/okx_range_footprints.db
```

默认 price bucket：

```text
price_step = 1 USDT
```

默认表名：

```text
ETH_USDT_SWAP_range_footprint_r0015_step1
ETH_USDT_SWAP_range_footprint_r0020_step1
ETH_USDT_SWAP_range_footprint_r0025_step1
```

优先推荐先跑 0.20% + step 1：

```bat
python tools\prebuild_okx_range_footprints.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.002 --price-step 1
```

三组一起跑：

```bat
python tools\prebuild_okx_range_footprints.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.002 0.0025 --price-step 1
```

如果以后想试更细价格桶：

```bat
python tools\prebuild_okx_range_footprints.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.002 --price-step 0.5
```

对应表名会变成：

```text
ETH_USDT_SWAP_range_footprint_r0020_step0_5
```

### one-pass multi-range 说明

`prebuild_okx_range_footprints.py` 也支持 one-pass multi-range。

这条命令：

```bat
python tools\prebuild_okx_range_footprints.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.002 0.0025 --price-step 1
```

实际是：

```text
读一个 raw trades chunk
    -> 同时生成 r0015 footprint
    -> 同时生成 r0020 footprint
    -> 同时生成 r0025 footprint
```

不是读三遍 tick。

日志里应该看到：

```text
[RANGE-FOOTPRINT-PREBUILD-START] ... ranges=r0015,r0020,r0025 ... mode=one_pass_multi_range
[RANGE-FOOTPRINT-MULTI-DAY-START] ... utc_day=2022-01-01 ...
[RANGE-FOOTPRINT-MULTI-DAY-DONE] ... rows_by_range=r0015:xxx,r0020:xxx,r0025:xxx ...
```

---

## 五、推荐完整执行流程

### 第一步：普通 trade bars

```bat
python tools\prebuild_okx_trade_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --timeframes 1m 5m
```

### 第二步：Range Bars，先跑 0.20%

```bat
python tools\prebuild_okx_range_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.002
```

### 第三步：Range Footprints，先跑 0.20% + step 1

```bat
python tools\prebuild_okx_range_footprints.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.002 --price-step 1
```

### 第四步：确认 0.20% 有价值后，再补 0.15% 和 0.25%

```bat
python tools\prebuild_okx_range_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.0025
```

```bat
python tools\prebuild_okx_range_footprints.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.0025 --price-step 1
```

如果你已经确定三组都要，直接这样也可以：

```bat
python tools\prebuild_okx_range_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.002 0.0025
```

```bat
python tools\prebuild_okx_range_footprints.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.002 0.0025 --price-step 1
```

---

## 六、常见问题

### 1. 为什么 rows_by_range 某些 range 是 0？

例如：

```text
rows_by_range=r0015:0,r0020:0,r0025:75
```

这通常表示：

```text
r0015 / r0020 这一天已经缓存过，所以本次没有重复写入。
r0025 这一天没缓存，所以本次新增写入 75 根。
```

如果想全部重建，加：

```text
--force-rebuild
```

---

### 2. 可以中途停止吗？

可以。

coverage 是按 UTC day 标记的。完整跑完某天后才会标记成功。

中途停止后，直接用同一条命令重跑即可。

---

### 3. 是否要先下载 raw trades？

强烈建议先下载。

预构建工具虽然可以在缺 raw zip 时自动下载，但一边下载一边聚合会更慢，也更容易遇到网络问题。

raw trades 建议放在：

```text
data/okx/raw/trades/ETH-USDT-SWAP/
```

---

### 4. 能不能同时跑 bars 和 footprints？

不建议。

虽然它们写入不同 DB，但会抢同一批 raw trades zip 的读取 IO。如果 raw 缺失，还可能同时触发下载。

推荐：

```text
先 bars，后 footprints。
```

---

### 5. 什么时候用 price_step = 0.5？

先用：

```text
price_step = 1
```

如果 footprint 太粗，看不出价格层级主动买卖差异，再试：

```text
price_step = 0.5
```

但是 `0.5` 会让 footprint 行数更多，DB 更大，查询更慢。

---

## 七、Python 读取示例

### 读取 Range Bars

```python
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader

loader = OKXRangeBarLoader(symbol="ETH-USDT-SWAP", range_pct=0.002)
bars = loader.fetch_data_by_date_range("2022-01-01", "2026-06-15")
print(bars.tail())
```

### 读取 Range Footprints

```python
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader

loader = OKXRangeFootprintLoader(symbol="ETH-USDT-SWAP", range_pct=0.002, price_step=1)
fp = loader.fetch_data_by_date_range("2022-01-01", "2026-06-15")
print(fp.tail())
```

### 读取普通 Trade Bars

```python
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader

loader = OKXTradeBarLoader(symbol="ETH-USDT-SWAP", timeframe="1m")
bars = loader.fetch_data_by_date_range("2022-01-01", "2026-06-15")
print(bars.tail())
```

---

## 八、推荐命令备忘

最常用的一套：

```bat
python tools\prebuild_okx_trade_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --timeframes 1m 5m
```

```bat
python tools\prebuild_okx_range_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.002
```

```bat
python tools\prebuild_okx_range_footprints.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.002 --price-step 1
```

三组 Range 一次性补齐：

```bat
python tools\prebuild_okx_range_bars.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.002 0.0025
```

```bat
python tools\prebuild_okx_range_footprints.py --symbol ETH-USDT-SWAP --start-date 2022-01-01 --end-date 2026-06-15 --range-pcts 0.0015 0.002 0.0025 --price-step 1
```
