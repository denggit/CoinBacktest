# CoinBacktest Analyze Tool

一个放在项目根目录下的轻量 K 线分析工具，用来快速查看 CoinBacktest 本地数据，并通过插件把事件标记到图上。

## 启动

```bash
python analyze_tool/server.py --host 127.0.0.1 --port 8765
```

浏览器打开：

```text
http://127.0.0.1:8765
```

## 当前能力

- 数据源通过现有 `src.data_feed`：
  - 普通 K 线：`src.data_feed.okx_loader.OKXDataLoader`
  - Trade Bar：`src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader`
  - Range Bar：`src.data_feed.okx_range_bar_loader.OKXRangeBarLoader`
- 前端 K 线图：
  - 鼠标滚轮缩放
  - 按住左右拖动
  - 鼠标悬停显示 OHLCV
  - 底部显示 volume
  - 下方面板显示 Trade Bar / Range Bar 更多字段
- 插件接口：
  - 内置 `长影线标记` 插件
  - 以后新增插件只需要实现 `AnalyzePlugin` 协议并注册到 `analyze_tool/plugins/__init__.py`

## 本地缓存说明

默认勾选“只读本地缓存，不自动下载/补数据”。这样不会因为看图触发大量下载或重建。

如果本地 DB 没覆盖当前时间段，可以取消这个勾选，让 loader 自己按项目现有逻辑补数据。重数据建议优先使用项目里的预构建命令提前生成：

```bash
python tools/prebuild_okx_trade_bars.py --symbol ETH-USDT-SWAP --timeframe 1m --start-date 2023-01-01 --end-date 2026-06-30
python tools/prebuild_okx_range_bars.py --symbol ETH-USDT-SWAP --range-pcts 0.0015 0.0020 0.0025 --start-date 2023-01-01 --end-date 2026-06-30
```

## 新增插件示例

```python
from analyze_tool.plugin_api import Marker, PluginParam, PluginRunResult

class MyPlugin:
    plugin_id = "my_plugin"
    name = "我的事件标记"
    description = "描述这个插件做什么"
    params = [PluginParam(name="color", label="颜色", kind="color", default="#facc15")]

    def run(self, df, params=None):
        markers = []
        for idx, row in df.iterrows():
            if row["close"] > row["open"]:
                markers.append(Marker(timestamp=idx.strftime("%Y-%m-%d %H:%M:%S"), label="上涨", color=params.get("color", "#facc15")))
        return PluginRunResult(markers=markers, summary={"matched": len(markers), "input_rows": len(df)})
```

然后在 `analyze_tool/plugins/__init__.py` 里注册：

```python
registry.register(MyPlugin())
```

## 验证

```bash
python analyze_tool/selftest.py
python -m py_compile analyze_tool/server.py analyze_tool/data_service.py analyze_tool/plugin_api.py analyze_tool/plugins/long_shadow.py analyze_tool/selftest.py
python tools/check_import_boundaries.py
python -m pytest tests/test_import_boundaries.py -q
```
