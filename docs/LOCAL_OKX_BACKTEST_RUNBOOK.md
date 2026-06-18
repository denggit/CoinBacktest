# Local OKX Historical Data A1 Research Replay Runbook

This runbook describes the offline research flow for GlacierPulse.

Goal:

```text
OKX official historical data
  -> local raw files
  -> local replay through the same research components used by main.py
  -> reports/backtests/<run-name>/research_events.jsonl + summary.json
  -> reports/backtests/<run-name>/research/phase1_candidates.jsonl
  -> reports/backtests/<run-name>/research/a1_reaction_events.jsonl
  -> optional unified research reports + zip
```

This is not a signal-trading backtester. It does not start `main.py`, does not connect WebSocket, does not use `IcebergTrader`, and does not simulate open/close orders.

The local replay path mirrors the current `main.py` research stage:

```text
trade tick -> MarketContext.apply_trade() -> A1AbsorptionEngine.on_trade()
book update -> MarketContext.apply_book_delta() -> A1AbsorptionEngine.on_book_update()
returned A1 research event -> local handle_research_event()
```

A1 iceberg events are recorded for research. They are not sent to execution.

## 1. Download OKX official files

OKX official historical data page says trade history is tick-level from September 2021 onwards, and order book is high-resolution L2 data from March 2023 onwards.

### 1.1 Trades

For OKX trade history, use this CDN URL template:

```text
https://www.okx.com/cdn/okex/traderecords/trades/daily/{yyyymmdd}/{symbol}-trades-{date}.zip
```

For `ETH-USDT-SWAP`, run:

```bash
python tools/download_okx_historical_data.py \
  --kind trades \
  --symbol ETH-USDT-SWAP \
  --start-date 2026-04-01 \
  --end-date 2026-05-25 \
  --url-template 'https://www.okx.com/cdn/okex/traderecords/trades/daily/{yyyymmdd}/{symbol}-trades-{date}.zip'
```

Files will be saved to:

```text
data/okx/raw/trades/ETH-USDT-SWAP/
```

### 1.2 Books

For OKX historical L2 order book data, use export-link mode. The command shape is close to the trades command, but it does not use a static URL template:

```bash
python tools/download_okx_historical_data.py \
  --kind books \
  --symbol ETH-USDT-SWAP \
  --start-date 2026-04-01 \
  --end-date 2026-05-25
```

For 5000-depth books:

```bash
python tools/download_okx_historical_data.py \
  --kind books \
  --symbol ETH-USDT-SWAP \
  --start-date 2025-05-01 \
  --end-date 2026-05-01 \
  --books-depth 5000
```

Books export mode requests official OKX historical-data export links and downloads the returned files into:

```text
data/okx/raw/books/ETH-USDT-SWAP/
```

If OKX returns zero download links, the tool fails clearly. In that case, use the OKX historical-data website to export the book files and then pass the official file URLs with `--url` or `--manifest`.

## 2. Run local A1 research replay

With trades and books:

```bash
python tools/backtest_local_data.py \
  --symbol ETH-USDT-SWAP \
  --trades-dir data/okx/raw/trades/ETH-USDT-SWAP \
  --books-dir data/okx/raw/books/ETH-USDT-SWAP \
  --run-name eth_swap_1y_a1_research
```

Output:

```text
reports/backtests/eth_swap_1y_a1_research/
  research_events.jsonl
  summary.json
  research/
    phase1_candidates.jsonl
    a1_reaction_events.jsonl
  runtime_state/
    a1_dynamic_params.json
```

The output meaning is:

```text
research_events.jsonl      = events returned by A1AbsorptionEngine and handled like main.py research events
phase1_candidates.jsonl    = Phase1 truth candidate recorder output
a1_reaction_events.jsonl   = A1 reaction research recorder output
summary.json               = replay statistics and safety flags
```

Optional unified research reports:

```bash
python tools/backtest_local_data.py \
  --symbol ETH-USDT-SWAP \
  --trades-dir data/okx/raw/trades/ETH-USDT-SWAP \
  --books-dir data/okx/raw/books/ETH-USDT-SWAP \
  --run-name eth_swap_1y_a1_research \
  --generate-reports \
  --kline data/okx/raw/klines/ETH-USDT-SWAP/ETH-USDT-SWAP-1m.csv
```

`--generate-reports` is replay post-processing only. It runs after replay finishes and after the local research runtime is closed. It does not change replay timestamps, trades/books merge order, A1 runtime decisions, book cleaning, or execution behavior.

Report CLI parameters:

```text
--generate-reports        Enable unified research report generation after replay. Default: disabled.
--kline PATH              Required when --generate-reports is enabled. 1m kline CSV for a1_edge and zone_truth.
--report-run-name NAME    Optional report run name under reports/. Default: backtests/<run-name>/research_reports.
--report-min-sample N     Minimum sample passed to the report tools. Default: 30.
--report-timezone TZ      Timezone passed only to report generation and kline parsing. Default: Asia/Shanghai.
```

Default report outputs for `--run-name eth_swap_1y_a1_research`:

```text
reports/backtests/eth_swap_1y_a1_research/research_reports/
reports/backtests/eth_swap_1y_a1_research/research_reports.zip
```

`summary.json` includes a `report_generation` object with status, input paths, report directory, zip path, kline path, and any empty research JSONL inputs created for report compatibility. Missing `research/phase1_candidates.jsonl` or `research/a1_reaction_events.jsonl` files are created as empty files; events are never fabricated.

Trades-only smoke test:

```bash
python tools/backtest_local_data.py \
  --symbol ETH-USDT-SWAP \
  --trades-dir data/okx/raw/trades/ETH-USDT-SWAP \
  --run-name eth_swap_trades_only_smoke \
  --allow-missing-books \
  --max-events 100000
```

Important: A1 iceberg settlement depends on order book updates. Trades-only mode is only a parser/replay smoke test, not a full A1 evaluation.

## 3. Supported input file types

`tools/backtest_local_data.py` supports:

```text
.csv
.jsonl
.ndjson
.json
.gz
.zip
```

The parser tries common OKX-style fields:

Trades:

```text
instId / inst_id / symbol
px / price
sz / size / qty / amount
side
ts / timestamp / time
```

Books:

```text
instId / inst_id / symbol
bids / bid / bid_levels
asks / ask / ask_levels
ts / timestamp / time
```

For `ETH-USDT-SWAP`, the default contract multiplier is `0.1`, matching the project assumption that one OKX ETH swap contract is 0.1 ETH.

## 4. Recommended first validation

Start with one day of trades:

```bash
python tools/download_okx_historical_data.py \
  --kind trades \
  --symbol ETH-USDT-SWAP \
  --start-date 2025-05-01 \
  --end-date 2025-05-01 \
  --url-template 'https://www.okx.com/cdn/okex/traderecords/trades/daily/{yyyymmdd}/{symbol}-trades-{date}.zip'
```

Then try one day of books:

```bash
python tools/download_okx_historical_data.py \
  --kind books \
  --symbol ETH-USDT-SWAP \
  --start-date 2025-05-01 \
  --end-date 2025-05-01
```

After both files are available, run full A1 research replay:

```bash
python tools/backtest_local_data.py \
  --symbol ETH-USDT-SWAP \
  --trades-dir data/okx/raw/trades/ETH-USDT-SWAP \
  --books-dir data/okx/raw/books/ETH-USDT-SWAP \
  --run-name eth_swap_one_day_a1_research_check \
  --progress-every 50000
```

Then inspect:

```text
reports/backtests/eth_swap_one_day_a1_research_check/summary.json
reports/backtests/eth_swap_one_day_a1_research_check/research_events.jsonl
reports/backtests/eth_swap_one_day_a1_research_check/research/phase1_candidates.jsonl
reports/backtests/eth_swap_one_day_a1_research_check/research/a1_reaction_events.jsonl
```

If `malformed_rows` is high, inspect the raw OKX file format and adjust field mapping in `normalize_trade()` or `normalize_book()`.

## 5. Design rule

Keep this research path separate from live trading:

```text
main.py                         = live WebSocket research runtime
backtest_local_data.py           = local historical research replay runtime
download_okx_historical_data.py  = official file acquisition
```

Do not import or instantiate `IcebergTrader` in the local replay entry. Do not add position, PnL, or signal-trading simulation into this tool. If future research needs virtual-position evaluation, enable it explicitly through research configuration and keep it separate from real execution.
