# SOXL ICT MSS R02 - Alpaca .env Hotfix

## Problem
`AlpacaStockLoader` only read credentials through `os.getenv()`. CoinBacktest's existing `config/env_loader.py` reads the repository-root `.env` into a dict and does not export those values into `os.environ`, so credentials present only in `.env` were invisible to the loader.

## Fix
- Do not modify `config/env_loader.py`.
- `AlpacaStockLoader` now resolves credentials in this order:
  1. explicit constructor arguments;
  2. process environment variables;
  3. CoinBacktest repository-root `.env` via `config.env_loader.load_env_config()`.
- Supported names: `APCA_API_KEY_ID`, `ALPACA_API_KEY_ID`, `ALPACA_API_KEY`; `APCA_API_SECRET_KEY`, `ALPACA_API_SECRET_KEY`, `ALPACA_SECRET_KEY`.

## Validation
- `PYTHONPATH=. pytest -q tests/data_feed/test_alpaca_stock_loader.py` -> 4 passed.
- Integration smoke with a temporary repository-root `.env` confirmed both credentials are loaded without exporting them into `os.environ`.
