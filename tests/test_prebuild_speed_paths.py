from __future__ import annotations

import pandas as pd

from src.data_feed.okx_tick_loader import OKXTickLoader


def test_tick_loader_minimal_mode_preserves_required_values_without_raw_json() -> None:
    raw = pd.DataFrame(
        {
            "ts": [1_700_000_000_000, 1_700_000_000_100],
            "px": [3000.0, 3000.5],
            "sz": [2.0, 3.0],
            "side": ["buy", "sell"],
            "tradeId": ["a", "b"],
            "extra_payload": ["large", "payload"],
        }
    )
    loader = OKXTickLoader(symbol="ETH-USDT-SWAP")
    full = loader._normalize_trades(raw, minimal=False)
    minimal = loader._normalize_trades(raw, minimal=True)

    assert list(minimal.columns) == ["ts_ms", "price", "size", "side"]
    assert minimal.to_dict("records") == full[["ts_ms", "price", "size", "side"]].to_dict("records")
    assert "raw_json" in full.columns
    assert "raw_json" not in minimal.columns
