from __future__ import annotations

import json

import pandas as pd

from analyze_tool.plugin_api import PluginRunResult
from analyze_tool.plugins.orderbook_liquidity_heatmap import _compact_heatmap_payload


def test_compact_heatmap_payload_preserves_cells_without_repeated_objects() -> None:
    frame = pd.DataFrame(
        {
            "bar_start_ms": [1, 1, 2],
            "bar_start": [pd.Timestamp("2026-01-01 08:00:00"), pd.Timestamp("2026-01-01 08:00:00"), pd.Timestamp("2026-01-01 08:15:00")],
            "bar_end": [pd.Timestamp("2026-01-01 08:15:00"), pd.Timestamp("2026-01-01 08:15:00"), pd.Timestamp("2026-01-01 08:30:00")],
            "source_bucket_start_ms": [1000, 1000, 2000],
            "source_bucket_end_ms": [1999, 1999, 2999],
            "source_lag_ms": [1, 1, 1],
            "side_code": [1, -1, 1],
            "side": ["bid", "ask", "bid"],
            "price_low": [1800.0, 1801.0, 1802.0],
            "intensity": [0.01, 0.5, 1.0],
            "display_depth": [0.1234567, 5.0, 10.0],
            "display_order_count": [1, 2, 3],
            "is_large_rolling": [False, True, True],
        }
    )

    payload = _compact_heatmap_payload(
        frame,
        depth_unit="base",
        color_mode="single",
        display_price_step=1.0,
    )

    assert payload["v"] == 1
    assert payload["starts"] == ["2026-01-01 08:00:00", "2026-01-01 08:15:00"]
    assert payload["c"] == [0, 0, 1]
    assert payload["p"] == [1801, 1800, 1802]
    assert payload["i"] == [5000, 100, 10000]
    assert payload["s"] == [-1, 1, 1]
    assert payload["d"][1] == 0.123457
    assert payload["l"] == [1, 0, 1]

    result = PluginRunResult(markers=[], heatmap_compact=payload).as_dict()
    assert result["heatmap"] == []
    assert result["heatmap_compact"]["c"] == [0, 0, 1]
    json.dumps(result, allow_nan=False)


def test_large_json_response_uses_valid_gzip() -> None:
    import gzip
    import io

    from analyze_tool.server import json_response

    class Handler:
        headers = {"Accept-Encoding": "gzip, deflate"}

        def __init__(self) -> None:
            self.status = None
            self.response_headers: dict[str, str] = {}
            self.wfile = io.BytesIO()

        def send_response(self, status: int) -> None:
            self.status = status

        def send_header(self, name: str, value: str) -> None:
            self.response_headers[name] = value

        def end_headers(self) -> None:
            pass

    handler = Handler()
    payload = {"ok": True, "values": list(range(50_000))}
    json_response(handler, payload)

    assert handler.status == 200
    assert handler.response_headers["Content-Encoding"] == "gzip"
    body = handler.wfile.getvalue()
    decoded = json.loads(gzip.decompress(body).decode("utf-8"))
    assert decoded == payload
    assert int(handler.response_headers["Content-Length"]) == len(body)
