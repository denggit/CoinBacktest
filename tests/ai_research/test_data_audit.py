from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_standalone_data_audit_is_superseded() -> None:
    entrypoint = REPO_ROOT / "research" / "eth_ai_trading" / "01_audit_trade_data.py"
    text = entrypoint.read_text(encoding="utf-8")
    assert "superseded standalone R01 audit" in text
    assert "01_trades_only_supervised_baseline.py" in text


def test_r01_active_path_uses_public_data_feed_only() -> None:
    root = REPO_ROOT / "src" / "ai_research" / "trades_baseline"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "OKXTradeBarLoader" in text
    assert "sqlite3" not in text
    assert "zipfile" not in text
    assert "OKXTickLoader" not in text
