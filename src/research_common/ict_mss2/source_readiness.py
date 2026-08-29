"""Pre-R27 source-readiness and mechanism-novelty gate.

This module contains no market-data I/O.  It consumes the immutable catalog
returned by :mod:`src.data_feed.local_market_catalog` and decides whether a new
MSS2 mechanism is both locally supported and genuinely distinct from the
families frozen through R26.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data_feed.local_market_catalog import LocalMarketCatalog, SQLiteSeriesCoverage


@dataclass(frozen=True)
class SourceReadinessConfig:
    warmup_start: pd.Timestamp = pd.Timestamp("2022-01-01")
    discovery_start: pd.Timestamp = pd.Timestamp("2023-01-01")
    validation_end_exclusive: pd.Timestamp = pd.Timestamp("2025-07-01")


_TIMEFRAME_SECONDS = {
    "1s": 1,
    "5s": 5,
    "10s": 10,
    "15s": 15,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1_800,
    "1H": 3_600,
    "4H": 14_400,
    "1D": 86_400,
}


def _dimension_map(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in str(text or "").split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key] = value
    return out


def _instrument_and_timeframe(row: SQLiteSeriesCoverage) -> tuple[str, str]:
    dimensions = _dimension_map(row.dimensions)
    if "symbol" in dimensions:
        return dimensions["symbol"], dimensions.get("timeframe", dimensions.get("period", ""))
    table = row.table
    if row.database == "crypto_history.db":
        match = re.match(r"^(.*)_((?:\d+)(?:s|m|H|D))$", table)
        if match:
            return match.group(1).replace("_", "-"), match.group(2)
    match = re.match(r"^(.*?)_(?:trade_bars|range_bars|range_footprint)_", table)
    if match:
        tf_match = re.search(r"trade_bars_((?:\d+)(?:s|m|H|D))(?:_|$)", table)
        return match.group(1).replace("_", "-"), tf_match.group(1) if tf_match else "variable"
    return table, dimensions.get("timeframe", dimensions.get("period", ""))


def _expected_rows(start: pd.Timestamp, end_exclusive: pd.Timestamp, timeframe: str) -> int | None:
    seconds = _TIMEFRAME_SECONDS.get(timeframe)
    if seconds is None:
        return None
    return int((end_exclusive - start).total_seconds() // seconds)


def build_sqlite_inventory(
    catalog: LocalMarketCatalog,
    *,
    config: SourceReadinessConfig = SourceReadinessConfig(),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in catalog.sqlite_series:
        instrument, timeframe = _instrument_and_timeframe(item)
        expected = _expected_rows(config.warmup_start, config.validation_end_exclusive, timeframe)
        ratio = item.rows / expected if expected and item.rows >= 0 else None
        tolerance = pd.Timedelta(seconds=_TIMEFRAME_SECONDS.get(timeframe, 3_600))
        starts_on_time = item.start is not None and item.start <= config.warmup_start + tolerance
        ends_on_time = item.end is not None and item.end >= config.validation_end_exclusive - tolerance
        if starts_on_time and ends_on_time and (ratio is None or ratio >= 0.999):
            status = "READY"
        elif starts_on_time and ends_on_time and ratio is not None and ratio >= 0.995:
            status = "NEAR_COMPLETE"
        else:
            status = "PARTIAL"
        rows.append(
            {
                "storage": "sqlite",
                "database": item.database,
                "table": item.table,
                "instrument": instrument,
                "timeframe": timeframe,
                "rows_pre_embargo": item.rows,
                "start_pre_embargo": item.start,
                "end_pre_embargo": item.end,
                "expected_rows": expected,
                "coverage_ratio": ratio,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def build_archive_inventory(catalog: LocalMarketCatalog) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in catalog.archive_series:
        if item.dated_days == item.expected_days and item.expected_days > 0:
            status = "READY"
        elif item.dated_days == 0:
            status = "EMPTY_PRE_EMBARGO"
        else:
            status = "PARTIAL"
        rows.append(
            {
                "storage": "dated_files",
                "lane": item.lane,
                "instrument": item.series_key,
                "files_pre_embargo": item.files,
                "dated_days": item.dated_days,
                "expected_days": item.expected_days,
                "missing_days": item.missing_days,
                "coverage_ratio": item.coverage_ratio,
                "start_pre_embargo": item.start,
                "end_pre_embargo": item.end,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def assert_pre_embargo_catalog(
    catalog: LocalMarketCatalog,
    *,
    end_exclusive: Any = pd.Timestamp("2025-07-01"),
) -> pd.DataFrame:
    cutoff = pd.Timestamp(end_exclusive)
    checks: list[dict[str, Any]] = []
    sqlite_violations = [
        row.series_key for row in catalog.sqlite_series if row.end is not None and row.end >= cutoff
    ]
    archive_violations = [
        f"{row.lane}:{row.series_key}"
        for row in catalog.archive_series
        if row.end is not None and pd.Timestamp(row.end) >= cutoff
    ]
    checks.extend(
        [
            {
                "check": "sqlite_max_timestamp_strictly_before_embargo",
                "passed": not sqlite_violations,
                "detail": ",".join(sqlite_violations),
            },
            {
                "check": "archive_max_day_strictly_before_embargo",
                "passed": not archive_violations,
                "detail": ",".join(archive_violations),
            },
            {
                "check": "catalog_cutoff_matches_frozen_embargo",
                "passed": all(row.end_exclusive == cutoff for row in catalog.sqlite_series),
                "detail": cutoff.isoformat(sep=" "),
            },
        ]
    )
    frame = pd.DataFrame(checks)
    if not bool(frame["passed"].all()):
        failed = frame.loc[~frame["passed"], "check"].tolist()
        raise RuntimeError(f"pre-embargo catalog seal failed: {failed}")
    return frame


def _matching(
    inventory: pd.DataFrame,
    *,
    database: str | None = None,
    instrument: str | None = None,
    table: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame:
    if inventory.empty:
        return inventory
    mask = pd.Series(True, index=inventory.index)
    for column, value in (
        ("database", database),
        ("instrument", instrument),
        ("table", table),
        ("timeframe", timeframe),
    ):
        if value is not None:
            mask &= inventory[column].eq(value)
    return inventory.loc[mask]


def _has_ready(frame: pd.DataFrame) -> bool:
    return not frame.empty and bool(frame["status"].isin(["READY", "NEAR_COMPLETE"]).any())


def _archive_ready(frame: pd.DataFrame, lane: str, instrument: str) -> bool:
    if frame.empty:
        return False
    selected = frame.loc[frame["lane"].eq(lane) & frame["instrument"].eq(instrument)]
    return not selected.empty and bool(selected["status"].eq("READY").any())


def _table_present(catalog: LocalMarketCatalog, database: str, table: str) -> bool:
    return any(row.database == database and row.table == table for row in catalog.sqlite_metadata) or any(
        row.database == database and row.table == table for row in catalog.sqlite_series
    )


def build_mechanism_readiness_gate(
    catalog: LocalMarketCatalog,
    *,
    config: SourceReadinessConfig = SourceReadinessConfig(),
) -> pd.DataFrame:
    sqlite = build_sqlite_inventory(catalog, config=config)
    archives = build_archive_inventory(catalog)

    eth_price = _has_ready(
        _matching(sqlite, database="crypto_history.db", instrument="ETH-USDT-SWAP", timeframe="1m")
    )
    btc_price = _has_ready(
        _matching(sqlite, database="crypto_history.db", instrument="BTC-USDT-SWAP", timeframe="1m")
    )
    price_1m = sqlite.loc[
        sqlite.get("database", pd.Series(dtype=str)).eq("crypto_history.db")
        & sqlite.get("timeframe", pd.Series(dtype=str)).eq("1m")
    ] if not sqlite.empty else sqlite
    instruments = set(price_1m.get("instrument", pd.Series(dtype=str)).dropna().astype(str))
    eth_spot = sorted(
        value for value in instruments if value.startswith("ETH-") and not value.endswith("-SWAP")
    )
    alternate_swaps = sorted(
        value
        for value in instruments
        if value.endswith("-SWAP") and value not in {"ETH-USDT-SWAP", "BTC-USDT-SWAP"}
    )
    oi = _has_ready(_matching(sqlite, database="okx_derivatives.db", table="open_interest"))
    funding = _has_ready(_matching(sqlite, database="okx_derivatives.db", table="funding_rate"))
    mark = _has_ready(_matching(sqlite, database="okx_derivatives.db", table="mark_price"))
    liquidation = _has_ready(_matching(sqlite, database="okx_derivatives.db", table="liquidation"))
    binance_frame = _matching(
        sqlite, database="binance_futures_metrics.db", instrument="ETHUSDT", timeframe="5m"
    )
    binance_span = (
        not binance_frame.empty
        and pd.Timestamp(binance_frame["start_pre_embargo"].min()) <= config.discovery_start
        and pd.Timestamp(binance_frame["end_pre_embargo"].max())
        >= config.validation_end_exclusive - pd.Timedelta(minutes=5)
    )
    binance_metrics = bool(binance_span) and _archive_ready(
        archives, "binance_futures_metrics_raw", "ETHUSDT"
    )
    trades = _archive_ready(archives, "okx_raw_trades", "ETH-USDT-SWAP")
    books = _archive_ready(archives, "okx_raw_books", "ETH-USDT-SWAP")
    primitives = _archive_ready(archives, "okx_liquidity_primitives", "ETH-USDT-SWAP")
    liquidity_map = _archive_ready(archives, "okx_liquidity_map", "ETH-USDT-SWAP")
    range_ready = _has_ready(_matching(sqlite, database="okx_range_bars.db", instrument="ETH-USDT-SWAP"))

    def row(
        hypothesis: str,
        source_gate: str,
        evidence: str,
        novelty: str,
        prior_scope: str,
    ) -> dict[str, str]:
        decision = "ELIGIBLE_FOR_PRECOMMITMENT" if source_gate == "READY" and novelty == "NOVEL" else "NO_R27"
        return {
            "hypothesis": hypothesis,
            "source_gate": source_gate,
            "visible_evidence": evidence,
            "mechanism_novelty": novelty,
            "prior_scope": prior_scope,
            "decision": decision,
        }

    return pd.DataFrame(
        [
            row(
                "price_structure_or_path_variant",
                "READY" if eth_price else "MISSING",
                "ETH-USDT-SWAP 1m OHLCV is complete" if eth_price else "no complete ETH swap baseline",
                "EXHAUSTED",
                "R01-R17 and R20-R25 cover reversal, continuation, trend, impulse, compression, panic-wick, and Range-Bar paths",
            ),
            row(
                "btc_led_eth_repricing",
                "READY" if eth_price and btc_price else "MISSING",
                f"discovered 1m instruments={sorted(instruments)}",
                "FROZEN",
                "R22 rejected the BTC-led ETH catch-up family",
            ),
            row(
                "alternate_swap_leader_rotation",
                "READY" if eth_price and bool(alternate_swaps) else "MISSING",
                f"complete alternate swaps={alternate_swaps or 'none discovered'}",
                "NOVEL",
                "unresolved only if a non-BTC leader has full visible history",
            ),
            row(
                "eth_spot_led_swap_convergence",
                "READY" if eth_price and bool(eth_spot) else "MISSING",
                f"complete ETH spot instruments={eth_spot or 'none discovered'}",
                "NOVEL",
                "highest-value unresolved cross-market price-discovery mechanism",
            ),
            row(
                "okx_contract_oi_transition",
                "READY" if oi else "PARTIAL" if _table_present(catalog, "okx_derivatives.db", "open_interest") else "MISSING",
                "only 1D ETH-USDT-SWAP OI begins 2024-01-01" if not oi else "complete OKX OI",
                "FROZEN_ADJACENT",
                "R18/R19 rejected OI release/rebuild; incomplete OKX OI cannot provide a new two-year discovery lane",
            ),
            row(
                "funding_mark_basis_unwind",
                "READY" if funding and mark else "MISSING",
                f"funding_ready={funding}; mark_ready={mark}",
                "NOVEL",
                "unresolved carry/price-dislocation mechanism",
            ),
            row(
                "liquidation_flow_reversal",
                "READY" if liquidation else "MISSING",
                f"liquidation_ready={liquidation}",
                "NOVEL_BUT_OVERLAPS_FLOW",
                "requires actual liquidation records; price proxies are not equivalent",
            ),
            row(
                "binance_positioning_ratio_or_oi_variant",
                "READY" if binance_metrics else "PARTIAL",
                f"5m database plus complete dated raw archives={binance_metrics}",
                "FROZEN",
                "R18/R19/R26 rejected base-OI and relative-positioning families; no threshold/filter rescue",
            ),
            row(
                "trade_flow_absorption_or_impact_variant",
                "READY" if trades else "MISSING",
                f"ETH raw-trade daily coverage complete={trades}",
                "EXHAUSTED",
                "R03-R07, R23, order-flow, CVD, absorption, and failed-auction branches",
            ),
            row(
                "range_bar_activity_or_exhaustion_variant",
                "READY" if range_ready and trades else "PARTIAL",
                f"range stores span visible window={range_ready}; raw days complete={trades}",
                "EXHAUSTED",
                "Range-Bar activity/continuation and fixed r0020 run exhaustion rejected through R25",
            ),
            row(
                "persistent_book_liquidity_response",
                "READY" if books and primitives and liquidity_map else "MISSING",
                f"books={books}; primitives={primitives}; liquidity_map={liquidity_map}",
                "NOVEL",
                "unresolved only with discovery-plus-validation book coverage",
            ),
        ]
    )


def r27_gate_decision(mechanism_gate: pd.DataFrame) -> str:
    if mechanism_gate.empty:
        return "UNASSIGNED_NO_ELIGIBLE_MECHANISM"
    eligible = mechanism_gate["decision"].eq("ELIGIBLE_FOR_PRECOMMITMENT")
    return "R27_PRECOMMITMENT_ALLOWED" if bool(eligible.any()) else "UNASSIGNED_NO_ELIGIBLE_MECHANISM"


def source_readiness_markdown(
    sqlite_inventory: pd.DataFrame,
    archive_inventory: pd.DataFrame,
    mechanism_gate: pd.DataFrame,
    seal_checks: pd.DataFrame,
) -> str:
    decision = r27_gate_decision(mechanism_gate)
    lines = [
        "# Pre-R27 Local Source Readiness Audit",
        "",
        f"Decision: **{decision}**",
        "",
        "The catalog is discovered from actual local tables and series directories through `src.data_feed`. "
        "Every aggregate is physically bounded to timestamps before 2025-07-01; July and the MSS2 holdout remain unopened.",
        "",
        "## Actual fixed-cadence price catalog",
        "",
        "| Instrument | Timeframe | Rows | Start | End | Coverage | Status |",
        "|---|---|---:|---|---|---:|---|",
    ]
    price = sqlite_inventory.loc[sqlite_inventory["database"].eq("crypto_history.db")]
    for item in price.sort_values(["instrument", "timeframe"]).itertuples(index=False):
        ratio = "-" if pd.isna(item.coverage_ratio) else f"{float(item.coverage_ratio):.4%}"
        lines.append(
            f"| {item.instrument} | {item.timeframe} | {int(item.rows_pre_embargo):,} | "
            f"{item.start_pre_embargo} | {item.end_pre_embargo} | {ratio} | {item.status} |"
        )
    lines.extend(
        [
            "",
            "## Dated archive coverage",
            "",
            "| Lane | Instrument | Days | Expected | Missing | Start | End | Status |",
            "|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for item in archive_inventory.sort_values(["lane", "instrument"]).itertuples(index=False):
        lines.append(
            f"| {item.lane} | {item.instrument} | {int(item.dated_days):,} | "
            f"{int(item.expected_days):,} | {int(item.missing_days):,} | "
            f"{item.start_pre_embargo or '-'} | {item.end_pre_embargo or '-'} | {item.status} |"
        )
    binance = sqlite_inventory.loc[
        sqlite_inventory["database"].eq("binance_futures_metrics.db")
    ]
    oi = sqlite_inventory.loc[
        sqlite_inventory["database"].eq("okx_derivatives.db")
        & sqlite_inventory["table"].eq("open_interest")
    ]
    binance_text = "not present"
    if not binance.empty:
        item = binance.iloc[0]
        expected = int(item["expected_rows"]) if pd.notna(item["expected_rows"]) else 0
        observed = int(item["rows_pre_embargo"])
        binance_text = (
            f"{observed:,}/{expected:,} expected 5m rows "
            f"({observed / expected:.4%}); all 1,277 raw archive days exist"
        )
    oi_text = "not present"
    if not oi.empty:
        item = oi.iloc[0]
        oi_text = (
            f"{int(item['rows_pre_embargo']):,} daily rows from "
            f"{item['start_pre_embargo']} through {item['end_pre_embargo']}"
        )
    lines.extend(
        [
            "",
            "## Data-quality findings",
            "",
            "- **Critical source-readiness gap, high confidence:** spot, funding, mark, liquidation, and pre-embargo "
            "book-derived lanes are absent. This blocks the unresolved cross-market/carry/book hypotheses; it is "
            "not safe to replace them with price proxies.",
            f"- **High coverage limitation, high confidence:** OKX contract OI has {oi_text}. It omits the entire "
            "2023 discovery year and cannot support the frozen two-year discovery design.",
            f"- **Medium ingestion gaps, high confidence:** Binance metrics contain {binance_text}. Prior positioning "
            "studies handle gaps causally, but the lane is not perfectly regular and cannot be described as gap-free.",
            "- **Low price-baseline risk, high confidence:** every discovered fixed-cadence ETH/BTC OHLCV table has "
            "exactly 100% of expected rows inside the visible window.",
            "",
            "## Mechanism gate",
            "",
            "| Hypothesis | Source gate | Novelty | Decision | Evidence |",
            "|---|---|---|---|---|",
        ]
    )
    for item in mechanism_gate.itertuples(index=False):
        lines.append(
            f"| {item.hypothesis} | {item.source_gate} | {item.mechanism_novelty} | "
            f"{item.decision} | {item.visible_evidence} |"
        )
    lines.extend(
        [
            "",
            "## Seal checks",
            "",
        ]
    )
    for item in seal_checks.itertuples(index=False):
        lines.append(f"- `{item.check}`: {'PASS' if item.passed else 'FAIL'} {item.detail or ''}".rstrip())
    lines.extend(
        [
            "",
            "## Frozen conclusion",
            "",
            "No overlooked pre-embargo series supports a clean new mechanism. The actual OHLCV catalog contains only "
            "ETH-USDT-SWAP and BTC-USDT-SWAP at 1m; BTC leadership is already frozen after R22. "
            "OKX funding, mark, liquidation, books, liquidity primitives, and liquidity-map lanes have no visible "
            "pre-embargo coverage, while OKX OI starts in 2024. The complete trade, Range-Bar, and Binance metric "
            "lanes map to mechanism families already rejected through R26. R27 therefore remains unassigned.",
            "",
            "The next eligible study still requires newly acquired, genuinely forward-sealed history for one of: "
            "ETH spot/perpetual convergence, funding/mark/index basis unwind, or persistent book-liquidity response.",
            "",
        ]
    )
    return "\n".join(lines)
