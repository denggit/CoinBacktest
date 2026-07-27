from __future__ import annotations

import random

import csv
import json
from datetime import date
from pathlib import Path

from src.data_feed.okx_books_loader import OKXBooksLoader
from src.liquidity_map.models import BookEvent, BookLevel
from src.liquidity_map.replay import OrderBookReplay


def _raw_dir(tmp_path: Path) -> Path:
    return tmp_path / "okx" / "raw" / "books" / "ETH-USDT-SWAP" / "l2_400" / "2026" / "06"


def test_jsonl_snapshot_update_parser(tmp_path: Path) -> None:
    folder = _raw_dir(tmp_path)
    folder.mkdir(parents=True)
    path = folder / "ETH-USDT-SWAP_books_2026-06-01.jsonl"
    rows = [
        {
            "action": "snapshot",
            "data": [{"ts": "1780272000000", "bids": [["1800", "10", "0", "2"]], "asks": [["1801", "12", "0", "3"]], "seqId": "10"}],
        },
        {
            "action": "update",
            "data": [{"ts": "1780272000100", "bids": [["1800", "8", "0", "2"]], "asks": [], "seqId": "11", "prevSeqId": "10"}],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    events = list(OKXBooksLoader(data_dir=tmp_path).iter_book_events(date(2026, 6, 1)))
    assert [event.action for event in events] == ["snapshot", "update"]
    assert events[0].bids[0].size_contracts == 10
    assert events[0].asks[0].order_count == 3
    assert events[1].prev_seq_id == 10


def test_nested_csv_without_action_is_snapshot(tmp_path: Path) -> None:
    folder = _raw_dir(tmp_path)
    folder.mkdir(parents=True)
    path = folder / "books_20260601.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ts", "bids", "asks"])
        writer.writeheader()
        writer.writerow({"ts": "1780272000000", "bids": json.dumps([[1800, 5, 0, 1]]), "asks": json.dumps([[1801, 6, 0, 1]])})
    events = list(OKXBooksLoader(data_dir=tmp_path).iter_book_events("2026-06-01"))
    assert len(events) == 1
    assert events[0].is_snapshot


def test_price_level_csv_without_action_is_full_snapshot(tmp_path: Path) -> None:
    folder = _raw_dir(tmp_path)
    folder.mkdir(parents=True)
    path = folder / "levels_2026-06-01.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["timestamp", "side", "price", "size", "order_count"])
        writer.writeheader()
        writer.writerow({"timestamp": "1780272000000", "side": "bid", "price": 1800, "size": 5, "order_count": 2})
        writer.writerow({"timestamp": "1780272000000", "side": "ask", "price": 1801, "size": 7, "order_count": 3})
    events = list(OKXBooksLoader(data_dir=tmp_path).iter_book_events("2026-06-01"))
    assert len(events) == 1
    assert events[0].is_snapshot
    assert len(events[0].bids) == len(events[0].asks) == 1


def test_sequence_gap_invalidates_until_next_snapshot() -> None:
    replay = OrderBookReplay(price_step=1.0, strict_sequence=True)
    snapshot = BookEvent(1000, "snapshot", bids=(BookLevel(100, 10),), asks=(BookLevel(101, 10),), seq_id=10)
    replay.apply(snapshot)
    _deltas, gap = replay.apply(BookEvent(1100, "update", bids=(BookLevel(100, 9),), seq_id=12, prev_seq_id=9))
    assert gap is True
    assert replay.valid is False
    replay.apply(BookEvent(1200, "snapshot", bids=(BookLevel(100, 8),), asks=(BookLevel(101, 8),), seq_id=13))
    assert replay.valid is True


def test_repeated_snapshots_produce_level_deltas() -> None:
    replay = OrderBookReplay(price_step=1.0)
    replay.apply(BookEvent(1000, "snapshot", bids=(BookLevel(100, 10),), asks=(BookLevel(101, 10),)))
    deltas, gap = replay.apply(BookEvent(1100, "snapshot", bids=(BookLevel(100, 8),), asks=(BookLevel(101, 13),)))
    assert gap is False
    by_side = {delta.side: delta for delta in deltas}
    assert by_side["bid"].removed_contracts == 2
    assert by_side["ask"].added_contracts == 3


def test_tikdat_style_csv_snapshot_flag_and_sequence(tmp_path: Path) -> None:
    folder = _raw_dir(tmp_path)
    folder.mkdir(parents=True)
    path = folder / "tikdat_20260601.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["timestamp", "snapshot", "asks", "bids", "seq_id", "prev_seq_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "1780272000000",
                "snapshot": "true",
                "asks": json.dumps([[1801, 10, 0, 2]]),
                "bids": json.dumps([[1800, 9, 0, 2]]),
                "seq_id": 10,
                "prev_seq_id": -1,
            }
        )
        writer.writerow(
            {
                "timestamp": "1780272000100",
                "snapshot": "false",
                "asks": json.dumps([[1801, 8, 0, 2]]),
                "bids": json.dumps([]),
                "seq_id": 11,
                "prev_seq_id": 10,
            }
        )
    events = list(OKXBooksLoader(data_dir=tmp_path).iter_book_events("2026-06-01"))
    assert [event.action for event in events] == ["snapshot", "update"]
    assert events[1].prev_seq_id == 10


def test_symbol_root_tar_gz_archive_is_discovered_and_parsed(tmp_path: Path) -> None:
    import io
    import tarfile

    folder = tmp_path / "okx" / "raw" / "books" / "ETH-USDT-SWAP"
    folder.mkdir(parents=True)
    archive = folder / "ETH-USDT-SWAP-L2orderbook-400lv-2026-06-01.tar.gz"
    payload = (
        "timestamp,snapshot,asks,bids,seq_id,prev_seq_id\n"
        '1780272000000,true,"[[1801,10,0,2]]","[[1800,9,0,2]]",10,-1\n'
    ).encode("utf-8")
    info = tarfile.TarInfo("ETH-USDT-SWAP-L2orderbook-400lv-2026-06-01.csv")
    info.size = len(payload)
    with tarfile.open(archive, "w:gz") as tf:
        tf.addfile(info, io.BytesIO(payload))

    loader = OKXBooksLoader(data_dir=tmp_path)
    assert loader.find_local_book_files("2026-06-01") == [archive]
    schema = loader.inspect_day_schema("2026-06-01")
    assert len(schema["files"]) == 1
    assert "timestamp,snapshot" in schema["files"][0]["sample"][0]
    events = list(loader.iter_book_events("2026-06-01"))
    assert len(events) == 1
    assert events[0].is_snapshot
    assert events[0].bids[0].size_contracts == 9


def test_root_archives_are_filtered_by_requested_depth(tmp_path: Path) -> None:
    import io
    import tarfile

    folder = tmp_path / "okx" / "raw" / "books" / "ETH-USDT-SWAP"
    folder.mkdir(parents=True)
    for depth in (400, 5000):
        archive = folder / f"ETH-USDT-SWAP-L2orderbook-{depth}lv-2026-06-01.tar.gz"
        payload = (
            json.dumps(
                {
                    "instId": "ETH-USDT-SWAP",
                    "action": "snapshot",
                    "ts": "1780272000000",
                    "asks": [["1801", "10", "2"]],
                    "bids": [["1800", "9", "2"]],
                }
            )
            + "\n"
        ).encode("utf-8")
        info = tarfile.TarInfo(f"ETH-USDT-SWAP-L2orderbook-{depth}lv-2026-06-01.data")
        info.size = len(payload)
        with tarfile.open(archive, "w:gz") as tf:
            tf.addfile(info, io.BytesIO(payload))

    loader400 = OKXBooksLoader(data_dir=tmp_path, depth=400)
    loader5000 = OKXBooksLoader(data_dir=tmp_path, depth=5000)
    assert [path.name for path in loader400.find_local_book_files("2026-06-01")] == [
        "ETH-USDT-SWAP-L2orderbook-400lv-2026-06-01.tar.gz"
    ]
    assert [path.name for path in loader5000.find_local_book_files("2026-06-01")] == [
        "ETH-USDT-SWAP-L2orderbook-5000lv-2026-06-01.tar.gz"
    ]
    event = next(loader5000.iter_book_events("2026-06-01"))
    assert event.is_snapshot
    assert event.asks[0].order_count == 2
    probe = loader5000.probe_day_events("2026-06-01", max_events=10)
    assert probe["requested_depth"] == 5000
    assert probe["action_counts"] == {"snapshot": 1}


def test_replay_drops_float_residual_after_every_exact_level_in_bin_is_removed() -> None:
    replay = OrderBookReplay(price_step=1.0)
    rng = random.Random(0)
    sizes = [rng.random() * 1000.0 for _ in range(100)]
    prices = [2000.0 + index / 100.0 for index in range(100)]
    replay.apply(
        BookEvent(
            1,
            "snapshot",
            bids=tuple(BookLevel(price, size, 1) for price, size in zip(prices, sizes)),
            asks=(BookLevel(2001.0, 1.0, 1),),
            seq_id=1,
        )
    )

    replay.apply(
        BookEvent(
            2,
            "update",
            bids=tuple(BookLevel(price, 0.0, 0) for price in prices),
            seq_id=2,
            prev_seq_id=1,
        )
    )

    assert replay.bids == {}
    assert 2000 not in replay.bid_bins
    assert list(replay.iter_binned_depth("bid")) == []
