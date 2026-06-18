#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run all backtest scripts under a directory sequentially.

Examples:
    python tools/run_backtest_dir.py backtest/hf --start-date 2026-01-01 --end-date 2026-06-15

    python tools/run_backtest_dir.py backtest/mf \
      --start-date 2023-01-01 --end-date 2026-06-15 \
      --initial-capital 1000 --fee-rate 0.0005

    # Everything after -- is passed to each backtest unchanged.
    python tools/run_backtest_dir.py backtest/lf --start-date 2023-01-01 --end-date 2026-06-15 -- --symbol ETH-USDT-SWAP
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


def configure_utf8_stdio() -> None:
    """Make runner-side console output tolerant of UTF-8/emoji text on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

def child_utf8_env() -> dict[str, str]:
    """Force child backtest Python processes to write UTF-8 to stdout/stderr."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def parse_yyyy_mm_dd(value: str | None, name: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD, got: {value!r}") from exc


@dataclass
class BacktestRunResult:
    script: str
    command: list[str]
    status: str
    returncode: int | None
    elapsed_seconds: float
    log_file: str


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run every Python backtest script under a directory sequentially and pass shared CLI args to each script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target_dir", help="Backtest directory to run, e.g. backtest/hf, backtest/mf, backtest/lf, or backtest.")
    parser.add_argument("--start-date", help="Passed to every backtest script as --start-date.")
    parser.add_argument("--end-date", help="Passed to every backtest script as --end-date.")
    parser.add_argument("--pattern", default="*.py", help="Script filename glob pattern.")
    parser.add_argument("--exclude", action="append", default=[], help="Filename or glob to exclude. Can be repeated.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run each backtest.")
    parser.add_argument("--out-dir", default="data/reports/_batch_runs", help="Directory for batch logs and summary files.")
    parser.add_argument("--timeout-sec", type=int, default=0, help="Per-script timeout. 0 means no timeout.")
    parser.add_argument("--sleep-sec", type=float, default=3.0, help="Sleep seconds between scripts to reduce API/rate-limit pressure.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed backtest.")
    parser.add_argument("--include-init", action="store_true", help="Include __init__.py files. Usually keep this off.")
    parser.add_argument("--sort", choices=["name", "path"], default="path", help="Execution order.")

    # parse_known_args lets users pass strategy-specific shared args directly:
    #   --initial-capital 1000 --fee-rate 0.0005
    # Those unknown args are appended to every backtest command.
    args, passthrough = parser.parse_known_args(argv)

    # Also support explicit separator style: -- --symbol ETH-USDT-SWAP
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def project_root_from_here() -> Path:
    # tools/run_backtest_dir.py -> tools -> project root
    return Path(__file__).resolve().parents[1]


def should_exclude(path: Path, patterns: Iterable[str], include_init: bool) -> bool:
    if path.name == "__init__.py" and not include_init:
        return True
    if path.name == Path(__file__).name:
        return True
    for pattern in patterns:
        if path.match(pattern) or path.name == pattern:
            return True
    return False


def discover_scripts(target_dir: Path, *, pattern: str, exclude: Iterable[str], include_init: bool, sort_mode: str) -> list[Path]:
    if not target_dir.exists():
        raise FileNotFoundError(f"target_dir does not exist: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"target_dir is not a directory: {target_dir}")

    scripts = [p for p in target_dir.rglob(pattern) if p.is_file() and not should_exclude(p, exclude, include_init)]
    if sort_mode == "name":
        return sorted(scripts, key=lambda p: (p.name, str(p)))
    return sorted(scripts, key=lambda p: str(p))


def build_common_args(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    common: list[str] = []
    if args.start_date:
        common.extend(["--start-date", args.start_date])
    if args.end_date:
        common.extend(["--end-date", args.end_date])
    common.extend(passthrough)
    return common


def safe_log_name(script: Path, root: Path) -> str:
    try:
        rel = script.relative_to(root)
    except ValueError:
        rel = script
    text = str(rel).replace("\\", "__").replace("/", "__")
    if text.endswith(".py"):
        text = text[:-3]
    return text + ".log"


def run_one(command: list[str], *, cwd: Path, log_file: Path, timeout_sec: int) -> BacktestRunResult:
    script = command[1] if len(command) > 1 else ""
    start = time.time()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.write(f"cwd={cwd}\n")
        log.write("=" * 120 + "\n")
        log.flush()

        try:
            proc = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=child_utf8_env(),
            )
            assert proc.stdout is not None
            deadline = start + timeout_sec if timeout_sec and timeout_sec > 0 else None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                if deadline and time.time() > deadline:
                    proc.kill()
                    log.write(f"\n[TIMEOUT] killed after {timeout_sec} seconds\n")
                    elapsed = time.time() - start
                    return BacktestRunResult(script=script, command=command, status="timeout", returncode=None, elapsed_seconds=elapsed, log_file=str(log_file))
            returncode = proc.wait()
            elapsed = time.time() - start
            status = "ok" if returncode == 0 else "failed"
            log.write("\n" + "=" * 120 + "\n")
            log.write(f"status={status} returncode={returncode} elapsed_seconds={elapsed:.2f}\n")
            return BacktestRunResult(script=script, command=command, status=status, returncode=returncode, elapsed_seconds=elapsed, log_file=str(log_file))
        except Exception as exc:
            elapsed = time.time() - start
            log.write(f"\n[RUNNER_ERROR] {type(exc).__name__}: {exc}\n")
            return BacktestRunResult(script=script, command=command, status="runner_error", returncode=None, elapsed_seconds=elapsed, log_file=str(log_file))


def write_summary(results: list[BacktestRunResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"

    json_path.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else ["script", "status"])
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_stdio()
    args, passthrough = parse_args(argv)

    try:
        start_dt = parse_yyyy_mm_dd(args.start_date, "--start-date")
        end_dt = parse_yyyy_mm_dd(args.end_date, "--end-date")
        if start_dt and end_dt and end_dt < start_dt:
            raise ValueError(f"--end-date must be >= --start-date, got {args.start_date} -> {args.end_date}")
    except ValueError as exc:
        print(f"[ARGUMENT_ERROR] {exc}", file=sys.stderr)
        return 2

    project_root = project_root_from_here()
    target_dir = Path(args.target_dir)
    if not target_dir.is_absolute():
        target_dir = project_root / target_dir

    scripts = discover_scripts(
        target_dir,
        pattern=args.pattern,
        exclude=args.exclude,
        include_init=bool(args.include_init),
        sort_mode=str(args.sort),
    )
    if not scripts:
        print(f"No backtest scripts found under: {target_dir}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_out_dir = project_root / args.out_dir / timestamp
    common_args = build_common_args(args, passthrough)

    print("=" * 120)
    print(f"Batch backtest target : {target_dir}")
    print(f"Scripts found         : {len(scripts)}")
    print("Execution mode        : sequential, one child process at a time")
    print(f"Sleep between scripts : {args.sleep_sec}s")
    print(f"Common args           : {' '.join(common_args) if common_args else '(none)'}")
    print(f"Batch output dir      : {batch_out_dir}")
    print("=" * 120)

    results: list[BacktestRunResult] = []
    for idx, script in enumerate(scripts, start=1):
        rel_script = script.relative_to(project_root)
        command = [args.python, "-X", "utf8", str(rel_script), *common_args]
        log_file = batch_out_dir / safe_log_name(script, project_root)

        print("\n" + "#" * 120)
        print(f"[{idx}/{len(scripts)}] Running: {' '.join(command)}")
        print(f"Log file: {log_file}")
        print("#" * 120)

        if args.dry_run:
            results.append(
                BacktestRunResult(
                    script=str(rel_script),
                    command=command,
                    status="dry_run",
                    returncode=None,
                    elapsed_seconds=0.0,
                    log_file=str(log_file),
                )
            )
            continue

        result = run_one(command, cwd=project_root, log_file=log_file, timeout_sec=int(args.timeout_sec))
        results.append(result)
        if args.fail_fast and result.status != "ok":
            print(f"[FAIL_FAST] stopped after failure: {rel_script}")
            break

        if idx < len(scripts) and args.sleep_sec > 0 and not args.dry_run:
            print(f"[SEQUENTIAL_SLEEP] sleeping {args.sleep_sec}s before next backtest...")
            time.sleep(float(args.sleep_sec))

    write_summary(results, batch_out_dir)

    ok = sum(1 for r in results if r.status == "ok")
    failed = sum(1 for r in results if r.status not in {"ok", "dry_run"})
    print("\n" + "=" * 120)
    print(f"Batch done: ok={ok} failed={failed} total={len(results)}")
    print(f"Summary: {batch_out_dir / 'summary.csv'}")
    print("=" * 120)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
