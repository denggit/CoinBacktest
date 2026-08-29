from __future__ import annotations

import asyncio
import logging
import signal
from datetime import datetime, timezone

from .alerts import (
    Alert,
    FedWatchState,
    bp_change,
    classify_macro,
    compare_fedwatch,
    fedwatch_comparison_detail,
    fedwatch_repricing_alerts,
    severity_for,
    threshold_triggered,
    yield_alerts,
)
from .browser import BrowserSession
from .config import MonitorConfig
from .emailer import AsyncEmailSender
from .models import (
    DxySnapshot,
    FedWatchSnapshot,
    TargetProbability,
    TreasurySnapshot,
    expected_target_rate,
    unavailable_observation,
    utc_now_iso,
)
from .schedule import new_york_time, poll_delay_seconds, polling_mode
from .sources import DxySource, FedWatchSource, SourceUnavailable, TreasurySource
from .storage import MacroStore


class MacroMonitor:
    def __init__(self, config: MonitorConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.store = MacroStore(config.db_path, retention_days=config.retention_days)
        self.browser = BrowserSession(headed=config.headed, verbose=config.verbose, logger=logger)
        self.fedwatch = FedWatchSource(self.browser, logger, retries=config.source_retries)
        self.treasury = TreasurySource(self.browser, logger, retries=config.source_retries)
        self.dxy = DxySource(self.browser, logger, retries=config.source_retries)
        self.email = AsyncEmailSender(config.email, logger)
        self._stop = asyncio.Event()
        self._latest_fedwatch: FedWatchSnapshot | None = None
        self._latest_treasury: TreasurySnapshot | None = None
        self._latest_dxy: DxySnapshot | None = None
        self._polling_mode: str | None = None

    async def start(self) -> None:
        self.logger.info("[macro] starting")
        self.logger.info("[macro] db=%s", self.config.db_path)
        self.logger.info(
            "[macro] polling schedule=America/New_York weekdays 07:00-19:00 after-hours=%.0fs weekends=%.0fs",
            self.config.off_hours_poll_seconds,
            self.config.weekend_poll_seconds,
        )
        self._log_polling_mode(datetime.now(timezone.utc))
        await self.browser.start()

    async def run_once(self) -> int:
        try:
            await self.start()
            fed_result, treasury_result, dxy_result = await asyncio.gather(
                self.fedwatch.fetch(),
                self.treasury.fetch(),
                self.dxy.fetch(),
                return_exceptions=True,
            )
            ok = False
            if isinstance(fed_result, FedWatchSnapshot):
                ok = True
                await self._handle_fedwatch(fed_result)
            else:
                await self._record_unavailable("cme_fedwatch", "fedwatch_source_status", fed_result)
            if isinstance(treasury_result, TreasurySnapshot):
                ok = True
                await self._handle_treasury(treasury_result)
            else:
                await self._record_unavailable("treasury_yields", "treasury_source_status", treasury_result)
            if isinstance(dxy_result, DxySnapshot):
                ok = True
                await self._handle_dxy(dxy_result)
            else:
                await self._record_unavailable("dxy_index", "dxy_source_status", dxy_result)
            if ok and not await self._evaluate_macro_confirmation():
                self.logger.info("[monitor] no alert")
            return 0 if ok else 1
        finally:
            await self.close()

    async def run_forever(self) -> None:
        tasks: set[asyncio.Task[None]] = set()
        try:
            await self.start()
            self._install_signal_handlers()
            tasks = {
                asyncio.create_task(self._fedwatch_loop(), name="fedwatch-poll"),
                asyncio.create_task(self._treasury_loop(), name="treasury-poll"),
                asyncio.create_task(self._dxy_loop(), name="dxy-poll"),
            }
            await self._stop.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.close()

    def stop(self) -> None:
        self._stop.set()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop)
            except (NotImplementedError, RuntimeError):
                pass

    async def _fedwatch_loop(self) -> None:
        while True:
            started = asyncio.get_running_loop().time()
            try:
                await self._handle_fedwatch(await self.fedwatch.fetch())
                await self._evaluate_macro_confirmation()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._record_unavailable("cme_fedwatch", "fedwatch_source_status", exc)
            await self._sleep_for_poll(self.config.fedwatch_poll_seconds, started)

    async def _treasury_loop(self) -> None:
        while True:
            started = asyncio.get_running_loop().time()
            try:
                await self._handle_treasury(await self.treasury.fetch())
                await self._evaluate_macro_confirmation()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._record_unavailable("treasury_yields", "treasury_source_status", exc)
            await self._sleep_for_poll(self.config.treasury_poll_seconds, started)

    async def _dxy_loop(self) -> None:
        while True:
            started = asyncio.get_running_loop().time()
            try:
                await self._handle_dxy(await self.dxy.fetch())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._record_unavailable("dxy_index", "dxy_source_status", exc)
            await self._sleep_for_poll(self.config.dxy_poll_seconds, started)

    async def _sleep_for_poll(self, normal_interval_seconds: float, started: float) -> None:
        now = datetime.now(timezone.utc)
        self._log_polling_mode(now)
        elapsed = asyncio.get_running_loop().time() - started
        await asyncio.sleep(
            poll_delay_seconds(
                normal_interval_seconds,
                self.config.off_hours_poll_seconds,
                self.config.weekend_poll_seconds,
                elapsed_seconds=elapsed,
                now=now,
            )
        )

    def _log_polling_mode(self, now: datetime) -> None:
        mode = polling_mode(now)
        if mode == self._polling_mode:
            return
        self._polling_mode = mode
        local = new_york_time(now)
        self.logger.info(
            "[macro] polling mode=%s ny_time=%s",
            mode,
            local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        )

    def _window_tolerance(self, normal_interval_seconds: float, minimum_seconds: int) -> int:
        return max(
            minimum_seconds,
            int(normal_interval_seconds * 2),
            int(self.config.off_hours_poll_seconds / 2),
        )

    async def _handle_fedwatch(self, snapshot: FedWatchSnapshot) -> None:
        previous: dict[int, FedWatchState] = {}
        for minutes in (15, 60):
            state = self._fedwatch_state_at_window(snapshot, minutes)
            if state is not None:
                previous[minutes] = state
        current = self._current_fedwatch_state(snapshot)
        self.store.insert_observations(snapshot.observations())
        self._latest_fedwatch = snapshot
        self.logger.info("[fedwatch] meeting=%s source=%s", snapshot.meeting_date, snapshot.source)
        for item in snapshot.probabilities:
            if self.config.verbose or item.probability > 0:
                self.logger.info("[fedwatch] target=%s%% probability=%.1f%%", item.target_range, item.probability)
        if snapshot.cut_probability is not None:
            self.logger.info(
                "[fedwatch] cut=%.1f%% hold=%s hike=%s expected_rate=%s",
                snapshot.cut_probability,
                f"{snapshot.hold_probability:.1f}%" if snapshot.hold_probability is not None else "n/a",
                f"{snapshot.hike_probability:.1f}%" if snapshot.hike_probability is not None else "n/a",
                f"{snapshot.expected_rate:.3f}%" if snapshot.expected_rate is not None else "n/a",
            )
            await self._emit_alerts(
                fedwatch_repricing_alerts(current, previous, self.config.thresholds),
                snapshot.timestamp_utc,
            )

    @staticmethod
    def _current_fedwatch_state(snapshot: FedWatchSnapshot) -> FedWatchState:
        return FedWatchState(
            cut_probability=snapshot.cut_probability,
            hold_probability=snapshot.hold_probability,
            hike_probability=snapshot.hike_probability,
            expected_rate=snapshot.expected_rate,
        )

    def _fedwatch_state_at_window(self, snapshot: FedWatchSnapshot, minutes: int) -> FedWatchState | None:
        anchor = self.store.window_value(
            "fedwatch_cut_probability",
            snapshot.timestamp_utc,
            minutes,
            meeting_date=snapshot.meeting_date,
            tolerance_seconds=self._window_tolerance(self.config.fedwatch_poll_seconds, 120),
        )
        if anchor is None:
            return None
        rows = self.store.fedwatch_observations_at(snapshot.meeting_date, anchor.timestamp_utc)
        values = {row.metric: row.value for row in rows if row.value is not None}
        probabilities = tuple(
            TargetProbability(row.target_range, float(row.value))
            for row in rows
            if row.metric == "fedwatch_target_probability"
            and row.target_range is not None
            and row.value is not None
        )
        expected_rate = values.get("fedwatch_expected_rate")
        if expected_rate is None:
            expected_rate = expected_target_rate(probabilities)
        cut = values.get("fedwatch_cut_probability")
        hike = values.get("fedwatch_hike_probability")
        if cut is None or hike is None:
            return None
        return FedWatchState(
            cut_probability=float(cut),
            hold_probability=float(values["fedwatch_hold_probability"])
            if values.get("fedwatch_hold_probability") is not None
            else None,
            hike_probability=float(hike),
            expected_rate=float(expected_rate) if expected_rate is not None else None,
        )

    async def _handle_treasury(self, snapshot: TreasurySnapshot) -> None:
        histories: dict[str, dict[int, float]] = {"us2y_yield": {}, "us10y_yield": {}}
        current_values = {"us2y_yield": snapshot.us2y_yield, "us10y_yield": snapshot.us10y_yield}
        for metric, current in current_values.items():
            if current is None:
                continue
            for minutes in (5, 15, 60):
                row = self.store.window_value(
                    metric,
                    snapshot.timestamp_utc,
                    minutes,
                    tolerance_seconds=self._window_tolerance(self.config.treasury_poll_seconds, 45),
                )
                if row and row.value is not None:
                    histories[metric][minutes] = row.value
        rows = snapshot.observations()
        if snapshot.us2y_yield is None:
            rows.append(unavailable_observation(snapshot.source_2y, "us2y_yield", snapshot.timestamp_utc))
        if snapshot.us10y_yield is None:
            rows.append(unavailable_observation(snapshot.source_10y, "us10y_yield", snapshot.timestamp_utc))
        self.store.insert_observations(rows)
        self._latest_treasury = snapshot
        if snapshot.us2y_yield is not None:
            self.logger.info("[treasury] US2Y=%.3f%% source=%s", snapshot.us2y_yield, snapshot.source_2y)
        else:
            self.logger.warning("[treasury] US2Y source unavailable")
        if snapshot.us10y_yield is not None:
            self.logger.info("[treasury] US10Y=%.3f%% source=%s", snapshot.us10y_yield, snapshot.source_10y)
        else:
            self.logger.warning("[treasury] US10Y source unavailable")
        if snapshot.spread is not None:
            self.logger.info("[treasury] spread=%+.1f bp", snapshot.spread * 100.0)
        alerts: list[Alert] = []
        if snapshot.us2y_yield is not None:
            alerts += yield_alerts("us2y_yield", snapshot.us2y_yield, histories["us2y_yield"], self.config.thresholds)
        if snapshot.us10y_yield is not None:
            alerts += yield_alerts("us10y_yield", snapshot.us10y_yield, histories["us10y_yield"], self.config.thresholds)
        await self._emit_alerts(alerts, snapshot.timestamp_utc)

    async def _handle_dxy(self, snapshot: DxySnapshot) -> None:
        rows = snapshot.observations()
        if snapshot.value is None:
            rows.append(unavailable_observation(snapshot.source, "dxy_index", snapshot.timestamp_utc))
        self.store.insert_observations(rows)
        self._latest_dxy = snapshot
        if snapshot.value is not None:
            self.logger.info("[dxy] DXY=%.3f source=%s", snapshot.value, snapshot.source)
        else:
            self.logger.warning("[dxy] source unavailable")

    async def _evaluate_macro_confirmation(self) -> bool:
        if not self._latest_treasury and not self._latest_fedwatch:
            return False
        emitted = False
        now = (
            self._latest_treasury.timestamp_utc
            if self._latest_treasury
            else self._latest_fedwatch.timestamp_utc
        )
        for minutes in (15, 60):
            fed_change = us2_change = None
            fed_comparison = None
            fed_limit = getattr(self.config.thresholds, f"fedwatch_{minutes}m_pct")
            us2_limit = getattr(self.config.thresholds, f"us2y_{minutes}m_bp")
            if self._latest_fedwatch:
                old_state = self._fedwatch_state_at_window(self._latest_fedwatch, minutes)
                if old_state is not None:
                    fed_comparison = compare_fedwatch(
                        old_state,
                        self._current_fedwatch_state(self._latest_fedwatch),
                    )
                    if fed_comparison is not None:
                        fed_change = fed_comparison.signed_change_pct
            if self._latest_treasury and self._latest_treasury.us2y_yield is not None:
                row = self.store.window_value(
                    "us2y_yield",
                    self._latest_treasury.timestamp_utc,
                    minutes,
                    tolerance_seconds=self._window_tolerance(self.config.treasury_poll_seconds, 45),
                )
                if row and row.value is not None:
                    us2_change = bp_change(self._latest_treasury.us2y_yield, row.value)
            classification = classify_macro(
                fedwatch_change_pct=fed_change,
                fedwatch_threshold_pct=fed_limit,
                us2y_change_bp=us2_change,
                us2y_threshold_bp=us2_limit,
            )
            if classification:
                direction = "MIXED" if classification.startswith("MIXED") else ("DOVISH" if "DOVISH" in classification else "HAWKISH")
                detail_parts = []
                if fed_comparison is not None:
                    detail_parts.append(fedwatch_comparison_detail(fed_comparison, minutes))
                if us2_change is not None:
                    detail_parts.append(f"US2Y {us2_change:+.1f} bp")
                component_severities: list[int] = []
                if fed_change is not None and threshold_triggered(fed_change, fed_limit):
                    component_severities.append(severity_for(fed_change, fed_limit))
                if us2_change is not None and threshold_triggered(us2_change, us2_limit):
                    component_severities.append(severity_for(us2_change, us2_limit))
                severity = max(component_severities, default=1)
                if classification.startswith("STRONG"):
                    severity = max(2, severity)
                await self._emit_alerts(
                    [Alert(f"MACRO_{minutes}M_{direction}", classification, "\n".join(detail_parts), direction.lower(), severity, minutes)],
                    now,
                )
                emitted = True
        return emitted

    async def _emit_alerts(self, alerts: list[Alert], timestamp_utc: str) -> None:
        for alert in alerts:
            self.logger.warning("[alert] %s | %s", alert.title.upper(), alert.detail)
            if self.config.email.enabled and self.store.cooldown_allows(
                alert.key,
                alert.severity,
                timestamp_utc,
                self.config.email.cooldown_seconds,
            ):
                body = (
                    f"{alert.title}\n\n{alert.detail}\n"
                    f"Window: {alert.window_minutes} minutes\n"
                    f"Observed: {timestamp_utc}\n\n"
                    "This monitor describes market repricing only; it is not a trading signal."
                )
                self.email.submit(f"[CoinBacktest Macro] {alert.title}", body)

    async def _record_unavailable(self, source: str, metric: str, error: object) -> None:
        detail = error.detail if isinstance(error, SourceUnavailable) else str(error)
        self.store.insert_observations([unavailable_observation(source, metric, utc_now_iso())])
        self.logger.warning("[macro] source unavailable source=%s error=%s", source, detail)

    async def close(self) -> None:
        await self.email.drain()
        await self.browser.close()
        self.store.close()
