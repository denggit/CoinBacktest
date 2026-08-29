"use strict";

const DISPLAY_TIME_ZONE = "Asia/Shanghai";

const metricView = {
  us2y_yield: { prefix: "us2", series: "us2y", decimals: 3, changeUnit: "bp" },
  us10y_yield: { prefix: "us10", series: "us10y", decimals: 3, changeUnit: "bp" },
  dxy_index: { prefix: "dxy", series: "dxy", decimals: 3, changeUnit: "%" },
};

const sourceNames = {
  cme_fedwatch_en: "CME FedWatch · EN",
  cme_fedwatch_cn: "CME FedWatch · CN",
  investing_us2y: "Investing.com",
  investing_us10y: "Investing.com",
  cnbc_us2y: "CNBC Public Quote",
  cnbc_us10y: "CNBC Public Quote",
  marketwatch_us2y: "MarketWatch",
  marketwatch_us10y: "MarketWatch",
  tradingview_us2y: "TradingView",
  tradingview_us10y: "TradingView",
  cnbc_dxy: "CNBC Public Quote",
  investing_dxy: "Investing.com",
  marketwatch_dxy: "MarketWatch",
  tradingview_dxy: "TradingView",
};

let currentSnapshot = null;
let chartRangeMinutes = 60;
let lastValues = {};
let chartGeometry = null;

const $ = (id) => document.getElementById(id);

function setConnection(state, label) {
  $("connection-pill").dataset.state = state;
  $("connection-state").textContent = label;
}

function formatSigned(value, decimals = 1) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(decimals)}`;
}

function timeAgo(timestamp) {
  if (!timestamp) return "—";
  const seconds = Math.max(0, Math.round((Date.now() - Date.parse(timestamp)) / 1000));
  if (seconds < 5) return "刚刚";
  if (seconds < 60) return `${seconds}s 前`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m 前`;
  return `${Math.floor(seconds / 3600)}h 前`;
}

function sourceLabel(source) {
  return sourceNames[source] || source || "等待数据";
}

function semanticTone(metric, change) {
  if (change == null) return "neutral";
  if (metric === "fedwatch_cut_probability") return change > 0 ? "dovish" : change < 0 ? "hawkish" : "neutral";
  if (metric === "us2y_yield") return change < 0 ? "dovish" : change > 0 ? "hawkish" : "neutral";
  if (metric === "dxy_index") return change > 0 ? "hawkish" : change < 0 ? "dovish" : "neutral";
  return change > 0 ? "up" : change < 0 ? "down" : "neutral";
}

function updateMetric(metric, data, series) {
  const view = metricView[metric];
  const valueNode = $(`${view.prefix}-value`);
  const value = data?.value;
  const nextText = value == null ? "—" : Number(value).toFixed(view.decimals);
  if (lastValues[metric] !== undefined && lastValues[metric] !== nextText) {
    const card = document.querySelector(`[data-metric="${metric}"]`);
    card.classList.remove("is-updating");
    void card.offsetWidth;
    card.classList.add("is-updating");
  }
  lastValues[metric] = nextText;
  valueNode.textContent = nextText;
  const card = document.querySelector(`[data-metric="${metric}"]`);
  card.dataset.live = String(value != null);
  const changesNode = $(`${view.prefix}-changes`);
  changesNode.replaceChildren();
  const entries = Object.entries(data?.changes || {});
  if (!entries.length) entries.push(["live", null]);
  for (const [windowName, change] of entries) {
    const chip = document.createElement("span");
    chip.className = `change-chip ${semanticTone(metric, change)}`;
    chip.textContent = change == null ? `${windowName} —` : `${windowName} ${formatSigned(change)} ${view.changeUnit}`;
    changesNode.appendChild(chip);
  }
  const source = $(`${view.prefix}-source`);
  if (source) source.textContent = sourceLabel(data?.source);
  $(`${view.prefix}-time`).textContent = timeAgo(data?.timestamp_utc);
  drawSparkline($(`${view.prefix}-sparkline`), series || [], semanticTone(metric, latestChange(data?.changes)));
}

function updateFedwatchCard(context, metrics, series) {
  const cut = context?.cut_probability;
  const hold = context?.hold_probability;
  const hike = context?.hike_probability;
  $("fed-cut-value").textContent = cut == null ? "—" : Number(cut).toFixed(1);
  $("fed-hold-value").textContent = hold == null ? "—" : Number(hold).toFixed(1);
  $("fed-hike-value").textContent = hike == null ? "—" : Number(hike).toFixed(1);
  $("fed-bias").textContent = context?.policy_bias == null
    ? "—"
    : `${formatSigned(context.policy_bias)} pct`;
  $("fed-expected").textContent = context?.expected_move_bp == null
    ? "—"
    : `${formatSigned(context.expected_move_bp)} bp`;
  const card = document.querySelector('[data-metric="fedwatch_context"]');
  card.dataset.live = String(cut != null && hold != null && hike != null);
  card.dataset.skew = context?.skew || "waiting";
  const signature = [cut, hold, hike, context?.policy_bias].join(":");
  if (lastValues.fedwatch_context !== undefined && lastValues.fedwatch_context !== signature) {
    card.classList.remove("is-updating");
    void card.offsetWidth;
    card.classList.add("is-updating");
  }
  lastValues.fedwatch_context = signature;
  const changesNode = $("fed-changes");
  changesNode.replaceChildren();
  for (const windowName of ["15m", "60m"]) {
    const change = context?.bias_changes?.[windowName];
    const chip = document.createElement("span");
    chip.className = `change-chip ${change == null ? "neutral" : change > 0 ? "dovish" : change < 0 ? "hawkish" : "neutral"}`;
    chip.textContent = change == null ? `Bias ${windowName} —` : `Bias ${windowName} ${formatSigned(change)} pct`;
    changesNode.appendChild(chip);
  }
  const cutMetric = metrics?.fedwatch_cut_probability;
  $("fed-source").textContent = sourceLabel(cutMetric?.source);
  $("fed-time").textContent = timeAgo(cutMetric?.timestamp_utc);
  const biasTone = context?.skew === "dovish" ? "dovish" : context?.skew === "hawkish" ? "hawkish" : "neutral";
  drawSparkline($("fed-sparkline"), series || [], biasTone);
}

function latestChange(changes) {
  if (!changes) return null;
  for (const key of ["5m", "15m", "60m"]) {
    if (changes[key] != null) return changes[key];
  }
  return null;
}

function canvasSize(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, ratio };
}

function drawSparkline(canvas, points, tone) {
  if (!canvas) return;
  const { width, height, ratio } = canvasSize(canvas);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  if (!points || points.length < 2) return;
  const values = points.map((point) => Number(point.value));
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) { min -= .01; max += .01; }
  const colors = { dovish: "#73e5d4", hawkish: "#ff7f78", up: "#f2b868", down: "#72a7ff", neutral: "#81909c" };
  const color = colors[tone] || colors.neutral;
  const pad = 2 * ratio;
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = pad + index / (points.length - 1) * (width - pad * 2);
    const y = pad + (max - Number(point.value)) / (max - min) * (height - pad * 2);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4 * ratio;
  ctx.stroke();
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, `${color}2b`);
  gradient.addColorStop(1, `${color}00`);
  ctx.lineTo(width - pad, height);
  ctx.lineTo(pad, height);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();
}

function updateRegime(regime, metrics, fedwatch) {
  const tone = regime?.code || "waiting";
  $("regime-panel").dataset.tone = tone;
  $("regime-label").textContent = regime?.label || "WAITING FOR DATA";
  $("regime-headline").textContent = regime?.headline || "等待监控后端数据";
  $("regime-detail").textContent = regime?.detail || "页面将自动接收新数据。";
  $("regime-action").textContent = regime?.action || "保持后端采集";
  $("regime-window").textContent = regime?.window || "LIVE";
  $("pricing-level").textContent = fedwatch?.dominant
    ? `${fedwatch.dominant_label} · ${fedwatch.skew_label} · EXPECTED ${formatSigned(fedwatch.expected_move_bp)} BP`
    : "WAITING FOR PRICING LEVEL";

  const fedChange = latestChange(fedwatch?.bias_changes);
  const us2Change = latestChange(metrics?.us2y_yield?.changes);
  updateSignalNode("fed-signal", fedChange == null ? "idle" : fedChange > 0 ? "dovish" : fedChange < 0 ? "hawkish" : "stable", fedChange == null ? "等待窗口" : `Bias ${formatSigned(fedChange)} pct`);
  updateSignalNode("us2-signal", us2Change == null ? "idle" : us2Change < 0 ? "dovish" : us2Change > 0 ? "hawkish" : "stable", us2Change == null ? "等待窗口" : `${formatSigned(us2Change)} bp`);
  const spread = metrics?.us10y_2y_spread?.value;
  updateSignalNode("curve-signal", spread == null ? "idle" : "stable", spread == null ? "等待数据" : `${formatSigned(spread)} bp`);
}

function updateAutomaticRegime(regime) {
  const container = $("auto-regime");
  const direction = regime?.direction || "waiting";
  container.dataset.tone = direction;
  $("auto-regime-label").textContent = regime?.label || "WAITING FOR DATA";
  $("auto-regime-window").textContent = (regime?.window || "15m").toUpperCase();
  const list = $("auto-signal-list");
  list.replaceChildren();
  const stateLabels = {
    hawkish: "鹰派确认",
    dovish: "鸽派确认",
    neutral: "未触发",
    mixed: "方向冲突",
    pending: "待确认",
  };
  (regime?.signals || []).forEach((signal) => {
    const row = document.createElement("div");
    row.className = "auto-signal";
    row.dataset.state = signal.state || "pending";
    const dot = document.createElement("i");
    const label = document.createElement("strong");
    label.textContent = signal.label;
    const detail = document.createElement("span");
    const windowName = regime?.window || "15m";
    let metricText = "";
    if (signal.key === "fedwatch") {
      metricText = signal.value == null ? "Bias —" : `Bias ${formatSigned(signal.value)} pct`;
    } else if (signal.key === "curve") {
      const shape = regime?.curve_shape && regime.curve_shape !== "unchanged"
        ? regime.curve_shape.toUpperCase()
        : "UNCHANGED";
      metricText = `${shape}${signal.change == null ? "" : ` · ${windowName} ${formatSigned(signal.change)} bp`}`;
    } else if (signal.key === "dxy") {
      metricText = signal.value == null ? "DXY —" : `${Number(signal.value).toFixed(3)}${signal.change == null ? "" : ` · ${windowName} ${formatSigned(signal.change, 2)}%`}`;
    } else {
      metricText = signal.value == null ? "—" : `${Number(signal.value).toFixed(3)}%${signal.change == null ? "" : ` · ${windowName} ${formatSigned(signal.change)} bp`}`;
    }
    detail.textContent = `${metricText} · ${stateLabels[signal.state] || "待确认"}`;
    row.append(dot, label, detail);
    list.appendChild(row);
  });
}

function updateSignalNode(id, state, label) {
  const node = $(id);
  node.dataset.state = state;
  node.querySelector("strong").textContent = label;
}

function updateMeeting(meeting) {
  $("meeting-date").textContent = meeting?.date || "—";
  $("meeting-likely").textContent = meeting?.most_likely_range
    ? `主区间 ${meeting.most_likely_range}% · ${Number(meeting.most_likely_probability).toFixed(1)}%`
    : "等待 FedWatch";
}

function updateProbabilities(probabilities, meeting, fedwatch) {
  const list = $("probability-list");
  list.replaceChildren();
  const visible = (probabilities || []).filter((item) => Number(item.probability) > 0.01);
  const total = (probabilities || []).reduce((sum, item) => sum + Number(item.probability), 0);
  $("distribution-total").textContent = probabilities?.length ? `Σ ${total.toFixed(1)}%` : "Σ —";
  $("probability-leader").textContent = meeting?.most_likely_range ? `${meeting.most_likely_range}%` : "—";
  $("probability-expected").textContent = fedwatch?.expected_move_bp == null ? "—" : `${formatSigned(fedwatch.expected_move_bp)} bp`;
  if (!visible.length) {
    const empty = document.createElement("div");
    empty.className = "probability-empty";
    empty.textContent = "等待 FedWatch 分布";
    list.appendChild(empty);
    return;
  }
  const leader = Math.max(...visible.map((item) => Number(item.probability)));
  visible.forEach((item) => {
    const row = document.createElement("div");
    row.className = `probability-row${Number(item.probability) === leader ? " leader" : ""}`;
    const meta = document.createElement("div");
    meta.className = "probability-meta";
    const range = document.createElement("span");
    range.textContent = `${item.target_range}%`;
    const value = document.createElement("strong");
    value.textContent = `${Number(item.probability).toFixed(1)}%`;
    meta.append(range, value);
    const track = document.createElement("div");
    track.className = "probability-track";
    const fill = document.createElement("div");
    fill.className = "probability-fill";
    fill.style.width = `${Math.min(100, Number(item.probability))}%`;
    track.appendChild(fill);
    row.append(meta, track);
    list.appendChild(row);
  });
}

function updateSources(sources) {
  const list = $("source-list");
  list.replaceChildren();
  if (!sources?.length) {
    const row = document.createElement("div");
    row.className = "source-row";
    row.innerHTML = '<span class="health-dot waiting"></span><strong>等待后端</strong><span>—</span><time>—</time>';
    list.appendChild(row);
    return;
  }
  sources.forEach((source) => {
    const row = document.createElement("div");
    row.className = "source-row";
    const dot = document.createElement("span");
    dot.className = `health-dot ${source.state}`;
    const label = document.createElement("strong");
    label.textContent = source.label;
    const provider = document.createElement("span");
    provider.textContent = `${sourceLabel(source.source)}${source.fallback ? " · FALLBACK" : ""}`;
    const time = document.createElement("time");
    time.textContent = timeAgo(source.timestamp_utc);
    row.append(dot, label, provider, time);
    list.appendChild(row);
  });
}

function updateSnapshot(snapshot) {
  currentSnapshot = snapshot;
  const metrics = snapshot.metrics || {};
  updateRegime(snapshot.regime, metrics, snapshot.fedwatch);
  updateAutomaticRegime(snapshot.automatic_regime);
  updateMeeting(snapshot.meeting);
  updateFedwatchCard(snapshot.fedwatch, metrics, snapshot.series?.fedwatch_bias);
  updateMetric("us2y_yield", metrics.us2y_yield, snapshot.series?.us2y);
  updateMetric("us10y_yield", metrics.us10y_yield, snapshot.series?.us10y);
  updateMetric("dxy_index", metrics.dxy_index, snapshot.series?.dxy);
  updateProbabilities(snapshot.probabilities, snapshot.meeting, snapshot.fedwatch);
  updateSources(snapshot.sources);
  $("legend-us2").textContent = metrics.us2y_yield?.value == null ? "—" : `${Number(metrics.us2y_yield.value).toFixed(3)}%`;
  $("legend-us10").textContent = metrics.us10y_yield?.value == null ? "—" : `${Number(metrics.us10y_yield.value).toFixed(3)}%`;
  drawYieldChart();
  setConnection(snapshot.connection?.state === "live" ? "live" : "degraded", snapshot.connection?.state === "live" ? "实时推送中" : "数据需关注");
  $("last-update").textContent = snapshot.connection?.last_observation_utc ? `最新 ${timeAgo(snapshot.connection.last_observation_utc)}` : "等待首笔数据";
}

function filteredSeries(points, minutes) {
  if (!points?.length) return [];
  const cutoff = Date.now() - minutes * 60_000;
  return points.filter((point) => Date.parse(point.timestamp_utc) >= cutoff);
}

function drawYieldChart() {
  const canvas = $("yield-chart");
  const { width, height, ratio } = canvasSize(canvas);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  const us2 = filteredSeries(currentSnapshot?.series?.us2y, chartRangeMinutes);
  const us10 = filteredSeries(currentSnapshot?.series?.us10y, chartRangeMinutes);
  $("chart-empty").hidden = us2.length > 1 || us10.length > 1;
  if (us2.length < 2 && us10.length < 2) { chartGeometry = null; return; }

  const all = [...us2, ...us10];
  let minTime = Math.min(...all.map((point) => Date.parse(point.timestamp_utc)));
  let maxTime = Math.max(...all.map((point) => Date.parse(point.timestamp_utc)));
  let minValue = Math.min(...all.map((point) => Number(point.value)));
  let maxValue = Math.max(...all.map((point) => Number(point.value)));
  if (minTime === maxTime) maxTime += 1000;
  if (minValue === maxValue) { minValue -= .01; maxValue += .01; }
  const valuePad = Math.max((maxValue - minValue) * .14, .006);
  minValue -= valuePad;
  maxValue += valuePad;
  const pad = { left: 48 * ratio, right: 14 * ratio, top: 14 * ratio, bottom: 25 * ratio };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const xFor = (timestamp) => pad.left + (Date.parse(timestamp) - minTime) / (maxTime - minTime) * plotW;
  const yFor = (value) => pad.top + (maxValue - Number(value)) / (maxValue - minValue) * plotH;

  ctx.font = `${9 * ratio}px JetBrains Mono, Consolas, monospace`;
  ctx.textBaseline = "middle";
  for (let index = 0; index <= 4; index += 1) {
    const y = pad.top + index / 4 * plotH;
    const value = maxValue - index / 4 * (maxValue - minValue);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.strokeStyle = "rgba(164,190,208,.10)";
    ctx.lineWidth = ratio;
    ctx.stroke();
    ctx.fillStyle = "#55616c";
    ctx.textAlign = "right";
    ctx.fillText(value.toFixed(3), pad.left - 8 * ratio, y);
  }
  for (let index = 0; index <= 3; index += 1) {
    const timestamp = minTime + index / 3 * (maxTime - minTime);
    const date = new Date(timestamp);
    ctx.fillStyle = "#55616c";
    ctx.textAlign = index === 0 ? "left" : index === 3 ? "right" : "center";
    ctx.fillText(date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: DISPLAY_TIME_ZONE }), pad.left + index / 3 * plotW, height - 8 * ratio);
  }

  const drawLine = (points, color) => {
    if (points.length < 2) return;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = xFor(point.timestamp_utc);
      const y = yFor(point.value);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.7 * ratio;
    ctx.lineJoin = "round";
    ctx.stroke();
  };
  drawLine(us2, "#73e5d4");
  drawLine(us10, "#f2b868");
  chartGeometry = { us2, us10, minTime, maxTime, xFor, yFor, pad, plotW, plotH, ratio };
}

function nearestPoint(points, targetTime) {
  return points.reduce((best, point) => Math.abs(Date.parse(point.timestamp_utc) - targetTime) < Math.abs(Date.parse(best.timestamp_utc) - targetTime) ? point : best, points[0]);
}

function handleChartPointer(event) {
  if (!chartGeometry) return;
  const canvas = $("yield-chart");
  const rect = canvas.getBoundingClientRect();
  const xCss = event.clientX - rect.left;
  const x = xCss * chartGeometry.ratio;
  const target = chartGeometry.minTime + Math.max(0, Math.min(1, (x - chartGeometry.pad.left) / chartGeometry.plotW)) * (chartGeometry.maxTime - chartGeometry.minTime);
  const point2 = chartGeometry.us2.length ? nearestPoint(chartGeometry.us2, target) : null;
  const point10 = chartGeometry.us10.length ? nearestPoint(chartGeometry.us10, target) : null;
  const anchor = point2 || point10;
  if (!anchor) return;
  const tooltip = $("chart-tooltip");
  const time = new Date(Date.parse(anchor.timestamp_utc)).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false, timeZone: DISPLAY_TIME_ZONE });
  tooltip.textContent = `${time}  ·  2Y ${point2 ? Number(point2.value).toFixed(3) : "—"}%  ·  10Y ${point10 ? Number(point10.value).toFixed(3) : "—"}%`;
  tooltip.hidden = false;
  tooltip.style.left = `${Math.min(rect.width - 165, Math.max(0, xCss + 12))}px`;
  tooltip.style.top = "12px";
}

function connectEvents() {
  if (!("EventSource" in window)) {
    setConnection("degraded", "轮询模式");
    setInterval(fetchSnapshot, 3000);
    return;
  }
  const events = new EventSource("/events");
  events.onopen = () => setConnection("live", "实时推送中");
  events.addEventListener("snapshot", (event) => {
    try { updateSnapshot(JSON.parse(event.data)); }
    catch (error) { console.error("Invalid dashboard snapshot", error); }
  });
  events.onerror = () => setConnection("offline", "正在重连");
}

async function fetchSnapshot() {
  try {
    const response = await fetch("/api/snapshot", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    updateSnapshot(await response.json());
  } catch (error) {
    setConnection("offline", "等待服务恢复");
  }
}

function tickClock() {
  $("local-clock").textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false, timeZone: DISPLAY_TIME_ZONE });
  if (currentSnapshot) {
    $("last-update").textContent = currentSnapshot.connection?.last_observation_utc
      ? `北京时间 · 最新 ${timeAgo(currentSnapshot.connection.last_observation_utc)}`
      : "北京时间 · 等待首笔数据";
    for (const metric of Object.values(metricView)) {
      const dataKey = Object.keys(metricView).find((key) => metricView[key] === metric);
      const timestamp = currentSnapshot.metrics?.[dataKey]?.timestamp_utc;
      $(`${metric.prefix}-time`).textContent = timeAgo(timestamp);
    }
    $("fed-time").textContent = timeAgo(currentSnapshot.metrics?.fedwatch_cut_probability?.timestamp_utc);
    updateSources(currentSnapshot.sources);
  }
}

document.querySelectorAll("[data-range]").forEach((button) => {
  button.addEventListener("click", () => {
    chartRangeMinutes = Number(button.dataset.range);
    document.querySelectorAll("[data-range]").forEach((item) => item.classList.toggle("active", item === button));
    drawYieldChart();
  });
});

$("yield-chart").addEventListener("mousemove", handleChartPointer);
$("yield-chart").addEventListener("mouseleave", () => { $("chart-tooltip").hidden = true; });
new ResizeObserver(() => {
  if (!currentSnapshot) return;
  drawYieldChart();
  for (const [metric, view] of Object.entries(metricView)) {
    drawSparkline($(`${view.prefix}-sparkline`), currentSnapshot.series?.[view.series] || [], semanticTone(metric, latestChange(currentSnapshot.metrics?.[metric]?.changes)));
  }
  const context = currentSnapshot.fedwatch;
  drawSparkline($("fed-sparkline"), currentSnapshot.series?.fedwatch_bias || [], context?.skew === "dovish" ? "dovish" : context?.skew === "hawkish" ? "hawkish" : "neutral");
}).observe(document.body);

setInterval(tickClock, 1000);
tickClock();
fetchSnapshot();
connectEvents();
