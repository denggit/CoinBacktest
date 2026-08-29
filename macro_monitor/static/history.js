"use strict";

const DISPLAY_TIME_ZONE = "Asia/Shanghai";

const METRICS = {
  fedwatch_bias: {
    title: "FedWatch Policy Bias",
    subtitle: "Cut 概率 − Hike 概率。正值代表鸽派倾向，负值代表鹰派倾向。",
    color: "#73e5d4",
  },
  us2y_yield: {
    title: "US 2Y Treasury Yield",
    subtitle: "美国 2 年期国债收益率；区间变化统一换算为 basis points。",
    color: "#72a7ff",
  },
  us10y_yield: {
    title: "US 10Y Treasury Yield",
    subtitle: "美国 10 年期国债收益率；区间变化统一换算为 basis points。",
    color: "#f2b868",
  },
  dxy_index: {
    title: "US Dollar Index",
    subtitle: "美元指数 DXY；区间变化显示为相对百分比。",
    color: "#bc9cff",
  },
};

const VALID_RANGES = new Set(["4h", "24h", "7d", "30d"]);
const params = new URLSearchParams(window.location.search);
const requestedMetric = params.get("metric");
const requestedRange = (params.get("range") || "24h").toLowerCase();

const state = {
  metric: Object.hasOwn(METRICS, requestedMetric) ? requestedMetric : "fedwatch_bias",
  range: VALID_RANGES.has(requestedRange) ? requestedRange : "24h",
  payload: null,
  loading: false,
  hoverIndex: null,
};

const elements = {
  status: document.getElementById("history-status"),
  statusText: document.getElementById("history-status-text"),
  clock: document.getElementById("history-clock"),
  title: document.getElementById("history-title"),
  subtitle: document.getElementById("history-subtitle"),
  current: document.getElementById("history-current"),
  unit: document.getElementById("history-unit"),
  change: document.getElementById("history-change"),
  changeUnit: document.getElementById("history-change-unit"),
  minimum: document.getElementById("history-min"),
  minimumUnit: document.getElementById("history-min-unit"),
  maximum: document.getElementById("history-max"),
  maximumUnit: document.getElementById("history-max-unit"),
  chartTitle: document.getElementById("history-chart-title"),
  source: document.getElementById("history-source"),
  latest: document.getElementById("history-latest"),
  chart: document.getElementById("history-chart"),
  empty: document.getElementById("history-empty"),
  tooltip: document.getElementById("history-tooltip"),
  coverage: document.getElementById("history-coverage"),
  sampling: document.getElementById("history-sampling"),
};

function updateClock() {
  elements.clock.textContent = new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false, timeZone: DISPLAY_TIME_ZONE,
  }).format(new Date());
}

function setStatus(status, text) {
  elements.status.dataset.state = status;
  elements.statusText.textContent = text;
}

function updateControls() {
  document.querySelectorAll("[data-history-metric]").forEach((link) => {
    const metric = link.dataset.historyMetric;
    link.classList.toggle("active", metric === state.metric);
    link.setAttribute("aria-current", metric === state.metric ? "page" : "false");
    link.href = `/history?metric=${encodeURIComponent(metric)}&range=${encodeURIComponent(state.range)}`;
  });
  document.querySelectorAll("[data-history-range]").forEach((button) => {
    const active = button.dataset.historyRange === state.range;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  const url = new URL(window.location.href);
  url.searchParams.set("metric", state.metric);
  url.searchParams.set("range", state.range);
  window.history.replaceState(null, "", url);
}

function formatValue(value, payload, signed = false) {
  if (!Number.isFinite(value)) return "—";
  const decimals = Number.isInteger(payload?.decimals) ? payload.decimals : 2;
  const prefix = signed && value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(decimals)}`;
}

function formatChange(value, unit) {
  if (!Number.isFinite(value)) return "—";
  const decimals = unit === "bp" ? 1 : 2;
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(decimals)}`;
}

function formatTimestamp(value, includeDate = true) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    ...(includeDate ? { month: "2-digit", day: "2-digit" } : {}),
    hour: "2-digit", minute: "2-digit", second: includeDate ? undefined : "2-digit", hour12: false,
    timeZone: DISPLAY_TIME_ZONE,
  }).format(date);
}

function changeTone(value) {
  if (!Number.isFinite(value) || Math.abs(value) < 1e-9) return "neutral";
  if (state.metric === "fedwatch_bias") return value > 0 ? "dovish" : "hawkish";
  if (state.metric === "us2y_yield" || state.metric === "us10y_yield") return value > 0 ? "hawkish" : "dovish";
  return value > 0 ? "up" : "down";
}

function renderSummary(payload) {
  const config = METRICS[state.metric];
  elements.title.textContent = config.title;
  elements.subtitle.textContent = config.subtitle;
  elements.chartTitle.textContent = `${config.title} · ${state.range.toUpperCase()}`;
  const signedCurrent = state.metric === "fedwatch_bias";
  elements.current.textContent = formatValue(payload.current, payload, signedCurrent);
  elements.unit.textContent = payload.unit || "—";
  elements.change.textContent = formatChange(payload.period_change, payload.change_unit);
  elements.changeUnit.textContent = payload.change_unit || "—";
  elements.minimum.textContent = formatValue(payload.minimum, payload, signedCurrent);
  elements.minimumUnit.textContent = payload.unit || "—";
  elements.maximum.textContent = formatValue(payload.maximum, payload, signedCurrent);
  elements.maximumUnit.textContent = payload.unit || "—";
  elements.change.closest("article").dataset.tone = changeTone(payload.period_change);
  elements.source.textContent = payload.source ? String(payload.source).replaceAll("_", " ").toUpperCase() : "暂无来源";
  elements.latest.textContent = formatTimestamp(payload.last_timestamp_utc);
  elements.coverage.textContent = payload.first_timestamp_utc
    ? `${formatTimestamp(payload.first_timestamp_utc)} → ${formatTimestamp(payload.last_timestamp_utc)}`
    : "所选范围暂无覆盖数据";
  elements.sampling.textContent = payload.raw_count
    ? `原始 ${payload.raw_count.toLocaleString("zh-CN")} 点 · 展示 ${payload.returned_count.toLocaleString("zh-CN")} 点${payload.meeting_date ? ` · FOMC ${payload.meeting_date}` : ""}`
    : "等待后台积累历史";
}

function chartGeometry(payload) {
  const rect = elements.chart.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const padding = { left: width < 520 ? 52 : 70, right: 22, top: 24, bottom: 42 };
  const plotWidth = Math.max(1, width - padding.left - padding.right);
  const plotHeight = Math.max(1, height - padding.top - padding.bottom);
  const points = payload?.points || [];
  const times = points.map((point) => new Date(point.timestamp_utc).getTime());
  const values = points.map((point) => Number(point.value));
  let xMin = times[0] || 0;
  let xMax = times[times.length - 1] || xMin + 1;
  if (xMin === xMax) xMax = xMin + 1;
  let yMin = values.length ? Math.min(...values) : 0;
  let yMax = values.length ? Math.max(...values) : 1;
  if (yMin === yMax) {
    const pad = Math.max(Math.abs(yMin) * .001, state.metric === "fedwatch_bias" ? .5 : .005);
    yMin -= pad;
    yMax += pad;
  } else {
    const pad = (yMax - yMin) * .1;
    yMin -= pad;
    yMax += pad;
  }
  const xFor = (time) => padding.left + ((time - xMin) / (xMax - xMin)) * plotWidth;
  const yFor = (value) => padding.top + ((yMax - value) / (yMax - yMin)) * plotHeight;
  return { width, height, padding, plotWidth, plotHeight, points, times, values, xMin, xMax, yMin, yMax, xFor, yFor };
}

function drawChart() {
  const payload = state.payload;
  const rect = elements.chart.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  elements.chart.width = Math.max(1, Math.round(rect.width * ratio));
  elements.chart.height = Math.max(1, Math.round(rect.height * ratio));
  const context = elements.chart.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  if (!payload?.ready || !payload.points?.length) return;

  const geometry = chartGeometry(payload);
  const { width, height, padding, plotWidth, plotHeight, points, times, values, xFor, yFor, yMin, yMax } = geometry;
  context.font = '9px "JetBrains Mono", Consolas, monospace';
  context.lineWidth = 1;
  context.textBaseline = "middle";

  for (let index = 0; index <= 4; index += 1) {
    const ratioY = index / 4;
    const y = padding.top + ratioY * plotHeight;
    context.strokeStyle = "rgba(164,190,208,.10)";
    context.beginPath(); context.moveTo(padding.left, y); context.lineTo(width - padding.right, y); context.stroke();
    const labelValue = yMax - ratioY * (yMax - yMin);
    context.fillStyle = "rgba(129,144,156,.82)";
    context.textAlign = "right";
    context.fillText(labelValue.toFixed(Number(payload.decimals)), padding.left - 10, y);
  }

  const xTicks = width < 620 ? 3 : 6;
  context.textBaseline = "top";
  for (let index = 0; index < xTicks; index += 1) {
    const tickRatio = xTicks === 1 ? 0 : index / (xTicks - 1);
    const timestamp = geometry.xMin + tickRatio * (geometry.xMax - geometry.xMin);
    const x = padding.left + tickRatio * plotWidth;
    context.strokeStyle = "rgba(164,190,208,.055)";
    context.beginPath(); context.moveTo(x, padding.top); context.lineTo(x, height - padding.bottom); context.stroke();
    context.fillStyle = "rgba(85,97,108,.95)";
    context.textAlign = index === 0 ? "left" : index === xTicks - 1 ? "right" : "center";
    const longRange = state.range === "7d" || state.range === "30d";
    const label = new Intl.DateTimeFormat("zh-CN", longRange
      ? { month: "2-digit", day: "2-digit", hour: "2-digit", hour12: false, timeZone: DISPLAY_TIME_ZONE }
      : { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: DISPLAY_TIME_ZONE }).format(new Date(timestamp));
    context.fillText(label, x, height - padding.bottom + 14);
  }

  const color = METRICS[state.metric].color;
  const gradient = context.createLinearGradient(0, padding.top, 0, height - padding.bottom);
  gradient.addColorStop(0, `${color}35`);
  gradient.addColorStop(1, `${color}00`);
  context.beginPath();
  points.forEach((point, index) => {
    const x = xFor(times[index]);
    const y = yFor(values[index]);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.lineTo(xFor(times[times.length - 1]), height - padding.bottom);
  context.lineTo(xFor(times[0]), height - padding.bottom);
  context.closePath();
  context.fillStyle = gradient;
  context.fill();

  context.beginPath();
  points.forEach((point, index) => {
    const x = xFor(times[index]);
    const y = yFor(values[index]);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.strokeStyle = color;
  context.lineWidth = 1.7;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.stroke();

  const last = points.length - 1;
  context.fillStyle = color;
  context.beginPath(); context.arc(xFor(times[last]), yFor(values[last]), 3.3, 0, Math.PI * 2); context.fill();

  if (Number.isInteger(state.hoverIndex) && state.hoverIndex >= 0 && state.hoverIndex < points.length) {
    const index = state.hoverIndex;
    const x = xFor(times[index]);
    const y = yFor(values[index]);
    context.strokeStyle = "rgba(237,245,246,.35)";
    context.lineWidth = 1;
    context.beginPath(); context.moveTo(x, padding.top); context.lineTo(x, height - padding.bottom); context.stroke();
    context.fillStyle = "#07090c";
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath(); context.arc(x, y, 4.5, 0, Math.PI * 2); context.fill(); context.stroke();
  }
}

function nearestPointIndex(pointerX) {
  const geometry = chartGeometry(state.payload);
  if (!geometry.points.length) return null;
  const targetTime = geometry.xMin + ((pointerX - geometry.padding.left) / geometry.plotWidth) * (geometry.xMax - geometry.xMin);
  let low = 0;
  let high = geometry.times.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (geometry.times[middle] < targetTime) low = middle + 1; else high = middle;
  }
  if (low > 0 && Math.abs(geometry.times[low - 1] - targetTime) < Math.abs(geometry.times[low] - targetTime)) return low - 1;
  return low;
}

function showTooltip(event) {
  if (!state.payload?.ready) return;
  const rect = elements.chart.getBoundingClientRect();
  const pointerX = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
  const index = nearestPointIndex(pointerX);
  if (index === null) return;
  state.hoverIndex = index;
  const point = state.payload.points[index];
  const signed = state.metric === "fedwatch_bias";
  elements.tooltip.textContent = `${formatTimestamp(point.timestamp_utc)}\n${formatValue(Number(point.value), state.payload, signed)} ${state.payload.unit}`;
  elements.tooltip.hidden = false;
  const tipWidth = 170;
  elements.tooltip.style.left = `${Math.max(8, Math.min(rect.width - tipWidth - 8, pointerX + 14))}px`;
  elements.tooltip.style.top = `${Math.max(8, event.clientY - rect.top - 58)}px`;
  drawChart();
}

function hideTooltip() {
  state.hoverIndex = null;
  elements.tooltip.hidden = true;
  drawChart();
}

async function loadHistory() {
  if (state.loading) return;
  state.loading = true;
  setStatus("connecting", "正在读取历史");
  try {
    const response = await fetch(`/api/history?metric=${encodeURIComponent(state.metric)}&range=${encodeURIComponent(state.range)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    state.payload = payload;
    state.hoverIndex = null;
    renderSummary(payload);
    elements.empty.hidden = Boolean(payload.ready);
    elements.empty.textContent = payload.detail || "所选范围暂无历史数据";
    elements.tooltip.hidden = true;
    drawChart();
    setStatus(payload.ready ? "live" : "degraded", payload.ready ? `已同步 · ${state.range.toUpperCase()}` : "等待历史数据");
  } catch (error) {
    elements.empty.hidden = false;
    elements.empty.textContent = "历史数据暂时无法读取，页面会自动重试";
    setStatus("offline", "读取失败 · 自动重试");
  } finally {
    state.loading = false;
  }
}

document.querySelectorAll("[data-history-range]").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.dataset.historyRange === state.range) return;
    state.range = button.dataset.historyRange;
    state.payload = null;
    updateControls();
    loadHistory();
  });
});

elements.chart.addEventListener("pointermove", showTooltip);
elements.chart.addEventListener("pointerleave", hideTooltip);
new ResizeObserver(drawChart).observe(elements.chart);

updateControls();
updateClock();
setInterval(updateClock, 1000);
loadHistory();
setInterval(loadHistory, 15000);
