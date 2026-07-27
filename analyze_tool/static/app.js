const state = {
  config: null,
  plugins: [],
  candles: [],
  markers: [],
  regions: [],
  tracks: [],
  bands: [],
  heatmap: [],
  priceRegions: [],
  showWallOverlay: false,
  heatmapByBar: new Map(),
  heatmapMaxDurationMs: 0,
  heatmapLayerCanvas: null,
  heatmapLayerKey: '',
  rowFields: {},
  pluginUi: {},
  timestampIndex: new Map(),
  visibleStart: 0,
  visibleEnd: 200,
  hoverIndex: null,
  selectedIndex: null,
  dragging: false,
  dragX: 0,
  dragY: 0,
  dragStartStart: 0,
  dragStartEnd: 0,
  dragMoved: false,
  dragThreshold: 6,
  dragMode: null,
  yMinManual: null,
  yMaxManual: null,
  priceDragAnchorFrac: 0.5,
  priceDragAnchorPrice: null,
  priceDragStartMin: null,
  priceDragStartMax: null,
  initialVisibleSpan: 320,
  heatmapColorMinPct: 0,
  heatmapColorMaxPct: 50,
  hoverHeatmapCells: [],
  selectedHeatmapCells: [],
  hoverPrice: null,
  alignmentAudit: null,
};

const $ = (id) => document.getElementById(id);
const canvas = $('chartCanvas');
const ctx = canvas.getContext('2d');
const tooltip = $('tooltip');

function setStatus(text, isError = false) {
  const el = $('status');
  el.textContent = text;
  el.style.color = isError ? '#fca5a5' : '#93a4b8';
}

function fmt(n, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return '-';
  const x = Number(n);
  if (Math.abs(x) >= 1000) return x.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return x.toLocaleString(undefined, { maximumFractionDigits: digits });
}


function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}


function heatmapCellPassesColorFloor(cell) {
  const raw = clamp(Number(cell?.intensity || 0), 0, 1);
  const min = clamp(Number(state.heatmapColorMinPct || 0) / 100, 0, 0.95);
  return raw > 0 && raw + 1e-12 >= min;
}
function effectiveHeatmapIntensity(cell) {
  const raw = clamp(Number(cell?.intensity || 0), 0, 1);
  const min = clamp(Number(state.heatmapColorMinPct || 0) / 100, 0, 0.95);
  const max = clamp(Number(state.heatmapColorMaxPct || 100) / 100, min + 0.01, 1.0);
  return clamp((raw - min) / Math.max(max - min, 1e-9), 0, 1);
}

function updateHeatmapColorLabels() {
  const minEl = $('heatmapColorMinValue');
  const maxEl = $('heatmapColorMaxValue');
  if (minEl) minEl.textContent = `${Math.round(state.heatmapColorMinPct)}%`;
  if (maxEl) maxEl.textContent = `${Math.round(state.heatmapColorMaxPct)}%`;
}

function setHeatmapColorRange(minPct, maxPct) {
  let min = clamp(Number(minPct), 0, 95);
  let max = clamp(Number(maxPct), 5, 100);
  if (max <= min + 1) {
    if (document.activeElement?.id === 'heatmapColorMin') min = Math.max(0, max - 1);
    else max = Math.min(100, min + 1);
  }
  state.heatmapColorMinPct = min;
  state.heatmapColorMaxPct = max;
  if ($('heatmapColorMin')) $('heatmapColorMin').value = String(min);
  if ($('heatmapColorMax')) $('heatmapColorMax').value = String(max);
  updateHeatmapColorLabels();
  state.heatmapLayerKey = '';
  draw();
}

function updateAlignmentAuditBadge(audit) {
  state.alignmentAudit = audit || null;
  const badge = $('alignmentAuditBadge');
  if (!badge) return;
  if (!audit) {
    badge.className = 'alignment-audit-badge hidden';
    badge.textContent = '对齐未检查';
    badge.removeAttribute('title');
    return;
  }
  const status = String(audit.status || 'unavailable');
  badge.className = `alignment-audit-badge ${status === 'pass' ? 'pass' : status === 'warning' ? 'warning' : 'unavailable'}`;
  const checked = Number(audit.checked_bars || 0);
  badge.textContent = status === 'pass' ? `对齐通过 · ${checked} bars` : status === 'warning' ? `对齐警告 · ${checked} bars` : `对齐仅时间检查 · ${checked} bars`;
  const details = [
    `时间P95延迟 ${fmt(audit.time_lag_p95_ms, 0)} ms`,
    `价格错位 ${audit.price_mismatches ?? '-'} / ${audit.price_checks ?? '-'}`,
    `Close-Mid P95 ${fmt(audit.close_mid_p95_bps, 3)} bps`,
    `允许价格格误差 ≤ $${fmt(audit.max_allowed_price_error, 4)}`,
  ];
  badge.title = details.join(' · ');
}

function htmlEscape(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function getJson(url, options = {}) {
  const res = await fetch(url, options);
  const text = await res.text();
  if (!text.trim()) {
    throw new Error(`后端返回空响应（HTTP ${res.status}）。服务进程可能因内存不足或连接中断而退出，请查看启动窗口日志。`);
  }
  let data;
  try {
    data = JSON.parse(text);
  } catch (err) {
    const preview = text.slice(0, 180).replace(/\s+/g, ' ');
    throw new Error(`后端JSON响应不完整（HTTP ${res.status}，收到 ${text.length.toLocaleString()} 字符）：${preview || err.message}`);
  }
  if (!res.ok || data.ok === false) {
    const msg = data.error || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function qs(params) {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') u.set(k, String(v));
  }
  return u.toString();
}

function currentDataParams() {
  return {
    data_type: $('dataType').value,
    symbol: $('symbol').value.trim(),
    timeframe: $('timeframe').value,
    range_pct: $('rangePct').value,
    start: $('start').value,
    end: $('end').value,
    limit: $('limit').value,
    local_only: $('localOnly').checked ? '1' : '0',
  };
}

function parseTimestampMs(value) {
  if (value === null || value === undefined) return NaN;
  if (typeof value === 'number') return Number(value);
  const text = String(value).trim();
  if (!text) return NaN;

  // Analyze Tool displays the project's UTC+8 wall-clock timestamps as naive
  // strings.  Treat those strings as a timezone-free chart coordinate instead
  // of asking the browser to apply its local timezone again.  This keeps Kline,
  // heatmap cells, markers and date navigation on exactly the same axis on every
  // operating system and browser timezone.
  const explicitZone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(text);
  if (explicitZone) {
    const parsed = Date.parse(text.replace(' ', 'T'));
    return Number.isFinite(parsed) ? parsed : NaN;
  }
  const match = /^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2})(?::(\d{2})(?::(\d{2})(?:\.(\d{1,3}))?)?)?)?$/.exec(text);
  if (!match) return NaN;
  const [, year, month, day, hour = '0', minute = '0', second = '0', millis = '0'] = match;
  return Date.UTC(
    Number(year), Number(month) - 1, Number(day),
    Number(hour), Number(minute), Number(second), Number(String(millis).padEnd(3, '0')),
  );
}

function localDateTimeInputValue(date) {
  const d = date instanceof Date ? date : new Date(date);
  if (!Number.isFinite(d.getTime())) return '';
  const pad = n => String(n).padStart(2, '0');
  // The chart axis is a project wall-clock coordinate, so use UTC getters to
  // round-trip the numeric coordinate without another browser timezone shift.
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}T${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
}

function candleIntervalMs() {
  if (state.candles.length >= 2) {
    const diffs = [];
    for (let i = 1; i < Math.min(state.candles.length, 100); i++) {
      const d = Number(state.candles[i].time) - Number(state.candles[i - 1].time);
      if (Number.isFinite(d) && d > 0) diffs.push(d);
    }
    if (diffs.length) {
      diffs.sort((a, b) => a - b);
      return diffs[Math.floor(diffs.length / 2)];
    }
  }
  const tf = $('timeframe')?.value || '1m';
  const match = /^(\d+)(s|m|h|d)$/i.exec(tf);
  if (!match) return 60000;
  const value = Number(match[1]);
  const unit = match[2].toLowerCase();
  return value * ({ s: 1000, m: 60000, h: 3600000, d: 86400000 }[unit] || 60000);
}

function visibleTimeRangeMs() {
  if (!state.candles.length || state.visibleEnd <= state.visibleStart) return { startMs: 0, endMs: 1 };
  const interval = candleIntervalMs();
  const startMs = Number(state.candles[state.visibleStart]?.time);
  const endMs = Number(state.candles[state.visibleEnd - 1]?.time) + interval;
  return { startMs, endMs: Math.max(startMs + 1, endMs) };
}

function xForTimestampMs(timestampMs, plot) {
  const { startMs, endMs } = visibleTimeRangeMs();
  return plot.left + ((timestampMs - startMs) / Math.max(1, endMs - startMs)) * plot.width;
}

function nearestCandleIndex(timestampMs) {
  if (!state.candles.length || !Number.isFinite(timestampMs)) return null;
  let lo = 0, hi = state.candles.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (Number(state.candles[mid].time) < timestampMs) lo = mid + 1;
    else hi = mid;
  }
  if (lo <= 0) return 0;
  if (lo >= state.candles.length) return state.candles.length - 1;
  const before = Number(state.candles[lo - 1].time);
  const after = Number(state.candles[lo].time);
  return Math.abs(timestampMs - before) <= Math.abs(after - timestampMs) ? lo - 1 : lo;
}

function lowerBoundCandleTime(timestampMs) {
  let lo = 0, hi = state.candles.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (Number(state.candles[mid].time) < timestampMs) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function rebuildHeatmapIndex() {
  const map = new Map();
  if (!state.candles.length || !state.heatmap.length) {
    state.heatmapByBar = map;
    return;
  }
  const interval = candleIntervalMs();
  for (const cell of state.heatmap) {
    let first = lowerBoundCandleTime(cell._startMs);
    if (first > 0 && Number(state.candles[first]?.time) > cell._startMs) first -= 1;
    let last = lowerBoundCandleTime(cell._endMs) - 1;
    first = Math.max(0, first);
    last = Math.min(state.candles.length - 1, Math.max(first, last));
    for (let i = first; i <= last; i++) {
      const barStart = Number(state.candles[i].time);
      const barEnd = barStart + interval;
      if (cell._endMs <= barStart || cell._startMs >= barEnd) continue;
      if (!map.has(i)) map.set(i, []);
      map.get(i).push(cell);
    }
  }
  state.heatmapByBar = map;
}

function lowerBoundHeatmapStart(timestampMs) {
  let lo = 0, hi = state.heatmap.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (state.heatmap[mid]._startMs < timestampMs) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function setVisibleCentered(index) {
  if (!state.candles.length || index === null) return;
  const span = Math.max(1, state.visibleEnd - state.visibleStart || state.initialVisibleSpan || 320);
  state.visibleStart = Math.round(index - span / 2);
  state.visibleEnd = state.visibleStart + span;
  clampVisible();
}

function resetChartView() {
  if (!state.candles.length) return;
  const span = Math.min(state.candles.length, Math.max(20, state.initialVisibleSpan || 320));
  state.visibleEnd = state.candles.length;
  state.visibleStart = Math.max(0, state.visibleEnd - span);
  resetPriceViewport();
  state.selectedIndex = null;
  updateDetails(null);
  draw();
}

function jumpToEdge(edge) {
  if (!state.candles.length) return;
  const span = Math.max(1, state.visibleEnd - state.visibleStart || state.initialVisibleSpan || 320);
  if (edge === 'start') {
    state.visibleStart = 0;
    state.visibleEnd = Math.min(state.candles.length, span);
  } else {
    state.visibleEnd = state.candles.length;
    state.visibleStart = Math.max(0, state.visibleEnd - span);
  }
  resetPriceViewport();
  draw();
}

function jumpToInputDate() {
  if (!state.candles.length) return;
  const raw = $('jumpDate').value;
  const target = parseTimestampMs(raw);
  if (!Number.isFinite(target)) {
    setStatus('请输入有效的跳转日期时间。', true);
    return;
  }
  const index = nearestCandleIndex(target);
  setVisibleCentered(index);
  resetPriceViewport();
  state.selectedIndex = index;
  updateDetails(index);
  draw();
}

async function init() {
  state.config = await getJson('/api/config');
  const dataType = $('dataType');
  dataType.innerHTML = state.config.data_types.map(x => `<option value="${x.id}">${htmlEscape(x.label)}</option>`).join('');
  dataType.value = state.config.defaults.data_type;
  $('symbol').value = state.config.defaults.symbol;
  $('rangePct').value = state.config.defaults.range_pct;
  fillTimeframes();
  dataType.addEventListener('change', () => { fillTimeframes(); renderPluginParams(); });

  const pluginData = await getJson('/api/plugins');
  state.plugins = pluginData.plugins || [];
  $('pluginSelect').innerHTML = state.plugins.map(p => `<option value="${p.id}">${htmlEscape(p.name)}</option>`).join('');
  $('pluginSelect').addEventListener('change', renderPluginParams);
  renderPluginParams();

  $('loadBtn').addEventListener('click', loadCandles);
  $('runPluginBtn').addEventListener('click', runPlugin);
  $('clearMarkersBtn').addEventListener('click', () => { state.markers = []; state.regions = []; state.tracks = []; state.bands = []; state.heatmap = []; state.priceRegions = []; state.showWallOverlay = false; state.heatmapByBar = new Map(); state.heatmapMaxDurationMs = 0; state.heatmapLayerKey = ''; state.rowFields = {}; state.pluginUi = {}; state.hoverHeatmapCells = []; state.selectedHeatmapCells = []; state.alignmentAudit = null; $('pluginSummary').textContent = ''; $('heatmapColorControls')?.classList.add('hidden'); $('wallOverlayControl')?.classList.add('hidden'); if ($('wallOverlayToggle')) $('wallOverlayToggle').checked = false; updateAlignmentAuditBadge(null); renderHeatmapCellDetail([]); updateDetails(state.selectedIndex); draw(); });
  $('resetViewBtn').addEventListener('click', resetChartView);
  $('autoPriceBtn').addEventListener('click', () => { resetPriceViewport(); draw(); });
  $('goStartBtn').addEventListener('click', () => jumpToEdge('start'));
  $('goEndBtn').addEventListener('click', () => jumpToEdge('end'));
  $('jumpDateBtn').addEventListener('click', jumpToInputDate);
  $('jumpDate').addEventListener('keydown', (e) => { if (e.key === 'Enter') jumpToInputDate(); });
  $('heatmapColorMin')?.addEventListener('input', (e) => setHeatmapColorRange(e.target.value, state.heatmapColorMaxPct));
  $('heatmapColorMax')?.addEventListener('input', (e) => setHeatmapColorRange(state.heatmapColorMinPct, e.target.value));
  $('resetHeatmapColorBtn')?.addEventListener('click', () => setHeatmapColorRange(Number(state.pluginUi.heatmap_color_min_pct ?? 0), Number(state.pluginUi.heatmap_color_max_pct ?? 50)));
  $('wallOverlayToggle')?.addEventListener('change', (e) => { state.showWallOverlay = Boolean(e.target.checked); draw(); });
  updateHeatmapColorLabels();
  setupCanvas();
  resizeCanvas();
  window.addEventListener('resize', () => { resizeCanvas(); draw(); });
  draw();
}

function fillTimeframes() {
  const dt = $('dataType').value;
  const isRange = dt === 'range_bar';
  $('timeframeWrap').classList.toggle('hidden', isRange);
  $('rangeWrap').classList.toggle('hidden', !isRange);
  const tf = $('timeframe');
  const list = state.config.timeframes[dt] || [];
  tf.innerHTML = list.map(x => `<option value="${x}">${x}</option>`).join('');
  tf.value = list.includes('1m') ? '1m' : list[0];
}

function selectedPlugin() {
  return state.plugins.find(p => p.id === $('pluginSelect').value);
}

function renderParamControl(param) {
  const id = `plugin_${param.name}`;
  if (param.kind === 'select') {
    const opts = (param.choices || []).map(c => `<option value="${htmlEscape(c.value)}" ${c.value === param.default ? 'selected' : ''}>${htmlEscape(c.label)}</option>`).join('');
    return `<label>${htmlEscape(param.label)}<select id="${id}" data-param="${htmlEscape(param.name)}">${opts}</select></label>`;
  }
  if (param.kind === 'color') {
    return `<label>${htmlEscape(param.label)}<input id="${id}" data-param="${htmlEscape(param.name)}" type="color" value="${htmlEscape(param.default || '#facc15')}" /></label>`;
  }
  return `<label>${htmlEscape(param.label)}<input id="${id}" data-param="${htmlEscape(param.name)}" type="number" value="${htmlEscape(param.default)}" min="${param.min ?? ''}" max="${param.max ?? ''}" step="${param.step ?? 'any'}" /></label>`;
}

function renderPluginParams() {
  const p = selectedPlugin();
  const wrap = $('pluginParams');
  if (!p) { wrap.innerHTML = '<div class="mini">暂无插件</div>'; return; }
  const params = p.params || [];
  const compactPlugins = new Set(['market_state_map_v0', 'estimated_liquidation_heatmap_v1', 'offline_orderbook_liquidity_heatmap_v1']);
  if (!compactPlugins.has(p.id)) {
    wrap.innerHTML = params.map(renderParamControl).join('');
    return;
  }
  let primaryNames;
  if (p.id === 'estimated_liquidation_heatmap_v1') {
    primaryNames = new Set(['display_mode']);
  } else if (p.id === 'offline_orderbook_liquidity_heatmap_v1') {
    primaryNames = new Set(['display_mode', 'normalization', 'depth_unit', 'manual_max', 'large_window_hours', 'large_percentile', 'display_price_step']);
  } else {
    primaryNames = new Set(['view_mode', 'show_watch_markers']);
  }
  const primary = params.filter(param => primaryNames.has(param.name));
  const advanced = params.filter(param => !primaryNames.has(param.name));
  wrap.innerHTML = `${primary.map(renderParamControl).join('')}
    <details class="plugin-advanced">
      <summary>高级参数</summary>
      <div class="plugin-advanced-body">${advanced.map(renderParamControl).join('')}</div>
    </details>`;
}

function pluginParams() {
  const out = {};
  document.querySelectorAll('#pluginParams [data-param]').forEach(el => {
    out[el.dataset.param] = el.value;
  });
  return out;
}

async function loadCandles() {
  try {
    setStatus('加载中...');
    const data = await getJson('/api/candles?' + qs(currentDataParams()));
    state.candles = (data.candles || []).map(candle => {
      const chartTime = parseTimestampMs(candle.timestamp);
      return { ...candle, source_time: candle.time, time: chartTime };
    }).filter(candle => Number.isFinite(Number(candle.time)))
      .sort((a, b) => Number(a.time) - Number(b.time));
    state.markers = [];
    state.regions = [];
    state.tracks = [];
    state.bands = [];
    state.heatmap = [];
    state.priceRegions = [];
    state.showWallOverlay = false;
    state.heatmapByBar = new Map();
    state.heatmapMaxDurationMs = 0;
    state.heatmapLayerKey = '';
    state.rowFields = {};
    state.pluginUi = {};
    state.hoverHeatmapCells = [];
    state.selectedHeatmapCells = [];
    state.hoverPrice = null;
    state.alignmentAudit = null;
    $('heatmapColorControls')?.classList.add('hidden');
    $('wallOverlayControl')?.classList.add('hidden');
    if ($('wallOverlayToggle')) $('wallOverlayToggle').checked = false;
    updateAlignmentAuditBadge(null);
    renderHeatmapCellDetail([]);
    state.timestampIndex = new Map(state.candles.map((c, i) => [c.timestamp, i]));
    state.hoverIndex = null;
    state.selectedIndex = null;
    const n = state.candles.length;
    state.initialVisibleSpan = Math.min(320, Math.max(1, n));
    state.visibleEnd = n;
    state.visibleStart = Math.max(0, n - state.initialVisibleSpan);
    state.yMinManual = null;
    state.yMaxManual = null;
    if (n) {
      const earliest = new Date(Number(state.candles[0].time));
      const latest = new Date(Number(state.candles[n - 1].time));
      $('jumpDate').min = localDateTimeInputValue(earliest);
      $('jumpDate').max = localDateTimeInputValue(latest);
      $('jumpDate').value = localDateTimeInputValue(latest);
    } else {
      $('jumpDate').min = '';
      $('jumpDate').max = '';
      $('jumpDate').value = '';
    }
    $('chartTitle').textContent = `${data.meta.symbol} · ${data.meta.data_type}`;
    $('chartSub').textContent = n ? `${data.meta.start} → ${data.meta.end} · ${n} bars · ${data.meta.table_name || ''}` : `无数据：${data.meta.table_name || ''}`;
    setStatus(n ? `已加载 ${n} 根，来源 ${data.meta.loader}` : '没有读到数据。确认本地 DB 覆盖，或取消“只读本地缓存”让 loader 自动补数据。', n === 0);
    updateDetails(null);
    draw();
  } catch (err) {
    setStatus(err.message, true);
  }
}

function decodeCompactHeatmap(payload) {
  if (!payload || Number(payload.v) !== 1) return [];
  const starts = payload.starts || [];
  const ends = payload.ends || [];
  const sourceStarts = payload.source_starts || [];
  const sourceEnds = payload.source_ends || [];
  const sourceLags = payload.source_lags || [];
  const columns = payload.c || [];
  const prices = payload.p || [];
  const intensities = payload.i || [];
  const sides = payload.s || [];
  const depths = payload.d || [];
  const orders = payload.o || [];
  const large = payload.l || [];
  const count = Math.min(columns.length, prices.length, intensities.length, sides.length, depths.length, orders.length);
  const step = Math.max(Number(payload.step || 1), Number.EPSILON);
  const scale = Math.max(Number(payload.scale || 10000), 1);
  const unit = payload.unit || 'ETH';
  const colorMode = payload.color_mode || 'single';
  const cells = new Array(count);
  for (let i = 0; i < count; i++) {
    const columnIndex = Number(columns[i]);
    const startTimestamp = starts[columnIndex];
    const endTimestamp = ends[columnIndex];
    const startMs = parseTimestampMs(startTimestamp);
    const endMs = parseTimestampMs(endTimestamp);
    const priceLow = Number(prices[i]) * step;
    const side = Number(sides[i]) === 1 ? 'bid' : 'ask';
    const intensity = clamp(Number(intensities[i]) / scale, 0, 1);
    const isLarge = Boolean(Number(large[i] || 0));
    const label = `${side === 'bid' ? '买盘' : '卖盘'}流动性${isLarge ? ' · 大流动性' : ''}`;
    cells[i] = {
      start_timestamp: startTimestamp,
      end_timestamp: endTimestamp,
      price_low: priceLow,
      price_high: priceLow + step,
      intensity,
      side,
      color: colorMode === 'single' ? '#f97316' : (side === 'bid' ? '#22d3ee' : '#fb7185'),
      label,
      confidence: 1,
      fields: {
        depth: Number(depths[i]),
        unit,
        order_count: Number(orders[i]),
        causal_depth_ratio: intensity,
        is_large_rolling: isLarge,
        source_snapshot_start: sourceStarts[columnIndex],
        source_snapshot_end: sourceEnds[columnIndex],
        source_lag_ms: Number(sourceLags[columnIndex] || 0),
      },
      _startMs: startMs,
      _endMs: endMs,
    };
  }
  return cells.filter(cell => Number.isFinite(cell._startMs) && Number.isFinite(cell._endMs) && cell._endMs > cell._startMs)
    .sort((a, b) => a._startMs - b._startMs || Number(a.price_low) - Number(b.price_low));
}

async function runPlugin() {
  const plugin = selectedPlugin();
  if (!plugin) return;
  try {
    setStatus('运行插件中...');
    const data = await getJson('/api/plugin-markers', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ data: currentDataParams(), plugin_id: plugin.id, params: pluginParams() })
    });
    state.markers = data.markers || [];
    state.regions = data.regions || [];
    state.tracks = data.tracks || [];
    state.bands = data.bands || [];
    if (data.heatmap_compact) {
      state.heatmap = decodeCompactHeatmap(data.heatmap_compact);
    } else {
      state.heatmap = (data.heatmap || []).map(cell => ({
        ...cell,
        _startMs: parseTimestampMs(cell.start_timestamp),
        _endMs: parseTimestampMs(cell.end_timestamp),
      })).filter(cell => Number.isFinite(cell._startMs) && Number.isFinite(cell._endMs) && cell._endMs > cell._startMs)
        .sort((a, b) => a._startMs - b._startMs || Number(a.price_low) - Number(b.price_low));
    }
    state.heatmapMaxDurationMs = state.heatmap.reduce((m, cell) => Math.max(m, cell._endMs - cell._startMs), 0);
    state.priceRegions = (data.price_regions || []).map(region => ({
      ...region,
      _startMs: parseTimestampMs(region.start_timestamp),
      _endMs: parseTimestampMs(region.end_timestamp),
    })).filter(region => Number.isFinite(region._startMs) && Number.isFinite(region._endMs)
      && region._endMs > region._startMs && Number(region.price_high) > Number(region.price_low))
      .sort((a, b) => a._startMs - b._startMs || Number(a.price_low) - Number(b.price_low));
    state.heatmapLayerKey = '';
    rebuildHeatmapIndex();
    state.rowFields = data.row_fields || {};
    const s = data.summary || {};
    state.pluginUi = s.ui || {};
    state.showWallOverlay = Boolean(state.pluginUi.wall_overlay_default);
    const wallToggle = $('wallOverlayToggle');
    const wallControl = $('wallOverlayControl');
    if (wallToggle) wallToggle.checked = state.showWallOverlay;
    if (wallControl) wallControl.classList.toggle('hidden', !state.pluginUi.wall_overlay_control || !state.priceRegions.length);
    const wallLabel = $('wallOverlayLabel');
    if (wallLabel) wallLabel.textContent = state.pluginUi.wall_overlay_label || '墙';
    state.hoverHeatmapCells = [];
    state.selectedHeatmapCells = [];
    const colorControls = $('heatmapColorControls');
    if (colorControls) colorControls.classList.toggle('hidden', !state.pluginUi.heatmap_color_controls || !state.heatmap.length);
    setHeatmapColorRange(
      Number(state.pluginUi.heatmap_color_min_pct ?? 0),
      Number(state.pluginUi.heatmap_color_max_pct ?? 50),
    );
    updateAlignmentAuditBadge(s.alignment_audit || null);
    renderHeatmapCellDetail([]);
    const advancedPanel = $('advancedDetailPanel');
    if (advancedPanel && state.pluginUi.advanced_collapsed !== false) advancedPanel.open = false;
    $('pluginSummary').textContent = s.display || `匹配 ${s.matched ?? 0} / ${s.input_rows ?? 0} 根；上影 ${s.upper_count ?? '-'}，下影 ${s.lower_count ?? '-'}`;
    setStatus(`插件完成：${plugin.name}，节点 ${state.markers.length} 个，区间 ${state.regions.length} 个，指标轨道 ${state.tracks.length} 条，状态色带 ${state.bands.length} 条，热力格 ${state.heatmap.length} 个，墙框 ${state.priceRegions.length} 个`);
    updateDetails(state.selectedIndex ?? state.hoverIndex);
    draw();
  } catch (err) {
    setStatus(err.message, true);
  }
}

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function setupCanvas() {
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (!state.candles.length) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const plot = plotArea();
    const axis = priceAxisArea(plot);
    const overPriceAxis = x >= axis.left && x <= axis.right && y >= axis.top && y <= axis.bottom;

    if (overPriceAxis || e.shiftKey) {
      const s = scales();
      const anchorPrice = priceForY(y, s, plot);
      zoomPriceViewport(anchorPrice, e.deltaY < 0 ? 0.88 : 1.14);
      draw();
      return;
    }

    const frac = Math.max(0, Math.min(1, (x - plot.left) / Math.max(plot.width, 1)));
    const span = state.visibleEnd - state.visibleStart;
    const zoom = e.deltaY < 0 ? 0.82 : 1.22;
    const newSpan = Math.max(20, Math.min(state.candles.length, Math.round(span * zoom)));
    const anchor = state.visibleStart + span * frac;
    state.visibleStart = Math.round(anchor - newSpan * frac);
    state.visibleEnd = state.visibleStart + newSpan;
    clampVisible();
    draw();
  }, { passive: false });

  canvas.addEventListener('dblclick', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const plot = plotArea();
    const axis = priceAxisArea(plot);
    if (x >= axis.left && x <= axis.right && y >= axis.top && y <= axis.bottom) {
      resetPriceViewport();
      draw();
    }
  });

  canvas.addEventListener('mousedown', (e) => {
    if (!state.candles.length || e.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const plot = plotArea();
    const axis = priceAxisArea(plot);
    const overPriceAxis = x >= axis.left && x <= axis.right && y >= axis.top && y <= axis.bottom;
    const overPlot = x >= plot.left && x <= plot.right && y >= plot.top && y <= plot.bottom;
    if (!overPriceAxis && !overPlot) return;

    const s = scales();
    state.dragging = true;
    state.dragMoved = false;
    state.dragX = e.clientX;
    state.dragY = e.clientY;
    state.dragStartStart = state.visibleStart;
    state.dragStartEnd = state.visibleEnd;
    state.priceDragStartMin = s.minP;
    state.priceDragStartMax = s.maxP;
    state.priceDragAnchorPrice = priceForY(y, s, plot);
    state.priceDragAnchorFrac = (state.priceDragAnchorPrice - s.minP) / Math.max(s.maxP - s.minP, 1e-9);
    state.dragMode = overPriceAxis ? (e.shiftKey ? 'price-pan' : 'price-scale') : (e.shiftKey ? 'pan-xy' : 'pan-x');
    canvas.style.cursor = overPriceAxis ? 'ns-resize' : 'grabbing';
  });

  window.addEventListener('mouseup', (e) => {
    if (!state.dragging) return;
    const wasDrag = state.dragMoved || Math.abs(e.clientX - state.dragX) > state.dragThreshold || Math.abs(e.clientY - state.dragY) > state.dragThreshold;
    const mode = state.dragMode;
    state.dragging = false;
    state.dragMoved = false;
    state.dragMode = null;
    canvas.style.cursor = 'crosshair';
    if (!wasDrag && (mode === 'pan-x' || mode === 'pan-xy')) handleCanvasClick(e);
  });

  window.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const inside = e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom;

    if (state.dragging && state.candles.length) {
      const dx = e.clientX - state.dragX;
      const dy = e.clientY - state.dragY;
      if (Math.abs(dx) > state.dragThreshold || Math.abs(dy) > state.dragThreshold) state.dragMoved = true;
      const plot = plotArea();
      const startMin = Number(state.priceDragStartMin);
      const startMax = Number(state.priceDragStartMax);
      const priceSpan = Math.max(startMax - startMin, 1e-9);

      if (state.dragMode === 'price-scale') {
        const anchor = Number.isFinite(state.priceDragAnchorPrice) ? state.priceDragAnchorPrice : (startMin + startMax) * 0.5;
        const factor = Math.exp(dy / 180);
        state.yMinManual = anchor - (anchor - startMin) * factor;
        state.yMaxManual = anchor + (startMax - anchor) * factor;
        clampManualPriceRange();
        draw();
        return;
      }

      if (state.dragMode === 'price-pan') {
        const shiftPrice = (dy / Math.max(plot.height, 1)) * priceSpan;
        state.yMinManual = startMin + shiftPrice;
        state.yMaxManual = startMax + shiftPrice;
        clampManualPriceRange();
        draw();
        return;
      }

      if (state.dragMode === 'pan-x' || state.dragMode === 'pan-xy') {
        const barSpan = state.dragStartEnd - state.dragStartStart;
        const barsPerPx = barSpan / Math.max(1, plot.width);
        const shiftBars = Math.round(-dx * barsPerPx);
        state.visibleStart = state.dragStartStart + shiftBars;
        state.visibleEnd = state.dragStartEnd + shiftBars;
        clampVisible();

        // Vertical movement is deliberately locked during normal dragging.
        // Hold Shift while starting the drag to enable XY panning.
        if (state.dragMode === 'pan-xy' && Math.abs(dy) > 1) {
          const shiftPrice = (dy / Math.max(plot.height, 1)) * priceSpan;
          state.yMinManual = startMin + shiftPrice;
          state.yMaxManual = startMax + shiftPrice;
          clampManualPriceRange();
        }
        draw();
        return;
      }
    }

    if (!inside || !state.candles.length) {
      tooltip.classList.add('hidden');
      canvas.style.cursor = 'default';
      state.hoverHeatmapCells = [];
      state.hoverPrice = null;
      if (!state.selectedHeatmapCells.length) renderHeatmapCellDetail([]);
      return;
    }
    const localX = e.clientX - rect.left;
    const localY = e.clientY - rect.top;
    const plot = plotArea();
    const axis = priceAxisArea(plot);
    const overPriceAxis = localX >= axis.left && localX <= axis.right && localY >= axis.top && localY <= axis.bottom;
    canvas.style.cursor = overPriceAxis ? 'ns-resize' : 'crosshair';
    if (overPriceAxis) {
      tooltip.classList.add('hidden');
      state.hoverHeatmapCells = [];
      state.hoverPrice = null;
      if (!state.selectedHeatmapCells.length) renderHeatmapCellDetail([]);
      return;
    }
    const index = indexAtX(localX);
    const s = scales();
    state.hoverPrice = priceForY(localY, s, plot);
    state.hoverHeatmapCells = heatmapCellsAtPoint(localX, localY, plot, s);
    state.hoverIndex = index;
    if (state.selectedIndex === null) updateDetails(index);
    renderHeatmapCellDetail(state.selectedHeatmapCells.length ? state.selectedHeatmapCells : state.hoverHeatmapCells);
    positionTooltip(index, localX, localY, state.hoverHeatmapCells);
    draw();
  });

  canvas.addEventListener('mouseleave', () => {
    if (!state.dragging) canvas.style.cursor = 'default';
    state.hoverIndex = null;
    state.hoverPrice = null;
    state.hoverHeatmapCells = [];
    tooltip.classList.add('hidden');
    renderHeatmapCellDetail(state.selectedHeatmapCells);
    draw();
  });
}

function handleCanvasClick(e) {
  if (!state.candles.length) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const idx = hitTestCandleColumn(x, y);
  if (idx === null) {
    state.selectedHeatmapCells = [];
    renderHeatmapCellDetail([]);
    setSelectedIndex(null);
    return;
  }
  const plot = plotArea();
  const s = scales();
  const cells = heatmapCellsAtPoint(x, y, plot, s);
  const sameIndex = state.selectedIndex === idx;
  const sameCell = cells.length && state.selectedHeatmapCells.length
    && cells[0]._startMs === state.selectedHeatmapCells[0]._startMs
    && Number(cells[0].price_low) === Number(state.selectedHeatmapCells[0].price_low)
    && String(cells[0].side) === String(state.selectedHeatmapCells[0].side);
  if (sameIndex && (!cells.length || sameCell)) {
    state.selectedHeatmapCells = [];
    renderHeatmapCellDetail([]);
    setSelectedIndex(null);
    return;
  }
  state.selectedHeatmapCells = cells;
  renderHeatmapCellDetail(cells);
  setSelectedIndex(idx);
}

function hitTestCandleColumn(x, y) {
  const plot = plotArea();
  const vol = volumeArea();
  if (x < plot.left || x > plot.right || y < plot.top || y > vol.bottom) return null;
  const idx = indexAtX(x);
  if (idx < state.visibleStart || idx >= state.visibleEnd) return null;
  return idx;
}

function setSelectedIndex(index) {
  state.selectedIndex = index;
  if (index === null) {
    state.selectedHeatmapCells = [];
    renderHeatmapCellDetail([]);
  }
  updateDetails(index);
  draw();
}

function clampVisible() {
  const n = state.candles.length;
  const span = state.visibleEnd - state.visibleStart;
  if (state.visibleStart < 0) { state.visibleStart = 0; state.visibleEnd = span; }
  if (state.visibleEnd > n) { state.visibleEnd = n; state.visibleStart = Math.max(0, n - span); }
}

function indicatorArea() {
  const rect = canvas.getBoundingClientRect();
  const count = Math.min(10, state.tracks.length);
  const height = count ? Math.max(84, Math.min(260, count * 26)) : 0;
  const bottom = rect.height - 84;
  return { left: 64, top: bottom - height, right: rect.width - 78, bottom, width: rect.width - 142, height };
}

function plotArea() {
  const rect = canvas.getBoundingClientRect();
  const indicators = indicatorArea();
  const top = 58;
  const bottom = state.tracks.length ? indicators.top - 10 : rect.height - 88;
  return { left: 64, top, right: rect.width - 78, bottom, width: rect.width - 142, height: Math.max(80, bottom - top) };
}

function volumeArea() {
  const rect = canvas.getBoundingClientRect();
  return { left: 64, top: rect.height - 72, right: rect.width - 78, bottom: rect.height - 30, width: rect.width - 142, height: 42 };
}

function priceAxisArea(plot) {
  const rect = canvas.getBoundingClientRect();
  return { left: plot.right + 2, top: plot.top, right: rect.width - 8, bottom: plot.bottom, width: Math.max(24, rect.width - plot.right - 10), height: plot.height };
}

function clampManualPriceRange() {
  if (!Number.isFinite(state.yMinManual) || !Number.isFinite(state.yMaxManual) || state.yMaxManual <= state.yMinManual) {
    state.yMinManual = null;
    state.yMaxManual = null;
  }
}

function priceForY(y, s, plot) {
  const frac = Math.max(0, Math.min(1, (plot.bottom - y) / Math.max(plot.height, 1)));
  return s.minP + frac * (s.maxP - s.minP);
}

function resetPriceViewport() {
  state.yMinManual = null;
  state.yMaxManual = null;
}

function zoomPriceViewport(anchorPrice, scaleFactor) {
  const s = scales();
  const minP = Number.isFinite(state.yMinManual) ? state.yMinManual : s.minP;
  const maxP = Number.isFinite(state.yMaxManual) ? state.yMaxManual : s.maxP;
  const span = Math.max(maxP - minP, 1e-9);
  const anchor = Number.isFinite(anchorPrice) ? anchorPrice : (minP + maxP) * 0.5;
  const nextSpan = Math.max(1e-9, span * scaleFactor);
  const lowerFrac = (anchor - minP) / span;
  const upperFrac = (maxP - anchor) / span;
  state.yMinManual = anchor - nextSpan * lowerFrac;
  state.yMaxManual = anchor + nextSpan * upperFrac;
  clampManualPriceRange();
}

function panPriceViewport(deltaPrice) {
  const s = scales();
  const minP = Number.isFinite(state.yMinManual) ? state.yMinManual : s.minP;
  const maxP = Number.isFinite(state.yMaxManual) ? state.yMaxManual : s.maxP;
  state.yMinManual = minP + deltaPrice;
  state.yMaxManual = maxP + deltaPrice;
  clampManualPriceRange();
}

function visibleCandles() {
  return state.candles.slice(state.visibleStart, state.visibleEnd);
}

function scales() {
  const vis = visibleCandles();
  const vols = vis.map(c => Number(c.volume)).filter(Number.isFinite);
  let autoMin = Infinity;
  let autoMax = -Infinity;
  for (const candle of vis) {
    const low = Number(candle.low);
    const high = Number(candle.high);
    if (Number.isFinite(low)) autoMin = Math.min(autoMin, low);
    if (Number.isFinite(high)) autoMax = Math.max(autoMax, high);
  }

  const { startMs, endMs } = visibleTimeRangeMs();
  if (state.heatmap.length) {
    const first = lowerBoundHeatmapStart(startMs - Math.max(state.heatmapMaxDurationMs, 1));
    for (let i = first; i < state.heatmap.length; i++) {
      const cell = state.heatmap[i];
      if (cell._startMs >= endMs) break;
      if (cell._endMs <= startMs || Number(cell.intensity || 0) < 0.35) continue;
      const low = Number(cell.price_low);
      const high = Number(cell.price_high);
      if (Number.isFinite(low)) autoMin = Math.min(autoMin, low);
      if (Number.isFinite(high)) autoMax = Math.max(autoMax, high);
    }
  }

  if (state.showWallOverlay && state.priceRegions.length) {
    for (const region of state.priceRegions) {
      if (region._endMs <= startMs || region._startMs >= endMs) continue;
      const low = Number(region.price_low);
      const high = Number(region.price_high);
      if (Number.isFinite(low)) autoMin = Math.min(autoMin, low);
      if (Number.isFinite(high)) autoMax = Math.max(autoMax, high);
    }
  }

  if (!Number.isFinite(autoMin) || !Number.isFinite(autoMax) || autoMax <= autoMin) {
    autoMin = 0;
    autoMax = 1;
  }
  const pad = Math.max((autoMax - autoMin) * 0.08, Math.abs(autoMax) * 0.0005, 1e-9);
  autoMin -= pad;
  autoMax += pad;
  clampManualPriceRange();
  const minP = Number.isFinite(state.yMinManual) ? state.yMinManual : autoMin;
  const maxP = Number.isFinite(state.yMaxManual) ? state.yMaxManual : autoMax;
  const maxV = Math.max(...vols, 1);
  return {
    minP,
    maxP,
    maxV,
    autoMinP: autoMin,
    autoMaxP: autoMax,
    usingManualY: Number.isFinite(state.yMinManual) && Number.isFinite(state.yMaxManual),
  };
}

function yForPrice(price, s, plot) {
  return plot.bottom - ((price - s.minP) / (s.maxP - s.minP)) * plot.height;
}

function xForIndex(i, plot) {
  const span = Math.max(1, state.visibleEnd - state.visibleStart);
  const local = i - state.visibleStart;
  return plot.left + (local + 0.5) * plot.width / span;
}

function indexAtX(x) {
  const plot = plotArea();
  const span = Math.max(1, state.visibleEnd - state.visibleStart);
  const local = Math.floor((x - plot.left) / plot.width * span);
  return Math.max(state.visibleStart, Math.min(state.visibleEnd - 1, state.visibleStart + local));
}

function draw() {
  resizeCanvas();
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  drawBackground(rect);
  if (!state.candles.length) {
    ctx.fillStyle = '#64748b';
    ctx.font = '14px system-ui';
    ctx.fillText('点击左侧“加载图表”。如果没有数据，先用项目已有 prebuild/download 工具准备本地 DB。', 72, 70);
    return;
  }
  const plot = plotArea();
  const vol = volumeArea();
  const s = scales();
  drawGrid(plot, vol, s);
  drawStateBands(plot);
  drawPriceHeatmap(plot, s);
  drawPriceRegions(plot, s);
  drawHoveredHeatmapCells(plot, s);
  drawRegions(plot);
  drawCandles(plot, vol, s);
  drawIndicatorTracks(plot);
  drawMarkers(plot, s);
  drawSelectedCandle(plot, s);
  drawAxes(plot, vol, s);
  drawCrosshair(plot, s);
}

function drawBackground(rect) {
  // CoinGlass-style warm ivory chart canvas.  Heat intensity is encoded by the
  // palette itself, not by blending bright colors into a dark background.
  const g = ctx.createLinearGradient(0, 0, 0, rect.height);
  g.addColorStop(0, '#fffefa');
  g.addColorStop(1, '#fbfaf2');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, rect.width, rect.height);
}

function drawGrid(plot, vol, s) {
  ctx.strokeStyle = 'rgba(51,65,85,0.10)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 5; i++) {
    const y = plot.top + plot.height * i / 5;
    ctx.moveTo(plot.left, y); ctx.lineTo(plot.right, y);
  }
  const count = 8;
  for (let i = 0; i <= count; i++) {
    const x = plot.left + plot.width * i / count;
    ctx.moveTo(x, plot.top); ctx.lineTo(x, vol.bottom);
  }
  ctx.moveTo(vol.left, vol.top); ctx.lineTo(vol.right, vol.top);
  ctx.stroke();
}

function drawCandles(plot, vol, s) {
  const span = state.visibleEnd - state.visibleStart;
  const step = plot.width / Math.max(1, span);
  const bodyW = Math.max(2, Math.min(14, step * 0.62));
  for (let i = state.visibleStart; i < state.visibleEnd; i++) {
    const c = state.candles[i];
    const x = xForIndex(i, plot);
    const o = Number(c.open), h = Number(c.high), l = Number(c.low), cl = Number(c.close), v = Number(c.volume || 0);
    const up = cl >= o;
    const color = up ? '#22c55e' : '#ef4444';
    const yH = yForPrice(h, s, plot), yL = yForPrice(l, s, plot), yO = yForPrice(o, s, plot), yC = yForPrice(cl, s, plot);
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1, Math.min(2, step * 0.12));
    ctx.beginPath(); ctx.moveTo(x, yH); ctx.lineTo(x, yL); ctx.stroke();
    const top = Math.min(yO, yC), bottom = Math.max(yO, yC);
    ctx.fillStyle = color;
    ctx.fillRect(x - bodyW / 2, top, bodyW, Math.max(1, bottom - top));
    ctx.globalAlpha = 0.32;
    const vh = Math.max(1, (v / s.maxV) * vol.height);
    ctx.fillRect(x - bodyW / 2, vol.bottom - vh, bodyW, vh);
    ctx.globalAlpha = 1;
  }
}

function drawIndicatorTracks(plot) {
  if (!state.tracks.length) return;
  const area = indicatorArea();
  if (area.height <= 0) return;
  const laneCount = Math.min(10, state.tracks.length);
  const laneHeight = area.height / laneCount;
  const span = Math.max(1, state.visibleEnd - state.visibleStart);
  const decimate = Math.max(1, Math.floor(span / Math.max(1, area.width)));

  ctx.save();
  ctx.strokeStyle = 'rgba(51,65,85,0.10)';
  ctx.fillStyle = 'rgba(248,250,252,0.88)';
  ctx.fillRect(area.left, area.top, area.width, area.height);

  for (let lane = 0; lane < laneCount; lane++) {
    const track = state.tracks[lane];
    const top = area.top + lane * laneHeight;
    const bottom = top + laneHeight;
    const minV = Number(track.min ?? 0);
    const maxV = Number(track.max ?? 1);
    const range = Math.max(1e-12, maxV - minV);
    const ref = Number(track.reference);

    ctx.strokeStyle = 'rgba(51,65,85,0.08)';
    ctx.beginPath();
    ctx.moveTo(area.left, bottom); ctx.lineTo(area.right, bottom);
    ctx.stroke();

    if (Number.isFinite(ref) && ref >= minV && ref <= maxV) {
      const yRef = bottom - ((ref - minV) / range) * laneHeight;
      ctx.strokeStyle = 'rgba(51,65,85,0.18)';
      ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(area.left, yRef); ctx.lineTo(area.right, yRef); ctx.stroke();
      ctx.setLineDash([]);
    }

    const values = Array.isArray(track.values) ? track.values : [];
    ctx.strokeStyle = track.color || '#38bdf8';
    ctx.lineWidth = 1.35;
    ctx.beginPath();
    let started = false;
    for (let i = state.visibleStart; i < state.visibleEnd; i += decimate) {
      const value = Number(values[i]);
      if (!Number.isFinite(value)) { started = false; continue; }
      const x = xForIndex(i, plot);
      const clipped = Math.max(minV, Math.min(maxV, value));
      const y = bottom - ((clipped - minV) / range) * Math.max(8, laneHeight - 5) - 2;
      if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    }
    ctx.stroke();

    const focusIndex = state.selectedIndex ?? state.hoverIndex ?? (state.visibleEnd - 1);
    const current = Number(values[focusIndex]);
    ctx.fillStyle = '#475569';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    const valueText = Number.isFinite(current) ? fmt(current, 3) : '-';
    ctx.fillText(`${track.label || track.id}: ${valueText}`, area.left + 5, top + 3);
  }
  ctx.restore();
}

function markerMap() {
  const m = new Map();
  for (const marker of state.markers) {
    const key = marker.timestamp;
    if (!m.has(key)) m.set(key, []);
    m.get(key).push(marker);
  }
  return m;
}

function indexForTimestamp(timestamp) {
  const exact = state.timestampIndex.get(timestamp);
  if (exact !== undefined) return exact;
  const target = parseTimestampMs(timestamp);
  if (!Number.isFinite(target) || !state.candles.length) return null;
  let lo = 0, hi = state.candles.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const value = Number(state.candles[mid].time);
    if (value < target) lo = mid + 1;
    else if (value > target) hi = mid - 1;
    else return mid;
  }
  return Math.max(0, Math.min(state.candles.length - 1, lo));
}

function indexBeforeTimestamp(timestamp) {
  const exact = state.timestampIndex.get(timestamp);
  if (exact !== undefined) return Math.max(0, exact - 1);
  const target = parseTimestampMs(timestamp);
  if (!Number.isFinite(target) || !state.candles.length) return null;
  let lo = 0, hi = state.candles.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (Number(state.candles[mid].time) < target) lo = mid + 1;
    else hi = mid;
  }
  return Math.max(0, Math.min(state.candles.length - 1, lo - 1));
}

function stateBandCategoriesAt(index) {
  const hits = [];
  for (const band of state.bands) {
    const codes = Array.isArray(band.codes) ? band.codes : [];
    const code = codes[index];
    if (code === null || code === undefined) continue;
    const category = (band.categories || []).find(item => Number(item.code) === Number(code));
    if (category) hits.push({ band, category });
  }
  return hits;
}

function drawStateBands(plot) {
  if (!state.bands.length) return;
  const span = Math.max(1, state.visibleEnd - state.visibleStart);
  const step = plot.width / span;
  let topStripOffset = 0;
  let bottomStripOffset = 0;
  for (const band of state.bands) {
    const codes = Array.isArray(band.codes) ? band.codes : [];
    const categories = new Map((band.categories || []).map(item => [Number(item.code), item]));
    const mode = band.render_mode || 'background';
    const height = Math.max(4, Number(band.height_px || 10));
    const position = band.position || 'bottom';
    let bandTop = plot.top;
    let bandHeight = plot.height;
    if (mode === 'strip') {
      if (position === 'top') {
        bandTop = plot.top + topStripOffset;
        topStripOffset += height + 1;
      } else {
        bottomStripOffset += height;
        bandTop = plot.bottom - bottomStripOffset;
        bottomStripOffset += 1;
      }
      bandHeight = height;
    }
    let i = state.visibleStart;
    while (i < state.visibleEnd) {
      const code = Number(codes[i]);
      if (!Number.isFinite(code) || !categories.has(code)) { i += 1; continue; }
      let j = i + 1;
      while (j < state.visibleEnd && Number(codes[j]) === code) j += 1;
      const category = categories.get(code);
      const x1 = xForIndex(i, plot) - step * 0.5;
      const x2 = xForIndex(j - 1, plot) + step * 0.5;
      ctx.save();
      const opacity = mode === 'strip' ? Number(category.opacity ?? 0.82) : Number(category.opacity ?? 0.07);
      ctx.fillStyle = hexToRgba(category.color || '#64748b', opacity);
      ctx.fillRect(x1, bandTop, Math.max(1, x2 - x1), bandHeight);
      if (mode === 'strip') {
        ctx.strokeStyle = 'rgba(226,232,240,0.16)';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x1, bandTop + 0.25, Math.max(1, x2 - x1), Math.max(1, bandHeight - 0.5));
      }
      ctx.restore();
      i = j;
    }
  }
}

function interpolateRgb(a, b, t) {
  return {
    r: Math.round(a.r + (b.r - a.r) * t),
    g: Math.round(a.g + (b.g - a.g) * t),
    b: Math.round(a.b + (b.b - a.b) * t),
  };
}

function liquidityHeatColor(intensity, alpha = 0.98) {
  // Palette sampled from the user-provided CoinGlass light-theme screenshots.
  // Low depth is warm ivory, medium depth becomes salmon/rose, and genuinely
  // large resting size remains dark burgundy-purple.  The RGB value carries
  // the intensity; opacity stays nearly constant so moving the saturation
  // limit never fades every cell together.
  const stops = [
    { t: 0.00, c: { r: 255, g: 254, b: 246 } },
    { t: 0.10, c: { r: 251, g: 248, b: 229 } },
    { t: 0.22, c: { r: 248, g: 239, b: 214 } },
    { t: 0.36, c: { r: 238, g: 183, b: 155 } },
    { t: 0.52, c: { r: 223, g: 130, b: 129 } },
    { t: 0.68, c: { r: 194, g: 82, b: 112 } },
    { t: 0.84, c: { r: 155, g: 57, b: 111 } },
    { t: 1.00, c: { r: 119, g: 45, b: 109 } },
  ];
  const raw = Math.max(0, Math.min(1, intensity));
  // Positive last-snapshot cells must remain distinguishable from the warm
  // white canvas.  This is display-only and monotone: it lifts the very pale
  // tail without changing depth ordering or any wall/backtest input.
  const x = raw <= 0 ? 0.06 : 0.06 + 0.94 * Math.pow(raw, 0.72);
  let left = stops[0], right = stops[stops.length - 1];
  for (let i = 1; i < stops.length; i++) {
    if (x <= stops[i].t) {
      left = stops[i - 1];
      right = stops[i];
      break;
    }
  }
  const local = right.t > left.t ? (x - left.t) / (right.t - left.t) : 0;
  const c = interpolateRgb(left.c, right.c, local);
  return `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha})`;
}

function heatmapLayerKey(plot, s, startMs, endMs) {
  const dpr = window.devicePixelRatio || 1;
  return [
    Math.round(plot.width), Math.round(plot.height), dpr,
    Math.round(startMs), Math.round(endMs),
    Number(s.minP).toFixed(8), Number(s.maxP).toFixed(8),
    state.heatmap.length,
    Number(state.heatmapColorMinPct).toFixed(2), Number(state.heatmapColorMaxPct).toFixed(2),
  ].join('|');
}

function ensureHeatmapLayer(plot, s) {
  const { startMs, endMs } = visibleTimeRangeMs();
  const key = heatmapLayerKey(plot, s, startMs, endMs);
  if (state.heatmapLayerCanvas && state.heatmapLayerKey === key) return state.heatmapLayerCanvas;

  const dpr = window.devicePixelRatio || 1;
  const layer = state.heatmapLayerCanvas || document.createElement('canvas');
  layer.width = Math.max(1, Math.ceil(plot.width * dpr));
  layer.height = Math.max(1, Math.ceil(plot.height * dpr));
  const lctx = layer.getContext('2d', { alpha: true });
  lctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  lctx.clearRect(0, 0, plot.width, plot.height);
  lctx.imageSmoothingEnabled = false;

  const first = lowerBoundHeatmapStart(startMs - Math.max(state.heatmapMaxDurationMs, 1));
  const timeSpan = Math.max(1, endMs - startMs);
  const priceSpan = Math.max(1e-9, s.maxP - s.minP);
  for (let i = first; i < state.heatmap.length; i++) {
    const cell = state.heatmap[i];
    if (cell._startMs >= endMs) break;
    if (cell._endMs <= startMs) continue;
    const low = Number(cell.price_low);
    const high = Number(cell.price_high);
    if (!Number.isFinite(low) || !Number.isFinite(high)) continue;
    if (high < s.minP || low > s.maxP) continue;

    const x1raw = ((Math.max(cell._startMs, startMs) - startMs) / timeSpan) * plot.width;
    const x2raw = ((Math.min(cell._endMs, endMs) - startMs) / timeSpan) * plot.width;
    const y1raw = plot.height - ((Math.min(high, s.maxP) - s.minP) / priceSpan) * plot.height;
    const y2raw = plot.height - ((Math.max(low, s.minP) - s.minP) / priceSpan) * plot.height;
    let left = Math.floor(Math.min(x1raw, x2raw));
    let right = Math.ceil(Math.max(x1raw, x2raw));
    let top = Math.floor(Math.min(y1raw, y2raw));
    let bottom = Math.ceil(Math.max(y1raw, y2raw));
    if (right <= left || bottom <= top) continue;

    if (right - left >= 3) { left += 0.35; right -= 0.35; }
    if (bottom - top >= 3) { top += 0.35; bottom -= 0.35; }

    if (!heatmapCellPassesColorFloor(cell)) continue;
    const intensity = effectiveHeatmapIntensity(cell);
    const confidence = Math.max(0, Math.min(1, Number(cell.confidence || 0)));
    // Use a stable, nearly opaque layer like CoinGlass.  Depth differences are
    // represented by palette position, while confidence only makes a small
    // opacity adjustment.  This prevents the whole map fading uniformly when
    // the upper saturation threshold changes.
    const alpha = 0.90 + 0.08 * confidence;
    const baseColor = String(cell.color || '#f97316').toLowerCase();
    lctx.fillStyle = baseColor === '#f97316'
      ? liquidityHeatColor(intensity, alpha)
      : hexToRgba(cell.color || '#f97316', alpha);
    lctx.fillRect(left, top, Math.max(0.6, right - left), Math.max(0.6, bottom - top));
  }

  state.heatmapLayerCanvas = layer;
  state.heatmapLayerKey = key;
  return layer;
}

function drawPriceHeatmap(plot, s) {
  if (!state.heatmap.length) return;
  const layer = ensureHeatmapLayer(plot, s);
  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.left, plot.top, plot.width, plot.height);
  ctx.clip();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(layer, plot.left, plot.top, plot.width, plot.height);
  ctx.restore();
}



function buildPriceRegionChains(startMs, endMs) {
  const visible = state.priceRegions
    .filter(region => region._endMs > startMs && region._startMs < endMs)
    .slice()
    .sort((a, b) => {
      const aId = String(a.fields?.wall_id ?? '');
      const bId = String(b.fields?.wall_id ?? '');
      if (aId !== bId) return aId.localeCompare(bId, undefined, { numeric: true });
      if (a.side !== b.side) return String(a.side || '').localeCompare(String(b.side || ''));
      return a._startMs - b._startMs || a._endMs - b._endMs;
    });

  const chains = [];
  for (const region of visible) {
    const wallId = String(region.fields?.wall_id ?? `region:${region._startMs}:${region._endMs}`);
    const previousChain = chains[chains.length - 1];
    const previousRegion = previousChain?.regions?.[previousChain.regions.length - 1];
    const configuredFadeMinutes = Number(region.fields?.maximum_fade_minutes ?? 0);
    const fadeToleranceMs = Math.max(1, configuredFadeMinutes * 60_000 + 1);
    const contiguous = previousRegion && region._startMs <= previousRegion._endMs + fadeToleranceMs;
    const sameIdentity = previousChain
      && previousChain.wallId === wallId
      && previousChain.side === region.side;

    if (!sameIdentity || !contiguous) {
      chains.push({
        wallId,
        side: region.side,
        regions: [region],
      });
      continue;
    }
    previousChain.regions.push(region);
  }
  return chains;
}

function wallStrokeStyle(region) {
  return { color: region.color || '#00AEEF', alpha: 0.97, dash: [] };
}

function drawWallRegionChain(chain, plot, s, startMs, endMs, timeSpan) {
  const regions = chain.regions || [];
  if (!regions.length) return;

  // V2.5.4 walls are deliberately one fixed rectangle.  Even when an older
  // payload contains several lifecycle slices, collapse them into the stable
  // wall-level bounds instead of drawing a stepped or fragmented outline.
  const firstRegion = regions[0];
  const lowCandidates = regions
    .map(region => Number(region.fields?.rectangle_price_low ?? region.price_low))
    .filter(Number.isFinite);
  const highCandidates = regions
    .map(region => Number(region.fields?.rectangle_price_high ?? region.price_high))
    .filter(Number.isFinite);
  if (!lowCandidates.length || !highCandidates.length) return;

  const low = Number(firstRegion.fields?.rectangle_price_low ?? lowCandidates[0]);
  const high = Number(firstRegion.fields?.rectangle_price_high ?? highCandidates[0]);
  if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) return;
  if (high < s.minP || low > s.maxP) return;

  const regionStart = Math.min(...regions.map(region => region._startMs));
  const regionEnd = Math.max(...regions.map(region => region._endMs));
  const x1 = plot.left + ((Math.max(regionStart, startMs) - startMs) / timeSpan) * plot.width;
  const x2 = plot.left + ((Math.min(regionEnd, endMs) - startMs) / timeSpan) * plot.width;
  if (x2 <= x1) return;

  const top = yForPrice(Math.min(high, s.maxP), s, plot);
  const bottom = yForPrice(Math.max(low, s.minP), s, plot);
  const rectTop = Math.min(top, bottom);
  const rectHeight = Math.max(2, Math.abs(bottom - top));
  const rectWidth = Math.max(1, x2 - x1);
  const style = wallStrokeStyle(firstRegion);

  const fillOpacity = clamp(Number(firstRegion.opacity ?? 0.018), 0, 0.12);
  if (fillOpacity > 0) {
    ctx.fillStyle = hexToRgba(style.color, fillOpacity);
    ctx.fillRect(x1, rectTop, rectWidth, rectHeight);
  }

  ctx.strokeStyle = hexToRgba(style.color, style.alpha);
  ctx.lineWidth = Math.max(1.4, Number(firstRegion.border_width || 1.8));
  ctx.setLineDash([]);
  ctx.strokeRect(x1 + 0.5, rectTop + 0.5, Math.max(1, rectWidth - 1), Math.max(1, rectHeight - 1));
}

function drawPriceRegions(plot, s) {
  if (!state.showWallOverlay || !state.priceRegions.length) return;
  const { startMs, endMs } = visibleTimeRangeMs();
  const timeSpan = Math.max(1, endMs - startMs);
  const chains = buildPriceRegionChains(startMs, endMs);
  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.left, plot.top, plot.width, plot.height);
  ctx.clip();
  for (const chain of chains) {
    drawWallRegionChain(chain, plot, s, startMs, endMs, timeSpan);
  }
  ctx.setLineDash([]);
  ctx.restore();
}

function timestampForX(x, plot) {
  const { startMs, endMs } = visibleTimeRangeMs();
  const frac = clamp((x - plot.left) / Math.max(plot.width, 1), 0, 1);
  return startMs + frac * (endMs - startMs);
}

function heatmapCellsAtPoint(x, y, plot, s) {
  if (!state.heatmap.length || x < plot.left || x > plot.right || y < plot.top || y > plot.bottom) return [];
  const timestampMs = timestampForX(x, plot);
  const price = priceForY(y, s, plot);
  const index = indexAtX(x);
  const candidates = state.heatmapByBar.get(index) || [];
  return candidates.filter(cell => {
    if (!heatmapCellPassesColorFloor(cell)) return false;
    const low = Number(cell.price_low);
    const high = Number(cell.price_high);
    return Number.isFinite(low) && Number.isFinite(high)
      && timestampMs >= cell._startMs && timestampMs < cell._endMs
      && price >= low && price < high;
  }).sort((a, b) => effectiveHeatmapIntensity(b) - effectiveHeatmapIntensity(a));
}

function drawHoveredHeatmapCells(plot, s) {
  const cells = state.selectedHeatmapCells.length ? state.selectedHeatmapCells : state.hoverHeatmapCells;
  if (!cells.length) return;
  const { startMs, endMs } = visibleTimeRangeMs();
  const timeSpan = Math.max(1, endMs - startMs);
  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.left, plot.top, plot.width, plot.height);
  ctx.clip();
  for (const cell of cells.slice(0, 4)) {
    const low = Number(cell.price_low), high = Number(cell.price_high);
    const x1 = plot.left + ((cell._startMs - startMs) / timeSpan) * plot.width;
    const x2 = plot.left + ((cell._endMs - startMs) / timeSpan) * plot.width;
    const y1 = yForPrice(high, s, plot);
    const y2 = yForPrice(low, s, plot);
    ctx.strokeStyle = cell.side === 'bid' ? '#67e8f9' : '#fda4af';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 2]);
    ctx.strokeRect(Math.min(x1, x2) + 0.5, Math.min(y1, y2) + 0.5, Math.max(1, Math.abs(x2 - x1) - 1), Math.max(1, Math.abs(y2 - y1) - 1));
  }
  ctx.setLineDash([]);
  ctx.restore();
}

function formatHeatmapCell(cell) {
  const fields = cell?.fields || {};
  const sideLabel = cell?.side === 'bid' ? '买盘' : cell?.side === 'ask' ? '卖盘' : (cell?.side || '-');
  const unit = fields.unit || '';
  const depth = fmt(fields.depth, 4);
  const orders = fmt(fields.order_count, 0);
  const threshold = fields.rolling_large_threshold === null || fields.rolling_large_threshold === undefined
    ? '-'
    : `${fmt(fields.rolling_large_threshold, 4)} ${unit}`;
  return {
    sideLabel,
    unit,
    depth,
    orders,
    threshold,
    priceRange: `[${fmt(cell.price_low, 4)}, ${fmt(cell.price_high, 4)})`,
    timeRange: `${cell.start_timestamp} → ${cell.end_timestamp}`,
    sourceRange: fields.source_snapshot_start && fields.source_snapshot_end
      ? `${fields.source_snapshot_start} → ${fields.source_snapshot_end}`
      : '-',
    distance: fields.distance_bps === null || fields.distance_bps === undefined ? '-' : `${fmt(fields.distance_bps, 2)} bps`,
    distanceBand: fields.distance_band || '-',
    lag: fields.source_lag_ms === null || fields.source_lag_ms === undefined ? '-' : `${fmt(fields.source_lag_ms, 0)} ms`,
    large: Boolean(fields.is_large_rolling),
    rolling: `${fmt(fields.rolling_window_hours, 0)}h P${fmt(fields.rolling_percentile, 1)}`,
    added: fmt(fields.added_base, 4),
    removed: fmt(fields.removed_base, 4),
    executed: fmt(fields.executed_base, 4),
  };
}

function renderHeatmapCellDetail(cells) {
  const card = $('heatmapCellCard');
  const target = $('heatmapCellDetail');
  const grid = $('detailGrid');
  if (!card || !target) return;
  if (!state.pluginUi.heatmap_detail_card || !cells?.length) {
    card.classList.add('hidden');
    if (grid) grid.classList.remove('has-heatmap-detail');
    target.className = 'heatmap-cell-detail empty';
    target.textContent = '鼠标放到热力格上查看价格、时间与挂单数量';
    return;
  }
  card.classList.remove('hidden');
  if (grid) grid.classList.add('has-heatmap-detail');
  target.className = 'heatmap-cell-detail';
  target.innerHTML = cells.slice(0, 4).map(cell => {
    const item = formatHeatmapCell(cell);
    const sideClass = cell.side === 'bid' ? 'side-bid' : 'side-ask';
    return `<div class="heatmap-cell-row">
      <div class="${sideClass}">${htmlEscape(item.sideLabel)}${item.large ? ' · 大流动性' : ''}</div>
      <div class="heatmap-cell-grid">
        <div><small>价格格</small>${htmlEscape(item.priceRange)}</div>
        <div><small>挂单</small>${htmlEscape(item.depth)} ${htmlEscape(item.unit)}</div>
        <div><small>订单数</small>${htmlEscape(item.orders)}</div>
        <div><small>距离</small>${htmlEscape(item.distance)} · ${htmlEscape(item.distanceBand)}</div>
        <div><small>K线列</small>${htmlEscape(item.timeRange)}</div>
        <div><small>源快照</small>${htmlEscape(item.sourceRange)} · lag ${htmlEscape(item.lag)}</div>
        <div><small>滚动阈值</small>${htmlEscape(item.threshold)} · ${htmlEscape(item.rolling)}</div>
        <div><small>增/减/成交</small>${htmlEscape(item.added)} / ${htmlEscape(item.removed)} / ${htmlEscape(item.executed)} ETH</div>
      </div>
    </div>`;
  }).join('');
}

function heatmapAtIndex(index) {
  if (index === null || index < 0) return [];
  return state.heatmapByBar.get(index) || [];
}

function drawRegions(plot) {
  if (!state.regions.length) return;
  const span = Math.max(1, state.visibleEnd - state.visibleStart);
  const step = plot.width / span;
  for (const region of state.regions) {
    const startIndex = indexForTimestamp(region.start_timestamp);
    const endIndex = indexForTimestamp(region.end_timestamp);
    if (startIndex === null || endIndex === null) continue;
    if (endIndex < state.visibleStart || startIndex >= state.visibleEnd) continue;
    const leftIndex = Math.max(startIndex, state.visibleStart);
    const rightIndex = Math.min(endIndex, state.visibleEnd - 1);
    const x1 = xForIndex(leftIndex, plot) - step * 0.5;
    const x2 = xForIndex(rightIndex, plot) + step * 0.5;
    const color = region.color || '#fb923c';
    const opacity = Number.isFinite(Number(region.opacity)) ? Number(region.opacity) : 0.08;
    ctx.save();
    ctx.fillStyle = hexToRgba(color, opacity);
    ctx.fillRect(x1, plot.top, Math.max(1, x2 - x1), plot.height);
    ctx.strokeStyle = hexToRgba(color, Math.min(0.55, opacity * 4.5));
    ctx.lineWidth = 1;
    ctx.strokeRect(x1, plot.top + 0.5, Math.max(1, x2 - x1), plot.height - 1);
    ctx.restore();
  }
}

function regionsAtIndex(index) {
  if (index === null || index < 0) return [];
  return state.regions.filter(region => {
    const startIndex = indexForTimestamp(region.start_timestamp);
    const endIndex = indexForTimestamp(region.end_timestamp);
    return startIndex !== null && endIndex !== null && index >= startIndex && index <= endIndex;
  });
}

function drawMarkers(plot, s) {
  if (!state.markers.length) return;
  const m = markerMap();
  const span = state.visibleEnd - state.visibleStart;
  const step = plot.width / Math.max(1, span);
  for (let i = state.visibleStart; i < state.visibleEnd; i++) {
    const c = state.candles[i];
    const list = m.get(c.timestamp);
    if (!list) continue;
    const x = xForIndex(i, plot);
    list.forEach((marker, order) => {
      const color = marker.color || '#facc15';
      const role = marker.role || 'node';
      ctx.save();
      ctx.fillStyle = hexToRgba(color, role === 'signal' ? 0.16 : 0.10);
      ctx.fillRect(x - step * 0.5, plot.top, Math.max(2, step), plot.height);
      ctx.strokeStyle = color;

      // 节点标记的实心线
      // ctx.globalAlpha = role === 'signal' ? 0.95 : 0.70;
      // ctx.lineWidth = role === 'signal' ? 1.6 : 1;
      // ctx.beginPath();
      // ctx.moveTo(x, plot.top);
      // ctx.lineTo(x, plot.bottom);
      // ctx.stroke();
      // ctx.globalAlpha = 1;

      const fallbackPrice = marker.position === 'low' ? Number(c.low) : Number(c.high);
      const markerPrice = Number.isFinite(Number(marker.price)) ? Number(marker.price) : fallbackPrice;
      const baseY = yForPrice(markerPrice, s, plot);
      const y = Math.max(plot.top + 10, Math.min(plot.bottom - 10, baseY + order * 12));
      drawStageSymbol(x, y, color, role, marker.position || 'top', marker.symbol || 'auto');
      if (span <= 220) drawCompactMarkerLabel(x, y, marker.label || '', color, marker.position || 'top', order);
      ctx.restore();
    });
  }
}

function drawStageSymbol(x, y, color, role, position, symbol = 'auto') {
  ctx.save();
  ctx.shadowBlur = role === 'signal' ? 12 : 7;
  ctx.shadowColor = color;
  ctx.fillStyle = color;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.beginPath();

  if (symbol === 'arrow_down') {
    ctx.moveTo(x - 3, y - 8);
    ctx.lineTo(x + 3, y - 8);
    ctx.lineTo(x + 3, y);
    ctx.lineTo(x + 7, y);
    ctx.lineTo(x, y + 9);
    ctx.lineTo(x - 7, y);
    ctx.lineTo(x - 3, y);
  } else if (symbol === 'arrow_up') {
    ctx.moveTo(x, y - 9);
    ctx.lineTo(x + 7, y);
    ctx.lineTo(x + 3, y);
    ctx.lineTo(x + 3, y + 8);
    ctx.lineTo(x - 3, y + 8);
    ctx.lineTo(x - 3, y);
    ctx.lineTo(x - 7, y);
  } else if (role === 'signal') {
    ctx.moveTo(x, y - 8); ctx.lineTo(x + 7, y + 5); ctx.lineTo(x - 7, y + 5);
  } else if (role === 'start') {
    ctx.moveTo(x, y + 8); ctx.lineTo(x + 6, y - 4); ctx.lineTo(x - 6, y - 4);
  } else {
    ctx.moveTo(x, y - 6); ctx.lineTo(x + 6, y); ctx.lineTo(x, y + 6); ctx.lineTo(x - 6, y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawCompactMarkerLabel(x, y, label, color, position, order) {
  if (!label) return;
  ctx.save();
  ctx.font = '10px system-ui';
  const text = label.length > 11 ? label.slice(0, 11) : label;
  const width = ctx.measureText(text).width + 8;
  const labelY = position === 'low' ? y + 10 + order * 2 : y - 18 - order * 2;
  const left = x - width / 2;
  ctx.fillStyle = 'rgba(255,255,255,0.94)';
  roundRect(ctx, left, labelY, width, 16, 5);
  ctx.fill();
  ctx.strokeStyle = hexToRgba(color, 0.72);
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x, labelY + 8);
  ctx.restore();
}

function drawMarkerHalo(x, y, color) {
  ctx.save();
  ctx.shadowBlur = 14;
  ctx.shadowColor = color;
  ctx.strokeStyle = hexToRgba(color, 0.92);
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.arc(x, y, 6.5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = hexToRgba(color, 0.16);
  ctx.beginPath();
  ctx.arc(x, y, 11, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawMarkerTag(x, y, color, dir) {
  ctx.save();
  ctx.translate(x, y);
  ctx.shadowBlur = 14;
  ctx.shadowColor = color;
  ctx.fillStyle = color;
  ctx.beginPath();
  if (dir === 'up') {
    ctx.moveTo(0, -7); ctx.lineTo(7, 4); ctx.lineTo(0, 8); ctx.lineTo(-7, 4);
  } else {
    ctx.moveTo(0, 7); ctx.lineTo(7, -4); ctx.lineTo(0, -8); ctx.lineTo(-7, -4);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawSelectedCandle(plot, s) {
  const i = state.selectedIndex;
  if (i === null || i < state.visibleStart || i >= state.visibleEnd) return;
  const c = state.candles[i];
  if (!c) return;
  const span = state.visibleEnd - state.visibleStart;
  const step = plot.width / Math.max(1, span);
  const x = xForIndex(i, plot);
  const o = Number(c.open), h = Number(c.high), l = Number(c.low), cl = Number(c.close);
  const yH = yForPrice(h, s, plot), yL = yForPrice(l, s, plot), yO = yForPrice(o, s, plot), yC = yForPrice(cl, s, plot);
  const bodyW = Math.max(4, Math.min(18, step * 0.76));

  ctx.save();
  // Click-lock marker adapted for the CoinGlass-style light canvas.
  ctx.fillStyle = 'rgba(15,23,42,0.055)';
  ctx.fillRect(x - step * 0.5, plot.top, Math.max(2, step), plot.height);
  ctx.strokeStyle = 'rgba(15,23,42,0.72)';
  ctx.globalAlpha = 0.86;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(x, plot.top);
  ctx.lineTo(x, plot.bottom);
  ctx.stroke();
  ctx.globalAlpha = 1;

  // Add a subtle white outline around the selected candle body/wick so the exact K line is unambiguous.
  ctx.shadowBlur = 10;
  ctx.shadowColor = 'rgba(15,23,42,0.22)';
  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(x, yH);
  ctx.lineTo(x, yL);
  ctx.stroke();
  ctx.strokeRect(x - bodyW / 2 - 1.5, Math.min(yO, yC) - 1.5, bodyW + 3, Math.max(3, Math.abs(yC - yO)) + 3);
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.max(0, Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2));
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function hexToRgba(hex, alpha) {
  const clean = String(hex).replace('#', '');
  const n = parseInt(clean.length === 3 ? clean.split('').map(x => x + x).join('') : clean, 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

function drawAxes(plot, vol, s) {
  ctx.fillStyle = '#475569';
  ctx.font = '12px system-ui';
  ctx.textAlign = 'left';
  for (let i = 0; i <= 5; i++) {
    const price = s.maxP - (s.maxP - s.minP) * i / 5;
    const y = plot.top + plot.height * i / 5;
    ctx.fillText(fmt(price, 2), plot.right + 10, y + 4);
  }
  ctx.fillText('VOL', vol.right + 10, vol.top + 12);
  ctx.textAlign = 'center';
  const ticks = 6;
  for (let i = 0; i <= ticks; i++) {
    const idx = Math.round(state.visibleStart + (state.visibleEnd - state.visibleStart - 1) * i / ticks);
    const c = state.candles[idx];
    if (!c) continue;
    const x = xForIndex(idx, plot);
    ctx.fillText(c.timestamp.slice(5, 16), x, vol.bottom + 20);
  }
}

function drawCrosshair(plot, s) {
  const i = state.hoverIndex;
  if (i === null || i < state.visibleStart || i >= state.visibleEnd) return;
  const c = state.candles[i];
  const x = xForIndex(i, plot);
  const y = yForPrice(Number.isFinite(state.hoverPrice) ? state.hoverPrice : Number(c.close), s, plot);
  ctx.strokeStyle = 'rgba(51,65,85,0.34)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x, plot.top); ctx.lineTo(x, plot.bottom);
  ctx.moveTo(plot.left, y); ctx.lineTo(plot.right, y);
  ctx.stroke();
  ctx.setLineDash([]);
}

function rowFieldValue(field, index) {
  const payload = (state.rowFields || {})[field];
  if (Array.isArray(payload)) return payload[index] ?? null;
  if (payload && Array.isArray(payload.values)) {
    const rawValue = payload.values[index] ?? null;
    if (rawValue === null || rawValue === undefined) return null;
    return (payload.categories || {})[String(rawValue)] ?? rawValue;
  }
  return null;
}

function compactStateBrief(index) {
  const direction = rowFieldValue('brief_direction', index);
  const phase = rowFieldValue('brief_phase', index);
  const process = rowFieldValue('brief_process', index);
  const processProbability = rowFieldValue('brief_process_probability', index);
  const processUplift = rowFieldValue('brief_process_probability_uplift', index);
  const processSamples = rowFieldValue('brief_process_samples', index);
  if (direction === null && phase === null && process === null) return null;
  return {
    direction: direction || '-',
    phase: phase || '-',
    process: process || '-',
    processProbability,
    processUplift,
    processSamples,
    reason1: rowFieldValue('brief_reason_1', index),
    reason2: rowFieldValue('brief_reason_2', index),
    reason3: rowFieldValue('brief_reason_3', index),
    advice: rowFieldValue('brief_advice', index),
    detail: rowFieldValue('brief_context_detail', index),
  };
}

function briefColor(kind, value) {
  const text = String(value || '');
  if (text.includes('下方多头清算')) return '#ef4444';
  if (text.includes('上方空头清算')) return '#22d3ee';
  if (text.includes('偏多') || text.includes('多头') || text.includes('上涨结构') || text.includes('向上突破')) return '#22c55e';
  if (text.includes('偏空') || text.includes('空头') || text.includes('下跌结构') || text.includes('向下突破')) return '#ef4444';
  if (text.includes('暂停') || text.includes('冲击')) return '#f87171';
  if (text.includes('冲突') || text.includes('衰减')) return '#facc15';
  if (kind === 'phase' && (text.includes('回撤') || text.includes('反弹') || text.includes('压缩'))) return '#38bdf8';
  return '#94a3b8';
}

function renderMarketStateBrief(index) {
  const card = $('marketStateCard');
  const grid = $('detailGrid');
  const target = $('marketStateDetail');
  const brief = index === null ? null : compactStateBrief(index);
  if (!card || !grid || !target || !brief || state.pluginUi.brief_available === false) {
    if (card) card.classList.add('hidden');
    if (grid) grid.classList.remove('has-state-brief');
    return;
  }
  card.classList.remove('hidden');
  grid.classList.add('has-state-brief');
  const briefLabels = Array.isArray(state.pluginUi.brief_labels) ? state.pluginUi.brief_labels : ['方向', '阶段', '过程'];
  const pills = [
    [briefLabels[0] || '方向', brief.direction, briefColor('direction', brief.direction)],
    [briefLabels[1] || '阶段', brief.phase, briefColor('phase', brief.phase)],
    [briefLabels[2] || '过程', brief.process, briefColor('process', brief.process)],
  ];
  const reasons = [brief.reason1, brief.reason2, brief.reason3].filter(Boolean);
  target.className = 'market-state-detail';
  target.innerHTML = `
    <div class="state-pill-row">${pills.map(([label, value, color]) => `<span class="state-pill" style="--pill-color:${color}"><small>${htmlEscape(label)}</small><b>${htmlEscape(value)}</b></span>`).join('')}</div>
    <div class="state-advice">${htmlEscape(brief.advice || '暂无简洁建议')}</div>
    <ul class="state-reasons">${reasons.map(reason => `<li>${htmlEscape(reason)}</li>`).join('')}</ul>
    <div class="state-disclaimer">${htmlEscape(state.pluginUi.brief_disclaimer || '多阶段过程与条件概率用于描述市场，不是开仓或平仓信号。')}</div>`;
}

function positionTooltip(index, x, y, pointedHeatCells = []) {
  const c = state.candles[index];
  if (!c) return;
  const marks = markerMap().get(c.timestamp) || [];
  const regions = regionsAtIndex(index);
  const heatCells = pointedHeatCells.length ? pointedHeatCells : [];
  const markerLine = marks.length ? `<br><span style="color:${marks[0].color}">◆ ${htmlEscape(marks.map(m => m.label).join(', '))}</span>` : '';
  let heatLine = '';
  if (heatCells.length) {
    heatLine = heatCells.slice(0, 2).map(cell => {
      const item = formatHeatmapCell(cell);
      const sideColor = cell.side === 'bid' ? '#67e8f9' : '#fda4af';
      return `<br><span style="color:${sideColor}">▦ ${htmlEscape(item.sideLabel)} ${htmlEscape(item.priceRange)} · ${htmlEscape(item.depth)} ${htmlEscape(item.unit)} · ${htmlEscape(item.orders)} 单${item.large ? ' · P阈值+' : ''}</span>`;
    }).join('');
  } else if (state.pluginUi.heatmap_hover && state.heatmap.length) {
    heatLine = `<br><span style="color:#94a3b8">▦ 当前鼠标价格 ${fmt(state.hoverPrice, 4)}，此处无热力格</span>`;
  }
  const regionLine = regions.length ? `<br><span style="color:${regions[0].color || '#fb923c'}">▧ ${htmlEscape(regions.map(r => `${r.label || 'episode'} · ${r.status || ''}`).join(', '))}</span>` : '';
  const bandHits = stateBandCategoriesAt(index);
  const bandLine = bandHits.length ? bandHits.map(hit => `<br><span style="color:${hit.category.color || '#cbd5e1'}">▰ ${htmlEscape(hit.band.label || hit.band.id)}：${htmlEscape(hit.category.label || hit.category.status || '')}</span>`).join('') : '';
  let trackLine = '';
  if (!state.pluginUi.compact) {
    const trackText = state.tracks.slice(0, 4).map(track => {
      const values = Array.isArray(track.values) ? track.values : [];
      return `${track.label || track.id} ${fmt(values[index], 3)}`;
    }).join(' · ');
    trackLine = trackText ? `<br><span style="color:#cbd5e1">${htmlEscape(trackText)}</span>` : '';
  }
  tooltip.innerHTML = `<b>${htmlEscape(c.timestamp)}</b><br>鼠标价格 ${fmt(state.hoverPrice, 4)}<br>O ${fmt(c.open)} · H ${fmt(c.high)}<br>L ${fmt(c.low)} · C ${fmt(c.close)}<br>V ${fmt(c.volume)}${markerLine}${heatLine}${regionLine}${bandLine}${trackLine}`;
  tooltip.style.left = Math.min(x + 18, canvas.getBoundingClientRect().width - 360) + 'px';
  tooltip.style.top = Math.max(12, y - 28) + 'px';
  tooltip.classList.remove('hidden');
}

function updateDetails(index) {
  const c = index === null ? null : state.candles[index];
  const locked = index !== null && state.selectedIndex === index;
  const title = $('currentKlineTitle');
  if (title) title.textContent = locked ? '当前 K 线 · 已锁定' : '当前 K 线';
  if (!c) {
    renderMarketStateBrief(null);
    $('ohlcvDetail').className = 'ohlcv-detail empty';
    $('ohlcvDetail').textContent = '鼠标移动到 K 线上查看数据；点击 K 线可锁定，点击空白处取消';
    $('extraDetail').className = 'extra-detail empty';
    $('extraDetail').textContent = 'Trade Bar / Range Bar 的更多字段会显示在这里';
    return;
  }
  renderMarketStateBrief(index);
  $('ohlcvDetail').className = 'ohlcv-detail';
  const change = Number(c.close) - Number(c.open);
  const changePct = Number(c.open) ? change / Number(c.open) : 0;
  const marks = markerMap().get(c.timestamp) || [];
  const regions = regionsAtIndex(index);
  const heatCells = heatmapAtIndex(index);
  const items = [
    ['Time', c.timestamp], ['Open', fmt(c.open)], ['High', fmt(c.high)], ['Low', fmt(c.low)], ['Close', fmt(c.close)], ['Volume', fmt(c.volume)], ['Change', fmt(change)], ['Change %', (changePct * 100).toFixed(3) + '%']
  ];
  if (marks.length) items.push(['Marker', marks.map(m => m.label).join(', ')]);
  if (heatCells.length) items.push([state.pluginUi.heatmap_label || 'Heatmap', `${heatCells.length} 格`]);
  if (regions.length) items.push(['Episode', regions.map(r => `${r.label || 'episode'} · ${r.status || ''}`).join(', ')]);
  $('ohlcvDetail').innerHTML = items.map(([k, v]) => `<div class="metric"><span>${htmlEscape(k)}</span><b>${htmlEscape(v)}</b></div>`).join('');
  const extra = { ...(c.extra || {}) };
  const bandHits = stateBandCategoriesAt(index);
  for (const bandHit of bandHits) {
    extra[`state_${bandHit.band.id || 'band'}`] = bandHit.category.label || bandHit.category.status;
    for (const [k, v] of Object.entries(bandHit.category.fields || {})) extra[`state_${k}`] = v;
  }
  for (const [field, payload] of Object.entries(state.rowFields || {})) {
    if (Array.isArray(payload)) {
      extra[`state_${field}`] = payload[index] ?? null;
      continue;
    }
    if (payload && Array.isArray(payload.values)) {
      const rawValue = payload.values[index] ?? null;
      const categories = payload.categories || {};
      extra[`state_${field}`] = rawValue === null ? null : (categories[String(rawValue)] ?? rawValue);
    }
  }
  for (const track of state.tracks) {
    const values = Array.isArray(track.values) ? track.values : [];
    extra[`state_${track.id || 'track'}`] = values[index] ?? null;
  }
  for (const mark of marks) {
    for (const [k, v] of Object.entries(mark.fields || {})) extra[`marker_${k}`] = v;
    if (mark.reason) extra.marker_reason = mark.reason;
  }
  for (const heatCell of heatCells.slice().sort((a, b) => Number(b.intensity || 0) - Number(a.intensity || 0)).slice(0, 5)) {
    const prefix = `heat_${heatCell.side || 'zone'}_${Number(heatCell.price_low).toFixed(2)}`;
    extra[prefix] = `${fmt(heatCell.price_low)}-${fmt(heatCell.price_high)} · 强度 ${fmt(heatCell.intensity, 3)}`;
    for (const [k, v] of Object.entries(heatCell.fields || {})) extra[`${prefix}_${k}`] = v;
  }
  for (const region of regions) {
    for (const [k, v] of Object.entries(region.fields || {})) extra[`episode_${k}`] = v;
    if (region.status) extra.episode_status = region.status;
  }
  const entries = Object.entries(extra);
  $('extraDetail').className = entries.length ? 'extra-detail' : 'extra-detail empty';
  $('extraDetail').innerHTML = entries.length ? entries.map(([k, v]) => `<div class="metric"><span>${htmlEscape(k)}</span><b>${htmlEscape(fmtMaybe(v))}</b></div>`).join('') : '没有额外字段';
}

function fmtMaybe(v) {
  if (typeof v === 'number') return fmt(v);
  return v ?? '-';
}

init().catch(err => setStatus(err.message, true));
