const state = {
  config: null,
  plugins: [],
  candles: [],
  markers: [],
  visibleStart: 0,
  visibleEnd: 200,
  hoverIndex: null,
  dragging: false,
  dragX: 0,
  dragStartStart: 0,
  dragStartEnd: 0,
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

function htmlEscape(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function getJson(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
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
  $('clearMarkersBtn').addEventListener('click', () => { state.markers = []; $('pluginSummary').textContent = ''; draw(); });
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

function renderPluginParams() {
  const p = selectedPlugin();
  const wrap = $('pluginParams');
  if (!p) { wrap.innerHTML = '<div class="mini">暂无插件</div>'; return; }
  wrap.innerHTML = (p.params || []).map(param => {
    const id = `plugin_${param.name}`;
    if (param.kind === 'select') {
      const opts = (param.choices || []).map(c => `<option value="${htmlEscape(c.value)}" ${c.value === param.default ? 'selected' : ''}>${htmlEscape(c.label)}</option>`).join('');
      return `<label>${htmlEscape(param.label)}<select id="${id}" data-param="${htmlEscape(param.name)}">${opts}</select></label>`;
    }
    if (param.kind === 'color') {
      return `<label>${htmlEscape(param.label)}<input id="${id}" data-param="${htmlEscape(param.name)}" type="color" value="${htmlEscape(param.default || '#facc15')}" /></label>`;
    }
    return `<label>${htmlEscape(param.label)}<input id="${id}" data-param="${htmlEscape(param.name)}" type="number" value="${htmlEscape(param.default)}" min="${param.min ?? ''}" max="${param.max ?? ''}" step="${param.step ?? 'any'}" /></label>`;
  }).join('');
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
    state.candles = data.candles || [];
    state.markers = [];
    state.hoverIndex = null;
    const n = state.candles.length;
    state.visibleEnd = n;
    state.visibleStart = Math.max(0, n - Math.min(320, n));
    $('chartTitle').textContent = `${data.meta.symbol} · ${data.meta.data_type}`;
    $('chartSub').textContent = n ? `${data.meta.start} → ${data.meta.end} · ${n} bars · ${data.meta.table_name || ''}` : `无数据：${data.meta.table_name || ''}`;
    setStatus(n ? `已加载 ${n} 根，来源 ${data.meta.loader}` : '没有读到数据。确认本地 DB 覆盖，或取消“只读本地缓存”让 loader 自动补数据。', n === 0);
    updateDetails(null);
    draw();
  } catch (err) {
    setStatus(err.message, true);
  }
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
    const s = data.summary || {};
    $('pluginSummary').textContent = `匹配 ${s.matched ?? 0} / ${s.input_rows ?? 0} 根；上影 ${s.upper_count ?? '-'}，下影 ${s.lower_count ?? '-'}`;
    setStatus(`插件完成：${plugin.name}，标记 ${state.markers.length} 根`);
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
    const plot = plotArea();
    const frac = Math.max(0, Math.min(1, (x - plot.left) / plot.width));
    const span = state.visibleEnd - state.visibleStart;
    const zoom = e.deltaY < 0 ? 0.82 : 1.22;
    const newSpan = Math.max(20, Math.min(state.candles.length, Math.round(span * zoom)));
    const anchor = state.visibleStart + span * frac;
    state.visibleStart = Math.round(anchor - newSpan * frac);
    state.visibleEnd = state.visibleStart + newSpan;
    clampVisible();
    draw();
  }, { passive: false });

  canvas.addEventListener('mousedown', (e) => {
    state.dragging = true;
    state.dragX = e.clientX;
    state.dragStartStart = state.visibleStart;
    state.dragStartEnd = state.visibleEnd;
  });
  window.addEventListener('mouseup', () => { state.dragging = false; });
  window.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const inside = e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom;
    if (state.dragging && state.candles.length) {
      const span = state.visibleEnd - state.visibleStart;
      const dx = e.clientX - state.dragX;
      const barsPerPx = span / Math.max(1, plotArea().width);
      const shift = Math.round(-dx * barsPerPx);
      state.visibleStart = state.dragStartStart + shift;
      state.visibleEnd = state.dragStartEnd + shift;
      clampVisible();
      draw();
      return;
    }
    if (!inside || !state.candles.length) {
      tooltip.classList.add('hidden');
      return;
    }
    const localX = e.clientX - rect.left;
    const localY = e.clientY - rect.top;
    const index = indexAtX(localX);
    state.hoverIndex = index;
    updateDetails(index);
    positionTooltip(index, localX, localY);
    draw();
  });
  canvas.addEventListener('mouseleave', () => {
    state.hoverIndex = null;
    tooltip.classList.add('hidden');
    draw();
  });
}

function clampVisible() {
  const n = state.candles.length;
  const span = state.visibleEnd - state.visibleStart;
  if (state.visibleStart < 0) { state.visibleStart = 0; state.visibleEnd = span; }
  if (state.visibleEnd > n) { state.visibleEnd = n; state.visibleStart = Math.max(0, n - span); }
}

function plotArea() {
  const rect = canvas.getBoundingClientRect();
  return { left: 64, top: 24, right: rect.width - 78, bottom: rect.height - 88, width: rect.width - 142, height: rect.height - 112 };
}

function volumeArea() {
  const rect = canvas.getBoundingClientRect();
  return { left: 64, top: rect.height - 76, right: rect.width - 78, bottom: rect.height - 30, width: rect.width - 142, height: 46 };
}

function visibleCandles() {
  return state.candles.slice(state.visibleStart, state.visibleEnd);
}

function scales() {
  const vis = visibleCandles();
  const highs = vis.map(c => Number(c.high)).filter(Number.isFinite);
  const lows = vis.map(c => Number(c.low)).filter(Number.isFinite);
  const vols = vis.map(c => Number(c.volume)).filter(Number.isFinite);
  let minP = Math.min(...lows), maxP = Math.max(...highs);
  if (!Number.isFinite(minP) || !Number.isFinite(maxP)) { minP = 0; maxP = 1; }
  const pad = Math.max((maxP - minP) * 0.08, Math.abs(maxP) * 0.0005, 1e-9);
  minP -= pad; maxP += pad;
  const maxV = Math.max(...vols, 1);
  return { minP, maxP, maxV };
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
    ctx.fillStyle = '#93a4b8';
    ctx.font = '14px system-ui';
    ctx.fillText('点击左侧“加载图表”。如果没有数据，先用项目已有 prebuild/download 工具准备本地 DB。', 72, 70);
    return;
  }
  const plot = plotArea();
  const vol = volumeArea();
  const s = scales();
  drawGrid(plot, vol, s);
  drawMarkers(plot);
  drawCandles(plot, vol, s);
  drawAxes(plot, vol, s);
  drawCrosshair(plot, s);
}

function drawBackground(rect) {
  const g = ctx.createLinearGradient(0, 0, 0, rect.height);
  g.addColorStop(0, 'rgba(15,23,42,0.15)');
  g.addColorStop(1, 'rgba(2,6,23,0.1)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, rect.width, rect.height);
}

function drawGrid(plot, vol, s) {
  ctx.strokeStyle = 'rgba(148,163,184,0.11)';
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

function markerMap() {
  const m = new Map();
  for (const marker of state.markers) {
    const key = marker.timestamp;
    if (!m.has(key)) m.set(key, []);
    m.get(key).push(marker);
  }
  return m;
}

function drawMarkers(plot) {
  if (!state.markers.length) return;
  const m = markerMap();
  const span = state.visibleEnd - state.visibleStart;
  const step = plot.width / Math.max(1, span);
  for (let i = state.visibleStart; i < state.visibleEnd; i++) {
    const c = state.candles[i];
    const list = m.get(c.timestamp);
    if (!list) continue;
    const x = xForIndex(i, plot);
    const color = list[0].color || '#facc15';
    ctx.fillStyle = hexToRgba(color, 0.16);
    ctx.fillRect(x - step * 0.5, plot.top, Math.max(2, step), plot.height);
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.72;
    ctx.beginPath(); ctx.moveTo(x, plot.top); ctx.lineTo(x, plot.bottom); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, plot.top + 10, 4, 0, Math.PI * 2); ctx.fill();
  }
}

function hexToRgba(hex, alpha) {
  const clean = String(hex).replace('#', '');
  const n = parseInt(clean.length === 3 ? clean.split('').map(x => x + x).join('') : clean, 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

function drawAxes(plot, vol, s) {
  ctx.fillStyle = '#93a4b8';
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
  const y = yForPrice(Number(c.close), s, plot);
  ctx.strokeStyle = 'rgba(226,232,240,0.35)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x, plot.top); ctx.lineTo(x, plot.bottom);
  ctx.moveTo(plot.left, y); ctx.lineTo(plot.right, y);
  ctx.stroke();
  ctx.setLineDash([]);
}

function positionTooltip(index, x, y) {
  const c = state.candles[index];
  if (!c) return;
  const marks = markerMap().get(c.timestamp) || [];
  tooltip.innerHTML = `<b>${htmlEscape(c.timestamp)}</b><br>O ${fmt(c.open)} · H ${fmt(c.high)}<br>L ${fmt(c.low)} · C ${fmt(c.close)}<br>V ${fmt(c.volume)}${marks.length ? `<br><span style="color:${marks[0].color}">● ${htmlEscape(marks.map(m => m.label).join(', '))}</span>` : ''}`;
  tooltip.style.left = Math.min(x + 18, canvas.getBoundingClientRect().width - 240) + 'px';
  tooltip.style.top = Math.max(12, y - 28) + 'px';
  tooltip.classList.remove('hidden');
}

function updateDetails(index) {
  const c = index === null ? null : state.candles[index];
  if (!c) {
    $('ohlcvDetail').className = 'ohlcv-detail empty';
    $('ohlcvDetail').textContent = '鼠标移动到 K 线上查看数据';
    $('extraDetail').className = 'extra-detail empty';
    $('extraDetail').textContent = 'Trade Bar / Range Bar 的更多字段会显示在这里';
    return;
  }
  $('ohlcvDetail').className = 'ohlcv-detail';
  const change = Number(c.close) - Number(c.open);
  const changePct = Number(c.open) ? change / Number(c.open) : 0;
  const items = [
    ['Time', c.timestamp], ['Open', fmt(c.open)], ['High', fmt(c.high)], ['Low', fmt(c.low)], ['Close', fmt(c.close)], ['Volume', fmt(c.volume)], ['Change', fmt(change)], ['Change %', (changePct * 100).toFixed(3) + '%']
  ];
  $('ohlcvDetail').innerHTML = items.map(([k, v]) => `<div class="metric"><span>${htmlEscape(k)}</span><b>${htmlEscape(v)}</b></div>`).join('');
  const extra = c.extra || {};
  const entries = Object.entries(extra);
  $('extraDetail').className = entries.length ? 'extra-detail' : 'extra-detail empty';
  $('extraDetail').innerHTML = entries.length ? entries.map(([k, v]) => `<div class="metric"><span>${htmlEscape(k)}</span><b>${htmlEscape(fmtMaybe(v))}</b></div>`).join('') : '没有额外字段';
}

function fmtMaybe(v) {
  if (typeof v === 'number') return fmt(v);
  return v ?? '-';
}

init().catch(err => setStatus(err.message, true));
