const state = {
  episode: null,
  events: [],
  activePaneId: 'setup30',
  selectedPrice: null,
  selectedRawPrice: null,
  selectedSnapField: null,
  selectedAnchorTime: null,
  selectedSourceTimeframe: '30m',
  selectedPaneId: 'setup30',
  playTimer: null,
  loading: false,
  stepInFlight: false,
  clock: null,
  panes: {},
};

const PANE_DEFAULTS = {
  setup30: { timeframe: '30m', role: '前置 Setup A', visibleCount: 90 },
  setup15: { timeframe: '15m', role: '前置 Setup B', visibleCount: 120 },
  exec2: { timeframe: '2m', role: '执行 A', visibleCount: 180 },
  exec1: { timeframe: '1m', role: '执行 B', visibleCount: 220 },
};

const $ = id => document.getElementById(id);

function setStatus(text, error = false) {
  $('status').textContent = text;
  $('status').style.color = error ? '#ff9da5' : '#8da0b6';
}
function fmtPrice(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return n.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4});
}
function parseWallTime(text) {
  const raw = String(text || '').trim().replace(' ', 'T');
  const ms = Date.parse(raw + (/[zZ]|[+-]\d\d:\d\d$/.test(raw) ? '' : 'Z'));
  return Number.isFinite(ms) ? ms : NaN;
}
function escapeHtml(text) {
  return String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
async function request(path, options = {}) {
  const res = await fetch(path, {headers: {'Content-Type': 'application/json'}, ...options});
  const text = await res.text();
  let payload = {};
  try { payload = text ? JSON.parse(text) : {}; }
  catch { throw new Error(`后端响应不是 JSON: ${text.slice(0, 160)}`); }
  if (!res.ok || payload.ok === false) throw new Error(payload.error || `HTTP ${res.status}`);
  return payload;
}
function post(path, body = {}) { return request(path, {method: 'POST', body: JSON.stringify(body)}); }

function activeStartMode() {
  return document.querySelector('input[name="startMode"]:checked')?.value || 'random';
}
function updateStartMode() {
  const random = activeStartMode() === 'random';
  $('randomBlock').classList.toggle('hidden', !random);
  $('specificBlock').classList.toggle('hidden', random);
}
function activePane() { return state.panes[state.activePaneId]; }
function setActivePane(paneId) {
  if (!state.panes[paneId]) return;
  state.activePaneId = paneId;
  for (const card of document.querySelectorAll('.chart-card')) card.classList.toggle('active', card.dataset.pane === paneId);
  const pane = state.panes[paneId];
  $('activePaneInfo').textContent = `当前图：${pane.role} · ${pane.timeframe}`;
  $('currentChart').textContent = `${pane.role} · ${pane.timeframe}`;
}
function selectedContextPayload(extra = {}) {
  return {
    anchor_time: state.selectedAnchorTime,
    anchor_timeframe: state.selectedSourceTimeframe,
    source_pane: state.selectedPaneId,
    magnet_enabled: Boolean($('magnetToggle')?.checked),
    snap_field: state.selectedSnapField,
    raw_clicked_price: state.selectedRawPrice,
    ...extra,
  };
}

async function createEpisode() {
  stopPlay();
  const mode = activeStartMode();
  const body = {symbol: $('symbol').value.trim(), mode};
  if (mode === 'random') {
    body.random_start = $('randomStart').value;
    body.random_end = $('randomEnd').value;
  } else {
    body.start_date = $('specificStart').value;
  }
  try {
    setStatus('正在建立 causal replay...');
    const data = await post('/api/episodes', body);
    state.episode = data.episode;
    state.clock = data.clock || null;
    state.selectedPrice = null;
    state.selectedRawPrice = null;
    state.selectedSnapField = null;
    state.selectedAnchorTime = null;
    for (const pane of Object.values(state.panes)) {
      pane.visibleOffset = 0;
      pane.hoverIndex = null;
    }
    $('selectedPrice').textContent = '在任意一张图点击 K 线附近的价格';
    $('episodeBadge').textContent = `Episode ${state.episode.id}`;
    await refreshSnapshots(true);
    setStatus('Episode 已建立。四张图共享 cursor，未来数据不会返回浏览器。');
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function refreshSnapshots(resetView = false) {
  if (!state.episode || state.loading) return;
  state.loading = true;
  try {
    const timeframes = [...new Set(Object.values(state.panes).map(p => p.timeframe))];
    const q = new URLSearchParams({timeframes: timeframes.join(','), limit: '700'});
    const data = await request(`/api/episodes/${state.episode.id}/snapshots?${q}`);
    state.episode = data.episode;
    state.clock = data.clock || state.clock;
    state.events = data.events || [];
    for (const pane of Object.values(state.panes)) {
      const chart = data.charts?.[pane.timeframe] || {};
      pane.bars = chart.bars || [];
      pane.source = chart.source || '-';
      if (resetView) pane.visibleOffset = 0;
      const meta = $(`meta-${pane.id}`);
      if (meta) meta.textContent = `${pane.timeframe} · ${pane.bars.length} bars · ${pane.source}`;
    }
    updateClockHeader();
    $('chartMeta').textContent = `共享 cursor · ${timeframes.join(' / ')} · HTF 仅 closed bar 可见 · playback cache`;
    renderTimeline();
    deriveCurrentState();
    drawAll();
  } finally {
    state.loading = false;
  }
}

function updateClockHeader() {
  if (!state.episode) return;
  $('cursorTime').textContent = `${state.episode.cursor_time} ET · decision cursor`;
  const c = state.clock || {};
  $('clockInfo').textContent = c.new_york
    ? `${c.market_phase || ''} · ${c.new_york} · 北京 ${c.beijing || '-'} · 开盘 09:30 ET`
    : 'SOXL · 07:30 ET 开始 · 09:30 ET 开盘';
}
function appendIncrementalBars(updates) {
  for (const pane of Object.values(state.panes)) {
    const incoming = updates?.[pane.timeframe] || [];
    if (!incoming.length) continue;
    const byTime = new Map(pane.bars.map(b => [b.time, b]));
    for (const bar of incoming) byTime.set(bar.time, bar);
    pane.bars = [...byTime.values()].sort((a, b) => String(a.time).localeCompare(String(b.time))).slice(-700);
    const meta = $(`meta-${pane.id}`);
    if (meta) meta.textContent = `${pane.timeframe} · ${pane.bars.length} bars · ${pane.source}`;
  }
}
async function step(minutes) {
  if (!state.episode || state.episode.status !== 'active' || state.stepInFlight) return;
  state.stepInFlight = true;
  try {
    const timeframes = [...new Set(Object.values(state.panes).map(p => p.timeframe))];
    const data = await post(`/api/episodes/${state.episode.id}/step`, {minutes, timeframes});
    state.episode = data.episode;
    state.clock = data.clock || state.clock;
    appendIncrementalBars(data.updates || {});
    updateClockHeader();
    drawAll();
    if (data.at_data_end) {
      stopPlay();
      setStatus('已到当天 16:00 ET 或本地数据边界。');
    }
  } catch (err) {
    stopPlay();
    setStatus(err.message, true);
  } finally {
    state.stepInFlight = false;
  }
}
function startPlay() {
  if (!state.episode || state.playTimer) return;
  $('playBtn').textContent = 'Ⅱ 暂停';
  const run = async () => {
    await step(1);
    if (state.playTimer) state.playTimer = setTimeout(run, Number($('playSpeed').value || 350));
  };
  state.playTimer = setTimeout(run, 30);
}
function stopPlay() {
  if (state.playTimer) clearTimeout(state.playTimer);
  state.playTimer = null;
  $('playBtn').textContent = '▶ 自动播放';
}

async function addEvent(eventType, price = null, payload = {}, timeframe = null) {
  if (!state.episode) return setStatus('请先建立 Episode', true);
  const sourceTf = timeframe || activePane()?.timeframe || '1m';
  try {
    const body = {event_type: eventType, timeframe: sourceTf, payload};
    if (price !== null && Number.isFinite(Number(price))) body.price = Number(price);
    const data = await post(`/api/episodes/${state.episode.id}/events`, body);
    if (data.event) state.events.push(data.event);
    renderTimeline();
    deriveCurrentState();
    drawAll();
  } catch (err) {
    setStatus(err.message, true);
  }
}
async function trade(side) {
  if (!state.episode) return setStatus('请先建立 Episode', true);
  const sourceTf = activePane()?.timeframe || '1m';
  try {
    const data = await post(`/api/episodes/${state.episode.id}/trade`, {side, timeframe: sourceTf});
    if (data.event) state.events.push(data.event);
    setStatus(`${side} 模拟成交 @ ${fmtPrice(data.fill_price)}（cursor 1m open；判断来源 ${sourceTf}）`);
    renderTimeline();
    deriveCurrentState();
    drawAll();
  } catch (err) {
    setStatus(err.message, true);
  }
}
function requireSelectedPrice() {
  if (!Number.isFinite(state.selectedPrice)) {
    setStatus('先在任意一张图上点击一个价格。', true);
    return false;
  }
  return true;
}

function eventDetail(ev) {
  const p = ev.payload || {};
  const src = ev.timeframe ? ` · src ${ev.timeframe}` : '';
  if (ev.event_type === 'LIQUIDITY') return `${p.kind || 'Liquidity'} · ${p.importance || 'normal'} · ${fmtPrice(ev.price)}${src}${p.anchor_time ? ` · anchor ${p.anchor_time}` : ''}`;
  if (ev.event_type === 'BIAS') return `${p.bias || '-'}${src}`;
  if (ev.event_type === 'TARGET') return `Delivery → ${fmtPrice(ev.price)}${src}`;
  if (ev.event_type === 'MARKER') return `${p.label || 'Shared line'} · ${fmtPrice(ev.price)}${src}${p.anchor_time ? ` · anchor ${p.anchor_time}` : ''}`;
  if (['LONG','SHORT','CLOSE'].includes(ev.event_type) && Number.isFinite(Number(ev.price))) return `${ev.event_type} @ ${fmtPrice(ev.price)} · market${src}`;
  if (['SL','TP','MOVE_SL'].includes(ev.event_type)) return `${ev.event_type} ${fmtPrice(ev.price)}${src}`;
  if (ev.event_type === 'NOTE') return `${p.text || ''}${src}`;
  if (ev.event_type === 'CLOSE' && p.reason === 'episode_end') return 'Episode ended';
  return `${Object.keys(p).length ? JSON.stringify(p) : ''}${src}`;
}
function renderTimeline() {
  const root = $('timeline');
  if (!state.events.length) {
    root.className = 'timeline empty';
    root.textContent = '还没有事件。';
    return;
  }
  root.className = 'timeline';
  root.innerHTML = state.events.slice().reverse().map(ev => `
    <div class="timeline-item ${escapeHtml(ev.event_type)}">
      <strong>${escapeHtml(ev.event_type)}</strong>
      <small>${escapeHtml(ev.event_time)}${ev.timeframe ? ` · ${escapeHtml(ev.timeframe)}` : ''}</small>
      <div class="event-detail">${escapeHtml(eventDetail(ev))}</div>
    </div>`).join('');
}
function deriveCurrentState() {
  let bias = '-', target = '-', position = 'Flat', sl = '-', tp = '-';
  for (const ev of state.events) {
    if (ev.event_type === 'BIAS') bias = `${ev.payload?.bias || '-'}${ev.timeframe ? ` (${ev.timeframe})` : ''}`;
    if (ev.event_type === 'TARGET') target = `${fmtPrice(ev.price)}${ev.timeframe ? ` (${ev.timeframe})` : ''}`;
    if (ev.event_type === 'LONG' || ev.event_type === 'SHORT') position = `${ev.event_type} @ ${fmtPrice(ev.price)}`;
    if (ev.event_type === 'CLOSE' || ev.event_type === 'INVALIDATE') position = 'Flat';
    if (ev.event_type === 'SL' || ev.event_type === 'MOVE_SL') sl = fmtPrice(ev.price);
    if (ev.event_type === 'TP') tp = fmtPrice(ev.price);
  }
  $('currentBias').textContent = bias;
  $('currentTarget').textContent = target;
  $('currentPosition').textContent = position;
  $('currentSl').textContent = sl;
  $('currentTp').textContent = tp;
}

function visibleBars(pane) {
  const total = pane.bars.length;
  if (!total) return {bars: [], start: 0};
  const count = Math.max(30, Math.min(pane.visibleCount, total));
  const maxOffset = Math.max(0, total - count);
  pane.visibleOffset = Math.max(0, Math.min(pane.visibleOffset, maxOffset));
  const end = total - pane.visibleOffset;
  const start = Math.max(0, end - count);
  return {bars: pane.bars.slice(start, end), start};
}
function plotRect(pane) {
  const w = pane.canvas.clientWidth, h = pane.canvas.clientHeight;
  return {left: 10, top: 10, right: w - 70, bottom: h - 24, width: Math.max(10, w - 80), height: Math.max(10, h - 34)};
}
function priceRange(bars) {
  if (!bars.length) return {min: 0, max: 1};
  let min = Infinity, max = -Infinity;
  for (const b of bars) {
    min = Math.min(min, Number(b.low));
    max = Math.max(max, Number(b.high));
  }
  const localSpan = Math.max(max - min, max * .001, .01);
  for (const ev of state.events) {
    const price = Number(ev.price);
    if (!Number.isFinite(price)) continue;
    if (price >= min - localSpan * .45 && price <= max + localSpan * .45) {
      min = Math.min(min, price);
      max = Math.max(max, price);
    }
  }
  const pad = Math.max((max - min) * .08, max * .0012, .01);
  return {min: min - pad, max: max + pad};
}
function yForPrice(price, range, plot) {
  return plot.bottom - ((price - range.min) / Math.max(1e-9, range.max - range.min)) * plot.height;
}
function priceForY(y, range, plot) {
  return range.min + ((plot.bottom - y) / Math.max(1, plot.height)) * (range.max - range.min);
}
function xForTime(timeText, bars, plot) {
  if (!bars.length) return plot.left;
  const target = parseWallTime(timeText);
  const first = parseWallTime(bars[0].time);
  const last = parseWallTime(bars[bars.length - 1].time);
  if (!Number.isFinite(target) || !Number.isFinite(first) || !Number.isFinite(last) || last <= first) return plot.right;
  return Math.max(plot.left, Math.min(plot.right, plot.left + ((target - first) / (last - first)) * plot.width));
}
function drawPriceLabel(ctx, x, y, text, color) {
  ctx.font = '9px system-ui';
  const width = ctx.measureText(text).width + 7;
  ctx.fillStyle = color;
  ctx.fillRect(x, y - 8, width, 16);
  ctx.fillStyle = '#fff';
  ctx.fillText(text, x + 3, y + 3);
}
function drawGrid(pane, plot, range, bars) {
  const ctx = pane.ctx;
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#172331';
  ctx.fillStyle = '#75879a';
  ctx.font = '9px system-ui';
  const lines = 5;
  for (let i = 0; i <= lines; i++) {
    const y = plot.top + (plot.height / lines) * i;
    ctx.beginPath(); ctx.moveTo(plot.left, y); ctx.lineTo(plot.right, y); ctx.stroke();
    const p = range.max - (range.max - range.min) * (i / lines);
    ctx.fillText(fmtPrice(p), plot.right + 4, y + 3);
  }
  const xLines = 5;
  for (let i = 0; i <= xLines; i++) {
    const x = plot.left + plot.width * (i / xLines);
    ctx.beginPath(); ctx.moveTo(x, plot.top); ctx.lineTo(x, plot.bottom); ctx.stroke();
    const idx = Math.min(bars.length - 1, Math.floor((bars.length - 1) * (i / xLines)));
    const label = bars[idx]?.time?.slice(5, 16) || '';
    ctx.fillText(label, Math.max(plot.left, x - 24), plot.bottom + 14);
  }
}
function eventStyle(type, payload) {
  if (type === 'LIQUIDITY') return payload?.kind === 'SSL' ? {color:'#5fd6e8', dash:[6,4]} : {color:'#f6b95d', dash:[6,4]};
  if (type === 'TARGET') return {color:'#b08cff', dash:[8,4]};
  if (type === 'MARKER') return {color:'#9aa9ba', dash:[4,3]};
  if (type === 'SL' || type === 'MOVE_SL') return {color:'#ff6b76', dash:[3,3]};
  if (type === 'TP') return {color:'#42d392', dash:[3,3]};
  return null;
}
function drawEvents(pane, plot, range, bars) {
  const ctx = pane.ctx;
  for (const ev of state.events) {
    const price = Number(ev.price);
    if (!Number.isFinite(price)) continue;
    const style = eventStyle(ev.event_type, ev.payload);
    if (style) {
      const y = yForPrice(price, range, plot);
      if (y >= plot.top - 10 && y <= plot.bottom + 10) {
        ctx.strokeStyle = style.color;
        ctx.setLineDash(style.dash);
        ctx.globalAlpha = .86;
        const anchor = ev.payload?.anchor_time || ev.event_time;
        const startX = xForTime(anchor, bars, plot);
        ctx.beginPath(); ctx.moveTo(startX, y); ctx.lineTo(plot.right, y); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
        let label = ev.event_type;
        if (ev.event_type === 'LIQUIDITY') label = ev.payload?.kind || 'LIQ';
        if (ev.event_type === 'MARKER') label = ev.payload?.label || 'LINE';
        const source = ev.timeframe ? `[${ev.timeframe}] ` : '';
        ctx.fillStyle = style.color;
        ctx.font = '9px system-ui';
        ctx.fillText(`${source}${label} ${fmtPrice(price)}`, plot.left + 4, Math.max(plot.top + 10, y - 4));
      }
    }
    if (ev.event_type === 'LONG' || ev.event_type === 'SHORT') {
      const y = yForPrice(price, range, plot);
      if (y < plot.top - 10 || y > plot.bottom + 10) continue;
      const x = xForTime(ev.event_time, bars, plot);
      const color = ev.event_type === 'LONG' ? '#42d392' : '#ff6b76';
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
      ctx.font = '9px system-ui';
      ctx.fillText(ev.event_type, Math.min(plot.right - 34, x + 5), y - 5);
    }
  }
}
function drawCrosshair(pane, plot, range, bars, idx) {
  const ctx = pane.ctx;
  idx = Math.max(0, Math.min(bars.length - 1, idx));
  const stepX = plot.width / Math.max(1, bars.length);
  const x = plot.left + (idx + .5) * stepX;
  ctx.setLineDash([3,3]); ctx.strokeStyle = '#41556d';
  ctx.beginPath(); ctx.moveTo(x, plot.top); ctx.lineTo(x, plot.bottom); ctx.stroke(); ctx.setLineDash([]);
  const b = bars[idx];
  pane.crosshair.classList.remove('hidden');
  pane.crosshair.textContent = `${b.time}  O ${fmtPrice(b.open)} H ${fmtPrice(b.high)} L ${fmtPrice(b.low)} C ${fmtPrice(b.close)}`;
}
function drawPane(pane) {
  const w = pane.canvas.clientWidth, h = pane.canvas.clientHeight;
  const ctx = pane.ctx;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#090f16'; ctx.fillRect(0, 0, w, h);
  const plot = plotRect(pane);
  const view = visibleBars(pane);
  const bars = view.bars;
  if (!bars.length) {
    ctx.fillStyle = '#72859a'; ctx.font = '11px system-ui';
    ctx.fillText('建立 Episode 后显示 K 线', 18, 30);
    return;
  }
  const range = priceRange(bars);
  drawGrid(pane, plot, range, bars);
  const stepX = plot.width / Math.max(1, bars.length);
  const bodyW = Math.max(1, Math.min(8, stepX * .66));
  bars.forEach((b, i) => {
    const x = plot.left + (i + .5) * stepX;
    const o = yForPrice(Number(b.open), range, plot), c = yForPrice(Number(b.close), range, plot);
    const hi = yForPrice(Number(b.high), range, plot), lo = yForPrice(Number(b.low), range, plot);
    const up = Number(b.close) >= Number(b.open);
    ctx.strokeStyle = up ? '#42d392' : '#ff6b76';
    ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, hi); ctx.lineTo(x, lo); ctx.stroke();
    const top = Math.min(o, c), bh = Math.max(1, Math.abs(c - o));
    ctx.fillRect(x - bodyW / 2, top, bodyW, bh);
  });
  drawEvents(pane, plot, range, bars);
  if (pane.hoverIndex !== null) drawCrosshair(pane, plot, range, bars, pane.hoverIndex);
  if (Number.isFinite(state.selectedPrice)) {
    const y = yForPrice(state.selectedPrice, range, plot);
    if (y >= plot.top && y <= plot.bottom) {
      ctx.setLineDash([5,4]);
      ctx.strokeStyle = pane.id === state.selectedPaneId ? '#6aa8ff' : 'rgba(106,168,255,.42)';
      ctx.beginPath(); ctx.moveTo(plot.left, y); ctx.lineTo(plot.right, y); ctx.stroke(); ctx.setLineDash([]);
      drawPriceLabel(ctx, plot.right + 3, y, fmtPrice(state.selectedPrice), pane.id === state.selectedPaneId ? '#3d7dde' : '#31445a');
    }
  }
}
function drawAll() { for (const pane of Object.values(state.panes)) drawPane(pane); }

function resizePane(pane) {
  const rect = pane.wrap.getBoundingClientRect();
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  pane.canvas.width = Math.max(1, Math.round(rect.width * dpr));
  pane.canvas.height = Math.max(1, Math.round(rect.height * dpr));
  pane.canvas.style.width = `${rect.width}px`;
  pane.canvas.style.height = `${rect.height}px`;
  pane.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawPane(pane);
}
function attachPaneEvents(pane) {
  const canvas = pane.canvas;
  canvas.addEventListener('pointermove', e => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const plot = plotRect(pane);
    const view = visibleBars(pane);
    if (pane.dragging) {
      const dx = x - pane.dragStartX;
      const perBar = plot.width / Math.max(1, view.bars.length);
      const delta = Math.round(dx / Math.max(1, perBar));
      pane.visibleOffset = Math.max(0, pane.dragStartOffset + delta);
      drawPane(pane);
      return;
    }
    if (x < plot.left || x > plot.right || !view.bars.length) {
      pane.hoverIndex = null;
      pane.crosshair.classList.add('hidden');
      drawPane(pane);
      return;
    }
    const local = Math.floor((x - plot.left) / plot.width * view.bars.length);
    pane.hoverIndex = Math.max(0, Math.min(view.bars.length - 1, local));
    drawPane(pane);
  });
  canvas.addEventListener('pointerdown', e => {
    setActivePane(pane.id);
    pane.dragging = true;
    pane.dragStartX = e.clientX - canvas.getBoundingClientRect().left;
    pane.dragStartOffset = pane.visibleOffset;
    canvas.setPointerCapture?.(e.pointerId);
  });
  canvas.addEventListener('pointerup', e => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const moved = Math.abs(x - pane.dragStartX) > 5;
    pane.dragging = false;
    if (moved) return;
    const plot = plotRect(pane), view = visibleBars(pane);
    if (x < plot.left || x > plot.right || y < plot.top || y > plot.bottom || !view.bars.length) return;
    const range = priceRange(view.bars);
    const rawPrice = priceForY(y, range, plot);
    const localIndex = Math.max(0, Math.min(view.bars.length - 1, Math.floor((x - plot.left) / plot.width * view.bars.length)));
    const bar = view.bars[localIndex];
    state.selectedRawPrice = rawPrice;
    state.selectedSnapField = null;
    state.selectedPrice = rawPrice;
    if ($('magnetToggle')?.checked && bar) {
      const candidates = [
        ['O', Number(bar.open)], ['H', Number(bar.high)], ['L', Number(bar.low)], ['C', Number(bar.close)],
      ].filter(([, price]) => Number.isFinite(price));
      candidates.sort((a, b) => Math.abs(a[1] - rawPrice) - Math.abs(b[1] - rawPrice));
      if (candidates.length) {
        state.selectedSnapField = candidates[0][0];
        state.selectedPrice = candidates[0][1];
      }
    }
    state.selectedAnchorTime = bar?.time || null;
    state.selectedSourceTimeframe = pane.timeframe;
    state.selectedPaneId = pane.id;
    const snap = state.selectedSnapField ? ` · 🧲 ${state.selectedSnapField}` : ' · 自由价格';
    $('selectedPrice').textContent = `已选 ${fmtPrice(state.selectedPrice)}${snap} · ${pane.timeframe} · anchor ${state.selectedAnchorTime || '-'}`;
    drawAll();
  });
  canvas.addEventListener('pointerleave', () => {
    if (!pane.dragging) {
      pane.hoverIndex = null;
      pane.crosshair.classList.add('hidden');
      drawPane(pane);
    }
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    pane.visibleCount = Math.max(30, Math.min(600, pane.visibleCount + (e.deltaY > 0 ? 20 : -20)));
    drawPane(pane);
  }, {passive: false});
  pane.card.addEventListener('click', () => setActivePane(pane.id));
}
function initPanes() {
  for (const [id, cfg] of Object.entries(PANE_DEFAULTS)) {
    const canvas = $(`canvas-${id}`);
    const wrap = $(`wrap-${id}`);
    const card = document.querySelector(`.chart-card[data-pane="${id}"]`);
    const pane = {
      id, role: cfg.role, timeframe: cfg.timeframe,
      bars: [], source: '-', visibleCount: cfg.visibleCount, visibleOffset: 0,
      hoverIndex: null, dragging: false, dragStartX: 0, dragStartOffset: 0,
      canvas, wrap, card, ctx: canvas.getContext('2d'), crosshair: $(`crosshair-${id}`),
    };
    state.panes[id] = pane;
    attachPaneEvents(pane);
    new ResizeObserver(() => resizePane(pane)).observe(wrap);
    resizePane(pane);
  }
  setActivePane(state.activePaneId);
}

function exportEpisode() {
  if (!state.episode) return;
  window.open(`/api/episodes/${state.episode.id}/export`, '_blank', 'noopener');
}
async function endEpisode() {
  if (!state.episode) return;
  stopPlay();
  try {
    const data = await post(`/api/episodes/${state.episode.id}/close`, {});
    state.episode = data.episode;
    $('episodeBadge').textContent = `${state.episode.id} · closed`;
    await refreshSnapshots(false);
    setStatus('Episode 已结束并保存。');
  } catch (err) {
    setStatus(err.message, true);
  }
}

for (const radio of document.querySelectorAll('input[name="startMode"]')) radio.addEventListener('change', updateStartMode);
$('newEpisodeBtn').addEventListener('click', createEpisode);
for (const select of document.querySelectorAll('.pane-tf')) {
  select.addEventListener('change', async () => {
    const pane = state.panes[select.dataset.pane];
    if (!pane) return;
    pane.timeframe = select.value;
    pane.visibleOffset = 0;
    pane.hoverIndex = null;
    setActivePane(pane.id);
    await refreshSnapshots(true);
  });
}
for (const btn of document.querySelectorAll('.pane-fit')) {
  btn.addEventListener('click', e => {
    e.stopPropagation();
    const pane = state.panes[btn.dataset.pane];
    if (!pane) return;
    pane.visibleCount = PANE_DEFAULTS[pane.id].visibleCount;
    pane.visibleOffset = 0;
    drawPane(pane);
  });
}
for (const b of document.querySelectorAll('.bias')) {
  b.addEventListener('click', () => addEvent('BIAS', null, {bias: b.dataset.bias}));
}
$('markLiquidityBtn').addEventListener('click', () => {
  if (!requireSelectedPrice()) return;
  addEvent('LIQUIDITY', state.selectedPrice, selectedContextPayload({
    kind: $('liquidityKind').value,
    importance: $('liquidityImportance').value,
  }), state.selectedSourceTimeframe);
});
$('markTargetBtn').addEventListener('click', () => {
  if (!requireSelectedPrice()) return;
  addEvent('TARGET', state.selectedPrice, selectedContextPayload({role: 'expected_delivery'}), state.selectedSourceTimeframe);
});
$('markLineBtn').addEventListener('click', () => {
  if (!requireSelectedPrice()) return;
  const label = $('markerLabel').value.trim() || 'Shared line';
  addEvent('MARKER', state.selectedPrice, selectedContextPayload({label}), state.selectedSourceTimeframe);
});
$('setSlBtn').addEventListener('click', () => {
  if (!requireSelectedPrice()) return;
  addEvent('SL', state.selectedPrice, selectedContextPayload({role: 'stop_loss'}), state.selectedSourceTimeframe);
});
$('setTpBtn').addEventListener('click', () => {
  if (!requireSelectedPrice()) return;
  addEvent('TP', state.selectedPrice, selectedContextPayload({role: 'take_profit'}), state.selectedSourceTimeframe);
});
$('watchBtn').addEventListener('click', () => addEvent('WATCH'));
$('waitBtn').addEventListener('click', () => addEvent('WAIT'));
$('skipBtn').addEventListener('click', () => addEvent('SKIP'));
$('invalidateBtn').addEventListener('click', () => addEvent('INVALIDATE'));
$('longBtn').addEventListener('click', () => trade('LONG'));
$('shortBtn').addEventListener('click', () => trade('SHORT'));
$('closeTradeBtn').addEventListener('click', () => trade('CLOSE'));
$('step1Btn').addEventListener('click', () => step(1));
$('step2Btn').addEventListener('click', () => step(2));
$('step5Btn').addEventListener('click', () => step(5));
$('step15Btn').addEventListener('click', () => step(15));
$('playBtn').addEventListener('click', () => state.playTimer ? stopPlay() : startPlay());
$('saveNoteBtn').addEventListener('click', async () => {
  const text = $('noteText').value.trim();
  if (!text) return;
  await addEvent('NOTE', null, {text});
  $('noteText').value = '';
});
$('exportBtn').addEventListener('click', exportEpisode);
$('endEpisodeBtn').addEventListener('click', endEpisode);

$('magnetToggle').addEventListener('change', () => {
  setStatus($('magnetToggle').checked ? '磁铁已开启：下一次点击自动吸附该 K 的 O/H/L/C。' : '磁铁已关闭：下一次点击使用自由价格。');
});

updateStartMode();
initPanes();
