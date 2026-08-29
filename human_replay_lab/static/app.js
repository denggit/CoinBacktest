const state = {
  episode: null,
  events: [],
  activeLimitOrders: [],
  tradeSummary: null,
  activePaneId: 'main',
  selectedPrice: null,
  selectedRawPrice: null,
  selectedSnapField: null,
  selectedAnchorTime: null,
  selectedAnchorIsPartial: false,
  selectedAnchorObservedThrough: null,
  selectedAnchorDisplayTime: null,
  selectedSourceTimeframe: '30m',
  selectedPaneId: 'main',
  playTimer: null,
  loading: false,
  stepInFlight: false,
  clock: null,
  panes: {},
  startMode: null,
  sequenceSymbol: null,
  timeframeSlots: [],
  activeTimeframeSlot: 0,
};

const SUPPORTED_TIMEFRAMES_UI = ['1m', '2m', '5m', '15m', '30m', '1H', '4H', '1D'];
const DEFAULT_TIMEFRAME_SLOTS = ['30m', '15m', '5m', '2m', '1m', '4H'];
const TIMEFRAME_SLOT_STORAGE_KEY = 'humanReplayLab.timeframeSlots.v1';
const ACTIVE_TIMEFRAME_SLOT_STORAGE_KEY = 'humanReplayLab.activeTimeframeSlot.v1';

const PANE_DEFAULTS = {
  main: { timeframe: '30m', role: '前置 Setup', visibleCount: 150 },
};

function roleForTimeframe(tf) {
  if (tf === '30m' || tf === '15m') return '前置 Setup';
  if (tf === '2m' || tf === '1m') return '执行';
  return '上下文';
}

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
function readPriceInput(id, {required = false} = {}) {
  const raw = String($(id)?.value || '').trim();
  if (!raw) {
    if (required) throw new Error(`${id === 'entryPriceInput' ? '挂单价格' : id} 不能为空`);
    return null;
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) throw new Error('价格必须是大于 0 的数字');
  return value;
}
function setTicketPrice(id, value) {
  if (!Number.isFinite(Number(value))) return;
  $(id).value = Number(value).toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
}
function updateTicketSelectedPrice() {
  const text = Number.isFinite(state.selectedPrice)
    ? `${fmtPrice(state.selectedPrice)}${state.selectedSnapField ? ` · 🧲 ${state.selectedSnapField}` : ' · 自由价格'} · ${state.selectedSourceTimeframe}`
    : '-';
  if ($('ticketSelectedPrice')) $('ticketSelectedPrice').textContent = `当前图上选中价：${text}`;
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
function isContinuous24x7Symbol(symbol = null) {
  return ['ETH-USDT-SWAP', 'XAU-USDT-SWAP'].includes(String(symbol || $('symbol')?.value || '').toUpperCase().trim());
}
function updateSessionProfileUi() {
  const continuous = isContinuous24x7Symbol();
  $('specificDateLabel')?.classList.toggle('hidden', continuous);
  $('specificDatetimeLabel')?.classList.toggle('hidden', !continuous);
  if ($('sessionProfileHint')) $('sessionProfileHint').textContent = continuous
    ? `${$('symbol')?.value || '所选标的'}：OKX 24/7，无固定开盘/收盘限制，周末也可 Replay；指定起点使用北京时间。Entry+SL+TP 成交后，TP/SL 任一命中会自动结束 Episode。`
    : 'SOXL：只在工作日创建 Episode，纽约 07:30 开始，16:00 结束；图表仍保留全部 OKX 盘外上下文。';
}
function updateSequenceUi() {
  const mode = activeStartMode();
  const sequential = mode === 'sequential';
  const canResume = sequential && state.startMode === 'sequential' && state.episode?.status === 'closed'
    && state.episode?.symbol === String($('symbol')?.value || '').trim();
  if ($('newEpisodeBtn')) $('newEpisodeBtn').textContent = canResume ? '继续下一 Episode' : (sequential ? '从所选日期开始 Episode' : '新建 Episode');
  if ($('sequenceProgress')) {
    $('sequenceProgress').classList.toggle('hidden', !sequential);
    $('sequenceProgress').textContent = canResume
      ? `顺序模式：下一 Episode 将从上一 Episode 结束时间之后继续。上一 Episode：${state.episode?.id || '-'}`
      : '顺序模式不会随机跳时间；完成一笔后继续点击这里，会沿时间轴向后推进。';
  }
}
function updateStartMode() {
  const mode = activeStartMode();
  $('randomBlock').classList.toggle('hidden', mode !== 'random');
  $('sequentialBlock')?.classList.toggle('hidden', mode !== 'sequential');
  $('specificBlock').classList.toggle('hidden', mode !== 'specific');
  const continuous = isContinuous24x7Symbol();
  if ($('sequentialTimeLabel')) $('sequentialTimeLabel').classList.toggle('hidden', !continuous);
  if ($('sequentialHint')) $('sequentialHint').textContent = continuous
    ? `${$('symbol')?.value || '所选标的'}：第一次从所选北京时间开始；以后每个 Episode 从上一 Episode 结束后的下一根可用 1m K 继续。`
    : 'SOXL：第一次从所选工作日 07:30 ET 开始；之后顺序跳到下一个有本地数据的工作日 07:30 ET。';
  updateSessionProfileUi();
  updateSequenceUi();
}
function activePane() { return state.panes[state.activePaneId]; }
function setActivePane(paneId) {
  if (!state.panes[paneId]) return;
  state.activePaneId = paneId;
  const pane = state.panes[paneId];
  pane.role = roleForTimeframe(pane.timeframe);
  $('activePaneInfo').textContent = `当前图：${pane.timeframe} · ${pane.role}`;
  $('currentChart').textContent = `${pane.timeframe} · ${pane.role}`;
  if ($('mainChartRole')) $('mainChartRole').textContent = pane.role;
  updateTimeframeUi(pane.timeframe);
}

function sanitizeTimeframeSlots(values) {
  const source = Array.isArray(values) ? values : [];
  const result = DEFAULT_TIMEFRAME_SLOTS.map((fallback, index) => {
    const candidate = String(source[index] || '').trim();
    return SUPPORTED_TIMEFRAMES_UI.includes(candidate) ? candidate : fallback;
  });
  return result;
}
function loadTimeframeSlotPreferences() {
  let slots = DEFAULT_TIMEFRAME_SLOTS.slice();
  let activeSlot = 0;
  try {
    const raw = window.localStorage?.getItem(TIMEFRAME_SLOT_STORAGE_KEY);
    if (raw) slots = sanitizeTimeframeSlots(JSON.parse(raw));
    const activeRaw = Number(window.localStorage?.getItem(ACTIVE_TIMEFRAME_SLOT_STORAGE_KEY));
    if (Number.isInteger(activeRaw) && activeRaw >= 0 && activeRaw < 6) activeSlot = activeRaw;
  } catch (_) {
    slots = DEFAULT_TIMEFRAME_SLOTS.slice();
    activeSlot = 0;
  }
  state.timeframeSlots = slots;
  state.activeTimeframeSlot = activeSlot;
}
function persistTimeframeSlotPreferences() {
  try {
    window.localStorage?.setItem(TIMEFRAME_SLOT_STORAGE_KEY, JSON.stringify(state.timeframeSlots));
    window.localStorage?.setItem(ACTIVE_TIMEFRAME_SLOT_STORAGE_KEY, String(state.activeTimeframeSlot));
  } catch (_) {}
}
function timeframeOptionsHtml(selected) {
  return SUPPORTED_TIMEFRAMES_UI.map(tf => `<option value="${tf}"${tf === selected ? ' selected' : ''}>${tf}</option>`).join('');
}
function renderTimeframeSlots() {
  if (!state.timeframeSlots.length) loadTimeframeSlotPreferences();
  for (const wrapper of document.querySelectorAll('.tf-slot')) {
    const index = Number(wrapper.dataset.slot);
    const select = wrapper.querySelector('.tf-slot-select');
    const role = wrapper.querySelector('.tf-slot-role');
    const tf = state.timeframeSlots[index] || DEFAULT_TIMEFRAME_SLOTS[index];
    if (select) {
      if (!select.options.length) select.innerHTML = timeframeOptionsHtml(tf);
      if (select.value !== tf) select.value = tf;
    }
    if (role) role.textContent = roleForTimeframe(tf);
    wrapper.classList.toggle('active', index === state.activeTimeframeSlot);
  }
}
function updateTimeframeUi(tf) {
  renderTimeframeSlots();
  const activeTf = state.timeframeSlots[state.activeTimeframeSlot];
  // If the chart was changed by a non-slot path, keep the active slot identity but
  // don't silently rewrite its saved configuration.
  for (const wrapper of document.querySelectorAll('.tf-slot')) {
    const index = Number(wrapper.dataset.slot);
    wrapper.classList.toggle('active', index === state.activeTimeframeSlot && activeTf === tf);
  }
}

async function changeMainTimeframe(tf, {slotIndex = null} = {}) {
  if (!SUPPORTED_TIMEFRAMES_UI.includes(tf)) return;
  if (Number.isInteger(slotIndex) && slotIndex >= 0 && slotIndex < 6) {
    state.activeTimeframeSlot = slotIndex;
    persistTimeframeSlotPreferences();
  }
  const pane = activePane();
  if (!pane || pane.timeframe === tf) { setActivePane('main'); renderTimeframeSlots(); return; }
  stopPlay();
  pane.timeframe = tf;
  pane.role = roleForTimeframe(tf);
  pane.visibleOffset = 0;
  pane.hoverIndex = null;
  setActivePane('main');
  renderTimeframeSlots();
  if (state.episode) await refreshSnapshots(true);
  else drawPane(pane);
}
async function configureTimeframeSlot(index, tf) {
  if (!Number.isInteger(index) || index < 0 || index >= 6 || !SUPPORTED_TIMEFRAMES_UI.includes(tf)) return;
  state.timeframeSlots[index] = tf;
  state.activeTimeframeSlot = index;
  persistTimeframeSlotPreferences();
  renderTimeframeSlots();
  await changeMainTimeframe(tf, {slotIndex: index});
}
async function resetTimeframeSlots() {
  state.timeframeSlots = DEFAULT_TIMEFRAME_SLOTS.slice();
  state.activeTimeframeSlot = 0;
  persistTimeframeSlotPreferences();
  renderTimeframeSlots();
  await changeMainTimeframe(state.timeframeSlots[0], {slotIndex: 0});
  setStatus('6 个快捷周期已恢复默认：30m / 15m / 5m / 2m / 1m / 4H。');
}
function selectedContextPayload(extra = {}) {
  return {
    anchor_time: state.selectedAnchorTime,
    anchor_timeframe: state.selectedSourceTimeframe,
    source_pane: state.selectedPaneId,
    anchor_is_partial: Boolean(state.selectedAnchorIsPartial),
    anchor_observed_through: state.selectedAnchorObservedThrough || null,
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
  } else if (mode === 'sequential') {
    const canResume = state.startMode === 'sequential' && state.episode?.status === 'closed' && state.episode?.symbol === body.symbol;
    if (canResume) {
      body.previous_episode_id = state.episode.id;
    } else if (isContinuous24x7Symbol(body.symbol)) {
      const day = $('sequentialStartDate').value;
      const clock = $('sequentialStartTime')?.value || '00:00';
      if (!day) { setStatus('请选择顺序 Replay 起始日期', true); return; }
      body.start_time = `${day}T${clock}`;
    } else {
      body.start_date = $('sequentialStartDate').value;
      if (!body.start_date) { setStatus('请选择顺序 Replay 起始日期', true); return; }
    }
  } else {
    if (isContinuous24x7Symbol(body.symbol)) body.start_time = $('specificStartTime').value;
    else body.start_date = $('specificStart').value;
  }
  try {
    setStatus('正在建立 causal replay...');
    const data = await post('/api/episodes', body);
    state.episode = data.episode;
    state.startMode = mode;
    state.sequenceSymbol = mode === 'sequential' ? body.symbol : null;
    state.clock = data.clock || null;
    state.selectedPrice = null;
    state.selectedRawPrice = null;
    state.selectedSnapField = null;
    state.selectedAnchorTime = null;
    state.selectedAnchorIsPartial = false;
    state.selectedAnchorObservedThrough = null;
    state.selectedAnchorDisplayTime = null;
    state.activeLimitOrders = [];
    state.tradeSummary = null;
    for (const pane of Object.values(state.panes)) {
      pane.visibleOffset = 0;
      pane.hoverIndex = null;
    }
    $('selectedPrice').textContent = '在主图点击 K 线附近的价格';
    for (const id of ['entryPriceInput','slPriceInput','tpPriceInput']) if ($(id)) $(id).value = '';
    updateTicketSelectedPrice();
    $('episodeBadge').textContent = `Episode ${state.episode.id}`;
    updateSequenceUi();
    await refreshSnapshots(true);
    setStatus('Episode 已建立。主图可随时切换周期；高周期形成中 K 只由已关闭 1m 更新。');
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
    state.activeLimitOrders = data.active_limit_orders || [];
    state.tradeSummary = data.trade_summary || state.tradeSummary;
    for (const pane of Object.values(state.panes)) {
      const chart = data.charts?.[pane.timeframe] || {};
      pane.bars = chart.bars || [];
      pane.source = chart.source || '-';
      if (resetView) pane.visibleOffset = 0;
      const meta = $(`meta-${pane.id}`);
      if (meta) meta.textContent = `${pane.timeframe} · ${pane.bars.length} bars · ${pane.source}`;
    }
    updateClockHeader();
    $('chartMeta').textContent = `${timeframes.join(' / ')} 主图 · 上下文不裁剪 session/周末 · HTF 形成中 K 按已关闭 1m 实时更新`;
    updateTimeframeUi(activePane()?.timeframe || '30m');
    renderTimeline();
    deriveCurrentState();
    renderTradeSummary();
    drawAll();
  } finally {
    state.loading = false;
  }
}

function updateClockHeader() {
  if (!state.episode) return;
  const c = state.clock || {};
  $('cursorTime').textContent = `${c.beijing || c.beijing_plain || '-'} · decision cursor`;
  const symbol = state.episode?.symbol || $('symbol')?.value || 'OKX';
  if (c.session_profile === 'crypto_24x7_until_bracket_exit' || isContinuous24x7Symbol(symbol)) {
    $('clockInfo').textContent = c.beijing
      ? `${symbol} · 24/7 · 北京时间 · Episode ${state.episode?.start_time ? '从当前随机/指定时点开始' : ''} · TP/SL 命中自动结束`
      : `${symbol} · 24/7 · 北京时间显示`;
  } else {
    $('clockInfo').textContent = c.beijing
      ? `${symbol} · ${c.market_phase || ''} · 北京时间 · Episode ${c.episode_start_bjt || '-'} 开始 · 美股开盘 ${c.market_open_bjt || '-'} · 结束 ${c.episode_end_bjt || '-'}`
      : `${symbol} · 北京时间显示`;
  }
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
    if (Array.isArray(data.trade_events) && data.trade_events.length) {
      state.events.push(...data.trade_events);
    }
    state.activeLimitOrders = data.active_limit_orders || [];
    state.tradeSummary = data.trade_summary || state.tradeSummary;
    updateClockHeader();
    renderTimeline();
    deriveCurrentState();
    renderTradeSummary();
    drawAll();
    if (data.trade_events?.length) {
      const closed = data.trade_events.find(ev => ev.event_type === 'TRADE_CLOSED');
      const filled = data.trade_events.find(ev => ev.event_type === 'ORDER_FILLED');
      if (closed) {
        const p = closed.payload || {};
        const r = Number.isFinite(Number(p.r_multiple)) ? ` · ${Number(p.r_multiple).toFixed(2)}R` : '';
        const net = Number.isFinite(Number(p.net_return_pct)) ? ` · net ${Number(p.net_return_pct).toFixed(2)}%` : '';
        setStatus(`交易已结束：${p.exit_reason || 'CLOSE'} @ ${fmtPrice(closed.price)}${r}${net}`);
      } else if (filled) {
        setStatus(`挂单已成交：${filled.payload?.side || ''} @ ${fmtPrice(filled.price)}`);
      }
    }
    if (data.auto_finalized || state.episode?.status === 'closed') {
      stopPlay();
      const latest = [...(data.trade_events || [])].reverse().find(ev => ev.event_type === 'TRADE_CLOSED');
      if (latest) {
        const p = latest.payload || {};
        setStatus(`${state.episode?.symbol || '24/7'} Episode 已自动完成：${p.exit_reason || 'TRADE_CLOSED'} @ ${fmtPrice(latest.price)}。结果和汇总已实时保存。`);
      }
      $('episodeBadge').textContent = `${state.episode.id} · closed`;
      updateSequenceUi();
    } else if (data.at_data_end) {
      stopPlay();
      setStatus(isContinuous24x7Symbol(state.episode?.symbol) ? `已到本地 ${state.episode?.symbol || '24/7'} 数据末端。` : `已到当天 ${state.clock?.episode_end_bjt || '-'} 北京时间或本地数据边界。`);
    }
  } catch (err) {
    stopPlay();
    setStatus(err.message, true);
  } finally {
    state.stepInFlight = false;
  }
}
async function rewind(minutes) {
  if (!state.episode || state.episode.status !== 'active' || state.stepInFlight) return;
  stopPlay();
  state.stepInFlight = true;
  try {
    const timeframes = [...new Set(Object.values(state.panes).map(p => p.timeframe))];
    const data = await post(`/api/episodes/${state.episode.id}/rewind`, {minutes, timeframes});
    state.episode = data.episode;
    state.clock = data.clock || state.clock;
    state.events = data.events || [];
    state.activeLimitOrders = data.active_limit_orders || [];
    state.tradeSummary = data.trade_summary || state.tradeSummary;
    for (const pane of Object.values(state.panes)) {
      const chart = data.charts?.[pane.timeframe];
      if (chart) {
        pane.bars = chart.bars || [];
        pane.source = chart.source || pane.source;
        pane.visibleOffset = 0;
      }
    }
    updateClockHeader();
    renderTimeline();
    deriveCurrentState();
    renderTradeSummary();
    drawAll();
    const discarded = Number(data.discarded_event_count || 0);
    setStatus(discarded
      ? `已回退 ${data.rewound_minutes || 0}m；目标时间之后 ${discarded} 个旧事件已归档为 discarded branch。`
      : `已回退 ${data.rewound_minutes || 0}m。`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    state.stepInFlight = false;
  }
}

const PLAY_SPEEDS = {
  // User-requested remap: 慢≈old正常, 正常≈old很快.  Fast modes
  // also batch several causal minutes into one HTTP request; the backend's
  // vectorized lifecycle scanner still stops exactly on the first fill/SL/TP.
  slow: {minutes: 1, delayMs: 300},
  normal: {minutes: 1, delayMs: 45},
  fast: {minutes: 2, delayMs: 20},
  veryfast: {minutes: 5, delayMs: 5},
};
function playSpeedConfig() {
  return PLAY_SPEEDS[$('playSpeed')?.value] || PLAY_SPEEDS.normal;
}
function startPlay() {
  if (!state.episode || state.playTimer) return;
  $('playBtn').textContent = 'Ⅱ 暂停';
  const run = async () => {
    const cfg = playSpeedConfig();
    await step(cfg.minutes);
    if (state.playTimer && state.episode?.status === 'active') {
      state.playTimer = setTimeout(run, cfg.delayMs);
    }
  };
  state.playTimer = setTimeout(run, 10);
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
    renderTradeSummary();
    drawAll();
  } catch (err) {
    setStatus(err.message, true);
  }
}
async function trade(side) {
  if (!state.episode) return setStatus('请先建立 Episode', true);
  const sourceTf = activePane()?.timeframe || '1m';
  const orderType = side === 'CLOSE' ? 'market' : ($('orderType')?.value || 'limit');
  try {
    const body = {side, timeframe: sourceTf, order_type: orderType};
    if (side !== 'CLOSE') {
      const sl = readPriceInput('slPriceInput');
      const tp = readPriceInput('tpPriceInput');
      if (sl !== null) body.stop_loss = sl;
      if (tp !== null) body.take_profit = tp;
      if (orderType === 'limit') {
        body.limit_price = readPriceInput('entryPriceInput', {required: true});
        body.entry_context = selectedContextPayload({
          intent: 'manual_limit_entry',
          entry_price_source: Number.isFinite(state.selectedPrice) && Math.abs(Number(state.selectedPrice) - Number(body.limit_price)) < 1e-9 ? 'chart_selection' : 'manual_input',
          selected_from: state.selectedSnapField || 'FREE',
        });
      }
    }
    const data = await post(`/api/episodes/${state.episode.id}/trade`, body);
    const emitted = Array.isArray(data.events) ? data.events : (data.event ? [data.event] : []);
    if (emitted.length) state.events.push(...emitted);
    state.activeLimitOrders = data.active_limit_orders || state.activeLimitOrders;
    state.tradeSummary = data.trade_summary || state.tradeSummary;
    const slText = body.stop_loss ? ` · SL ${fmtPrice(body.stop_loss)}` : '';
    const tpText = body.take_profit ? ` · TP ${fmtPrice(body.take_profit)}` : '';
    if (data.status === 'pending') {
      setStatus(`${side} LIMIT 已挂 @ ${fmtPrice(body.limit_price)}${slText}${tpText} · 等待后续 1m 触价成交`);
    } else {
      setStatus(`${side} ${orderType.toUpperCase()} 成交 @ ${fmtPrice(data.fill_price)}${slText}${tpText}（判断来源 ${sourceTf}）`);
    }
    renderTimeline();
    deriveCurrentState();
    renderTradeSummary();
    drawAll();
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function cancelLatestLimitOrder() {
  if (!state.episode) return setStatus('请先建立 Episode', true);
  if (!state.activeLimitOrders.length) return setStatus('当前没有待成交 LIMIT 挂单。', true);
  try {
    const data = await post(`/api/episodes/${state.episode.id}/cancel-order`, {});
    if (data.event) state.events.push(data.event);
    state.activeLimitOrders = data.active_limit_orders || [];
    state.tradeSummary = data.trade_summary || state.tradeSummary;
    renderTimeline();
    deriveCurrentState();
    renderTradeSummary();
    drawAll();
    setStatus('已撤销最近一张待成交 LIMIT 挂单。');
  } catch (err) {
    setStatus(err.message, true);
  }
}
function requireSelectedPrice() {
  if (!Number.isFinite(state.selectedPrice)) {
    setStatus('先在主图上点击一个价格。', true);
    return false;
  }
  return true;
}

function eventDetail(ev) {
  const p = ev.payload || {};
  const src = ev.timeframe ? ` · src ${ev.timeframe}` : '';
  if (ev.event_type === 'LIQUIDITY') return `${p.kind || 'Liquidity'} · ${p.importance || 'normal'} · ${fmtPrice(ev.price)}${src}${p.anchor_time_bjt ? ` · anchor ${p.anchor_time_bjt} 北京时间` : (p.anchor_time ? ` · anchor ${p.anchor_time}` : '')}`;
  if (ev.event_type === 'BIAS') return `${p.bias || '-'}${src}`;
  if (ev.event_type === 'TARGET') return `Delivery → ${fmtPrice(ev.price)}${src}`;
  if (ev.event_type === 'MARKER') return `${p.label || 'Shared line'} · ${fmtPrice(ev.price)}${src}${p.anchor_time_bjt ? ` · anchor ${p.anchor_time_bjt} 北京时间` : (p.anchor_time ? ` · anchor ${p.anchor_time}` : '')}`;
  if (ev.event_type === 'ANNOTATION_DELETE') return `已删除 ${p.target_kind || p.target_label || p.target_event_type || 'annotation'} · event #${p.target_event_id || '-'}${Number.isFinite(Number(p.target_price)) ? ` · ${fmtPrice(p.target_price)}` : ''}${src}`;
  if (ev.event_type === 'ANNOTATION_LINE_VISIBILITY') return `${p.visible ? '恢复' : '隐藏'}图线 · event #${p.target_event_id || '-'}${Number.isFinite(Number(p.target_price)) ? ` · ${fmtPrice(p.target_price)}` : ''}${src}`;
  if (ev.event_type === 'LIMIT_ORDER') return `${p.side || '-'} LIMIT @ ${fmtPrice(ev.price)}${p.stop_loss ? ` · SL ${fmtPrice(p.stop_loss)}` : ''}${p.take_profit ? ` · TP ${fmtPrice(p.take_profit)}` : ''} · pending${src}${p.entry_context?.snap_field ? ` · 🧲 ${p.entry_context.snap_field}` : ''}`;
  if (ev.event_type === 'REWIND') return `回退 ${p.rewound_minutes || 0}m · ${p.to_cursor_bjt || p.to_cursor || ev.event_time_bjt || ev.event_time} 北京时间 · archived ${(p.discarded_event_ids || []).length} events`;
  if (ev.event_type === 'LIMIT_CANCEL') return `${p.side || '-'} LIMIT @ ${fmtPrice(ev.price)} · cancelled${src}`;
  if (ev.event_type === 'LIMIT_EXPIRED') return `${p.side || '-'} LIMIT @ ${fmtPrice(ev.price)} · Episode结束时未成交${src}`;
  if (ev.event_type === 'ORDER_FILLED') return `${p.side || '-'} ${String(p.order_type || '').toUpperCase()} filled @ ${fmtPrice(ev.price)}${src}`;
  if (ev.event_type === 'TRADE_OPEN') return `${p.side || '-'} trade opened @ ${fmtPrice(ev.price)}${p.initial_stop_loss ? ` · SL ${fmtPrice(p.initial_stop_loss)}` : ''}${p.initial_take_profit ? ` · TP ${fmtPrice(p.initial_take_profit)}` : ''}${src}`;
  if (ev.event_type === 'TAKE_PROFIT_HIT') return `TP HIT @ ${fmtPrice(ev.price)} · trade ${p.trade_id || '-'}${src}`;
  if (ev.event_type === 'STOP_LOSS_HIT') return `SL HIT @ ${fmtPrice(ev.price)} · trade ${p.trade_id || '-'}${src}`;
  if (ev.event_type === 'TRADE_EXIT_AMBIGUOUS') return `同一根 1m 同时触及 SL/TP · 保守按 SL ${fmtPrice(ev.price)} 处理${src}`;
  if (ev.event_type === 'MANUAL_EXIT') return `手动平仓 @ ${fmtPrice(ev.price)}${src}`;
  if (ev.event_type === 'TRADE_CLOSED') {
    const r = Number.isFinite(Number(p.r_multiple)) ? ` · ${Number(p.r_multiple).toFixed(2)}R` : '';
    const net = Number.isFinite(Number(p.net_return_pct)) ? ` · net ${Number(p.net_return_pct).toFixed(2)}%` : '';
    const path = Number.isFinite(Number(p.mfe_pct)) ? ` · MFE ${Number(p.mfe_pct).toFixed(2)}% / MAE ${Number(p.mae_pct || 0).toFixed(2)}%` : '';
    return `${p.exit_reason || 'CLOSED'} · ${p.side || '-'} ${fmtPrice(p.entry_price)} → ${fmtPrice(p.exit_price)}${r}${net}${path}`;
  }
  if (ev.event_type === 'EPISODE_SUMMARY') {
    const legacyClosedPending = state.episode?.status === 'closed' && !('unfilled_orders' in p) ? Number(p.pending_orders || 0) : 0;
    const pending = legacyClosedPending ? 0 : Number(p.pending_orders || 0);
    const unfilled = Number(p.unfilled_orders || 0) + legacyClosedPending;
    return `Closed ${p.closed_trades || 0} · W ${p.wins || 0} / L ${p.losses || 0} · active ${p.active_trades || 0} · pending ${pending}${unfilled ? ` · unfilled ${unfilled}` : ''}`;
  }
  if (['LONG','SHORT','CLOSE'].includes(ev.event_type) && Number.isFinite(Number(ev.price))) {
    const kind = p.order_type === 'limit' ? `limit fill${p.limit_price ? ` (LMT ${fmtPrice(p.limit_price)})` : ''}` : 'market';
    return `${ev.event_type} @ ${fmtPrice(ev.price)} · ${kind}${src}`;
  }
  if (['SL','TP','MOVE_SL'].includes(ev.event_type)) return `${ev.event_type} ${fmtPrice(ev.price)}${src}`;
  if (ev.event_type === 'NOTE') return `${p.text || ''}${src}`;
  if (ev.event_type === 'CLOSE' && p.reason === 'episode_end') return 'Episode ended';
  return `${Object.keys(p).length ? JSON.stringify(p) : ''}${src}`;
}
function canDeleteAnnotation(ev) {
  return ['LIQUIDITY', 'TARGET', 'MARKER'].includes(ev?.event_type);
}
function annotationLineVisibilityMap() {
  const visibility = new Map();
  for (const ev of state.events) {
    if (ev?.event_type !== 'ANNOTATION_LINE_VISIBILITY') continue;
    const id = Number(ev.payload?.target_event_id);
    if (!Number.isFinite(id) || id <= 0) continue;
    visibility.set(id, ev.payload?.visible !== false);
  }
  return visibility;
}
function isAnnotationLineVisible(eventId, visibility = null) {
  const map = visibility || annotationLineVisibilityMap();
  return map.has(Number(eventId)) ? map.get(Number(eventId)) : true;
}
async function setAnnotationLineVisible(eventId, visible) {
  if (!state.episode || state.episode.status !== 'active') return;
  const target = state.events.find(ev => Number(ev.id) === Number(eventId));
  if (!target || !canDeleteAnnotation(target)) return;
  const label = target.event_type === 'LIQUIDITY' ? (target.payload?.kind || 'Liquidity') : (target.payload?.label || target.event_type);
  try {
    const data = await post(`/api/episodes/${state.episode.id}/annotation-line`, {event_id: Number(eventId), visible: Boolean(visible)});
    state.events = data.events || state.events;
    state.activeLimitOrders = data.active_limit_orders || state.activeLimitOrders;
    renderTimeline();
    deriveCurrentState();
    drawAll();
    setStatus(`${visible ? '已恢复' : '已从图上移除'} ${label} @ ${fmtPrice(target.price)}；Decision Timeline 和训练记录仍保留。`);
  } catch (err) {
    setStatus(err.message, true);
  }
}
async function deleteAnnotation(eventId) {
  if (!state.episode || state.episode.status !== 'active') return;
  const target = state.events.find(ev => Number(ev.id) === Number(eventId));
  if (!target || !canDeleteAnnotation(target)) return;
  const label = target.event_type === 'LIQUIDITY' ? (target.payload?.kind || 'Liquidity') : target.event_type;
  try {
    const data = await post(`/api/episodes/${state.episode.id}/delete-annotation`, {event_id: Number(eventId)});
    state.events = data.events || [];
    state.activeLimitOrders = data.active_limit_orders || state.activeLimitOrders;
    renderTimeline();
    deriveCurrentState();
    drawAll();
    setStatus(`已删除记录 ${label} @ ${fmtPrice(target.price)}；该 Decision 已从当前分支归档到 discarded_events。`);
  } catch (err) {
    setStatus(err.message, true);
  }
}
function renderTimeline() {
  const root = $('timeline');
  if (!state.events.length) {
    root.className = 'timeline empty';
    root.textContent = '还没有事件。';
    return;
  }
  root.className = 'timeline';
  const lineVisibility = annotationLineVisibilityMap();
  const timelineEvents = state.events.filter(ev => ev.event_type !== 'ANNOTATION_LINE_VISIBILITY');
  root.innerHTML = timelineEvents.slice().reverse().map(ev => {
    const lineVisible = canDeleteAnnotation(ev) ? isAnnotationLineVisible(ev.id, lineVisibility) : true;
    return `
    <div class="timeline-item ${escapeHtml(ev.event_type)}${!lineVisible ? ' line-hidden' : ''}">
      <div class="timeline-item-head">
        <strong>${escapeHtml(ev.event_type)}</strong>
        ${canDeleteAnnotation(ev) ? `<span class="annotation-actions">
          <button type="button" class="annotation-line-btn" data-event-id="${Number(ev.id)}" data-visible="${lineVisible ? '1' : '0'}" title="${lineVisible ? '只移除图上的线，保留Decision记录' : '恢复图上的线'}">${lineVisible ? '删线' : '恢复线'}</button>
          <button type="button" class="annotation-delete-btn" data-event-id="${Number(ev.id)}" title="删除Decision记录并归档">删记录</button>
        </span>` : ''}
      </div>
      <small>${escapeHtml(ev.event_time_bjt || ev.event_time)} 北京时间${ev.timeframe ? ` · ${escapeHtml(ev.timeframe)}` : ''}${!lineVisible ? ' · 图线已隐藏' : ''}</small>
      <div class="event-detail">${escapeHtml(eventDetail(ev))}</div>
    </div>`;
  }).join('');
  for (const button of root.querySelectorAll('.annotation-line-btn')) {
    button.addEventListener('click', () => setAnnotationLineVisible(Number(button.dataset.eventId), button.dataset.visible !== '1'));
  }
  for (const button of root.querySelectorAll('.annotation-delete-btn')) {
    button.addEventListener('click', () => deleteAnnotation(Number(button.dataset.eventId)));
  }
}
function renderTradeSummary() {
  const summary = state.tradeSummary || {};
  if ($('outcomeClosed')) $('outcomeClosed').textContent = Number(summary.closed_trades || 0);
  if ($('outcomeWins')) $('outcomeWins').textContent = Number(summary.wins || 0);
  if ($('outcomeLosses')) $('outcomeLosses').textContent = Number(summary.losses || 0);
  if ($('outcomeAmbiguous')) $('outcomeAmbiguous').textContent = Number(summary.ambiguous || 0);
  const latest = summary.latest_closed_trade;
  if (!$('latestOutcome')) return;
  if (!latest) {
    $('latestOutcome').textContent = summary.active_trades ? `当前有 ${summary.active_trades} 笔持仓尚未结束。` : '还没有已结束交易。';
    return;
  }
  const p = latest.payload || {};
  const r = Number.isFinite(Number(p.r_multiple)) ? `${Number(p.r_multiple).toFixed(2)}R` : 'R -';
  const net = Number.isFinite(Number(p.net_return_pct)) ? `${Number(p.net_return_pct).toFixed(2)}% net` : 'net -';
  const mfe = Number.isFinite(Number(p.mfe_pct)) ? `MFE ${Number(p.mfe_pct).toFixed(2)}%` : '';
  const mae = Number.isFinite(Number(p.mae_pct)) ? `MAE ${Number(p.mae_pct).toFixed(2)}%` : '';
  $('latestOutcome').textContent = `${p.exit_reason || 'CLOSED'} · ${p.side || '-'} ${fmtPrice(p.entry_price)} → ${fmtPrice(p.exit_price)} · ${r} · ${net}${mfe ? ` · ${mfe}` : ''}${mae ? ` / ${mae}` : ''}`;
}

function deriveCurrentState() {
  let bias = '-', target = '-', sl = '-', tp = '-';
  const activeTrades = new Map();
  let legacyPosition = 'Flat';
  for (const ev of state.events) {
    if (ev.event_type === 'BIAS') bias = `${ev.payload?.bias || '-'}${ev.timeframe ? ` (${ev.timeframe})` : ''}`;
    if (ev.event_type === 'TARGET') target = `${fmtPrice(ev.price)}${ev.timeframe ? ` (${ev.timeframe})` : ''}`;
    if (ev.event_type === 'TRADE_OPEN') activeTrades.set(String(ev.payload?.trade_id || ev.id), ev);
    if (ev.event_type === 'TRADE_CLOSED') activeTrades.delete(String(ev.payload?.trade_id || ''));
    if (ev.event_type === 'LONG' || ev.event_type === 'SHORT') legacyPosition = `${ev.event_type} @ ${fmtPrice(ev.price)}`;
    if (ev.event_type === 'CLOSE' && ev.payload?.reason !== 'episode_end') legacyPosition = 'Flat';
    if (ev.event_type === 'SL' || ev.event_type === 'MOVE_SL') sl = fmtPrice(ev.price);
    if (ev.event_type === 'TP') tp = fmtPrice(ev.price);
  }
  const active = [...activeTrades.values()];
  const position = active.length
    ? active.map(ev => `${ev.payload?.side || '?'} @ ${fmtPrice(ev.payload?.entry_price ?? ev.price)}`).join(' | ')
    : ((state.tradeSummary?.active_trades || 0) > 0 ? legacyPosition : 'Flat');
  $('currentBias').textContent = bias;
  $('currentTarget').textContent = target;
  $('currentPosition').textContent = position;
  const pending = state.activeLimitOrders.map(order => {
    const p = order.payload || {};
    return `${p.side || '?'} @ ${fmtPrice(order.price)}${p.stop_loss ? ` / SL ${fmtPrice(p.stop_loss)}` : ''}${p.take_profit ? ` / TP ${fmtPrice(p.take_profit)}` : ''}`;
  }).join(' | ');
  $('currentPending').textContent = pending || '-';
  $('currentSl').textContent = active.length || position !== 'Flat' ? sl : '-';
  $('currentTp').textContent = active.length || position !== 'Flat' ? tp : '-';
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
  const activeLimitIds = new Set(state.activeLimitOrders.map(order => String(order.payload?.order_id || '')));
  for (const ev of state.events) {
    if (ev.event_type === 'LIMIT_ORDER' && !activeLimitIds.has(String(ev.payload?.order_id || ''))) continue;
    const price = Number(ev.price);
    if (!Number.isFinite(price)) continue;
    if (price >= min - localSpan * .45 && price <= max + localSpan * .45) {
      min = Math.min(min, price);
      max = Math.max(max, price);
    }
  }
  for (const order of state.activeLimitOrders) {
    const p = order.payload || {};
    for (const value of [p.stop_loss, p.take_profit]) {
      const price = Number(value);
      if (!Number.isFinite(price)) continue;
      if (price >= min - localSpan * .45 && price <= max + localSpan * .45) {
        min = Math.min(min, price);
        max = Math.max(max, price);
      }
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
    const label = (bars[idx]?.time_bjt || bars[idx]?.time || '').slice(5, 16);
    ctx.fillText(label, Math.max(plot.left, x - 24), plot.bottom + 14);
  }
}
function eventStyle(type, payload) {
  if (type === 'LIQUIDITY') return payload?.kind === 'SSL' ? {color:'#5fd6e8', dash:[6,4]} : {color:'#f6b95d', dash:[6,4]};
  if (type === 'TARGET') return {color:'#b08cff', dash:[8,4]};
  if (type === 'MARKER') return {color:'#9aa9ba', dash:[4,3]};
  if (type === 'SL' || type === 'MOVE_SL') return {color:'#ff6b76', dash:[3,3]};
  if (type === 'TP') return {color:'#42d392', dash:[3,3]};
  if (type === 'LIMIT_ORDER') return {color:'#5fd6e8', dash:[10,5]};
  return null;
}
function drawEvents(pane, plot, range, bars) {
  const ctx = pane.ctx;
  const activeLimitIds = new Set(state.activeLimitOrders.map(order => String(order.payload?.order_id || '')));
  const lineVisibility = annotationLineVisibilityMap();
  for (const ev of state.events) {
    if (canDeleteAnnotation(ev) && !isAnnotationLineVisible(ev.id, lineVisibility)) continue;
    if (ev.event_type === 'LIMIT_ORDER' && !activeLimitIds.has(String(ev.payload?.order_id || ''))) continue;
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
        if (ev.event_type === 'LIMIT_ORDER') label = `${ev.payload?.side || ''} LMT`;
        const source = ev.timeframe ? `[${ev.timeframe}] ` : '';
        ctx.fillStyle = style.color;
        ctx.font = '9px system-ui';
        ctx.fillText(`${source}${label} ${fmtPrice(price)}`, plot.left + 4, Math.max(plot.top + 10, y - 4));
      }
    }
    if (ev.event_type === 'LIMIT_ORDER') {
      const startX = xForTime(ev.event_time, bars, plot);
      const p = ev.payload || {};
      for (const [kind, value, color] of [['SL', p.stop_loss, '#ff6b76'], ['TP', p.take_profit, '#42d392']]) {
        const bracketPrice = Number(value);
        if (!Number.isFinite(bracketPrice)) continue;
        const y = yForPrice(bracketPrice, range, plot);
        if (y < plot.top - 10 || y > plot.bottom + 10) continue;
        ctx.strokeStyle = color; ctx.globalAlpha = .62; ctx.setLineDash([2,5]);
        ctx.beginPath(); ctx.moveTo(startX, y); ctx.lineTo(plot.right, y); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
        ctx.fillStyle = color; ctx.font = '9px system-ui';
        ctx.fillText(`PENDING ${kind} ${fmtPrice(bracketPrice)}`, plot.left + 4, Math.max(plot.top + 10, y - 4));
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
    if (ev.event_type === 'TRADE_CLOSED') {
      const y = yForPrice(price, range, plot);
      if (y < plot.top - 10 || y > plot.bottom + 10) continue;
      const x = xForTime(ev.event_time, bars, plot);
      const reason = ev.payload?.exit_reason || 'EXIT';
      const color = reason === 'TAKE_PROFIT' ? '#42d392' : '#ff6b76';
      ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = color; ctx.font = '9px system-ui';
      const r = Number.isFinite(Number(ev.payload?.r_multiple)) ? ` ${Number(ev.payload.r_multiple).toFixed(2)}R` : '';
      ctx.fillText(`${reason === 'TAKE_PROFIT' ? 'TP' : reason === 'STOP_LOSS' ? 'SL' : 'EXIT'}${r}`, Math.min(plot.right - 52, x + 6), y - 6);
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
  const forming = b.is_partial ? ` · 形成中 · 已看到 ${b.observed_through_bjt || b.observed_through || '-'} 北京时间` : '';
  pane.crosshair.textContent = `${b.time_bjt || b.time} 北京时间  O ${fmtPrice(b.open)} H ${fmtPrice(b.high)} L ${fmtPrice(b.low)} C ${fmtPrice(b.close)}${forming}`;
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
  const latest = bars[bars.length - 1];
  if ($('formingStatus')) $('formingStatus').textContent = latest?.is_partial ? `● 形成中 · 已看到 ${(latest.observed_through_bjt || latest.observed_through || '').slice(11,16) || '-'} 北京时间` : '✓ 最新 K 已收盘';
  const range = priceRange(bars);
  drawGrid(pane, plot, range, bars);
  const stepX = plot.width / Math.max(1, bars.length);
  const bodyW = Math.max(1, Math.min(8, stepX * .66));
  bars.forEach((b, i) => {
    const x = plot.left + (i + .5) * stepX;
    const o = yForPrice(Number(b.open), range, plot), c = yForPrice(Number(b.close), range, plot);
    const hi = yForPrice(Number(b.high), range, plot), lo = yForPrice(Number(b.low), range, plot);
    const up = Number(b.close) >= Number(b.open);
    const color = up ? '#42d392' : '#ff6b76';
    ctx.strokeStyle = color;
    ctx.fillStyle = color; ctx.lineWidth = b.is_partial ? 1.35 : 1;
    ctx.globalAlpha = b.is_partial ? .82 : 1;
    ctx.beginPath(); ctx.moveTo(x, hi); ctx.lineTo(x, lo); ctx.stroke();
    const top = Math.min(o, c), bh = Math.max(1, Math.abs(c - o));
    if (b.is_partial) {
      ctx.globalAlpha = .42;
      ctx.fillRect(x - bodyW / 2, top, bodyW, bh);
      ctx.globalAlpha = 1;
      ctx.setLineDash([2,2]);
      ctx.strokeStyle = '#d8e7f6';
      ctx.strokeRect(x - bodyW / 2, top, bodyW, bh);
      ctx.setLineDash([]);
    } else {
      ctx.fillRect(x - bodyW / 2, top, bodyW, bh);
    }
    ctx.globalAlpha = 1;
  });
  const last = bars[bars.length - 1];
  if (last?.is_partial) {
    ctx.fillStyle = '#90bdf8';
    ctx.font = '9px system-ui';
    ctx.fillText(`形成中 · ${pane.timeframe}`, Math.max(plot.left + 6, plot.right - 78), plot.top + 12);
  }
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
    state.selectedAnchorDisplayTime = bar?.time_bjt || bar?.time || null;
    state.selectedAnchorIsPartial = Boolean(bar?.is_partial);
    state.selectedAnchorObservedThrough = bar?.observed_through || null;
    state.selectedSourceTimeframe = pane.timeframe;
    state.selectedPaneId = pane.id;
    const snap = state.selectedSnapField ? ` · 🧲 ${state.selectedSnapField}` : ' · 自由价格';
    const partialTag = state.selectedAnchorIsPartial ? ` · 形成中@${String((bar?.observed_through_bjt || state.selectedAnchorObservedThrough || '')).slice(11,16)} 北京时间` : '';
    $('selectedPrice').textContent = `已选 ${fmtPrice(state.selectedPrice)}${snap} · ${pane.timeframe} · anchor ${state.selectedAnchorDisplayTime || state.selectedAnchorTime || '-'} 北京时间${partialTag}`;
    updateTicketSelectedPrice();
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
  loadTimeframeSlotPreferences();
  const savedTf = state.timeframeSlots[state.activeTimeframeSlot] || PANE_DEFAULTS.main.timeframe;
  PANE_DEFAULTS.main.timeframe = savedTf;
  for (const [id, cfg] of Object.entries(PANE_DEFAULTS)) {
    const canvas = $(`canvas-${id}`);
    const wrap = $(`wrap-${id}`);
    const card = document.querySelector(`.chart-card[data-pane="${id}"]`);
    const pane = {
      id, role: roleForTimeframe(cfg.timeframe), timeframe: cfg.timeframe,
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
  renderTimeframeSlots();
}


async function loadHealth({preserveSymbol = false} = {}) {
  try {
    const current = preserveSymbol ? String($('symbol')?.value || '').trim() : '';
    const query = current ? `?${new URLSearchParams({symbol: current})}` : '';
    const data = await request(`/api/health${query}`);
    const symbols = Array.isArray(data.symbols) ? data.symbols : [];
    const select = $('symbol');
    if (!preserveSymbol || !select.options.length || (symbols.length && !symbols.includes(select.value))) {
      const desired = current && symbols.includes(current) ? current : (data.symbol || symbols[0] || '');
      select.innerHTML = symbols.map(s => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`).join('');
      if (desired) select.value = desired;
    }
    const selected = String(select.value || data.symbol || '').trim();
    let health = data;
    if (selected && selected !== data.symbol) {
      health = await request(`/api/health?${new URLSearchParams({symbol: selected})}`);
    }
    const c = health.coverage || {};
    const first = c.first_episode_date || '';
    const last = c.last_episode_date || '';
    for (const id of ['randomStart', 'randomEnd', 'specificStart', 'sequentialStartDate']) {
      const input = $(id);
      if (!input) continue;
      if (first) input.min = first;
      if (last) input.max = last;
    }
    if (first && last) {
      // Use a bounded recent window by default so 24/7 symbols do not scan years of 1m
      // rows just to choose one random blind-replay day.
      const lastDate = new Date(`${last}T00:00:00Z`);
      const recent = new Date(lastDate.getTime() - 90 * 24 * 60 * 60 * 1000);
      const recentText = recent.toISOString().slice(0, 10);
      $('randomStart').value = recentText > first ? recentText : first;
      $('randomEnd').value = last;
      $('specificStart').value = last;
      if ($('sequentialStartDate')) $('sequentialStartDate').value = first;
      if ($('specificStartTime')) {
        const endBjt = String(c.available_end_bjt || '').replace(' ', 'T').slice(0,16);
        $('specificStartTime').value = endBjt || `${last}T12:00`;
      }
    }
    updateSessionProfileUi();
    updateSequenceUi();
    const profileText = c.session_profile === 'crypto_24x7_until_bracket_exit' ? '24/7 · 周末可用 · TP/SL 自动结束' : '工作日 07:30 ET Episode';
    $('sourceInfo').textContent = first && last
      ? `${c.symbol || selected} · ${profileText} · OKX 本地 1m ${Number(c.rows_1m || 0).toLocaleString()} bars · 北京时间 ${c.available_start_bjt || first} → ${c.available_end_bjt || last} · 图表上下文保留全部可用 bars`
      : `${selected || '所选标的'} 没有可用 Episode。`;
    setStatus(`${c.symbol || selected || 'OKX'} 本地数据已就绪。`);
  } catch (err) {
    $('sourceInfo').textContent = `OKX 数据不可用：${err.message}`;
    setStatus(err.message, true);
  }
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
    state.events = data.events || state.events;
    state.tradeSummary = data.summary || state.tradeSummary;
    $('episodeBadge').textContent = `${state.episode.id} · closed`;
    updateSequenceUi();
    renderTimeline();
    deriveCurrentState();
    renderTradeSummary();
    drawAll();
    const open = Number(state.tradeSummary?.active_trades || 0);
    const pending = Number(state.tradeSummary?.pending_orders || 0);
    const unfilled = Number(state.tradeSummary?.unfilled_orders || 0);
    setStatus(open || pending ? `Episode 已最终汇总；仍有 ${open} 笔未结束持仓、${pending} 张活动挂单。` : (unfilled ? `Episode 已 closed；${unfilled} 张未成交挂单已标记为 UNFILLED/EXPIRED。` : 'Episode 已最终汇总并标记为 closed。所有操作此前已实时保存。'));
  } catch (err) {
    setStatus(err.message, true);
  }
}

for (const radio of document.querySelectorAll('input[name="startMode"]')) radio.addEventListener('change', updateStartMode);
$('symbol').addEventListener('change', async () => {
  if (state.episode?.symbol !== $('symbol').value) { state.startMode = null; state.sequenceSymbol = null; } stopPlay(); updateSessionProfileUi(); await loadHealth({preserveSymbol: true}); });
for (const id of ['sequentialStartDate', 'sequentialStartTime']) {
  $(id)?.addEventListener('change', () => {
    if (activeStartMode() === 'sequential') { state.startMode = null; state.sequenceSymbol = null; updateSequenceUi(); }
  });
}
$('newEpisodeBtn').addEventListener('click', createEpisode);
for (const select of document.querySelectorAll('.tf-slot-select')) {
  select.addEventListener('change', async e => {
    e.stopPropagation();
    const slotIndex = Number(select.dataset.slot);
    await configureTimeframeSlot(slotIndex, select.value);
  });
  select.addEventListener('click', e => {
    const slotIndex = Number(select.dataset.slot);
    state.activeTimeframeSlot = slotIndex;
    persistTimeframeSlotPreferences();
    renderTimeframeSlots();
  });
}
for (const wrapper of document.querySelectorAll('.tf-slot')) {
  wrapper.addEventListener('click', async e => {
    if (e.target.closest('select')) return;
    const slotIndex = Number(wrapper.dataset.slot);
    const tf = state.timeframeSlots[slotIndex] || DEFAULT_TIMEFRAME_SLOTS[slotIndex];
    await changeMainTimeframe(tf, {slotIndex});
  });
}
$('resetTimeframeSlotsBtn')?.addEventListener('click', resetTimeframeSlots);
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
$('cancelOrderBtn').addEventListener('click', cancelLatestLimitOrder);
$('closeTradeBtn').addEventListener('click', () => trade('CLOSE'));
$('rewind15Btn').addEventListener('click', () => rewind(15));
$('rewind5Btn').addEventListener('click', () => rewind(5));
$('rewind1Btn').addEventListener('click', () => rewind(1));
$('step1Btn').addEventListener('click', () => step(1));
$('step2Btn').addEventListener('click', () => step(2));
$('step5Btn').addEventListener('click', () => step(5));
$('step15Btn').addEventListener('click', () => step(15));
$('step30Btn').addEventListener('click', () => step(30));
$('step60Btn').addEventListener('click', () => step(60));
$('fillEntryBtn').addEventListener('click', () => { if (requireSelectedPrice()) setTicketPrice('entryPriceInput', state.selectedPrice); });
$('fillSlBtn').addEventListener('click', () => { if (requireSelectedPrice()) setTicketPrice('slPriceInput', state.selectedPrice); });
$('fillTpBtn').addEventListener('click', () => { if (requireSelectedPrice()) setTicketPrice('tpPriceInput', state.selectedPrice); });
$('clearTicketBtn').addEventListener('click', () => {
  for (const id of ['entryPriceInput','slPriceInput','tpPriceInput']) $(id).value = '';
  setStatus('下单价格已清空。');
});
$('playBtn').addEventListener('click', () => state.playTimer ? stopPlay() : startPlay());
$('saveNoteBtn').addEventListener('click', async () => {
  const text = $('noteText').value.trim();
  if (!text) return;
  await addEvent('NOTE', null, {text});
  $('noteText').value = '';
});
$('exportBtn').addEventListener('click', exportEpisode);
$('endEpisodeBtn').addEventListener('click', endEpisode);


function updateOrderButtons() {
  const mode = $('orderType')?.value || 'limit';
  $('longBtn').textContent = mode === 'limit' ? '挂多 LONG' : '市价 LONG';
  $('shortBtn').textContent = mode === 'limit' ? '挂空 SHORT' : '市价 SHORT';
  if ($('entryPriceInput')) {
    $('entryPriceInput').disabled = mode !== 'limit';
    $('entryPriceInput').placeholder = mode === 'limit' ? '例如 134.72' : 'MARKET 使用 cursor 的 1m open';
  }
  if ($('fillEntryBtn')) $('fillEntryBtn').disabled = mode !== 'limit';
}
$('orderType').addEventListener('change', updateOrderButtons);

$('magnetToggle').addEventListener('change', () => {
  setStatus($('magnetToggle').checked ? '磁铁已开启：下一次点击自动吸附该 K 的 O/H/L/C。' : '磁铁已关闭：下一次点击使用自由价格。');
});

updateStartMode();
updateOrderButtons();
initPanes();
renderTradeSummary();
loadHealth();
