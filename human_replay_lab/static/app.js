'use strict';

const $ = id => document.getElementById(id);
const LIMIT_FEE = 0.0002;
const MARKET_FEE = 0.0005;
const MARKET_SLIPPAGE = 0.0002;
const ACCOUNT_KEY = 'coinbacktest.replay.account.v2';
const EPISODE_KEY = 'coinbacktest.replay.lastEpisode.v2';
const PREFS_KEY = 'coinbacktest.replay.ui.v2';
const TIMEFRAME_MS = {'1m':60000,'2m':120000,'5m':300000,'15m':900000,'30m':1800000,'1H':3600000,'4H':14400000,'1D':86400000};
const PRESET_COLORS = ['#f4f7fb','#f05263','#ff9800','#f5dd42','#4caf5a','#0f9f8f','#16b8c8','#3478f6','#673ab7','#9c27b0','#e91e63','#f3b8c7','#ffd59a','#fff0a6','#a8d5ac','#86d3c9','#90d7df','#9ec5f5','#b7a4df','#d09bdd','#ef9abb','#8f98a8','#ff7b83','#ffc15a','#80c785','#3db6a7','#39aabd','#5798f2','#8d64cb','#bd61c4','#e75a91','#243047'];
const MAGNET_MODES = ['weak','strong','off'];
const MAGNET_WEAK_THRESHOLD_PX = 12;

const state = {
  episode: null,
  clock: null,
  executionPrice: null,
  bars: [],
  timeframe: '30m',
  events: [],
  activeLimitOrders: [],
  activeTrades: [],
  tradeSummary: null,
  health: null,
  visibleCount: 220,
  visibleOffset: 0,
  hoverIndex: null,
  hoverPoint: null,
  pointer: null,
  panning: false,
  tool: 'cursor',
  drawingDraft: null,
  drawings: [],
  selectedDrawingId: null,
  hoverDrawingId: null,
  drawingInteraction: null,
  drawingsLocked: false,
  magnetMode: 'weak',
  autoScale: true,
  manualPriceRange: null,
  historyLoading: false,
  historyExhausted: false,
  pendingCalloutPoint: null,
  pendingPositionSide: null,
  pendingPositionPoints: null,
  pendingPositionDrawingId: null,
  loading: false,
  playTimer: null,
  playGeneration: 0,
  pauseOnEvent: true,
  chartTimezone: 'beijing',
  account: null,
  accountWasStored: false,
  ticketAccountOverridden: false,
};

const canvas = $('chartCanvas');
const ctx = canvas.getContext('2d');

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
}

function money(value) {
  const n = Number(value || 0);
  return `${n < 0 ? '-' : ''}$${Math.abs(n).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
}

function price(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return n >= 1000 ? n.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}) : n.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 4});
}

function compactNumber(value) {
  const n = Number(value || 0);
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(2);
}

function wallTime(value) {
  const text = String(value || '').replace(' ', 'T');
  const parsed = Date.parse(`${text}Z`);
  return Number.isFinite(parsed) ? parsed : 0;
}

async function request(url, options = {}) {
  const response = await fetch(url, {cache: 'no-store', ...options});
  let data = {};
  try { data = await response.json(); } catch (_) { /* ignored */ }
  if (!response.ok || data.ok === false) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

function post(url, body) {
  return request(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
}

function setStatus(text, kind = 'ok') {
  $('statusText').textContent = text;
  const host = $('statusText').parentElement;
  host.classList.toggle('error', kind === 'error');
  host.classList.toggle('busy', kind === 'busy');
}

function loadPreferences() {
  try {
    const prefs = JSON.parse(localStorage.getItem(PREFS_KEY) || '{}');
    state.timeframe = ['1m','2m','5m','15m','30m','1H','4H','1D'].includes(prefs.timeframe) ? prefs.timeframe : '30m';
    state.magnetMode = MAGNET_MODES.includes(prefs.magnetMode) ? prefs.magnetMode : (prefs.magnet === false ? 'off' : 'weak');
    state.drawingsLocked = Boolean(prefs.drawingsLocked);
    state.visibleCount = Number.isFinite(Number(prefs.visibleCount)) ? Math.max(30, Math.min(5000, Number(prefs.visibleCount))) : 220;
    state.autoScale = prefs.autoScale !== false;
    state.pauseOnEvent = prefs.pauseOnEvent !== false;
    state.chartTimezone = prefs.chartTimezone === 'new_york' ? 'new_york' : 'beijing';
    if (['1000','500','250','160','80','40','16'].includes(String(prefs.playSpeed))) $('playSpeed').value = String(prefs.playSpeed);
  } catch (_) { /* use defaults */ }
  updateMagnetButton();
  $('lockToggle').classList.toggle('active', state.drawingsLocked);
  $('lockToggle').setAttribute('aria-pressed', String(state.drawingsLocked));
  updateAutoScaleButton();
  updateTimeframeButtons();
  $('pauseOnEvent').checked = state.pauseOnEvent;
  $('chartTimezone').value = state.chartTimezone;
}

function savePreferences() {
  localStorage.setItem(PREFS_KEY, JSON.stringify({timeframe: state.timeframe, magnetMode: state.magnetMode, magnet: state.magnetMode !== 'off', drawingsLocked: state.drawingsLocked, visibleCount: state.visibleCount, autoScale: state.autoScale, pauseOnEvent: state.pauseOnEvent, chartTimezone: state.chartTimezone, playSpeed: $('playSpeed').value}));
}

function updateMagnetButton() {
  const button=$('magnetToggle'),mode=MAGNET_MODES.includes(state.magnetMode)?state.magnetMode:'weak';
  const details={weak:{badge:'弱',label:'弱磁：靠近当前 K 线 OHLC 时吸附'},strong:{badge:'强',label:'强磁：始终吸附当前 K 线最近的 OHLC'},off:{badge:'关',label:'磁铁关闭：自由价格'}}[mode];
  state.magnetMode=mode;button.dataset.magnetMode=mode;button.classList.toggle('active',mode!=='off');button.setAttribute('aria-pressed',String(mode!=='off'));button.setAttribute('aria-label',details.label);button.title=`${details.label}（点击切换）`;$('magnetModeBadge').textContent=details.badge;
}

function defaultAccount() {
  return {initialBalance: 10000, balance: 10000, realizedPnl: 0, totalFees: 0, totalSlippage: 0, trades: [], processedTradeIds: []};
}

function loadAccount() {
  try {
    const raw = localStorage.getItem(ACCOUNT_KEY);
    state.accountWasStored = Boolean(raw);
    state.account = raw ? {...defaultAccount(), ...JSON.parse(raw)} : defaultAccount();
    if (!Array.isArray(state.account.trades)) state.account.trades = [];
    if (!Array.isArray(state.account.processedTradeIds)) state.account.processedTradeIds = [];
  } catch (_) { state.account = defaultAccount(); }
  $('accountInput').value = Number(state.account.balance || 10000).toFixed(2);
  renderAccount();
}

function saveAccount() {
  localStorage.setItem(ACCOUNT_KEY, JSON.stringify(state.account));
}

function syncAccountInputs(previousBalance = null, {forceTicket = false} = {}) {
  const balance = Number(state.account?.balance);
  if (!Number.isFinite(balance) || balance <= 0) return;
  $('accountInput').value = balance.toFixed(2);
  const ticketValue = Number($('positionAccount').value);
  const followedPreviousBalance = Number.isFinite(Number(previousBalance))
    && Number.isFinite(ticketValue)
    && Math.abs(ticketValue - Number(previousBalance)) < 0.005;
  const mayFollowEquity = !state.pendingPositionDrawingId
    && (!state.ticketAccountOverridden || followedPreviousBalance);
  if (forceTicket || mayFollowEquity) {
    $('positionAccount').value = balance.toFixed(2);
    state.ticketAccountOverridden = false;
    updateRiskPreview();
  }
}

function closedTradeEvents() {
  return state.events.filter(event => event.event_type === 'TRADE_CLOSED');
}

function syncAccountFromEvents() {
  const previousBalance = Number(state.account.balance);
  const processed = new Set(state.account.processedTradeIds.map(String));
  const closed = closedTradeEvents();
  if (!state.accountWasStored && closed.length && !state.account.trades.length) {
    const firstSize = Number(closed[0].payload?.account_size);
    if (Number.isFinite(firstSize) && firstSize > 0) {
      state.account.initialBalance = firstSize;
      state.account.balance = firstSize;
      $('accountInput').value = firstSize.toFixed(2);
    }
    state.accountWasStored = true;
  }
  let changed = false;
  for (const event of closed) {
    const id = String(event.payload?.trade_id || event.id);
    if (processed.has(id)) continue;
    const pnl = Number(event.payload?.net_pnl);
    if (!Number.isFinite(pnl)) { processed.add(id); state.account.processedTradeIds.push(id); continue; }
    const fees = Number(event.payload?.total_fees || 0);
    const slippage = Number(event.payload?.slippage_cost || 0);
    state.account.balance += pnl;
    state.account.realizedPnl += pnl;
    state.account.totalFees += fees;
    state.account.totalSlippage += slippage;
    state.account.trades.unshift({
      id,
      episodeId: state.episode?.id || '',
      side: event.payload?.side || '',
      pnl,
      fees,
      slippage,
      totalCosts: Number(event.payload?.total_costs || fees + slippage),
      plannedRisk: Number(event.payload?.planned_risk_amount || 0),
      riskOverrun: Number(event.payload?.risk_overrun_amount || 0),
      r: event.payload?.r_multiple,
      reason: event.payload?.exit_reason || 'EXIT',
      entry: event.payload?.entry_price,
      exit: event.payload?.exit_price,
      time: event.event_time_bjt || event.event_time,
      setup: state.events.find(e => e.event_type === 'TRADE_OPEN' && e.payload?.trade_id === event.payload?.trade_id)?.payload?.entry_context?.setup || null,
    });
    processed.add(id);
    state.account.processedTradeIds.push(id);
    changed = true;
  }
  if (changed) {
    syncAccountInputs(previousBalance);
    saveAccount();
    setStatus('交易结果已写入本地账户；下一笔 Account Size 已跟随最新权益', 'ok');
  }
  renderAccount();
}

function reconcileEpisodeAccount(activeEvents) {
  if (!state.episode) return;
  const previousBalance = Number(state.account.balance);
  const activeIds = new Set(activeEvents.filter(e => e.event_type === 'TRADE_CLOSED').map(e => String(e.payload?.trade_id || e.id)));
  const removed = state.account.trades.filter(t => t.episodeId === state.episode.id && !activeIds.has(String(t.id)));
  if (!removed.length) return;
  for (const trade of removed) {
    state.account.balance -= Number(trade.pnl || 0);
    state.account.realizedPnl -= Number(trade.pnl || 0);
    state.account.totalFees -= Number(trade.fees || 0);
    state.account.totalSlippage -= Number(trade.slippage || 0);
  }
  const removedIds = new Set(removed.map(t => String(t.id)));
  state.account.trades = state.account.trades.filter(t => !removedIds.has(String(t.id)));
  state.account.processedTradeIds = state.account.processedTradeIds.filter(id => !removedIds.has(String(id)));
  syncAccountInputs(previousBalance);
  saveAccount();
}

function renderAccount() {
  if (!state.account) return;
  const a = state.account;
  $('equityValue').textContent = money(a.balance);
  $('pnlValue').textContent = `${a.realizedPnl >= 0 ? '+' : ''}${money(a.realizedPnl)}`;
  $('pnlValue').className = `pnl ${a.realizedPnl > 0 ? 'positive' : a.realizedPnl < 0 ? 'negative' : 'neutral'}`;
  $('netProfitMetric').textContent = money(a.realizedPnl);
  $('netProfitMetric').style.color = a.realizedPnl > 0 ? '#16c7a3' : a.realizedPnl < 0 ? '#f05263' : '';
  const wins = a.trades.filter(t => t.pnl > 0);
  const losses = a.trades.filter(t => t.pnl < 0);
  const decided = wins.length + losses.length;
  $('winRateMetric').textContent = decided ? `${(wins.length / decided * 100).toFixed(1)}%` : '—';
  const avgWin = wins.length ? wins.reduce((s,t) => s + t.pnl, 0) / wins.length : 0;
  const avgLoss = losses.length ? losses.reduce((s,t) => s + Math.abs(t.pnl), 0) / losses.length : 0;
  $('payoffMetric').textContent = avgWin && avgLoss ? `${(avgWin / avgLoss).toFixed(2)} : 1` : '—';
  const rValues = a.trades.map(t => Number(t.r)).filter(Number.isFinite);
  $('avgRMetric').textContent = rValues.length ? `${(rValues.reduce((s,v) => s + v, 0) / rValues.length).toFixed(2)}R` : '—';
  $('tradeCountMetric').textContent = String(a.trades.length);
  $('feesMetric').textContent = `${money(a.totalFees)} / ${money(a.totalSlippage)}`;
  $('feesMetric').title = `手续费 ${money(a.totalFees)} · 滑点 ${money(a.totalSlippage)} · 合计 ${money(Number(a.totalFees)+Number(a.totalSlippage))}`;
  const move = Math.max(-50, Math.min(50, a.initialBalance > 0 ? a.realizedPnl / a.initialBalance * 100 : 0));
  const track = $('equityTrack');
  track.style.marginLeft = move < 0 ? `${50 + move}%` : '50%';
  track.style.width = `${Math.max(1, Math.abs(move))}%`;
  track.style.background = move < 0 ? '#f05263' : '#16c7a3';
  renderTradeLog();
}

function renderTradeLog() {
  const host=$('tradeLog'),invalidations=state.events.filter(event=>event.event_type==='LIMIT_CANCEL'&&event.payload?.reason==='take_profit_before_entry');
  const rows=state.account.trades.map(trade=>{
    const win=trade.pnl>=0,reason=trade.reason==='TAKE_PROFIT'?'TP':trade.reason==='STOP_LOSS'?'SL':trade.reason==='AMBIGUOUS_BOTH_HIT'?'SL*':'EXIT';
    return {time:trade.time||'',html:`<article class="trade-row"><div class="trade-top"><strong>${escapeHtml(trade.side)} · ${reason}</strong><span class="result ${win?'win':'loss'}">${trade.pnl>=0?'+':''}${money(trade.pnl)}</span></div><div class="trade-detail"><span>${price(trade.entry)} → ${price(trade.exit)}</span><span>${Number.isFinite(Number(trade.r))?`${Number(trade.r).toFixed(2)}R`:'—'}</span><span>fee ${money(trade.fees)} · slip ${money(trade.slippage)}</span></div><small>${escapeHtml(trade.time||'')} 北京时间${Number(trade.riskOverrun)>0?` · 超出 1R ${money(trade.riskOverrun)}`:''}</small>${renderSetupReview(trade.setup)}</article>`};
  });
  for(const event of invalidations){const p=event.payload||{},time=event.event_time_bjt||event.event_time||'';rows.push({time,html:`<article class="trade-row invalidated"><div class="trade-top"><strong>${escapeHtml(p.side||'')} · 挂单失效</strong><span class="result missed">TP 先到</span></div><div class="trade-detail"><span>Entry ${price(p.limit_price)}</span><span>TP ${price(p.take_profit)}</span><span>未成交 · 0 fee</span></div><small>${escapeHtml(time)} 北京时间 · Replay 自动撤单</small></article>`});}
  if(!rows.length){host.className='trade-log empty-list';host.textContent='成交结果和自动失效的挂单会记录在这里。';return;}
  host.className='trade-log';host.innerHTML=rows.sort((a,b)=>wallTime(b.time)-wallTime(a.time)).slice(0,30).map(row=>row.html).join('');
}

function drawingStorageKey() {
  return `coinbacktest.replay.drawings.${state.episode?.id || 'draft'}`;
}

function loadDrawings() {
  loadSetupDraft();
  try { state.drawings = JSON.parse(localStorage.getItem(drawingStorageKey()) || '[]'); }
  catch (_) { state.drawings = []; }
  if (!Array.isArray(state.drawings)) state.drawings = [];
  state.drawings = state.drawings.map(drawing => ({lineWidth: 2, locked: false, ...drawing}));
  state.selectedDrawingId = null;
}

function saveDrawings() {
  localStorage.setItem(drawingStorageKey(), JSON.stringify(state.drawings));
}

function syncPositionDrawings() {
  let changed=false;
  for (const drawing of state.drawings.filter(item=>item.type==='position')) {
    const order=state.activeLimitOrders.find(item=>String((item.payload||{}).order_id||'')===String(drawing.orderId||''));
    const trade=state.activeTrades.find(item=>String(item.trade_id||'')===String(drawing.tradeId||'')||(drawing.orderId&&String((item.payload||{}).order_id||'')===String(drawing.orderId)));
    if (order) {
      const p=order.payload||{};drawing.entry=Number(order.price);drawing.stop=Number(p.stop_loss);drawing.take=Number(p.take_profit);drawing.status='pending';
      changed=true;
    } else if (trade) {
      const p=trade.payload||{};drawing.tradeId=trade.trade_id;drawing.entry=Number(p.entry_price||trade.price);drawing.stop=Number(p.current_stop_loss??p.initial_stop_loss);drawing.take=Number(p.current_take_profit??p.initial_take_profit);drawing.status='open';
      changed=true;
    } else if(drawing.orderId){const cancelled=[...state.events].reverse().find(event=>event.event_type==='LIMIT_CANCEL'&&String(event.payload?.order_id||'')===String(drawing.orderId)&&event.payload?.reason==='take_profit_before_entry');if(cancelled&&drawing.status!=='cancelled'){drawing.status='cancelled';drawing.cancelReason='take_profit_before_entry';changed=true;}else if(!cancelled&&drawing.status!=='closed'){drawing.status='closed';changed=true;}}
    else if(drawing.tradeId&&drawing.status!=='closed'){drawing.status='closed';changed=true;}
  }
  if (changed) saveDrawings();
}

function addDrawing(drawing) {
  const created={id: `${Date.now()}-${Math.random().toString(16).slice(2,8)}`, color: '#7fa8f2', lineWidth: 2, locked: false, createdCursor: state.episode?.cursor_time || null, ...drawing};
  state.drawings.push(created);
  saveDrawings();
  drawChart();
  return created;
}

function updateTimeframeButtons() {
  document.querySelectorAll('#timeframeNav button').forEach(button => button.classList.toggle('active', button.dataset.timeframe === state.timeframe));
  const label = {'1H':'60','4H':'240','1D':'D'}[state.timeframe] || state.timeframe.replace('m','');
  $('activeTimeframeLabel').textContent = label;
  const minutes=timeframeMinutes();$('stepBtn').title=`下一根 ${state.timeframe} K 线（推进 ${minutes>=1440?'1 天':minutes>=60?`${minutes/60} 小时`:`${minutes} 分钟`}）`;$('rewindBtn').title=`回退一根 ${state.timeframe} K 线`;
}

function timeframeMinutes(tf=state.timeframe) {return Math.max(1,Math.round((TIMEFRAME_MS[tf]||60000)/60000));}

async function loadHealth() {
  const selected = $('symbol').value || 'ETH-USDT-SWAP';
  const data = await request(`/api/health?symbol=${encodeURIComponent(selected)}`);
  state.health = data;
  const options = data.symbols || [];
  $('symbol').innerHTML = options.map(symbol => `<option value="${escapeHtml(symbol)}">${escapeHtml(symbol.replace('-USDT-SWAP','USDT'))} · OKX</option>`).join('');
  $('symbol').value = options.includes(selected) ? selected : (options.includes('ETH-USDT-SWAP') ? 'ETH-USDT-SWAP' : options[0]);
  const c = data.coverage || {};
  $('coverageInfo').textContent = c.available_start_bjt ? `本地 1m：${c.available_start_bjt} → ${c.available_end_bjt}（${Number(c.rows_1m || 0).toLocaleString()} 根）` : '未找到本地行情覆盖信息';
  const current = $('startDatetime').value;
  if (!current && c.last_episode_date) $('startDatetime').value = `${c.last_episode_date}T00:00`;
  updateInstrumentName();
}

function updateInstrumentName() {
  const symbol = state.episode?.symbol || $('symbol').value || 'ETH-USDT-SWAP';
  const short = symbol.replace('-USDT-SWAP','USDT');
  $('instrumentName').textContent = `${short} Perpetual Swap Contract`;
}

function openStartDialog() {
  stopPlay();
  $('continueOption').disabled = !(state.episode?.status === 'closed' && state.episode.symbol === $('symbol').value);
  if ($('startMode').value === 'continue' && $('continueOption').disabled) {
    $('startMode').value = 'sequential';
    $('startDatetimeRow').classList.remove('hidden');
  }
  if (!$('startDialog').open) $('startDialog').showModal();
}

async function createEpisode(event) {
  event?.preventDefault();
  if (state.loading) return;
  const mode = $('startMode').value;
  const body = {symbol: $('symbol').value || 'ETH-USDT-SWAP', mode};
  if (mode === 'continue') {
    if (state.episode?.status !== 'closed' || state.episode.symbol !== body.symbol) return setStatus('请先结束同一品种的上次训练', 'error');
    body.mode = 'sequential';
    body.previous_episode_id = state.episode.id;
  }
  if (mode === 'random') {
    body.random_start = $('randomStart').value;
    body.random_end = $('randomEnd').value;
  } else {
    body.start_time = $('startDatetime').value.replace('T',' ');
  }
  state.loading = true; setStatus('正在建立因果回放窗口…', 'busy');
  try {
    const data = await post('/api/episodes', body);
    state.episode = data.episode; state.clock = data.clock;
    state.visibleOffset = 0; state.historyExhausted = false; state.events = []; state.activeLimitOrders = []; state.activeTrades = [];
    localStorage.setItem(EPISODE_KEY, state.episode.id);
    loadDrawings();
    await refreshSnapshot(true);
    loadTradeTicket('LONG',null,null,null,{scroll:false});
    $('startDialog').close();
    setStatus('Replay 已就绪；所有订单和结果会自动保存', 'ok');
  } catch (error) { setStatus(error.message, 'error'); }
  finally { state.loading = false; }
}

async function restoreEpisode() {
  const episodeId = localStorage.getItem(EPISODE_KEY);
  if (!episodeId) return false;
  try {
    const data = await request(`/api/episodes/${episodeId}/snapshots?timeframes=${encodeURIComponent(state.timeframe)}&limit=700`);
    applySnapshot(data, true);
    loadDrawings();
    syncPositionDrawings();
    loadTradeTicket('LONG',null,null,null,{scroll:false});
    setStatus(state.episode.status === 'active' ? '已恢复上次未完成的 Replay' : '已恢复上次训练记录', 'ok');
    return true;
  } catch (_) {
    localStorage.removeItem(EPISODE_KEY);
    return false;
  }
}

function applySnapshot(data, resetView = false) {
  state.episode = data.episode;
  state.clock = data.clock;
  state.executionPrice = Number.isFinite(Number(data.execution_price)) ? Number(data.execution_price) : null;
  state.bars = data.charts?.[state.timeframe]?.bars || data.bars || [];
  state.events = data.events || [];
  state.activeLimitOrders = data.active_limit_orders || [];
  state.activeTrades = data.active_trades || [];
  state.tradeSummary = data.trade_summary || null;
  state.historyExhausted = false;
  if (resetView) state.visibleOffset = 0;
  $('emptyState').classList.toggle('hidden', Boolean(state.episode));
  updateEpisodeUi();
  syncPositionDrawings();
  syncAccountFromEvents();
  renderActivePlans();
  resizeCanvas();
  if($('entryPrice').value)syncMarketTicketToExecution();
}

async function refreshSnapshot(resetView = false) {
  if (!state.episode) return;
  const data = await request(`/api/episodes/${state.episode.id}/snapshots?timeframes=${encodeURIComponent(state.timeframe)}&limit=700`);
  applySnapshot(data, resetView);
}

async function loadMoreHistory() {
  if (!state.episode || state.historyLoading || state.historyExhausted || !state.bars.length) return;
  state.historyLoading = true;
  const before = state.bars[0].time;
  const episodeId = state.episode.id, timeframe = state.timeframe, cursor = state.episode.cursor_time;
  setStatus('正在加载更早的本地 K 线…', 'busy');
  try {
    const data = await request(`/api/episodes/${state.episode.id}/history?timeframe=${encodeURIComponent(state.timeframe)}&before=${encodeURIComponent(before)}&limit=900`);
    if (state.episode?.id !== episodeId || state.timeframe !== timeframe || state.episode.cursor_time !== cursor) return;
    const known = new Set(state.bars.map(bar => bar.time));
    const older = (data.bars || []).filter(bar => !known.has(bar.time));
    if (older.length) {
      state.bars = [...older, ...state.bars];
      state.bars.sort((a,b) => wallTime(a.time) - wallTime(b.time));
      setStatus(`已加载更早的 ${older.length.toLocaleString()} 根 K 线；可继续向左拖动`, 'ok');
    } else {
      state.historyExhausted = true;
      setStatus('已到本地行情最早边界', 'ok');
    }
    if (data.has_more === false) state.historyExhausted = true;
    drawChart();
  } catch (error) { setStatus(error.message, 'error'); }
  finally { state.historyLoading = false; }
}

function updateEpisodeUi() {
  $('returnToCurrentBtn').disabled = !state.bars.length;
  updateInstrumentName(); updateTimeframeButtons();
  updateTradeTicketAvailability();
  updateEndEpisodeAvailability();
  if (!state.episode) return;
  $('episodeLabel').textContent = `${state.episode.id} · ${state.episode.status}`;
  $('cursorClock').textContent = state.chartTimezone === 'new_york' ? `纽约 ${state.clock?.new_york || state.episode.cursor_time}` : `北京时间 ${state.clock?.beijing_plain || String(state.clock?.beijing || '').replace(' CST','') || state.episode.cursor_time || '—'}`;
  const latest = state.bars[state.bars.length - 1];
  $('formingBadge').textContent = latest?.is_partial ? '● FORMING' : state.episode.status === 'active' ? '● REPLAY' : 'CLOSED';
  $('formingBadge').classList.toggle('live', state.episode.status === 'active');
  updateLegend(latest);
}

function updateTradeTicketAvailability() {
  const closed=Boolean(state.episode&&state.episode.status!=='active');
  $('tradeTicket').classList.toggle('disabled',closed);
  document.querySelectorAll('#tradeTicket input, #tradeTicket button').forEach(control=>{control.disabled=closed;});
  if(closed)$('tradeTicketSource').textContent='训练已结束';
}

function updateEndEpisodeAvailability() {
  const button=$('endEpisodeBtn'),active=state.episode?.status==='active',waiting=Boolean(active&&state.activeTrades.length);
  button.disabled=!active||waiting;
  button.textContent=waiting?'持仓等待 TP / SL':active?'结束本次训练':state.episode?'训练已结束':'结束本次训练';
  button.title=waiting?'持仓只能由 Replay 自动触发止盈或止损':'结束当前 Replay 训练';
}

function updateLegend(bar) {
  if (!bar) return;
  const up = Number(bar.close) >= Number(bar.open);
  const cls = up ? 'up' : 'down';
  $('ohlcLine').innerHTML = `O <span class="${cls}">${price(bar.open)}</span>&nbsp;&nbsp; H <span class="${cls}">${price(bar.high)}</span>&nbsp;&nbsp; L <span class="${cls}">${price(bar.low)}</span>&nbsp;&nbsp; C <span class="${cls}">${price(bar.close)}</span>`;
  $('volumeValue').textContent = compactNumber(bar.volume);
}

function appendBars(updates) {
  let added = 0;
  for (const bar of updates?.[state.timeframe] || []) {
    const index = state.bars.findIndex(item => item.time === bar.time);
    if (index >= 0) state.bars[index] = bar; else { state.bars.push(bar); added += 1; }
  }
  state.bars.sort((a,b) => wallTime(a.time) - wallTime(b.time));
  if (added && state.visibleOffset > 0) state.visibleOffset += added;
}

async function step(minutes = null) {
  if (!state.episode || state.episode.status !== 'active' || state.loading) return;
  const amount = Number(minutes || $('stepSize').value || 1);
  state.loading = true; setStatus(`向前推进 ${amount} 分钟…`, 'busy');
  try {
    const data = await post(`/api/episodes/${state.episode.id}/step`, {minutes: amount, timeframes: [state.timeframe], pause_on_event: state.pauseOnEvent});
    state.episode = data.episode; state.clock = data.clock; state.executionPrice=Number.isFinite(Number(data.execution_price))?Number(data.execution_price):null;appendBars(data.updates);
    if (Array.isArray(data.trade_events)) state.events.push(...data.trade_events);
    state.activeLimitOrders = data.active_limit_orders || [];
    state.activeTrades = data.active_trades || [];
    state.tradeSummary = data.trade_summary || state.tradeSummary;
    syncPositionDrawings(); updateEpisodeUi(); syncAccountFromEvents(); renderActivePlans(); syncMarketTicketToExecution();drawChart();
    const closed=data.trade_events?.find(e=>e.event_type==='TRADE_CLOSED'),invalidated=data.trade_events?.find(e=>e.event_type==='LIMIT_CANCEL'&&e.payload?.reason==='take_profit_before_entry');
    if (closed) setStatus(`${closed.payload?.exit_reason || '交易结束'} · ${Number(closed.payload?.r_multiple || 0).toFixed(2)}R · ${money(closed.payload?.net_pnl)} · Replay 继续，可准备下一笔`, 'ok');
    else if(invalidated)setStatus(`${invalidated.payload?.side||''} 挂单已自动撤销：价格先到 TP ${price(invalidated.payload?.take_profit)}，Entry 尚未成交`,'ok');
    else if (data.at_data_end) setStatus('已到本地数据边界', 'error');
    else setStatus(amount===timeframeMinutes()?`已生成下一根 ${state.timeframe} K 线 · ${data.lifecycle_resolution==='cached_1m_sequence'?'订单路径已用 1m 排序':'周期直达'} · 自动保存`:`已推进 ${data.advanced_minutes || amount} 分钟 · 自动保存`, 'ok');
    if (data.paused_on_event) {
      stopPlay();
      const reason = closed ? '止盈 / 止损' : invalidated ? '挂单失效' : '订单成交';
      setStatus(`${reason} · 已暂停在事件时刻（推进 ${data.advanced_minutes} 分钟），可检查图表后继续`, 'ok');
    }
    if (data.at_data_end || state.episode.status !== 'active') stopPlay();
  } catch (error) { stopPlay(); setStatus(error.message, 'error'); }
  finally { state.loading = false; }
}

async function rewind(minutes=null) {
  stopPlay();
  if (!state.episode || state.episode.status !== 'active' || state.loading) return;
  state.loading = true; setStatus('正在回退并归档未来分支…', 'busy');
  try {
    const rewindMinutes=Number(minutes||timeframeMinutes());const data = await post(`/api/episodes/${state.episode.id}/rewind`, {minutes: rewindMinutes, timeframes: [state.timeframe]});
    reconcileEpisodeAccount(data.events || []);
    state.drawings = state.drawings.filter(d => !d.createdCursor || wallTime(d.createdCursor) <= wallTime(data.episode.cursor_time));
    saveDrawings();
    applySnapshot(data, false);
    setStatus(`已回退一根 ${state.timeframe} K 线（${data.rewound_minutes || 0} 分钟）；放弃的未来事件已归档`, 'ok');
  } catch (error) { setStatus(error.message, 'error'); }
  finally { state.loading = false; }
}

function startPlay() {
  if (!state.episode || state.episode.status !== 'active' || state.playTimer || state.loading || document.querySelector('dialog[open]')) return;
  const generation = ++state.playGeneration;
  $('playBtn').textContent = 'Ⅱ'; $('playBtn').classList.add('playing');
  const loop = async () => {
    await step(timeframeMinutes());
    if (generation === state.playGeneration && state.playTimer && state.episode?.status === 'active') state.playTimer = setTimeout(loop, Number($('playSpeed').value || 1000));
  };
  state.playTimer = setTimeout(loop, 0);
}

function stopPlay() {
  state.playGeneration += 1;
  if (state.playTimer) clearTimeout(state.playTimer);
  state.playTimer = null; $('playBtn').textContent = '▶'; $('playBtn').classList.remove('playing');
}

async function changeTimeframe(tf) {
  stopPlay();
  if (state.loading) return setStatus('当前操作完成后再切换周期', 'busy');
  if (tf === state.timeframe) return;
  const previousTimeframe = state.timeframe;
  state.loading = true;
  state.timeframe = tf; state.visibleOffset = 0; state.historyExhausted = false; state.manualPriceRange = null; savePreferences(); updateTimeframeButtons();
  if (state.episode) {
    setStatus(`切换到 ${tf}…`, 'busy');
    try { await refreshSnapshot(true); setStatus(`${tf} K 线已加载`, 'ok'); }
    catch (error) { state.timeframe = previousTimeframe; savePreferences(); updateTimeframeButtons(); setStatus(error.message, 'error'); }
  } else drawChart();
  state.loading = false;
}

function chartTime(bar) {
  return state.chartTimezone === 'new_york' ? bar?.time : (bar?.time_bjt || bar?.time);
}

function visibleBars() {
  const total = state.bars.length;
  if (!total) return [];
  const count = Math.max(30, Math.min(5000, Math.round(state.visibleCount)));
  const maxBlank = Math.max(0, Math.min(count - 20, Math.floor(count * .78)));
  const maxOffset = Math.max(0, total - count);
  state.visibleOffset = Math.max(-maxBlank, Math.min(state.visibleOffset, maxOffset));
  const blankSlots = Math.max(0, -Math.round(state.visibleOffset));
  const desiredBars = Math.max(1, count - blankSlots);
  const end = total - Math.max(0, Math.round(state.visibleOffset));
  const bars = state.bars.slice(Math.max(0, end - desiredBars), end);
  bars._slots = count;
  bars._blankSlots = blankSlots;
  bars._startSlot = Math.max(0, desiredBars - bars.length);
  return bars;
}

function returnToCurrentTime() {
  if (!state.bars.length) return;
  state.visibleOffset = 0;
  state.hoverIndex = null;
  state.hoverPoint = null;
  $('crosshairTooltip').classList.add('hidden');
  updateLegend(state.bars[state.bars.length - 1]);
  drawChart();
}

function plotGeometry() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const mainBottom = Math.max(160, h * .77);
  return {
    width: w, height: h,
    main: {left: 4, top: 8, right: w - 68, bottom: mainBottom, width: Math.max(20,w - 72), height: Math.max(80,mainBottom - 8)},
    volume: {left: 4, top: mainBottom + 14, right: w - 68, bottom: h - 25, width: Math.max(20,w - 72), height: Math.max(20,h - mainBottom - 39)},
  };
}

function automaticPriceRange(bars) {
  if (!bars.length) return {min: 0, max: 1};
  // Autoscale follows visible candles only. Overlays must never squeeze price action.
  const min = Math.min(...bars.map(b => Number(b.low))), max = Math.max(...bars.map(b => Number(b.high)));
  const pad = Math.max((max-min)*.08,max*.0008,.01);
  return {min:min-pad,max:max+pad};
}

function currentPriceRange(bars) {
  const automatic = automaticPriceRange(bars);
  if (state.autoScale) return automatic;
  if (!state.manualPriceRange || !Number.isFinite(state.manualPriceRange.min) || !Number.isFinite(state.manualPriceRange.max) || state.manualPriceRange.max <= state.manualPriceRange.min) {
    state.manualPriceRange = {...automatic};
  }
  return {...state.manualPriceRange};
}

function updateAutoScaleButton() {
  const button = $('autoScaleBtn');
  if (!button) return;
  button.classList.toggle('active', state.autoScale);
  button.setAttribute('aria-pressed', String(state.autoScale));
  button.textContent = state.autoScale ? '自动' : '手动';
  button.title = state.autoScale ? '自动坐标：仅适配当前可见 K 线，画线和订单不参与缩放' : '手动坐标：可上下拖动；在价格轴或按 Alt 滚轮纵向缩放';
}

function yForPrice(value, range, plot) { return plot.bottom - (Number(value)-range.min)/Math.max(1e-9,range.max-range.min)*plot.height; }
function priceForY(y, range, plot) { return range.min + (plot.bottom-y)/Math.max(1,plot.height)*(range.max-range.min); }
// Coordinate math must retain subpixel slots when many candles share a pixel.
function barSlotWidth(bars, plot) { return plot.width / Math.max(1, bars._slots || bars.length); }
function xForIndex(index, bars, plot) { return plot.left + ((bars._startSlot || 0) + index + .5) * barSlotWidth(bars,plot); }
function xForTime(time, bars, plot) {
  if (!bars.length) return plot.left;
  const direct=bars.findIndex(bar=>String(bar.time)===String(time));if(direct>=0)return xForIndex(direct,bars,plot);
  const lastIndex=bars.length-1,target=wallTime(time),last=wallTime(bars[lastIndex].time),step=TIMEFRAME_MS[state.timeframe]||60000,slot=barSlotWidth(bars,plot);
  return xForIndex(lastIndex,bars,plot)+(target-last)/step*slot;
}

function wallTextFromMs(ms) { return new Date(ms).toISOString().slice(0,19).replace('T',' '); }
function shiftWallTime(time, deltaMs) { return wallTextFromMs(wallTime(time) + deltaMs); }

function timeForX(x,bars,plot) {
  if (!bars.length) return null;
  const lastIndex=bars.length-1,step=TIMEFRAME_MS[state.timeframe]||60000,slot=barSlotWidth(bars,plot),lastX=xForIndex(lastIndex,bars,plot);
  return wallTextFromMs(wallTime(bars[lastIndex].time)+(x-lastX)/Math.max(1e-9,slot)*step);
}

function canvasPointFromEvent(event, {snap = state.magnetMode} = {}) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  const bars = visibleBars(), geo = plotGeometry(), plot = geo.main, range = currentPriceRange(bars);
  if (!bars.length || x < plot.left || x > plot.right || y < plot.top || y > plot.bottom) return null;
  const slot=(x-plot.left)/Math.max(1e-9,barSlotWidth(bars,plot))-(bars._startSlot||0);
  const rawIndex=Math.floor(slot), index=Math.max(0,Math.min(bars.length-1,rawIndex));
  const bar=rawIndex>=0&&rawIndex<bars.length?bars[rawIndex]:null; let value = priceForY(y,range,plot); let snapField = null;
  const magnetMode=snap===false?'off':snap===true?state.magnetMode:snap;
  if (magnetMode !== 'off' && bar) {
    const candidates = [['O',bar.open],['H',bar.high],['L',bar.low],['C',bar.close]].map(([k,v]) => [k,Number(v)]).sort((a,b)=>Math.abs(a[1]-value)-Math.abs(b[1]-value));
    const nearest=candidates[0],withinWeakRange=nearest&&Math.abs(yForPrice(nearest[1],range,plot)-y)<=MAGNET_WEAK_THRESHOLD_PX;
    if (nearest&&(magnetMode==='strong'||withinWeakRange)) [snapField,value] = nearest;
  }
  const snappedToBar=Boolean(snapField)&&Boolean(bar),time=snappedToBar?bar.time:timeForX(x,bars,plot),pointX=snappedToBar?xForIndex(index,bars,plot):x;
  return {time,time_bjt:snappedToBar?(bar.time_bjt||time):time,price:value,index:rawIndex,snapField,x:pointX,y:yForPrice(value,range,plot)};
}

function drawGrid(bars, range, geo) {
  ctx.lineWidth = 1; ctx.font = '10px Inter, sans-serif'; ctx.textBaseline = 'middle';
  for (let i=0;i<=6;i++) {
    const y=geo.main.top+geo.main.height*i/6; ctx.strokeStyle='#171b23'; ctx.beginPath(); ctx.moveTo(geo.main.left,y); ctx.lineTo(geo.main.right,y); ctx.stroke();
    const value=range.max-(range.max-range.min)*i/6; ctx.fillStyle='#8a929e'; ctx.fillText(price(value),geo.main.right+7,y);
  }
  for (let i=0;i<=7;i++) {
    const x=geo.main.left+geo.main.width*i/7; ctx.strokeStyle='#151920'; ctx.beginPath(); ctx.moveTo(x,geo.main.top); ctx.lineTo(x,geo.volume.bottom); ctx.stroke();
    const slot=Math.round((bars._slots||bars.length)*i/7)-(bars._startSlot||0); const index=Math.max(0,Math.min(bars.length-1,slot)); const future=timeForX(x,bars,geo.main); const label=String(chartTime(bars[index]) || future || '').slice(5,16);
    ctx.fillStyle='#747d8b'; ctx.textAlign=i===0?'left':i===7?'right':'center'; ctx.fillText(label,x,geo.height-11);
  }
  ctx.textAlign='left';
}

function drawCandles(bars, range, geo) {
  const step=barSlotWidth(bars,geo.main), body=Math.max(1,Math.min(9,step*.68));
  const maxVol=Math.max(1,...bars.map(b=>Number(b.volume||0)));
  bars.forEach((bar,index)=>{
    const x=xForIndex(index,bars,geo.main), up=Number(bar.close)>=Number(bar.open), color=up?'#17bda0':'#ef4e60';
    const high=yForPrice(bar.high,range,geo.main), low=yForPrice(bar.low,range,geo.main), open=yForPrice(bar.open,range,geo.main), close=yForPrice(bar.close,range,geo.main);
    ctx.strokeStyle=color; ctx.fillStyle=color; ctx.globalAlpha=bar.is_partial?.72:1; ctx.beginPath(); ctx.moveTo(x,high); ctx.lineTo(x,low); ctx.stroke();
    const top=Math.min(open,close), height=Math.max(1,Math.abs(close-open));
    if (bar.is_partial) { ctx.fillStyle=up?'rgba(23,189,160,.3)':'rgba(239,78,96,.3)'; ctx.fillRect(x-body/2,top,body,height); ctx.strokeRect(x-body/2,top,body,height); }
    else ctx.fillRect(x-body/2,top,body,height);
    const volHeight=Number(bar.volume||0)/maxVol*geo.volume.height; ctx.globalAlpha=.42; ctx.fillStyle=color; ctx.fillRect(x-body/2,geo.volume.bottom-volHeight,body,volHeight); ctx.globalAlpha=1;
  });
  const latest=bars[bars.length-1]; if (latest) {
    const y=yForPrice(latest.close,range,geo.main), color=Number(latest.close)>=Number(latest.open)?'#17bda0':'#ef4e60';
    ctx.strokeStyle=color; ctx.setLineDash([3,3]); ctx.globalAlpha=.62; ctx.beginPath(); ctx.moveTo(Math.max(geo.main.left,geo.main.right-90),y); ctx.lineTo(geo.main.right,y); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha=1;
    drawAxisPrice(y,price(latest.close),color,geo);
  }
}

function drawAxisPrice(y,text,color,geo) {
  ctx.font='10px Inter, sans-serif'; const width=Math.min(65,ctx.measureText(text).width+11); ctx.fillStyle=color; ctx.fillRect(geo.main.right+1,y-9,width,18); ctx.fillStyle='#fff'; ctx.textBaseline='middle'; ctx.fillText(text,geo.main.right+6,y);
}

function drawPositionPlan(side, entry, stop, take, startTime, label, bars, range, geo, alpha=.2) {
  const x=Math.max(geo.main.left,xForTime(startTime,bars,geo.main)), right=geo.main.right;
  const ey=yForPrice(entry,range,geo.main), sy=yForPrice(stop,range,geo.main), ty=yForPrice(take,range,geo.main);
  ctx.globalAlpha=alpha; ctx.fillStyle='#17bda0'; ctx.fillRect(x,Math.min(ey,ty),Math.max(1,right-x),Math.abs(ey-ty)); ctx.fillStyle='#ef4e60'; ctx.fillRect(x,Math.min(ey,sy),Math.max(1,right-x),Math.abs(ey-sy)); ctx.globalAlpha=1;
  ctx.strokeStyle='#f4f7fb';ctx.lineWidth=1.5;ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(x,ey);ctx.lineTo(right,ey);ctx.stroke();
}

function plannedTradeMetrics({side='LONG',account,riskPct,entry,stop,take,orderType='limit',entryExecuted=false}) {
  const isLong=side==='LONG',rawEntry=Number(entry),stopLevel=Number(stop),takeLevel=Number(take),size=Number(account),risk=Number(riskPct);
  const entryExec=orderType==='market'&&!entryExecuted?rawEntry*(isLong?1+MARKET_SLIPPAGE:1-MARKET_SLIPPAGE):rawEntry;
  const stopExec=stopLevel*(isLong?1-MARKET_SLIPPAGE:1+MARKET_SLIPPAGE),entryFeeRate=orderType==='market'?MARKET_FEE:LIMIT_FEE;
  const riskAmount=size*risk/100,riskPerUnit=Math.abs(entryExec-stopLevel),qty=riskPerUnit>0?riskAmount/riskPerUnit:0;
  const stopLossPerUnit=Math.abs(entryExec-stopExec)+entryExec*entryFeeRate+stopExec*MARKET_FEE,stopNetLoss=stopLossPerUnit*qty,costOverrun=Math.max(0,stopNetLoss-riskAmount);
  const grossRewardPerUnit=isLong?takeLevel-entryExec:entryExec-takeLevel,netRewardPerUnit=grossRewardPerUnit-entryExec*entryFeeRate-takeLevel*LIMIT_FEE;
  const netRR=riskPerUnit>0?netRewardPerUnit/riskPerUnit:0,reward=netRewardPerUnit*qty;
  return {account:size,riskAmount,qty,rr:netRR,reward,entryExec,stopExec,riskPerUnit,stopNetLoss,costOverrun,netRewardPerUnit,stopPct:Math.abs(stopLevel-entryExec)/entryExec*100,takePct:Math.abs(takeLevel-entryExec)/entryExec*100};
}

function positionMetrics(drawing) {
  return plannedTradeMetrics({side:drawing.side||'LONG',account:Number(drawing.accountSize||state.account?.balance||10000),riskPct:Number(drawing.riskPct||1),entry:Number(drawing.entry),stop:Number(drawing.stop),take:Number(drawing.take),orderType:drawing.orderType||'limit',entryExecuted:['open','closed'].includes(drawing.status)});
}

function positionPixels(drawing,bars,range,geo) {
  const step=TIMEFRAME_MS[state.timeframe]||60000,start=drawing.a?.time||bars[bars.length-1]?.time,end=drawing.endTime||shiftWallTime(start,step*18);
  const x1=xForTime(start,bars,geo.main);let x2=xForTime(end,bars,geo.main);
  if (x2-x1<56) x2=x1+56;
  return {x1,x2,entryY:yForPrice(drawing.entry,range,geo.main),stopY:yForPrice(drawing.stop,range,geo.main),takeY:yForPrice(drawing.take,range,geo.main)};
}

function positionHandles(p) {
  return {
    take: {x:p.x1,y:p.takeY},
    entry: {x:p.x1,y:p.entryY},
    stop: {x:p.x1,y:p.stopY},
    width: {x:p.x2,y:p.entryY},
  };
}

function positionIsPlan(drawing) {
  return !drawing.orderId&&!drawing.tradeId&&drawing.status!=='pending'&&drawing.status!=='open'&&drawing.status!=='closed';
}

function positionActionPoint(p) {
  return {x:(p.x1+p.x2)/2,y:p.entryY};
}

function drawPositionLabel(text,x,y,color) {
  ctx.font='600 9px Inter, sans-serif';const width=Math.min(250,ctx.measureText(text).width+12),left=x-width;
  ctx.fillStyle=color;ctx.fillRect(left,y-9,width,18);ctx.fillStyle='#f4f7fb';ctx.textBaseline='middle';ctx.fillText(text,left+6,y);
}

function drawEditablePosition(drawing,bars,range,geo,selected) {
  const p=positionPixels(drawing,bars,range,geo),left=Math.max(geo.main.left,p.x1),right=Math.min(geo.main.right,p.x2),width=Math.max(1,right-left);
  if (right<geo.main.left||left>geo.main.right) return;
  ctx.fillStyle='rgba(20,184,143,.22)';ctx.fillRect(left,Math.min(p.entryY,p.takeY),width,Math.abs(p.entryY-p.takeY));
  ctx.fillStyle='rgba(239,78,96,.23)';ctx.fillRect(left,Math.min(p.entryY,p.stopY),width,Math.abs(p.entryY-p.stopY));
  ctx.lineWidth=Number(drawing.lineWidth||2);
  ctx.strokeStyle='#f4f7fb';ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(left,p.entryY);ctx.lineTo(right,p.entryY);ctx.stroke();
  // Keep placement and dragging unobstructed. Details appear only after the
  // finished Position is hovered or selected.
  const detailed=Boolean(!drawing.preview&&!state.drawingInteraction&&(selected||state.hoverDrawingId===drawing.id));
  if (!detailed) return;
  const m=positionMetrics(drawing);
  for (const [y,color] of [[p.takeY,'#1fc7a3'],[p.stopY,'#f05a69']]) {ctx.strokeStyle=color;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();}
  ctx.setLineDash([]);ctx.strokeStyle=drawing.color||'#b8c1cf';ctx.globalAlpha=.72;ctx.strokeRect(left,Math.min(p.takeY,p.stopY),width,Math.abs(p.takeY-p.stopY));ctx.globalAlpha=1;
  const status=positionIsPlan(drawing)?'PLAN':drawing.status==='open'?'OPEN':drawing.status==='cancelled'?'TP FIRST · CANCELLED':drawing.status==='closed'?'CLOSED':drawing.orderType==='market'?'MARKET':'LIMIT';
  drawPositionLabel(`Target: +${money(m.reward)} (+${m.takePct.toFixed(2)}%) · ${m.rr.toFixed(2)}R`,right,p.takeY,'rgba(19,143,116,.96)');
  drawPositionLabel(`${drawing.side} · ${status} · Qty ${m.qty.toFixed(4)} ETH`,right,p.entryY,'rgba(65,75,91,.97)');
  drawPositionLabel(`Stop: 1R -${money(m.riskAmount)} · 预计净亏 -${money(m.stopNetLoss)}（费用/滑点额外）`,right,p.stopY,'rgba(180,55,68,.97)');
  if(positionIsPlan(drawing)){
    const action=positionActionPoint({x1:left,x2:right,entryY:p.entryY});ctx.fillStyle='#111722';ctx.strokeStyle=drawing.side==='SHORT'?'#f05263':'#16c7a3';ctx.lineWidth=1.6;ctx.beginPath();ctx.arc(action.x,action.y,8,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.strokeStyle='#f4f7fb';ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(action.x-3.5,action.y);ctx.lineTo(action.x+3.5,action.y);ctx.moveTo(action.x,action.y-3.5);ctx.lineTo(action.x,action.y+3.5);ctx.stroke();
  }
  const locked=state.drawingsLocked||drawing.locked;
  if (!locked) {
    const handles=positionHandles({x1:left,x2:right,entryY:p.entryY,stopY:p.stopY,takeY:p.takeY});
    for (const [name,point] of Object.entries(handles)) {
      ctx.fillStyle='#0b0f15';ctx.strokeStyle='#f4f7fb';ctx.lineWidth=1.4;ctx.beginPath();
      if(name==='entry')ctx.arc(point.x,point.y,4.5,0,Math.PI*2);else ctx.rect(point.x-4,point.y-4,8,8);
      ctx.fill();ctx.stroke();
    }
  } else {ctx.fillStyle='#e5b966';ctx.font='bold 10px Inter, sans-serif';ctx.fillText('LOCKED',left+6,Math.min(p.takeY,p.stopY)+12);}
}

function drawTradeOverlays(bars, range, geo) {
  for (const order of state.activeLimitOrders) {
    const p=order.payload||{}, stop=Number(p.stop_loss), take=Number(p.take_profit), entry=Number(order.price);
    const represented=state.drawings.some(d=>d.type==='position'&&d.orderId&&d.orderId===p.order_id);
    if (!represented&&[entry,stop,take].every(Number.isFinite)) drawPositionPlan(p.side,entry,stop,take,order.event_time,`${p.side} · LIMIT PENDING`,bars,range,geo,.16);
  }
  for (const trade of state.activeTrades) {
    const p=trade.payload||{}, entry=Number(p.entry_price||trade.price), stop=Number(p.current_stop_loss??p.initial_stop_loss), take=Number(p.current_take_profit??p.initial_take_profit);
    const represented=state.drawings.some(d=>d.type==='position'&&((d.tradeId&&d.tradeId===trade.trade_id)||(d.orderId&&d.orderId===p.order_id)));
    if (!represented&&[entry,stop,take].every(Number.isFinite)) drawPositionPlan(p.side,entry,stop,take,trade.event_time,`${p.side} · OPEN · ${Number(p.quantity||0).toFixed(3)} ETH`,bars,range,geo,.2);
  }
}

function tradeMarkerTime(event,entry=false) {
  const payload=event.payload||{};
  if(payload.trigger_bar_time)return payload.trigger_bar_time;
  if(entry&&payload.entry_event_id){const source=state.events.find(item=>Number(item.id)===Number(payload.entry_event_id));if(source?.payload?.trigger_bar_time)return source.payload.trigger_bar_time;}
  return entry?(payload.entry_time||event.event_time):(payload.exit_time||event.event_time);
}

function roundedMarkerPath(x,y,width,height,radius=4) {
  const right=x+width,bottom=y+height,r=Math.min(radius,width/2,height/2);ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(right,y,right,bottom,r);ctx.arcTo(right,bottom,x,bottom,r);ctx.arcTo(x,bottom,x,y,r);ctx.arcTo(x,y,right,y,r);ctx.closePath();
}

function drawEntryMarker(x,y,side,geo) {
  const isLong=side==='LONG',color=isLong?'#17c99f':'#ef4e78',text=isLong?'B':'S',size=18;
  const preferredTop=isLong?y+7:y-size-7,top=Math.max(geo.main.top+2,Math.min(geo.main.bottom-size-2,preferredTop)),left=x-size/2;
  ctx.save();ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x,isLong?top:top+size);ctx.stroke();roundedMarkerPath(left,top,size,size,4);ctx.fillStyle=color;ctx.fill();ctx.fillStyle='#07100e';ctx.font='800 11px Inter, sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(text,x,top+size/2+.5);ctx.restore();
}

function drawExitMarker(x,y,event,geo) {
  const payload=event.payload||{},reason=String(payload.exit_reason||''),isTp=reason==='TAKE_PROFIT',isAmbiguous=reason==='AMBIGUOUS_BOTH_HIT',label=isTp?'TP':isAmbiguous?'SL*':'SL',color=isTp?'#17c99f':isAmbiguous?'#e4ad54':'#f05268';
  const rawMultiple=payload.r_multiple,multiple=rawMultiple===null||rawMultiple===undefined||rawMultiple===''?NaN:Number(rawMultiple),rText=Number.isFinite(multiple)?` ${multiple>=0?'+':''}${multiple.toFixed(2)}R`:'';
  ctx.save();ctx.fillStyle='#080a0e';ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,5.5,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.font='700 10px Inter, sans-serif';ctx.textBaseline='middle';const text=`${label}${rText}`,toLeft=x+ctx.measureText(text).width+14>geo.main.right,textY=y-9<geo.main.top+5?y+11:y-9;ctx.textAlign=toLeft?'right':'left';ctx.fillStyle=color;ctx.fillText(text,x+(toLeft?-9:9),textY);ctx.restore();
}

function drawTradeMarkers(bars,range,geo) {
  const seenEntries=new Set();
  for(const event of state.events.filter(item=>item.event_type==='TRADE_OPEN')){
    const payload=event.payload||{},tradeId=String(payload.trade_id||event.id);if(seenEntries.has(tradeId))continue;seenEntries.add(tradeId);const side=String(payload.side||'').toUpperCase(),markerPrice=Number(payload.entry_price??event.price),markerTime=tradeMarkerTime(event,true);if(!['LONG','SHORT'].includes(side)||!Number.isFinite(markerPrice)||!markerTime)continue;const x=xForTime(markerTime,bars,geo.main),y=yForPrice(markerPrice,range,geo.main);if(x<geo.main.left||x>geo.main.right||y<geo.main.top||y>geo.main.bottom)continue;drawEntryMarker(x,y,side,geo);
  }
  for (const event of state.events.filter(e=>e.event_type==='TRADE_CLOSED')) {
    const markerPrice=Number(event.payload?.exit_price??event.price),markerTime=tradeMarkerTime(event,false),y=yForPrice(markerPrice,range,geo.main),x=xForTime(markerTime,bars,geo.main);if(!markerTime||x<geo.main.left||x>geo.main.right||y<geo.main.top||y>geo.main.bottom)continue;drawExitMarker(x,y,event,geo);
  }
}

function pointPixels(point,bars,range,geo,preferCapturedX=false) {const capturedX=Number(point.x),x=preferCapturedX&&Number.isFinite(capturedX)?capturedX:xForTime(point.time,bars,geo.main);return {x,y:yForPrice(point.price,range,geo.main)};}

function rayEndPoint(a,b,geo) {
  const dx=b.x-a.x,dy=b.y-a.y;if(!dx&&!dy)return {...b};
  const candidates=[];
  if(dx>0)candidates.push((geo.main.right-a.x)/dx);else if(dx<0)candidates.push((geo.main.left-a.x)/dx);
  if(dy>0)candidates.push((geo.main.bottom-a.y)/dy);else if(dy<0)candidates.push((geo.main.top-a.y)/dy);
  const positive=candidates.filter(value=>Number.isFinite(value)&&value>=0),t=positive.length?Math.min(...positive):1;
  return {x:a.x+dx*t,y:a.y+dy*t};
}

function drawLineExtended(a,b,mode,geo) {
  let end={...b};
  if(mode==='ray')end=rayEndPoint(a,b,geo);
  if(mode==='horizontal-ray')end={x:geo.main.right,y:a.y};
  ctx.save();ctx.beginPath();ctx.rect(geo.main.left,geo.main.top,geo.main.width,geo.main.height);ctx.clip();ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(end.x,end.y);ctx.stroke();ctx.restore();
}

function calloutGeometry(a,text,geo) {
  ctx.save();ctx.font='10px Inter, sans-serif';const width=Math.min(210,ctx.measureText(String(text||'Callout')).width+14);ctx.restore();
  const height=22,toLeft=a.x+13+width>geo.main.right-4,toBottom=a.y-36<geo.main.top+4;
  const preferredLeft=toLeft?a.x-width-13:a.x+13,preferredTop=toBottom?a.y+14:a.y-36;
  const left=Math.max(geo.main.left+4,Math.min(geo.main.right-width-4,preferredLeft)),top=Math.max(geo.main.top+4,Math.min(geo.main.bottom-height-4,preferredTop));
  return {left,top,width,height,elbow:{x:toLeft?a.x-15:a.x+15,y:toBottom?a.y+18:a.y-18}};
}

function rectangleHandles(a,b) {
  const left=Math.min(a.x,b.x),right=Math.max(a.x,b.x),top=Math.min(a.y,b.y),bottom=Math.max(a.y,b.y),mx=(left+right)/2,my=(top+bottom)/2;
  return {nw:{x:left,y:top},n:{x:mx,y:top},ne:{x:right,y:top},e:{x:right,y:my},se:{x:right,y:bottom},s:{x:mx,y:bottom},sw:{x:left,y:bottom},w:{x:left,y:my}};
}

function drawControlPoint(point,size=8) {ctx.fillStyle='#0b0f15';ctx.strokeStyle='#4b91ff';ctx.lineWidth=2;ctx.fillRect(point.x-size/2,point.y-size/2,size,size);ctx.strokeRect(point.x-size/2,point.y-size/2,size,size);}

function drawOneDrawing(drawing,bars,range,geo,draft=false) {
  if (!drawing.a) return;
  const a=pointPixels(drawing.a,bars,range,geo,draft), b=pointPixels(drawing.b||drawing.a,bars,range,geo,draft);
  const selected=state.selectedDrawingId===drawing.id; ctx.save(); ctx.lineWidth=Number(drawing.lineWidth||2)+(selected ? .35 : 0); ctx.strokeStyle=drawing.color||'#7fa8f2'; ctx.fillStyle=drawing.color||'#7fa8f2'; ctx.globalAlpha=draft?.7:1;
  if (drawing.type==='trend'||drawing.type==='ray'||drawing.type==='horizontal-ray') drawLineExtended(a,b,drawing.type,geo);
  else if (drawing.type==='rectangle') { ctx.fillStyle=`${drawing.color||'#7fa8f2'}18`; ctx.fillRect(Math.min(a.x,b.x),Math.min(a.y,b.y),Math.abs(b.x-a.x),Math.abs(b.y-a.y)); ctx.strokeRect(Math.min(a.x,b.x),Math.min(a.y,b.y),Math.abs(b.x-a.x),Math.abs(b.y-a.y));if(selected&&!state.drawingsLocked&&!drawing.locked)for(const point of Object.values(rectangleHandles(a,b)))drawControlPoint(point); }
  else if (drawing.type==='vertical') { ctx.setLineDash([5,4]); ctx.beginPath(); ctx.moveTo(a.x,geo.main.top); ctx.lineTo(a.x,geo.volume.bottom); ctx.stroke(); ctx.setLineDash([]); }
  else if (drawing.type==='ruler') {
    ctx.setLineDash([4,3]); drawLineExtended(a,b,'trend',geo); ctx.setLineDash([]); const delta=drawing.b.price-drawing.a.price, pct=delta/drawing.a.price*100, barsCount=Math.round(Math.abs(wallTime(drawing.b.time)-wallTime(drawing.a.time))/(TIMEFRAME_MS[state.timeframe]||60000));
    const text=`${delta>=0?'+':''}${price(delta)} · ${pct>=0?'+':''}${pct.toFixed(2)}% · ${barsCount} bars`; ctx.font='10px Inter, sans-serif'; const w=ctx.measureText(text).width+12; ctx.fillStyle='rgba(32,48,73,.95)'; ctx.fillRect((a.x+b.x)/2-w/2,(a.y+b.y)/2-20,w,18); ctx.fillStyle='#cfe0ff'; ctx.fillText(text,(a.x+b.x)/2-w/2+6,(a.y+b.y)/2-11);
  } else if (drawing.type==='callout') {
    const text=String(drawing.text||'Callout'),box=calloutGeometry(a,text,geo);ctx.beginPath();ctx.arc(a.x,a.y,3,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(box.elbow.x,box.elbow.y);ctx.stroke();ctx.font='10px Inter, sans-serif';ctx.fillStyle='rgba(26,34,48,.96)';ctx.fillRect(box.left,box.top,box.width,box.height);ctx.strokeRect(box.left,box.top,box.width,box.height);ctx.fillStyle='#d8e2f2';ctx.textBaseline='middle';ctx.fillText(text,box.left+7,box.top+box.height/2);
  } else if (drawing.type==='position') drawEditablePosition(drawing,bars,range,geo,selected);
  if (selected && !state.drawingsLocked && !drawing.locked && !['position','rectangle'].includes(drawing.type)) { for (const point of [a,b]) drawControlPoint(point); }
  ctx.restore();
}

function drawDrawings(bars,range,geo) {
  for (const drawing of state.drawings) drawOneDrawing(drawing,bars,range,geo,false);
  if (state.drawingDraft?.a) {
    if(['long-position','short-position'].includes(state.drawingDraft.type)){const side=state.drawingDraft.type==='long-position'?'LONG':'SHORT',entry=state.drawingDraft.a.price,stop=(state.drawingDraft.preview||state.drawingDraft.a).price,risk=Math.abs(entry-stop)||entry*.005,take=side==='LONG'?entry+risk*2:entry-risk*2;drawEditablePosition({type:'position',side,entry,stop,take,riskPct:1,accountSize:state.account?.balance,a:state.drawingDraft.a,endTime:shiftWallTime(state.drawingDraft.a.time,(TIMEFRAME_MS[state.timeframe]||60000)*12),color:'#9fc0ff',lineWidth:1,preview:true},bars,range,geo,false);}
    else drawOneDrawing({id:'draft',type:state.drawingDraft.type,a:state.drawingDraft.a,b:state.drawingDraft.preview||state.drawingDraft.a,color:'#9fc0ff'},bars,range,geo,true);
  }
}

function drawCrosshair(bars,range,geo) {
  if (state.hoverIndex===null||!state.hoverPoint||!bars.length) return;
  const x=Math.max(geo.main.left,Math.min(geo.main.right,state.hoverPoint.x)), y=state.hoverPoint.y;
  ctx.save(); ctx.strokeStyle='#475263'; ctx.globalAlpha=.7; ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(x,geo.main.top); ctx.lineTo(x,geo.volume.bottom); ctx.moveTo(geo.main.left,y); ctx.lineTo(geo.main.right,y); ctx.stroke(); ctx.restore();
  if(state.hoverPoint.snapField){const snapX=Number.isFinite(state.hoverPoint.snapX)?state.hoverPoint.snapX:x;ctx.save();ctx.fillStyle='#080a0e';ctx.strokeStyle='#4b91ff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(snapX,y,5,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#4b91ff';ctx.beginPath();ctx.arc(snapX,y,1.5,0,Math.PI*2);ctx.fill();ctx.restore();}
  drawAxisPrice(y,price(priceForY(y,range,geo.main)),'#3c4656',geo);
}

function drawChart() {
  const w=canvas.clientWidth,h=canvas.clientHeight; ctx.clearRect(0,0,w,h); ctx.fillStyle='#080a0e'; ctx.fillRect(0,0,w,h);
  const bars=visibleBars(); if (!bars.length) {$('drawingEditBar')?.classList.add('hidden');return;}
  const geo=plotGeometry(), range=currentPriceRange(bars); drawGrid(bars,range,geo); drawCandles(bars,range,geo); drawTradeOverlays(bars,range,geo); drawDrawings(bars,range,geo); drawCrosshair(bars,range,geo);drawTradeMarkers(bars,range,geo);updateDrawingEditBar(bars,range,geo);
}

function resizeCanvas() {
  const rect=$('chartWrap').getBoundingClientRect(), dpr=Math.max(1,Math.min(window.devicePixelRatio||1,2));
  canvas.width=Math.max(1,Math.round(rect.width*dpr)); canvas.height=Math.max(1,Math.round(rect.height*dpr)); canvas.style.width=`${rect.width}px`; canvas.style.height=`${rect.height}px`; ctx.setTransform(dpr,0,0,dpr,0,0); drawChart();
}

function setTool(tool) {
  state.tool=tool; state.drawingDraft=null; state.pendingPositionSide=null;state.hoverIndex=null;state.hoverPoint=null;state.hoverDrawingId=null;
  $('crosshairTooltip').classList.add('hidden');
  document.querySelectorAll('.tool[data-tool]').forEach(button=>button.classList.toggle('active',button.dataset.tool===tool));
  const hints={trend:'点击起点，再点击终点；按住 Shift 画水平线',ray:'点击起点，再点击方向点',rectangle:'点击两个对角',vertical:'点击放置垂直线','horizontal-ray':'点击价格放置水平射线',ruler:'点击两个点测量价格与 K 线数','long-position':'先点 Entry，再点 Stop Loss','short-position':'先点 Entry，再点 Stop Loss',callout:'点击图表后输入标注'};
  if (hints[tool]) { $('drawingHint').textContent=hints[tool]; $('drawingHint').classList.remove('hidden'); } else $('drawingHint').classList.add('hidden');
  canvas.style.cursor=tool==='cursor'?'crosshair':'cell'; drawChart();
}

function finishTwoPointTool(point,event) {
  const draft=state.drawingDraft;
  if (!draft) { state.drawingDraft={type:state.tool,a:point,preview:point}; return; }
  const end={...point}; if (state.tool==='trend'&&event.shiftKey) end.price=draft.a.price;
  if (state.tool==='long-position'||state.tool==='short-position') {
    const side=state.tool==='long-position'?'LONG':'SHORT';
    const valid=side==='LONG'?end.price<draft.a.price:end.price>draft.a.price;
    if (!valid) { setStatus(`${side} 的 Stop 必须在 Entry ${side==='LONG'?'下方':'上方'}`, 'error'); state.drawingDraft=null; drawChart(); return; }
    const risk=Math.abs(draft.a.price-end.price),take=side==='LONG'?draft.a.price+risk*2:draft.a.price-risk*2;
    addDrawing({type:'position',side,entry:draft.a.price,stop:end.price,take,riskPct:1,accountSize:state.account?.balance||10000,orderType:'limit',a:draft.a,b:end,endTime:shiftWallTime(draft.a.time,(TIMEFRAME_MS[state.timeframe]||60000)*18),status:'plan',color:side==='LONG'?'#17bda0':'#ef4e60'});
    state.drawingDraft=null;state.selectedDrawingId=null;setTool('cursor');setStatus(`${side} Position 计划已创建；悬停后点击中间的 ＋ 可载入右侧模拟交易`,'ok');return;
  }
  addDrawing({type:state.tool,a:draft.a,b:end}); state.drawingDraft=null; setTool('cursor');
}

function handleChartClick(event) {
  if (!state.episode) return openStartDialog();
  if (state.tool==='cursor') { const rect=canvas.getBoundingClientRect(),hit=hitTestDrawing(event.clientX-rect.left,event.clientY-rect.top); state.selectedDrawingId=hit?.id||null;if(hit?.part==='activate'){const drawing=selectedDrawing();if(drawing)loadTradeTicketFromDrawing(drawing);}drawChart(); return; }
  const point=canvasPointFromEvent(event); if (!point) return;
  if (state.tool==='vertical') { addDrawing({type:'vertical',a:point,b:point}); setTool('cursor'); return; }
  if (state.tool==='horizontal-ray') { addDrawing({type:'horizontal-ray',a:point,b:point}); setTool('cursor'); return; }
  if (state.tool==='callout') { state.pendingCalloutPoint=point; $('calloutText').value=''; $('calloutDialog').showModal(); return; }
  finishTwoPointTool(point,event);
}

function distanceToSegment(px,py,a,b) {
  const dx=b.x-a.x,dy=b.y-a.y; if (!dx&&!dy) return Math.hypot(px-a.x,py-a.y); const t=Math.max(0,Math.min(1,((px-a.x)*dx+(py-a.y)*dy)/(dx*dx+dy*dy))); return Math.hypot(px-(a.x+t*dx),py-(a.y+t*dy));
}

function hitTestDrawing(x,y) {
  const bars=visibleBars(),geo=plotGeometry(),range=currentPriceRange(bars);
  for (const drawing of [...state.drawings].reverse()) {
    if (!drawing.a) continue; const a=pointPixels(drawing.a,bars,range,geo),b=pointPixels(drawing.b||drawing.a,bars,range,geo);
    if (drawing.type==='position') {
      const p=positionPixels(drawing,bars,range,geo),left=Math.max(geo.main.left,p.x1),right=Math.min(geo.main.right,p.x2);if(right<left)continue;const handles=positionHandles({...p,x1:left,x2:right});
      if (x>=left-8&&x<=right+8) {
        const action=positionActionPoint({...p,x1:left,x2:right});if(positionIsPlan(drawing)&&Math.hypot(x-action.x,y-action.y)<12)return{id:drawing.id,part:'activate'};
        for (const part of ['take','entry','stop','width']) if (Math.hypot(x-handles[part].x,y-handles[part].y)<11) return {id:drawing.id,part};
        if (y>=Math.min(p.takeY,p.stopY)-6&&y<=Math.max(p.takeY,p.stopY)+6) return {id:drawing.id,part:'body'};
      }
      continue;
    }
    if(drawing.type==='rectangle'){
      if(state.selectedDrawingId===drawing.id)for(const [name,point] of Object.entries(rectangleHandles(a,b)))if(Math.hypot(x-point.x,y-point.y)<10)return{id:drawing.id,part:`rect-${name}`};
      const left=Math.min(a.x,b.x),right=Math.max(a.x,b.x),top=Math.min(a.y,b.y),bottom=Math.max(a.y,b.y);if(x>=left-5&&x<=right+5&&y>=top-5&&y<=bottom+5)return{id:drawing.id,part:'body'};continue;
    }
    if (Math.hypot(x-a.x,y-a.y)<9) return {id:drawing.id,part:'a'};
    if (drawing.type!=='vertical'&&drawing.type!=='callout'&&Math.hypot(x-b.x,y-b.y)<9) return {id:drawing.id,part:'b'};
    if (drawing.type==='vertical'&&Math.abs(x-a.x)<7) return {id:drawing.id,part:'body'};
    if (drawing.type==='callout'){const box=calloutGeometry(a,drawing.text,geo);if(Math.hypot(x-a.x,y-a.y)<28||(x>=box.left-4&&x<=box.left+box.width+4&&y>=box.top-4&&y<=box.top+box.height+4))return{id:drawing.id,part:'body'};}
    if (['trend','ray','horizontal-ray','ruler'].includes(drawing.type)) {
      const extended=drawing.type==='ray'?rayEndPoint(a,b,geo):{...b};if(drawing.type==='horizontal-ray'){extended.x=geo.main.right;extended.y=a.y;}
      if(distanceToSegment(x,y,a,extended)<7)return {id:drawing.id,part:'body'};
    }
  }
  return null;
}

function selectedDrawing() { return state.drawings.find(drawing=>drawing.id===state.selectedDrawingId)||null; }

function updateDrawingEditBar(bars=visibleBars(),range=currentPriceRange(bars),geo=plotGeometry()) {
  const bar=$('drawingEditBar'),drawing=selectedDrawing();
  if (!drawing||!bars.length) {bar.classList.add('hidden');return;}
  $('drawingColor').value=drawing.color||'#7fa8f2';bar.style.setProperty('--drawing-color',drawing.color||'#7fa8f2');$('drawingWidth').value=String(drawing.lineWidth||2);document.querySelectorAll('#colorPalette [data-color]').forEach(button=>button.classList.toggle('active',button.dataset.color===(drawing.color||'#7fa8f2')));
  $('drawingLockBtn').classList.toggle('active',Boolean(drawing.locked));$('drawingLockBtn').title=drawing.locked?'解锁当前绘图':'锁定当前绘图';
  let anchor=pointPixels(drawing.a,bars,range,geo);if(drawing.type==='position'){const p=positionPixels(drawing,bars,range,geo);anchor={x:(p.x1+p.x2)/2,y:Math.min(p.entryY,p.stopY,p.takeY)};}else if(drawing.b){const b=pointPixels(drawing.b,bars,range,geo);anchor={x:(anchor.x+b.x)/2,y:Math.min(anchor.y,b.y)};}
  bar.classList.remove('hidden');const width=bar.offsetWidth||170;bar.style.left=`${Math.max(6,Math.min(canvas.clientWidth-width-6,anchor.x-width/2))}px`;bar.style.top=`${Math.max(42,Math.min(canvas.clientHeight-44,anchor.y-46))}px`;
}

function shiftDrawingPoint(point,deltaTime,deltaPrice) {
  return point?{...point,time:shiftWallTime(point.time,deltaTime),price:Number(point.price)+deltaPrice}:point;
}

function positionPricesChanged(drawing,original) {
  return ['entry','stop','take'].some(key=>{
    const current=Number(drawing?.[key]),previous=Number(original?.[key]);
    return !Number.isFinite(current)||!Number.isFinite(previous)||Math.abs(current-previous)>Math.max(1e-9,Math.abs(previous)*1e-10);
  });
}

function moveDrawingInteraction(event,x,y) {
  const interaction=state.drawingInteraction,drawing=selectedDrawing();if(!interaction||!drawing)return;
  const bars=visibleBars(),geo=plotGeometry(),range=interaction.range,step=TIMEFRAME_MS[state.timeframe]||60000,deltaTime=Math.round((x-interaction.startX)/Math.max(1e-9,barSlotWidth(bars,geo.main)))*step;
  const currentPrice=priceForY(y,range,geo.main),deltaPrice=currentPrice-interaction.startPrice,original=interaction.original;
  if (drawing.type==='position') {
    const pricePart=['take','stop','entry','body'].includes(interaction.part),snappedPoint=pricePart?canvasPointFromEvent(event,{snap:state.magnetMode}):null,dragPrice=snappedPoint?.price??currentPrice;
    if(snappedPoint?.snapField)state.hoverPoint={x,y:snappedPoint.y,snapX:snappedPoint.x,snapField:snappedPoint.snapField};
    const gap=Math.max((range.max-range.min)/Math.max(1,geo.main.height)*2,Math.abs(Number(drawing.entry))*1e-8,1e-8),isLong=drawing.side==='LONG';
    if (interaction.part==='take') drawing.take=isLong?Math.max(dragPrice,Number(drawing.entry)+gap):Math.min(dragPrice,Number(drawing.entry)-gap);
    else if (interaction.part==='stop') drawing.stop=isLong?Math.min(dragPrice,Number(drawing.entry)-gap):Math.max(dragPrice,Number(drawing.entry)+gap);
    else if (interaction.part==='entry'&&drawing.status!=='open') {
      const lower=(isLong?Number(drawing.stop):Number(drawing.take))+gap,upper=(isLong?Number(drawing.take):Number(drawing.stop))-gap;
      if(lower<upper)drawing.entry=Math.max(lower,Math.min(upper,dragPrice));
    }
    else if (interaction.part==='width') {
      const point=canvasPointFromEvent(event,{snap:false}),minimumEnd=wallTime(drawing.a.time)+step*4;
      if(point)drawing.endTime=wallTextFromMs(Math.max(minimumEnd,wallTime(point.time)));
    }
    else if (interaction.part==='body') {
      drawing.a=shiftDrawingPoint(original.a,deltaTime,0);drawing.endTime=shiftWallTime(original.endTime||shiftWallTime(original.a.time,step*18),deltaTime);
      if(drawing.status!=='open'){const snappedDelta=dragPrice-interaction.startPrice;drawing.entry=Number(original.entry)+snappedDelta;drawing.stop=Number(original.stop)+snappedDelta;drawing.take=Number(original.take)+snappedDelta;}
    }
    return;
  }
  const point=canvasPointFromEvent(event,{snap:state.magnetMode});
  if(drawing.type==='rectangle'&&interaction.part.startsWith('rect-')&&point){
    const key=interaction.part.slice(5),aTime=wallTime(original.a.time),bTime=wallTime(original.b.time);let leftTime=Math.min(aTime,bTime),rightTime=Math.max(aTime,bTime),topPrice=Math.max(Number(original.a.price),Number(original.b.price)),bottomPrice=Math.min(Number(original.a.price),Number(original.b.price));
    if(key.includes('w'))leftTime=wallTime(point.time);if(key.includes('e'))rightTime=wallTime(point.time);if(key.includes('n'))topPrice=point.price;if(key.includes('s'))bottomPrice=point.price;
    const lo=Math.min(leftTime,rightTime),hi=Math.max(leftTime,rightTime),low=Math.min(topPrice,bottomPrice),high=Math.max(topPrice,bottomPrice);drawing.a={...original.a,time:wallTextFromMs(lo),price:high};drawing.b={...original.b,time:wallTextFromMs(hi),price:low};return;
  }
  if ((interaction.part==='a'||interaction.part==='b')&&point) drawing[interaction.part]={...point};
  else if (interaction.part==='body') {drawing.a=shiftDrawingPoint(original.a,deltaTime,deltaPrice);drawing.b=shiftDrawingPoint(original.b,deltaTime,deltaPrice);}
}

async function syncEditedPosition(drawing,original) {
  if (!state.episode||drawing.status==='closed') return;
  const valid=drawing.side==='LONG'?drawing.stop<drawing.entry&&drawing.take>drawing.entry:drawing.take<drawing.entry&&drawing.stop>drawing.entry;
  if(!valid){Object.assign(drawing,original);saveDrawings();drawChart();return setStatus(`${drawing.side} 的 Stop / Entry / Target 关系无效`,'error');}
  if(!drawing.orderId&&!drawing.tradeId){saveDrawings();drawChart();return setStatus('Position 计划已更新到本地；点击中间的 ＋ 才会载入交易面板','ok');}
  const body={order_id:drawing.orderId||undefined,trade_id:drawing.tradeId||undefined,timeframe:state.timeframe,limit_price:drawing.entry,stop_loss:drawing.stop,take_profit:drawing.take,account_size:drawing.accountSize,risk_pct:drawing.riskPct,limit_fee_rate:LIMIT_FEE,market_fee_rate:MARKET_FEE,market_slippage_rate:MARKET_SLIPPAGE};
  try {const data=await post(`/api/episodes/${state.episode.id}/update-order`,body);state.events.push(...(data.events||[]));state.activeLimitOrders=data.active_limit_orders||[];state.activeTrades=data.active_trades||[];syncPositionDrawings();renderActivePlans();saveDrawings();drawChart();setStatus('Position 价格已更新，Replay 将按新价位执行','ok');}
  catch(error){Object.assign(drawing,original);saveDrawings();drawChart();setStatus(error.message,'error');}
}

function latestReplayPrice() {
  if(Number.isFinite(Number(state.executionPrice))&&Number(state.executionPrice)>0)return Number(state.executionPrice);
  const bar=state.bars[state.bars.length-1];return Number(bar?.close||bar?.open||0);
}

function setTicketOrderType(type) {
  const value=type==='market'?'market':'limit';$('entryType').value=value;
  document.querySelectorAll('[data-ticket-order]').forEach(button=>button.classList.toggle('active',button.dataset.ticketOrder===value));
  updateRiskPreview();
}

function loadTradeTicket(side='LONG',entry=null,stop=null,drawing=null,{scroll=true}={}) {
  const reference=Number(entry)||latestReplayPrice(),stopValue=Number(stop),hasStop=Number.isFinite(stopValue)&&stopValue>0,riskDistance=hasStop?Math.abs(reference-stopValue):reference*.005;
  const normalizedSide=side==='SHORT'?'SHORT':'LONG',resolvedStop=hasStop?stopValue:(normalizedSide==='LONG'?reference-riskDistance:reference+riskDistance),resolvedTake=drawing?Number(drawing.take):(normalizedSide==='LONG'?reference+riskDistance*2:reference-riskDistance*2);
  state.pendingPositionSide=normalizedSide;state.pendingPositionDrawingId=drawing?.id||null;state.pendingPositionPoints=drawing?{entry:drawing.a,stop:drawing.b}:null;
  $('positionAccount').value=Number(drawing?.accountSize||state.account?.balance||10000).toFixed(2);state.ticketAccountOverridden=false;$('riskPct').value=String(drawing?.riskPct||1);$('entryPrice').value=Number(reference||0).toFixed(4);$('stopPrice').value=Number(resolvedStop||0).toFixed(4);$('takePrice').value=Number(resolvedTake||0).toFixed(4);
  $('tradeTicketSource').textContent=drawing?`Position · ${normalizedSide}`:'手动输入';setTicketOrderType(drawing?.orderType||'limit');updateRiskPreview();
  if(scroll)$('tradeTicket').scrollIntoView({behavior:'smooth',block:'start'});
}

function loadTradeTicketFromDrawing(drawing) {
  loadTradeTicket(drawing.side,Number(drawing.entry),Number(drawing.stop),drawing);setStatus('Position 参数已载入右侧；确认后点击开多或开空才会提交','ok');
}

function useLatestTradePrice() {
  const next=latestReplayPrice(),previous=Number($('entryPrice').value);if(!Number.isFinite(next)||next<=0)return;
  const delta=Number.isFinite(previous)?next-previous:0;$('entryPrice').value=next.toFixed(4);$('stopPrice').value=(Number($('stopPrice').value)+delta).toFixed(4);$('takePrice').value=(Number($('takePrice').value)+delta).toFixed(4);updateRiskPreview();
}

function syncMarketTicketToExecution() {
  if($('entryType').value==='market'&&!state.pendingPositionDrawingId)useLatestTradePrice();else updateRiskPreview();
}

function updateRiskPreview() {
  const side=state.pendingPositionSide||'LONG',account=Number($('positionAccount').value),riskPct=Number($('riskPct').value),typedEntry=Number($('entryPrice').value),stop=Number($('stopPrice').value),take=Number($('takePrice').value),type=$('entryType').value;
  const causalEntry=type==='market'?latestReplayPrice():typedEntry,metrics=plannedTradeMetrics({side,account,riskPct,entry:causalEntry,stop,take,orderType:type});
  $('plannedRisk').textContent=money(metrics.riskAmount);$('plannedRisk').title='1R 只按 Entry 到原始 SL 的价格距离计算，不含手续费和滑点';$('stopNetLoss').textContent=Number.isFinite(metrics.stopNetLoss)?money(metrics.stopNetLoss):'—';$('stopNetLoss').title=Number.isFinite(metrics.costOverrun)?`比 1R 额外支出 ${money(metrics.costOverrun)}`:'';$('estimatedQty').textContent=Number.isFinite(metrics.qty)?`${metrics.qty.toFixed(4)} ETH`:'— ETH';$('targetRR').textContent=Number.isFinite(metrics.rr)?metrics.rr.toFixed(2):'—';
  $('targetRR').title='净目标收益除以纯价格风险 1R；手续费从最终净收益额外扣除';
  $('executionPreview').textContent=type==='market'?`Replay 1m Open ${price(causalEntry)} · 预计成交 ${price(metrics.entryExec)}（${side==='LONG'?'+':'−'}0.02% 滑点）`:`限价挂单 ${price(typedEntry)} · 无滑点`;
}

function validatePositionTicket(side) {
  const specs=[
    ['positionAccount','Account Size',value=>value>0],
    ['riskPct','Risk',value=>value>=.01&&value<=100],
    ['entryPrice','Entry',value=>value>0],
    ['stopPrice','Stop Loss',value=>value>0],
    ['takePrice','Take Profit',value=>value>0],
  ];
  const values={};
  for(const [id,label,isValid] of specs){const input=$(id),value=Number(input.value);if(!Number.isFinite(value)||!isValid(value)){setStatus(`${label} 输入无效，请检查数值`,'error');input.focus({preventScroll:true});return null;}values[id]=value;}
  const type=$('entryType').value,rawEntry=type==='market'?latestReplayPrice():values.entryPrice,entry=type==='market'?rawEntry*(side==='LONG'?1+MARKET_SLIPPAGE:1-MARKET_SLIPPAGE):rawEntry,stop=values.stopPrice,take=values.takePrice,validBracket=side==='LONG'?stop<entry&&take>entry:take<entry&&stop>entry;
  if(!validBracket){setStatus(`${side} 需要 ${side==='LONG'?'Stop < 预计成交价 < Take Profit':'Take Profit < 预计成交价 < Stop'}`,'error');$('stopPrice').focus({preventScroll:true});return null;}
  return {account:values.positionAccount,riskPct:values.riskPct,entry:rawEntry,stop,take};
}

async function placePosition(side,event) {
  stopPlay();
  event?.preventDefault();if(!state.episode)return openStartDialog();
  if(state.episode.status!=='active'){updateTradeTicketAvailability();return setStatus('本次训练已结束；请开始新训练后再下单','error');}
  if(state.loading)return;
  state.pendingPositionSide=side;
  const ticket=validatePositionTicket(side);if(!ticket)return;
  const orderType=$('entryType').value,{entry,stop,take,account,riskPct}=ticket;
  const sourceDrawing=state.pendingPositionDrawingId?state.drawings.find(d=>d.id===state.pendingPositionDrawingId&&positionIsPlan(d)):null;
  const anchor=sourceDrawing?.a||{time:state.episode.cursor_time,price:entry};
  const body={side,timeframe:state.timeframe,order_type:orderType,stop_loss:stop,take_profit:take,account_size:account,risk_pct:riskPct,limit_fee_rate:LIMIT_FEE,market_fee_rate:MARKET_FEE,market_slippage_rate:MARKET_SLIPPAGE,entry_context:{anchor_time:anchor.time,source:sourceDrawing?'position_plan':'trade_ticket'}};
  body.entry_context.setup = readSetupDraft();
  if (orderType==='limit') body.limit_price=entry;
  state.loading=true;document.querySelectorAll('[data-open-side]').forEach(button=>button.disabled=true);setStatus(`正在提交 ${side} ${orderType==='limit'?'限价挂单':'市价单'}…`,'busy');
  try {
    const data=await post(`/api/episodes/${state.episode.id}/trade`,body); state.events.push(...(data.events||[])); state.activeLimitOrders=data.active_limit_orders||[]; state.activeTrades=data.active_trades||[]; state.tradeSummary=data.trade_summary||state.tradeSummary;
    const activeTrade=[...state.activeTrades].reverse().find(t=>(t.payload||{}).side===side),linked={side,entry,stop,take,riskPct,accountSize:account,orderType,orderId:data.order_id||null,tradeId:activeTrade?.trade_id||null,status:data.status==='pending'?'pending':'open',color:side==='LONG'?'#17bda0':'#ef4e60'};
    if(sourceDrawing)Object.assign(sourceDrawing,linked);else addDrawing({type:'position',...linked,a:anchor,b:{...anchor,price:stop},endTime:shiftWallTime(anchor.time,(TIMEFRAME_MS[state.timeframe]||60000)*18)});
    syncPositionDrawings();
    state.pendingPositionDrawingId=null;state.pendingPositionPoints=null;$('tradeTicketSource').textContent=`已提交 · ${side}`;saveDrawings();setTool('cursor');renderActivePlans();drawChart();
    setStatus(data.status==='pending'?`${side} 挂单已放置，等待 K 线触发`:`${side} 已按市价成交`, 'ok');
  } catch(error) { setStatus(error.message,'error'); }
  finally {state.loading=false;updateTradeTicketAvailability();}
}

function renderActivePlans() {
  const host=$('activePlans'), items=[];
  for (const order of state.activeLimitOrders) {
    const p=order.payload||{},side=p.side||'',cls=side==='SHORT'?'short':'long';
    items.push(`<article class="order-card ${cls}"><div class="order-top"><strong class="${cls}">${escapeHtml(side)} · LIMIT</strong><small>等待成交</small></div><div class="order-levels"><div><span>Entry</span><b>${price(order.price)}</b></div><div><span>Stop</span><b>${price(p.stop_loss)}</b></div><div><span>Target</span><b>${price(p.take_profit)}</b></div></div><button data-cancel-order="${escapeHtml(p.order_id||'')}">撤销挂单</button></article>`);
  }
  for (const trade of state.activeTrades) {
    const p=trade.payload||{},side=p.side||'',cls=side==='SHORT'?'short':'long';
    items.push(`<article class="order-card ${cls}"><div class="order-top"><strong class="${cls}">${escapeHtml(side)} · OPEN</strong><small>${Number(p.quantity||0).toFixed(4)} ETH</small></div><div class="order-levels"><div><span>Entry</span><b>${price(p.entry_price)}</b></div><div><span>Stop</span><b>${price(p.current_stop_loss??p.initial_stop_loss)}</b></div><div><span>Target</span><b>${price(p.current_take_profit??p.initial_take_profit)}</b></div></div><span class="auto-exit-note">Replay 自动止盈 / 止损 · 禁止手动平仓</span></article>`);
  }
  $('openCount').textContent=String(items.length); host.className=items.length?'card-list':'card-list empty-list'; host.innerHTML=items.length?items.join(''):'还没有挂单或持仓。可从 Position 计划的 ＋ 载入，或直接使用上方模拟交易。';updateEndEpisodeAvailability();
}

async function cancelOrder(orderId) {
  try { const data=await post(`/api/episodes/${state.episode.id}/cancel-order`,{order_id:orderId}); state.events.push(data.event); state.activeLimitOrders=data.active_limit_orders||[]; state.activeTrades=data.active_trades||[]; renderActivePlans(); drawChart(); setStatus('挂单已撤销','ok'); }
  catch(error){setStatus(error.message,'error');}
}

async function endEpisode() {
  if (!state.episode||state.episode.status!=='active') return;
  try { const data=await post(`/api/episodes/${state.episode.id}/close`,{}); state.episode=data.episode; state.tradeSummary=data.summary; await refreshSnapshot(); stopPlay(); setStatus('本次训练已结束并生成汇总','ok'); }
  catch(error){setStatus(error.message,'error');}
}

function exportEpisode() { if (state.episode) window.open(`/api/episodes/${state.episode.id}/export`,'_blank','noopener'); }

function deleteSelectedDrawing() {
  const drawing=selectedDrawing();if(!drawing)return;
  if(state.drawingsLocked)return setStatus('请先关闭全局绘图锁','error');
  state.drawings=state.drawings.filter(item=>item.id!==drawing.id);state.selectedDrawingId=null;saveDrawings();drawChart();setStatus('选中的绘图已删除','ok');
}

canvas.addEventListener('pointerdown',event=>{
  $('colorPalette').classList.add('hidden');
  const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top,bars=visibleBars(),range=currentPriceRange(bars),hit=state.tool==='cursor'?hitTestDrawing(x,y):null;
  const geo=plotGeometry();state.pointer={x,y,startX:x,startY:y,startOffset:state.visibleOffset,startRange:{...range},startPrice:priceForY(y,range,geo.main),axisScaling:!state.autoScale&&x>geo.main.right};
  if(hit){state.selectedDrawingId=hit.id;const drawing=selectedDrawing(),fixedOpenEntry=drawing?.type==='position'&&drawing.status==='open'&&hit.part==='entry';if(fixedOpenEntry)setStatus('该 Position 已成交，Entry 价格已固定；仍可拖动 TP、SL 或右侧宽度','ok');else if(hit.part!=='activate'&&!state.drawingsLocked&&!drawing?.locked){state.drawingInteraction={id:hit.id,part:hit.part,startX:x,startY:y,startPrice:state.pointer.startPrice,range:{...range},original:JSON.parse(JSON.stringify(drawing))};}state.panning=false;}
  else state.panning=state.tool==='cursor'&&!state.pointer.axisScaling;
  canvas.setPointerCapture?.(event.pointerId);drawChart();
});
canvas.addEventListener('pointermove',event=>{
  const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top,bars=visibleBars(),geo=plotGeometry(); state.hoverPoint={x,y};
  const hoverHit=state.tool==='cursor'?hitTestDrawing(x,y):null;state.hoverDrawingId=hoverHit?.id||state.drawingInteraction?.id||null;
  if(state.tool==='cursor')canvas.style.cursor=hoverHit?.part==='activate'?'pointer':hoverHit?'move':'crosshair';
  if(state.drawingInteraction)moveDrawingInteraction(event,x,y);
  if(state.pointer?.axisScaling){const start=state.pointer.startRange,anchor=state.pointer.startPrice,factor=Math.exp(Math.max(-1.4,Math.min(1.4,(y-state.pointer.startY)*.008))),span=(start.max-start.min)*factor,ratio=(anchor-start.min)/(start.max-start.min);state.manualPriceRange={min:anchor-span*ratio,max:anchor+span*(1-ratio)};}
  if (state.panning&&state.pointer) { const dx=x-state.pointer.startX,dy=y-state.pointer.startY; if (Math.abs(dx)>3) {state.visibleOffset=state.pointer.startOffset+Math.round(dx/Math.max(1e-9,barSlotWidth(bars,geo.main)));visibleBars();}if(!state.autoScale&&Math.abs(dy)>3){const span=state.pointer.startRange.max-state.pointer.startRange.min,shift=dy/Math.max(1,geo.main.height)*span;state.manualPriceRange={min:state.pointer.startRange.min+shift,max:state.pointer.startRange.max+shift};} }
  const hoverBars=visibleBars(),rawIndex=Math.floor((x-geo.main.left)/Math.max(1e-9,barSlotWidth(hoverBars,geo.main))-(hoverBars._startSlot||0));
  if (hoverBars.length&&x>=geo.main.left&&x<=geo.main.right&&rawIndex>=0&&rawIndex<hoverBars.length) state.hoverIndex=rawIndex; else state.hoverIndex=null;
  if (state.drawingDraft) { const point=canvasPointFromEvent(event); if(point){state.hoverPoint={x,y:point.y,snapX:point.x,snapField:point.snapField};state.drawingDraft.preview=state.tool==='trend'&&event.shiftKey?{...point,price:state.drawingDraft.a.price}:point;} }
  const bar=state.hoverIndex===null?null:hoverBars[state.hoverIndex],tip=$('crosshairTooltip'),showTip=bar&&state.tool==='cursor'&&!state.drawingInteraction; if(showTip){updateLegend(bar);tip.textContent=`${chartTime(bar)}  O ${price(bar.open)}  H ${price(bar.high)}  L ${price(bar.low)}  C ${price(bar.close)}  Vol ${compactNumber(bar.volume)}`; tip.style.left=`${Math.min(canvas.clientWidth-390,x+14)}px`; tip.style.top=`${Math.max(70,Math.min(canvas.clientHeight-34,y+14))}px`; tip.classList.remove('hidden');}else tip.classList.add('hidden');
  drawChart();
});
canvas.addEventListener('pointerup',async event=>{
  const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top,moved=state.pointer&&Math.hypot(x-state.pointer.startX,y-state.pointer.startY)>5,interaction=state.drawingInteraction;
  state.panning=false;state.drawingInteraction=null;state.pointer=null;if(state.hoverPoint)state.hoverPoint={...state.hoverPoint,snapField:null};
  if(interaction){const drawing=state.drawings.find(item=>item.id===interaction.id);if(moved){saveDrawings();const pricesChanged=drawing?.type==='position'&&positionPricesChanged(drawing,interaction.original);if(pricesChanged&&drawing.status!=='closed'&&interaction.part!=='width')await syncEditedPosition(drawing,interaction.original);else{drawChart();setStatus(drawing?.type==='position'?(pricesChanged?'已结束的 Position 仅更新本地绘图':'Position 显示位置已保存；交易价格没有变化'):'绘图位置已保存到本地','ok');}}else drawChart();return;}
  if(!moved)handleChartClick(event);
  const bars=visibleBars(),maxOffset=Math.max(0,state.bars.length-Math.max(30,state.visibleCount));if(bars[0]?.time===state.bars[0]?.time&&state.visibleOffset>=maxOffset-1)loadMoreHistory();
});
canvas.addEventListener('pointerleave',()=>{state.hoverIndex=null;state.hoverPoint=null;state.hoverDrawingId=null;$('crosshairTooltip').classList.add('hidden');updateLegend(state.bars[state.bars.length-1]);drawChart();});
canvas.addEventListener('wheel',event=>{event.preventDefault();const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top,geo=plotGeometry(),bars=visibleBars();if(!state.autoScale&&(event.altKey||x>geo.main.right)){const range=currentPriceRange(bars),anchor=priceForY(y,range,geo.main),factor=Math.exp(Math.max(-.8,Math.min(.8,event.deltaY*.0015))),newSpan=(range.max-range.min)*factor,ratio=(anchor-range.min)/(range.max-range.min);state.manualPriceRange={min:anchor-newSpan*ratio,max:anchor+newSpan*(1-ratio)};}else{const delta=Math.max(8,Math.round(state.visibleCount*.1));state.visibleCount=Math.max(30,Math.min(5000,state.visibleCount+(event.deltaY>0?delta:-delta)));visibleBars();}savePreferences();drawChart();},{passive:false});

document.querySelectorAll('.tool[data-tool]').forEach(button=>button.addEventListener('click',()=>setTool(button.dataset.tool)));
$('magnetToggle').addEventListener('click',()=>{const index=MAGNET_MODES.indexOf(state.magnetMode);state.magnetMode=MAGNET_MODES[(index+1)%MAGNET_MODES.length];updateMagnetButton();savePreferences();const message={weak:'弱磁已开启：靠近 K 线 OHLC 时吸附',strong:'强磁已开启：始终吸附当前 K 线最近的 OHLC',off:'磁铁已关闭：自由价格'}[state.magnetMode];setStatus(message,'ok');});
$('lockToggle').addEventListener('click',()=>{state.drawingsLocked=!state.drawingsLocked;$('lockToggle').classList.toggle('active',state.drawingsLocked);$('lockToggle').setAttribute('aria-pressed',String(state.drawingsLocked));savePreferences();drawChart();setStatus(state.drawingsLocked?'全部绘图已锁定':'全部绘图已解锁','ok');});
$('trashBtn').addEventListener('click',()=>{if(state.drawingsLocked)return setStatus('请先关闭全局绘图锁','error');const count=state.drawings.length;state.drawings=[];state.selectedDrawingId=null;saveDrawings();drawChart();setStatus(count?`已一次删除 ${count} 个绘图`:'当前没有绘图','ok');});
$('autoScaleBtn').addEventListener('click',()=>{const bars=visibleBars();if(state.autoScale)state.manualPriceRange=automaticPriceRange(bars);state.autoScale=!state.autoScale;if(state.autoScale)state.manualPriceRange=null;updateAutoScaleButton();savePreferences();drawChart();setStatus(state.autoScale?'自动坐标已开启：纵向移动与缩放已锁定':'自动坐标已关闭：可上下拖动，价格轴滚轮可纵向缩放','ok');});
$('drawingColor').addEventListener('input',event=>{const drawing=selectedDrawing();if(!drawing)return;drawing.color=event.target.value;$('drawingEditBar').style.setProperty('--drawing-color',drawing.color);saveDrawings();drawChart();});
$('colorPaletteBtn').addEventListener('click',event=>{event.stopPropagation();$('colorPalette').classList.toggle('hidden');});
$('colorPalette').addEventListener('click',event=>{const button=event.target.closest('[data-color]'),drawing=selectedDrawing();if(!button||!drawing)return;drawing.color=button.dataset.color;$('drawingColor').value=drawing.color;$('colorPalette').classList.add('hidden');saveDrawings();drawChart();});
$('drawingWidth').addEventListener('change',event=>{const drawing=selectedDrawing();if(!drawing)return;drawing.lineWidth=Number(event.target.value);saveDrawings();drawChart();});
$('drawingLockBtn').addEventListener('click',()=>{const drawing=selectedDrawing();if(!drawing)return;drawing.locked=!drawing.locked;saveDrawings();drawChart();setStatus(drawing.locked?'当前绘图已锁定':'当前绘图已解锁','ok');});
$('drawingDeleteBtn').addEventListener('click',deleteSelectedDrawing);
document.querySelectorAll('#timeframeNav button').forEach(button=>button.addEventListener('click',()=>changeTimeframe(button.dataset.timeframe)));
document.querySelectorAll('.range-presets button').forEach(button=>button.addEventListener('click',()=>{state.visibleCount=Number(button.dataset.visible);state.visibleOffset=0;savePreferences();drawChart();}));
$('fitChartBtn').addEventListener('click',()=>{state.visibleCount=Math.min(240,state.bars.length||220);state.visibleOffset=0;state.autoScale=true;state.manualPriceRange=null;updateAutoScaleButton();savePreferences();drawChart();});
$('returnToCurrentBtn').addEventListener('click', returnToCurrentTime);
$('chartWrap').addEventListener('pointermove', event => {
  const button = $('returnToCurrentBtn'), rect = button.getBoundingClientRect();
  const near = event.clientX >= rect.left - 64 && event.clientX <= rect.right + 40 && event.clientY >= rect.top - 64 && event.clientY <= rect.bottom + 40;
  button.classList.toggle('is-near', near && !state.panning && !state.drawingInteraction && !state.drawingDraft);
});
$('chartWrap').addEventListener('pointerleave', () => $('returnToCurrentBtn').classList.remove('is-near'));
$('newSessionBtn').addEventListener('click',openStartDialog);$('emptyStartBtn').addEventListener('click',openStartDialog);$('replayModeBtn').addEventListener('click',()=>state.episode?$('replayBar').scrollIntoView({behavior:'smooth'}):openStartDialog());
$('startMode').addEventListener('change',()=>{const random=$('startMode').value==='random',continuing=$('startMode').value==='continue';$('randomRows').classList.toggle('hidden',!random);$('startDatetimeRow').classList.toggle('hidden',random||continuing);});
$('createEpisodeBtn').addEventListener('click',createEpisode);
$('symbol').addEventListener('change',async()=>{await loadHealth();setStatus('Symbol 将在新训练中生效','ok');});
$('playBtn').addEventListener('click',()=>state.playTimer?stopPlay():startPlay());$('stepBtn').addEventListener('click',()=>{stopPlay();step(timeframeMinutes());});$('jumpStepBtn').addEventListener('click',()=>{stopPlay();step();});$('rewindBtn').addEventListener('click',()=>rewind(timeframeMinutes()));$('stepSize').addEventListener('change',()=>{$('stepSizeLabel').textContent=$('stepSize').selectedOptions[0].textContent;});
$('positionForm').addEventListener('input',event=>{if(event.target===$('positionAccount'))state.ticketAccountOverridden=true;updateRiskPreview();});$('positionForm').addEventListener('submit',event=>event.preventDefault());
document.querySelectorAll('[data-ticket-order]').forEach(button=>button.addEventListener('click',()=>{setTicketOrderType(button.dataset.ticketOrder);if(button.dataset.ticketOrder==='market')useLatestTradePrice();}));
$('useMarketPriceBtn').addEventListener('click',useLatestTradePrice);document.querySelectorAll('[data-open-side]').forEach(button=>button.addEventListener('click',event=>placePosition(button.dataset.openSide,event)));
$('saveCalloutBtn').addEventListener('click',event=>{event.preventDefault();const text=$('calloutText').value.trim();if(!text||!state.pendingCalloutPoint)return;addDrawing({type:'callout',a:state.pendingCalloutPoint,b:state.pendingCalloutPoint,text,color:'#e0ad59'});state.pendingCalloutPoint=null;$('calloutDialog').close();setTool('cursor');setStatus('Callout 已保存到本地','ok');});
document.querySelectorAll('[data-dialog-close]').forEach(button=>button.addEventListener('click',()=>{const dialog=button.closest('dialog');if(dialog?.open)dialog.close('cancel');}));
document.querySelectorAll('dialog.modal').forEach(dialog=>dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close('cancel');}));
$('calloutDialog').addEventListener('cancel',()=>{state.pendingCalloutPoint=null;if(state.tool==='callout')setTool('cursor');});
$('calloutDialog').addEventListener('close',()=>{if($('calloutDialog').returnValue==='cancel'){state.pendingCalloutPoint=null;if(state.tool==='callout')setTool('cursor');}});
$('activePlans').addEventListener('click',event=>{const cancel=event.target.closest('[data-cancel-order]');if(cancel)cancelOrder(cancel.dataset.cancelOrder);});
$('applyAccountBtn').addEventListener('click',()=>{const value=Number($('accountInput').value);if(!Number.isFinite(value)||value<=0)return setStatus('Account Size 必须大于 0','error');state.account.balance=value;state.account.initialBalance=value-state.account.realizedPnl;syncAccountInputs(null,{forceTicket:true});saveAccount();renderAccount();setStatus('当前 Account Size 已更新；下一笔仓位会自动使用它','ok');});
$('resetStatsBtn').addEventListener('click',()=>{const current=Number($('accountInput').value)||state.account.balance;state.account={initialBalance:current,balance:current,realizedPnl:0,totalFees:0,totalSlippage:0,trades:[],processedTradeIds:closedTradeEvents().map(e=>String(e.payload?.trade_id||e.id))};syncAccountInputs(null,{forceTicket:true});saveAccount();renderAccount();setStatus('账户统计已重置；SQLite 交易历史未删除','ok');});
$('exportBtn').addEventListener('click',exportEpisode);$('endEpisodeBtn').addEventListener('click',endEpisode);
window.addEventListener('resize',resizeCanvas);
function handleReplayKey(event) {
  if (event.key === 'Escape') {
    stopPlay();
    const dialog = document.querySelector('dialog[open]');
    if (dialog) { event.preventDefault(); dialog.close('cancel'); return; }
    state.drawingDraft = null; setTool('cursor'); return;
  }
  if (event.repeat || event.ctrlKey || event.metaKey || event.altKey || document.querySelector('dialog[open]') || ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName) || document.activeElement?.isContentEditable) return;
  if ((event.key === 'Delete' || event.key === 'Backspace') && state.selectedDrawingId && !state.drawingsLocked && !selectedDrawing()?.locked) { event.preventDefault(); deleteSelectedDrawing(); }
  if (event.code === 'Space') { event.preventDefault(); state.playTimer ? stopPlay() : startPlay(); }
  if (event.key === 'ArrowRight') { event.preventDefault(); stopPlay(); step(timeframeMinutes()); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); rewind(timeframeMinutes()); }
}
window.addEventListener('keydown', handleReplayKey);
document.addEventListener('visibilitychange', () => { if (document.hidden) stopPlay(); });
$('pauseOnEvent').addEventListener('change', () => { state.pauseOnEvent = $('pauseOnEvent').checked; savePreferences(); });
$('playSpeed').addEventListener('change', savePreferences);
$('chartTimezone').addEventListener('change', () => { state.chartTimezone = $('chartTimezone').value; savePreferences(); updateEpisodeUi(); drawChart(); });
$('tradeTicket').addEventListener('focusin', stopPlay);

const SETUP_FIELDS = ['breakoutLiquidity','retestLiquidity','trendLiquidity','smcLiquidity','setupNotes'];
function readSetupDraft() {
  return {method: 'InterEquity manual liquidity thesis', ...Object.fromEntries(SETUP_FIELDS.map(id => [id, $(id).value])), recorded_cursor: state.episode?.cursor_time || null};
}
function renderSetupReview(setup) {
  if (!setup) return '';
  const labels = {breakoutLiquidity:'Breakout',retestLiquidity:'Break & retest',trendLiquidity:'Trend line',smcLiquidity:'SMC'};
  const statuses = {unassessed:'待判断',swept:'判断已扫',remaining:'判断仍有',not_applicable:'不适用'};
  const rows = Object.entries(labels).map(([id,label]) => `${label}：${statuses[setup[id]] || '待判断'}`).join(' · ');
  return `<details class="setup-review"><summary>入场前计划</summary><p>${escapeHtml(rows)}</p><p>${escapeHtml(setup.setupNotes || '未填写理由')}</p></details>`;
}
function loadSetupDraft() {
  let draft = {};
  try { draft = JSON.parse(localStorage.getItem(`coinbacktest.replay.setup.${state.episode?.id || 'draft'}`) || '{}'); } catch (_) { /* use defaults */ }
  for (const id of SETUP_FIELDS) $(id).value = draft[id] || (id === 'setupNotes' ? '' : 'unassessed');
}
$('setupChecklist').addEventListener('input', () => {
  localStorage.setItem(`coinbacktest.replay.setup.${state.episode?.id || 'draft'}`, JSON.stringify(readSetupDraft()));
});

async function init() {
  $('colorPalette').innerHTML=PRESET_COLORS.map(color=>`<button type="button" data-color="${color}" style="--swatch:${color}" title="${color}" aria-label="颜色 ${color}"></button>`).join('');
  loadPreferences(); loadAccount();$('positionAccount').value=Number(state.account.balance||10000).toFixed(2);resizeCanvas();
  try { await loadHealth(); const restored=await restoreEpisode(); if(!restored){$('emptyState').classList.remove('hidden');drawChart();} }
  catch(error){setStatus(`无法读取本地 Replay 服务：${error.message}`,'error');}
}

init();
