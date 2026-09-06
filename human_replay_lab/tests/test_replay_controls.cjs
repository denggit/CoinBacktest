// Runtime regressions for the browser controller, using isolated DOM/timer doubles.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');

function priceRangeHarness() {
  const scope = {state:{autoScale:true,manualPriceRange:null,drawings:[],activeLimitOrders:[],activeTrades:[]}};
  vm.createContext(scope);
  vm.runInContext(source.slice(source.indexOf('function automaticPriceRange('),source.indexOf('function updateAutoScaleButton(')),scope);
  return scope;
}

test('drawings and order levels do not expand the visible candle scale',()=>{
  const scope=priceRangeHarness(),bars=[{low:2900,high:3000},{low:2980,high:3050}];
  const before=scope.currentPriceRange(bars);
  scope.state.drawings=[{type:'horizontal-ray',a:{price:2800}},{type:'trend',a:{price:3100},b:{price:3200}},{type:'position',entry:3020,stop:2700,take:3400}];
  scope.state.activeLimitOrders=[{price:2750,payload:{stop_loss:2500,take_profit:3500}}];
  scope.state.activeTrades=[{price:3020,payload:{initial_stop_loss:2600,initial_take_profit:3600}}];
  const after=scope.currentPriceRange(bars);
  assert.equal(after.min,before.min);assert.equal(after.max,before.max);
  assert.equal(after.min,2888);assert.equal(after.max,3062);
});

test('autoscale follows the visible window while manual scale stays fixed',()=>{
  const scope=priceRangeHarness();
  const first=scope.currentPriceRange([{low:100,high:110}]);
  const next=scope.currentPriceRange([{low:200,high:210}]);
  assert.ok(Math.abs(next.min-first.min-100)<1e-9);assert.ok(Math.abs(next.max-first.max-100)<1e-9);
  scope.state.autoScale=false;scope.state.manualPriceRange={min:50,max:500};
  const manual=scope.currentPriceRange([{low:200,high:210}]);
  assert.equal(manual.min,50);assert.equal(manual.max,500);
});

function harness() {
  const elements = new Map();
  const timers = new Map();
  let nextTimer = 0;
  const state = {episode:{id:'test',status:'active'},timeframe:'30m',loading:false,playTimer:null,playGeneration:0,pauseOnEvent:true,events:[]};
  const scope = {
    state, document:{querySelector:()=>null},
    $:id=>{if(!elements.has(id))elements.set(id,{value:id==='playSpeed'?'1000':'5',classList:{add(){},remove(){}}});return elements.get(id);},
    setTimeout:fn=>{timers.set(++nextTimer,fn);return nextTimer;},clearTimeout:id=>timers.delete(id),
    timeframeMinutes:()=>30, setStatus(){}, money:String,
    post:async()=>({episode:{id:'test',status:'active'},updates:{},active_trades:[],active_limit_orders:[],advanced_minutes:30}),
    appendBars(){},syncPositionDrawings(){},updateEpisodeUi(){},syncAccountFromEvents(){},renderActivePlans(){},syncMarketTicketToExecution(){},drawChart(){},
    savePreferences(){},updateTimeframeButtons(){},refreshSnapshot:async()=>{},
  };
  vm.createContext(scope);
  vm.runInContext(source.slice(source.indexOf('async function step('),source.indexOf('async function rewind(')),scope);
  vm.runInContext(source.slice(source.indexOf('function startPlay()'),source.indexOf('function chartTime(')),scope);
  return {scope,state,timers,elements};
}

test('autoplay pauses after a fill and forwards the event-pause preference',async()=>{
  const {scope,state,timers}=harness();
  let sent;
  scope.post=async(url,body)=>{sent=body;return {episode:state.episode,advanced_minutes:11,paused_on_event:true,trade_events:[{event_type:'ORDER_FILLED'}]};};
  scope.startPlay();
  await timers.get(state.playTimer)();
  assert.equal(sent.pause_on_event,true);
  assert.equal(state.playTimer,null);
  assert.equal(timers.size,0);
  assert.equal(state.events.length,1);
});

test('data boundary stops an active episode from polling forever',async()=>{
  const {scope,state,timers}=harness();
  scope.post=async()=>({episode:state.episode,at_data_end:true,advanced_minutes:0});
  scope.startPlay();await timers.get(state.playTimer)();
  assert.equal(state.playTimer,null);
});

test('pause during an in-flight request cannot restart the old playback loop',async()=>{
  const {scope,state,timers}=harness();
  let resolve;
  scope.post=()=>new Promise(done=>resolve=done);
  scope.startPlay();const pending=timers.get(state.playTimer)();
  scope.stopPlay();scope.startPlay();
  assert.equal(state.playTimer,null); // busy start is ignored
  resolve({episode:state.episode});await pending;
  assert.equal(timers.size,0);
  assert.equal(state.loading,false);
});

test('timeframe changes wait for a pending step and preserve the current bars',async()=>{
  const {scope,state}=harness();state.loading=true;
  let requested=false;scope.refreshSnapshot=async()=>requested=true;
  await scope.changeTimeframe('1m');
  assert.equal(state.timeframe,'30m');assert.equal(requested,false);
});

test('failed timeframe fetch restores the previous timeframe',async()=>{
  const {scope,state}=harness();scope.refreshSnapshot=async()=>{throw new Error('offline');};
  await scope.changeTimeframe('1m');
  assert.equal(state.timeframe,'30m');assert.equal(state.loading,false);
});

test('dialogs prevent hidden playback shortcuts from starting playback',()=>{
  const {scope,state}=harness();scope.document.querySelector=()=>({open:true});
  scope.startPlay();assert.equal(state.playTimer,null);
});
