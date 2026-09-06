const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');

function harness(count) {
  const start=Date.UTC(2026,7,1);
  const bars=Array.from({length:count},(_,i)=>({time:new Date(start+i*60000).toISOString().slice(0,19).replace('T',' '),open:100,high:101,low:99,close:100}));
  bars._slots=count;bars._startSlot=0;
  const geo={main:{left:4,right:1004,width:1000,top:0,bottom:200,height:200}};
  const range={min:90,max:110};
  const scope={state:{timeframe:'1m',magnetMode:'weak'},TIMEFRAME_MS:{'1m':60000},MAGNET_WEAK_THRESHOLD_PX:12,
    canvas:{getBoundingClientRect:()=>({left:50,top:25})},visibleBars:()=>bars,plotGeometry:()=>geo,currentPriceRange:()=>range};
  vm.createContext(scope);
  vm.runInContext(source.slice(source.indexOf('function wallTime('),source.indexOf('async function request(')),scope);
  vm.runInContext(source.slice(source.indexOf('function yForPrice('),source.indexOf('function drawGrid(')),scope);
  vm.runInContext(source.slice(source.indexOf('function pointPixels('),source.indexOf('function rayEndPoint(')),scope);
  return {scope,bars,geo,range};
}

for (const count of [500,1000,5000]) {
  for (const magnet of ['weak','strong','off']) {
    test(`${count} visible bars / ${magnet}: placed and restored anchor stays under the mouse`,()=>{
      const {scope,bars,geo,range}=harness(count);
      scope.state.magnetMode=magnet;
      const index=Math.floor(count*.8),x=scope.xForIndex(index,bars,geo.main);
      const point=scope.canvasPointFromEvent({clientX:x+50,clientY:125});
      assert.equal(point.index,index);
      assert.equal(point.time,bars[index].time);
      // Saved drawings render from time, not their old pixel x coordinate.
      const restored=JSON.parse(JSON.stringify(point));delete restored.x;
      const rendered=scope.pointPixels(restored,bars,range,geo);
      assert.ok(Math.abs(rendered.x-x)<.001,`anchor jumped from ${x} to ${rendered.x}`);
    });
  }
}

test('dense chart with left padding and right blank space uses the rendered slots',()=>{
  const {scope,bars,geo}=harness(1000);
  bars._slots=5000;bars._startSlot=2000;
  const index=750,x=scope.xForIndex(index,bars,geo.main);
  const point=scope.canvasPointFromEvent({clientX:x+50,clientY:125});
  assert.equal(point.time,bars[index].time);
  assert.equal(point.index,index);
  const blankX=scope.xForIndex(bars.length+100,bars,geo.main);
  const blank=scope.canvasPointFromEvent({clientX:blankX+50,clientY:125});
  assert.equal(blank.snapField,null);
  assert.ok(Math.abs(scope.xForTime(blank.time,bars,geo.main)-blankX)<.001);
});
