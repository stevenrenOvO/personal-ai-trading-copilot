// Run the shipped script in a minimal DOM with stubbed HTTP; no live server/trades.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {webcrypto} = require('node:crypto');

const html = fs.readFileSync(path.join(__dirname, '../backend/dashboard/static/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace(/\s*\(async function init\(\) \{[\s\S]*?\}\)\(\);\s*$/, '');

function makePage(storage = new Map()) {
  const elements = new Map();
  class Element {
    constructor(id = '', tag = 'div', attrs = '') {
      this.id = id; this.tagName = tag.toUpperCase(); this.value = ''; this.disabled = false;
      this.textContent = ''; this.className = ''; this.attrs = {}; this._html = '';
      this.dataset = {tab: attrs.match(/data-tab="([^"]+)"/)?.[1]};
      const classes = new Set((attrs.match(/class="([^"]+)"/)?.[1] || '').split(' '));
      this.classList = {add: c => classes.add(c), remove: c => classes.delete(c), contains: c => classes.has(c)};
    }
    set innerHTML(value) {
      this._html = value;
      if (this.tagName === 'SELECT' && value.includes('<option')) this.value = value.match(/value="([^"]*)"/)[1];
    }
    get innerHTML() { return this._html; }
    setAttribute(key, value) { this.attrs[key] = value; }
    removeAttribute(key) { delete this.attrs[key]; }
    addEventListener() {}
    prepend(el) { elements.set(el.id, el); }
    click() { return this.onclick?.(); }
  }
  for (const match of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    elements.set(match[3], new Element(match[3], match[1], match[2]));
  }
  const tabs = Array.from(html.matchAll(/<button (data-tab="[^"]+"[^>]*)>/g),
    match => new Element('', 'button', match[1]));
  const sections = [...elements.values()].filter(el => el.tagName === 'SECTION');
  const document = {
    hidden:false,
    getElementById:id => elements.get(id),
    createElement:tag => new Element('', tag),
    querySelector:selector => selector === '#tabs button.active' ? tabs.find(el => el.classList.contains('active')) : null,
    querySelectorAll:selector => {
      if (selector === '#tabs button') return tabs;
      if (selector === '.section') return sections;
      if (selector === '.form-row label') return [];
      const prefix = selector.match(/^\[id\^="([^"]+)"\]$/)?.[1];
      return prefix ? [...elements.values()].filter(el => el.id.startsWith(prefix)) : [];
    },
  };
  let responder = () => { throw new Error('Unstubbed HTTP request'); };
  const context = vm.createContext({document, window:{addEventListener() {}}, console,
    sessionStorage:{getItem:key => storage.get(key) || null, setItem:(key, value) => storage.set(key, value), removeItem:key => storage.delete(key)},
    crypto:webcrypto, Uint8Array, AbortController, setTimeout, clearTimeout, setInterval() {},
    fetch:(url, opts) => responder(url, opts),
  });
  vm.runInContext(script, context, {filename:'dashboard.js'});
  elements.get('date').value = '20261013';
  return {elements, setFetch:fn => {responder = fn;}, run:code => vm.runInContext(code, context), storage};
}
const response = (data, status = 200) => ({ok:status < 400, status, json:async () => data});
const wrapped = data => response({status:'ok', data});
const deferred = () => { let resolve; const promise = new Promise(r => {resolve = r;}); return {promise, resolve}; };
const flush = () => new Promise(resolve => setImmediate(resolve));
function reviewReply(url, trades = []) {
  if (url === '/trades') return wrapped(trades);
  if (url === '/profile') return wrapped({stats:{}});
  if (url === '/positions') return wrapped({positions:{}});
  return wrapped({findings:[]});
}

async function decision_fields() {
  const page = makePage(); const {elements:e} = page;
  const values = {'pt-qty':'9900', 'dj-code':'000001.sz', 'dj-action':'WAIT', 'dj-date':'2026-09-18', 'dj-qty':'7', 'dj-price':'', 'dj-reason':'等待数据'};
  for (const [id, value] of Object.entries(values)) e.get(id).value = value;
  let submitted;
  page.setFetch((url, opts) => {
    if (opts.method === 'POST') { submitted = JSON.parse(opts.body); return wrapped({trade_id:'test'}); }
    return reviewReply(url);
  });
  await e.get('dj-submit').onclick(); await flush();
  assert.equal(submitted.trade_date, '20260918');
  assert.equal(submitted.quantity, 7);
  assert.equal(submitted.price, null);
  assert.equal(submitted.ts_code, '000001.SZ');
  assert.match(submitted.request_id, /^decision_[a-f0-9]{32}$/);
  assert.match(e.get('dj-result').innerHTML, /2026-09-18/);
}

async function decision_retry() {
  let page = makePage(); let e = page.elements;
  const values = {'dj-code':'000001.SZ', 'dj-action':'WAIT', 'dj-date':'20260918', 'dj-qty':'', 'dj-price':'', 'dj-reason':'先观察'};
  for (const [id, value] of Object.entries(values)) e.get(id).value = value;
  let first;
  page.setFetch((url, opts) => {first = JSON.parse(opts.body); throw new Error('response lost');});
  await e.get('dj-submit').onclick();
  assert.match(e.get('dj-result').innerHTML, /结果尚未确认/);
  assert.equal(e.get('dj-submit').disabled, false);
  // A page reload must retain the same id for an uncertain write.
  page = makePage(page.storage); e = page.elements;
  for (const [id, value] of Object.entries(values)) e.get(id).value = value;
  let second;
  page.setFetch((url, opts) => {
    if (opts.method === 'POST') { second = JSON.parse(opts.body); return wrapped({trade_id:'test'}); }
    return reviewReply(url);
  });
  await e.get('dj-submit').onclick(); await flush();
  assert.equal(first.request_id, second.request_id);
  assert.equal(page.storage.size, 0);
}

async function stale_date() {
  const page = makePage(); const {elements:e} = page;
  const pending = new Map();
  page.setFetch(url => {const item = deferred(); pending.set(url, item); return item.promise;});
  const older = page.run("loadMarket('20261013')");
  e.get('date').value = '20261014';
  const newer = page.run("loadMarket('20261014')");
  for (const [url, item] of pending) if (url.includes('20261014')) item.resolve(wrapped({state:'新日期结果', emotion_cycle:'新阶段'}));
  await newer;
  for (const [url, item] of pending) if (url.includes('20261013')) item.resolve(wrapped({state:'旧日期结果', emotion_cycle:'旧阶段'}));
  await older;
  assert.match(e.get('market-grid').innerHTML, /新日期结果/);
  assert.doesNotMatch(e.get('market-grid').innerHTML, /旧日期结果/);
  assert.match(e.get('emotion-detail').innerHTML, /新阶段/);
  assert.equal(e.get('market').attrs['aria-busy'], 'false');
}

async function audit_failure() {
  const page = makePage(); const {elements:e} = page;
  const trades = Array.from({length:7}, (_, i) => ({trade_id:String(i), ts_code:'000001.SZ', strategy_id:i === 0 ? '<img src=x onerror=alert(1)>' : 'manual'}));
  let inFlight = 0; let maxInFlight = 0; const requests = [];
  page.setFetch(url => {
    if (!url.endsWith('/audit')) return reviewReply(url, trades);
    inFlight++; maxInFlight = Math.max(maxInFlight, inFlight);
    const item = deferred(); requests.push(item);
    return item.promise.finally(() => {inFlight--;});
  });
  const loaded = page.run('loadReview()'); await flush();
  assert.equal(requests.length, 4);
  for (const item of requests.slice()) item.resolve(response({detail:'审计服务失败'}, 503));
  await flush();
  assert.equal(requests.length, 7);
  for (const item of requests.slice(4)) item.resolve(wrapped({findings:[]}));
  await loaded;
  assert.equal(maxInFlight, 4);
  assert.match(e.get('review-audit').innerHTML, /4 条审计未完成/);
  assert.doesNotMatch(e.get('review-audit').innerHTML, /本次审计范围内未发现问题/);
  assert.doesNotMatch(e.get('review-trades').innerHTML, /<img src=x/);
  assert.match(e.get('review-trades').innerHTML, /&lt;img/);
}

async function failure_status() {
  const page = makePage(); const {elements:e} = page;
  page.setFetch(() => response({detail:'数据暂不可用'}, 503));
  await page.run("loadMarket('20261013')");
  assert.match(e.get('net-status').textContent, /数据加载失败/);
  assert.match(e.get('market-grid').innerHTML, /数据获取失败/);
  page.setFetch(() => response({status:'degraded', version:'1.1', checks:{history:{status:'warning', message:'历史数据待补齐'},
    ingest:{status:'warning',message:'历史数据正在补齐',progress:{processed:200,total:5000,downloaded:190,failed:10}}}}));
  await page.run('loadSystemStatus()');
  assert.match(e.get('system-label').textContent, /部分数据待完善/);
  assert.match(e.get('system-details').innerHTML, /历史数据待补齐/);
  assert.match(e.get('system-details').innerHTML, /已检查 200\/5000 个代码，已更新 190，失败待重试 10/);
  assert.match(e.get('net-status').textContent, /数据加载失败/);
}

async function date_selection_during_refresh() {
  const page = makePage(); const {elements:e} = page;
  const pending = new Map();
  page.setFetch(url => {const item = deferred(); pending.set(url, item); return item.promise;});
  const loading = page.run('initDates()');
  e.get('date').value = '20261012';
  pending.get('/data/latest').resolve(wrapped({date:'20261013', phase:'盘后'}));
  pending.get('/data/history/dates').resolve(wrapped({dates:['20261012', '20261013']}));
  await loading;
  assert.equal(e.get('date').value, '20261012');
  assert.equal(e.get('pt-date').value, '20261013');
  assert.equal(e.get('dj-date').value, '20261012');
}

async function refresh_waits_for_view() {
  const page = makePage(); const {elements:e} = page;
  const wait = deferred();
  page.setFetch(url => {
    if (url === '/ready') return response({status:'ready', version:'1.1', checks:{}});
    if (url === '/data/latest') return wrapped({date:'20261013'});
    if (url === '/data/history/dates') return wrapped({dates:['20261013']});
    return wait.promise;
  });
  const refresh = e.get('refresh').onclick(); await flush();
  assert.equal(e.get('refresh').disabled, true);
  wait.resolve(wrapped({})); await refresh;
  assert.equal(e.get('refresh').disabled, false);
}

async function unconfirmed_market() {
  const page = makePage(); const {elements:e} = page;
  e.get('date').value = '20261023';
  page.setFetch(url => url.startsWith('/market/')
    ? response({status:'ok', data:{state:'数据准备中', total_count:null, score:50,
        advance_count:null, decline_count:null, flat_count:null, limit_up_count:null,
        limit_down_count:null, confidence:0, recommended_exposure:0.5},
        meta:{source:'tencent', status:'ok', quote_date:'20261022', quote_time:'16:15:00', session_phase:'盘前'}})
    : wrapped({emotion_cycle:'数据不足'}));
  await page.run("loadMarket('20261023')");
  assert.match(e.get('market-scope').innerHTML, /所选日期行情未确认/);
  assert.doesNotMatch(e.get('market-scope').innerHTML, /数据正常/);
  assert.doesNotMatch(e.get('market-grid').innerHTML, /null|undefined|50%|>50<|>0%/);
  assert.match(e.get('market-grid').innerHTML, /待确认\/待确认\/待确认/);
  assert.match(e.get('market-grid').innerHTML, /市场参考仓位<\/div><div class="value ">待确认/);
}

async function health_without_samples() {
  const page = makePage(); const {elements:e} = page;
  const baseline = {strategy_id:'ma_cross', win_rate:0, max_drawdown:0, recent_20_trades:[], recent_50_trades:[]};
  page.setFetch(() => wrapped([baseline]));
  await page.run('loadHealth()');
  assert.match(e.get('health-table').innerHTML, /样本不足/);
  assert.doesNotMatch(e.get('health-table').innerHTML, /0\.0%/);
  page.setFetch(() => wrapped([{...baseline, recent_20_trades:[-1], recent_50_trades:[-1]}]));
  await page.run('loadHealth()');
  assert.match(e.get('health-table').innerHTML, /0\.0%/);
}

async function available_dates() {
  const page = makePage(); const {elements:e} = page;
  e.get('date').value = '';
  page.setFetch(url => {
    if (url === '/data/latest') return wrapped({date:'20261019', current_trading_day:'20261023',
      phase:'盘前', latest_complete_date:'20261019', snapshot_date:'20261022',
      display_reason:'日线补数中，默认查看最近完整日', history_dates:[
        {date:'20261019', complete:true, coverage_ratio:0.99},
        {date:'20261020', complete:false, coverage_ratio:0.02},
        {date:'20261022', complete:false, coverage_ratio:0.01, has_snapshot:true},
      ]});
    if (url === '/data/history/dates') return wrapped({dates:['20261019','20261020','20261022']});
    return wrapped({});
  });
  await page.run('initDates()');
  assert.equal(e.get('date').value, '20261019');
  assert.equal(e.get('pt-date').value, '20261023');
  assert.equal(e.get('bt-end').value, '20261019');
  assert.match(e.get('date').innerHTML, /value="20261020" disabled/);
  assert.doesNotMatch(e.get('date').innerHTML, /value="20261022" disabled/);
  assert.match(e.get('date-description').textContent, /当前交易日 2026-10-23/);
  assert.match(e.get('date-description').textContent, /正在查看 2026-10-19/);
  await e.get('latest-snapshot').onclick();
  assert.equal(e.get('date').value, '20261022');
  assert.match(e.get('date-description').textContent, /日线结构待补齐/);
  assert.equal(e.get('latest-complete').hidden, false);
  await e.get('latest-complete').onclick();
  assert.equal(e.get('date').value, '20261019');
}

async function dashboard_basis() {
  const page = makePage(); const {elements:e} = page;
  e.get('date').value = '20261022';
  page.setFetch(url => {
    if (url.startsWith('/market/')) return response({status:'ok', data:{state:'中性',total_count:5217,
      advance_count:2000,decline_count:3000,limit_up_count:40,limit_down_count:5},
      meta:{quote_date:'20261022', status:'final', source:'tencent'}});
    const data = url.startsWith('/emotion/') ? {emotion_cycle:'旧日发酵',recommended_exposure:0.8}
      : url.startsWith('/sectors') ? [{name:'旧日板块', score:90}]
      : url.startsWith('/board/') ? {grade:'A'}
      : {can_attack:true, total_exposure_cap:0.8, plans:[{name:'旧日机会',tier:'A'}]};
    return response({status:'ok', data, meta:{as_of:'20261019',kind:'historical',status:'ok'}});
  });
  await page.run("loadDashboard('20261022')");
  assert.match(e.get('d-market').innerHTML, /5,217 只/);
  assert.equal(e.get('d-can').textContent, '所选日结构待确认');
  assert.equal(e.get('d-emotion').textContent, '待确认');
  assert.equal(e.get('d-board').textContent, '待确认');
  assert.equal(e.get('d-exposure').textContent, '等待确认');
  assert.doesNotMatch(e.get('d-sectors').innerHTML, /旧日板块/);
  assert.doesNotMatch(e.get('d-opps').innerHTML, /旧日机会/);
  assert.match(e.get('d-can-reason').textContent, /2026-10-19/);
}

async function refresh_new_complete_day() {
  const page = makePage(); const {elements:e} = page;
  e.get('date').value = '';
  let info = {date:'20261019',current_trading_day:'20261023',latest_complete_date:'20261019',
    history_dates:[{date:'20261019',complete:true},{date:'20261022',complete:false}]};
  page.setFetch(url => url === '/data/latest' ? wrapped(info)
    : url === '/data/history/dates' ? wrapped({dates:['20261019','20261022']}) : wrapped({}));
  await page.run('initDates()');
  assert.equal(e.get('date').value, '20261019');
  info = {...info,date:'20261022',latest_complete_date:'20261022',
    history_dates:[{date:'20261019',complete:true},{date:'20261022',complete:true}]};
  await page.run('initDates()');
  assert.equal(e.get('date').value, '20261022');
  e.get('date').value = '20261019';
  await e.get('date').onchange();
  await page.run('initDates()');
  assert.equal(e.get('date').value, '20261019');
}

async function analysis_coverage_labels() {
  const page = makePage(); const {elements:e} = page;
  e.get('date').value = '';
  page.setFetch(url => url === '/data/latest' ? wrapped({date:'20261023',current_trading_day:'20261023',latest_complete_date:'20261023',
    history_dates:[{date:'20261023',complete:true,coverage_ratio:0.998}]}) : wrapped({dates:['20261023']}));
  await page.run('initDates()');
  assert.match(e.get('date').innerHTML, /日线可分析 · 99.8%/);
  assert.match(e.get('date-description').textContent, /覆盖 99.8% · 仍有少量缺失/);
  assert.doesNotMatch(e.get('date').innerHTML, /日线完整/);
}

async function structured_ai_results() {
  const page = makePage(); const {elements:e} = page;
  e.get('ai-q').value = '最强板块';
  page.setFetch(() => wrapped({answer:'已通过 get_sector_strength 获取数据',tool_calls:['get_sector_strength'],data:[
    {name:'银行<script>alert(1)</script>',date:'20261013',score:82.6,avg_pct_chg:1.25,leader_name:'平安银行',risk:'low'}]}));
  await e.get('ai-ask').onclick();
  const output = e.get('ai-output').innerHTML;
  assert.match(output, /板块强度 · 查询结果/);
  assert.match(output, /平均涨跌幅/);
  assert.match(output, /1.25%/);
  assert.match(output, /平安银行/);
  assert.match(output, /&lt;script&gt;/);
  assert.doesNotMatch(output, /<script>|get_sector_strength/);
  page.setFetch(() => wrapped({tool_calls:['get_market_state'],data:{state:'偏强',total_count:5217,recommended_exposure:0.6}}));
  await e.get('ai-ask').onclick();
  assert.match(e.get('ai-output').innerHTML, /5,217/);
  assert.match(e.get('ai-output').innerHTML, /60.0%/);
}

async function structured_ai_empty_and_stale() {
  const page = makePage(); const {elements:e} = page;
  e.get('ai-q').value = '交易记录';
  page.setFetch(() => wrapped({tool_calls:['get_trade_history'],data:[]}));
  await e.get('ai-ask').onclick();
  assert.match(e.get('ai-output').innerHTML, /暂无匹配记录/);
  page.setFetch(() => wrapped({tool_calls:['get_emotion_state'],data:{date:'20261012',emotion_cycle:'偏弱',basis_note:'日线待确认'}}));
  await e.get('ai-ask').onclick();
  assert.match(e.get('ai-output').innerHTML, /当前结果依据 2026-10-12，仅供历史复盘/);
  page.setFetch(() => wrapped({tool_calls:['get_personal_profile'],data:{stats:{total_trades:0,win_rate:0,profit_factor:0}}}));
  await e.get('ai-ask').onclick();
  assert.match(e.get('ai-output').innerHTML, /样本不足/);
  assert.doesNotMatch(e.get('ai-output').innerHTML, /0.0%/);
}

async function readiness_index_and_demo_labels() {
  const page = makePage(); const {elements:e} = page;
  page.setFetch(() => response({status:'ready',version:'1.1',data_mode:'demo',checks:{
    indexes:{status:'ok',message:'20261023 · 3/3 项指数日线已更新'}}}));
  await page.run('loadSystemStatus()');
  assert.equal(e.get('demo-banner').hidden, false);
  assert.match(e.get('system-details').innerHTML, /指数日线/);
  assert.match(e.get('system-details').innerHTML, /3\/3/);
  page.setFetch(() => response({status:'ready',version:'1.1',data_mode:'production',checks:{}}));
  await page.run('loadSystemStatus()');
  assert.equal(e.get('demo-banner').hidden, true);
}

async function unknown_ai_trade_permission() {
  const page = makePage(); const {elements:e} = page;
  page.setFetch(() => wrapped({basis_date:'20261013',can_trade_today:null,recommended_exposure:null,emotion:'数据不足'}));
  await page.run("loadBriefing('20261013')");
  assert.match(e.get('ai-output').innerHTML, /等待当日确认/);
  assert.doesNotMatch(e.get('ai-output').innerHTML, /danger\">观望/);
  e.get('ai-q').value = '今天能不能做';
  page.setFetch(() => wrapped({tool_calls:['get_market_state'],data:{date:'20261013',score:75,can_attack:null,analysis_basis_date:'20261012'}}));
  await e.get('ai-ask').onclick();
  assert.match(e.get('ai-output').innerHTML, /等待当日确认/);
  assert.match(e.get('ai-output').innerHTML, /交易计划依据日期/);
}

const scenarios = {decision_fields, decision_retry, stale_date, audit_failure, failure_status, date_selection_during_refresh, refresh_waits_for_view, unconfirmed_market, health_without_samples, available_dates, dashboard_basis, refresh_new_complete_day, analysis_coverage_labels, structured_ai_results, structured_ai_empty_and_stale, readiness_index_and_demo_labels, unknown_ai_trade_permission};
const scenario = scenarios[process.argv[2]];
if (!scenario) throw new Error('Unknown frontend scenario');
scenario().then(() => process.stdout.write('ok\n')).catch(err => {console.error(err); process.exitCode = 1;});
