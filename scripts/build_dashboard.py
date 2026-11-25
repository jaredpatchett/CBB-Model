#!/usr/bin/env python3
"""
Build docs/dashboard.html from docs/data/latest.json -- the CBB version of
the CFB dashboard, same look, same ideas:
  Edge Board   KPIs, Bet Card (spreads + totals, grade badges, bet-small tags),
               every game with a tap-to-open breakdown (team numbers, four
               factors, tempo, rest, what moves the model's number)
  Totals       official overs/unders, edge key from the backtest
  Player Props official props first, grade, hit chance, edge, matchup
  My Tracker   your own bets (this browser), +TRK from any play
  Performance  record by market and grade, history, backtest edge tiers,
               model accuracy
The page carries all its data; no server needed (GitHub Pages).
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CBB EDGE</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Saira+Condensed:ital,wght@0,700;0,800;1,800;1,900&family=Inter:wght@400;500;600;700;800;900&family=Azeret+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#07090A;--panel:#0E1311;--panel2:#0A0E0C;--chip:#141B18;--rule:#1C2621;--rule2:#26332C;--text:#F1F5F2;--dim:#C9D3CD;--muted:#8B9A92;--muted2:#66756D;
--neon:#39F07A;--blue:#5EC8FF;--orange:#FF9F5E;--amber:#FFB020;--red:#FF6B6B;--purple:#C48CFF;--gold:#FFD24A;
--display:'Saira Condensed','Arial Narrow',sans-serif;--body:'Inter',system-ui,sans-serif;--mono:'Azeret Mono',ui-monospace,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:var(--body);font-size:14px;-webkit-font-smoothing:antialiased}
button{font-family:inherit}
.top{position:sticky;top:0;z-index:20;background:rgba(7,9,10,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--rule);display:flex;align-items:center;gap:18px;padding:0 24px;height:64px}
.brand{font-family:var(--display);font-style:italic;font-weight:900;font-size:30px;letter-spacing:.02em;display:flex;align-items:center;gap:10px;white-space:nowrap}
.brand i{width:8px;height:26px;background:var(--neon);transform:skew(-18deg);display:inline-block;box-shadow:10px 0 0 var(--neon)}.brand b{color:var(--neon);font-weight:900}
.nav{display:flex;gap:4px;overflow-x:auto}.nav button{background:none;border:0;color:var(--muted);font-weight:600;font-size:13px;padding:22px 12px;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap}
.nav button.on{color:var(--neon);border-bottom-color:var(--neon)}
.asof{margin-left:auto;font-size:11px;color:var(--muted);white-space:nowrap}
.wrap{max-width:1320px;margin:0 auto;padding:20px 24px 90px}
.demo{background:rgba(255,176,32,.12);border:1px solid rgba(255,176,32,.5);color:var(--amber);border-radius:10px;padding:10px 14px;font-size:12px;font-weight:600;margin-bottom:16px}
h2{font-family:var(--display);font-style:italic;font-weight:900;font-size:26px;letter-spacing:.03em;text-transform:uppercase;margin:0}
.sh{display:flex;align-items:baseline;gap:12px;margin:22px 0 12px;flex-wrap:wrap}.sh .sub{font-size:11px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;font-weight:700}
.sh:before{content:'';width:4px;height:22px;background:var(--neon);border-radius:2px;align-self:center}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.kpi{background:var(--panel);border:1px solid var(--rule);border-radius:14px;padding:16px 18px}
.kpi .l{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:800}.kpi .v{font-size:28px;font-weight:900;margin-top:6px}.kpi .s{font-size:11px;color:var(--muted);margin-top:4px}
.kpi .v.g{color:var(--neon)}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 12px}.tab{background:var(--chip);border:1px solid var(--rule2);color:var(--dim);border-radius:8px;padding:7px 12px;font-weight:700;font-size:12px;cursor:pointer}.tab.on{background:var(--neon);color:#06120A;border-color:var(--neon)}
.grid2{display:grid;grid-template-columns:1.25fr 1fr;gap:18px;align-items:start}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:14px;overflow:hidden}
.grp{padding:10px 14px;font-size:12px;color:var(--muted);border-bottom:1px solid var(--rule);display:flex;gap:8px;align-items:center}
.pill{font-size:10px;font-weight:900;letter-spacing:.08em;border-radius:999px;padding:3px 9px}.pill.sp{background:rgba(57,240,122,.14);color:var(--neon);border:1px solid rgba(57,240,122,.5)}.pill.tot{background:rgba(94,200,255,.14);color:var(--blue);border:1px solid rgba(94,200,255,.5)}.pill.pr{background:rgba(196,140,255,.14);color:var(--purple);border:1px solid rgba(196,140,255,.5)}
.row{display:grid;grid-template-columns:auto 1fr auto auto;gap:12px;align-items:center;padding:12px 14px;border-bottom:1px solid var(--rule);border-left:3px solid var(--purple);cursor:pointer}
.row.A{border-left-color:var(--neon)}.row.C{border-left-color:var(--red)}.row:hover{background:#111915}
.play{font-family:var(--display);font-weight:800;font-size:20px;letter-spacing:.02em}.note{font-size:11px;color:var(--muted);margin-top:2px}
.edge{font-size:19px;font-weight:900;text-align:right}.edge small{display:block;font-size:10px;color:var(--muted);font-weight:700}
.under{color:var(--blue)}.over{color:var(--orange)}.pos{color:var(--neon)}.neg{color:var(--red)}
.gb{display:inline-flex;align-items:center;gap:4px;font-weight:900;font-size:13px;border-radius:999px;padding:3px 10px;white-space:nowrap}
.gb.A{background:rgba(57,240,122,.18);color:var(--neon);border:1px solid rgba(57,240,122,.7)}.gb.B{background:rgba(196,140,255,.12);color:var(--purple);border:1px solid rgba(196,140,255,.5)}.gb.C{background:rgba(255,90,90,.12);color:var(--red);border:1px solid rgba(255,90,90,.55)}
.tag{font-size:10.5px;font-weight:800;padding:2px 8px;border-radius:999px;border:1px solid var(--rule2);background:var(--chip);color:var(--dim);white-space:nowrap}
.tag.new{background:var(--neon);color:#06120A;border-color:var(--neon)}.tag.small{color:var(--amber);border-color:rgba(255,176,32,.5)}.tag.bad{color:var(--amber);border-color:rgba(255,176,32,.5)}.tag.good{color:var(--neon);border-color:rgba(57,240,122,.45)}
.tags{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}
.trk{background:var(--chip);border:1px solid var(--rule2);color:var(--neon);border-radius:7px;padding:5px 9px;font-weight:800;font-size:11px;cursor:pointer}.trk.in{color:var(--muted)}
table{width:100%;border-collapse:collapse}th{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);text-align:left;padding:10px 12px;border-bottom:1px solid var(--rule);font-weight:800}
td{padding:10px 12px;border-bottom:1px solid var(--rule);font-size:13px}tr.click{cursor:pointer}tr.click:hover td{background:#111915}td.num,th.num{text-align:right;font-family:var(--mono);font-size:12.5px}
.bd{background:var(--panel2);border-top:1px solid var(--rule);padding:16px 18px}
.bd h4{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--neon);margin:14px 0 6px;font-weight:900}.bd h4:first-child{margin-top:0}
.cmp{display:grid;grid-template-columns:1fr 70px 70px;gap:4px 10px;font-size:12.5px}.cmp div{padding:3px 0;border-bottom:1px solid var(--rule)}.cmp .h{color:var(--muted);font-size:10px;letter-spacing:.1em;text-transform:uppercase;font-weight:800}.cmp .n{text-align:right;font-family:var(--mono)}.cmp .w{color:var(--neon);font-weight:800}
.drv{display:grid;grid-template-columns:170px 1fr 52px;gap:8px;align-items:center;font-size:12px;padding:3px 0}.bar{position:relative;height:10px;background:rgba(255,255,255,.05);border-radius:5px}.bar:after{content:'';position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:rgba(255,255,255,.25)}
.bar i{position:absolute;top:0;bottom:0;border-radius:5px}.bar i.up{background:var(--orange)}.bar i.dn{background:var(--blue)}
.p{font-size:13px;line-height:1.55;color:var(--dim);margin:6px 0}
.empty{padding:22px;color:var(--muted);font-size:13px;text-align:center}
.keyrow{display:grid;grid-template-columns:110px 1fr 70px 70px auto;gap:10px;align-items:center;padding:6px 8px;border-radius:8px;font-size:13px}.keyrow.top{background:rgba(255,210,74,.08);border:1px solid rgba(255,210,74,.4)}
.kbar{height:10px;background:rgba(255,255,255,.06);border-radius:5px;overflow:hidden}.kbar i{display:block;height:100%;background:var(--neon)}.keyrow.top .kbar i{background:var(--gold)}
.dots{display:flex;gap:3px}.dot{font-size:9px;font-weight:800;width:22px;height:18px;display:inline-flex;align-items:center;justify-content:center;border-radius:4px}.dot.u{background:rgba(57,240,122,.2);color:var(--neon)}.dot.d{background:rgba(255,90,90,.2);color:var(--red)}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:14px;padding:16px 18px;margin-bottom:14px}.panel h3{margin:0 0 4px;font-family:var(--display);font-style:italic;font-weight:900;font-size:20px;text-transform:uppercase}.pn{font-size:11.5px;color:var(--muted);margin-bottom:10px}
.mnav{display:none}
input.f{background:var(--chip);border:1px solid var(--rule2);color:var(--text);border-radius:8px;padding:7px 10px;font-size:12px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}.nav,.asof{display:none}.top{padding:0 16px}.wrap{padding:14px 12px 96px}
.mnav{display:flex;position:fixed;bottom:0;left:0;right:0;background:rgba(7,9,10,.97);border-top:1px solid var(--rule);z-index:30;overflow-x:auto}
.mnav button{flex:1;min-width:70px;background:none;border:0;color:var(--muted);font-size:11px;font-weight:700;padding:12px 4px 18px}.mnav button.on{color:var(--neon)}
.row{grid-template-columns:auto 1fr auto}.row .hide-m{display:none}.drv{grid-template-columns:110px 1fr 44px}.keyrow{grid-template-columns:1fr 60px 56px}.keyrow .kbar{display:none}.keyrow .dots{grid-column:1/-1}
.hide-m{display:none}td,th{padding:8px 6px}}
.foot{max-width:1100px;margin:24px auto 90px;padding:0 16px;font-size:11px;color:var(--muted,#8a94a6);text-align:center;line-height:1.5}
</style></head><body>
<div class="top"><div class="brand"><i></i>CBB <b>EDGE</b></div><div class="nav" id="nav"></div><div class="asof" id="asof"></div></div>
<div class="wrap" id="app"></div><div class="foot">For entertainment and information only. Not betting advice. Past results don&rsquo;t guarantee future results. 21+. Gambling problem? Call 1-800-GAMBLER.</div><div class="mnav" id="mnav"></div>
<script>
window.MODEL_DATA = __DATA__;
(function(){
var D = window.MODEL_DATA, S = {page:'edge', open:null, totShow:'all', propShow:'official', propMk:'ALL', hist:'all', perfMk:'spread'};
try { var sp = localStorage.getItem('cbb_page'); if (sp) S.page = sp; } catch(e) {}
var PAGES = [['edge','Edge Board'],['totals','Totals'],['props','Player Props'],['tracker','My Tracker'],['perf','Model Performance']];
var GM = {A:['&#128293;','Bet it'],B:['&#127922;','Degen play, not a bet'],C:['&#128683;','Don&rsquo;t play']};
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function sg(x,d){if(x==null||isNaN(x))return '&ndash;';return (x>0?'+':'')+Number(x).toFixed(d==null?1:d);}
function kick(iso){try{return new Date(iso).toLocaleString([], {weekday:'short',month:'numeric',day:'numeric',hour:'numeric',minute:'2-digit'});}catch(e){return iso;}}
var SCHOOL={};(D.games||[]).forEach(function(g){if(g.home_school)SCHOOL[g.home]=g.home_school;if(g.away_school)SCHOOL[g.away]=g.away_school;});
(D.spreads||[]).concat(D.totals||[],D.props||[]).forEach(function(p){if(p.home_cbbd)SCHOOL[p.home]=p.home_cbbd;if(p.away_cbbd)SCHOOL[p.away]=p.away_cbbd;});
function short(n){return SCHOOL[n]||n;}
function badge(g){return g?'<span class="gb '+g+'">'+GM[g][0]+' '+g+'</span>':'';}
function rec(o){if(!o||!(o.w+o.l))return '0-0';return o.w+'-'+o.l+(o.p?'-'+o.p:'');}
function gameOf(p){for(var i=0;i<D.games.length;i++){var g=D.games[i];if(g.home===p.home&&g.away===p.away&&g.kickoff===p.kickoff)return i;}return -1;}
// ---- tracker (this browser only)
function trk(){try{return JSON.parse(localStorage.getItem('cbb_trk')||'[]');}catch(e){return [];}}
function trkSave(t){try{localStorage.setItem('cbb_trk',JSON.stringify(t));}catch(e){}}
function pid(p){return [p.market,p.home,p.away,p.kickoff,p.side,p.player||''].join('|');}
function inTrk(p){return trk().some(function(x){return x.id===pid(p);});}
window.__trk=function(i,list){var p=(list==='props'?D.props:list==='totals'?D.totals:D.spreads)[i];var t=trk();if(!t.some(function(x){return x.id===pid(p);})){t.push({id:pid(p),label:label(p),grade:p.grade,market:p.market,kickoff:p.kickoff,line:p.line,price:p.price||-110,res:null,at:new Date().toISOString()});trkSave(t);}render();};
window.__res=function(id,r){var t=trk();t.forEach(function(x){if(x.id===id)x.res=x.res===r?null:r;});trkSave(t);render();};
window.__del=function(id){trkSave(trk().filter(function(x){return x.id!==id;}));render();};
function label(p){
  if(p.market==='spread'){var team=p.side==='home'?p.home:p.away;var ln=p.side==='home'?p.line:-p.line;return short(team)+' '+(ln>0?'+':'')+ln;}
  if(p.market==='total')return (p.side==='under'?'UNDER ':'OVER ')+p.line;
  return p.player+' '+(p.side==='over'?'O':'U')+p.line+' '+p.prop_market.toLowerCase();}
function trkBtn(i,list,p){return inTrk(p)?'<span class="trk in">&#10003; TRK</span>':'<button class="trk" onclick="event.stopPropagation();window.__trk('+i+',&quot;'+list+'&quot;)">+TRK</button>';}
function tags(p){var t=[];if(p.new)t.push('<span class="tag new">NEW</span>');if(p.tier)t.push('<span class="tag">'+esc(p.tier)+(p.market==='prop'?'% edge':' pts')+'</span>');
  if(p.move>0)t.push('<span class="tag small" title="The market moved toward us since the model flagged it: the best number is gone.">line moved our way &middot; bet small</span>');
  if(p.move<0)t.push('<span class="tag bad">line moved against us</span>');
  if(p.matchup)t.push('<span class="tag '+((p.mu_pts>0)?'good':(p.mu_pts<0?'bad':''))+'">vs '+esc(p.matchup)+'</span>');return t.join('');}
// ---- breakdown
function teamName(g,side){return short(side==='home'?g.home:g.away);}
function breakdown(g){
  if(!g||!g.enough_games)return '<div class="bd"><p class="p">Not enough games yet for one of these teams (the model needs 2+ this season).</p></div>';
  var H=g.home_stats||{},A=g.away_stats||{},h=teamName(g,'home'),a=teamName(g,'away');
  var rows=[['Offense (pts/100 poss)','r_adj_off',1,1],['Defense (pts/100 allowed)','r_adj_def',1,-1],['Tempo (poss/40)','r_tempo',1,0],
    ['eFG% (offense)','r_efg_o',100,1],['eFG% allowed','r_efg_d',100,-1],['Turnover rate','r_tov_o',100,-1],['Forced TO rate','r_tov_d',100,1],
    ['Off. rebound %','r_orb_o',100,1],['Off. reb % allowed','r_orb_d',100,-1],['FT rate','r_ftr_o',100,1],['3PA rate','r_three_rate_o',100,0],['Last season net rating','prior_net',1,1],['Games played','games_prior',1,0]];
  var cmp='<div class="cmp"><div class="h">Per game, recent weighted</div><div class="h n">'+esc(a)+'</div><div class="h n">'+esc(h)+'</div>'+rows.map(function(r){
    var av=A[r[1]],hv=H[r[1]];var f=function(v){return v==null?'&ndash;':(Number(v)*r[2]).toFixed(1);};
    var better=r[3]&&av!=null&&hv!=null?((av-hv)*r[3]>0?'a':(av-hv)*r[3]<0?'h':''):'';
    return '<div>'+r[0]+'</div><div class="n'+(better==='a'?' w':'')+'">'+f(av)+'</div><div class="n'+(better==='h'?' w':'')+'">'+f(hv)+'</div>';}).join('')+'</div>';
  var why=[];var tp=(A.r_tempo+H.r_tempo)/2;
  if(H.r_adj_off!=null){why.push('<b>Efficiency:</b> '+esc(a)+' scores '+A.r_adj_off.toFixed(1)+' per 100 possessions and allows '+A.r_adj_def.toFixed(1)+'; '+esc(h)+' scores '+H.r_adj_off.toFixed(1)+' and allows '+H.r_adj_def.toFixed(1)+' (adjusted for who they played). That gap per 100 trips is the core of the number.');}
  if(tp)why.push('<b>Tempo:</b> about '+tp.toFixed(1)+' possessions expected. '+(tp>70?'A fast game: more possessions means more points and more variance, which helps the underdog.':tp<65?'A slow, grinding game: fewer possessions means fewer points and less variance, which helps the favorite.':'An average pace.'));
  if(H.r_tov_o!=null){var tovEdge=(A.r_tov_d-H.r_tov_o)*100,tovEdge2=(H.r_tov_d-A.r_tov_o)*100;why.push('<b>Turnovers:</b> '+(Math.abs(tovEdge)>Math.abs(tovEdge2)?esc(a)+'&rsquo;s defense forces turnovers at '+(A.r_tov_d*100).toFixed(1)+'% against the '+esc(h)+' offense that gives it away '+(H.r_tov_o*100).toFixed(1)+'% of the time.':esc(h)+'&rsquo;s defense forces turnovers at '+(H.r_tov_d*100).toFixed(1)+'% against the '+esc(a)+' offense at '+(A.r_tov_o*100).toFixed(1)+'%.')+' Live-ball turnovers turn into easy points the other way.');}
  if(H.r_orb_o!=null){why.push('<b>Glass:</b> '+esc(h)+' grabs '+(H.r_orb_o*100).toFixed(1)+'% of its misses; '+esc(a)+' '+(A.r_orb_o*100).toFixed(1)+'%. Second chances matter most in close games.');}
  if(g.home_rest!=null){var r1=g.home_rest,r2=g.away_rest;if(Math.min(r1,r2)<=1)why.push('<b>Rest:</b> '+(r1<=1?esc(h):esc(a))+' is on a back-to-back. Tired legs show up in jump shooting late.');}
  if(g.neutral)why.push('<b>Neutral court:</b> no home-court edge.');
  var drv=function(list,lab){if(!list||!list.length)return '';var mx=Math.max.apply(null,list.map(function(d){return Math.abs(d[1]);}).concat([1]));
    return '<h4>What moves the model&rsquo;s '+lab+'</h4>'+list.map(function(d){var w=Math.abs(d[1])/mx*50,up=d[1]>0;return '<div class="drv"><span>'+esc(d[0])+'</span><span class="bar"><i class="'+(up?'up':'dn')+'" style="'+(up?'left':'right')+':50%;width:'+w.toFixed(0)+'%"></i></span><b class="'+(up?'over':'under')+'" style="text-align:right">'+sg(d[1])+'</b></div>';}).join('');};
  var plays='';[['spread_play','Spread'],['total_play','Total']].forEach(function(x){var p=g[x[0]];if(!p)return;
    var lab=x[1]==='Spread'?(p.side==='home'?h:a)+' '+(function(){var ln=p.side==='home'?g.market_spread:-g.market_spread;return (ln>0?'+':'')+ln;})():(p.side==='under'?'UNDER ':'OVER ')+g.market_total;
    plays+='<div class="p">'+(p.official?badge(p.grade):'<span class="tag">lean</span>')+' <b>'+esc(lab)+'</b> &middot; edge '+p.edge+' pts'+(p.official?'':' &middot; <i>not official: '+esc(p.why_not)+'</i>')+'</div>';});
  return '<div class="bd"><h4>The plays</h4>'+(plays||'<div class="p">No lean.</div>')+
    '<div class="p">Model: '+esc(h)+' '+sg(-g.model_margin)+' (market '+(g.market_spread==null?'&ndash;':sg(g.market_spread))+'), total '+g.model_total+' (market '+(g.market_total==null?'&ndash;':g.market_total)+'), '+esc(h)+' wins '+(g.home_win_prob*100).toFixed(0)+'%</div>'+
    '<h4>Why</h4>'+why.map(function(x){return '<p class="p">'+x+'</p>';}).join('')+'<h4>Team numbers</h4>'+cmp+drv(g.drivers_spread,'spread')+drv(g.drivers_total,'total')+'</div>';}
window.__open=function(k){S.open=S.open===k?null:k;render();};
window.__go=function(p){S.page=p;S.open=null;try{localStorage.setItem('cbb_page',p);}catch(e){}render();window.scrollTo(0,0);};
window.__set=function(k,v){S[k]=v;render();};
function playRow(p,i,list){var gi=gameOf(p),key=list+i,open=S.open===key,g=D.games[gi];
  var edgeTxt=p.market==='prop'?sg(p.edge)+'%':p.edge;
  var sub=p.market==='spread'?short(p.away)+' at '+short(p.home)+' &middot; '+kick(p.kickoff)+' &middot; model '+sg(p.model)+' &middot; cover '+(p.cover_prob?(p.cover_prob*100).toFixed(0)+'%':'')
    :p.market==='total'?short(p.away)+' at '+short(p.home)+' &middot; '+kick(p.kickoff)+' &middot; model '+p.model
    :esc(short(p.team))+' vs '+esc(short(p.opponent))+' &middot; '+kick(p.kickoff)+' &middot; proj '+p.model+' (avg '+p.avg+', '+p.minutes+' min) &middot; hits '+(p.hit_prob*100).toFixed(0)+'%';
  return '<div class="row '+p.grade+'" onclick="window.__open(&quot;'+key+'&quot;)"><div>'+badge(p.grade)+'</div><div><div class="play">'+esc(label(p))+'</div><div class="note">'+sub+'</div><div class="tags">'+tags(p)+'</div></div>'+
    '<div class="edge '+(p.market==='total'?p.side:p.market==='prop'?'pos':'pos')+'">'+edgeTxt+'<small>'+(p.market==='prop'?'edge':'pts '+(p.market==='total'?p.side:'edge'))+'</small></div><div class="hide-m">'+trkBtn(i,list,p)+'</div></div>'+
    (open?(p.market==='prop'?'<div class="bd"><p class="p">'+esc(p.player)+' is projected for <b>'+p.model+'</b> '+esc(p.prop_market.toLowerCase())+' against a line of '+p.line+' ('+(p.hit_prob*100).toFixed(0)+'% to go '+p.side+'). He averages '+p.avg+' in '+p.minutes+' minutes over '+p.games+' games, and the opponent&rsquo;s defense grades as '+esc(p.matchup)+'.</p><div class="p">'+trkBtn(i,list,p)+'</div></div>':breakdown(g)):'');}
function card(list,title,cls,arr,empty){return '<div class="card"><div class="grp"><span class="pill '+cls+'">'+title+'</span> official, bet the &#128293; A&rsquo;s <span style="margin-left:auto">'+arr.length+'</span></div>'+
  (arr.length?arr.map(function(p,i){return playRow(p,i,list);}).join(''):'<div class="empty">'+empty+'</div>')+'</div>';}
function kpis(){var R=D.record||{},M=R.markets||{};var a=0;['spread','total','prop'].forEach(function(m){var o=(M[m]||{}).grades||{};a+=0;});
  var nA=D.spreads.filter(function(p){return p.grade==='A';}).length+D.totals.filter(function(p){return p.grade==='A';}).length+D.props.filter(function(p){return p.official&&p.grade==='A';}).length;
  var all=R.all||{};return '<div class="kpis"><div class="kpi"><div class="l">&#128293; A plays today</div><div class="v g">'+nA+'</div><div class="s">'+(D.spreads.length+D.totals.length+D.props.filter(function(p){return p.official;}).length)+' official in all</div></div>'+
  '<div class="kpi"><div class="l">Games on the slate</div><div class="v">'+D.games.length+'</div><div class="s">'+D.games.filter(function(g){return g.enough_games;}).length+' with enough data</div></div>'+
  '<div class="kpi"><div class="l">Official record</div><div class="v">'+rec(all)+'</div><div class="s">'+(all.w+all.l?sg(all.units)+'u &middot; '+sg(all.roi)+'%':'tracking starts with the first graded play')+'</div></div>'+
  '<div class="kpi"><div class="l">Model accuracy</div><div class="v">'+(((D.lab||{}).accuracy||{}).margin_mae||'&ndash;')+'</div><div class="s">avg miss on the margin (market '+(((D.lab||{}).accuracy||{}).market_spread_mae||'&ndash;')+')</div></div></div>';}
function pageEdge(){
  var games=D.games.map(function(g,i){var sp=g.spread_play,tp=g.total_play,key='g'+i,open=S.open===key;
    return '<tr class="click" onclick="window.__open(&quot;'+key+'&quot;)"><td><b>'+esc(short(g.away))+'</b> at <b>'+esc(short(g.home))+'</b><div class="note">'+kick(g.kickoff)+(g.neutral?' &middot; neutral':'')+'</div></td>'+
      '<td class="num">'+(g.market_spread==null?'&ndash;':sg(g.market_spread))+'</td><td class="num">'+(g.model_spread==null?'&ndash;':sg(g.model_spread))+'</td>'+
      '<td class="num hide-m">'+(g.market_total==null?'&ndash;':g.market_total)+'</td><td class="num hide-m">'+(g.model_total==null?'&ndash;':g.model_total)+'</td>'+
      '<td>'+(sp&&sp.official?badge(sp.grade):'')+' '+(tp&&tp.official?'<span class="tag">'+(tp.side==='under'?'U':'O')+' '+tp.grade+'</span>':'')+'</td></tr>'+
      (open?'<tr><td colspan="6" style="padding:0">'+breakdown(g)+'</td></tr>':'');}).join('');
  return kpis()+'<div class="grid2"><div><div class="sh"><h2>Bet Card</h2><span class="sub">spreads &amp; totals &middot; tap for the breakdown</span></div>'+
    card('spreads','SPREADS','sp',D.spreads,'No official spreads right now.')+'<div style="height:12px"></div>'+card('totals','TOTALS','tot',D.totals,'No official totals right now.')+
    '<div class="pn" style="margin-top:10px">Moneylines are shown in the breakdown but never official (backtest: -6.1%, and the model&rsquo;s win% runs about 10 points high).</div></div>'+
    '<div><div class="sh"><h2>Slate</h2><span class="sub">every game &middot; tap to open</span></div><div class="card"><table><tr><th>Game</th><th class="num">Mkt</th><th class="num">Model</th><th class="num hide-m">Total</th><th class="num hide-m">Model</th><th>Plays</th></tr>'+games+'</table></div></div></div>';}
function edgeKey(block,side){var b=(((D.lab||{})[block]||{}).vs_open||{});if(side)b=b[side]||{};var T=(b.tiers||[]).filter(function(t){return t.n;});if(!T.length)return '<div class="pn">Run the backtest (Actions &rarr; Backtest lab) to fill this in.</div>';
  var mx=Math.max.apply(null,T.map(function(t){return Math.abs(t.roi);}).concat([1]));
  return T.map(function(t){var top=t.every&&t.roi>=10&&Object.keys(t.seasons).length>=3;return '<div class="keyrow'+(top?' top':'')+'"><span>'+(top?'&#127942; ':'')+esc(t.label)+'</span><span class="kbar"><i style="width:'+(Math.max(0,t.roi)/mx*100).toFixed(0)+'%"></i></span><b class="'+(t.roi>=0?'pos':'neg')+'" style="text-align:right">'+sg(t.roi)+'%</b><span class="note">'+t.w+'-'+t.l+'</span><span class="dots">'+Object.keys(t.seasons).sort().map(function(y){var v=t.seasons[y];return '<i class="dot '+(v>=0?'u':'d')+'" title="'+y+': '+sg(v)+'%">'+String(y).slice(2)+'</i>';}).join('')+'</span></div>';}).join('');}
function pageTotals(){var arr=S.totShow==='a'?D.totals.filter(function(p){return p.grade==='A';}):D.totals;
  return '<div class="sh"><h2>Game Totals</h2><span class="sub">official overs &amp; unders</span></div>'+
  '<div class="tabs"><button class="tab'+(S.totShow==='a'?' on':'')+'" onclick="window.__set(&quot;totShow&quot;,&quot;a&quot;)">&#128293; A plays</button><button class="tab'+(S.totShow!=='a'?' on':'')+'" onclick="window.__set(&quot;totShow&quot;,&quot;all&quot;)">All official</button></div>'+
  card('totals','TOTALS','tot',arr,'No official totals right now.')+
  '<div class="panel" style="margin-top:14px"><h3>Edge key &middot; unders</h3><div class="pn">How each edge size did in the backtest vs opening totals, by season.</div>'+edgeKey('total','under')+'</div>'+
  '<div class="panel"><h3>Edge key &middot; overs</h3>'+edgeKey('total','over')+'</div>';}
function pageProps(){var mk=S.propMk,arr=D.props.filter(function(p){return (S.propShow==='official'?p.official:true)&&(mk==='ALL'||p.prop_market===mk);});
  var mks=['ALL','Points','Rebounds','Assists','Threes'];
  return '<div class="sh"><h2>Player Props</h2><span class="sub">points &middot; rebounds &middot; assists &middot; threes</span></div>'+
   '<div class="pn" style="margin:0 0 10px">Since 11/25: Points props and props priced -150 or worse top out at &#127922; B (A&rsquo;s weren&rsquo;t beating the juice).</div>'+
  '<div class="tabs">'+['official','all'].map(function(v){return '<button class="tab'+(S.propShow===v?' on':'')+'" onclick="window.__set(&quot;propShow&quot;,&quot;'+v+'&quot;)">'+(v==='official'?'Official':'All lines')+'</button>';}).join('')+'<span style="width:10px"></span>'+
  mks.map(function(m){return '<button class="tab'+(mk===m?' on':'')+'" onclick="window.__set(&quot;propMk&quot;,&quot;'+m+'&quot;)">'+m+'</button>';}).join('')+'</div>'+
  '<div class="card"><div class="grp"><span class="pill pr">PROPS</span> '+(S.propShow==='official'?'official plays':'every line the model priced')+'<span style="margin-left:auto">'+arr.length+'</span></div>'+
  (arr.length?arr.slice(0,150).map(function(p){return playRow(p,D.props.indexOf(p),'props');}).join(''):'<div class="empty">No props posted yet. Books post college props mostly on bigger games, usually the day before.</div>')+'</div>'+
  '<div class="pn" style="margin-top:10px">Every prop line is saved and graded from the box score. Backtest (Nov 11-17): &#128293; A&rsquo;s 45-22 (+25%), B&rsquo;s and C&rsquo;s lost. One early-season week, so treat props as watch-only until the live record backs it up.</div>';}
function pageTracker(){var t=trk().slice().sort(function(a,b){return a.kickoff<b.kickoff?1:-1;});var w=0,l=0,u=0;
  t.forEach(function(x){if(x.res==='W'){w++;var pr=Number(x.price)||-110;u+=pr>0?pr/100:100/Math.abs(pr);}else if(x.res==='L'){l++;u-=1;}});
  return '<div class="sh"><h2>My Tracker</h2><span class="sub">your bets &middot; saved in this browser</span></div>'+
  '<div class="kpis"><div class="kpi"><div class="l">Record</div><div class="v">'+w+'-'+l+'</div></div><div class="kpi"><div class="l">Units</div><div class="v '+(u>=0?'g':'')+'">'+sg(u,2)+'</div></div><div class="kpi"><div class="l">Pending</div><div class="v">'+t.filter(function(x){return !x.res;}).length+'</div></div><div class="kpi"><div class="l">ROI</div><div class="v">'+(w+l?sg(u/(w+l)*100)+'%':'&ndash;')+'</div></div></div>'+
  '<div class="card" style="margin-top:14px">'+(t.length?'<table><tr><th>Play</th><th class="hide-m">When</th><th>Result</th><th></th></tr>'+t.map(function(x){var id=esc(JSON.stringify(x.id));return '<tr><td>'+badge(x.grade)+' '+esc(x.label)+'</td><td class="hide-m note">'+kick(x.kickoff)+'</td><td>'+['W','L','P'].map(function(r){return '<button class="tab'+(x.res===r?' on':'')+'" onclick=\'window.__res('+id+',"'+r+'")\'>'+r+'</button>';}).join(' ')+'</td><td><button class="trk" onclick=\'window.__del('+id+')\'>&times;</button></td></tr>';}).join('')+'</table>':'<div class="empty">Tap +TRK on any play to track it here.</div>')+'</div>';}
function pagePerf(){var R=D.record||{},M=(R.markets||{})[S.perfMk]||{},G=M.grades||{};var mk=S.perfMk;
  var grow=function(g){var o=G[g]||{},n=(o.w||0)+(o.l||0);return '<tr><td>'+badge(g)+' '+GM[g][1]+'</td><td class="num">'+rec(o)+'</td><td class="num '+((o.units||0)>=0?'pos':'neg')+'">'+(n?sg(o.units)+'u':'&ndash;')+'</td><td class="num">'+(o.roi==null?'&ndash;':sg(o.roi)+'%')+'</td><td class="num hide-m">'+(n?Math.round(o.w/n*100)+'%':'&ndash;')+'</td></tr>';};
  var hist=(R.history||[]).filter(function(h){return h.market===mk&&(S.hist==='all'||h.final_grade===S.hist);}).slice(0,60);
  var acc=(D.lab||{}).accuracy||{},pa=(D.lab||{}).props_accuracy||{},diag=D.diagnostics||{};
  return '<div class="sh"><h2>Model Performance</h2><span class="sub">official plays, graded automatically</span></div>'+
  '<div class="tabs">'+[['spread','Spreads'],['total','Totals'],['prop','Props']].map(function(m){return '<button class="tab'+(mk===m[0]?' on':'')+'" onclick="window.__set(&quot;perfMk&quot;,&quot;'+m[0]+'&quot;)">'+m[1]+'</button>';}).join('')+'</div>'+
  '<div class="panel"><h3>Record by grade</h3><div class="pn">'+(R.since?'Tracked since '+new Date(R.since).toLocaleDateString()+'. ':'')+'Graded at the last line before tip, 1 unit a play.</div><table><tr><th>Grade</th><th class="num">Record</th><th class="num">Units</th><th class="num">ROI</th><th class="num hide-m">Hit</th></tr>'+['A','B','C'].map(grow).join('')+'</table></div>'+
  pageAs()+
  '<div class="panel"><h3>History</h3><div class="tabs">'+['all','A','B','C'].map(function(g){return '<button class="tab'+(S.hist===g?' on':'')+'" onclick="window.__set(&quot;hist&quot;,&quot;'+g+'&quot;)">'+(g==='all'?'All':GM[g][0]+' '+g)+'</button>';}).join('')+'</div>'+
  (hist.length?'<table>'+hist.map(function(h){return '<tr><td><b class="'+(h.result==='W'?'pos':h.result==='L'?'neg':'')+'">'+h.result+'</b></td><td>'+badge(h.final_grade)+' '+esc(h.market==='prop'?h.player+' '+(h.side==='over'?'O':'U')+h.final_line+' '+(h.prop_market||'').toLowerCase():h.market==='total'?(h.side==='under'?'U ':'O ')+h.final_line:short(h.side==='home'?h.home:h.away)+' '+(function(){var l=h.side==='home'?h.final_line:-h.final_line;return (l>0?'+':'')+l;})())+'</td><td class="note hide-m">'+esc(short(h.away))+' at '+esc(short(h.home))+'</td><td class="num">'+(h.actual!=null?h.actual:h.final_score||'')+'</td></tr>';}).join('')+'</table>':'<div class="pn">No graded plays yet.</div>')+'</div>'+
  (mk!=='prop'?'<div class="panel"><h3>Edge key &middot; '+(mk==='spread'?'spreads':'totals')+'</h3><div class="pn">Backtest vs opening lines, each season predicted by a model trained only on earlier seasons. &#127942; = up every season.</div>'+edgeKey(mk)+'</div>':'')+
  '<div class="panel"><h3>Accuracy</h3><div class="pn">Average miss in points (lower is better), backtest.</div><table><tr><th></th><th class="num">Model</th><th class="num">Market</th></tr><tr><td>Margin</td><td class="num">'+(acc.margin_mae||'&ndash;')+'</td><td class="num">'+(acc.market_spread_mae||'&ndash;')+'</td></tr><tr><td>Total</td><td class="num">'+(acc.total_mae||'&ndash;')+'</td><td class="num">'+(acc.market_total_mae||'&ndash;')+'</td></tr>'+
  Object.keys(pa).map(function(k){return '<tr><td>'+k+' (vs recent avg)</td><td class="num">'+pa[k].model_mae+'</td><td class="num">'+pa[k].recent_avg_mae+'</td></tr>';}).join('')+'</table></div>';}
function pageAs(){
  // every A play flagged since the model went live: official ones plus the pre-launch days (labeled)
  var off=((D.record||{}).history||[]).filter(function(h){return h.final_grade==='A';}).map(function(h){h._pre=0;return h;});
  var pre=(((D.prelaunch||{}).history)||[]).filter(function(h){return h.final_grade==='A';}).map(function(h){h._pre=1;return h;});
  var all=off.concat(pre).sort(function(a,b){return a.kickoff<b.kickoff?1:-1;});
  if(!all.length)return '';
  var by={};all.forEach(function(h){var k=h.market;by[k]=by[k]||{w:0,l:0,u:0};if(h.result==='W')by[k].w++;else if(h.result==='L')by[k].l++;by[k].u+=+(h.units||0);});
  var names={spread:'Spreads',total:'Totals',prop:'Props'};
  return '<div class="panel"><h3>&#128293; Every A play since launch</h3><div class="pn">All A grades the model has flagged, graded at the last line before tip, 1 unit each. '+
    'Plays marked <b>pre-launch</b> came before the official rules were set on 11/19 (then a 7+ pt spread edge was an A; now a spread needs a 10+ pt edge) and are not in the official record.</div>'+
    '<table><tr><th>Market</th><th class="num">W-L</th><th class="num">Units</th></tr>'+Object.keys(by).map(function(k){var b=by[k];return '<tr><td>'+(names[k]||k)+'</td><td class="num">'+b.w+'-'+b.l+'</td><td class="num '+(b.u>=0?'pos':'neg')+'">'+(b.u>=0?'+':'')+b.u.toFixed(1)+'u</td></tr>';}).join('')+'</table>'+
    '<table style="margin-top:10px">'+all.slice(0,80).map(function(h){var what=h.market==='prop'?esc(h.player||'')+' '+esc(h.prop_market||'')+' '+esc(h.side||'')+' '+h.final_line:
      esc(h.away||'')+' @ '+esc(h.home||'')+' &middot; '+(h.market==='total'?esc(h.side||'')+' '+h.final_line:esc(h.side==='home'?h.home:h.away)+' '+(h.side==='home'?(h.final_line>0?'+':'')+h.final_line:((-h.final_line)>0?'+':'')+(-h.final_line)));
      return '<tr><td><b class="'+(h.result==='W'?'pos':h.result==='L'?'neg':'')+'">'+(h.result||'&ndash;')+'</b></td><td>'+what+(h._pre?' <span class="pn">pre-launch</span>':'')+'</td><td class="num">'+(h.final_score||'')+'</td><td class="num">'+new Date(h.kickoff).toLocaleDateString()+'</td></tr>';}).join('')+'</table></div>';}
function render(){
  document.getElementById('nav').innerHTML=PAGES.map(function(p){return '<button class="'+(S.page===p[0]?'on':'')+'" onclick="window.__go(&quot;'+p[0]+'&quot;)">'+p[1]+'</button>';}).join('');
  document.getElementById('mnav').innerHTML=PAGES.map(function(p){return '<button class="'+(S.page===p[0]?'on':'')+'" onclick="window.__go(&quot;'+p[0]+'&quot;)">'+p[1].replace('Model ','').replace('Player ','')+'</button>';}).join('');
  document.getElementById('asof').textContent='Updated '+kick(D.meta.generated_at);
  var banner=D.meta.demo?'<div class="demo">&#9888; DEMO DATA: every number on this page is made up to preview the layout. Real numbers appear after the first run with API keys. Official rules are provisional until the backtest sets them.</div>':
    '<div class="demo" style="border-color:rgba(94,200,255,.4);color:var(--blue);background:rgba(94,200,255,.08)">Rules (from the backtest): unders 10+ and spreads 10+ = &#128293; A, unders 5-10 = &#127922; B. The edge is vs opening lines, so bet early.</div>';
  var body={edge:pageEdge,totals:pageTotals,props:pageProps,tracker:pageTracker,perf:pagePerf}[S.page]||pageEdge;
  document.getElementById('app').innerHTML=banner+body();}
render();
})();
</script></body></html>"""


def main():
    data = json.load(open(f"{config.DOCS_DATA_DIR}/latest.json"))
    html = TEMPLATE.replace("__DATA__", json.dumps(data, default=str).replace("</", "<\\/"))
    with open("docs/dashboard.html", "w") as f:
        f.write(html)
    with open("docs/index.html", "w") as f:
        f.write('<meta http-equiv="refresh" content="0; url=dashboard.html">')
    print(f"Wrote docs/dashboard.html ({len(data.get('games', []))} games)")


if __name__ == "__main__":
    main()
