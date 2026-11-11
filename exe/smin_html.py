"""smin.json + f00..f50.jpg -> index.html (로컬, 작은 json 만 읽음). 사용: python exe/smin_html.py <smin.json> <출력 index.html>"""
import json, sys
d = json.load(open(sys.argv[1]))
T = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>wolf H9 (=H8) 프레임별 σ_min 분포</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{font-family:'Noto Sans CJK KR','Malgun Gothic',sans-serif;margin:12px;color:#222}
#wrap{display:flex;gap:16px;align-items:flex-start}#img{width:600px;height:600px;border:1px solid #ddd}
#right{width:760px}.ctl{display:flex;gap:12px;align-items:center;margin:8px 0}#sl{flex:1}
#st{font-size:14px;line-height:1.6;background:#f6f6f6;padding:8px;border-radius:6px}small{color:#555}</style></head><body>
<h3>wolf H9 (=H8) 한 판 · 프레임별 입자 변형 구배 F 의 최소 특이값 σ_min 분포</h3>
<div class="ctl"><button id="pb">재생</button><input id="sl" type="range" min="0" max="__NF__" value="0"><b id="fl"></b></div>
<div id="wrap"><div><img id="img"><br><small>왼쪽: 3DGS 렌더 (R2 result/anchorflow/v20261009/fem_H8_wolf_cmp.mp4 의 H8 칸을 잘라 씀)</small></div>
<div id="right"><div id="st"></div><div id="h1" style="height:300px"></div><div id="h2" style="height:300px"></div></div></div>
<p><small>σ_min = 입자 F 의 최소 특이값. det F &lt; 0 인 입자는 부호를 붙여 −σ_min 으로 계산했다. 히스토그램의 x 범위와 bin 은 전 프레임 공통으로 고정했고 프레임마다 정규화하지 않았다 (세로축 = 입자 수).
프레임 0 의 F 는 항등. __GTNOTE__</small></p>
<script>
const D=__DATA__;const NF=D.pred.length-1;
function ctr(r){const [lo,hi,n]=r,w=(hi-lo)/n;return Array.from({length:n},(_,i)=>lo+w*(i+.5))}
const xs=ctr(D.srange),xd=ctr(D.drange),ws=(D.srange[1]-D.srange[0])/D.srange[2],wd=(D.drange[1]-D.drange[0])/D.drange[2];
let ymaxS=0,ymaxD=0;
for(const k of ['pred','gt']){if(!D[k])continue;for(const r of D[k]){ymaxS=Math.max(ymaxS,...r.sh);ymaxD=Math.max(ymaxD,...r.dh)}}
// x 표시 범위: 전 프레임 입자의 99.9% 가 들어가는 고정 범위
function lim(key,ctrs,maxfrac){let tot=new Array(ctrs.length).fill(0);for(const k of ['pred','gt']){if(!D[k])continue;for(const r of D[k])r[key].forEach((v,i)=>tot[i]+=v)}
 const s=tot.reduce((a,b)=>a+b,0);let a=0,lo=0,hi=ctrs.length-1;while(a+tot[lo]<s*(1-maxfrac)/2){a+=tot[lo];lo++}a=0;while(a+tot[hi]<s*(1-maxfrac)/2){a+=tot[hi];hi--}return [ctrs[Math.max(lo-2,0)],ctrs[Math.min(hi+2,ctrs.length-1)]]}
const xrS=lim('sh',xs,0.999),xrD=lim('dh',xd,0.999);
function fig(div,x,w,key,f,xr,ymax,title,xt,vl){const tr=[{x:x,y:D.pred[f][key],type:'bar',width:w,name:'예측 (H8)',marker:{color:'#2a6fdb',opacity:.75}}];
 if(D.gt)tr.push({x:x,y:D.gt[f][key],type:'scatter',mode:'lines',line:{color:'#777',width:2,shape:'hv'},name:'정답 (GT)'});
 const lay={title:{text:title,font:{size:14}},margin:{t:34,l:60,r:10,b:40},xaxis:{range:xr,title:xt},yaxis:{range:[0,ymax*1.05],title:'입자 수'},bargap:0,legend:{x:.72,y:.98},
  shapes:vl?[{type:'line',x0:vl,x1:vl,y0:0,y1:1,yref:'paper',line:{color:'#c33',dash:'dot',width:1}}]:[]};
 Plotly.react(div,tr,lay,{displaylogo:false})}
function f3(v){return v.toFixed(3)}
function show(f){f=+f;document.getElementById('img').src='f'+String(f).padStart(2,'0')+'.jpg';document.getElementById('fl').textContent='프레임 '+f+' / '+NF;
 const p=D.pred[f],g=D.gt&&D.gt[f];
 let h='<b>예측 (H8)</b> σ_min 중앙값 '+f3(p.med)+' · 10 백분위 '+f3(p.p10)+' · σ_min&lt;0.5 입자 몫 '+(100*p.lt05).toFixed(1)+'% · det F&lt;0 몫 '+(100*p.neg).toFixed(2)+'%';
 if(g)h+='<br><b style="color:#666">정답 (GT)</b> σ_min 중앙값 '+f3(g.med)+' · 10 백분위 '+f3(g.p10)+' · σ_min&lt;0.5 입자 몫 '+(100*g.lt05).toFixed(1)+'%';else h+='<br>정답 F 없음 (생략): '+D.gt_note;
 h+='<br><small>입자 '+p.n+' 개 · 표시 범위 밖 σ_min 최소 '+f3(p.smin)+' 최대 '+f3(p.smax)+'</small>';document.getElementById('st').innerHTML=h;
 fig('h1',xs,ws,'sh',f,xrS,ymaxS,'σ_min 분포 (부호 붙임, 회색 윤곽선 = 정답 · 붉은 점선 = 0.5)','σ_min',0.5);
 fig('h2',xd,wd,'dh',f,xrD,ymaxD,'det F 분포','det F',1)}
const sl=document.getElementById('sl'),pb=document.getElementById('pb');let tm=null;
sl.oninput=()=>show(sl.value);
pb.onclick=()=>{if(tm){clearInterval(tm);tm=null;pb.textContent='재생'}else{pb.textContent='정지';tm=setInterval(()=>{sl.value=(+sl.value+1)%(NF+1);show(sl.value)},250)}};
show(0);
</script></body></html>'''
gtn = "정답 F 는 정답 h5 의 F 로 구함, 회색 윤곽선." if d.get("gt") else "정답 F 는 구할 수 없어 생략했다: " + d["gt_note"]
T = T.replace("__NF__", str(len(d["pred"]) - 1)).replace("__GTNOTE__", gtn).replace("__DATA__", json.dumps(d, separators=(",", ":")))
open(sys.argv[2], "w").write(T)
