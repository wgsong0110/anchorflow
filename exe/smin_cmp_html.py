"""smin.json + g00..g50.jpg(정답)/h00..h50.jpg(H9) -> index.html (정답 대 H9 나란히). 사용: python exe/smin_cmp_html.py <smin.json> <출력 index.html>"""
import json, sys
d = json.load(open(sys.argv[1]))
T = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>wolf 정답 대 H9 · 입자 F 분포</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{font-family:'Noto Sans CJK KR','Malgun Gothic',sans-serif;margin:12px;color:#222}
.ctl{display:flex;gap:12px;align-items:center;margin:8px 0}#sl{flex:1}
#cols{display:flex;gap:16px}.col{width:50%}.col h3{margin:4px 0;text-align:center}.col img{width:420px;display:block;margin:0 auto;border:1px solid #ddd}
.st{font-size:14px;line-height:1.6;padding:8px;border-radius:6px;margin:6px 0}#sg{background:#eee}#sp{background:#e8f0fc}
.note{background:#fff8e1;border:1px solid #e6c84a;padding:8px;border-radius:6px;font-size:14px;line-height:1.55}small{color:#555}</style></head><body>
<h3>wolf 모래 · MPM 정답 대 우리 H9 (=H8) · 프레임별 입자 변형 구배 F 분포</h3>
<div class="note"><b>정답의 σ≈1 이 정상 기준이다.</b> 모래 탄성은 자중 응력이 약 10 kPa 이고 E = __E__ Pa 라, 정답의 탄성 변형은 σ ≈ 1, det F ≈ 1 (J − 1 ≈ −2e-4) 인 것이 물리적으로 맞는 값이다 (절 CM110). 이 쪽을 기준으로 H9 가 얼마나 벗어나는지를 본다.
정답 F 는 <b>탄성 F</b> 이다: 정답 h5 의 f_tensor = PhysGaussian particle_F 이고 sand_return_mapping (Drucker-Prager) 을 거친 뒤의 탄성 F_e 이며 (소성 몫은 이미 빠짐). H9 쪽은 실행기가 덤프한 F 를 소성 사영 전후 구분 없이 그대로 썼다.</div>
<div class="ctl"><button id="pb">재생</button><input id="sl" type="range" min="0" max="__NF__" value="0"><b id="fl"></b></div>
<div id="cols">
<div class="col"><h3>MPM 정답</h3><img id="ig"><div class="st" id="sg"></div><div id="g1" style="height:300px"></div><div id="g2" style="height:300px"></div></div>
<div class="col"><h3>우리 H9</h3><img id="ip"><div class="st" id="sp"></div><div id="p1" style="height:300px"></div><div id="p2" style="height:300px"></div></div></div>
<div id="en" style="height:360px"></div>
<p><small>히스토그램: 가로 범위와 bin 은 두 쪽·전 프레임 공통으로 고정, 정규화하지 않음 (세로 = 입자 수, 두 쪽 같은 세로 범위, 세로축은 로그). σ_min = F 의 최소 특이값 (det F &lt; 0 이면 −σ_min). 프레임 0 의 F 는 항등. 입자 __NP__ 개.
탄성 에너지 Σ V_p ψ(F_p), ψ = μ Σ ε_i² + λ/2 (Σ ε_i)², ε_i = log σ_i (μ = __MU__, λ = __LAM__, V_p = m_p/밀도) 를 두 쪽 같은 식으로 덤프 F 에 적용했다 (0 은 1e-3 으로 표시). 렌더: R2 result/anchorflow/v20261009/fem_H8_wolf_cmp.mp4 의 정답 칸·H8 칸.</small></p>
<script>
const D=__DATA__;const NF=D.pred.length-1;
function ctr(r){const [lo,hi,n]=r,w=(hi-lo)/n;return Array.from({length:n},(_,i)=>lo+w*(i+.5))}
const xs=ctr(D.srange),xd=ctr(D.drange),ws=(D.srange[1]-D.srange[0])/D.srange[2],wd=(D.drange[1]-D.drange[0])/D.drange[2];
let ymaxS=1,ymaxD=1;for(const k of ['pred','gt'])for(const r of D[k]){ymaxS=Math.max(ymaxS,...r.sh);ymaxD=Math.max(ymaxD,...r.dh)}
function lim(key,ctrs){let tot=new Array(ctrs.length).fill(0);for(const k of ['pred','gt'])for(const r of D[k])r[key].forEach((v,i)=>tot[i]+=v);
 const s=tot.reduce((a,b)=>a+b,0);let a=0,lo=0,hi=ctrs.length-1;while(a+tot[lo]<s*0.0005){a+=tot[lo];lo++}a=0;while(a+tot[hi]<s*0.0005){a+=tot[hi];hi--}return [ctrs[Math.max(lo-2,0)],ctrs[Math.min(hi+2,ctrs.length-1)]]}
const xrS=lim('sh',xs),xrD=lim('dh',xd);
function fig(div,x,w,y,color,xr,ymax,title,xt,vl){
 Plotly.react(div,[{x:x,y:y.map(v=>v>0?v:null),type:'bar',width:w,marker:{color:color}}],{title:{text:title,font:{size:14}},margin:{t:34,l:60,r:10,b:40},xaxis:{range:xr,title:xt},yaxis:{type:'log',range:[0,Math.log10(ymax*1.2)],title:'입자 수 (로그)'},bargap:0,
 shapes:[{type:'line',x0:vl,x1:vl,y0:0,y1:1,yref:'paper',line:{color:'#c33',dash:'dot',width:1}}]},{displaylogo:false})}
function f3(v){return v.toFixed(3)}
function txt(r){return 'σ_min 중앙값 <b>'+f3(r.med)+'</b> · 10 백분위 <b>'+f3(r.p10)+'</b> · σ_min&lt;0.5 몫 <b>'+(100*r.lt05).toFixed(2)+'%</b><br>det F 중앙값 '+f3(r.dmed)+' · det F&lt;0 몫 '+(100*r.neg).toFixed(2)+'%'}
function pad(f){return String(f).padStart(2,'0')}
function show(f){f=+f;document.getElementById('ig').src='g'+pad(f)+'.jpg';document.getElementById('ip').src='h'+pad(f)+'.jpg';document.getElementById('fl').textContent='프레임 '+f+' / '+NF;
 const g=D.gt[f],p=D.pred[f];document.getElementById('sg').innerHTML=txt(g);document.getElementById('sp').innerHTML=txt(p);
 fig('g1',xs,ws,g.sh,'#777',xrS,ymaxS,'정답 σ_min 분포 (붉은 점선 = 0.5)','σ_min',0.5);fig('g2',xd,wd,g.dh,'#777',xrD,ymaxD,'정답 det F 분포','det F',1);
 fig('p1',xs,ws,p.sh,'#2a6fdb',xrS,ymaxS,'H9 σ_min 분포 (붉은 점선 = 0.5)','σ_min',0.5);fig('p2',xd,wd,p.dh,'#2a6fdb',xrD,ymaxD,'H9 det F 분포','det F',1);
 en(f)}
const sl=document.getElementById('sl'),pb=document.getElementById('pb');let tm=null;
sl.oninput=()=>show(sl.value);
pb.onclick=()=>{if(tm){clearInterval(tm);tm=null;pb.textContent='재생'}else{pb.textContent='정지';tm=setInterval(()=>{sl.value=(+sl.value+1)%(NF+1);show(sl.value)},250)}};
function en(f){const x=D.pred.map((_,i)=>i);
 Plotly.react('en',[{x:x,y:D.pred.map(r=>Math.max(r.E,1e-3)),mode:'lines',name:'우리 H9',line:{color:'#2a6fdb',width:2}},{x:x,y:D.gt.map(r=>Math.max(r.E,1e-3)),mode:'lines',name:'MPM 정답',line:{color:'#777',width:2}}],
 {title:{text:'프레임별 전체 탄성 에너지 Σ V_p ψ(F_p) [J] (세로 로그축, 세로선 = 현재 프레임)',font:{size:14}},margin:{t:34,l:70,r:10,b:40},xaxis:{title:'프레임'},yaxis:{type:'log',title:'에너지 [J]'},
 shapes:[{type:'line',x0:f,x1:f,y0:0,y1:1,yref:'paper',line:{color:'#c33',width:2}}]},{displaylogo:false})}
show(0);
</script></body></html>'''
T = T.replace("__E__", "%.3g" % d["E"]).replace("__MU__", "%.4g" % d["mu"]).replace("__LAM__", "%.4g" % d["lam"]).replace("__NP__", str(d["Np"])).replace("__NF__", str(len(d["pred"]) - 1)).replace("__DATA__", json.dumps(d, separators=(",", ":")))
open(sys.argv[2], "w").write(T)
