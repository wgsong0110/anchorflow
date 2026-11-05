"""절 CM13.4: 수박 단면 손상 변수 (logJp) 인터랙티브 뷰어 -- 단일 HTML (plotly CDN, 자료는 gzip+base64 로 내장).

  python3 exe/wmd_viewer.py --npz wmd_slices.npz --stats wmd_stats.json --out viewer.html [--cap 800] [--zstep 1]
프레임·z 슬라이더, 정답 (GF 다시 돌림) | 우리 (2) SNH 단면 나란히, 같은 viridis·공통 색 범위 (모든 프레임·모든 z), 확대·이동, 입자 위 값 표시.
표시용 부분표본 상한 --cap (층마다, 고정 씨앗; 화면에 "표시 n / 전체 N").
"""
import argparse
import base64
import gzip
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--stats", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cap", type=int, default=800)
ap.add_argument("--zstep", type=int, default=1, help="층 번호를 이 간격으로만 (1 = 모든 층)")
ap.add_argument("--vmin", type=float, default=-0.2)
ap.add_argument("--vmax", type=float, default=0.4)
ap.add_argument("--label_gt", default="정답 (GF 공식 러너 다시 돌림)")
ap.add_argument("--label_ours", default="우리 (2) Stable Neo-Hookean")
a = ap.parse_args()
Z = np.load(a.npz)
xy, v, idx = Z["xy"], Z["v"], Z["idx"]
dx, z0, T = float(Z["dx"]), float(Z["z0"]), int(Z["T"])
rng = np.random.default_rng(1)
XY, V, TAB = [], [], []
off = 0
lo, hi = xy.min(0), xy.max(0)
for side, t, k, s0, n, full in idx:
    if k % a.zstep:
        continue
    sel = np.arange(s0, s0 + n)
    if n > a.cap:
        sel = np.sort(rng.choice(sel, a.cap, replace=False))
    XY.append(xy[sel]); V.append(v[sel])
    TAB.append([int(side), int(t), int(k), off, len(sel), int(full)]); off += len(sel)
XY = np.concatenate(XY); V = np.concatenate(V)
q = np.round((XY - lo) / (hi - lo) * 65535 - 32768).astype(np.int16)
vq = np.round(np.clip(np.nan_to_num(V, nan=9.0), -3.2, 3.2) * 10000).astype(np.int16)
blob = base64.b64encode(gzip.compress(q.tobytes() + vq.tobytes(), 9)).decode()
S = json.load(open(a.stats))
meta = dict(T=T, dx=dx, z0=z0, lo=lo.tolist(), hi=hi.tolist(), n=int(off), tab=TAB, vmin=a.vmin, vmax=a.vmax, cap=a.cap,
            gt0=[s["gt0"] for s in S["gt"]], ours0=[s["gt0"] for s in S["ours"]], lg=a.label_gt, lo_=a.label_ours,
            ks=sorted({r[2] for r in TAB}), zstep=a.zstep)
html = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>수박 단면 손상 변수 (logJp) -- 정답 | 우리</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{font-family:"Noto Sans KR",sans-serif;margin:12px;color:#0b0b0b;background:#fcfcfb}
.row{display:flex;gap:18px;align-items:center;margin:6px 0}.row label{min-width:90px}input[type=range]{width:520px}
#info{color:#52514e;font-size:13px;line-height:1.5}</style></head><body>
<h3 style="margin:4px 0">수박 단면의 손상 변수 logJp (NACC 경화 상태) -- 정답 | 우리</h3>
<div id="info"></div>
<div class="row"><label>프레임 <b id="tl"></b></label><input id="ts" type="range" min="0" step="1"><button id="pl">재생</button></div>
<div class="row"><label>높이 z <b id="zl"></b></label><input id="zs" type="range" min="0" step="1"></div>
<div id="cnt" style="font-size:13px;color:#52514e"></div>
<div id="pp" style="width:100%;height:640px"></div>
<script>
const M=__META__;const B64="__BLOB__";
function sideIdx(){const m={};for(const r of M.tab){m[r[0]+"_"+r[1]+"_"+r[2]]=r;}return m;}
const IDX=sideIdx();
async function load(){const bin=Uint8Array.from(atob(B64),c=>c.charCodeAt(0));
 const ds=new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip")));const buf=await ds.arrayBuffer();
 const q=new Int16Array(buf,0,M.n*2);const vq=new Int16Array(buf,M.n*4,M.n);return {q,vq};}
document.getElementById("info").innerHTML=
 "손상 변수 = 입자별 logJp (GF NACC return mapping 의 경화 상태, 처음 −0.04). logJp ≥ 0 이면 응집력 p0 = κ(1e-5 + sinh(ξ·max(−logJp,0))) 이 사라짐 = 손상 입자; GF 러너는 logJp > 0.4 입자를 렌더에서 숨김.<br>"+
 "단면 = 높이 z 를 중심으로 두께 "+M.dx.toFixed(5)+" (정답 MPM 격자 칸 dx = 2/300, 시뮬 공간 단위) 의 수평 층 입자, 위에서 본 (x, y). z = 바닥 0.02 + (k + ½)·dx, 슬라이더 한 칸 = "+M.zstep+" 층 ("+(M.zstep*M.dx).toFixed(5)+").<br>"+
 "색: viridis, 공통 범위 "+M.vmin+" ~ "+M.vmax+" (모든 프레임·모든 z·두 편 같음, 밖은 끝 색). 표시용 부분표본: 층마다 최대 "+M.cap+" 개 (고정 씨앗) -- 아래에 표시/전체 개수.<br>"+
 "정답 = GF 공식 러너를 같은 config·3DGS 로 다시 돌려 logJp 를 더 덤프한 궤적 (원래 정답 h5 에는 logJp 가 없음; GF 도 비결정이라 위치가 원래 정답과 조금 다름 -- doc 절 CM13.4).";
const ts=document.getElementById("ts"),zs=document.getElementById("zs");ts.max=M.T-1;ts.value=0;
zs.max=M.ks.length-1;zs.value=Math.floor(M.ks.length/4);
load().then(({q,vq})=>{
 function pts(side,t,k){const r=IDX[side+"_"+t+"_"+k];if(!r)return {x:[],y:[],c:[],n:0,full:0};
  const x=new Float32Array(r[4]),y=new Float32Array(r[4]),c=new Float32Array(r[4]);
  for(let i=0;i<r[4];i++){const j=r[3]+i;x[i]=(q[2*j]+32768)/65535*(M.hi[0]-M.lo[0])+M.lo[0];y[i]=(q[2*j+1]+32768)/65535*(M.hi[1]-M.lo[1])+M.lo[1];c[i]=vq[j]/10000;}
  return {x,y,c,n:r[4],full:r[5]};}
 const cs="Viridis";
 function tr(p,ax,show){return {type:"scattergl",mode:"markers",x:p.x,y:p.y,xaxis:"x"+ax,yaxis:"y"+ax,
  marker:{size:4,color:p.c,colorscale:cs,cmin:M.vmin,cmax:M.vmax,showscale:show,colorbar:show?{title:{text:"logJp",side:"right"},len:0.9,tickvals:[-0.2,-0.04,0,0.2,0.4],ticktext:["−0.2","−0.04 처음","0 응집력 0","0.2","0.4 GF 숨김"]}:undefined},
  hovertemplate:"x %{x:.4f}<br>y %{y:.4f}<br>logJp %{marker.color:.4f}<extra></extra>"};}
 const lay={grid:{rows:1,columns:2,pattern:"independent"},margin:{t:40,l:50,r:20,b:40},paper_bgcolor:"#fcfcfb",plot_bgcolor:"#fcfcfb",dragmode:"pan",
  xaxis:{range:[M.lo[0],M.hi[0]],title:"x"},yaxis:{range:[M.lo[1],M.hi[1]],scaleanchor:"x",title:"y"},
  xaxis2:{range:[M.lo[0],M.hi[0]],title:"x",matches:"x"},yaxis2:{range:[M.lo[1],M.hi[1]],scaleanchor:"x2",matches:"y"},
  annotations:[{text:M.lg,xref:"x domain",yref:"y domain",x:0.5,y:1.06,showarrow:false},{text:M.lo_,xref:"x2 domain",yref:"y2 domain",x:0.5,y:1.06,showarrow:false}]};
 let first=true;
 function draw(){const t=+ts.value,k=M.ks[+zs.value];const g=pts(0,t,k),o=pts(1,t,k);
  document.getElementById("tl").textContent=t;document.getElementById("zl").textContent=(M.z0+(k+0.5)*M.dx).toFixed(4)+" (층 "+k+")";
  document.getElementById("cnt").textContent="정답: 표시 "+g.n+" / 층 전체 "+g.full+" 입자, 이 프레임 손상 (logJp > 0) 몫 "+(100*M.gt0[t]).toFixed(2)+" %   |   우리: 표시 "+o.n+" / 층 전체 "+o.full+" 입자, 손상 몫 "+(100*(M.ours0[t]??NaN)).toFixed(2)+" %";
  const d=[tr(g,"",false),tr(o,"2",true)];
  if(first){Plotly.newPlot("pp",d,lay,{scrollZoom:true,displaylogo:false});first=false;}else{Plotly.react("pp",d,document.getElementById("pp").layout);}}
 ts.oninput=draw;zs.oninput=draw;draw();
 let timer=null;document.getElementById("pl").onclick=()=>{if(timer){clearInterval(timer);timer=null;return;}
  timer=setInterval(()=>{ts.value=(+ts.value+1)%M.T;draw();},150);};
});
</script></body></html>"""
open(a.out, "w").write(html.replace("__META__", json.dumps(meta)).replace("__BLOB__", blob))
print(f"[뷰어] {a.out} 점 {off} 층 {len(TAB)} 크기 {len(blob) / 1e6:.1f} MB", flush=True)
