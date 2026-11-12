"""wolf H9 프레임별 입자 '탄성 에너지 밀도 - 운동 에너지' 산점도 단일 HTML 뷰어 (그림 base64 포함).
사용: python exe/wolffields_html.py <psi 디렉토리 (sample.npz fields.json)> <vid 디렉토리 (h9_plain.mp4 gt_plain.mp4)> <출력 html> <임시 폴더>"""
import base64, glob, json, os, subprocess, sys
import numpy as np
pd, vd, out, TMP = sys.argv[1:5]
os.makedirs(TMP, exist_ok=True)
Z = np.load(f"{pd}/sample.npz"); J = json.load(open(f"{pd}/fields.json"))
EPS, EK = J["eps"], J["ek"]
lx = lambda a: np.log10(a + EPS); ly = lambda a: np.log10(a + EK)
XL = (min(lx(Z["P"][1:]).min(), lx(Z["G"][1:]).min()), max(lx(Z["P"]).max(), lx(Z["G"]).max()))
YL = (J["k_lo"], max(ly(Z["KP"]).max(), ly(Z["KG"]).max()))


def q16(a, lo, hi): return np.round(np.clip((a - lo) / (hi - lo), 0, 1) * 65535).astype("<u2")


b = lambda a: base64.b64encode(a.tobytes()).decode()
Rq = np.where(np.isfinite(Z["R"]), np.round(np.clip((Z["R"] - J["r_lo"]) / (J["r_hi"] - J["r_lo"]), 0, 1) * 254), 255).astype(np.uint8)
imgs = {}
for k in ("h9_plain", "gt_plain"):
    d = f"{TMP}/{k}"; os.makedirs(d, exist_ok=True)
    for p in glob.glob(f"{d}/*.jpg"): os.remove(p)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{vd}/{k}.mp4", "-vf", "scale=360:-2", "-q:v", "4", f"{d}/%04d.jpg"], check=True)
    imgs[k] = ["data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode() for p in sorted(glob.glob(f"{d}/*.jpg"))]
D = dict(T=J["L"] + 1, n=int(len(Z["idx"])), N=J["N"], xl=[float(v) for v in XL], yl=[float(v) for v in YL], rlo=J["r_lo"], rhi=J["r_hi"],
         px=b(q16(lx(Z["P"]), *XL)), py=b(q16(ly(Z["KP"]), *YL)), gx=b(q16(lx(Z["G"]), *XL)), gy=b(q16(ly(Z["KG"]), *YL)), r=b(Rq),
         Eel=[J["E_el_pred"], J["E_el_gt"]], Eke=[J["E_ke_pred"], J["E_ke_gt"]], rmed=J["r_med"], img_p=imgs["h9_plain"], img_g=imgs["gt_plain"])
HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>wolf 입자 탄성 에너지 밀도 - 운동 에너지</title>
<style>html,body{margin:0;height:100vh;font:14px sans-serif;background:#fff;overflow:hidden}
#top{height:6vh;display:flex;align-items:center;gap:12px;padding:0 12px;border-bottom:1px solid #ccc}#top input[type=range]{flex:1}
#imgs{height:24vh;display:flex;justify-content:space-around;align-items:center}#imgs div{text-align:center}#imgs img{height:20vh}
#plots{height:60vh;display:flex}#plots div{flex:1;text-align:center}canvas{width:98%;height:54vh}
#note{height:8vh;padding:0 12px;font-size:12px;color:#444;overflow:hidden}</style></head><body>
<div id="top"><button id="pb">재생</button><span>프레임 <b id="fn">0</b></span><input id="sl" type="range" min="0" max="0" value="0"><span id="rm"></span></div>
<div id="imgs"><div>우리 H9<br><img id="ip"></div><div>정답<br><img id="ig"></div></div>
<div id="plots"><div><b>우리 H9</b> <span id="ep"></span><br><canvas id="c0" width="900" height="600"></canvas></div><div><b>정답</b> <span id="eg"></span><br><canvas id="c1" width="900" height="600"></canvas></div></div>
<div id="note">가로축 탄성 에너지 밀도 log10(ψ+1e-3) [J/m³], 세로축 입자 운동 에너지 log10(½mv²) [J], 두 축 모두 로그, 범위는 전 프레임·두 판 공통 고정. 점 색(우리 쪽만) = 입자가 든 칸의 높이 비 r (0.3~1.1 고정, 1 이 흰색, 회색은 칸 없음).
점은 그리기용으로 무작위 <span id="nn"></span> 개를 두 판 같은 번호로 뽑은 표본이며 화면 표시용이다. 계산·지표(전체값)에는 쓰지 않았고 전체 입자 <span id="NN"></span> 개로 계산한 값이다.</div>
<script>const D=__DATA__;
function dec(s,t){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return t==2?new Uint16Array(a.buffer):a}
const px=dec(D.px,2),py=dec(D.py,2),gx=dec(D.gx,2),gy=dec(D.gy,2),R=dec(D.r,1),n=D.n,T=D.T;
document.getElementById('nn').textContent=n;document.getElementById('NN').textContent=D.N;
const sl=document.getElementById('sl');sl.max=T-1;
function rc(v){ // RdBu 발산형 (1 이 흰색)
 if(v==255)return'rgb(190,190,190)';const t=v/254;const lo=[178,24,43],mid=[247,247,247],hi=[33,102,172];let a,b,u;
 if(t<.5){a=lo;b=mid;u=t/.5}else{a=mid;b=hi;u=(t-.5)/.5}return`rgb(${a.map((x,i)=>Math.round(x+(b[i]-x)*u)).join(',')})`}
const rcs=[];for(let v=0;v<256;v++)rcs.push(rc(v));
function draw(ci,X,Y,t,col){const c=document.getElementById('c'+ci),g=c.getContext('2d'),W=c.width,H=c.height,m=[70,20,20,50];g.clearRect(0,0,W,H);
 const pw=W-m[0]-m[1],ph=H-m[2]-m[3];g.strokeStyle='#888';g.strokeRect(m[0],m[2],pw,ph);g.fillStyle='#000';g.font='16px sans-serif';g.textAlign='center';
 const xl=D.xl,yl=D.yl;for(let k=Math.ceil(xl[0]);k<=xl[1];k++){const x=m[0]+(k-xl[0])/(xl[1]-xl[0])*pw;g.fillText('1e'+k,x,H-m[3]+20);g.beginPath();g.moveTo(x,m[2]);g.lineTo(x,m[2]+ph);g.strokeStyle='#eee';g.stroke()}
 g.textAlign='right';for(let k=Math.ceil(yl[0]);k<=yl[1];k++){const y=m[2]+ph-(k-yl[0])/(yl[1]-yl[0])*ph;g.fillText('1e'+k,m[0]-4,y+5);g.beginPath();g.moveTo(m[0],y);g.lineTo(m[0]+pw,y);g.strokeStyle='#eee';g.stroke()}
 g.textAlign='center';g.fillText('탄성 에너지 밀도 (J/m³, 로그)',m[0]+pw/2,H-8);g.save();g.translate(14,m[2]+ph/2);g.rotate(-Math.PI/2);g.fillText('입자 운동 에너지 (J, 로그)',0,0);g.restore();
 const o=t*n;for(let i=0;i<n;i++){g.fillStyle=col?rcs[R[o+i]]:'#4a6fa5';g.globalAlpha=.7;g.fillRect(m[0]+X[o+i]/65535*pw-1.5,m[2]+ph-Y[o+i]/65535*ph-1.5,3,3)}g.globalAlpha=1}
function show(t){document.getElementById('fn').textContent=t;sl.value=t;document.getElementById('ip').src=D.img_p[t];document.getElementById('ig').src=D.img_g[t];
 document.getElementById('ep').textContent=`탄성 에너지 합 ${D.Eel[0][t].toPrecision(3)} J · 운동 에너지 합 ${D.Eke[0][t].toPrecision(3)} J`;
 document.getElementById('eg').textContent=`탄성 에너지 합 ${D.Eel[1][t].toPrecision(3)} J · 운동 에너지 합 ${D.Eke[1][t].toPrecision(3)} J`;
 document.getElementById('rm').textContent=`칸 높이 비 중앙값 ${D.rmed[t].toFixed(3)}`;draw(0,px,py,t,true);draw(1,gx,gy,t,false)}
sl.oninput=()=>show(+sl.value);let tm=null;const pb=document.getElementById('pb');
pb.onclick=()=>{if(tm){clearInterval(tm);tm=null;pb.textContent='재생'}else{pb.textContent='정지';tm=setInterval(()=>show((+sl.value+1)%T),150)}};show(0);</script></body></html>"""
open(out, "w").write(HTML.replace("__DATA__", json.dumps(D)))
print("저장", out, os.path.getsize(out) // 1024, "KB", flush=True)
