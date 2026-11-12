"""wolf H9 탄성-정규화 산점도 단일 HTML 뷰어, 점 색 기준을 버튼으로: 단색 / 속도 / 높이 / x / y (그림 base64 포함).
(a) 입자: 입자 탄성 에너지 밀도 ψ (J/m³) 대 입자에 나눈 정규화 에너지 (J), 색 = 표본 입자의 프레임 끝 속도 크기 |v| · 높이 (바닥 법선 방향, 바닥 기준) · 월드 x · 월드 y.
(b) 칸: 칸 탄성 에너지 대 칸 정규화 에너지 (J), 색 = 칸 중심 (칸 입자 질량 가중 평균 위치) 의 높이·x·y, 칸 평균 속도 (칸 입자 질량 가중 평균 속도 벡터) 의 크기.
축·색 범위는 전 프레임 공통 고정 (색 범위는 입자·칸 같은 기준끼리 하나, 프레임 1~끝 분위수). 표본은 그리기용만.
입력 = 기록 판 디렉토리 (lib/anchorflow/fem_wolfip.py 의 ip/f*.npz: psi·e_el·e_reg·el_c·reg_c·sidx·sx·sv·cx·cv·cm·fl_n·fl_pn).
사용: python exe/wolfip_html_color.py <판 디렉토리> <일반 3DGS plain.mp4> <출력 html> <임시 폴더>"""
import base64, glob, json, os, shutil, subprocess, sys
import numpy as np
from PIL import Image
from tqdm import tqdm
rd, pv, out, TMP = sys.argv[1:5]
os.makedirs(TMP, exist_ok=True)
FF = shutil.which("ffmpeg")
if not FF:
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
fs = sorted(glob.glob(f"{rd}/ip/f*.npz")); L = len(fs); T = L + 1
assert L >= 1 and fs[-1].endswith(f"f{L:03d}.npz"), (L, fs[-1:])
z0 = np.load(fs[0]); IDX = z0["sidx"]; n = len(IDX); N = len(z0["psi"])
rng = np.random.default_rng(1)
PSI = np.zeros((T, n)); PREG = np.zeros((T, n)); PC = {k: np.full((T, n), np.nan) for k in ("v", "h", "x", "y")}
CEL, CREG, CC, COFF, NC = [], [], {k: [] for k in ("v", "h", "x", "y")}, [0, 0], [0]
Eel, Ereg = [0.0], [0.0]
for i, f in enumerate(tqdm(fs, desc="프레임", ncols=70)):
    z = np.load(f); t = i + 1
    assert np.array_equal(z["sidx"], IDX), "표본 번호가 프레임마다 다름"
    nr = z["fl_n"].astype(np.float64); nr /= np.linalg.norm(nr); pn = float(z["fl_pn"])
    PSI[t] = z["psi"][IDX]; PREG[t] = z["e_reg"][IDX]
    sx, sv = z["sx"].astype(np.float64), z["sv"].astype(np.float64)
    PC["v"][t] = np.linalg.norm(sv, axis=1); PC["h"][t] = sx @ nr - pn; PC["x"][t] = sx[:, 0]; PC["y"][t] = sx[:, 1]
    ok = np.nonzero(z["cm"] > 0)[0]
    sel = np.sort(rng.choice(ok, min(20000, len(ok)), replace=False))
    cx, cv = z["cx"][sel].astype(np.float64), z["cv"][sel].astype(np.float64)
    CEL.append(z["el_c"][sel]); CREG.append(z["reg_c"][sel])
    CC["v"].append(np.linalg.norm(cv, axis=1)); CC["h"].append(cx @ nr - pn); CC["x"].append(cx[:, 0]); CC["y"].append(cx[:, 1])
    COFF.append(COFF[-1] + len(sel)); NC.append(int(len(ok)))
    Eel.append(float(z["e_el"].astype(np.float64).sum())); Ereg.append(float(z["e_reg"].astype(np.float64).sum()))
CEL, CREG = np.concatenate(CEL), np.concatenate(CREG)
CC = {k: np.concatenate(v) for k, v in CC.items()}


def rng_log(a):
    v = a[a > 0]
    return (float(np.floor(np.log10(np.percentile(v, 0.1)))), float(np.ceil(np.log10(v.max())))) if v.size else (-12.0, 0.0)


def q16(a, lo, hi):
    c = np.round(np.clip((np.log10(np.maximum(a, 1e-300)) - lo) / (hi - lo), 0, 1) * 65534).astype("<u2")
    c[~(a > 0)] = 65535
    return c


PX, PY, CXR, CYR = rng_log(PSI[1:]), rng_log(PREG[1:]), rng_log(CEL), rng_log(CREG)
CR = {}
for k in ("v", "h", "x", "y"):                                                     # 색 범위: 입자 표본 (프레임 1~끝) + 칸 표본 합쳐 하나, 고정
    a = np.concatenate([PC[k][1:].ravel(), CC[k]]); a = a[np.isfinite(a)]
    CR[k] = (0.0, float(np.percentile(a, 99.5))) if k == "v" else tuple(float(q) for q in np.percentile(a, (0.5, 99.5)))


def q8(a, lo, hi):
    c = np.round(np.clip((a - lo) / (hi - lo), 0, 1) * 254).astype(np.uint8); c[~np.isfinite(a)] = 255
    return c


b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
d = f"{TMP}/img"; os.makedirs(d, exist_ok=True)
for p in glob.glob(f"{d}/*"): os.remove(p)
subprocess.run([FF, "-v", "error", "-y", "-i", pv, f"{d}/%04d.png"], check=True)
pngs = sorted(glob.glob(f"{d}/*.png")); BOX = None
for p in pngs:
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.int16); m = np.abs(a - a[0, 0]).max(2) > 12
    if m.any():
        ys, xs = np.nonzero(m); b = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        BOX = b if BOX is None else (min(BOX[0], b[0]), min(BOX[1], b[1]), max(BOX[2], b[2]), max(BOX[3], b[3]))
w0, h0 = Image.open(pngs[0]).size; BOX = BOX or (0, 0, w0, h0)
mx, my = int(0.06 * (BOX[2] - BOX[0])) + 4, int(0.06 * (BOX[3] - BOX[1])) + 4
BOX = (max(0, BOX[0] - mx), max(0, BOX[1] - my), min(w0, BOX[2] + mx), min(h0, BOX[3] + my))
imgs = []
for i, p in enumerate(pngs):
    im = Image.open(p).convert("RGB").crop(BOX); im = im.resize((400, max(2, round(400 * im.size[1] / im.size[0]))), Image.LANCZOS)
    im.save(f"{d}/{i:04d}.jpg", quality=85); imgs.append("data:image/jpeg;base64," + base64.b64encode(open(f"{d}/{i:04d}.jpg", "rb").read()).decode())
assert len(imgs) == T, (len(imgs), T)
D = dict(T=T, n=n, N=N, px=PX, py=PY, cx=CXR, cy=CYR, coff=COFF, ncell=NC,
         a_x=b64(q16(PSI, *PX)), a_y=b64(q16(PREG, *PY)), b_x=b64(q16(CEL, *CXR)), b_y=b64(q16(CREG, *CYR)),
         pc={k: b64(q8(PC[k], *CR[k])) for k in CR}, cc={k: b64(q8(CC[k], *CR[k])) for k in CR}, cr=CR,
         Eel=Eel, Ereg=Ereg, img=imgs)
HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>wolf H9 탄성 대 정규화 에너지 (색 기준)</title>
<style>html,body{margin:0;height:100vh;font:13px sans-serif;background:#fff;color:#222;overflow:hidden}
#top{height:6vh;display:flex;align-items:center;gap:10px;padding:0 12px;border-bottom:1px solid #ddd}#top input[type=range]{flex:1}
.cb button{font-size:13px;padding:2px 8px;border:1px solid #bbb;background:#f6f6f6;cursor:pointer}.cb button.on{background:#2f6db5;color:#fff;border-color:#2f6db5}
#mid{height:52vh;display:flex}#mid .pl{flex:1;text-align:center;min-width:0}#mid canvas{width:100%;height:47vh}
#imgc{width:22vw;display:flex;flex-direction:column;align-items:center;justify-content:center}#imgc img{max-width:21vw;max-height:40vh}
#barr{height:7vh;display:flex;align-items:center;justify-content:center;border-top:1px solid #eee}#bar{width:60vw;height:6.5vh}
#bot{height:24vh;display:flex}#bot canvas{width:78vw;height:23vh}#hov{width:22vw;padding:8px;font-size:12px}
#note{height:11vh;padding:0 12px;font-size:12px;color:#555;overflow:hidden;border-top:1px solid #ddd}</style></head><body>
<div id="top"><button id="pb">재생</button><span>프레임 <b id="fn">0</b> / <span id="fT"></span></span><input id="sl" type="range" min="0" max="0" value="0">
<span class="cb">색 기준: <button data-k="none">단색</button><button data-k="v">속도</button><button data-k="h">높이</button><button data-k="x">x</button><button data-k="y">y</button></span><span id="tot"></span></div>
<div id="mid"><div class="pl"><b>(a) 입자</b> <span id="za"></span><br><canvas id="ca" width="900" height="560"></canvas></div>
<div class="pl"><b>(b) 칸</b> <span id="zb"></span><br><canvas id="cb" width="900" height="560"></canvas></div>
<div id="imgc"><div>일반 3DGS (이 프레임)</div><img id="im"></div></div>
<div id="barr"><canvas id="bar" width="1200" height="70"></canvas></div>
<div id="bot"><canvas id="cc" width="1500" height="280"></canvas><div id="hov">곡선 위에 마우스를 올리면 그 프레임 값, 누르면 그 프레임으로.</div></div>
<div id="note">(a) 가로 = 입자 탄성 에너지 밀도 ψ (그 팔분 칸 가우스 점, J/m³), 세로 = 입자에 나눈 정규화 에너지 (J). (b) 가로 = 칸 탄성 에너지 Σ V_q ψ_q (J), 세로 = 칸 정규화 에너지 (면 값을 양쪽 칸에 반씩, J). 두 축 로그, 축 범위 전 프레임 공통 고정, 0 인 값은 축 가장자리에 회색.
색: 속도 = 프레임 끝 입자 속도 크기 |v| (칸은 칸 입자 질량 가중 평균 속도 벡터의 크기), 높이 = 바닥 평면 법선 방향 거리 (바닥 = 0; 칸은 칸 입자 질량 가중 평균 위치), x·y = 월드 좌표. 색 범위는 입자·칸 공통, 전 프레임 고정 (프레임별 정규화 없음; 속도 0~99.5 분위수, 위치 0.5~99.5 분위수, 범위 밖은 끝 색).
점은 그리기용 무작위 표본이다 -- 입자 2만 개 (모든 프레임 같은 번호), 칸은 프레임마다 격자가 달라 프레임마다 무작위 2만 칸. 표본은 계산에 쓰지 않았고 곡선·전체 합은 전체 입자 <span id="NN"></span> 개로 계산했다. 정규화 = 실제로 켜진 면 항 (P1X 면 고스트 질량 몫). 이 뷰어의 값은 위치·속도를 함께 기록한 같은 명령의 다시 돈 판.</div>
<script>const D=__DATA__;
function d16(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return new Uint16Array(a.buffer)}
function d8(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a}
const ax=d16(D.a_x),ay=d16(D.a_y),bx=d16(D.b_x),by=d16(D.b_y),T=D.T,n=D.n;const PC={},CC={};for(const k in D.pc){PC[k]=d8(D.pc[k]);CC[k]=d8(D.cc[k])}
document.getElementById('NN').textContent=D.N;document.getElementById('fT').textContent=T-1;
const sl=document.getElementById('sl');sl.max=T-1;
const INK='#222',MUT='#888',GRID='#eee',C1='#2f6db5',C2='#c8553d';
// 순차 색표 (YlGnBu, 밝은 끝은 흰 바탕에서 보이게 0.15 부터)
const ST=[[255,255,217],[237,248,177],[199,233,180],[127,205,187],[65,182,196],[29,145,192],[34,94,168],[37,52,148],[8,29,88]];
function cmap(t){t=0.15+0.85*t;const x=t*(ST.length-1),i=Math.min(ST.length-2,Math.floor(x)),u=x-i;return'rgb('+ST[i].map((v,j)=>Math.round(v+(ST[i+1][j]-v)*u)).join(',')+')'}
const LUT=[];for(let v=0;v<255;v++)LUT.push(cmap(v/254));LUT.push('#bbb');
const NAME={v:['속도 크기','m/s'],h:['높이 (바닥 기준, 바닥 법선 방향)','m'],x:['x (월드)','m'],y:['y (월드)','m']};
let CK='none';
function frame(g,W,H,m,xl,yl,xt,yt){g.clearRect(0,0,W,H);const pw=W-m[0]-m[1],ph=H-m[2]-m[3];g.font='15px sans-serif';g.fillStyle=INK;
 g.textAlign='center';for(let k=Math.ceil(xl[0]);k<=xl[1];k++){const x=m[0]+(k-xl[0])/(xl[1]-xl[0])*pw;g.strokeStyle=GRID;g.beginPath();g.moveTo(x,m[2]);g.lineTo(x,m[2]+ph);g.stroke();g.fillText('1e'+k,x,H-m[3]+18)}
 g.textAlign='right';for(let k=Math.ceil(yl[0]);k<=yl[1];k++){const y=m[2]+ph-(k-yl[0])/(yl[1]-yl[0])*ph;g.strokeStyle=GRID;g.beginPath();g.moveTo(m[0],y);g.lineTo(m[0]+pw,y);g.stroke();g.fillText('1e'+k,m[0]-5,y+5)}
 g.strokeStyle=MUT;g.strokeRect(m[0],m[2],pw,ph);g.textAlign='center';g.fillText(xt,m[0]+pw/2,H-6);g.save();g.translate(16,m[2]+ph/2);g.rotate(-Math.PI/2);g.fillText(yt,0,0);g.restore();return[pw,ph]}
function scat(id,X,Y,C,o0,o1,xl,yl,xt,yt,col,zid,what){const c=document.getElementById(id),g=c.getContext('2d'),W=c.width,H=c.height,m=[78,16,14,46];
 const [pw,ph]=frame(g,W,H,m,xl,yl,xt,yt);let z=0;g.globalAlpha=.6;
 for(let i=o0;i<o1;i++){const zx=X[i]==65535,zy=Y[i]==65535;if(zx||zy)z++;g.fillStyle=(zx||zy)?'#aaa':(C?LUT[C[i]]:col);
  const x=m[0]+(zx?0:X[i]/65534)*pw,y=m[2]+ph-(zy?0:Y[i]/65534)*ph;g.fillRect(x-1.5,y-1.5,3,3)}g.globalAlpha=1;
 document.getElementById(zid).textContent=`표본 ${o1-o0} ${what} 중 0 값 (가장자리 회색) ${z}`}
function bar(){const c=document.getElementById('bar'),g=c.getContext('2d'),W=c.width,H=c.height;g.clearRect(0,0,W,H);g.font='15px sans-serif';g.fillStyle=INK;g.textAlign='center';
 if(CK=='none'){g.fillText('단색: (a) 입자 파랑, (b) 칸 주황 -- 색 기준 버튼으로 속도·높이·x·y 를 고른다',W/2,H/2+5);return}
 const [lo,hi]=D.cr[CK],x0=150,x1=W-150;for(let i=0;i<=254;i++){g.fillStyle=LUT[i];g.fillRect(x0+(x1-x0)*i/255,8,(x1-x0)/255+1,20)}
 g.strokeStyle=MUT;g.strokeRect(x0,8,x1-x0,20);g.fillStyle=INK;for(let j=0;j<=5;j++){const v=lo+(hi-lo)*j/5,x=x0+(x1-x0)*j/5;g.fillRect(x,28,1,5);g.fillText(v.toPrecision(3),x,48)}
 g.fillText(`${NAME[CK][0]} (${NAME[CK][1]}) -- 범위 ${lo.toPrecision(3)} ~ ${hi.toPrecision(3)} 전 프레임 고정, 입자·칸 공통`,W/2,66)}
const ev=a=>a.map(v=>v>0?Math.log10(v):null);const le=ev(D.Eel),lr=ev(D.Ereg);
const all=le.concat(lr).filter(v=>v!==null);const cyl=[Math.floor(Math.min(...all)),Math.ceil(Math.max(...all))];
function curves(t){const c=document.getElementById('cc'),g=c.getContext('2d'),W=c.width,H=c.height,m=[78,170,12,40];
 const [pw,ph]=frame(g,W,H,m,[0,T-1],cyl,'',' 전체 에너지 (J, 로그)');g.clearRect(m[0]-30,H-m[3]+2,pw+60,22);
 g.fillStyle=INK;g.textAlign='center';for(let k=0;k<T;k+=5)g.fillText(k,m[0]+k/(T-1)*pw,H-m[3]+18);g.fillText('프레임',m[0]+pw/2,H-4);
 const X=k=>m[0]+k/(T-1)*pw,Y=v=>m[2]+ph-(v-cyl[0])/(cyl[1]-cyl[0])*ph;
 [[le,C1,'전체 탄성'],[lr,C2,'전체 정규화']].forEach(([a,cl,nm],j)=>{g.strokeStyle=cl;g.lineWidth=2;g.beginPath();let s=false;
  a.forEach((v,k)=>{if(v===null){s=false;return}if(!s){g.moveTo(X(k),Y(v));s=true}else g.lineTo(X(k),Y(v))});g.stroke();
  if(a[t]!==null){g.fillStyle=cl;g.beginPath();g.arc(X(t),Y(a[t]),5,0,7);g.fill();g.strokeStyle='#fff';g.lineWidth=2;g.stroke()}
  g.fillStyle=cl;g.fillRect(W-m[1]+14,m[2]+14+j*26,18,4);g.fillStyle=INK;g.textAlign='left';g.fillText(nm,W-m[1]+38,m[2]+20+j*26)});
 g.strokeStyle=MUT;g.lineWidth=1;g.setLineDash([4,4]);g.beginPath();g.moveTo(X(t),m[2]);g.lineTo(X(t),m[2]+ph);g.stroke();g.setLineDash([])}
const fmt=v=>v===0?'0':v.toPrecision(4);
function show(t){document.getElementById('fn').textContent=t;sl.value=t;document.getElementById('im').src=D.img[t];
 document.getElementById('tot').textContent=t?`전체 탄성 ${fmt(D.Eel[t])} J · 전체 정규화 ${fmt(D.Ereg[t])} J`:'프레임 0: 걸음 전';
 const pc=CK=='none'?null:PC[CK],cc=CK=='none'?null:CC[CK];
 scat('ca',ax,ay,pc,t*n,(t+1)*n,D.px,D.py,'입자 탄성 에너지 밀도 ψ (J/m³, 로그)','입자 정규화 에너지 (J, 로그)',C1,'za','입자');
 scat('cb',bx,by,cc,D.coff[t],D.coff[t+1],D.cx,D.cy,'칸 탄성 에너지 (J, 로그)','칸 정규화 에너지 (J, 로그)',C2,'zb','칸');bar();curves(t)}
document.querySelectorAll('.cb button').forEach(b=>{b.onclick=()=>{CK=b.dataset.k;document.querySelectorAll('.cb button').forEach(x=>x.classList.toggle('on',x===b));show(+sl.value)}});
document.querySelector('.cb button[data-k=none]').classList.add('on');
const cc=document.getElementById('cc');function fr_of(e){const r=cc.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*cc.width;return Math.max(0,Math.min(T-1,Math.round((x-78)/(cc.width-78-170)*(T-1))))}
cc.onmousemove=e=>{const k=fr_of(e);document.getElementById('hov').innerHTML=`프레임 <b>${k}</b><br>전체 탄성 ${fmt(D.Eel[k])} J<br>전체 정규화 ${fmt(D.Ereg[k])} J`};
cc.onclick=e=>show(fr_of(e));sl.oninput=()=>show(+sl.value);let tm=null;const pb=document.getElementById('pb');
pb.onclick=()=>{if(tm){clearInterval(tm);tm=null;pb.textContent='재생'}else{pb.textContent='정지';tm=setInterval(()=>show((+sl.value+1)%T),200)}};show(1);</script></body></html>"""
open(out, "w").write(HTML.replace("__DATA__", json.dumps(D)))
print(f"[색 범위] {CR}", flush=True)
print("저장", out, os.path.getsize(out) // 1024, "KB", flush=True)
