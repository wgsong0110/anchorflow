"""wolf H9 탄성-정규화 산점도 단일 HTML 뷰어 (그림 base64 포함).
(a) 입자: 입자 탄성 에너지 밀도 ψ (J/m³) 대 입자에 나눈 정규화 에너지 (J) · (b) 칸: 칸 탄성 에너지 Σ_q∈칸 V_q ψ_q (J) 대 칸 정규화 에너지 (면 반씩, J).
두 축 로그, 범위는 전 프레임 공통 고정. 0 인 값은 축 가장자리에 회색으로 (수는 화면에). 아래에 프레임별 전체 탄성·정규화 에너지 곡선 (같은 로그 축 하나), 옆에 그 프레임 렌더.
사용: python exe/wolfip_html.py <an 디렉토리 (sample.npz ip.json)> <plain.mp4> <출력 html> <임시 폴더>"""
import base64, glob, json, os, shutil, subprocess, sys
import numpy as np
ad, pv, out, TMP = sys.argv[1:5]
os.makedirs(TMP, exist_ok=True)
FF = shutil.which("ffmpeg")
if not FF:
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
Z = np.load(f"{ad}/sample.npz"); J = json.load(open(f"{ad}/ip.json"))
T = J["T"]; n = int(len(Z["idx"]))


def rng(a):
    v = a[a > 0]
    return (float(np.floor(np.log10(np.percentile(v, 0.1)))), float(np.ceil(np.log10(v.max())))) if v.size else (-12.0, 0.0)


def q16(a, lo, hi):
    c = np.round(np.clip((np.log10(np.maximum(a, 1e-300)) - lo) / (hi - lo), 0, 1) * 65534).astype("<u2")
    c[~(a > 0)] = 65535                                                            # 0 (또는 음수) 표시
    return c


b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
PX, PY = rng(Z["psi"][1:]), rng(Z["reg"][1:])
CX, CY = rng(Z["cel"]), rng(Z["creg"])
d = f"{TMP}/img"; os.makedirs(d, exist_ok=True)
for p in glob.glob(f"{d}/*.jpg"): os.remove(p)
for p in glob.glob(f"{d}/*.png"): os.remove(p)
subprocess.run([FF, "-v", "error", "-y", "-i", pv, f"{d}/%04d.png"], check=True)
from PIL import Image
pngs = sorted(glob.glob(f"{d}/*.png")); BOX = None
for p in pngs:                                                                     # 모든 프레임 같은 자르기 상자 (배경과 다른 픽셀 합집합 + 여백)
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
D = dict(T=T, n=n, N=J["N"], px=PX, py=PY, cx=CX, cy=CY, coff=[int(q) for q in Z["coff"]], ncell=J["cell_n"],
         a_x=b64(q16(Z["psi"], *PX)), a_y=b64(q16(Z["reg"], *PY)), b_x=b64(q16(Z["cel"], *CX)), b_y=b64(q16(Z["creg"], *CY)),
         Eel=J["E_el"], Ereg=J["E_reg"], img=imgs)
HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>wolf H9 탄성 대 정규화 에너지</title>
<style>html,body{margin:0;height:100vh;font:13px sans-serif;background:#fff;color:#222;overflow:hidden}
#top{height:6vh;display:flex;align-items:center;gap:12px;padding:0 12px;border-bottom:1px solid #ddd}#top input[type=range]{flex:1}
#mid{height:58vh;display:flex}#mid .pl{flex:1;text-align:center;min-width:0}#mid canvas{width:100%;height:52vh}
#imgc{width:22vw;display:flex;flex-direction:column;align-items:center;justify-content:center}#imgc img{max-width:21vw;max-height:40vh}
#bot{height:26vh;display:flex}#bot canvas{width:78vw;height:25vh}#hov{width:22vw;padding:8px;font-size:12px}
#note{height:10vh;padding:0 12px;font-size:12px;color:#555;overflow:hidden;border-top:1px solid #ddd}
button{font-size:13px}</style></head><body>
<div id="top"><button id="pb">재생</button><span>프레임 <b id="fn">0</b> / <span id="fT"></span></span><input id="sl" type="range" min="0" max="0" value="0"><span id="tot"></span></div>
<div id="mid"><div class="pl"><b>(a) 입자</b> <span id="za"></span><br><canvas id="ca" width="900" height="620"></canvas></div>
<div class="pl"><b>(b) 칸</b> <span id="zb"></span><br><canvas id="cb" width="900" height="620"></canvas></div>
<div id="imgc"><div>일반 3DGS (이 프레임)</div><img id="im"></div></div>
<div id="bot"><canvas id="cc" width="1500" height="300"></canvas><div id="hov">곡선 위에 마우스를 올리면 그 프레임 값, 누르면 그 프레임으로.</div></div>
<div id="note">(a) 가로 = 입자 탄성 에너지 밀도 ψ (그 팔분 칸 가우스 점, J/m³), 세로 = 입자에 나눈 정규화 에너지 (J). (b) 가로 = 칸 탄성 에너지 Σ V_q ψ_q (J), 세로 = 칸 정규화 에너지 (면 값을 양쪽 칸에 반씩, J).
두 축 로그, 범위는 전 프레임 공통 고정. 0 인 값은 축 가장자리에 회색 점으로 두었다.
점은 그리기용으로만 뽑은 무작위 표본이다 -- 입자는 2만 개를 모든 프레임 같은 번호로, 칸은 프레임마다 격자가 달라 프레임마다 무작위 2만 칸. 표본은 계산에 쓰지 않았고 아래 곡선·전체 합은 전체 입자 <span id="NN"></span> 개로 계산했다.
정규화 = 실제로 켜진 면 항 (고스트 질량 몫; wolf κ 0 이라 P1X 강성·가벼움 벌점 0). 프레임 0 은 걸음 전이라 비어 있다.</div>
<script>const D=__DATA__;
function dec(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return new Uint16Array(a.buffer)}
const ax=dec(D.a_x),ay=dec(D.a_y),bx=dec(D.b_x),by=dec(D.b_y),T=D.T,n=D.n;
document.getElementById('NN').textContent=D.N;document.getElementById('fT').textContent=T-1;
const sl=document.getElementById('sl');sl.max=T-1;
const INK='#222',MUT='#888',GRID='#eee',C1='#2f6db5',C2='#c8553d';
function frame(g,W,H,m,xl,yl,xt,yt){g.clearRect(0,0,W,H);const pw=W-m[0]-m[1],ph=H-m[2]-m[3];g.font='15px sans-serif';g.fillStyle=INK;
 g.textAlign='center';for(let k=Math.ceil(xl[0]);k<=xl[1];k++){const x=m[0]+(k-xl[0])/(xl[1]-xl[0])*pw;g.strokeStyle=GRID;g.beginPath();g.moveTo(x,m[2]);g.lineTo(x,m[2]+ph);g.stroke();g.fillText('1e'+k,x,H-m[3]+18)}
 g.textAlign='right';for(let k=Math.ceil(yl[0]);k<=yl[1];k++){const y=m[2]+ph-(k-yl[0])/(yl[1]-yl[0])*ph;g.strokeStyle=GRID;g.beginPath();g.moveTo(m[0],y);g.lineTo(m[0]+pw,y);g.stroke();g.fillText('1e'+k,m[0]-5,y+5)}
 g.strokeStyle=MUT;g.strokeRect(m[0],m[2],pw,ph);g.textAlign='center';g.fillText(xt,m[0]+pw/2,H-6);g.save();g.translate(16,m[2]+ph/2);g.rotate(-Math.PI/2);g.fillText(yt,0,0);g.restore();return[pw,ph]}
function scat(id,X,Y,o0,o1,xl,yl,xt,yt,col,zid,what){const c=document.getElementById(id),g=c.getContext('2d'),W=c.width,H=c.height,m=[78,16,14,46];
 const [pw,ph]=frame(g,W,H,m,xl,yl,xt,yt);let z=0;g.globalAlpha=.55;
 for(let i=o0;i<o1;i++){const zx=X[i]==65535,zy=Y[i]==65535;if(zx||zy)z++;g.fillStyle=(zx||zy)?'#aaa':col;
  const x=m[0]+(zx?0:X[i]/65534)*pw,y=m[2]+ph-(zy?0:Y[i]/65534)*ph;g.fillRect(x-1.5,y-1.5,3,3)}g.globalAlpha=1;
 document.getElementById(zid).textContent=`표본 ${o1-o0} ${what} 중 0 값 (가장자리 회색) ${z}`}
const ev=(a)=>a.map(v=>v>0?Math.log10(v):null);const le=ev(D.Eel),lr=ev(D.Ereg);
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
 document.getElementById('tot').textContent=t?`전체 탄성 ${fmt(D.Eel[t])} J · 전체 정규화 ${fmt(D.Ereg[t])} J · 칸 ${D.ncell[t]} 개`:'프레임 0: 걸음 전';
 scat('ca',ax,ay,t*n,(t+1)*n,D.px,D.py,'입자 탄성 에너지 밀도 ψ (J/m³, 로그)','입자 정규화 에너지 (J, 로그)',C1,'za','입자');
 scat('cb',bx,by,D.coff[t],D.coff[t+1],D.cx,D.cy,'칸 탄성 에너지 (J, 로그)','칸 정규화 에너지 (J, 로그)',C2,'zb','칸');curves(t)}
const cc=document.getElementById('cc');function fr_of(e){const r=cc.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*cc.width;return Math.max(0,Math.min(T-1,Math.round((x-78)/(cc.width-78-170)*(T-1))))}
cc.onmousemove=e=>{const k=fr_of(e);document.getElementById('hov').innerHTML=`프레임 <b>${k}</b><br>전체 탄성 ${fmt(D.Eel[k])} J<br>전체 정규화 ${fmt(D.Ereg[k])} J`};
cc.onclick=e=>show(fr_of(e));sl.oninput=()=>show(+sl.value);let tm=null;const pb=document.getElementById('pb');
pb.onclick=()=>{if(tm){clearInterval(tm);tm=null;pb.textContent='재생'}else{pb.textContent='정지';tm=setInterval(()=>show((+sl.value+1)%T),200)}};show(1);</script></body></html>"""
open(out, "w").write(HTML.replace("__DATA__", json.dumps(D)))
print("저장", out, os.path.getsize(out) // 1024, "KB", flush=True)
