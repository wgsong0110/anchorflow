"""레고 (A) 원래 FCR 폭증 프레임 (인자, 기본 21·22) 걸음·뉴턴 시점별 그림 (exe/stab_divsteps.sh 덤프 wd_f<프레임>.npz + _pos.npz).
사용: python exe/divsteps_plot.py <덤프 디렉토리> <PNG 경로 (확장자 빼고)> <HTML 경로>
시점 = 프레임마다 걸음 1·2 x [시작점 (관성 예측), 뉴턴 1 끝, 뉴턴 2 끝, 걸음 끝]. 노드 이동 = 그 시점으로 들어온 노드 변위 변화 (앞 시점 대비; 프레임 첫 시점은 프레임 시작 대비).
"""
import base64
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from matplotlib import font_manager
import os
for _f in ("NotoSansCJKkr-Regular.otf", "NotoSansCJKkr-Bold.otf"):
    if os.path.exists(os.path.expanduser(f"~/.fonts/{_f}")):
        font_manager.fontManager.addfont(os.path.expanduser(f"~/.fonts/{_f}"))
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
D, PNG, HTML = sys.argv[1], sys.argv[2], sys.argv[3]
FR = tuple(int(v) for v in sys.argv[4].split(",")) if len(sys.argv) > 4 else (21, 22)
TAG = sys.argv[5] if len(sys.argv) > 5 else ""
SN = ["시작점 (관성 예측)", "뉴턴 1 끝", "뉴턴 2 끝", "걸음 끝"]

snaps = []                                                                        # 시점 목록
for f in FR:
    z = np.load(f"{D}/wd_f{f}.npz")
    p = np.load(f"{D}/wd_f{f}_pos.npz")
    K, nt = int(z["K"][0]), int(z["nt"][0])
    h = float(z["cellh"][0])
    A = p["A"]
    lo, nb = p["lo"].astype(float), p["nb"].astype(float)
    nX = p["n_X"]
    Ys = []
    for k in range(K):
        Ys.append(z["n_Y0"][k])
        for i in range(nt):
            Ys.append(z["n_Y"][k * nt + i])
        Ys.append(z["n_U"][k])
    qJ = z["q_J"]; qnp = z["q_np"]; qinv = p["q_inv"]
    corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], float)
    gbox = (lo[None] + corners * (nb[None] - 1)) @ A.T                             # 노드 상자 모서리 (월드)
    Ai = np.linalg.inv(A)
    for s, Y in enumerate(Ys):
        k, ph = s // (nt + 2), s % (nt + 2)
        prev = Ys[s - 1] if s > 0 else np.zeros_like(Y)
        mv = np.linalg.norm(Y - prev, axis=1) / h
        big = mv > 1.0
        px = p["p_xs"][s]
        okp = np.isfinite(px).all(1)
        fb = np.floor(px[okp] @ Ai.T - 0.5)
        rebox = int(np.prod(fb.max(0) - fb.min(0) + 3)) if okp.any() else -1       # 그 시점 입자로 다시 잡으면 노드 상자 크기
        inv = qJ[qinv, s] <= 0
        full = qnp[qinv] > 0
        qx = p["q_xs"][s]
        snaps.append(dict(f=f, k=k, ph=ph, name=f"프레임 {f} 걸음 {k + 1} {SN[ph]}", px=px,
                          qf=qx[inv & full], qe=qx[inv & ~full], nf=int((inv & full).sum()), ne=int((inv & ~full).sum()),
                          a0=(nX + prev)[big], a1=(nX + Y)[big], amv=mv[big], mvmax=float(mv.max()), nbox=int(np.prod(nb)), rebox=rebox, gbox=gbox,
                          nr=p["nr"], pn=float(p["pn"][0]), h=h))
    print(f, "시작 위치로 다시 잡은 상자", int(np.prod(np.ptp(np.floor(p["p_x0"] @ Ai.T - 0.5), 0) + 3)), "실제", int(np.prod(nb)))

# 공통 축 범위: 모든 시점 입자 + 화살표 끝 (비유한 제외)
allp = np.concatenate([s["px"][np.isfinite(s["px"]).all(1)] for s in snaps] + [s["a1"] for s in snaps if len(s["a1"])])
lo3, hi3 = np.percentile(allp, 0, 0), np.percentile(allp, 100, 0)
ctr, half = (lo3 + hi3) / 2, (hi3 - lo3).max() / 2 * 1.02
# 확대 범위: 프레임 21 첫 시점 입자 경계 (몸통)
p0 = snaps[0]["px"][np.isfinite(snaps[0]["px"]).all(1)]
zc, zh = (p0.min(0) + p0.max(0)) / 2, (p0.max(0) - p0.min(0)).max() / 2 * 1.15
vmax = max(s["mvmax"] for s in snaps)
NORM = mcolors.LogNorm(vmin=1.0, vmax=max(vmax, 1.01))
CM = plt.get_cmap("plasma")
up = snaps[0]["nr"] / np.linalg.norm(snaps[0]["nr"])
print("바닥 법선", up, "최대 노드 이동 칸", vmax)
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def floor_quad(nr, pn, c, r):
    """바닥 평면 nr·x = pn 의 사각형 (c 중심 반경 r)."""
    nr = nr / np.linalg.norm(nr)
    t1 = np.cross(nr, [1, 0, 0] if abs(nr[0]) < 0.9 else [0, 1, 0]); t1 /= np.linalg.norm(t1)
    t2 = np.cross(nr, t1)
    c0 = c - (c @ nr - pn) * nr
    return [c0 + r * (a * t1 + b * t2) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def panel(ax, s, c, r, view, title=None):
    px = s["px"]
    ax.scatter(px[:, 0], px[:, 1], px[:, 2], s=0.05, c="0.55", alpha=0.08, linewidths=0, rasterized=True)
    if len(s["qe"]):
        ax.scatter(*s["qe"].T, s=3, c="#f28e2b", alpha=0.9, linewidths=0, rasterized=True)
    if len(s["qf"]):
        ax.scatter(*s["qf"].T, s=3, c="#d62728", alpha=0.9, linewidths=0, rasterized=True)
    if len(s["a1"]):
        segs = np.stack([s["a0"], s["a1"]], 1)
        ax.add_collection3d(Line3DCollection(segs, colors=CM(NORM(s["amv"])), linewidths=0.6))
        ax.scatter(*s["a1"].T, s=1.5, c=CM(NORM(s["amv"])), linewidths=0, rasterized=True)
    g = s["gbox"]
    ax.add_collection3d(Line3DCollection([[g[a], g[b]] for a, b in EDGES], colors="k", linewidths=0.6, linestyles="--"))
    ax.add_collection3d(Poly3DCollection([floor_quad(s["nr"], s["pn"], c, r)], facecolors=(0.3, 0.5, 0.8, 0.12), edgecolors=(0.3, 0.5, 0.8, 0.4)))
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(*view)
    ax.tick_params(labelsize=5, pad=0)
    if title:
        ax.set_title(title, fontsize=7, pad=2)


# 보는 방향: 위 = 바닥 법선 쪽 축
upax = int(np.argmax(np.abs(up)))
VIEWS = {2: [(8, -90, 0, "정면"), (8, 0, 0, "측면")], 1: [(8, -90, 90, "정면"), (8, 0, 90, "측면")]}.get(upax, [(8, -90, 0, "정면"), (8, 0, 0, "측면")])


def title(s):
    return (f"{s['name']}\n뒤집힌 점 찬 {s['nf']} · 빈 {s['ne']}\n노드 상자 {s['nbox']:,} (그 시점 입자로 {s['rebox']:,})\n최대 노드 이동 {s['mvmax']:.1f} 칸")


def sheet(sel, c, r, path, sup):
    nc = 8
    nrow = (len(sel) + nc - 1) // nc
    fig = plt.figure(figsize=(nc * 2.6, nrow * 2 * 2.5 + 0.8))
    for j, s in enumerate(sel):
        rr, cc = divmod(j, nc)
        for v, (e, a, ro, nm) in enumerate(VIEWS):
            ax = fig.add_subplot(nrow * 2, nc, (rr * 2 + v) * nc + cc + 1, projection="3d")
            panel(ax, s, c, r, (e, a, ro), title(s) if v == 0 else None)
            ax.text2D(0.02, 0.02, nm, transform=ax.transAxes, fontsize=6)
    fig.suptitle(sup, fontsize=10)
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CM)
    cax = fig.add_axes([0.92, 0.3, 0.008, 0.4])
    fig.colorbar(sm, cax=cax).set_label("노드 이동 (칸, 1 칸 넘는 노드만, 로그)", fontsize=7)
    fig.text(0.01, 0.005, "회색 = 입자 (그 시점 노드 변위로 보간), 빨강 = 뒤집힌 가우스 점 (det ≤ 0) 찬 팔분, 주황 = 빈 팔분, 화살표 = 앞 시점 -> 이 시점 노드 위치, 검은 점선 = 이 프레임 노드 상자, 파랑 면 = 바닥. 모든 칸 공통 축·색 범위.", fontsize=7)
    fig.subplots_adjust(left=0.01, right=0.91, top=0.93, bottom=0.03, wspace=0.02, hspace=0.25)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("저장", path)


FS = "·".join(str(f) for f in FR)
sheet(snaps, ctr, half, PNG + ".png", f"레고 (A) 원래 FCR{TAG}, 노드 상자 폭증 앞뒤 프레임 {FS} -- 걸음·뉴턴 시점별 (전체 범위)")
sheet(snaps, zc, zh, PNG + "_zoom.png", f"같은 그림 확대{TAG} (프레임 {FR[0]} 시작 몸통 범위, 모든 칸 같은 범위)")

# HTML (plotly.js, 시점 슬라이더)
def b64(a, dt=np.float32):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode()


J = []
for s in snaps:
    px = s["px"]; okp = np.isfinite(px).all(1)
    J.append(dict(name=s["name"], title=title(s).replace("\n", " · "), px=b64(px[okp]), qf=b64(s["qf"]), qe=b64(s["qe"]), a0=b64(s["a0"]), a1=b64(s["a1"]), amv=b64(s["amv"]),
                  box=s["gbox"].tolist()))
fq = floor_quad(snaps[0]["nr"], snaps[0]["pn"], ctr, half)
meta = dict(c=ctr.tolist(), r=float(half), zc=zc.tolist(), zr=float(zh), vmax=float(vmax), edges=EDGES, floor=[q.tolist() for q in fq], up=up.tolist())
html = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>레고 (A) 원래 FCR 폭증 프레임 __FS__ 시점별</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{font-family:'Apple SD Gothic Neo','Noto Sans KR',sans-serif;margin:8px}#pl{width:100%;height:82vh}#t{font-size:14px;margin:4px 0}</style></head><body>
<div>시점 <input type="range" id="s" min="0" max="__N__" value="0" style="width:60%"> <button id="pv">◀</button><button id="nx">▶</button>
 <b>__TAG__</b> <label><input type="checkbox" id="zm"> 확대 (프레임 __F0__ 시작 몸통 범위)</label> <label><input type="checkbox" id="sp" checked> 입자 표시</label></div>
<div id="t"></div><div id="pl"></div>
<div style="font-size:12px">회색 = 입자 (그 시점 노드 변위로 보간), 빨강 = 뒤집힌 가우스 점 (det ≤ 0) 찬 팔분, 주황 = 빈 팔분, 선 = 앞 시점 -> 이 시점 노드 위치 (1 칸 넘는 노드, 색 = 이동 칸, 로그, 모든 시점 공통 1~__VM__), 점선 = 노드 상자, 파랑 = 바닥. 축 범위 모든 시점 공통.</div>
<script>
const S=__S__, M=__M__;
function f32(b){const u=Uint8Array.from(atob(b),c=>c.charCodeAt(0));return new Float32Array(u.buffer);}
function xyz(a){const x=[],y=[],z=[];for(let i=0;i<a.length;i+=3){x.push(a[i]);y.push(a[i+1]);z.push(a[i+2]);}return [x,y,z];}
const cache={};
function col(v){const t=Math.log(Math.max(v,1))/Math.log(M.vmax);const c=[[13,8,135],[126,3,168],[204,71,120],[248,149,64],[240,249,33]];const p=Math.min(t,0.9999)*(c.length-1),i=Math.floor(p),f=p-i;return 'rgb('+c[i].map((q,j)=>Math.round(q+(c[i+1][j]-q)*f)).join(',')+')';}
function traces(k){const s=S[k];if(!cache[k]){const P=xyz(f32(s.px)),QF=xyz(f32(s.qf)),QE=xyz(f32(s.qe)),A0=f32(s.a0),A1=f32(s.a1),MV=f32(s.amv);
 const ax=[],ay=[],az=[],ac=[],mx=[],my=[],mz=[],mc=[],mt=[];
 for(let i=0;i<MV.length;i++){ax.push(A0[3*i],A1[3*i],null);ay.push(A0[3*i+1],A1[3*i+1],null);az.push(A0[3*i+2],A1[3*i+2],null);const c=col(MV[i]);ac.push(c,c,c);mx.push(A1[3*i]);my.push(A1[3*i+1]);mz.push(A1[3*i+2]);mc.push(Math.log10(MV[i]));mt.push('이동 '+MV[i].toFixed(2)+' 칸');}
 const bx=[],by=[],bz=[];for(const [a,b] of M.edges){bx.push(s.box[a][0],s.box[b][0],null);by.push(s.box[a][1],s.box[b][1],null);bz.push(s.box[a][2],s.box[b][2],null);}
 cache[k]=[{type:'scatter3d',mode:'markers',name:'입자',x:P[0],y:P[1],z:P[2],marker:{size:1,color:'rgb(140,140,140)',opacity:0.15},hoverinfo:'skip'},
 {type:'scatter3d',mode:'markers',name:'뒤집힌 점 (찬 팔분)',x:QF[0],y:QF[1],z:QF[2],marker:{size:2.5,color:'#d62728'}},
 {type:'scatter3d',mode:'markers',name:'뒤집힌 점 (빈 팔분)',x:QE[0],y:QE[1],z:QE[2],marker:{size:2.5,color:'#f28e2b'}},
 {type:'scatter3d',mode:'lines',name:'노드 이동 > 1 칸',x:ax,y:ay,z:az,line:{color:ac,width:3},hoverinfo:'skip'},
 {type:'scatter3d',mode:'markers',name:'노드 (이동 끝)',x:mx,y:my,z:mz,text:mt,hoverinfo:'text',marker:{size:2,color:mc,cmin:0,cmax:Math.log10(M.vmax),colorscale:'Plasma',colorbar:{title:'log10 이동 (칸)',len:0.5}}},
 {type:'scatter3d',mode:'lines',name:'노드 상자',x:bx,y:by,z:bz,line:{color:'black',width:2,dash:'dash'},hoverinfo:'skip'},
 {type:'mesh3d',name:'바닥',x:M.floor.map(q=>q[0]),y:M.floor.map(q=>q[1]),z:M.floor.map(q=>q[2]),i:[0,0],j:[1,2],k:[2,3],color:'rgb(80,130,200)',opacity:0.15,hoverinfo:'skip'}];}
 const t=cache[k].slice();if(!document.getElementById('sp').checked)t[0]=Object.assign({},t[0],{visible:false});return t;}
let cam=null;
function draw(){const k=+document.getElementById('s').value;document.getElementById('t').textContent=(k+1)+'/'+S.length+' · '+S[k].title;
 const z=document.getElementById('zm').checked,c=z?M.zc:M.c,r=z?M.zr:M.r;
 const ax=(i)=>({range:[c[i]-r,c[i]+r],autorange:false});
 const lay={margin:{l:0,r:0,t:0,b:0},uirevision:'keep',legend:{x:0,y:1},scene:{xaxis:ax(0),yaxis:ax(1),zaxis:ax(2),aspectmode:'cube',camera:{up:{x:M.up[0],y:M.up[1],z:M.up[2]}}}};
 Plotly.react('pl',traces(k),lay);}
['s','zm','sp'].forEach(i=>document.getElementById(i).addEventListener('input',draw));
document.getElementById('pv').onclick=()=>{const e=document.getElementById('s');e.value=Math.max(0,+e.value-1);draw();};
document.getElementById('nx').onclick=()=>{const e=document.getElementById('s');e.value=Math.min(S.length-1,+e.value+1);draw();};
draw();
</script></body></html>"""
html = html.replace("__FS__", FS).replace("__TAG__", TAG).replace("__F0__", str(FR[0])).replace("__N__", str(len(snaps) - 1)).replace("__VM__", f"{vmax:.0f}").replace("__S__", json.dumps(J)).replace("__M__", json.dumps(meta))
open(HTML, "w").write(html)
print("저장", HTML)
for s in snaps:
    print(f"{s['name']}: 찬 {s['nf']} 빈 {s['ne']} 상자 {s['nbox']} 입자로 {s['rebox']} 최대 이동 {s['mvmax']:.2f} 칸, >1 칸 노드 {len(s['amv'])}")
