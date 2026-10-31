# 절 AG.14: wolf 입자 적분점 판 (클램프 끔) 이 격자 상자 8 배 한도로 멈출 때 날아간 입자 고르기 + 뷰어 (torch 없이, 로컬 arm64)
# 사용: python3 exe/fem_ag14_wolf_viz.py <실행 디렉토리 (akpl_track.pt, pred.pt, summary.json)> <이름 A|B> <출력 html> <그림 디렉토리> <정답 wolf.pt>
# 문턱은 정답 (PG MPM) 궤적의 같은 프레임 값으로 정한다 (분포 그림에 함께 표시):
#   (가) 멈춘 형상 (= 멈춘 프레임 시작 = 직전 프레임 끝) 에서 첫 프레임 상자 (격자 상자 식과 같은 5 % 여유) 밖 거리 d_out > 정답 같은 프레임 d_out 최대
#   (나) 직전 프레임 동안 위치 변위 |Δx| > 정답 같은 프레임 |Δx| 최대 (그 프레임에 실제로 가장 많이 움직인 입자보다 더 움직인 것)
# 처음 이상해진 곳: 서브스텝 위치 증분 / h_s 가 정답 최대 속력 (멈춘 형상까지, v) 을 처음 넘은 프레임 (정확) 과 서브스텝
#   (서브스텝 통째 기록이 있는 마지막 3 프레임이면 정확, 아니면 그 이상인 첫 기록 단계 1·3·10·30·100·1000 m/s 의 첫 넘음, 그것도 없으면 그 프레임 최대 서브스텝).
import sys, os, json, base64, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptload_np as ptload
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f_ in ["/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf"]:
    if os.path.exists(f_): fm.fontManager.addfont(f_); plt.rcParams["font.family"] = fm.FontProperties(fname=f_).get_name()
plt.rcParams["axes.unicode_minus"] = False
D, NAME, OUT, FD, GTP = sys.argv[1:6]; os.makedirs(FD, exist_ok=True)
tk = ptload.load(os.path.join(D, "akpl_track.pt")); pr = ptload.load(os.path.join(D, "pred.pt"))
summ = json.load(open(os.path.join(D, "summary.json")))
P = pr["pred"].astype(np.float64); x0 = tk["x0"].astype(np.float64); done = P.shape[0]; Np = x0.shape[0]
stop = int(summ["diverged"]); N = int(tk["N"]); hs = float(tk["hs"]); LV = list(tk["lv"])
assert stop == done + 1, (stop, done)
frames = list(tk["frame"]); nf = len(frames)
pt = np.array(tk["plane_pt"], dtype=np.float64); nr = np.array(tk["plane_n"], dtype=np.float64)
hgt = lambda x: (x - pt) @ nr
xs = P[done - 1]; xp = P[done - 2] if done >= 2 else x0
# (가) 첫 프레임 상자 (box() 와 같은 5 % 여유) 밖 거리
lo0, hi0 = x0.min(0), x0.max(0); pad = 0.05 * (hi0 - lo0).max(); b0, b1 = lo0 - pad, hi0 + pad
dout = np.linalg.norm(np.maximum(0, np.maximum(b0 - xs, xs - b1)), axis=1)
disp = np.linalg.norm(xs - xp, axis=1)
GTd = ptload.load(GTP); GX = GTd["x"].astype(np.float64); GV = GTd["v"].astype(np.float64)
assert np.abs(GX[0] - x0).max() == 0
gxs, gxp = GX[done], GX[done - 1]                                       # 정답 같은 상태 (프레임 done 끝)
g_dout = np.linalg.norm(np.maximum(0, np.maximum(b0 - gxs, gxs - b1)), axis=1)
g_disp = np.linalg.norm(gxs - gxp, axis=1)
thr_o, thr_d = float(g_dout.max()), float(g_disp.max())
gap_o, gap_d = None, None
selo = dout > thr_o; seld = disp > thr_d
sel = selo | seld; idx = np.nonzero(sel)[0]
# 상자 한도를 만든 최소 집합 (밖으로 먼 것부터 빼며 남은 상자 부피 <= 8 x 첫 상자, 부피는 축 정렬 근사)
def boxvol(x):
    l, h_ = x.min(0), x.max(0); p = 0.05 * (h_ - l).max(); return np.prod(h_ - l + 2 * p)
V0 = boxvol(x0); order = np.argsort(-dout); keep = np.ones(Np, bool); npeel = 0
for k in order[:5000]:
    if boxvol(xs[keep]) <= 8 * V0: break
    keep[k] = False; npeel += 1
# 그림: 분포 + 문턱
def hist(v, thr, gap, ttl, fn, xl, gv=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    lv = np.log10(np.maximum(v, 1e-12))
    ax[0].hist(lv, bins=200, color="#888", label="우리 (멈춘 실행)"); ax[0].set_yscale("log"); ax[0].set_xlabel(f"log10 {xl}"); ax[0].set_ylabel("입자 수")
    if gv is not None:
        ax[0].hist(np.log10(np.maximum(gv, 1e-12)), bins=200, histtype="step", color="#2a6fd0", label="정답 같은 프레임"); ax[0].legend(fontsize=9)
    s = np.sort(v)[::-1]; ax[1].loglog(np.arange(1, s.size + 1), np.maximum(s, 1e-12), color="#333", label="우리 (멈춘 실행)"); ax[1].set_xlabel("순위 (큰 것부터)"); ax[1].set_ylabel(xl)
    if gv is not None:
        g = np.sort(gv)[::-1]; ax[1].loglog(np.arange(1, g.size + 1), np.maximum(g, 1e-12), color="#2a6fd0", label="정답 같은 프레임")
    ax[1].legend(fontsize=9)
    ax0 = np.log10(max(thr, 1e-12))
    ax[0].axvline(ax0, color="#d00", ls="--"); ax[1].axhline(max(thr, 1e-12), color="#d00", ls="--")
    ax[1].set_title(f"문턱 = 정답 같은 프레임 최대 {thr:.3g} m, 넘은 입자 {(v > thr).sum()}", fontsize=10)
    fig.suptitle(ttl); fig.tight_layout(); fig.savefig(fn, dpi=110); plt.close(fig)
hist(dout, thr_o, gap_o, f"wolf {NAME}: 멈춘 형상 (프레임 {stop} 시작) 의 첫 프레임 상자 밖 거리", os.path.join(FD, f"wolf_{NAME}_dout_hist.png"), "상자 밖 거리 (m)", g_dout)
hist(disp, thr_d, gap_d, f"wolf {NAME}: 직전 프레임 ({stop - 1}) 동안 위치 변위", os.path.join(FD, f"wolf_{NAME}_disp_hist.png"), "|Δx| (m)", g_disp)
# 처음 이상해진 프레임·서브스텝
MX = tk["mx"].astype(np.float64) / hs                                   # [nf, Np] 프레임별 서브스텝 최대 속력 (m/s)
q = float(np.linalg.norm(GV[:done + 1], axis=-1).max())                 # 정답 최대 속력 (멈춘 형상까지)
kv = next((k for k, l in enumerate(LV) if l >= q), len(LV) - 1); vthr = q
CR = tk["cr"].astype(np.int64)                                          # [nf, len(LV), Np]
cnt, mr, mrs, dmin = tk["cnt"].astype(np.int64), tk["mr"].astype(np.float64), tk["mrs"].astype(np.float64), tk["dmin"].astype(np.float64)
Xst = np.concatenate([x0[None], P[:-1]], 0)                             # 프레임 f (0 기준) 시작 위치
RF = [int(r) for r in tk["ring_frames"]]; RING = tk["ring"]             # [nr, N, Np] log10 |Δu| (float16)
lq = np.log10(q * hs)
_mv = np.linalg.norm(np.diff(GX, axis=0), axis=-1) > 1e-9                # [L, Np] 정답에서 그 프레임 동안 움직였는가
t_rel = np.where(_mv.any(0), np.argmax(_mv, 0) + 1, -1)                 # 정답에서 처음 움직인 프레임 (1 기준; 순차 방출로 풀린 프레임), 없으면 -1
recs = []
for i in idx:
    hit = np.nonzero(MX[:, i] > q)[0]                                   # 프레임 최대 속력 > 정답 최대 속력 (정확)
    f = int(hit[0]) if hit.size else int(np.argmax(MX[:, i]))
    if frames[f] in RF:                                                 # 서브스텝 통째 기록이 있는 프레임: 정확한 첫 서브스텝
        col = RING[RF.index(frames[f]), :, i].astype(np.float64); jj = np.nonzero(col > lq)[0]
        s1, s_how = (int(jj[0]) if jj.size else int(np.argmax(col))), "정확"
    elif CR[f, kv, i] >= 0:
        s1, s_how = int(CR[f, kv, i]), f"{LV[kv]:g} m/s 단계 첫 넘음"
    else:
        s1, s_how = int(tk["am"][f, i]), "그 프레임 최대 서브스텝"
    L = nf - 1
    recs.append(dict(i=int(i), box=bool(selo[i]), disp=bool(seld[i]), f1=frames[f] + 1, s1=s1, s_how=s_how, c1=int(cnt[f, i]), mr1=float(mr[f, i]), ms1=float(mrs[f, i]),
                     h1=float(hgt(Xst[f, i])), d1=float(dmin[f, i]), h0=float(hgt(x0[i])), c0=int(cnt[0, i]), mr0=float(mr[0, i]),
                     cL=int(cnt[L, i]), mrL=float(mr[L, i]), dL=float(dmin[L, i]), dout=float(dout[i]), disp_m=float(disp[i]), hL=float(hgt(xs[i])),
                     vmax=float(MX[:, i].max()), t=int(frames[f] * N + s1), tr=int(t_rel[i])))
# 비교 집단 (고르지 않은 입자) 같은 프레임 통계
def qs(a): return [round(float(x), 4) for x in np.quantile(a, [0.1, 0.5, 0.9])]
f1s = np.array([r["f1"] for r in recs]); fi = {fr + 1: k for k, fr in enumerate(frames)}
fm_ = int(np.bincount(f1s).argmax())
ns = ~sel; k0 = fi[fm_]
moved = (MX[k0] > 0.01) & ns
info = dict(name=NAME, run=D, stop_frame=stop, done=done, Np=int(Np), n_mark=int(idx.size), n_box=int(selo.sum()), n_disp=int(seld.sum()), n_both=int((selo & seld).sum()),
            thr_dout=thr_o, gap_dout=gap_o, thr_disp=thr_d, gap_disp=gap_d, n_peel_min=int(npeel), vthr=vthr, vthr_basis=q,
            first_frame_hist={str(k): int(v) for k, v in zip(*np.unique(f1s, return_counts=True))},
            sel_stats_at_first=dict(cnt=qs([r["c1"] for r in recs]), mr=qs([r["mr1"] for r in recs]), h=qs([r["h1"] for r in recs]), cnt1_frac=float(np.mean([r["c1"] == 1 for r in recs]))),
            sel_stats_first_frame=dict(cnt=qs([r["c0"] for r in recs]), h0=qs([r["h0"] for r in recs])),
            pop_stats_frame=dict(frame=fm_, cnt_all=qs(cnt[k0, ns]), mr_all=qs(mr[k0, ns]), h_all=qs(hgt(Xst[k0, ns])), cnt1_frac_all=float((cnt[k0, ns] == 1).mean()),
                                 cnt_moving=qs(cnt[k0, moved]) if moved.any() else None, mr_moving=qs(mr[k0, moved]) if moved.any() else None, n_moving=int(moved.sum())),
            pop_first_frame=dict(cnt=qs(cnt[0, ns]), h0=qs(hgt(x0[ns]))),
            mark=recs)
def _rd(o):
    if isinstance(o, float): return float(f"{o:.4g}")
    if isinstance(o, dict): return {k: _rd(v) for k, v in o.items()}
    if isinstance(o, list): return [_rd(v) for v in o]
    return o
recs = _rd(recs); info["mark"] = recs
json.dump(info, open(os.path.join(FD, f"wolf_{NAME}_mark_info.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "mark"}, ensure_ascii=False, indent=1))
# 뷰어
NS = 300000; sub = np.unique(np.linspace(0, Np - 1, min(NS, Np)).round().astype(np.int64))
up = nr.tolist()
ext0 = float((hi0 - lo0).max()); axl = lambda e: 10 ** np.floor(np.log10(e * 0.4))
def robust_box(x):
    l, h_ = np.quantile(x, 0.005, 0), np.quantile(x, 0.995, 0); return l.tolist(), h_.tolist()
fe = min(r["f1"] for r in recs); xb = x0 if fe == 1 else P[fe - 2]
SH = [("첫 프레임 형상", x0, False), (f"폭주 시작 프레임 {fe} 시작 형상", xb, True), (f"멈춘 프레임 {stop} 형상", xs, True)]
los, his, axl_, axs_, bmul = [], [], [], [], []
for _, xx, _c in SH:
    l_, h_ = robust_box(xx); los.append(l_); his.append(h_); e_ = float(np.max(np.array(h_) - np.array(l_)))
    axl_.append(float(axl(e_))); axs_.append(f"{axl(e_):g} m"); bmul.append(max(1.0, e_ / ext0))
meta = dict(n_all=int(Np), n_body=int(sub.size), n_mark=int(idx.size), n_box=info["n_box"], n_disp=info["n_disp"], n_both=info["n_both"], N=N, stop_frame=stop,
            vthr=vthr, up=up, frame_dt=float(tk["h"]), plane_pt=pt.tolist(), lo=los, hi=his, clip_r=1.5, clip=[c for _, _, c in SH], shapes=[n for n, _, _ in SH],
            rule=f"(가) 첫 프레임 상자 밖 거리 > {thr_o:.3g} m, (나) 프레임 {stop - 1} 동안 변위 > {thr_d:.3g} m (둘 다 정답 같은 프레임의 최대값)",
            axis_len=axl_, axis_lbl=axs_, bsize_mul=bmul, mark=recs)
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a.astype(np.float32)).tobytes()).decode()
T = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz", "wolf_divparticles_tmpl.html"), encoding="utf-8").read()
ttl = f"wolf (PG 모래) 명시적 FEM 입자 적분점 판 {NAME}: 날아간 입자 (멈춘 프레임 {stop})"
html = (T.replace("__TITLE__", ttl).replace("__META__", json.dumps(meta, ensure_ascii=False))
        .replace("__BODYS__", json.dumps([b64(xx[sub]) for _, xx, _ in SH])).replace("__MARKS__", json.dumps([b64(xx[idx]) for _, xx, _ in SH])))
open(OUT, "w", encoding="utf-8").write(html)
print("뷰어", OUT, "본체", sub.size, "날아간", idx.size, "MB", round(os.path.getsize(OUT) / 1e6, 2))
