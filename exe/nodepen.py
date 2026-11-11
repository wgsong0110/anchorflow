"""절 CM111 진단 (읽기만, 결과 불변): 튄 입자의 받침 노드가 바닥 밑으로 내려간 노드였는가.
사용: python exe/nodepen.py <판이름> <판 디렉토리 (pred.pt·e1.jsonl·nodes/f*.npz)> <cfg.json> <data wolf.pt> <out.json>
  노드 덤프 (exe/fem_an_rt_sim.py --dump_nodes): 프레임 f 의 걸음 = 프레임 f-1 -> f. 격자는 걸음 시작 입자 위치로 만든 고정 격자 (노드 L = A (정수 번호 + bmin)), 노드 걸음 끝 위치 = A L + U.
  노드 높이 h = x·n - PN (n·PN = 바닥 법선·점). 바닥 밑 = h < 0, 깊이 = -h. 세 가지로 잼:
    E = 걸음 끝 위치, M = 프레임 안 모든 걸음 끝 (U 사본) 중 가장 낮은 값 (걸음 중간), C = 걸음 시작 격자 위치는 바닥 위(h0 >= 0)인데 끝은 아래 (E 이면서 내려간 것).
  입자 받침 = base (pb) + {0,1,2}^3 27 노드, 가중 = 이차 B-스플라인 (fx).
  튐 정의는 exe/floorpen.py 와 같음 (새 이탈, 속도 급증). 창: W1 = 그 프레임 걸음, W3 = f-2..f 걸음, P3 = f-3..f-1 걸음.
"""
import json, math, os, sys
import numpy as np
import torch

name, rd, cfgp, datap, outp = sys.argv[1:6]
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
cfg = json.load(open(cfgp))
pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
NR = np.array(pl["normal"], dtype=np.float64); NR = NR / np.linalg.norm(NR)
PT = np.array([float(q) for q in pl["point"]], dtype=np.float64); PN = float(PT @ NR)
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100)); DH = 0.5 * DXG
H = float(cfg["frame_dt"])
print(f"[{name}] 바닥 점 {PT.tolist()} 법선 {NR.tolist()} d̂ {DH}", flush=True)
d = torch.load(os.path.join(rd, "pred.pt"), map_location="cpu", weights_only=False)
P = d["pred"]
X0 = torch.load(datap, map_location="cpu", weights_only=False)["x"][0]
steps = sorted(int(f[1:4]) for f in os.listdir(os.path.join(rd, "nodes")) if f.endswith(".npz"))
T = min(P.shape[0], max(steps)); N = P.shape[1]
print(f"[{name}] pred {tuple(P.shape)} 노드 덤프 {len(steps)} 개 (최대 f{max(steps)}) 사용 T {T}", flush=True)
frames = [X0.float()] + [P[i].float() for i in range(T)]
NRt = torch.tensor(NR, dtype=torch.float32)

# ---------- 튐 (floorpen.py 와 같음)
e1 = {}
if os.path.exists(os.path.join(rd, "e1.jsonl")):
    for ln in open(os.path.join(rd, "e1.jsonl")):
        r = json.loads(ln); e1[r["f"]] = r
CELLS = {"NODE_H(cell2*DXG*sqrt2)": 2 * DXG * math.sqrt(2), "2*DXG": 2 * DXG, "DXG": DXG, "DXG*sqrt2": DXG * math.sqrt(2)}


def farinfo(x):
    ok = torch.isfinite(x).all(-1)
    xs = x[ok][::7].double()
    q = torch.quantile(xs, torch.tensor([0.001, 0.999], dtype=torch.float64), dim=0).float()
    dd = torch.clamp(torch.maximum(q[0] - x, x - q[1]), min=0).norm(dim=-1)
    return torch.where(ok, dd, torch.full_like(dd, float("inf")))


DD = [farinfo(frames[f]) for f in range(len(frames))]
score = {}
for k, c in CELLS.items():
    s = 0; tot = 0
    for f in range(1, len(frames)):
        if f in e1:
            s += abs(int((DD[f] > 4 * c).sum()) - e1[f]["nfar"]); tot += e1[f]["nfar"]
    score[k] = (s, tot)
kbest = min(score, key=lambda k: score[k][0]) if e1 else "DXG*sqrt2"; CELL = CELLS[kbest]
print(f"[{name}] 칸 후보 대조 {score} -> {kbest} {CELL:.5f}", flush=True)

THR = [1.0, 3.0]
V = [None]
for f in range(1, T + 1):
    v = (frames[f] - frames[f - 1]).norm(dim=-1) / H
    V.append(torch.where(torch.isfinite(v), v, torch.zeros_like(v)))
jump = {t: {} for t in THR}      # f -> bool 마스크
newfar = {}
farprev = None
for f in range(1, T + 1):
    far = DD[f] > 4 * CELL
    newfar[f] = far & (~farprev if farprev is not None else torch.ones_like(far)); farprev = far
    if f >= 2:
        for t in THR:
            jump[t][f] = (V[f] - V[f - 1]) > t
print(f"[{name}] 튐 수: " + ", ".join(f"급증>{t:g}: {sum(int(m.sum()) for m in jump[t].values())}" for t in THR) + f", 새 이탈 {sum(int(m.sum()) for m in newfar.values())}", flush=True)

# ---------- 프레임(걸음)별 노드 -> 입자 받침 값
OFF = torch.tensor([(i, j, k) for i in range(3) for j in range(3) for k in range(3)], dtype=torch.int64)       # (27, 3)
depE, depM, depC, wsE, mpE = {}, {}, {}, {}, {}
rows = []
for g in range(1, T + 1):
    z = np.load(os.path.join(rd, "nodes", f"f{g:03d}.npz"))
    ip = z["ip"]; bmin = ip[0:3].astype(np.int64); nb = ip[6:9].astype(np.int64)
    A = z["A"].astype(np.float64)
    act = torch.from_numpy(z["act"].astype(np.int64)); mn = torch.from_numpy(z["mn"].astype(np.float64))
    Ust = torch.from_numpy(z["Ust"].astype(np.float64))      # (ns, nact, 3) 걸음 끝마다 U
    ns = Ust.shape[0]
    i0 = act // int(nb[1] * nb[2]); j0 = (act // int(nb[2])) % int(nb[1]); k0 = act % int(nb[2])
    Lg = torch.stack([i0, j0, k0], 1).double() + torch.from_numpy(bmin).double()
    x0 = Lg @ torch.from_numpy(A).T                                                                             # 걸음 시작 격자 위치
    h0 = x0 @ torch.from_numpy(NR) - PN
    hst = h0[None] + Ust @ torch.from_numpy(NR)                                                                 # (ns, nact) 걸음 끝마다 높이
    hE = hst[-1]; hM = torch.minimum(hst.min(0).values, hE)
    # 질량 백분위 (활성 노드 안 순위)
    rk = torch.empty(len(mn), dtype=torch.float64); rk[torch.argsort(mn)] = torch.arange(len(mn), dtype=torch.float64)
    pct = rk / max(len(mn) - 1, 1) * 100
    NB = int(nb[0] * nb[1] * nb[2])
    boxE = torch.full((NB,), float("inf"), dtype=torch.float32); boxM = boxE.clone(); boxC = boxE.clone(); boxP = torch.zeros(NB, dtype=torch.float32)
    boxE[act] = hE.float(); boxM[act] = hM.float(); boxC[act] = torch.where((h0 >= 0) & (hE < 0), hE, torch.full_like(hE, float("inf"))).float(); boxP[act] = pct.float()
    # 입자 받침 27 노드
    pb = torch.from_numpy(z["pb"].astype(np.int64)); fx = torch.from_numpy(z["fx"].astype(np.float32))
    nid = ((pb[:, None, 0] + OFF[None, :, 0]) * int(nb[1]) + (pb[:, None, 1] + OFF[None, :, 1])) * int(nb[2]) + (pb[:, None, 2] + OFF[None, :, 2])   # (N,27)
    w = []
    for a in range(3):
        f_ = fx[:, a]
        w.append(torch.stack([0.5 * (1.5 - f_) ** 2, 0.75 - (f_ - 1) ** 2, 0.5 * (f_ - 0.5) ** 2], 1))
    W = (w[0][:, :, None, None] * w[1][:, None, :, None] * w[2][:, None, None, :]).reshape(-1, 27)
    hE_p = boxE[nid]; hM_p = boxM[nid]; hC_p = boxC[nid]
    dE = torch.clamp(-hE_p, min=0).max(1).values * 1e3                                                         # mm
    dM = torch.clamp(-hM_p, min=0).max(1).values * 1e3
    dC = torch.clamp(-hC_p, min=0).max(1).values * 1e3
    under = hE_p < 0
    wsh = (W * under).sum(1) / W.sum(1)
    deepest = torch.argmin(hE_p, 1)
    mp = boxP[nid.gather(1, deepest[:, None]).squeeze(1)]
    mp = torch.where(dE > 0, mp, torch.full_like(mp, float("nan")))
    depE[g], depM[g], depC[g], wsE[g], mpE[g] = dE, dM, dC, wsh, mp
    # 걸음 통계
    uE = hE < 0; uM = hM < 0; uC = (h0 >= 0) & (hE < 0)
    rows.append(dict(f=g, ns=int(ns), n_active=int(len(mn)), n_under_E=int(uE.sum()), n_under_M=int(uM.sum()), n_cross_C=int(uC.sum()), n_start_below=int((h0 < 0).sum()),
                     n_under_E_dh=int((hE < -DH).sum()), depth_max_E_mm=float(max(0.0, -hE.min())) * 1e3, depth_max_M_mm=float(max(0.0, -hM.min())) * 1e3,
                     depth_max_C_mm=float(max(0.0, -torch.where(uC, hE, torch.zeros_like(hE)).min())) * 1e3,
                     mass_under_E=float(mn[uE].sum() / mn.sum()), frac_part_under_E=float((dE > 0).float().mean()), frac_part_under_M=float((dM > 0).float().mean()),
                     frac_part_cross_C=float((dC > 0).float().mean()), n_part_under_E_dh=int((dE > DH * 1e3).sum()),
                     frac_part_wshare_gt_half=float((wsh > 0.5).float().mean()), redo=int(z["redo"]), ksK=int(z["ksK"]), dxL_mm=float(np.linalg.norm(A[:, 0])) * 1e3,
                     n_deep_E=int((hE < -1.5 * float(np.linalg.norm(A[:, 0]))).sum()), mass_deep_E=float(mn[hE < -1.5 * float(np.linalg.norm(A[:, 0]))].sum() / mn.sum()),
                     masspct_med_under_E=(float(pct[uE].median()) if bool(uE.any()) else None)))
    if g % 5 == 0 or g == T:
        print(f"[{name}] f{g}: " + json.dumps({k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in rows[-1].items()}), flush=True)

# ---------- 튄 입자별 창 검사
WIN = {"W1": lambda f: [f], "W3": lambda f: [f - 2, f - 1, f], "P3": lambda f: [f - 3, f - 2, f - 1]}


def winmax(D, f, kind):
    ss = [g for g in WIN[kind](f) if 1 <= g <= T]
    if not ss: return None
    return torch.stack([D[g] for g in ss]).max(0).values


def winarg(f, kind):
    """창 안에서 E 깊이가 가장 큰 걸음의 (깊이, 가중 몫, 질량 백분위)"""
    ss = [g for g in WIN[kind](f) if 1 <= g <= T]
    if not ss: return None
    dd = torch.stack([depE[g] for g in ss]); gi = dd.argmax(0)
    pick = lambda D: torch.stack([D[g] for g in ss]).gather(0, gi[None]).squeeze(0)
    return dd.max(0).values, pick(wsE), pick(mpE)


def q(x, qs=(0.5, 0.9, 1.0)):
    x = x[torch.isfinite(x)]
    return [float(torch.quantile(x, p)) for p in qs] if x.numel() else []


BE = torch.tensor([0.0, 5e-3, 10e-3, 20e-3, 30e-3, 50e-3, 100e-3, 200e-3])      # 입자 높이 구간 경계 (m): 구간 = bucketize
DEEP = 1.5 * rows[0]["dxL_mm"]      # mm: 입자가 다 바닥 위면 격자만으로는 닿을 수 없는 깊이 (받침 최대 1.5 칸 아래)
hP = [(frames[f].double() @ torch.from_numpy(NR) - PN).float() for f in range(len(frames))]
print(f"[{name}] 깊은 노드 문턱 {DEEP:.1f} mm (1.5 칸)", flush=True)


def metr(f, kind):
    """창 안 입자별 [E 어느 걸음이든 바닥 밑 받침 노드, C 내려간 노드, 깊은 노드] 불리언 (3, N) 과 깊이 mm"""
    e = winmax(depE, f, kind)
    if e is None: return None, None
    cc = winmax(depC, f, kind)
    return torch.stack([e > 0, cc > 0, e > DEEP]), e


MN = ["E", "C", "deep"]


def cohort(masks, excl):
    """masks: f -> 그 프레임에서 튄 입자. 창별: 관측 수, 같은 프레임·같은 높이 구간 (직전 프레임 입자 높이) 의 튀지 않은 입자 비율로 기대한 수, 깊이·가중 몫·질량 백분위"""
    o = dict(n=sum(int(m.sum()) for m in masks.values()))
    own = []
    for f, m in masks.items():
        if f >= 2 and bool(m.any()): own.append(hP[f - 1][m])
    own = torch.cat(own) if own else torch.zeros(0)
    o["own_below_floor_prev"] = int((own < 0).sum()); o["own_h_mm_q50_90"] = [float(torch.quantile(own, p)) * 1e3 for p in (0.5, 0.9)] if own.numel() else []
    for kind in WIN:
        obs = [0, 0, 0]; exp = [0.0, 0.0, 0.0]; n = 0; dep, wsl, mpl = [], [], []; ddep, dws, dmp = [], [], []
        for f, m in masks.items():
            if f < 2 or not bool(m.any()): continue
            M, e = metr(f, kind)
            if M is None: continue
            wa = winarg(f, kind)
            n += int(m.sum())
            bi = torch.bucketize(hP[f - 1], BE, right=True)
            nm = ~excl[f]
            for bb in range(len(BE) + 1):
                ev = m & (bi == bb); ne = int(ev.sum())
                if ne == 0: continue
                ref = nm & (bi == bb)
                if int(ref.sum()) == 0: continue
                for i in range(3): exp[i] += ne * float(M[i][ref].float().mean())
            for i in range(3): obs[i] += int(M[i][m].sum())
            sel = m & (e > 0); dep.append(wa[0][sel]); wsl.append(wa[1][sel]); mpl.append(wa[2][sel])
            sd = m & (e > DEEP); ddep.append(wa[0][sd]); dws.append(wa[1][sd]); dmp.append(wa[2][sd])
        cat = lambda L_: torch.cat(L_) if L_ else torch.zeros(0)
        o[kind] = dict(n=n, obs={k_: obs[i] for i, k_ in enumerate(MN)}, exp={k_: exp[i] for i, k_ in enumerate(MN)},
                       depth_mm_q50_90_max=q(cat(dep)), wshare_q50_90_max=q(cat(wsl)), masspct_q50_90_max=q(cat(mpl)),
                       deep_depth_mm_q50_90_max=q(cat(ddep)), deep_wshare_q50_90_max=q(cat(dws)), deep_masspct_q50_90_max=q(cat(dmp)))
    return o


def baseline(excl):
    """사건이 아닌 (입자, 프레임) 쌍: 같은 창 정의로 비율 (프레임 범위 f0.. 전부)"""
    o = {}
    for kind in WIN:
        n = 0; c = [0, 0, 0]
        for f in range(2, T + 1):
            M, e = metr(f, kind)
            if M is None: continue
            keep = ~excl[f]; n += int(keep.sum())
            for i in range(3): c[i] += int(M[i][keep].sum())
        o[kind] = dict(n=n, **{f"rate_{k_}": c[i] / max(n, 1) for i, k_ in enumerate(MN)})
    return o


out = dict(name=name, dh_mm=DH * 1e3, dxg=DXG, frame_dt=H, cell=CELL, cell_name=kbest, nframes=T, nparticles=N, rows=rows)
excl = {f: torch.zeros(N, dtype=torch.bool) for f in range(2, T + 1)}
for f in excl:
    for t in THR: excl[f] |= jump[t][f]
    excl[f] |= newfar[f]
out["deep_mm"] = DEEP
out["cohort_jump3"] = cohort(jump[3.0], excl)
out["cohort_jump1"] = cohort(jump[1.0], excl)
out["cohort_newfar"] = cohort({f: m for f, m in newfar.items() if f >= 2}, excl)
out["baseline_nonevent"] = baseline(excl)
# 모든 입자 기준 (사건 포함)
# 사건 프레임 목록 (프레임별 튄 수)
out["events_per_frame"] = dict(jump3=[int(jump[3.0][f].sum()) if f in jump[3.0] else 0 for f in range(1, T + 1)],
                                jump1=[int(jump[1.0][f].sum()) if f in jump[1.0] else 0 for f in range(1, T + 1)],
                                newfar=[int(newfar[f].sum()) for f in range(1, T + 1)])
# 사건별 깊이 순위 (W3 창, 가장 깊은 20 개: 프레임, 입자, 깊이 mm, 가중 몫, 질량 백분위)
top = []
for key, masks in (("jump3", jump[3.0]), ("newfar", newfar)):
    cand = []
    for f, m in masks.items():
        if f < 2 or not bool(m.any()): continue
        wa = winarg(f, "W3")
        idx = torch.nonzero(m & (wa[0] > 0)).squeeze(1)
        for p in idx[:2000].tolist():
            cand.append((float(wa[0][p]), f, p, float(wa[1][p]), float(wa[2][p])))
    cand.sort(reverse=True)
    top.append((key, [dict(f=c[1], p=c[2], depth_mm=c[0], wshare=c[3], masspct=c[4]) for c in cand[:20]]))
out["top_depth"] = dict(top)
json.dump(out, open(outp, "w"), ensure_ascii=False, indent=1)
print(f"[{name}] 저장 {outp}", flush=True)
