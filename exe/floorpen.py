"""절 CM110 진단 (읽기만, 결과 불변): wolf 덤프 pred.pt 로 '바닥 밑 깊이 들어감 -> 튐' 이 맞는지.
사용: python exe/floorpen.py <판이름> <pred.pt> <cfg.json> <data wolf.pt> <e1.jsonl> <frames_done> <out.json>
  바닥 높이 h = (x - PT)·NR (cfg 의 surface_collider), d̂ = 0.5 grid_lim / n_grid (실행기 FL["DH"] 와 같음).
  이탈 = exe/escape1_wrap.py 와 같은 정의 (몸 상자 0.1~99.9 백분위 [입자 7 개마다] 에서 4 칸 넘게 떨어짐, '새 이탈' = 이 프레임에 처음), 칸은 e1.jsonl nfar 와 맞는 후보를 고름.
  튐 = 입자 속도 (프레임 변위 / frame_dt) 가 전 프레임보다 THR m/s 넘게 오름.
"""
import json, math, sys
import torch

name, pp, cfgp, datap, e1p, nd, outp = sys.argv[1:8]
nd = int(nd)
cfg = json.load(open(cfgp))
pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
NR = torch.tensor(pl["normal"], dtype=torch.float64); NR = NR / NR.norm()
PT = torch.tensor([float(q) for q in pl["point"]], dtype=torch.float64)
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100)); DH = 0.5 * DXG
H = float(cfg["frame_dt"])
print(f"[{name}] 바닥 점 {PT.tolist()} 법선 {NR.tolist()} DXG {DXG} d̂ {DH} frame_dt {H}", flush=True)
d = torch.load(pp, map_location="cpu", weights_only=False)
P = d["pred"]
X0 = torch.load(datap, map_location="cpu", weights_only=False)["x"][0]
print(f"[{name}] pred {tuple(P.shape)} frames_done {nd}  data x0 {tuple(X0.shape)}", flush=True)
T = min(nd, P.shape[0])
e1 = {}
for ln in open(e1p):
    r = json.loads(ln); e1[r["f"]] = r
CELLS = {"NODE_H(cell2*DXG*sqrt2)": 2 * DXG * math.sqrt(2), "2*DXG": 2 * DXG, "DXG": DXG, "DXG*sqrt2": DXG * math.sqrt(2)}
THR = [1.0, 3.0, 10.0]
BINS = [-1e9, -5 * DH, -DH, 0.0, DH, 1e9]
BN = ["< -5d̂", "[-5d̂,-d̂)", "[-d̂,0)", "[0,d̂)", ">= d̂"]


def hgt(x):
    h = (x.double() - PT) @ NR
    return torch.where(torch.isfinite(h), h, torch.full_like(h, 1e9))


def farinfo(x):
    ok = torch.isfinite(x).all(-1)
    xs = x[ok][::7].double()
    q = torch.quantile(xs, torch.tensor([0.001, 0.999], dtype=torch.float64), dim=0).float()
    dd = torch.clamp(torch.maximum(q[0] - x, x - q[1]), min=0).norm(dim=-1)
    dd = torch.where(ok, dd, torch.full_like(dd, float("inf")))
    return dd


# 1 차: 칸 후보 고르기 (e1.jsonl nfar 대조)
frames = [X0.float()] + [P[i].float() for i in range(T)]
DD = [farinfo(frames[f]) for f in range(len(frames))]
score = {}
for k, c in CELLS.items():
    s = 0; tot = 0
    for f in range(1, len(frames)):
        if f in e1:
            s += abs(int((DD[f] > 4 * c).sum()) - e1[f]["nfar"]); tot += e1[f]["nfar"]
    score[k] = (s, tot)
print(f"[{name}] 칸 후보 대조 (nfar 차 합, e1 nfar 합):", score, flush=True)
kbest = min(score, key=lambda k: score[k][0]); CELL = CELLS[kbest]
print(f"[{name}] 칸 = {kbest} {CELL:.5f}", flush=True)

rows = []
hist = []            # 직전 프레임들의 높이
vprev = None
farprev = None
ev = {t: dict(n=0, w1=0, w3=0, w3d=0, pairs=[]) for t in THR}
evfar = dict(n=0, w1=0, w3=0, w3d=0, pairs=[])
binpf = {t: [0] * len(BN) for t in THR}; binjump = {t: [0] * len(BN) for t in THR}
binfar = [0] * len(BN); binfarn = [0] * len(BN)
everunder_w3 = 0; pf = 0
for f in range(len(frames)):
    x = frames[f]; h = hgt(x)
    fin = torch.isfinite(x).all(-1)
    r = dict(f=f, n_under=int((h < 0).sum()), depth_max_mm=float(max(0.0, -h.min())) * 1e3, depth_over_dh=float(max(0.0, -h.min())) / DH,
             n_band=int(((h >= 0) & (h < DH)).sum()), n_under_dh=int((h < -DH).sum()), hmin_mm=float(h.min()) * 1e3)
    if f >= 1:
        v = ((x - frames[f - 1]).norm(dim=-1) / H); v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
        far = DD[f] > 4 * CELL
        r["n_far"] = int(far.sum()); r["dmax"] = float(DD[f][fin].max()) if bool(fin.any()) else None
        new = far & (~farprev if farprev is not None else torch.ones_like(far))
        r["n_newfar"] = int(new.sum()); farprev = far
        r["vmax"] = float(v.max()); r["v_p99"] = float(torch.quantile(v[::7], 0.99))
        if vprev is not None:
            dv = v - vprev
            r["dvmax"] = float(dv.max())
            for t in THR:
                r[f"n_jump{int(t)}"] = int((dv > t).sum())
        vprev = v
        if len(hist) >= 1:
            w1 = hist[-1]; w3 = torch.stack(hist[-3:]).min(0).values
            # 구간 칸별 표본 수 (기저율)
            if vprev is not None and f >= 2:
                for t in THR:
                    jm = (v - hist_v) > t
                    e = ev[t]; n = int(jm.sum()); e["n"] += n
                    e["w1"] += int((jm & (w1 < 0)).sum()); e["w3"] += int((jm & (w3 < 0)).sum()); e["w3d"] += int((jm & (w3 < -DH)).sum())
                    if n:
                        idx = jm & (w3 < 0)
                        e["pairs"].append(torch.stack([-w3[idx].float(), (v - hist_v)[idx].float(), v[idx].float()], 1))
                    bi = torch.bucketize(w3.float(), torch.tensor(BINS[1:-1], dtype=torch.float32), right=True)
                    binpf[t] = [a + int((bi == k).sum()) for k, a in enumerate(binpf[t])]
                    binjump[t] = [a + int(((bi == k) & jm).sum()) for k, a in enumerate(binjump[t])]
                if f == 2: pass
                everunder_w3 += int((w3 < 0).sum()); pf += int(w3.numel())
            # 새 이탈
            n = int(new.sum()); evfar["n"] += n
            evfar["w1"] += int((new & (w1 < 0)).sum()); evfar["w3"] += int((new & (w3 < 0)).sum()); evfar["w3d"] += int((new & (w3 < -DH)).sum())
            if n:
                idx = new & (w3 < 0)
                evfar["pairs"].append(torch.stack([-w3[idx].float(), v[idx].float()], 1))
            bi = torch.bucketize(w3.float(), torch.tensor(BINS[1:-1], dtype=torch.float32), right=True)
            for k in range(len(BN)):
                binfar[k] += int(((bi == k) & new).sum()); binfarn[k] += int((bi == k).sum())
        hist_v = v
    hist.append(h); hist = hist[-3:]
    rows.append(r)
    if f % 5 == 0 or f == len(frames) - 1:
        print(f"[{name}] f{f}: " + json.dumps({k: (round(v_, 3) if isinstance(v_, float) else v_) for k, v_ in r.items()}), flush=True)


def spear(a, b):
    if a.numel() < 3: return None
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double()
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-30))


out = dict(name=name, dh=DH, dxg=DXG, frame_dt=H, cell=CELL, cell_name=kbest, cell_score=score, thr=THR, bins=BN, nframes=T, rows=rows, base_rate_w3=everunder_w3 / max(pf, 1))
for t in THR:
    e = ev[t]; pr = torch.cat(e["pairs"]) if e["pairs"] else torch.zeros(0, 3)
    qs = {}
    if pr.shape[0]:
        for q in (0.5, 0.9, 0.99):
            qs[str(q)] = [float(torch.quantile(pr[::max(1, pr.shape[0] // 1000000), c], q)) for c in range(3)]
    out[f"jump{int(t)}"] = dict(n=e["n"], w1=e["w1"], w3=e["w3"], w3_deeper_dh=e["w3d"], n_under_pairs=int(pr.shape[0]),
                                 spearman_depth_dv=spear(pr[:, 0], pr[:, 1]) if pr.shape[0] else None, spearman_depth_v=spear(pr[:, 0], pr[:, 2]) if pr.shape[0] else None,
                                 q_depth_dv_v=qs, bin_particleframes=binpf[t], bin_jumps=binjump[t])
pr = torch.cat(evfar["pairs"]) if evfar["pairs"] else torch.zeros(0, 2)
out["newfar"] = dict(n=evfar["n"], w1=evfar["w1"], w3=evfar["w3"], w3_deeper_dh=evfar["w3d"], n_under_pairs=int(pr.shape[0]),
                     spearman_depth_v=spear(pr[:, 0], pr[:, 1]) if pr.shape[0] else None,
                     q_depth_v={str(q): [float(torch.quantile(pr[:, c], q)) for c in range(2)] for q in (0.5, 0.9)} if pr.shape[0] else {},
                     bin_particleframes=binfarn, bin_events=binfar)
json.dump(out, open(outp, "w"), ensure_ascii=False, indent=1)
print(f"[{name}] 저장 {outp}", flush=True)
