"""절 CM112 분석 (CPU, 읽기만): 노드 덤프에서 활성 칸 (입자가 든 이차 B-스플라인 칸) 의 세로 두께 h 를 프레임 시작·끝에 재고, 층별 줄어든 비율 r = h끝/h시작 를 낸다.
사용: python exe/cellh_an.py <판이름> <판 디렉토리 (nodes/f*.npz)> <cfg.json> <out.json>
 - 칸 = 입자 받침 base (pb) 와 분수 좌표 fx (>=1 이면 위쪽 칸) 로 정한 8 꼭짓점 노드. 노드 높이 h = x·n - PN (n = 바닥 법선, 걸음 시작 = 격자 점, 끝 = + 걸음 끝 U).
 - 칸 두께 h = 8 꼭짓점 높이의 최대 - 최소. 층 = floor(칸 바닥 (시작 꼭짓점 최소 높이) / dx), 0 미만은 0.
 - 프레임 f = 걸음 f-1 -> f. 누적 비율 = 입자별 r (그 입자가 든 칸) 의 프레임 곱, 층은 입자의 첫 프레임 층.
"""
import json, os, sys
import numpy as np

name, rd, cfgp, outp = sys.argv[1:5]
cfg = json.load(open(cfgp))
pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
NR = np.array(pl["normal"], dtype=np.float64); NR /= np.linalg.norm(NR)
PN = float(np.array([float(q) for q in pl["point"]]) @ NR)
steps = sorted(int(f[1:4]) for f in os.listdir(os.path.join(rd, "nodes")) if f.endswith(".npz"))
OFF = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64)
print(f"[{name}] 법선 {NR.tolist()} 바닥 높이 {PN:.4f} 덤프 {len(steps)} 개", flush=True)
Rcum = None; L0 = None
frames = []


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else None


for g in steps:
    z = np.load(os.path.join(rd, "nodes", f"f{g:03d}.npz"))
    ip = z["ip"]; bmin = ip[0:3].astype(np.int64); nb = ip[6:9].astype(np.int64)
    A = z["A"].astype(np.float64); act = z["act"].astype(np.int64)
    dx = float(np.linalg.norm(A[:, 0]))
    nbx = int(nb[0] * nb[1] * nb[2])
    i0 = act // int(nb[1] * nb[2]); j0 = (act // int(nb[2])) % int(nb[1]); k0 = act % int(nb[2])
    x0 = (np.stack([i0, j0, k0], 1).astype(np.float64) + bmin) @ A.T
    h0 = x0 @ NR - PN
    hE = h0 + z["Ust"][-1].astype(np.float64) @ NR
    B0 = np.full(nbx, np.nan); BE = np.full(nbx, np.nan); B0[act] = h0; BE[act] = hE
    pb = z["pb"].astype(np.int64); fx = z["fx"].astype(np.float32)
    N = len(pb)
    if Rcum is None:
        Rcum = np.ones(N); L0 = np.full(N, -1)
    c = pb + (fx >= 1.0)                                                                  # 칸 낮은 모서리 노드 (상자 안 번호)
    nid = ((c[:, None, 0] + OFF[None, :, 0]) * int(nb[1]) + (c[:, None, 1] + OFF[None, :, 1])) * int(nb[2]) + (c[:, None, 2] + OFF[None, :, 2])
    ok = (nid >= 0).all(1) & (nid < nbx).all(1)
    nid = np.where(ok[:, None], nid, 0)
    a0 = B0[nid]; aE = BE[nid]
    ok &= np.isfinite(a0).all(1) & np.isfinite(aE).all(1)
    hs = a0.max(1) - a0.min(1); he = aE.max(1) - aE.min(1); hb = a0.min(1)
    key = nid[:, 0]
    # 칸 단위 (고유 칸)
    kk, first = np.unique(np.where(ok, key, -1), return_index=True)
    sel = first[kk >= 0]
    cs, ce, cb = hs[sel], he[sel], hb[sel]
    cl = np.clip(np.floor(cb / dx + 1e-6), 0, None).astype(int)
    r = ce / cs
    # 입자 단위 누적
    rp = np.where(ok, he / np.where(hs > 0, hs, 1), 1.0)
    Rcum = Rcum * rp
    lp = np.clip(np.floor(hb / dx + 1e-6), 0, None).astype(int)
    if (L0 < 0).any():
        m = L0 < 0
        L0[m] = np.where(ok[m], lp[m], 0)
    rec = dict(f=g, dx=dx, n_cells=int(len(sel)), n_part=int(N), n_part_bad=int((~ok).sum()), hs_med=float(np.median(cs)), hs_min=float(cs.min()), hs_max=float(cs.max()),
               layers={}, cum={})
    for L in range(int(cl.max()) + 1):
        m = cl == L
        if m.sum():
            rl = r[m]
            rec["layers"][L] = dict(n=int(m.sum()), med=q(rl, 50), p10=q(rl, 10), min=float(rl.min()), p90=q(rl, 90), he_med=float(np.median(ce[m])))
    for L in range(int(L0.max()) + 1):
        m = L0 == L
        if m.sum():
            rl = Rcum[m]
            rec["cum"][L] = dict(n=int(m.sum()), med=q(rl, 50), p10=q(rl, 10), min=float(rl.min()))
    frames.append(rec)
    print(f"[{name}] f{g}: 칸 {len(sel)} 시작두께/dx 중앙 {np.median(cs)/dx:.3f} [{cs.min()/dx:.3f}, {cs.max()/dx:.3f}] 층 {len(rec['layers'])} "
          + " ".join(f"L{L}:{v['med']:.3f}" for L, v in list(rec["layers"].items())[:6]), flush=True)
json.dump(dict(name=name, normal=NR.tolist(), floor=PN, frames=frames), open(outp, "w"))
print(f"[{name}] 끝 {outp}", flush=True)
