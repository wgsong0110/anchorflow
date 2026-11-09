"""절 CM87 wolf 더미 표 (읽기만, torch 없이 -- lib/anchorflow/ptload_np.py): 판 디렉토리 (pred.pt·fem.jsonl·summary.json·bbox.txt) 를 무리로 묶어
발산·오차·질량중심 (절대 x·n = fem.jsonl hp[0], 바닥 위 = − 480 mm)·더미 꼭대기 (99 % · 최대, 바닥 위)·안식각·발자국 면적·1 cm 복셀당 입자 (채움)·이탈 (bbox 4 칸 밖 > 20 개)
정답은 같은 함수 (pred[k] = 정답 x[k+1]).
사용: python exe/wolfh9_heap.py <wolf.pt> <무리이름=판디렉토리[,판디렉토리...]> ... > 표.md   (+ heap_rows.json)
정의: 높이 h = z − 0.48 (바닥 법선 +z). 꼭대기 = h 의 99 백분위·최대. 발자국 면적 = 1 cm² 칸 중 입자 3 개 이상인 칸 수 (cm²).
  안식각 = 칸 꼭대기 높이 지도 (칸 최대 h) 의 중앙 차분 기울기 각, 꼭대기 높이가 지도 최대의 20~80 % 인 칸의 중앙값 (°).
  채움 = 입자 수 / 점유 1 cm 복셀 수 (부피 보존이면 정답 ≈ 2.2). 프로파일 = 평면 질량중심에서 5 cm 띠별 칸 꼭대기 중앙 (mm)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import ptload_np as PL                                              # noqa: E402

FL, C = 0.48, 0.01


def met(x):
    ok = np.isfinite(x).all(1); x = x[ok].astype(np.float64); h = x[:, 2] - FL
    r = dict(comu=h.mean() * 1e3, zmax=h.max() * 1e3, z99=np.quantile(h, 0.99) * 1e3)
    ij = np.floor(x[:, :2] / C).astype(np.int64); key = ij[:, 0] * 100000 + ij[:, 1]
    u, inv = np.unique(key, return_inverse=True)
    top = np.full(u.size, -1.0); np.maximum.at(top, inv, h); cnt = np.bincount(inv); m = cnt >= 3
    r["area"] = float(m.sum())
    r["ppv"] = x.shape[0] / np.unique(np.c_[ij, np.floor(h / C).astype(np.int64)], axis=0).shape[0]
    I = u // 100000; J = u - I * 100000; i0, j0 = I.min(), J.min()
    G = np.full((I.max() - i0 + 3, J.max() - j0 + 3), np.nan); G[I[m] - i0 + 1, J[m] - j0 + 1] = top[m]
    gx = (G[2:, 1:-1] - G[:-2, 1:-1]) / (2 * C); gy = (G[1:-1, 2:] - G[1:-1, :-2]) / (2 * C)
    s = np.degrees(np.arctan(np.hypot(gx, gy))); hc = G[1:-1, 1:-1]; hm = np.nanmax(G)
    f = np.isfinite(s) & (hc > 0.2 * hm) & (hc < 0.8 * hm)
    r["rep"] = float(np.median(s[f])) if f.any() else float("nan")
    cx, cy = x[:, 0].mean(), x[:, 1].mean(); rr = np.hypot((I[m] + 0.5) * C - cx, (J[m] + 0.5) * C - cy)
    r["prof"] = [round(float(np.median(top[m][(rr >= a) & (rr < a + 0.05)])) * 1e3) if ((rr >= a) & (rr < a + 0.05)).sum() > 3 else None for a in np.arange(0, 0.6, 0.05)]
    return r


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def bbox(d):
    p = os.path.join(d, "bbox.txt")
    if not os.path.exists(p):
        return None
    for ln in open(p):
        if "pred.pt" in ln:
            D = json.loads(ln[ln.index("{"):]); return max((v["nfar"], v["dmax"]) for v in D.values())


FR = (10, 20, 30, 40, 50)
d = PL.load(sys.argv[1]); X = np.asarray(d["x"])
GT = {f: met(np.asarray(X[f])) for f in FR}
rows = {"정답": dict(n=1, gt=GT)}
for a in sys.argv[2:]:
    g, ds = a.split("=", 1)
    R = []
    for dd in ds.split(","):
        s = json.load(open(os.path.join(dd, "summary.json"))) if os.path.exists(os.path.join(dd, "summary.json")) else {}
        fj = [q for q in jl(os.path.join(dd, "fem.jsonl")) if "t" in q]
        hp = {f: fj[f - 1]["hp"][0] * 1e3 for f in FR if f - 1 < len(fj) and "hp" in fj[f - 1]}
        hg = {f: fj[f - 1]["hg"][0] * 1e3 for f in FR if f - 1 < len(fj) and "hg" in fj[f - 1]}
        M = {}
        pp = os.path.join(dd, "pred.pt")
        if os.path.exists(pp):
            P = PL.load(pp); P = np.asarray(P["pred"] if isinstance(P, dict) else P)
            M = {f: met(np.asarray(P[f - 1])) for f in FR if f - 1 < P.shape[0] and np.isfinite(P[f - 1]).all(1).sum() > 1000}
        e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
        R.append(dict(dir=dd, div=s.get("diverged"), err=e, hp=hp, hg=hg, m=M, bb=bbox(dd)))
    rows[g] = dict(n=len(R), runs=R)
json.dump(rows, open("heap_rows.json", "w"), ensure_ascii=False, default=str)


def rng(v, p=0):
    v = [q for q in v if q is not None and np.isfinite(q)]
    if not v:
        return "-"
    return f"{np.mean(v):.{p}f}" + (f" [{min(v):.{p}f}~{max(v):.{p}f}]" if len(v) > 1 else "")


hg50 = hg30 = None
for g, D in rows.items():
    for r in D.get("runs", []):
        hg50 = hg50 or r["hg"].get(50); hg30 = hg30 or r["hg"].get(30)
print("## wolf 더미 (질량중심 = fem.jsonl hp[0] 질량 가중 x·n 절대 mm, 바닥 z 480 mm; 바닥 위 = − 480)\n")
print("| 무리 | 판 | 발산 | 오차 % | 질량중심 f30 · f50 절대 mm (바닥 위) | 꼭대기 f50 99 % · 최대 mm (바닥 위) | 안식각 f50 ° | 발자국 면적 f30 · f50 cm² | 채움 f50 입자/cm³ | 이탈 판 · 최대 개수 |")
print("|---|---|---|---|---|---|---|---|---|---|")
G5 = GT[50]
print(f"| 정답 | - | - | - | {hg30 or float('nan'):.0f} · {hg50 or float('nan'):.0f} ({(hg50 or 480) - 480:.0f}) | {G5['z99']:.0f} · {G5['zmax']:.0f} | {G5['rep']:.1f} | {GT[30]['area']:.0f} · {G5['area']:.0f} | {G5['ppv']:.2f} | - |")
for g, D in rows.items():
    if g == "정답":
        continue
    R = D["runs"]; ok = [r for r in R if not r["div"]]
    f30 = [r["hp"].get(30) for r in ok]; f50 = [r["hp"].get(50) for r in ok]
    t99 = [r["m"].get(50, {}).get("z99") for r in ok]; tmx = [r["m"].get(50, {}).get("zmax") for r in ok]
    rep = [r["m"].get(50, {}).get("rep") for r in ok]; a30 = [r["m"].get(30, {}).get("area") for r in R]; a50 = [r["m"].get(50, {}).get("area") for r in ok]
    ppv = [r["m"].get(50, {}).get("ppv") for r in ok]
    esc = [r["bb"] for r in R if r["bb"]]; ne = sum(1 for b in esc if b[0] > 20); mx = max((b[0] for b in esc), default=0)
    nd = sum(1 for r in R if r["div"])
    print(f"| {g} | {len(R)} | {nd} | {rng([100 * r['err'] for r in ok if r['err'] is not None], 2)} | {rng(f30)} · {rng(f50)} ({rng([q - 480 for q in f50 if q])}) | {rng(t99)} · {rng(tmx)} | {rng(rep, 1)} | {rng(a30)} · {rng(a50)} | {rng(ppv, 2)} | {ne}/{len(R)} · {mx} |")
print("\n## 높이 프로파일 (f50, 평면 질량중심에서 5 cm 띠별 칸 꼭대기 중앙 mm; 첫 판)\n")
print(f"- 정답: {G5['prof']}")
for g, D in rows.items():
    if g != "정답" and D["runs"][0]["m"].get(50):
        print(f"- {g}: {D['runs'][0]['m'][50]['prof']}")
print("\n## 프레임별 (첫 판): 질량중심 바닥 위 (균일 가중) mm · 채움 · 꼭대기 99 % mm\n")
print("| 무리 | " + " | ".join(f"f{f}" for f in FR) + " |")
print("|---|" + "---|" * len(FR))
print("| 정답 | " + " | ".join(f"{GT[f]['comu']:.0f} · {GT[f]['ppv']:.2f} · {GT[f]['z99']:.0f}" for f in FR) + " |")
for g, D in rows.items():
    if g == "정답":
        continue
    M = D["runs"][0]["m"]
    print(f"| {g} | " + " | ".join((f"{M[f]['comu']:.0f} · {M[f]['ppv']:.2f} · {M[f]['z99']:.0f}" if f in M else "-") for f in FR) + " |")
