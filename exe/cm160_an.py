#!/usr/bin/env python3
"""절 CM160 분석 (instance 에서; numpy·scipy·torch): bread 찢어짐 가설 (1e-20 에서 틈의 가벼운 노드가 양쪽 조각을 이어 붙임) 확인.
  gt   <bread.pt> <out>                      : 정답 찢어짐 곡선 (프레임마다) -> <out>/gt_stats.json, 기록 프레임 목록 <out>/frames.txt
  part <bread.pt> <A 디렉토리> <B 디렉토리> <out> : 입자 수준 (찢어진 입자 수·연결 성분·틈 너비) 정답·A·B, 기록 프레임의 입자 사본 -> part_stats.json, parts_f*.npz
  node <A 디렉토리> <B 디렉토리> <out>       : 노드 수준 (저질량 노드 위치·탄성 힘·단면 합력·가우스 점 응력) -> node_stats.json, cm160_tables.md
정의: 찢어진 입자 = 0 프레임 8 최근접 이웃과의 거리 비 중앙값 > 1.5 (절 AG.17 와 같음). 연결 성분 = 0 프레임 8-NN 변 중 현재 길이 비 < 문턱인 변으로 이은 입자 그래프."""
import json
import os
import sys

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

K = 8
NB_TH = (1.5, 2.0, 3.0)


def load_gt(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["x"].float().numpy()                                   # (T, N, 3), 프레임 0..T-1


def knn0(x0):
    t = cKDTree(x0)
    dd, ii = t.query(x0, k=K + 1)
    return ii[:, 1:], dd[:, 1:]


def stretch(x, nb, d0):
    return np.median(np.linalg.norm(x[:, None, :] - x[nb], axis=-1) / np.maximum(d0, 1e-9), axis=1)


def ratios(x, nb, d0):
    return np.linalg.norm(x[:, None, :] - x[nb], axis=-1) / np.maximum(d0, 1e-9)


def comps(x, nb, d0, th):
    r = ratios(x, nb, d0)
    N = x.shape[0]
    ii = np.repeat(np.arange(N), K)[(r < th).ravel()]
    jj = nb.ravel()[(r < th).ravel()]
    g = coo_matrix((np.ones(len(ii), np.int8), (ii, jj)), shape=(N, N))
    n, lab = connected_components(g, directed=False)
    sz = np.bincount(lab)
    return int(n), int((sz >= 100).sum()), int(sz.max()), lab


def gap_width(x, ax):
    c = np.sort(x[:, ax])
    lo, hi = int(len(c) * 0.01), int(len(c) * 0.99)
    d = np.diff(c[lo:hi])
    k = int(d.argmax())
    return float(d.max()), float(c[lo + k]), float(c[lo + k + 1])


def pull_axis(x):
    ext = (x[-1].max(0) - x[-1].min(0)) / np.maximum(x[0].max(0) - x[0].min(0), 1e-9)
    return int(np.argmax(ext)), ext.tolist()


def tear_hist(x, nb, d0, ax, lo, hi, bw=0.02):
    s = stretch(x, nb, d0)
    t = s > 1.5
    h, e = np.histogram(x[t, ax], bins=np.arange(lo, hi + bw, bw))
    return int(t.sum()), h.tolist(), e.tolist()


def stage_gt(a):
    xg = load_gt(a[0]); out = a[1]; os.makedirs(out, exist_ok=True)
    nb, d0 = knn0(xg[0])
    ax, ext = pull_axis(xg)
    n_t = []
    for t in range(1, xg.shape[0]):
        n_t.append(int((stretch(xg[t], nb, d0) > 1.5).sum()))
    n_t = np.array(n_t)                                             # 프레임 1..T-1
    mx = int(n_t.max())
    fa = int(np.argmax(n_t >= max(100, 0.002 * xg.shape[1]))) + 1
    fb = int(np.argmax(n_t >= 0.9 * mx)) + 1
    frames = sorted({int(round(v)) for v in np.linspace(fa, fb, 6)} | {int(xg.shape[0] - 1), max(fa - 5, 1)})
    if os.environ.get("XS_FRAMES"):
        frames = [int(q) for q in os.environ["XS_FRAMES"].split(",")]
    json.dump(dict(axis=ax, ext=ext, n_torn=n_t.tolist(), f_start=fa, f_end=fb, frames=frames), open(f"{out}/gt_stats.json", "w"))
    open(f"{out}/frames.txt", "w").write(",".join(str(f) for f in frames))
    print("[gt] axis", ax, "ext", ext, "torn max", mx, "start", fa, "end", fb, "frames", frames, flush=True)


def stage_part(a):
    xg = load_gt(a[0]); dA, dB, out = a[1], a[2], a[3]
    xa = torch.load(f"{dA}/pred.pt", weights_only=False)["pred"].float().numpy()
    xb = torch.load(f"{dB}/pred.pt", weights_only=False)["pred"].float().numpy()
    frames = [int(q) for q in open(f"{out}/frames.txt").read().split(",")]
    nb, d0 = knn0(xg[0])
    ax, _ = pull_axis(xg)
    lo, hi = float(xg[0][:, ax].min()) - 0.1, float(xg[:, :, ax].max()) + 0.1
    res = []
    for t in sorted(set(list(range(5, xg.shape[0], 5)) + frames)):
        r = dict(f=t)
        for nm, x in (("gt", xg[t]), ("A", xa[t - 1]), ("B", xb[t - 1])):
            s = stretch(x, nb, d0)
            r[nm] = dict(torn=int((s > 1.5).sum()), s_med=float(np.median(s)), s_p99=float(np.percentile(s, 99)))
            for th in NB_TH:
                n, nbig, mxs, _ = comps(x, nb, d0, th)
                r[nm][f"comp{th}"] = [n, nbig, mxs]
            gw, g0, g1 = gap_width(x, ax)
            r[nm]["gap"] = [gw, g0, g1]
            r[nm]["ymin"], r[nm]["ymax"] = float(x[:, ax].min()), float(x[:, ax].max())
            if t in frames:
                r[nm]["torn_hist"] = tear_hist(x, nb, d0, ax, lo, hi)
        res.append(r)
        if t in frames:
            np.savez_compressed(f"{out}/parts_f{t:03d}.npz", gt=xg[t].astype(np.float16), A=xa[t - 1].astype(np.float16), B=xb[t - 1].astype(np.float16))
            print("[part]", t, {k: (r[k]["torn"], r[k]["comp2.0"], round(r[k]["gap"][0], 4)) for k in ("gt", "A", "B")}, flush=True)
    yn = None
    for r in sorted(res, key=lambda q: q["f"]):
        if r["gt"]["gap"][0] > 0.004:
            yn = 0.5 * (r["gt"]["gap"][1] + r["gt"]["gap"][2]); break
    err = lambda x_: float(np.mean(np.linalg.norm(x_ - xg[1:1 + len(x_)], axis=-1).mean(1)))
    json.dump(dict(axis=ax, yneck=yn, rows=res, mean_dist_A=err(xa), mean_dist_B=err(xb)), open(f"{out}/part_stats.json", "w"))


def stage_node(a):
    """부호: 풀이기 f 는 증분 포텐셜의 기울기 (= −힘). 힘 = −fel. 이음 방향 힘 = 틈 쪽을 향한 힘 (양수 = 틈 반대편 조각 쪽으로 당김).
    단면 인장 T(y*) = y* 보다 위 노드의 탄성 힘 축 성분 합의 부호 반전 (양수 = 위쪽 조각이 아래쪽으로 당겨짐 = 양쪽을 이어 붙임)."""
    dA, dB, out = a[0], a[1], a[2]
    ps = json.load(open(f"{out}/part_stats.json")); ax = ps["axis"]; yneck = ps["yneck"]
    frames = [int(q) for q in open(f"{out}/frames.txt").read().split(",")]
    rows = {r["f"]: r for r in ps["rows"]}
    res = []
    md = []
    for f in frames:
        zA, zB = np.load(f"{dA}/tear/f{f:03d}.npz"), np.load(f"{dB}/tear/f{f:03d}.npz")
        pr = np.load(f"{out}/parts_f{f:03d}.npz")
        dx = float(zA["dx"]); mth = float(zA["mthr"])
        rg = rows[f]
        ygt = 0.5 * (rg["gt"]["gap"][1] + rg["gt"]["gap"][2]) if rg["gt"]["gap"][0] > 0.004 else yneck
        W, Wn = 0.06, 2 * dx
        r = dict(f=f, ygt=ygt, dx=dx, mthrA=mth, gt_torn=rg["gt"]["torn"])
        keyA = {tuple(k) for k in zA["ijk"][zA["m"] < mth]}
        for nm, z, part in (("A", zA, pr["A"].astype(np.float32)), ("B", zB, pr["B"].astype(np.float32))):
            yg = 0.5 * (rg[nm]["gap"][1] + rg[nm]["gap"][2]) if rg[nm]["gap"][0] > 0.004 else ygt
            pos, m, fe = z["pos"].astype(np.float64), z["m"].astype(np.float64), z["fel"].astype(np.float64)
            y = pos[:, ax]
            low = m < mth
            sl = np.abs(y - yg) < W
            tree = cKDTree(part)
            dmin = tree.query(pos[low], k=1)[0] / dx if low.any() else np.zeros(0)
            sgn = np.sign(y - yg)
            fy = -fe[:, ax]                                                 # 힘 (축 성분)
            br = -sgn * fy                                                  # 틈 쪽을 향하면 양수
            fm = np.linalg.norm(fe, axis=1)
            inA = np.array([tuple(k) in keyA for k in z["ijk"]])
            q = z["q"].astype(np.float64); qy = z["q_pos"][:, ax].astype(np.float64)
            sig = q[:, ax]                                                  # 코시 응력 σ_축축 (성분 순서 xx yy zz xy xz yz, 축 1 = yy)
            qs = np.abs(qy - yg) < W; qn = np.abs(qy - yg) < Wn
            mfull = np.zeros(int(z["nn"])); mfull[z["aid"]] = m
            qlow = mfull[z["q_id"]] < mth
            def st(v):
                return [float(np.mean(v)), float(np.percentile(v, 90)), float(np.max(v)), int(len(v))] if len(v) else [None, None, None, 0]
            o = np.argsort(-y)
            cs = np.cumsum(fy[o]) * -1.0                                    # 위쪽 노드 힘 합의 부호 반전 -> 양수 = 아래로 당김
            ys = y[o]
            def T_at(v):
                k = np.searchsorted(-ys, -v)
                return float(cs[min(max(k - 1, 0), len(cs) - 1)])
            Tprof = [(round(float(v), 3), T_at(v)) for v in np.arange(yg - 0.3, yg + 0.31, 0.05)]
            ll = low & sl
            r[nm] = dict(
                yg=float(yg), n_act=int(len(m)), n_low=int(low.sum()), n_low_sl=int(ll.sum()), n_act_sl=int(sl.sum()),
                low_ext_frac=float((dmin > 1.5).mean()) if len(dmin) else None,
                low_fel_mag_med=float(np.median(fm[low])) if low.any() else None, nonlow_fel_mag_med=float(np.median(fm[~low])),
                low_sl_bridge_mean=float(np.mean(br[ll])) if ll.any() else None,
                low_sl_bridge_frac=float(np.mean(br[ll] > 0)) if ll.any() else None,
                low_sl_bridge_sum=float(np.sum(br[ll])) if ll.any() else None,
                low_sl_fel_mag_med=float(np.median(fm[ll])) if ll.any() else None,
                nonlow_sl_bridge_mean=float(np.mean(br[~low & sl])) if (~low & sl).any() else None,
                nonlow_sl_bridge_sum=float(np.sum(br[~low & sl])) if (~low & sl).any() else None,
                sameA_low_n=int(inA.sum()), sameA_low_sl_n=int((inA & sl).sum()),
                sameA_low_sl_bridge_mean=float(np.mean(br[inA & sl])) if (inA & sl).any() else None,
                sameA_low_sl_bridge_frac=float(np.mean(br[inA & sl] > 0)) if (inA & sl).any() else None,
                sameA_low_sl_fel_mag_med=float(np.median(fm[inA & sl])) if (inA & sl).any() else None,
                T_profile=Tprof, T_gap=T_at(yg), T_far=[T_at(yg - 0.25), T_at(yg + 0.25)],
                T_min_in_prof=float(min(t for _, t in Tprof)),
                sig_slab=st(sig[qs]), sig_narrow=st(sig[qn]), sig_all=st(sig),
                sig_slab_lowcenter=st(sig[qs & qlow]), sig_slab_normal=st(sig[qs & ~qlow]),
                sig_pos_frac_narrow=float(np.mean(sig[qn] > 0)) if qn.any() else None,
            )
        res.append(r)
        g = lambda v: "-" if v is None else f"{v:.3g}"
        md.append(f"| {f} | {ygt:.3f} | {rg['gt']['gap'][0]:.3f} / {rg['A']['gap'][0]:.3f} / {rg['B']['gap'][0]:.3f} | " + " | ".join(
            f"{r[n]['n_low']} / {r[n]['n_low_sl']} | {g(r[n]['low_sl_bridge_mean'])} | {g(r[n]['T_gap'])} | {g(r[n]['sig_narrow'][0])} / {g(r[n]['sig_narrow'][1])}" for n in ("A", "B")) + " |")
    json.dump(res, open(f"{out}/node_stats.json", "w"), ensure_ascii=False)
    open(f"{out}/cm160_tables.md", "w").write(
        "| 프레임 | 정답 틈 y | 틈 너비 정답 / A / B | A 저질량 노드 전체 / 틈 안 | A 틈 안 저질량 이음 방향 힘 평균 | A 단면 인장 T | A 틈 σ 평균 / p90 | "
        "B 저질량 노드 전체 / 틈 안 | B 틈 안 저질량 이음 방향 힘 평균 | B 단면 인장 T | B 틈 σ 평균 / p90 |\n" + "|" + "---|" * 11 + "\n" + "\n".join(md) + "\n")
    print(open(f"{out}/cm160_tables.md").read())


if __name__ == "__main__":
    {"gt": stage_gt, "part": stage_part, "node": stage_node}[sys.argv[1]](sys.argv[2:])
