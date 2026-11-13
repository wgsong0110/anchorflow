"""절 CM131 분석 (로컬, numpy 만): 판마다 R2 lbours/wolfcv/cm131/<판>/{esc,rm}/fNNN.npz 를 한 프레임씩 /tmp 로 받아 걸음·뉴턴 평가별 노드 잔차를 노드 부류로 나눔.
사용: python3 exe/cm131_an.py --runs h9,g9,h5,k5 --frames 5:30 --out <json> [--tmp /tmp/cm131w] [--keep 0]
노드 부류 (걸음 k 마다; 노드 = mn > 0, fem_escwhy·fem_resmode 와 같은 nid):
  질량비 m/m_ref (m_ref = 그 프레임 노드 질량 95 % 분위), 받침 입자 수 (입자 프레임 시작 위치의 2 차 B-스플라인 27 노드), 꼭짓점 몫 (둘레 가우스 점 부피 중 DP 꼭짓점 몫, 커밋 앞 QS),
  확장: 외부 (W8 ≥ 0) · 주인 (외부 노드의 외삽 가중 c_oj ≠ 0 인 노드; 하중 Σ c² ) · 나머지, 고스트 면 인접 (g3fm 면의 36 노드), 바닥 장벽 (BK > 0).
잔차: R0 (안 접은 r0 = −T g, 절 CM117 |g|/|g0| 와 같은 버퍼), RF (접은 r = 뉴턴이 보는 것). 평가 = OK[k, e, 0] 인 e (sp_g2 와 같은 판정).
정체 모드 가늠 (평가 e -> e+1, 실제 갱신 Δy = Y[e+1] − Y[e], Δg = −(R0[e+1] − R0[e])):
  할선 강성 S = Σ Δg·Δy / Σ Mi |Δy|² (Mi = meff/τ²; 1 이면 그 부류의 참 응답이 질량뿐 = 거의 자유), 헤시안 대각 몫 Hd = Σ Δy·B Δy / Σ Mi |Δy|² (대각 블록만 -- 결합 빼 과대),
  자유 걸음 몫 φ = Σ Δy·r / Σ |r|²/Mi (1 = 질량만 있는 노드의 정확한 뉴턴 걸음을 다 감).
"""
import argparse
import json
import os
import subprocess

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="h9,g9,h5,k5")
ap.add_argument("--frames", default="5:30")
ap.add_argument("--out", required=True)
ap.add_argument("--r2", default="r2:storage/result/anchorflow/lbours/wolfcv/cm131")
ap.add_argument("--tmp", default="/tmp/cm131w")
ap.add_argument("--keep", type=int, default=0)
ap.add_argument("--pn", type=float, default=0.48)
a = ap.parse_args()
os.makedirs(a.tmp, exist_ok=True)
f0, f1 = [int(q) for q in a.frames.split(":")]


def fetch(run, sub, f):
    p = os.path.join(a.tmp, f"{run}_{sub}_f{f:03d}.npz")
    if not os.path.exists(p):
        r = subprocess.run(["rclone", "copyto", f"{a.r2}/{run}/{sub}/f{f:03d}.npz", p, "-q"])
        if r.returncode != 0 or not os.path.exists(p):
            return None
    return p


def key(c):
    return (c[:, 0].astype(np.int64) * 4096 + c[:, 1]) * 4096 + c[:, 2]


def ext_weights(x):
    """축 거리 x (−2..2) -> 블록 노드 (−1,0,1) 가중 (fem_wolfh8 차수 2)."""
    if x == 2:
        return (1.0, -3.0, 3.0)
    if x == -2:
        return (3.0, -3.0, 1.0)
    w = [0.0, 0.0, 0.0]
    w[x + 1] = 1.0
    return tuple(w)


def classes(z, m, k):
    IP = z["IP"]
    nb1, nb2 = int(IP[7]), int(IP[8])
    nid = z["nid"].astype(np.int64)
    nn = len(nid)
    nrel = np.stack([nid // (nb1 * nb2), (nid // nb2) % nb1, nid % nb2], 1)
    sidx = np.argsort(nid)

    def idx_of(g):
        j = np.searchsorted(nid, g, sorter=sidx)
        j = np.clip(j, 0, nn - 1)
        jj = sidx[j]
        return np.where(nid[jj] == g, jj, -1)
    mn = z["mn"].astype(np.float64)
    mref = np.quantile(mn, 0.95)
    # 받침 입자 수
    x = z["p0_x"]
    ok = np.isfinite(x).all(1)
    y = x[ok] @ z["Ai"].T
    b = np.floor(y - 0.5).astype(np.int64) - IP[0:3]
    cnt = np.zeros(nn)
    for i in range(3):
        for j in range(3):
            for l in range(3):
                g = ((b[:, 0] + i) * nb1 + (b[:, 1] + j)) * nb2 + (b[:, 2] + l)
                jj = idx_of(g)
                cnt += np.bincount(jj[jj >= 0], minlength=nn)[:nn]
    # 꼭짓점 몫 (커밋 앞 QS 상태), 칸 -> 노드
    cb = z["cb"].astype(np.int64)
    nc = len(cb)
    ck = key(cb)
    order = np.argsort(ck)
    cks = ck[order]
    QS = z["QS"][k]
    st = QS[:, 3].astype(int).reshape(-1, 8)
    Vc = z["Vq"].reshape(-1, 8)
    Vap = np.zeros(nn)
    Vall = np.zeros(nn)
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                c = nrel - np.array([di, dj, dk])
                kk = key(c)
                j = np.clip(np.searchsorted(cks, kk), 0, nc - 1)
                okc = cks[j] == kk
                ci = order[j]
                v = np.where(okc[:, None], Vc[ci], 0)
                Vall += v.sum(1)
                Vap += (v * (st[ci] == 2)).sum(1)
    fap = np.where(Vall > 0, Vap / np.maximum(Vall, 1e-30), 0.0)
    # 확장: 외부·주인 (하중 Σ c²)
    W8 = z["W8"][k].astype(np.int64)
    ext = W8 >= 0
    load = np.zeros(nn)
    for o in np.nonzero(ext)[0]:
        bb = W8[o]
        bc = np.array([bb // (nb1 * nb2), (bb // nb2) % nb1, bb % nb2])
        xs = nrel[o] - bc - 1
        if np.any(np.abs(xs) > 2):
            continue
        w0, w1, w2 = ext_weights(int(xs[0])), ext_weights(int(xs[1])), ext_weights(int(xs[2]))
        for p_ in range(3):
            for q_ in range(3):
                for r_ in range(3):
                    ww = w0[p_] * w1[q_] * w2[r_]
                    if ww != 0.0:
                        jj = idx_of(bb + (p_ * nb1 + q_) * nb2 + r_)
                        if jj >= 0:
                            load[jj] += ww * ww
    owner = (load > 0) & ~ext
    # 고스트 면 인접 (g3fm 비트 a: 칸 c 와 c + e_a 사이 면 -> 36 노드)
    gadj = np.zeros(nn, bool)
    if "g3fm" in m.files:
        fm = m["g3fm"][:nc].astype(np.int64)
        for a_ in range(3):
            cs = cb[(fm >> a_) & 1 == 1]
            if len(cs) == 0:
                continue
            ext_ = [3, 3, 3]
            ext_[a_] = 4
            for di in range(ext_[0]):
                for dj in range(ext_[1]):
                    for dk in range(ext_[2]):
                        g = ((cs[:, 0] + di) * nb1 + (cs[:, 1] + dj)) * nb2 + (cs[:, 2] + dk)
                        jj = idx_of(g)
                        gadj[jj[jj >= 0]] = True
    bar = z["BK"][k] > 0
    pos = (nrel + IP[0:3]) @ z["A"].T
    hz = pos[:, 2] + z["U0"][k][:, 2] - a.pn
    return dict(mr=mn / mref, cnt=cnt, fap=fap, ext=ext, owner=owner, load=load, gadj=gadj, bar=bar, hz=hz, mref=mref)


BINS = {
    "질량비": ("mr", [0, 1e-3, 1e-2, 0.1, 0.5, 1e9]),
    "받침 입자": ("cnt", [0, 1, 9, 65, 257, 1e12]),
    "꼭짓점 몫": ("fap", [0, 0.25, 0.5, 0.75, 0.95, 1.01]),
}


def masks(C):
    out = {}
    for nm, (fld, edges) in BINS.items():
        v = C[fld]
        for lo, hi in zip(edges[:-1], edges[1:]):
            out[f"{nm} [{lo:g},{hi:g})"] = (v >= lo) & (v < hi)
    out["확장 외부"] = C["ext"]
    out["확장 주인"] = C["owner"]
    out["확장 무관"] = ~C["ext"] & ~C["owner"]
    out["고스트 면 인접"] = C["gadj"]
    out["고스트 면 밖"] = ~C["gadj"]
    out["바닥 장벽"] = C["bar"]
    out["바닥 3 cm 안"] = C["hz"] < 0.03
    out["저질량 (<0.1) · 꼭짓점 ≥0.5"] = (C["mr"] < 0.1) & (C["fap"] >= 0.5)
    out["저질량 (<0.1) · 고스트 인접"] = (C["mr"] < 0.1) & C["gadj"]
    out["전체"] = np.ones_like(C["ext"])
    return out


def n2(v):
    return (v.astype(np.float64) ** 2).sum(-1)


res = {}
for run in a.runs.split(","):
    rows = []
    for f in range(f0, f1 + 1):
        pe, pr = fetch(run, "esc", f), fetch(run, "rm", f)
        if pe is None or pr is None:
            print(f"[{run}] f{f} 없음 (esc {pe is not None} rm {pr is not None})", flush=True)
            continue
        z, m = np.load(pe), np.load(pr)
        if "nid" not in z.files or "nid" not in m.files:
            print(f"[{run}] f{f} 기록 없음", flush=True)
            continue
        if len(z["nid"]) != len(m["nid"]) or not np.array_equal(z["nid"], m["nid"]):
            print(f"[{run}] f{f} nid 다름 -- 건너뜀", flush=True)
            continue
        K = min(int(z["K"]), int(m["K"]), m["R0"].shape[0])
        for k in range(K):
            C = classes(z, m, k)
            M = masks(C)
            OK = m["OK"][k]
            ev = [e for e in range(OK.shape[0]) if OK[e, 0] == 1]
            if not ev:
                continue
            R0, RF, Y, DL = m["R0"][k].astype(np.float64), m["RF"][k].astype(np.float64), m["Y"][k].astype(np.float64), m["DL"][k].astype(np.float64)
            Mi = m["ME"][k].astype(np.float64) * float(m["IT2"][k])
            B = m["B"][k].astype(np.float64).reshape(-1, 3, 3)
            e0, eL = ev[0], ev[-1]
            g0, gL = n2(R0[e0]), n2(R0[eL])
            f0_, fL = n2(RF[e0]), n2(RF[eL])
            row = dict(f=f, k=k, nev=len(ev), ev=ev, gr=float(np.sqrt(gL.sum() / max(g0.sum(), 1e-300))), grF=float(np.sqrt(fL.sum() / max(f0_.sum(), 1e-300))),
                       nn=int(len(Mi)), mref=float(C["mref"]), cls={})
            # 이웃 평가 사이: 코사인, 부류별
            cos = []
            for e, e2 in zip(ev[:-1], ev[1:]):
                a1, a2 = R0[e].ravel(), R0[e2].ravel()
                cos.append(float(a1 @ a2 / max(np.linalg.norm(a1) * np.linalg.norm(a2), 1e-300)))
            row["cos"] = cos
            pr_ = gL / max(gL.sum(), 1e-300)
            row["neff_last"] = float(1.0 / max((pr_ ** 2).sum(), 1e-300))
            for nm, mk in M.items():
                if not mk.any():
                    continue
                d = dict(n=int(mk.sum()), sh0=float(g0[mk].sum() / max(g0.sum(), 1e-300)), shL=float(gL[mk].sum() / max(gL.sum(), 1e-300)),
                         shF0=float(f0_[mk].sum() / max(f0_.sum(), 1e-300)), shFL=float(fL[mk].sum() / max(fL.sum(), 1e-300)),
                         rho=float(np.sqrt(gL[mk].sum() / max(g0[mk].sum(), 1e-300))), rhoF=float(np.sqrt(fL[mk].sum() / max(f0_[mk].sum(), 1e-300))))
                S, Hd, phi = [], [], []
                for e, e2 in zip(ev[:-1], ev[1:]):
                    dy = Y[e2][mk] - Y[e][mk]
                    dg = -(R0[e2][mk] - R0[e][mk])
                    den = (Mi[mk][:, None] * dy ** 2).sum()
                    if den > 0:
                        S.append(float((dg * dy).sum() / den))
                        Hd.append(float(np.einsum("ni,nij,nj->", dy, B[mk], dy) / den))
                    r = R0[e][mk]
                    fden = (n2(r) / np.maximum(Mi[mk], 1e-300)).sum()
                    if fden > 0:
                        phi.append(float((dy * r).sum() / fden))
                d.update(S=S, Hd=Hd, phi=phi)
                row["cls"][nm] = d
            rows.append(row)
        print(f"[{run}] f{f}: 걸음 {K} 평가 {[r_['nev'] for r_ in rows if r_['f'] == f]} |g|/|g0| {[round(r_['gr'], 4) for r_ in rows if r_['f'] == f]}"
              f" 접은 {[round(r_['grF'], 4) for r_ in rows if r_['f'] == f]}", flush=True)
        if not a.keep:
            os.remove(pe)
            os.remove(pr)
    res[run] = rows
    json.dump(res, open(a.out, "w"), ensure_ascii=False)


# 요약 표
def med(v):
    v = [q for q in v if q is not None and np.isfinite(q)]
    return float(np.median(v)) if v else float("nan")


print("\n== 요약 (프레임 %d~%d, 걸음 중앙)" % (f0, f1))
for run, rows in res.items():
    if not rows:
        continue
    print(f"\n[{run}] 걸음 {len(rows)} 평가 수 중앙 {med([r['nev'] for r in rows])} |g|/|g0| (안 접음) 중앙 {med([r['gr'] for r in rows]):.3f} (접음) {med([r['grF'] for r in rows]):.3f}"
          f" 이웃 평가 잔차 코사인 중앙 {med([c for r in rows for c in r['cos']]):.3f} 끝 잔차 유효 노드 수 중앙 {med([r['neff_last'] for r in rows]):.0f} / 노드 {med([r['nn'] for r in rows]):.0f}")
    names = list(rows[0]["cls"].keys())
    for r in rows[1:]:
        for nm in r["cls"]:
            if nm not in names:
                names.append(nm)
    print(f"  {'부류':<28} {'노드 몫':>7} {'r0² 몫 처음':>10} {'끝':>6} {'접은 처음':>8} {'끝':>6} {'부류 |r|비':>9} {'접은':>6} {'할선 S':>8} {'대각 Hd':>9} {'자유 몫 φ':>9}")
    for nm in names:
        ds = [r["cls"][nm] for r in rows if nm in r["cls"]]
        nsh = med([d["n"] / r["nn"] for r in rows for d in ([r["cls"][nm]] if nm in r["cls"] else [])])
        print(f"  {nm:<28} {nsh:7.3f} {med([d['sh0'] for d in ds]):10.3f} {med([d['shL'] for d in ds]):6.3f} {med([d['shF0'] for d in ds]):8.3f} {med([d['shFL'] for d in ds]):6.3f}"
              f" {med([d['rho'] for d in ds]):9.3f} {med([d['rhoF'] for d in ds]):6.3f} {med([q for d in ds for q in d['S']]):8.2f} {med([q for d in ds for q in d['Hd']]):9.2f} {med([q for d in ds for q in d['phi']]):9.3f}")
