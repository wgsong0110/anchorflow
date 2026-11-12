"""절 CM111 집계 (CPU, 읽기만): lib/anchorflow/fem_cellforce.py 의 <판>/cf/f*.npz (칸별 항별 일반화 힘) -> <O>/cellforce_<판>.json.
사용: python exe/cellforce_an.py <O> <판> [<판> ...]
 - 항 묶음: 관성 (중력 뺀 m(y−ŷ)/τ²) · 중력 · 질량하한 ((M_eff − m)/τ²) · 탄성 (요소 커널) · 장벽 (rt_bar_n) · 정규화 (나머지 이름 통) · 밖 ('~' 통 = s.k 밖에서 f 를 바꾼 몫)
   · 합 (= 남은 기울기, 우리 재합) · 검사 (풀이기 잔차 r0 의 법선 성분으로 만든 같은 값).
 - F > 0 = 칸을 누르는 쪽, < 0 = 벌리는 (버티는) 쪽. 단위 N (일반화 힘, d_c 단위 길이).
 - 구간: 프레임 묶음 (f5 · f8~9 · f10~30 · f31~50) x (두께 비 r 구간 | 층). 칸마다 그 프레임의 r·층.
"""
import json
import os
import sys

import numpy as np

O = sys.argv[1]
RUNS = sys.argv[2:]
PH = [("f5", [5]), ("f8~9", [8, 9]), ("f10~30", list(range(10, 31))), ("f31~50", list(range(31, 51)))]
RE = [-np.inf, 0.5, 0.7, 0.9, 0.98, 1.02, np.inf]
RL = ["<0.5", "0.5~0.7", "0.7~0.9", "0.9~0.98", "0.98~1.02", ">1.02"]
LL = ["0", "1", "2", "3", "4", "5+"]
GR = ["관성", "중력", "질량하한", "탄성", "장벽", "정규화", "밖", "합", "검사"]
PHYS = ["관성", "중력", "질량하한", "탄성", "장벽", "정규화", "밖"]


def group_of(nm):
    if nm in ("관성", "중력", "질량하한"):
        return nm
    if nm == "탄성":
        return "탄성"
    if nm == "rt_bar_n":
        return "장벽"
    if nm == "합":
        return "합"
    if nm == "풀이기잔차":
        return "검사"
    if nm.startswith("~"):
        return "밖"
    return "정규화"


def stats(G, m):
    """G: {묶음: [칸]}, m: 칸 고름 -> 묶음별 중앙·평균·양수 몫·몫 중앙."""
    n = int(m.sum())
    o = dict(n=n)
    if n == 0:
        return o
    den = sum(np.abs(G[g][m]) for g in PHYS)
    den = np.where(den > 0, den, 1.0)
    for g in GR:
        v = G[g][m]
        o[g] = dict(med=float(np.median(v)), mean=float(v.mean()), pos=float((v > 0).mean()), p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
                    share=float(np.median(v / den)) if g in PHYS + ["합"] else None)
    return o


for run in RUNS:
    d = os.path.join(O, run, "cf")
    if not os.path.isdir(d):
        print(f"[{run}] cf 없음", flush=True)
        continue
    fs = sorted(int(f[1:4]) for f in os.listdir(d) if f.endswith(".npz"))
    FR = {}
    names = None
    for f in fs:
        z = np.load(os.path.join(d, f"f{f:03d}.npz"))
        terms = [str(q) for q in z["terms"]]
        names = terms
        rec = dict(r=z["r"].astype(np.float64), lay=z["lay"].astype(np.int64), m=z["m"].astype(np.float64), ne=z["ne"], nt=z["nt"])
        for sl, key in (("시작", "F0"), ("끝", "F1")):
            F = z[key].astype(np.float64)
            G = {g: np.zeros(F.shape[1]) for g in GR}
            raw = {}
            for i, nm in enumerate(terms):
                G[group_of(nm)] += F[i]
                raw[nm] = F[i]
            rec[sl] = G; rec[sl + "_raw"] = raw
        FR[f] = rec
    out = dict(run=run, frames=fs, terms=names, phases={}, series={}, check={}, regnames={})
    # 정규화·밖 이름별 크기 (끝 칸, 모든 프레임 칸 |F| 합)
    for nm in (names or []):
        g = group_of(nm)
        if g in ("정규화", "밖", "탄성", "장벽"):
            out["regnames"][nm] = dict(group=g, sumabs=float(sum(np.abs(FR[f]["끝_raw"][nm]).sum() for f in fs)))
    for ph, pf in PH:
        pf = [f for f in pf if f in FR]
        if not pf:
            continue
        o = dict(frames=pf)
        for sl in ("시작", "끝"):
            G = {g: np.concatenate([FR[f][sl][g] for f in pf]) for g in GR}
            r = np.concatenate([FR[f]["r"] for f in pf]); lay = np.concatenate([FR[f]["lay"] for f in pf])
            ne = np.concatenate([FR[f]["ne"] for f in pf]); nt = np.concatenate([FR[f]["nt"] for f in pf])
            o[sl] = dict(all=stats(G, np.ones(len(r), bool)), rbin={}, layer={}, rbin_inner={})
            for i, lb in enumerate(RL):
                m = (r >= RE[i]) & (r < RE[i + 1])
                o[sl]["rbin"][lb] = stats(G, m)
                o[sl]["rbin_inner"][lb] = stats(G, m & (ne == 0) & (nt == 0))             # 확장 묶인 노드·바닥 T ≠ I 노드 없는 칸만
            for i, lb in enumerate(LL):
                m = (lay == i) if i < 5 else (lay >= 5)
                o[sl]["layer"][lb] = stats(G, m)
        out["phases"][ph] = o
    for f in fs:
        R = FR[f]
        s_ = {}
        for sl in ("시작", "끝"):
            G = R[sl]
            s_[sl] = {g: float(np.median(G[g])) for g in GR}
            m = R["r"] < 0.9
            s_[sl + "_r<0.9"] = {g: float(np.median(G[g][m])) for g in GR} if m.any() else None
        s_["n"] = int(len(R["r"])); s_["n_r<0.9"] = int((R["r"] < 0.9).sum()); s_["r_med"] = float(np.median(R["r"]))
        dif = np.abs(R["끝"]["합"] - R["끝"]["검사"]); ref = np.abs(R["끝"]["검사"]).max()
        s_["chk"] = [float(dif.max()), float(ref)]
        out["series"][f] = s_
    jl = os.path.join(O, run, "cf.jsonl")
    if os.path.exists(jl):
        L = [json.loads(q) for q in open(jl) if q.strip()]
        last = {}
        for q in L:
            last[q["fr"]] = q                                                                 # 같은 프레임을 다시 하면 마지막 (받아들인 시도)
        out["check"] = {int(k): dict(itb=v.get("itb"), stat={sl: {kk: vv for kk, vv in st.items() if kk != "sumabs"} for sl, st in (v.get("stat") or {}).items()}, ncell=v.get("ncell"),
                                     sumabs=(v.get("stat") or {}).get("끝", {}).get("sumabs")) for k, v in last.items()}
        out["n_lines"] = len(L)
    json.dump(out, open(os.path.join(O, f"cellforce_{run}.json"), "w"), ensure_ascii=False)
    P = out["phases"].get("f10~30") or next(iter(out["phases"].values()), None)
    print(f"[{run}] 프레임 {len(fs)} 개, 항 {names}", flush=True)
    if P:
        for lb in RL:
            st = P["끝"]["rbin"][lb]
            if st.get("n"):
                print(f"[{run}] 끝 r {lb} n {st['n']}: " + " ".join(f"{g} {st[g]['med']:.3g}" for g in GR), flush=True)
    print(f"[{run}] 끝 -> {O}/cellforce_{run}.json", flush=True)
