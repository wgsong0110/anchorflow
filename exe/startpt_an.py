"""절 CM113 집계 (CPU, 읽기만): lib/anchorflow/fem_startpt.py 의 <판>/sp/f*.npz -> <O>/startpt_<판>.json.
사용: python exe/startpt_an.py <O> <판> [<판> ...]
 - 걸음마다 사본 차례 (U_이전끝 -> rt_ga_step0 뒤 Ut·y -> y 를 받는 커널 뒤 y ... -> y0_시작점·yh_기준 -> y_갱신 (뉴턴 반복) ...), 앞 y 와 같은 y 사본은 뺌.
 - 사본마다 층별 (0..4, 5+, 전체) 칸 r 통계: 수, r<0.98 몫, r<0.9 몫, 중앙, 10 백분위, 최소.
 - y 사본 사이 변화 (전 y -> 이 y): r 이 0.005 넘게 준 칸 수, 그 칸들의 Δr 중앙과 구속 노드 몫 Δr_C (T≈0 꼭짓점만 옮김) · 자유 노드 몫 Δr_F 중앙.
 - 평가별 항 (절 CM111 F 정의, 양수 = 누름) 중앙값: 시작점에서 눌린 칸 (r(y0) < 0.98) 과 나머지.
"""
import json
import os
import sys

import numpy as np

O = sys.argv[1]
RUNS = sys.argv[2:]
LL = ["0", "1", "2", "3", "4", "5+", "전체"]


def lmask(lay, i):
    return np.ones(len(lay), bool) if i == 6 else ((lay == i) if i < 5 else (lay >= 5))


def rstat(r, lay):
    o = {}
    for i, lb in enumerate(LL):
        m = lmask(lay, i) & np.isfinite(r)
        v = r[m]
        if len(v) == 0:
            continue
        o[lb] = dict(n=int(len(v)), f098=float((v < 0.98).mean()), f09=float((v < 0.9).mean()), med=float(np.median(v)), p10=float(np.percentile(v, 10)), min=float(v.min()))
    return o


def group(nm):
    if nm in ("관성", "중력", "질량하한", "합", "풀이기잔차"):
        return nm
    if nm == "탄성":
        return "탄성"
    if nm == "rt_bar_n":
        return "장벽"
    if nm.startswith("~"):
        return "밖"
    return "정규화"


for run in RUNS:
    d = os.path.join(O, run, "sp")
    if not os.path.isdir(d):
        print(f"[{run}] sp 없음", flush=True)
        continue
    fs = sorted(int(f[1:4]) for f in os.listdir(d) if f.endswith(".npz"))
    out = dict(run=run, frames={})
    for f in fs:
        z = np.load(os.path.join(d, f"f{f:03d}.npz"), allow_pickle=False)
        steps = json.loads(str(z["steps"]))
        lay = z["lay"].astype(np.int64)
        terms = [str(q) for q in z["terms"]]
        G = sorted(set(group(t) for t in terms), key=lambda g: ["관성", "중력", "질량하한", "탄성", "장벽", "정규화", "밖", "합", "풀이기잔차"].index(g))
        FO = dict(rEnd=rstat(z["rEnd"].astype(np.float64), lay), steps=[])
        for st in steps:
            k = st["k"]
            R = z[f"k{k}_r"].astype(np.float64); RC = z[f"k{k}_rC"].astype(np.float64); RF = z[f"k{k}_rF"].astype(np.float64)
            ncons = z[f"k{k}_ncons"]; nxt = z[f"k{k}_next"]
            seq = []
            prevr = None; prev_tag = None
            r0 = None
            for j, tg in enumerate(st["tags"]):
                r = R[j]; kind = tg[0]
                isy = not (kind.startswith("Ut") or kind.startswith("yh"))
                if isy and prevr is not None and np.array_equal(r, prevr) and not kind.startswith("y0"):
                    continue
                e = dict(j=j, kind=kind, kernel=tg[1], it=int(tg[2]), r=rstat(r, lay))
                if isy and prevr is not None:
                    dr = r - prevr
                    m = dr < -0.005
                    e["dec"] = dict(n=int(m.sum()), n_new098=int(((r < 0.98) & (prevr >= 0.98)).sum()),
                                    dr_med=float(np.median(dr[m])) if m.any() else None,
                                    drC_med=float(np.median(RC[j][m] - prevr[m])) if m.any() else None,
                                    drF_med=float(np.median(RF[j][m] - prevr[m])) if m.any() else None,
                                    cons_corner_med=float(np.median(ncons[m])) if m.any() else None, ext_corner_med=float(np.median(nxt[m])) if m.any() else None,
                                    lay={LL[i]: int((m & lmask(lay, i)).sum()) for i in range(6)})
                if kind.startswith("y0"):
                    r0 = r
                seq.append(e)
                if isy:
                    prevr = r
            S = dict(k=k, ncons=st["ncons"], nact=st["nact"], seq=seq, ev={})
            if f"k{k}_F" in z.files and r0 is not None:
                F = z[f"k{k}_F"].astype(np.float64)
                for ei, e in enumerate(st["ev"]):
                    Gs = {g: np.zeros(F.shape[2]) for g in G}
                    for ti, t in enumerate(terms):
                        Gs[group(t)] += F[ei, ti]
                    o = {}
                    for nm, m in (("y0 눌린 칸 (r<0.98)", r0 < 0.98), ("y0 r<0.9", r0 < 0.9), ("나머지", r0 >= 0.98)):
                        if m.any():
                            o[nm] = dict(n=int(m.sum()), **{g: dict(med=float(np.median(Gs[g][m])), pos=float((Gs[g][m] > 0).mean())) for g in G})
                    S["ev"][str(e)] = o
            FO["steps"].append(S)
        out["frames"][str(f)] = FO
        # 요약 줄: 처음 r<0.98 칸을 들여온 단계
        for S in FO["steps"]:
            msg = []
            for e in S["seq"]:
                a = e["r"].get("전체", {})
                dec = e.get("dec") or {}
                msg.append(f"{e['kind']}:{e['kernel']}{'' if e['it'] < 0 else e['it']} f098={a.get('f098', 0):.4f} min={a.get('min', 1):.3f}" + (f" new={dec.get('n_new098')}" if dec else ""))
            print(f"[{run}] f{f} k{S['k']} 구속 {S['ncons']}/{S['nact']}: " + " | ".join(msg), flush=True)
    json.dump(out, open(os.path.join(O, f"startpt_{run}.json"), "w"), ensure_ascii=False)
    print(f"[{run}] 끝 -> {O}/startpt_{run}.json", flush=True)
