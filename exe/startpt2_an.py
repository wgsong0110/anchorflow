"""절 CM114 집계 (CPU, 읽기만): lib/anchorflow/fem_startpt.py (+ 절 CM114 몫) 의 <판>/sp/f*.npz -> <O>/startpt2_<판>.json.
사용: python exe/startpt2_an.py <O> <판> [<판> ...]
 (1) 되먹임: 프레임 f 칸마다 Ũ r (관성 예측), 걸음 시작 속도장 세로 압축률 (위 4 꼭짓점 평균 n·v − 아래 4 평균) / 시작 두께 (1/s, 음수 = 줄어듦),
     앞 프레임 끝 r (그 칸 입자들이 앞 프레임에 들었던 칸의 프레임 끝 r 평균; 입자 차례는 프레임 사이 고정).
     -> 순위 상관 (스피어만), 앞 끝 r < 0.98 여부별 Ũ r < 0.98 몫, Ũ 눌린 칸 중 앞 끝 눌린 몫, 압축률 구간별 Ũ r 중앙.
 (2) mt_floor_n 처음 눌림: Ũ 에서는 r ≥ 0.98 인데 mt_floor_n 뒤 r < 0.98 인 칸 (f4 는 처음 눌린 20 칸) 의 꼭짓점 구속 몫 k_구속 (kb: 1 풀림, 0 묶임),
     바닥 고착 몫, mt_floor_n 이 덮어쓴 법선 변위 (y − Ũ), 칸 입자 중 아직 방출 안 된 몫 -- 모든 칸과 비교.
"""
import json
import os
import sys

import numpy as np

O = sys.argv[1]
RUNS = sys.argv[2:]


def rank(a):
    o = np.empty(len(a)); o[np.argsort(a, kind="mergesort")] = np.arange(len(a)); return o


def spear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return None
    return float(np.corrcoef(rank(a[m]), rank(b[m]))[0, 1])


def kbclass(kb):
    return dict(free=int((kb > 0.999).sum()), partial=int(((kb > 1e-3) & (kb <= 0.999)).sum()), held=int((kb <= 1e-3).sum()))


for run in RUNS:
    d = os.path.join(O, run, "sp")
    fs = sorted(int(f[1:4]) for f in os.listdir(d) if f.endswith(".npz"))
    Z = {f: np.load(os.path.join(d, f"f{f:03d}.npz")) for f in fs}
    out = dict(run=run, fb={}, mt={})
    for f in fs:
        z = Z[f]
        st = json.loads(str(z["steps"]))[0]
        tags = [t[0] + ":" + t[1] for t in st["tags"]]
        R = z["k0_r"]
        if "Ut_관성예측:rt_ga_step0" not in tags:
            continue
        rU = R[tags.index("Ut_관성예측:rt_ga_step0")].astype(np.float64)
        rmt = R[tags.index("y:mt_floor_n")].astype(np.float64) if "y:mt_floor_n" in tags else rU
        vr = z["k0_vrate"].astype(np.float64) if "k0_vrate" in z.files else None
        o = dict(ncell=int(len(rU)), nU098=int((rU < 0.98).sum()))
        if vr is not None:
            o["corr_vrate_rU"] = spear(vr, rU)
            q = np.percentile(vr, [1, 5, 25, 50, 75, 95, 99])
            o["vrate_q"] = [float(x) for x in q]
            o["vrate_compU"] = float(np.median(vr[rU < 0.98])) if (rU < 0.98).any() else None
            o["vrate_rest"] = float(np.median(vr[rU >= 0.98]))
        if f - 1 in Z and "pcell" in Z[f - 1].files:
            zp = Z[f - 1]
            pc0, pc1 = zp["pcell"].astype(np.int64), z["pcell"].astype(np.int64)
            ok = (pc0 >= 0) & (pc1 >= 0)
            rend = zp["rEnd"].astype(np.float64)
            pr = rend[pc0[ok]]
            nc = len(rU)
            s_ = np.bincount(pc1[ok], weights=pr, minlength=nc); c_ = np.bincount(pc1[ok], minlength=nc)
            prev = np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)
            pm = prev < 0.98; um = rU < 0.98
            o.update(corr_prev_rU=spear(prev, rU), corr_prev_vrate=spear(prev, vr) if vr is not None else None,
                     n_prev098=int(np.nansum(pm)), pU_given_prev=float(um[pm].mean()) if pm.any() else None,
                     pU_given_notprev=float(um[~pm & np.isfinite(prev)].mean()), prev_among_U=float(pm[um].mean()) if um.any() else None,
                     rU_med_prev=float(np.median(rU[pm])) if pm.any() else None, rU_med_notprev=float(np.median(rU[~pm & np.isfinite(prev)])))
        out["fb"][str(f)] = o
        # (2)
        if "k0_kb" in z.files:
            new = (rmt < 0.98) & (rU >= 0.98)
            kb = z["k0_kb"].astype(np.float64); stk = z["k0_stk"].astype(np.float64)
            dmt = z["k0_dmt"].astype(np.float64) if "k0_dmt" in z.files else np.zeros_like(kb)
            held = z["held"].astype(np.float64)
            lay = z["lay"].astype(np.int64)

            def desc(m):
                if not m.any():
                    return dict(n=0)
                kk = kb[m].reshape(-1)
                pc = ((kb[m] > 1e-3) & (kb[m] <= 0.999)).sum(1)
                return dict(n=int(m.sum()), corner=kbclass(kk), cell_any_partial=float((pc > 0).mean()), cell_partial_med=float(np.median(pc)),
                            cell_mixed=float((((kb[m] > 0.999).any(1)) & ((kb[m] <= 0.999).any(1))).mean()),
                            kb_min_med=float(np.median(kb[m].min(1))), kb_max_med=float(np.median(kb[m].max(1))),
                            stk_max_med=float(np.median(stk[m].max(1))), stk_any=float((stk[m].max(1) > 1e-3).mean()),
                            held_med=float(np.median(held[m])), held_any=float((held[m] > 0).mean()), held_mixed=float(((held[m] > 0) & (held[m] < 1)).mean()),
                            dmt_absmax_med=float(np.median(np.abs(dmt[m]).max(1))) if dmt.any() else None,
                            dmt_top_minus_bot_med=None, lay={str(i): int((m & (lay == i)).sum()) for i in range(6)})
            o2 = dict(new=desc(new), all=desc(np.ones(len(rU), bool)))
            if "k0_dmt_tb" in z.files:
                tb = z["k0_dmt_tb"].astype(np.float64)
                for nm in ("new", "all"):
                    mm = new if nm == "new" else np.ones(len(rU), bool)
                    if mm.any():
                        o2[nm]["dmt_top_minus_bot_med"] = float(np.median(tb[mm]))
                        o2[nm]["dmt_tb_neg"] = float((tb[mm] < -1e-6).mean())
            out["mt"][str(f)] = o2
        print(f"[{run}] f{f}: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if not isinstance(v, list)}, ensure_ascii=False), flush=True)
        if str(f) in out["mt"]:
            n_ = out["mt"][str(f)]["new"]
            print(f"[{run}] f{f} mt 처음 눌림: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in n_.items()}, ensure_ascii=False), flush=True)
            a_ = out["mt"][str(f)]["all"]
            print(f"[{run}] f{f} 모든 칸: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in a_.items() if k != "lay"}, ensure_ascii=False), flush=True)
    json.dump(out, open(os.path.join(O, f"startpt2_{run}.json"), "w"), ensure_ascii=False)
    print(f"[{run}] 끝", flush=True)
