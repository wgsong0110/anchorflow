"""프레임별 맞춤 (exe/perframe.sh): 프레임당 반복 수 판정과 요약.

판정 (--n 1000): 두 궤적 모두 --min_frames 프레임 이상 끝났고 그 프레임의 E* (estar_T.json) 가 있을 때만 판정한다.
  프레임마다 갱신 중 증분 포텐셜 곡선(records.json 의 curve: 반복 k, 그 반복 갱신 전 E)에서
    g80  = mean E(k in [0.7N, 0.8N]) - E*,   g100 = mean E(k in [0.9N, N]) - E*
    정체율 = (g80 - g100) / g80   (마지막 20% 반복에서 E - E* 가 줄어든 비율; g80 <= 0 이면 0)
  정체율 중앙값(두 궤적 전 프레임) < 0.05 이면 "충분"(DECISION n1000), 아니면 DECISION n3000.
요약 (--summary): 반복 수별·궤적별 위치 오차 평균(기존 출력만 최적화·다섯 궤적 학습과 비교), E - E*, 출력 오차, 모델-출력만최적화 거리.
  다섯 궤적 모델의 프레임별 위치 오차는 lego_psi_spstick/roll_mm_T.pt (pred vs gt) 로 다시 재서 multi_err_T.json 에 쓴다.
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--u", required=True)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--min_frames", type=int, default=10)
ap.add_argument("--summary", action="store_true")
ap.add_argument("--thr", type=float, default=0.05)
a = ap.parse_args()
TR = ("L000", "0004")
REF = {"L000": (11.47, 12.21), "0004": (6.92, 25.57)}


def load(T, N):
    p = os.path.join(a.u, f"n{N}", T, "records.json")
    return json.load(open(p)) if os.path.exists(p) else []


def estar(T):
    p = os.path.join(a.u, f"estar_{T}.json")
    if not os.path.exists(p):
        return {}
    try:
        return {r["state"]: r["E_min"] for r in json.load(open(p))}
    except Exception:
        return {}


def stall(rec, es, N):
    cv = rec["curve"]
    e80 = [e for k, e, _ in cv if 0.7 * N <= k <= 0.8 * N]
    e100 = [e for k, e, _ in cv if 0.9 * N <= k <= N]
    if not e80 or not e100:
        return None
    g80, g100 = np.mean(e80) - es, np.mean(e100) - es
    return 0.0 if g80 <= 0 else float((g80 - g100) / g80)


if not a.summary:
    rows = []
    for T in TR:
        R, E = load(T, a.n), estar(T)
        ok = [r for r in R if r["t"] in E]
        print(f"[{T}] 끝난 프레임 {len(R)}, E* 있는 것 {len(ok)}")
        if len(ok) < a.min_frames:
            raise SystemExit(1)
        for r in ok:
            s = stall(r, E[r["t"]], a.n)
            ex = 100 * (r["Eu_after"] / E[r["t"]] - 1)
            rows.append((T, r["t"], s, ex, r["Eu_before"] - E[r["t"]], r["Eu_after"] - E[r["t"]]))
            print(f"  {T} 프레임 {r['t']:2d}: 갱신 전 E-E* {rows[-1][4]:.3e} -> 뒤 {rows[-1][5]:.3e} (초과 {ex:+.2f}%), 마지막 20% 정체율 {s}")
    ss = [r[2] for r in rows if r[2] is not None]
    med = float(np.median(ss))
    print(f"[판정 근거] 정체율 = 반복 {int(0.8 * a.n)}->{a.n} 에서 E-E* 가 줄어든 비율; 중앙값 {med:.4f} (문턱 {a.thr}), "
          f"초과% 중앙값 {np.median([r[3] for r in rows]):.2f}%, 프레임 {len(rows)}")
    print(f"DECISION {'n1000' if med < a.thr else 'n3000'} 정체율중앙 {med:.4f} 문턱 {a.thr} 프레임 {len(rows)}")
    raise SystemExit(0)

import torch  # noqa: E402

out = {}
for T in TR:
    mp = os.path.join(a.u, f"multi_err_{T}.json")
    if not os.path.exists(mp):
        D = torch.load(f"/home/dkta/work/lbours/lego_psi_spstick/roll_mm_{T}.pt", map_location="cpu", weights_only=False)
        ext = float(D["EXT"])
        me = [float((D["pred"][i].float() - D["gt"][i].float()).norm(dim=-1).mean()) / ext for i in range(D["pred"].shape[0])]
        json.dump(me, open(mp, "w"))
    me = json.load(open(mp))
    E = estar(T)
    for N in (1000, 3000):
        R = load(T, N)
        if not R:
            continue
        pe = [r["pos_err_model"] for r in R]
        ex = [100 * (r["Eu_after"] / E[r["t"]] - 1) for r in R if r["t"] in E]
        exb = [100 * (r["Eu_before"] / E[r["t"]] - 1) for r in R if r["t"] in E]
        st = [s for s in (stall(r, E[r["t"]], N) for r in R if r["t"] in E) if s is not None]
        out[f"{T}_n{N}"] = dict(frames=len(R), pos_err_mean=100 * float(np.mean(pe)),
                                ov_err_mean=100 * float(np.mean([r["ov_err"] for r in R])),
                                multi_err_mean=100 * float(np.mean(me[:len(R)])),
                                out_err_after_mean=100 * float(np.mean([r["out_err_after"] for r in R])),
                                out_err_before_mean=100 * float(np.mean([r["out_err_before"] for r in R])),
                                dist_model_ov_mean=100 * float(np.mean([r["dist_model_ov"] for r in R])),
                                excess_after_median=float(np.median(ex)) if ex else None,
                                excess_before_median=float(np.median(exb)) if exb else None,
                                stall_median=float(np.median(st)) if st else None)
        o = out[f"{T}_n{N}"]
        print(f"[{T} 반복 {N}] {o['frames']} 프레임  위치 오차 평균 {o['pos_err_mean']:.2f}% (출력만 최적화 {REF[T][0]}%, 다섯 궤적 {REF[T][1]}%; "
              f"다시 잰 값 {o['ov_err_mean']:.2f}%/{o['multi_err_mean']:.2f}%)  출력 오차 {o['out_err_before_mean']:.3f}% -> {o['out_err_after_mean']:.3f}%  "
              f"E-E* 초과 중앙 {o['excess_before_median']}% -> {o['excess_after_median']}%  모델-출력만최적화 거리 평균 {o['dist_model_ov_mean']:.2f}%  "
              f"정체율 중앙 {o['stall_median']}")
json.dump(out, open(os.path.join(a.u, "summary.json"), "w"), ensure_ascii=False, indent=1)
print("SUMMARY_DONE")
