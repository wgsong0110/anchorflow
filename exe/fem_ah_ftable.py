"""절 AH 최종 판 표: 궤적별 완주 (비측정 r1~3 + 단독 t1~3), 오차 (완주 평균; 회별), 단독 ms·fps, 기준 (≤ 16.7 ms · < 10 %), 절 AF 공통 설정 오차, N 0.5 배·2 배 오차.
사용: python exe/fem_ah_ftable.py <lbours 루트> <판 (스윕 이름, 예 gkr_d025_vc75)> <판 (측정 이름, 예 fin)>"""
import json
import os
import sys

root, V, FN = sys.argv[1], sys.argv[2], sys.argv[3]


def ld(sub, tag):
    f = os.path.join(root, sub, tag, "summary.json")
    return json.load(open(f)) if os.path.exists(f) else None


def oth(sub, tag):
    v = []
    for f in ("gpu_at_start.txt", "gpu_at_end.txt"):
        p = os.path.join(root, sub, tag, f)
        if os.path.exists(p):
            v += [int(q.split(",")[1].split()[0]) for q in open(p) if q.split(",")[0].strip() != "0"]
    return max(v) if v else None


e_ = lambda s: "-" if s is None else (f"발산 {s['diverged']}" if s["diverged"] else f"{100 * s['err_mean']:.2f}")
print("| 장면 | 완주 (비측정 + 단독) | 오차 (완주 평균; 회별 비측정 / 단독) | 단독 ms (평균; 회별) | fps | 기준 | 절 AF 공통 설정 오차 | N 0.5 배 (lego 133·ficus 320) / 2 배 (534·1280) 오차 | 측정 중 카드 1~6 최대 MiB |")
print("|---|---|---|---|---|---|---|---|---|")
ng = nt = 0
for t in [f"{i:04d}" for i in range(20)] + ["L000", "F"]:
    sub = "fem_ah_ficus" if t == "F" else "fem_ah_lego"
    A = [ld(sub, f"ah_{V}_{t}_r{r}") for r in (1, 2, 3)]
    B = [ld(sub, f"ah_{FN}_{t}_t{r}") for r in (1, 2, 3)]
    X = [ld(sub, f"ah_{V}_x{k}_{t}_r1") for k in ("2", "0.5")]                      # x2 = h_s 2 배 = N 0.5 배 (lego 133), x0.5 = N 2 배 (534)
    S = [s for s in A + B if s]
    fin = [s for s in S if not s["diverged"]]
    err = sum(s["err_mean"] for s in fin) / len(fin) if fin else None
    BT = [s for s in B if s]
    ms = sum(s["ms_mean"] for s in BT) / len(BT) if BT else None
    good = err is not None and len(fin) == len(S) and len(S) == 6 and ms is not None and ms <= 1000 / 60 and err < 0.10
    nt += 1; ng += good
    if t == "F":
        af = "0.59 (절 AD 판, 같은 N)"
    elif t == "L000":
        af = e_(ld("fem_ah_lego", "ah_b0_L000_r1")) + " (N267)"
    else:
        fa = os.path.join(root, "fem_af_lego", f"af_{t}_c267_r1", "summary.json")
        af = e_(json.load(open(fa))) if os.path.exists(fa) else "-"
    om = max((oth(sub, f"ah_{FN}_{t}_t{r}") or 0) for r in (1, 2, 3))
    nm = {"F": "ficus F_pg_v0", "L000": "lego L000"}.get(t, f"lego {t}")
    print(f"| {nm} | {len(fin)}/{len(S)} ({sum(1 for s in A if s and not s['diverged'])}+{sum(1 for s in BT if not s['diverged'])}) | "
          + (f"{100 * err:.2f} % " if err is not None else "- ") + "(" + ", ".join(e_(s) for s in A) + " / " + ", ".join(e_(s) for s in B) + ")"
          + (f" | {ms:.2f} (" + ", ".join(f"{s['ms_mean']:.2f}" for s in BT) + f") | {1000 / ms:.1f}" if ms else " | - | -")
          + f" | {'만족' if good else '미달'} | {af} | {e_(X[0])} / {e_(X[1])} | {om} |")
print(f"\n기준 만족 {ng}/{nt} (lego 홀드아웃 20 + L000 + ficus)")
