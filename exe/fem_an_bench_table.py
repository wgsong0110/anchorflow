"""절 AN 속도 표: <루트>/<판>/fem.jsonl (sec_ms) + smi.csv -> 판마다 프레임당 구간 시간 (첫 2 프레임 뺀 평균), 뉴턴·PCG 반복, 헤시안 곱·MG 사이클 한 번 시간, 카드 사용률."""
import glob
import json
import os
import sys

import numpy as np

R = sys.argv[1]
info = open(os.path.join(R, "info.txt")).read().split() if os.path.exists(os.path.join(R, "info.txt")) else ["?", "?"]
G = int(info[1]) if len(info) > 1 and info[1].isdigit() else None
print(" ".join(info))
KEYS = ["P2G", "바닥", "요소", "대각", "MG구성", "PCG", "선탐색", "G2P", "검사"]
NS = "ns" in os.path.basename(R.rstrip("/"))
print(f"카드 {G} (비측정이면 다른 실행과 함께; 사용률 = 실행 동안 1 초 표본 평균)")
print("| 판 | 프레임 ms (동기화 포함) | " + " | ".join(f"{k} ms" for k in KEYS) + " | 동기화·할당·기타 ms | 걸음당 뉴턴 | 뉴턴당 PCG | 헤시안 곱 한 번 ms (프레임당 횟수) | 거친 층 곱 ms/프레임 | MG 사이클 한 번 ms | 오차 % (끝 프레임) | 평균 오차 % | 걸음 끝 최대 |g|/|g0| (프레임 평균) | 이 카드 사용률 % (다른 카드 평균) |")
print("|---|" + "---|" * (len(KEYS) + 11))
for d in sorted(glob.glob(os.path.join(R, "*", "fem.jsonl"))):
    V = os.path.basename(os.path.dirname(d))
    rs = [json.loads(l) for l in open(d)]
    rr = rs[2:] if len(rs) > 3 else rs
    if not rr:
        continue
    fm = 1e3 * np.mean([r["sec"] for r in rr])
    sec = {k: np.mean([r.get("sec_ms", {}).get(k, [0, 0])[0] for r in rr]) for k in KEYS}
    oth = fm - sum(sec.values())
    nts = [n for r in rr for n in r.get("nts", [])]
    cgs = [c for r in rr for st in r.get("cgs", []) for c in st]
    mvt = sum(r.get("sec_ms", {}).get("헤시안곱", [0, 0])[0] for r in rr); mvc = sum(r.get("sec_ms", {}).get("헤시안곱", [0, 0])[1] for r in rr)
    mvcz = np.mean([r.get("sec_ms", {}).get("헤시안곱_거친층", [0, 0])[0] for r in rr])
    km = {}                                                                            # 동기화 없는 판: 커널 이벤트 시간 (ms/프레임)
    for r in rr:
        for k_, (t_, c_) in r.get("kernel_ms", {}).items():
            km[k_] = km.get(k_, 0.0) + t_ / len(rr)
    mgt = sum(r.get("sec_ms", {}).get("MG사이클", [0, 0])[0] for r in rr); mgc = sum(r.get("sec_ms", {}).get("MG사이클", [0, 0])[1] for r in rr)
    ut = uo = float("nan")
    cp_ = os.path.join(os.path.dirname(d), "card.txt")
    if os.path.exists(cp_):
        G = int(open(cp_).read().split()[1])
    sp = os.path.join(os.path.dirname(d), "smi.csv")
    if os.path.exists(sp) and G is not None:
        a = np.array([[float(q) for q in l.split(",")[:2]] for l in open(sp) if l.strip() and l.split(",")[0].strip().isdigit()])
        if a.size:
            ut = a[a[:, 0] == G, 1].mean()
            uo = a[(a[:, 0] != G) & (a[:, 0] > 0), 1].mean()
    print(f"| {V} | {fm:.0f} | " + " | ".join(f"{sec[k]:.1f}" for k in KEYS) + f" | {oth:.1f} | {np.mean(nts):.1f} | {np.mean(cgs) if cgs else 0:.1f} | "
          f"{mvt / max(mvc, 1):.3f} ({mvc / len(rr):.0f}) | {mvcz:.1f} | {mgt / max(mgc, 1):.2f} | {100 * rs[-1]['err']:.3f} | {100 * np.mean([r['err'] for r in rs]):.3f} | {np.mean([r.get('gr', float('nan')) for r in rs]):.2e} | {ut:.0f} ({uo:.0f}) |" + (" 커널 ms/프레임 " + ", ".join(f"{k_} {t_:.2f}" for k_, t_ in sorted(km.items())) if km else ""))
