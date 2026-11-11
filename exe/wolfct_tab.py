"""절 CM108 wolfct 표 (로컬, numpy 만): 다시 풀기 판 cw.jsonl (fem_condwhy + fem_wolfct FD 기록) -> 마크다운.

사용: python exe/wolfct_tab.py <결과 루트> <판 이름 ...>
표 1 (FD 대조, 판마다 모든 상태·모든 뉴턴 반복 합): 상태 (탄성·꼬리·꼭짓점) 별 점 수 몫, 꺾인 점 몫, 점마다 상대 오차 ‖T − FD‖/‖FD‖ 평균 (10 에서 자름) -- T0 지금 판 (탄성) · T1 정확한 야코비안 · T2 새 판 (대칭 + PSD);
  꼭짓점은 FD = 0 이라 ‖T‖/‖T0‖ 로. 첫 반복만 따로 (괄호).
표 2 (수렴): εN 1e-2 도달 반복 (상태별), 끝 r 중앙, ρ = r_{k+1}/r_k 중앙, 경험 차수 p = log(r_{k+1}/r_k)/log(r_k/r_{k−1}) 중앙 (r 이 줄어드는 반복끼리), PCG 상대 잔차 중앙.
"""
import json
import math
import os
import sys

import numpy as np

EPS = 1e-2
CL = ("탄성", "꼬리", "꼭짓점")


def load(d):
    fn = os.path.join(d, "cw.jsonl")
    return [json.loads(l) for l in open(fn) if l.strip()] if os.path.exists(fn) else None


def g(x, f="%.2g"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f % x)


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def fdsum(L, first=False):
    A = np.zeros((3, 12))
    for q in L:
        for st in q["steps"]:
            for o in st["its"]:
                if "fd" in o and (not first or o["it"] == 0):
                    A += np.asarray(o["fd"])
    return A


def fdcell(A, c):
    a = A[c]
    if c == 2:
        return f"{g(math.sqrt(a[5] / a[5]) if a[5] > 0 else None)} · {g(math.sqrt(a[8] / a[5]) if a[5] > 0 else None)} · {g(math.sqrt(a[6] / a[5]) if a[5] > 0 else None)}"
    if a[0] <= 0:
        return "-"
    return " · ".join(g(a[k] / a[0]) for k in (9, 10, 11))                          # 점마다 상대 오차 평균 (10 에서 자름)


def conv(L):
    reach, rend, rho, pord, prel = [], [], [], [], []
    for q in L:
        rs = q["steps"][0]["rs"] if q["steps"] else []
        its = q["steps"][0]["its"] if q["steps"] else []
        reach.append(next((i for i, r in enumerate(rs) if r is not None and r < EPS), None))
        rend.append(rs[-1] if rs else None)
        for i in range(len(rs) - 1):
            if rs[i] > 0 and rs[i + 1] > 0:
                rho.append(rs[i + 1] / rs[i])
        for i in range(1, len(rs) - 1):
            a, b = rs[i] / rs[i - 1], rs[i + 1] / rs[i]
            if 0 < a < 1 and 0 < b < 1:
                pord.append(math.log(b) / math.log(a))
        prel += [o["Prel"] for o in its if o.get("Prel") is not None]
    nit = max((len(q["steps"][0]["rs"]) for q in L if q["steps"]), default=0)
    return reach, med(rend), med(rho), med(pord), med(prel), nit, [q["f"] for q in L]


def main():
    root, names = sys.argv[1], sys.argv[2:]
    print("FD 대조 (상대 오차 T0 지금 판 · T1 정확한 야코비안 · T2 새 판; 꼭짓점은 ‖T0‖·‖T1‖·‖T2‖ / ‖T0‖; 모든 반복 [첫 반복])\n")
    print("| 판 | 점 몫 탄성 · 꼬리 · 꼭짓점 · 꺾임 | 탄성 | 꼬리 | 꼭짓점 |")
    print("|---|---|---|---|---|")
    for n in names:
        L = load(os.path.join(root, n))
        if not L:
            continue
        A, A0 = fdsum(L), fdsum(L, True)
        if A[:, 0].sum() <= 0:
            continue
        tot = A[:, 0].sum() + A[:, 7].sum()
        sh = " · ".join(g(A[c, 0] / tot, "%.2f") for c in range(3)) + " · " + g(A[:, 7].sum() / tot, "%.3f")
        print(f"| {n} | {sh} | " + " | ".join(f"{fdcell(A, c)} [{fdcell(A0, c)}]" for c in range(3)) + " |")
    print("\n수렴 (εN 1e-2 도달 반복, 상태별; 끝 r = 마지막 반복 ‖g‖/F; ρ·p 중앙)\n")
    print("| 판 | 상태 프레임 | εN 도달 반복 | 끝 r 중앙 | ρ 중앙 | 차수 p 중앙 | PCG 상대 잔차 중앙 |")
    print("|---|---|---|---|---|---|---|")
    for n in names:
        L = load(os.path.join(root, n))
        if not L:
            print(f"| {n} | (없음) ||||||")
            continue
        reach, rend, rho, p, prel, nit, fr = conv(L)
        print(f"| {n} | {' '.join(str(x) for x in fr)} | {' · '.join('>%d' % (nit - 1) if x is None else str(x) for x in reach)} | {g(rend)} | {g(rho)} | {g(p)} | {g(prel)} |")


if __name__ == "__main__":
    main()
