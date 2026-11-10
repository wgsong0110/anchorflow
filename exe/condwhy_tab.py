"""절 CM98 condwhy 표 (로컬, numpy 만): 판 디렉토리들의 cw.jsonl (fem_condwhy 기록) -> 마크다운 표.

사용: python exe/condwhy_tab.py <결과 루트> <판 이름 ...>  (루트/이름/cw.jsonl)
열: 첫 반복 κ (프레임별·중앙·최대), 첫 반복 λmin·λmax 중앙, 끊김 (모든 뉴턴 반복의 PCG 풀이 중 pAp ≤ 0 · rz ≤ 0 으로 처음 끊긴 풀이 수 / 풀이 수),
첫 반복 끊김 프레임 수, εN 1e-2 도달 반복 (프레임별), 끝 r 중앙, PCG 상대 잔차 중앙.
"""
import json
import os
import sys

import numpy as np

EPS = 1e-2


def load(d):
    fn = os.path.join(d, "cw.jsonl")
    if not os.path.exists(fn):
        return None
    return [json.loads(l) for l in open(fn) if l.strip()]


def g(x, f="%.2g"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f % x)


def row(name, L):
    kap, lmin, lmax, reach, rend, prel = [], [], [], [], [], []
    nsol = npap = nrz = nboth = 0
    f0brk = []
    fr = []
    for q in L:
        fr.append(q["f"])
        its = q["steps"][0]["its"] if q["steps"] else []
        it0 = next((o for o in its if not o.get("stop")), None)
        kap.append(it0.get("kap") if it0 else None)
        lmin.append(it0.get("lmin") if it0 else None)
        lmax.append(it0.get("lmax") if it0 else None)
        b0 = it0.get("brk") if it0 else None
        f0brk.append(b0["kind"] if b0 else "")
        rs = q["steps"][0]["rs"] if q["steps"] else []
        k = next((i for i, r in enumerate(rs) if r is not None and r < EPS), None)
        reach.append(k)
        rend.append(rs[-1] if rs else None)
        for o in its:
            if o.get("stop") or "nrun" not in o:
                continue
            nsol += 1
            b = o.get("brk")
            if b:
                npap += b["kind"] == "pAp"; nrz += b["kind"] == "rz"; nboth += b["kind"] == "both"
            if o.get("Prel") is not None:
                prel.append(o["Prel"])
    kk = [x for x in kap if x is not None]
    nr = len(rs) if L else 0
    return dict(name=name, fr=fr, kap=kap, kmed=(float(np.median(kk)) if kk else None), kmax=(max(kk) if kk else None),
                lmin=(float(np.median([x for x in lmin if x is not None])) if any(x is not None for x in lmin) else None),
                lmax=(float(np.median([x for x in lmax if x is not None])) if any(x is not None for x in lmax) else None),
                reach=reach, rend=(float(np.median([x for x in rend if x is not None])) if any(x is not None for x in rend) else None),
                prel=(float(np.median(prel)) if prel else None), nsol=nsol, npap=npap, nrz=nrz, nboth=nboth, f0brk=f0brk, nit=nr)


def main():
    root, names = sys.argv[1], sys.argv[2:]
    print("| 판 | κ 첫 반복 (프레임별) | κ 중앙 · 최대 | λmin · λmax 중앙 | 첫 반복 끊김 | 끊긴 풀이 pAp≤0 · rz≤0 · 둘 다 / 풀이 | εN 도달 반복 | 끝 r 중앙 | PCG 상대 잔차 중앙 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n in names:
        L = load(os.path.join(root, n))
        if not L:
            print(f"| {n} | (없음) |||||||| ")
            continue
        r = row(n, L)
        print(f"| {n} | {' · '.join(g(x) for x in r['kap'])} | {g(r['kmed'])} · {g(r['kmax'])} | {g(r['lmin'])} · {g(r['lmax'])} | "
              f"{' '.join(x or '.' for x in r['f0brk'])} | {r['npap']} · {r['nrz']} · {r['nboth']} / {r['nsol']} | "
              f"{' · '.join('>%d' % (r['nit'] - 1) if x is None else str(x) for x in r['reach'])} | {g(r['rend'])} | {g(r['prel'])} |")


if __name__ == "__main__":
    main()
