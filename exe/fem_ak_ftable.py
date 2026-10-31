"""절 AK 최종 판 표 (R2 로컬 마운트): 판 V (기본 pqf3) 의 lego 21 + ficus -- 비측정 3 회, 깨끗한 단독 3 회 (ms), h_s 2 배·0.5 배. 기준 = 완주 + 오차 < 10 % + 단독 평균 ≤ 16.7 ms.
사용: python exe/fem_ak_ftable.py [판]"""
import json
import os
import sys

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
V = sys.argv[1] if len(sys.argv) > 1 else "pqf3"
TX = sys.argv[2] if len(sys.argv) > 2 else ""                                       # lego 태그 접미 (예 _np10: 바닥 인자만 바꾼 판; ficus 는 바닥이 없어 기본 판 그대로)
SC = ["L000"] + ["%04d" % k for k in range(20)] + ["ficus"]


def sj(sc, kind, r, sx=""):
    root, fk = ("fem_ak_ficus", "") if sc == "ficus" else ("fem_ak_lego", "_z" + TX)
    p = f"{L}/{root}/{sc}_{V}{fk}{sx}_{kind}{r}/summary.json"
    return json.load(open(p)) if os.path.exists(p) else None


def cell(s):
    if s is None:
        return None, "-"
    if s.get("diverged"):
        return None, f"발산 {s['diverged']}"
    return 100 * s["err_mean"], f"{100 * s['err_mean']:.2f}"


print(f"판 {V}{TX}")
print(f"| 장면 | 완주 (비측정 + 단독) | 오차 % (비측정 3 회 / 단독 3 회) | 단독 ms 평균 (회별) | fps | h_s 2 배 / 0.5 배 오차 | 기준 |")
print("|---|---|---|---|---|---|---|")
ok_n = 0
done_all = 0
for sc in SC:
    rr = [sj(sc, "r", r) for r in (1, 2, 3)]
    if TX and sc != "ficus":                                                         # 비측정 3 회 = _np10_r1 + _np10b_r1·r2
        rr = [sj(sc, "r", 1), sj(sc, "r", 1, "b"), sj(sc, "r", 2, "b")]
    tt = [sj(sc, "t", r) for r in (1, 2, 3)]
    es, txt, nd, nt = [], [], 0, 0
    for s in rr + tt:
        e, tx = cell(s)
        txt.append(tx)
        if s is not None:
            nt += 1
            if e is not None:
                es.append(e)
                nd += 1
    ms = [s["ms_mean"] for s in tt if s is not None and not s.get("diverged")]
    msm = sum(ms) / len(ms) if ms else float("nan")
    x2, x5 = cell(sj(sc, "r", 1, "_x2"))[1], cell(sj(sc, "r", 1, "_x0.5"))[1]
    em = sum(es) / len(es) if es else float("nan")
    good = nd == nt and nt > 0 and em < 10 and msm <= 16.7
    ok_n += good
    done_all += (nd == nt)
    print(f"| {sc} | {nd}/{nt} | {em:.2f} ({', '.join(txt[:3])} / {', '.join(txt[3:])}) | {msm:.2f} ({', '.join('%.2f' % q for q in ms)}) | {1e3 / msm:.1f} | {x2} / {x5} | {'만족' if good else '미달'} |")
print(f"\n기준 만족 {ok_n}/{len(SC)} (lego 21 + ficus), 모든 회 완주 {done_all}/{len(SC)}")
