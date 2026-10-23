#!/usr/bin/env python3
"""출력 전처리 짧은 시험(2 프레임) 요약 (사용자 10-23): 기준판(pfip_noin L000) 같은 프레임과 같은 반복에서 E, E/E* 비교.
  python3 exe/precond_smoke_sum.py <기준판 records.json> 이름=records.json ...
"""
import json
import sys

base = json.load(open(sys.argv[1]))
IT = [0, 100, 250, 500, 1000, 1500, 2000]


def at(c, k):
    return min(c, key=lambda p: abs(p[0] - k))[1]


def first_below(c, e):
    for k, v in c:
        if v <= e:
            return k
    return None


out = []
for kv in sys.argv[2:]:
    nm, p = kv.split("=", 1)
    r = json.load(open(p))
    out.append(f"== {nm}")
    for q in r:
        b = base[q["t"]]
        es, bs = q["E_star"], b["E_star"]
        row = " ".join(f"{k}:{at(q['obj_curve'], k) / es:.3f}({at(b['obj_curve'], k) / bs:.3f})" for k in IT)
        fb = first_below(q["obj_curve"], b["obj_last"])
        out.append(f"  t={q['t']} 반복별 E/E* 이번(기준판): {row}")
        out.append(f"       끝 E/E* {q['ratio_E_Estar']:.3f} (기준판 {b['ratio_E_Estar']:.3f}), 기준판 끝 E 에 닿은 반복 {fb}, "
                   f"위치 오차 {100 * q['pos_err_model']:.3f}% (기준판 {100 * b['pos_err_model']:.3f}%), det 최소 {q['F_stats']['det_min']:.3f}, "
                   f"맞춤 {q['sec_fit']:.0f} s (기준판 {b['sec_fit']:.0f} s)")
        pc = {k: v for k, v in q.get("precond", {}).items() if k not in ("t", "mode")}
        out.append("       전처리 " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in pc.items()))
print("\n".join(out))
