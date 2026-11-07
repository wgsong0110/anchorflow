"""절 CM27 회귀 표: python exe/wolffix_regtab.py <출력 디렉토리> <판 이름>... -> 마크다운 (표준 출력). torch 없이 summary.json·period.txt 만."""
import json
import os
import sys

O = sys.argv[1]
VNS = sys.argv[2:]
SC = [("L", "레고"), ("F", "ficus"), ("B", "bread"), ("M", "수박 (절 CM24 (2) 판 기준)")]


def sm(n):
    p = os.path.join(O, n, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def er(s):
    if not s:
        return "없음"
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    d = s.get("diverged")
    return (f"{100 * e:.2f}" if e is not None else "-") + (f" (발산 f{d})" if d else "")


def ms(s):
    return (f"{s['ms_mean']:.2f}/{s.get('ms_max', float('nan')):.2f}" + (f" (발산 f{s['diverged']})" if s.get("diverged") else "")) if s and s.get("ms_mean") is not None else "-"


def per(n):
    p = os.path.join(O, n, "period.txt")
    if not os.path.exists(p):
        return ""
    t = [l.strip() for l in open(p) if l.strip()]
    return f" [{t[-1][:80]}]" if t else ""


for vn in VNS:
    print(f"## 판 {vn}\n")
    print("| 씬 | off 오차 % | 판 오차 % | off 해시 · k1new | 단독 ms off a · b | 단독 ms 판 a · b |")
    print("|---|---|---|---|---|---|")
    for s, nm in SC:
        if s == "M":
            off = " · ".join(er(sm(f"r_M_off_{r}")) for r in "ab"); on = " · ".join(er(sm(f"r_M_{vn}_{r}")) for r in "ab"); hh = "-"
        else:
            so, sv = sm(f"r_{s}_off"), sm(f"r_{s}_{vn}")
            off, on = er(so) + per(f"r_{s}_off"), er(sv) + per(f"r_{s}_{vn}")
            k = os.path.join(O, f"r_{s}_off", "k1new_summary.json")
            hh = f"{(so or {}).get('x_sha1')} · {json.load(open(k)).get('x_sha1') if os.path.exists(k) else '-'}"
        zo = " · ".join(ms(sm(f"z_{s}_off{vn}_{r}")) for r in "ab"); zv = " · ".join(ms(sm(f"z_{s}_{vn}_{r}")) for r in "ab")
        print(f"| {nm} | {off} | {on} | {hh} | {zo} | {zv} |")
    print()
