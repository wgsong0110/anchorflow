"""절 AN.16 큰 dt 표: <묶음 디렉토리>/<방식>K<k>_<궤적>/ -> 방식 x K 표.
방식: be (후퇴 오일러, 우리), hh (HHT ρ 0.7, 우리), ex (명시적 B-스플라인 기준선 bnv), mpm (정답 MPM 솔버).
칸: 평균 오차 % 또는 x<발산 프레임> (우리 lego/ficus), ficus 는 주기 비 / 진폭 비 / 오차 %, 우리 판은 프레임당 ms (비측정) 도.
사용: python exe/fem_an_bigdt_table.py <dir>"""
import glob
import json
import math
import os
import re
import sys

R = sys.argv[1]
KS = [1, 2, 4, 8]
runs = {}
for d in glob.glob(os.path.join(R, "*K*_*")):
    m = re.match(r"(be|hh|ex|mpm)K(\d+)_(.+)$", os.path.basename(d))
    if m:
        runs[(m.group(1), int(m.group(2)), m.group(3))] = d


def mpm_div(d):
    p = os.path.join(d, "err.json")
    if not os.path.exists(p):
        lg = os.path.join(d, "log.txt")
        tail = open(lg, errors="ignore").read().strip().splitlines()[-1][:80] if os.path.exists(lg) else ""
        return None, None, "중단: " + tail
    s = json.load(open(p))
    er = s["err"]
    for i, e in enumerate(er):
        if (not math.isfinite(e)) or e > 2.0:
            return i + 1, None, ""
    return None, sum(er) / len(er), ""


def cell(v, K, t):
    d = runs.get((v, K, t))
    if d is None:
        return "-"
    if v == "mpm":
        dv, e, msg = mpm_div(d)
        if msg:
            return msg
        return f"x{dv}" if dv else f"{100 * e:.2f}"
    sp = os.path.join(d, "summary.json")
    if not os.path.exists(sp):
        return "미완" 
    s = json.load(open(sp))
    e = s.get("err_mean")
    o = f"x{s.get('diverged')}" if e is None else f"{100 * e:.2f}"
    if t == "ficus" and e is not None:
        pp = os.path.join(d, "period.txt")
        m = re.search(r"정답 대비 ([\d.]+)\), 최대 진폭 [\d.]+ m \(비 ([\d.]+)\)", open(pp).read()) if os.path.exists(pp) else None
        o = (f"{m.group(1)} / {m.group(2)} / {o}" if m else "주기? / " + o)
    if v in ("be", "hh") and s.get("ms_mean"):
        o += f" ({s['ms_mean']:.1f} ms)"
    return o


NAMES = {"be": "우리 후퇴 오일러 (bej 설정)", "hh": "우리 HHT ρ 0.7 (hhj 설정)", "ex": "명시적 B-스플라인 (bnv)", "mpm": "정답 MPM 솔버"}
print("오차 % = 롤아웃 평균 |x−x_gt|/EXT. x숫자 = 그 프레임에서 발산 (비유한 또는 오차 > 2). ms = 프레임당 시간 (결정적 판, 비측정). 주기 비 / 진폭 비 / 오차 %.\n")
for t, vs in (("lego L000", [("be", "L000"), ("hh", "L000"), ("ex", "L000"), ("mpm", "L000")]),
              ("lego 0005", [("be", "0005"), ("hh", "0005")]), ("lego 0013", [("be", "0013"), ("hh", "0013")]),
              ("lego 0016", [("be", "0016"), ("hh", "0016")]),
              ("ficus", [("be", "ficus"), ("hh", "ficus"), ("ex", "ficus"), ("mpm", "ficus")])):
    print(f"### {t}\n")
    print("| 방식 | " + " | ".join(f"K {K} (dt = frame_dt/{K})" for K in KS) + " |")
    print("|---|" + "---|" * len(KS))
    for v, tt in vs:
        print(f"| {NAMES[v]} | " + " | ".join(cell(v, K, tt) for K in KS) + " |")
    print()
