"""절 BD 큰 dt 표 (bigdt2): <묶음 디렉토리>/<방식>K<k>_<궤적>/ -> 방식 x K 표 + 정답 실행기 원래 dt 확인 + 후퇴 오일러 발산 진단 (dq*).
방식: hh (우리 채택 판 HHT ρ 0.8, 절 AN.22 / 재질 씬은 절 CM3), be (같은 플래그, 적분기만 후퇴 오일러), ex (명시적 B-스플라인 bnv), mpm (정답 MPM, K0 = config 서브스텝).
칸: 평균 오차 % 또는 x<발산 프레임>, ficus 는 주기 비 / 진폭 비 / 오차 %; 우리 판은 프레임당 ms (결정적 판, 비측정).
사용: python exe/fem_an_bigdt2_table.py <dir>"""
import glob
import json
import math
import os
import re
import sys

R = sys.argv[1]
KS = [1, 2, 4, 8]
runs = {}
for d in glob.glob(os.path.join(R, "*_*")):
    m = re.match(r"(be|hh|ex|mpm)K(\d+)_(.+)$", os.path.basename(d))
    if m:
        runs[(m.group(1), int(m.group(2)), m.group(3))] = d


def done(d):
    p = os.path.join(d, "run_info.txt")
    return os.path.exists(p) and "rc=" in open(p).read()


def mpm_cell(d):
    p = os.path.join(d, "err.json")
    if not os.path.exists(p):
        if not done(d):
            return "미완"
        lg = os.path.join(d, "log.txt")
        ln = [q for q in open(lg, errors="ignore").read().splitlines() if q.strip() and "Warp CUDA error" not in q] if os.path.exists(lg) else []
        return "중단: " + (ln[-1][:80] if ln else "?")
    s = json.load(open(p))
    dv = s.get("div")
    if dv:
        return f"x{dv['frame']} ({dv['reason'].split(' (')[0]})"
    er = s["err"]
    for i, e in enumerate(er):
        if (not math.isfinite(e)) or e > 2.0:
            return f"x{i + 1}"
    return f"{100 * sum(er) / len(er):.2f}"


def cell(v, K, t):
    d = runs.get((v, K, t))
    if d is None:
        return "-"
    if v == "mpm":
        return mpm_cell(d)
    sp = os.path.join(d, "summary.json")
    if not os.path.exists(sp):
        if not done(d):
            return "미완"
        lg = os.path.join(d, "log.txt")
        ln = [q for q in open(lg, errors="ignore").read().splitlines() if q.strip()] if os.path.exists(lg) else []
        return "중단: " + (ln[-1][:80] if ln else "?")
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


NAMES = {"be": "우리 같은 플래그 적분기만 후퇴 오일러", "hh": "우리 채택 판 (HHT ρ 0.8)", "ex": "명시적 B-스플라인 (bnv)", "mpm": "정답 MPM 솔버 (공식 config, 서브스텝 = frame_dt/K)"}
print("오차 % = 롤아웃 평균 |x−x_gt|/EXT. x숫자 = 그 프레임에서 발산 (비유한, 오차 > 2, 정답 MPM 은 서브스텝 끝 입자가 격자 배열 밖). "
      "ms = 프레임당 시간 (결정적 판, 비측정). ficus = 주기 비 / 진폭 비 / 오차 %.\n")
print("우리 채택 판: lego·ficus = 절 AN.22 최종 (HHT ρ 0.8, 띠 고착 장벽 바닥, κ 0.1, --mv2 --mgbar 2 --lsfast --fk, εN 1e-2 · εP 1e-1, 걸음당 상한 lego 2x3 · ficus 2x20 -- K 와 무관), "
      "wolf·bread = 절 CM3 공통 묶음 (시작점 ŷ, --hspec --ncomp, 상한 wolf 2x4 · bread 2x1).\n")
print("## 정답 MPM 실행기 확인 (원래 서브스텝, 같은 실행기 exe/gt_sens.py)\n")
for t in ("L000", "ficus", "wolf", "bread"):
    d = runs.get(("mpm", 0, t))
    if d is None:
        continue
    p = os.path.join(d, "err.json")
    extra = ""
    if os.path.exists(p):
        s = json.load(open(p))
        extra = f" -- 서브스텝 {s['nsub']} x {s['dt']:.3e} s, {s['frames_done']} 프레임, {s['sec']:.0f} s"
    print(f"- {t}: {mpm_cell(d)}{extra}")
print()
for t, vs in (("lego L000", [("hh", "L000"), ("be", "L000"), ("ex", "L000"), ("mpm", "L000")]),
              ("lego 0005", [("hh", "0005"), ("be", "0005")]), ("lego 0013", [("hh", "0013"), ("be", "0013")]),
              ("lego 0016", [("hh", "0016"), ("be", "0016")]),
              ("ficus", [("hh", "ficus"), ("be", "ficus"), ("ex", "ficus"), ("mpm", "ficus")]),
              ("wolf (모래)", [("hh", "wolf"), ("mpm", "wolf")]), ("bread (plasticine)", [("hh", "bread"), ("mpm", "bread")])):
    print(f"### {t}\n")
    print("| 방식 | " + " | ".join(f"K {K} (dt = frame_dt/{K})" for K in KS) + " |")
    print("|---|" + "---|" * len(KS))
    for v, tt in vs:
        print(f"| {NAMES[v]} | " + " | ".join(cell(v, K, tt) for K in KS) + " |")
    print()

DQ = {"dq0": "재현 (절 AN.16 bej 설정 K 8)", "dq1": "허용치 0 (상한까지)", "dq2": "ω 재사용 끔 (--mg_wreuse --mg_lreuse 뺌)", "dq3": "블록 야코비 (MG 없음)",
      "dq4": "뉴턴 2", "dq5": "PCG 10", "dq6": "비결정 판"}
print("## 후퇴 오일러 K 8 발산 진단 (절 AN.16 bej 설정, --diag; 발산 프레임의 걸음별 첫 몇 반복)\n")
print("| 판 | 0005 | 0013 |\n|---|---|---|")
for k, nm in DQ.items():
    row = []
    for t in ("0005", "0013"):
        d = os.path.join(R, f"{k}_{t}")
        sp = os.path.join(d, "summary.json")
        if not os.path.exists(sp):
            row.append("미완" if not done(d) else "중단")
            continue
        s = json.load(open(sp))
        row.append(f"x{s['diverged']} (발산 전 {100 * s['err_mean_done']:.2f})" if s.get("diverged") else f"{100 * s['err_mean']:.2f}")
    print(f"| {nm} | " + " | ".join(row) + " |")
print()
for k in DQ:
    for t in ("0005", "0013"):
        fj = os.path.join(R, f"{k}_{t}", "fem.jsonl")
        if not os.path.exists(fj):
            continue
        L = [json.loads(q) for q in open(fj) if q.strip()]
        if not L or not (isinstance(L[-1].get("err"), float) and not math.isfinite(L[-1]["err"])):
            continue
        for rec in L[-2:]:
            dg = rec.get("diag", {})
            its = dg.get("it", [])[:4]
            short = [{q: (round(v, 4) if isinstance(v, float) else v) for q, v in it.items() if q in ("g2", "rz0", "rz1", "gd", "E0", "st")} for it in its]
            print(f"- {k}_{t} 프레임 {rec['t'] + 1} 오차 {rec['err']} 반복 {rec.get('it')} 첫 반복들 {short}")
