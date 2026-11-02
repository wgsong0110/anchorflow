"""절 AN.err2 표 (칸이 "a / b" 면 1·2 회차): lego·낙하·ficus x 판 (R2 lbours/fem_anerr/r2 + 1 차 기준 판). 사용: python exe/fem_anerr_r2table.py <lbours 경로>"""
import glob
import json
import os
import re
import sys

B = sys.argv[1]
T = ["0001", "0003", "0016", "0017", "0018", "0005", "0013", "L000"]
V = [("명시적 bnv", "fem_ak_bselc/{t}_bnv"), ("명시적 gqw", "fem_ak_bselc/{t}_gqw"), ("실시간 판 중점 K1", "fem_an/{t}_rl1n2c20"),
     ("후퇴 K1", "fem_anerr/k1fac/g_be_K1_{t}"), ("후퇴 K2", "fem_anerr/r2/g_be_K2_{t}"), ("후퇴 K4", "fem_anerr/r2/g_be_K4_{t}"), ("후퇴 K8", "fem_anerr/r2/g_be_K8_{t}"),
     ("후퇴 K267", "fem_anerr/kbe/g_be_K267_{t}")]
for nm, tag in [("θ0.55", "t55"), ("θ0.6", "t60"), ("TR-BDF2", "trb")]:
    for k in (1, 2):
        V.append((f"{nm}+하한 K{k}", f"fem_anerr/r2/{tag}_K{k}_{{t}}"))
V += [("θ0.55+하한 적응 K1/4", "fem_anerr/r2/t55ad4_K1_{t}"), ("θ0.55+하한 적응 K1/8", "fem_anerr/r2/t55ad8_K1_{t}"), ("θ0.55+하한 고착 평균 K1", "fem_anerr/r2/t55av_K1_{t}"),
      ("θ0.55+하한 적응 K1/8 엄격 (0.5 m/s, 0.02 m)", "fem_anerr/r2b/t55ad8s_K1_{t}")]


def one(p):
    f = os.path.join(B, p, "summary.json")
    lg = os.path.join(B, p, "log.txt")
    if os.path.exists(f):
        s = json.load(open(f))
        if s.get("diverged"):
            return f"x{s['diverged']}", None, s
        if s.get("frames_done", 0) >= s.get("frames", 1):
            e = 100 * s["err_mean"]
            return f"{e:.2f}", e, s
    if os.path.exists(lg) and "OutOfMemoryError" in open(lg, errors="ignore").read():
        return "폭주", None, None
    return "", None, None


print("| 판 | " + " | ".join(T) + " | <10% | 낙하 튀어 오름 mm / 오차 % | ficus 주기 비 / 오차 % |")
print("|---|" + "---|" * (len(T) + 3))
for nm, pat in V:
    c, n, k = [], 0, 0
    rp = pat.replace("fem_anerr/r2/", "fem_anerr/r2b/").replace("fem_anerr/k1fac/", "fem_anerr/r2b/")
    for t in T:
        s, e, _ = one(pat.format(t=t))
        if rp != pat:                                                                  # 두 번째 회차 (원자 합 순서로 회차마다 갈림)
            s2, e2, _ = one(rp.format(t=t))
            if s2:
                s = f"{s} / {s2}"
        c.append(s)
        if s:
            n += 1
            k += int(e is not None and e < 10)
    dr = ""
    p = os.path.join(B, pat.format(t="drop1"))
    s, e, _ = one(pat.format(t="drop1"))
    if os.path.exists(os.path.join(p, "drop_ana.txt")):
        m = re.search(r"튀어 오름 ([-\d.]+) mm", open(os.path.join(p, "drop_ana.txt")).read().strip().split("\n")[-1])
        dr = f"{m.group(1) if m else '?'} / {s}"
    elif s:
        dr = s
    fc = ""
    p = os.path.join(B, pat.format(t="ficus"))
    s, e, _ = one(pat.format(t="ficus"))
    if os.path.exists(os.path.join(p, "period.txt")):
        m = re.search(r"정답 대비 ([\d.]+)", open(os.path.join(p, "period.txt")).read())
        fc = f"{m.group(1) if m else '?'} / {s}"
    elif s:
        fc = s
    print(f"| {nm} | " + " | ".join(c) + f" | {k}/{n} | {dr} | {fc} |")
