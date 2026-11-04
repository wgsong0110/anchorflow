"""절 CM 표: python exe/fem_cm_table.py judge <묶음 디렉토리> | time <묶음 디렉토리>
judge: 판 x 씬 (오차 % / 발산 프레임 x / ficus 주기 비) -- 하위 디렉토리 <판>_<씬>; time: 이름별 평균·최대 (첫 2 프레임 뺀 재생 GPU 시간), 오차, 프레임당 뉴턴·PCG, 카드 0 다른 프로세스."""
import json
import os
import re
import sys

import numpy as np

SC = ["L000", "0016", "0019", "ficus", "wolf", "bread"] + ["%04d" % i for i in range(20) if i not in (16, 19)]


def load(d):
    s = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else None
    fr = [json.loads(l) for l in open(os.path.join(d, "fem.jsonl"))] if os.path.exists(os.path.join(d, "fem.jsonl")) else []
    return s, fr


def per(d):
    p = os.path.join(d, "period.txt")
    if not os.path.exists(p):
        return ""
    t = open(p).read()
    m = re.findall(r"정답 대비 ([0-9.]+)\)[^\n]*?진폭 [0-9.]+ m \(비 ([0-9.]+)\)", t)
    return ("주기 %s 진폭 %s" % m[-1]) if m else t.strip().splitlines()[-1][:120]


def ms_of(fr):
    ms = [q.get("ms") for q in fr[2:] if q.get("ms") is not None and np.isfinite(q.get("ms"))]
    return (float(np.mean(ms)), float(np.max(ms))) if ms else (None, None)


def it_of(fr):
    it = [q["it"] for q in fr if q.get("it")]
    if not it:
        return ""
    return "N %.1f P %.1f (최대 %d)" % (np.mean([q["nN"] for q in it]), np.mean([q["nP"] for q in it]), max(q["nP"] for q in it))


def cell(d):
    s, fr = load(d)
    if s is None:
        return "?" if not os.path.exists(os.path.join(d, "log.txt")) else "오류"
    if s.get("diverged"):
        return "x%d (%.2f)" % (s["diverged"], 100 * s["err_mean_done"])
    c = "%.2f" % (100 * s["err_mean"])
    p = per(d)
    if p:
        c = p + " / " + c
    return c


def judge(R):
    runs = sorted(os.listdir(R))
    vs = sorted({r.rsplit("_", 1)[0] for r in runs if "_" in r and os.path.isdir(os.path.join(R, r))})
    sc = [t for t in SC if any(r.endswith("_" + t) for r in runs)]
    print("| 판 | " + " | ".join(sc) + " | lego <10 % |")
    print("|---" * (len(sc) + 2) + "|")
    for v in vs:
        cs, ok, n = [], 0, 0
        for t in sc:
            d = os.path.join(R, v + "_" + t)
            cs.append(cell(d) if os.path.isdir(d) else "")
            if os.path.isdir(d) and t not in ("ficus", "wolf", "bread"):
                s, _ = load(d)
                if s is not None:
                    n += 1; ok += int(not s.get("diverged") and s["err_mean"] is not None and 100 * s["err_mean"] < 10)
        print("| %s | %s | %d/%d |" % (v, " | ".join(cs), ok, n))
    print()
    print("프레임당 반복 (뉴턴 합 N · PCG 합 P 평균, 최대 P) -- 판정 판:")
    for v in vs:
        xs = []
        for t in sc:
            d = os.path.join(R, v + "_" + t)
            if os.path.isdir(d):
                xs.append("%s %s" % (t, it_of(load(d)[1])))
        print("- %s: %s" % (v, "; ".join(xs)))


def timing(R):
    print("| 이름 | 평균 ms | 최대 ms | 오차 % | 반복 | 해시 | 카드 0 다른 프로세스 |")
    print("|---|---|---|---|---|---|---|")
    for v in sorted(os.listdir(R)):
        d = os.path.join(R, v)
        if not os.path.isdir(d):
            continue
        s, fr = load(d)
        mm, mx = ms_of(fr)
        ct = open(os.path.join(d, "card.txt")).read() if os.path.exists(os.path.join(d, "card.txt")) else ""
        m = re.search(r"다른 계산 프로세스 (\d+)", ct)
        e = "" if s is None else (("x%d " % s["diverged"]) if s.get("diverged") else "") + "%.2f" % (100 * s["err_mean_done"])
        print("| %s | %s | %s | %s | %s | %s | %s |" % (v, "" if mm is None else "%.2f" % mm, "" if mx is None else "%.2f" % mx, e, it_of(fr), (s or {}).get("x_sha1", "")[:10], m.group(1) if m else "?"))


if __name__ == "__main__":
    (judge if sys.argv[1] == "judge" else timing)(sys.argv[2])
