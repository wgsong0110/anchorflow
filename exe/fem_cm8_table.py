"""절 CM8 표: python exe/fem_cm8_table.py <판정 묶음 디렉토리> <판 이름...>
행 = 씬 (lego L000·0000..0019, drop1, wolf, ficus, bread), 열 = 판 (<변형><c>, 하위 디렉토리 <판>_<씬>; ficus·bread 는 판 o 가 있으면 그것), 칸 = 평균 오차 % 또는 x발산 프레임 (발산 전 평균), 낙하 = 반발 / 튀어 오름 mm, ficus = 주기·진폭 비, wolf = f50 질량 중심 z."""
import json
import os
import re
import sys

B, NMS = sys.argv[1], sys.argv[2:]
LG = ["L000"] + ["%04d" % i for i in range(20)]
ROWS = LG + ["drop1", "wolf", "ficus", "bread"]


def find(nm, t):
    for v in (nm, "o"):
        d = os.path.join(B, v + "_" + t)
        if os.path.isdir(d):
            return d
    return None


def cell(nm, t):
    d = find(nm, t)
    if d is None:
        return "", None
    p = os.path.join(d, "summary.json")
    if not os.path.exists(p):
        ri = open(os.path.join(d, "run_info.txt")).read() if os.path.exists(os.path.join(d, "run_info.txt")) else ""
        return ("실패 (" + ri.split("rc=")[-1].split()[0] + ")") if "rc=" in ri else "도는 중", None
    s = json.load(open(p))
    dv = s.get("diverged")
    e = 100 * (s["err_mean_done"] if dv else s["err_mean"])
    c = ("**x%d** (%.2f)" % (dv, e)) if dv else "%.2f" % e
    if t == "drop1":
        q = os.path.join(d, "drop_ana.txt")
        if os.path.exists(q):
            m = re.search(r"\n[^\n]*?:[^\n]*반발 계수 ([0-9.\-]+), 튀어 오름 ([0-9.\-]+) mm", "\n" + "\n".join(l for l in open(q).read().splitlines() if not l.startswith("mpm:")))
            if m:
                c += " / 반발 %s · %s mm" % m.groups()
    if t == "ficus":
        q = os.path.join(d, "period.txt")
        if os.path.exists(q):
            m = re.findall(r"정답 대비 ([0-9.]+)\)[^\n]*?진폭 [0-9.]+ m \(비 ([0-9.]+)\)", open(q).read())
            if m:
                c += " (주기 %s 진폭 %s)" % m[-1]
    if t == "wolf":
        fr = [json.loads(l) for l in open(os.path.join(d, "fem.jsonl"))] if os.path.exists(os.path.join(d, "fem.jsonl")) else []
        hp = [(q["hp"], q["hg"]) for q in fr if q.get("hp") and q.get("hg")]
        if hp:
            c += " (끝 질량중심 z %.3f / 정답 %.3f, 최대 z %.3f / %.3f)" % (hp[-1][0][0], hp[-1][1][0], hp[-1][0][1], hp[-1][1][1])
    return c, (dv, e)


print("| 씬 | " + " | ".join(NMS) + " |")
print("|---|" + "---|" * len(NMS))
st = {nm: [] for nm in NMS}
for t in ROWS:
    cs = []
    for nm in NMS:
        c, v = cell(nm, t)
        cs.append(c)
        if t in LG and v is not None:
            st[nm].append(v)
    print("| %s | %s |" % (t, " | ".join(cs)))
print()
for nm in NMS:
    v = st[nm]
    ok = [e for dv, e in v if not dv]
    print("- %s: lego 끝난 %d / 21, 발산 %d (%s), <10 %% %d, 발산 없는 평균 %.2f %%" % (nm, len(v), sum(1 for dv, e in v if dv),
          " ".join(t for t in LG if (cell(nm, t)[1] or (0, 0))[0]), sum(1 for e in ok if e < 10), sum(ok) / max(len(ok), 1)))
