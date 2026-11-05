"""절 CM9 표 (--gq_mass 켜고 끔): python exe/fem_cm9_table.py <묶음 디렉토리 cm9> <시간표 디렉토리 cm9z>
수박 (옛 재질 실행기, 절 AS 전체 판): 판 w0* (끔)·w1* (켬) -- 발산 프레임, 평균·프레임별 오차, 거의 빈 노드 수 (0 < 질량 < 문턱), 질량·운동량 합 차, 발산 진단 (최대 변위 노드 칸 수·질량 비).
회귀 (새 실행기, 채택 판 (B) 장벽 + --y0eb, 결정적): 판 r0 (끔, 절 CM8 aa5a7ef 끝 해시와 같아야 함)·r1 (켬)."""
import json
import os
import re
import sys

B, BZ = sys.argv[1], sys.argv[2]
REF = {"L000": "1c76f894a3", "0016": "f89f6fb659", "drop1": "d1932fd19b", "wolf": "fa318a68b4", "ficus": "9f9f3ad2d3", "bread": "899ac753fc"}   # 절 CM8 b4·b5·o 끝 해시 (aa5a7ef)


def load(d):
    s = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else None
    fr = []
    if os.path.exists(os.path.join(d, "fem.jsonl")):
        for l in open(os.path.join(d, "fem.jsonl")):
            try:
                fr.append(json.loads(l))
            except Exception:
                pass
    return s, fr


def state(d):
    ri = open(os.path.join(d, "run_info.txt")).read() if os.path.exists(os.path.join(d, "run_info.txt")) else ""
    if os.path.exists(os.path.join(d, "card.txt")):
        ri += open(os.path.join(d, "card.txt")).read()
    return ("실패 (rc " + ri.split("rc=")[-1].split()[0] + ")") if "rc=" in ri else "도는 중"


def emean(s):
    dv = s.get("diverged")
    e = 100 * (s["err_mean_done"] if dv else s["err_mean"])
    return ("**x%d** (%.2f)" % (dv, e)) if dv else "%.3f" % e


def fl(v, f="%.2f"):
    return "-" if v is None else f % v


print("## 수박 (셀 1.5, 절 AS 전체 판, 옛 재질 실행기, 비결정 -- 같은 명령 두 번 a·b)")
print("| 판 | gq_mass | 끝난 프레임 | 평균 오차 % (x = 발산 프레임) | 거의 빈 노드 수 프레임 1..6 | 거의 빈 노드 평균 · 최대 (질량 있는 노드 평균) | 최대 변위 노드 칸 수 최대 (그 노드 질량 / 중앙값) | 질량 · 운동량 합 상대 차 최대 | 평균 ms (비측정) |")
print("|---|---|---|---|---|---|---|---|---|")
WR = sorted(n for n in os.listdir(B) if re.match(r"w[01][a-z]?_wm$", n)) if os.path.isdir(B) else []
SER = {}
for n in WR:
    d = os.path.join(B, n)
    s, fr = load(d)
    on = "켬" if n.startswith("w1") else "끔"
    if s is None and not fr:
        print("| %s | %s | %s |  |  |  |  |  |  |" % (n, on, state(d)))
        continue
    nl = [q.get("nlow") for q in fr]
    na = [q.get("nact") for q in fr if q.get("nact") is not None]
    nlv = [q for q in nl if q is not None]
    um = [(q.get("umax_cell") or 0.0, q.get("umax_mrel")) for q in fr if q.get("umax_cell") is not None]
    umx = max(um, key=lambda t: t[0]) if um else (None, None)
    gq = [q["gqchk"] for q in fr if q.get("gqchk")]
    gm = max((g[0] for g in gq), default=None); gv = max((g[1] for g in gq), default=None)
    SER[n] = [100 * q["err"] for q in fr]
    print("| %s | %s | %d | %s | %s | %s · %s (%s) | %s (%s) | %s · %s | %s |" % (
        n, on, len(fr), emean(s) if s else state(d), " ".join(str(q) for q in nl[:6]),
        fl(sum(nlv) / len(nlv) if nlv else None, "%.1f"), fl(max(nlv) if nlv else None, "%d"), fl(sum(na) / len(na) if na else None, "%.0f"),
        fl(umx[0], "%.2f"), fl(umx[1], "%.3g"), fl(gm, "%.1e"), fl(gv, "%.1e"), fl(s.get("ms_mean") if s else None, "%.1f")))
print()
print("프레임별 오차 % (프레임 1..10, 그 뒤 10 프레임마다):")
for n, e in SER.items():
    idx = list(range(min(10, len(e)))) + list(range(19, len(e), 10))
    print("- %s: %s" % (n, " ".join("f%d %.2f" % (i + 1, e[i]) for i in idx)))
print()
print("## 수박 단독 시간 (카드 0, gpu0.lock, 판마다 카드 0 의 다른 계산 프로세스 0 일 때 시작; 비결정, 재생 GPU 시간, 첫 2 프레임 뺌)")
print("| 판 | gq_mass | 끝난 프레임 | 평균 · 최대 ms | 평균 오차 % | 실행 중 카드 0 다른 프로세스 |")
print("|---|---|---|---|---|---|")
for n in sorted(os.listdir(BZ)) if os.path.isdir(BZ) else []:
    d = os.path.join(BZ, n)
    if not os.path.isdir(d):
        continue
    s, fr = load(d)
    ct = open(os.path.join(d, "card.txt")).read() if os.path.exists(os.path.join(d, "card.txt")) else ""
    m = re.search(r"다른 계산 프로세스 (\d+)", ct)
    if s is None:
        print("| %s | %s | %d | %s |  |  |" % (n, "켬" if n.startswith("zw1") else "끔", len(fr), state(d)))
        continue
    print("| %s | %s | %d | %s · %s | %s | %s |" % (n, "켬" if n.startswith("zw1") else "끔", len(fr), fl(s.get("ms_mean"), "%.1f"), fl(s.get("ms_max"), "%.1f") if s.get("ms_max") else "-", emean(s), m.group(1) if m else "?"))
print()
print("## 회귀 (채택 판 (B): 장벽 + --y0eb, lego·낙하 2x4, wolf 2x5, ficus 2x24, bread 2x2; 결정적, 비측정)")
print("| 씬 | 끔 r0: 오차 % · 끝 해시 (절 CM8 해시와 같음?) | 켬 r1: 오차 % · 끝 해시 | 켬 거의 빈 노드 평균 · 최대 (끔) | 켬 질량 · 운동량 · 가속도 몫 합 상대 차 최대 | 지표 끔 / 켬 |")
print("|---|---|---|---|---|---|")
for t in ["L000", "0016", "drop1", "ficus", "wolf", "bread"]:
    row, ext = [], []
    nlo = {}
    gqs = "-"
    for v in ("r0", "r1"):
        d = os.path.join(B, v + "_" + t)
        if not os.path.isdir(d):
            row.append(""); ext.append(""); continue
        s, fr = load(d)
        if s is None:
            row.append(state(d)); ext.append(""); continue
        h = s.get("x_sha1", "")
        same = (" (같음)" if h == REF[t] else " (**다름**, CM8 %s)" % REF[t]) if v == "r0" else ""
        row.append("%s · %s%s" % (emean(s), h, same))
        nl = [q.get("nlow") for q in fr if q.get("nlow") is not None]
        nlo[v] = (sum(nl) / len(nl), max(nl)) if nl else None
        if v == "r1":
            gq = [q["gqchk"] for q in fr if q.get("gqchk")]
            if gq:
                gqs = " · ".join(fl(max((g[i] for g in gq if g[i] is not None), default=None), "%.1e") for i in range(3))
        x = ""
        if t == "drop1" and os.path.exists(os.path.join(d, "drop_ana.txt")):
            m = re.search(r"반발 계수 ([0-9.\-]+), 튀어 오름 ([0-9.\-]+) mm", "\n".join(l for l in open(os.path.join(d, "drop_ana.txt")).read().splitlines() if not l.startswith("mpm:")))
            x = "반발 %s · %s mm" % m.groups() if m else ""
        if t == "ficus" and os.path.exists(os.path.join(d, "period.txt")):
            m = re.findall(r"정답 대비 ([0-9.]+)\)[^\n]*?진폭 [0-9.]+ m \(비 ([0-9.]+)\)", open(os.path.join(d, "period.txt")).read())
            x = "주기 %s 진폭 %s" % m[-1] if m else ""
        if t == "wolf":
            hp = [(q["hp"], q["hg"]) for q in fr if q.get("hp") and q.get("hg")]
            x = "끝 질량중심 z %.3f / 정답 %.3f" % (hp[-1][0][0], hp[-1][1][0]) if hp else ""
        ext.append(x)
    nn = "%s (%s)" % (("%.1f · %d" % nlo["r1"]) if nlo.get("r1") else "-", ("%.1f · %d" % nlo["r0"]) if nlo.get("r0") else "-")
    print("| %s | %s | %s | %s | %s | %s / %s |" % (t, row[0], row[1], nn, gqs, ext[0], ext[1]))
print()
print("(거의 빈 노드 = 0 < 노드 질량 < 문턱 (--mlow 1e-3 x 첫 프레임 입자 P2G 노드 질량 중앙값, 정의 그대로); 탄성 씬 (lego·낙하·ficus) 은 문턱이 없어 안 셈.)")
