"""절 CM17 표 (astick): python exe/fem_astick_table.py <묶음 디렉토리 astick>
판: a = (A) 가우스 점 고착 원래 (절 CM8 a6 명령, 새 코드 플래그 없음), u1 = (i) 법선 한쪽 구속 + 접선 고착 (--fl_up 1), u2 = (ii) 법선 한쪽 구속 + 접선 자유 (--fl_up 2).
씬 lego L000·0000·0013·0018 (결정적). 디렉토리 <판>_<씬> (summary.json·fem.jsonl: pen·hg·hp·pd·fup).
튀어 오름 = 질량 중심 높이 (바닥 법선 z, 질량 가중, 실행기 --hgt) 의 첫 착지 (처음으로 높이가 내려가다 다시 오르는 프레임 = 첫 극소) 뒤 최대 상승 (그 극소 대비). 정답은 같은 fem.jsonl 의 hg.
바닥 아래 = 바닥 평면 부호 거리 < 0 인 입자의 최대 깊이 (실행기 --pen). 법선 구속 노드 = 걸음 끝 법선 고착 노드 수 (fem.jsonl fup.act, 질량 > 0 노드)."""
import json
import os
import sys

import numpy as np

B = sys.argv[1]
VS = ("a", "u1", "u2")
VN = {"a": "(A) 원래", "u1": "(i) 법선 한쪽 + 접선 고착", "u2": "(ii) 법선 한쪽 + 접선 자유"}
SCN = ("L000", "0000", "0013", "0018")
A6 = {"L000": "x23 (GPU 메모리 부족, 발산 전 1.46; 절 CM8 cm8av v_L000)", "0000": "6.36", "0013": "4.53", "0018": "10.80"}
A6H = {"0000": "f444ba6042", "0013": "48a7997a03", "0018": "623d5873fd"}


def sm(d):
    p = os.path.join(d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def jl(d):
    p = os.path.join(d, "fem.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


P = lambda v, t: os.path.join(B, f"{v}_{t}")


def state(d):
    s = sm(d)
    if s is None:
        ri = os.path.join(d, "run_info.txt")
        t = open(ri).read() if os.path.exists(ri) else ""
        return ("실패 (" + t.split("rc=")[-1].split()[0] + ")") if "rc=" in t else ("없음" if not t else "도는 중")
    dv = s.get("diverged")
    e = 100 * (s["err_mean_done"] if dv else s["err_mean"])
    why = " GPU 메모리 부족" if s.get("oom") else ""
    return ("**x%d**%s (발산 전 %.2f)" % (dv, why, e)) if dv else "%.2f" % e


def bounce(z):
    z = [q for q in z if q is not None]
    if len(z) < 3:
        return None
    for i in range(1, len(z) - 1):
        if z[i] < z[i - 1] and z[i + 1] >= z[i]:
            rest = z[i:]
            return i + 1, 1000 * (max(rest) - z[i]), int(np.argmax(rest)) + i + 1
    return None


def fmt_b(b):
    return "-" if b is None else "%.1f mm (착지 프레임 %d, 최고 %d)" % (b[1], b[0], b[2])


print("# 절 CM17 astick -- (A) 가우스 점 고착 바닥의 법선 성분을 한쪽 구속으로 (위 허용), lego 결정적")
print()
print("## 1. 판정 (칸 = 평균 오차 % 또는 x발산 프레임; 끝 상태 해시)")
print()
print("| 씬 | 원래 (A) 기존 값 (절 CM8 a6) | " + " | ".join(VN[v] for v in VS) + " | 해시 " + " / ".join(VS) + " | a6 해시 |")
print("|---|---|" + "---|" * (len(VS) + 2))
for t in SCN:
    H = [(((sm(P(v, t)) or {}).get("x_sha1") or "-") if not (sm(P(v, t)) or {}).get("diverged") else "(발산)") for v in VS]   # 발산 판의 끝 상태 해시는 뜻 없음
    print(f"| {t} | {A6[t]} | " + " | ".join(state(P(v, t)) for v in VS) + " | " + " / ".join(H) + f" | {A6H.get(t, '-')} |")
print()
print("- 끄면 같은지: 바꾸기 전 커밋 a448383 의 같은 명령 (진단 플래그 있음·없음 둘 다) 과 (A) 해시 대조 = R2 lbours/fem_cm/astick/chk (marks/astick_chk.log). a6 해시는 절 CM8 (커밋 aa5a7ef, 클러스터) 값.")
print()
print("## 2. 바닥 아래·튀어 오름·법선 구속 (프레임 평균 / 최대)")
print()
print("| 씬 | 판 | 바닥 아래 최대 깊이 mm | 아래 입자 수 | 정답 최대 깊이 mm | 튀어 오름 (질량 중심 z) | 정답 튀어 오름 | 법선 구속 노드 (걸음 끝, 프레임 안 걸음 최대) | 후보 노드 | 넣기 · 빼기 (프레임당) |")
print("|---|---|---|---|---|---|---|---|---|---|")


def mm(v, f="%.1f"):
    v = [q for q in v if q is not None]
    return ((f + " / " + f) % (np.mean(v), max(v))) if v else "-"


GTB = {}
for t in SCN:                                                                          # 정답 튀어 오름은 가장 길게 돈 판의 hg (정답은 판과 무관 -- 짧게 끝난 판에서 재면 잘림)
    L_ = max((jl(P(v, t)) for v in VS), key=len)
    GTB[t] = bounce([q["hg"][0] for q in L_ if "hg" in q])
for t in SCN:
    for v in VS:
        J = jl(P(v, t))
        if not J:
            continue
        pe = [q["pen"] for q in J if "pen" in q]
        hp = [q["hp"][0] if "hp" in q else None for q in J]
        fu = [q["fup"] for q in J if "fup" in q]
        act = [max(q["act"]) for q in fu if q["act"]]
        cand = [max(q["cand"]) for q in fu if q["cand"]]
        oo = [sum(q["on"]) for q in fu]; ff = [sum(q["off"]) for q in fu]
        print(f"| {t} | {VN[v]} | {mm([1000 * p['dmax'] for p in pe if p['dmax'] is not None], '%.2f')} | {mm([p['np'] for p in pe])} | {mm([1000 * p['gdmax'] for p in pe], '%.2f')} | "
              f"{fmt_b(bounce(hp))} ({len(J)} 프레임) | {fmt_b(GTB[t])} | {mm(act) if fu else '-'} | {mm(cand) if fu else '-'} | {('%.1f · %.1f' % (np.mean(oo), np.mean(ff))) if fu else '-'} |")
print()
print("## 3. 발산·멈춘 판 진단 (실행기 --pdiag: 이상 입자 = J ≤ 0.05·J > 5·비유한; 처음 이상 프레임과 마지막 프레임)")
print()
print("| 씬 | 판 | 멈춘 프레임 · 까닭 | 처음 이상 프레임: 이상 입자 수 · J≤0 수 · 중심 (x, y, z) · 바닥 위 높이 [최소, 중앙, 최대] | 마지막 유한 프레임: 이상 입자 · J≤0 · 최악 (J 최소) 입자 위치 · 바닥 위 높이 · 노드 상자 |")
print("|---|---|---|---|---|")
for t in SCN:
    for v in VS:
        s = sm(P(v, t)); J = jl(P(v, t))
        if not s or not s.get("diverged"):
            continue
        why = "GPU 메모리 부족 (노드 상자)" if s.get("oom") else ("비유한" if J and not np.isfinite(J[-1].get("err", 0) or np.nan) else "오차 > 200 %")
        pd = [(q["t"] + 1, q["pd"]) for q in J if "pd" in q]
        first = next(((f, p) for f, p in pd if p.get("nbad")), None)
        c1 = "-" if first is None else "프레임 %d: %d · %d · (%s) · [%s]" % (first[0], first[1]["nbad"], first[1]["nJneg"], ", ".join("%.3f" % x for x in first[1].get("bad_ctr", [])),
                                                                     ", ".join("%.3f" % x for x in first[1].get("bad_h", [])))
        pdf = [(f, p) for f, p in pd if (p.get("worst") or {}).get("x") and all(np.isfinite(p["worst"]["x"]))]   # 마지막 유한 프레임 (비유한 프레임 바로 앞)
        lf, lp = pdf[-1] if pdf else (pd[-1] if pd else (None, {}))
        w = lp.get("worst") or {}
        c2 = "-" if not pd else "프레임 %d: %d · %d · (%s) · %s · %s" % (lf, lp.get("nbad", 0), lp.get("nJneg", 0), ", ".join("%.3f" % x for x in w.get("x", [])),
                                                                     ("%.3f" % w["h"]) if "h" in w else "-", next((q.get("nn", "-") for q in J if q["t"] + 1 == lf), "-"))
        print(f"| {t} | {VN[v]} | {s['diverged']} · {why} | {c1} | {c2} |")
print()
print("## 4. 프레임별 질량 중심 z (mm; 정답 / " + " / ".join(VN[v] for v in VS) + ") -- L000")
print()
JS = {v: jl(P(v, "L000")) for v in VS}
n = max([len(J) for J in JS.values()] + [0])
ref = next((J for J in JS.values() if J and "hg" in J[0]), None)
if ref:
    print("| 프레임 | 정답 | " + " | ".join(VN[v] for v in VS) + " |")
    print("|---|---|" + "---|" * len(VS))
    for i in range(0, n, 2):
        g = ref[i]["hg"][0] if i < len(ref) and "hg" in ref[i] else None
        cells = []
        for v in VS:
            J = JS[v]
            cells.append("%.1f" % (1000 * J[i]["hp"][0]) if i < len(J) and "hp" in J[i] else "-")
        print(f"| {i + 1} | {'%.1f' % (1000 * g) if g is not None else '-'} | " + " | ".join(cells) + " |")
