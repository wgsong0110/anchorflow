"""절 CM18 표 (astick_snh): python exe/fem_astick_snh_table.py <묶음 디렉토리 astick_snh>
판 (lego L000 만, 결정적, 절 CM17 명령 그대로 + 아래):
  sa = (A) + SNH (--elx snh), su1 = (i) + SNH (--fl_up 1 --elx snh), su2 = (ii) + SNH (--fl_up 2 --elx snh)
  fa·fu1·fu2 = 절 CM17 의 (A)·(i)·(ii) (FCR) 같은 명령 + 진단 --qjd 만 (가우스 점 J 를 재려고 다시 돌림; 절 CM17 결과 cm17/<a|u1|u2>_L000 과 프레임별 오차가 같은지 대조 = 끄면 같음)
디렉토리 <판>_L000 (summary.json·fem.jsonl: pen·hg·hp·pd·qj·fup, pred.pt 덤프).
튀어 오름 = 질량 중심 높이 (바닥 법선 z, --hgt) 의 첫 극소 뒤 최대 상승 (절 CM17 표와 같은 식). 정답 = 가장 길게 돈 판의 hg.
처음 뒤집힘 = 입자 (--pdiag: J = det F ≤ 0 이 처음 생긴 프레임, 그 프레임 J 최소 입자) · 가우스 점 (--qjd: 프레임 끝 커밋된 U 의 det X_q ≤ 0 이 처음 생긴 프레임, J 최소 점).
처음 비유한 = 덤프 pred.pt 에서 위치가 처음 비유한이 된 프레임의 그 입자들의 바로 앞 프레임 위치 (중심·바닥 위 높이)."""
import json
import os
import sys

import numpy as np

B = sys.argv[1]
SN = ("sa", "su1", "su2")
FC = ("fa", "fu1", "fu2")
CM = {"fa": "a", "fu1": "u1", "fu2": "u2"}
VN = {"sa": "(A) + SNH", "su1": "(i) + SNH", "su2": "(ii) + SNH", "fa": "(A) FCR (절 CM17)", "fu1": "(i) FCR (절 CM17)", "fu2": "(ii) FCR (절 CM17)"}
ALL = FC + SN
NR, PN = None, None


def P(v):
    return os.path.join(B, f"{v}_L000")


def sm(d):
    p = os.path.join(d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def jl(d):
    p = os.path.join(d, "fem.jsonl")
    out = []
    if os.path.exists(p):
        for l in open(p):
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def state(d):
    s = sm(d)
    if s is None:
        ri = os.path.join(d, "run_info.txt")
        t = open(ri).read() if os.path.exists(ri) else ""
        return ("실패 (" + t.split("rc=")[-1].split()[0] + ")") if "rc=" in t else ("없음" if not t else "도는 중")
    dv = s.get("diverged")
    e = 100 * (s["err_mean_done"] if dv else s["err_mean"])
    if not dv:
        return "완주 %.2f" % e
    J = jl(d)
    why = "GPU 메모리 부족 (노드 상자 %s)" % (J[-1].get("nn") if J else "-") if s.get("oom") else ("비유한" if J and not np.isfinite(J[-1].get("err", 0) or np.nan) else "오차 > 200 %")
    return "**x%d** %s (발산 전 %.2f)" % (dv, why, e)


def bounce(z):
    z = [q for q in z if q is not None]
    for i in range(1, len(z) - 1):
        if z[i] < z[i - 1] and z[i + 1] >= z[i]:
            rest = z[i:]
            return i + 1, 1000 * (max(rest) - z[i]), int(np.argmax(rest)) + i + 1
    return None


def fmt_b(b):
    return "없음 (첫 극소 없음)" if b is None else "%.1f mm (착지 %d, 최고 %d)" % (b[1], b[0], b[2])


def mm(v, f="%.1f"):
    v = [q for q in v if q is not None]
    return ((f + " / " + f) % (np.mean(v), max(v))) if v else "-"


def fx(x):
    return "(" + ", ".join("%.3f" % q for q in x) + ")"


def first_nonfinite(d):
    p = os.path.join(d, "pred.pt")
    if not os.path.exists(p):
        return None
    import torch
    X = torch.load(p, map_location="cpu", weights_only=False)["pred"].double()
    bad = ~torch.isfinite(X).all(-1)
    fr = torch.nonzero(bad.any(-1)).squeeze(1)
    if not fr.numel():
        return "없음"
    f = int(fr[0])
    ids = torch.nonzero(bad[f]).squeeze(1)
    if f == 0:
        return "프레임 1: %d 입자 (앞 프레임 덤프 없음)" % ids.numel()
    xp = X[f - 1, ids]
    ok = torch.isfinite(xp).all(-1); xp = xp[ok]
    h = xp @ torch.tensor(NR, dtype=torch.float64) - PN
    return "프레임 %d: %d 입자 (전체 %d), 바로 앞 프레임 중심 %s, 높이 [%.3f, %.3f, %.3f] m" % (f + 1, ids.numel(), X.shape[1], fx(xp.mean(0).tolist()), float(h.min()), float(h.median()), float(h.max()))


for v in ALL:
    s = sm(P(v))
    if s and NR is None:
        try:
            import torch
            dd = torch.load(s["args"]["data"], map_location="cpu", weights_only=False)
            pl = [q for q in dd["cfg"]["boundary_conditions"] if q.get("type") == "surface_collider"][0]   # 실행기 --floor vel 과 같은 평면
            NR = np.array(pl["normal"], dtype=np.float64); NR = (NR / np.linalg.norm(NR)).tolist(); PN = float(np.dot([float(q) for q in pl["point"]], NR))
        except Exception:
            pass
if NR is None:
    NR, PN = [0.0, 0.0, 1.0], 0.0

print("# 절 CM18 astick_snh -- A 바닥 ((A) 원래·(i) --fl_up 1·(ii) --fl_up 2) x 탄성 FCR / Stable Neo-Hookean (--elx snh, μ_s = μ, λ_s = λ + μ), lego L000, 결정적")
print()
print("## 1. 판정 (평균 오차 % 또는 x 멈춘 프레임)")
print()
print("| 바닥 | FCR (절 CM17 결과) | FCR 다시 (+ --qjd) 와 절 CM17 대조 | SNH |")
print("|---|---|---|---|")
for f_, s_, nm in (("fa", "sa", "(A) 원래"), ("fu1", "su1", "(i) 법선 한쪽 + 접선 고착"), ("fu2", "su2", "(ii) 법선 한쪽 + 접선 자유")):
    c17 = os.path.join(B, "cm17", CM[f_] + "_L000")
    J0, J1 = jl(c17), jl(P(f_))
    e0 = [q.get("err") for q in J0]; e1 = [q.get("err") for q in J1]
    same = (len(e0) == len(e1) and all((a == b) or (a != a and b != b) for a, b in zip(e0, e1)) and [q.get("pd") for q in J0] == [q.get("pd") for q in J1]
            and (sm(c17) or {}).get("diverged") == (sm(P(f_)) or {}).get("diverged"))
    print(f"| {nm} | {state(c17)} | {('같음 (프레임 %d 개 오차·입자 진단 비트 같음)' % len(e1)) if same and e1 else ('**다름** (%d 대 %d 프레임)' % (len(e0), len(e1)))} | {state(P(s_))} |")
print()
print("## 2. 바닥 아래·튀어 오름 (프레임 평균 / 최대)")
print()
JS = {v: jl(P(v)) for v in ALL}
Lg = max(JS.values(), key=len)
GTB = bounce([q["hg"][0] for q in Lg if "hg" in q])
print(f"정답 튀어 오름 {fmt_b(GTB)} (정답 13.4 mm 기준), 정답 바닥 아래 최대 깊이 {mm([1000 * q['pen']['gdmax'] for q in Lg if 'pen' in q], '%.2f')} mm")
print()
print("| 판 | 바닥 아래 최대 깊이 mm | 아래 입자 수 | 튀어 오름 (질량 중심 z) | 돈 프레임 |")
print("|---|---|---|---|---|")
for v in ALL:
    J = JS[v]
    if not J:
        print(f"| {VN[v]} | - | - | - | 0 |"); continue
    pe = [q["pen"] for q in J if "pen" in q]
    print(f"| {VN[v]} | {mm([1000 * p['dmax'] for p in pe if p['dmax'] is not None], '%.2f')} | {mm([p['np'] for p in pe])} | {fmt_b(bounce([q['hp'][0] if 'hp' in q else None for q in J]))} | {len(J)} |")
print()
print("## 3. 처음 뒤집힘·처음 비유한")
print()
print("| 판 | 처음 J ≤ 0 입자 (프레임: 수, J 최소 입자 위치, 바닥 위 높이 m) | 처음 J ≤ 0 가우스 점 (프레임: 수 / 부피 > 0 점, J 최소 점 위치, 높이 m; J ≤ 0 점 높이 [최소, 중앙, 최대]) | 처음 비유한 위치 (덤프) | 마지막 프레임 J ≤ 0 입자 · 가우스 점 |")
print("|---|---|---|---|---|")
for v in ALL:
    J = JS[v]
    if not J:
        continue
    fp = next(((q["t"] + 1, q["pd"]) for q in J if q.get("pd", {}).get("nJneg")), None)
    cp_ = "없음" if fp is None else "프레임 %d: %d, %s, %s" % (fp[0], fp[1]["nJneg"], fx(fp[1]["worst"]["x"]) if fp[1].get("worst") else "-", ("%.3f" % fp[1]["worst"]["h"]) if fp[1].get("worst") else "-")
    fq = next(((q["t"] + 1, q["qj"]) for q in J if q.get("qj", {}).get("nneg")), None)
    cq = ("- (--qjd 없음)" if not any("qj" in q for q in J) else "없음") if fq is None else "프레임 %d: %d / %d, %s, %.3f; [%s]" % (
        fq[0], fq[1]["nneg"], fq[1]["nv"], fx(fq[1]["x"]), fq[1]["h"], ", ".join("%.3f" % q for q in fq[1].get("neg_h", [])))
    try:
        nf = first_nonfinite(P(v))
    except Exception as ex:
        nf = "읽기 실패 %s" % str(ex)[:80]
    s = sm(P(v)) or {}
    if s.get("oom") and nf == "없음":
        nf = "없음 (프레임 %d GPU 메모리 부족으로 멈춤)" % s["diverged"]
    last = J[-1]
    print(f"| {VN[v]} | {cp_} | {cq} | {nf} | {last.get('pd', {}).get('nJneg', '-')} · {last.get('qj', {}).get('nneg', '-')} |")
print()
print("## 4. 프레임별 J ≤ 0 가우스 점 수 (프레임 끝, 부피 > 0) / J ≤ 0 입자 수")
print()
n = max(len(J) for J in JS.values())
print("| 프레임 | " + " | ".join(VN[v] for v in ALL) + " |")
print("|---|" + "---|" * len(ALL))
for i in range(n):
    row = []
    for v in ALL:
        J = JS[v]
        if i < len(J):
            row.append("%s / %s" % (J[i].get("qj", {}).get("nneg", "-"), J[i].get("pd", {}).get("nJneg", "-")))
        else:
            row.append("")
    print(f"| {i + 1} | " + " | ".join(row) + " |")
print()
print("## 5. 프레임별 질량 중심 z (mm) · 오차 %")
print()
print("| 프레임 | 정답 z | " + " | ".join(VN[v] + " z · 오차" for v in ALL) + " |")
print("|---|---|" + "---|" * len(ALL))
for i in range(0, n, 2):
    g = Lg[i]["hg"][0] if i < len(Lg) and "hg" in Lg[i] else None
    row = []
    for v in ALL:
        J = JS[v]
        row.append(("%.1f · %.2f" % (1000 * J[i]["hp"][0], 100 * J[i]["err"])) if i < len(J) and "hp" in J[i] and J[i].get("err") is not None and J[i]["err"] == J[i]["err"] else ("" if i >= len(J) else "x"))
    print(f"| {i + 1} | {'%.1f' % (1000 * g) if g is not None else '-'} | " + " | ".join(row) + " |")
