"""절 CM93 묶음 wolfh12 진단 표 (로컬, json 만 읽음): python exe/wolfh12_tab.py <무리=판디렉토리[,판...]> ... > 표.md
판마다: 걸음 끝 잔차 비 (fem.jsonl uf.g2, 걸음 있는 프레임) 중앙·덜 풀린 걸음 몫, 뉴턴·PCG 수, 풀이 체적 비 J_qX/J_q0 (fi.jsonl, f20~40 평균·f1~50 곱),
잔차와 압축의 순위 상관 (프레임별 log g2 대 1 − J_qX/J_q0), w12.jsonl (f30): J < 0.9 점 수·V 가중 평균 tr·평균 키르히호프 응력 p (kPa)·헤시안 체적 곡률 비 h⁺/h·Σ V Ψ_v (J).
"""
import json
import math
import os
import statistics as stt
import sys


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
    for k, i in enumerate(o):
        r[i] = float(k)
    return r


def spear(a, b):
    if len(a) < 4:
        return float("nan")
    ra, rb = rank(a), rank(b); ma, mb = stt.mean(ra), stt.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 0 else float("nan")


def one(d):
    F = [q for q in jl(os.path.join(d, "fem.jsonl")) if "t" in q]
    I = jl(os.path.join(d, "fi.jsonl")); W = jl(os.path.join(d, "w12.jsonl"))
    s = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else {}
    g, fl, nN, nP, cr, rows = [], 0, [], [], [], []
    for k, f in enumerate(F):
        if f["it"]["nN"] == 0:
            continue
        g2 = f["uf"]["g2"][0]; g.append(g2); fl += f["uf"]["fail"][0]; nN.append(f["it"]["nN"]); nP.append(f["it"]["nP"])
        if k < len(I) and I[k].get("Jq0"):
            rows.append((math.log10(max(g2, 1e-12)), 1 - I[k]["JqX"] / I[k]["Jq0"]))
    jr = [i["JqX"] / i["Jq0"] for i in I if i.get("Jq0")]
    w30 = W[29] if len(W) >= 30 else (W[-1] if W else None)
    c = (w30 or {}).get("c09", {})
    a = (w30 or {}).get("all", {})
    return dict(err=s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done"), div=s.get("diverged"), nf=len(F),
                g2=stt.median(g) if g else float("nan"), fail=fl / len(g) if g else float("nan"), nN=stt.mean(nN) if nN else 0, nP=stt.mean(nP) if nP else 0,
                jr2040=stt.mean(jr[19:40]) if len(jr) >= 40 else float("nan"), jprod=math.prod(jr) if jr else float("nan"),
                rho=spear([r[0] for r in rows], [r[1] for r in rows]),
                n09=c.get("n", 0), tr09=c.get("tr"), p09=(c.get("p") or 0) / 1e3 if c.get("n") else None, hr09=c.get("hr"), Ev09=c.get("Ev"),
                Evall=a.get("Ev"), medA=I[-1].get("medA") if I else None)


def f_(v, p=2):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{p}f}"


def g_(v):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.3g}"


print("| 무리 | 판 | 발산 | 오차 % | 걸음 끝 잔차 비 중앙 | 덜 풀린 걸음 몫 | 뉴턴 · PCG (프레임 평균) | 풀이 J_qX/J_q0 f20~40 평균 | 풀이 체적 비 곱 f1~50 | 순위 상관 (log 잔차, 압축) | f30 J<0.9 점 수 | 그 점 tr · p kPa · h⁺/h | Σ V Ψ_v (J<0.9 · 전체) | 입자 J 중앙 끝 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for a in sys.argv[1:]:
    nm, ds = a.split("=", 1)
    for d in ds.split(","):
        r = one(d)
        print(f"| {nm} | {os.path.basename(d.rstrip('/'))} | {r['div']} | {f_(100 * r['err'] if r['err'] is not None else None)} | {g_(r['g2'])} | {f_(r['fail'])} | {r['nN']:.1f} · {r['nP']:.1f} | {f_(r['jr2040'], 3)} | {g_(r['jprod'])} | {f_(r['rho'])} | {r['n09']} | {f_(r['tr09'])} · {g_(r['p09'])} · {f_(r['hr09'])} | {g_(r['Ev09'])} · {g_(r['Evall'])} | {f_(r['medA'], 3)} |")
