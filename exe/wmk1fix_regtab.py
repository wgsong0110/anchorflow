"""절 CM24 회귀 표: 레고·ficus·wolf·bread 기본 판 K 1 (off, k1new 명령 그대로) 대 판 (같은 명령 + 판 플래그). 결정적 판정 판 + 비결정 단독 시간.
사용: python exe/wmk1fix_regtab.py <출력 디렉토리> <판 이름>...
"""
import json
import os
import sys
from collections import Counter

O = sys.argv[1]
VS = sys.argv[2:]
SC = [("L", "레고"), ("F", "ficus"), ("W", "wolf"), ("B", "bread")]


def js(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def jl(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except Exception:
        return []


def res(s):
    if s is None:
        return "없음"
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else ("완주 %d" % s["frames_done"])


def zms(n):
    r = jl(f"{O}/{n}/fem.jsonl")
    ms = [q["ms"] for q in r][2:]
    ms = [q for q in ms if q == q]
    return (f"{sum(ms) / len(ms):.2f} · {max(ms):.2f}") if ms else "-"


def zt(s, v, r):
    if v != "off":
        return zms(f"z_{s}_{v}_{r}")
    out = [f"{w}: {zms(f'z_{s}_off{w}_{r}')}" for w in VS if os.path.exists(f"{O}/z_{s}_off{w}_{r}")]
    if os.path.exists(f"{O}/z_{s}_off_{r}"):
        out.append("ks3: " + zms(f"z_{s}_off_{r}"))
    return "; ".join(out) or "-"


print("## 4. 회귀 -- 레고·ficus·wolf·bread 기본 판 K 1 (결정적 판정 판; 단독 시간은 비결정 a · b, ms 평균 · 최대 (첫 2 프레임 뺌); off 단독 시간은 판 묶음마다 같은 카드에서 판과 번갈아 잰 값)\n")
print("| 씬 | 판 | 결과 | 오차 % | 끝 해시 (off 는 k1new K1 과 대조) | 덜 풀린 걸음 프레임 / 프레임 | 최종 K 분포 | 단독 ms a | 단독 ms b |")
print("|---|---|---|---|---|---|---|---|---|")
for s, nm in SC:
    for v in ["off"] + VS:
        n = f"r_{s}_{v}"
        sm = js(f"{O}/{n}/summary.json")
        rows = jl(f"{O}/{n}/fem.jsonl")
        hs = sm.get("x_sha1") if sm else None
        if v == "off":
            k = js(f"{O}/{n}/k1new_summary.json")
            hs = f"{hs} ({'같음' if (k and k.get('x_sha1') == hs) else ('다름 ' + str(k.get('x_sha1')) if k else 'k1new 없음')})"
        uf = [q for q in rows if q.get("uf")]
        nf = sum(1 for q in uf if any(q["uf"]["fail"]))
        ks = Counter(q["ks"]["K"] for q in rows if q.get("ks"))
        per = ""
        if s == "F":
            p = f"{O}/{n}/period.txt"
            if os.path.exists(p):
                per = " (" + " ".join(open(p).read().split("\n")[-3:]).strip()[:120] + ")"
        em = f"{100 * sm['err_mean_done']:.2f}" if sm else "-"
        print(f"| {nm} | {'K 1 기본 판 (off)' if v == 'off' else v} | {res(sm)} | {em}{per} | {hs} | {f'{nf} / {len(rows)}' if uf else '-'} | {dict(sorted(ks.items())) if ks else '-'} | {zt(s, v, 'a')} | {zt(s, v, 'b')} |")
