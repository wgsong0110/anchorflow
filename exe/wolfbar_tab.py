"""절 CM107 묶음 wolfbar 표 (로컬, numpy 없이 json 만): R2 lbours/wolfbar 판 디렉토리 (wb.jsonl · fem.jsonl · summary.json) -> 마크다운.

사용: python exe/wolfbar_tab.py <내려받은 디렉토리> 판무리=이름,이름 ... > 표.md
  예: python exe/wolfbar_tab.py ~/wb H8=b8_1,b8_2 H5=b5_1,b5_2 N=bN_1,bN_2
정의 (fem_wolfbar.py): A = 장벽 활성 칸 (8 팔분 점 중 하나라도 이 프레임 어느 뉴턴 반복에서 장벽이 걸림, bcf bit 0) 의 점, N = 나머지 점유 칸 점.
프레임 부피 변화 (가우스 점, 풀이 + 반환 사상) = 마지막 걸음 ΣV·JY − 첫 걸음 ΣV·J0 (같은 프레임의 마지막 시도; K > 1 시도는 무리 구성이 걸음마다 조금 다름).
바닥 위 질량중심 = fem.jsonl hp[0] − 0.48 m (절 CM87.0).
"""
import json
import os
import sys

FLOOR = 0.48


def load(d):
    W = {}
    p = os.path.join(d, "wb.jsonl")
    if os.path.exists(p):
        for l in open(p):
            try:
                o = json.loads(l)
            except Exception:
                continue
            W[o["fr"] + 1] = o                                                           # t = fi h (fi 0 부터) -> 프레임 fi + 1 (fem.jsonl 줄 번호와 같게); 같은 프레임 여러 시도 -> 마지막 시도
    F = {}
    p = os.path.join(d, "fem.jsonl")
    if os.path.exists(p):
        for i, l in enumerate(open(p)):
            try:
                F[i + 1] = json.loads(l)
            except Exception:
                pass
    S = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else {}
    return W, F, S


def frame_rows(W):
    R = []
    for fr in sorted(W):
        o = W[fr]
        st = [q for q in o["steps"] if q.get("A") is not None]
        if not st:
            continue
        a, b = st[0], st[-1]
        row = dict(fr=fr, K=o["K"])
        for g in ("A", "N"):
            g0, g1 = a.get(g) or {}, b.get(g) or {}
            vj0 = g0.get("VJ0", 0.0)
            vjy = (g1.get("VJ0", 0.0) + g1.get("dVs", 0.0) + g1.get("dVr", 0.0)) if g1.get("n") else 0.0
            row[g] = dict(n=g1.get("n", 0), V=g1.get("V", 0.0), VJ0=vj0, dV=vjy - vj0 if g1.get("n") else 0.0, dVs=sum((q.get(g) or {}).get("dVs", 0.0) for q in st[-1:]),
                          dVr=sum((q.get(g) or {}).get("dVr", 0.0) for q in st), inv=g1.get("inv", 0), c09=g1.get("c09", 0), fb=g1.get("fb", 0.0), W=g1.get("W", 0.0),
                          fbmax=g1.get("fbmax", 0.0), nfb=g1.get("nfb", 0), nfbin=g1.get("nfbin", 0), nkq=max((q.get(g) or {}).get("nkq", 0) for q in st),
                          kqmax=max((q.get(g) or {}).get("kqmax", 0) for q in st), nkqs=max((q.get(g) or {}).get("nkqs", 0) for q in st))
        row["bands"] = b.get("bands")
        row["bands0"] = a.get("bands")
        r2 = [q.get("r2") for q in st if q.get("r2")]
        row["r2"] = r2[-1] if r2 else None
        row["g2"] = b.get("g2"); row["fail"] = max((q.get("fail") or 0) for q in st); row["smin"] = min((q.get("smin") if q.get("smin") is not None else 1.0) for q in st)
        yk = [q.get("yk") for q in st if q.get("yk")]
        row["yk"] = yk[0] if yk else None
        row["ncell"] = b.get("ncell"); row["nnode"] = b.get("nnode")
        R.append(row)
    return R


def com(F, fr):
    o = F.get(fr)
    if not o or not o.get("hp"):
        return None
    return 1000.0 * (float(o["hp"][0]) - FLOOR)


def fmt(x, p=1):
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != x:
            return "nan"
        return f"{x:.{p}f}"
    return str(x)


def first(R, f):
    for r in R:
        if f(r):
            return r["fr"]
    return None


def summarize(name, d):
    W, F, S = load(d)
    R = frame_rows(W)
    if not R:
        return dict(name=name, n=0, S=S)
    V0 = R[0]["A"]["VJ0"] + R[0]["N"]["VJ0"]
    o = dict(name=name, n=len(R), last=R[-1]["fr"], div=bool(S.get("diverged")), done=S.get("frames_done"), err=S.get("err_mean") or S.get("mean_err"))
    cA = sum(r["A"]["dV"] for r in R); cN = sum(r["N"]["dV"] for r in R)
    o.update(V0=V0, lossA=-cA / V0, lossN=-cN / V0, shareA=(cA / (cA + cN)) if (cA + cN) else None)
    o["volA"] = sum(r["A"]["VJ0"] for r in R) / max(sum(r["A"]["VJ0"] + r["N"]["VJ0"] for r in R), 1e-30)   # 프레임 평균 A 부피 몫
    # 시작: 처음으로 프레임 부피 손실이 전체의 0.5 % 를 넘는 무리·프레임, 처음 뒤집힌 점, 처음 J 10 % 넘게 줄어든 점
    o["f_lossA"] = first(R, lambda r: -r["A"]["dV"] > 0.005 * V0)
    o["f_lossN"] = first(R, lambda r: -r["N"]["dV"] > 0.005 * V0)
    o["f_invA"] = first(R, lambda r: r["A"]["inv"] > 0); o["f_invN"] = first(R, lambda r: r["N"]["inv"] > 0)
    o["f_c09A"] = first(R, lambda r: r["A"]["c09"] > 0); o["f_c09N"] = first(R, lambda r: r["N"]["c09"] > 0)
    o["invA"] = sum(r["A"]["inv"] for r in R); o["invN"] = sum(r["N"]["inv"] for r in R)
    # 높이 띠 누적
    nb = len(R[-1]["bands"]) if R[-1].get("bands") else 0
    o["band_loss"] = [-sum((r["bands"][i][3] + r["bands"][i][4]) for r in R if r.get("bands")) / V0 for i in range(nb)]
    o["band_vol"] = [sum(r["bands"][i][2] for r in R if r.get("bands")) / max(sum(sum(b[2] for b in r["bands"]) for r in R if r.get("bands")), 1e-30) for i in range(nb)]
    # 잔차
    rr = [r for r in R if r.get("r2") and r["r2"][2] > 0]
    o["r2A"] = sum(r["r2"][0] / r["r2"][2] for r in rr) / len(rr) if rr else None
    fl = [r for r in rr if r["fail"]]
    o["r2A_fail"] = sum(r["r2"][0] / r["r2"][2] for r in fl) / len(fl) if fl else None
    o["nfail"] = sum(1 for r in R if r["fail"]); o["nfr"] = len(R)
    g2 = sorted(r["g2"] for r in R if r.get("g2") is not None)
    o["g2med"] = g2[len(g2) // 2] if g2 else None
    o["nodeA"] = sum(r["nnode"][0] / max(r["nnode"][0] + r["nnode"][1], 1) for r in R if r.get("nnode")) / max(1, sum(1 for r in R if r.get("nnode")))
    # 장벽 힘
    fbw = [r["A"]["fb"] / r["A"]["W"] for r in R if r["A"]["W"] > 0]
    o["fbW_med"] = sorted(fbw)[len(fbw) // 2] if fbw else None; o["fbW_max"] = max(fbw) if fbw else None
    o["nfbin"] = sum(r["A"]["nfbin"] for r in R)
    # 시작점 섞기·보폭 자름
    yk = [r["yk"] for r in R if r.get("yk")]
    o["ykA"] = sum(q[0] for q in yk) / len(yk) if yk else None; o["ykN"] = sum(q[1] for q in yk) / len(yk) if yk else None
    o["bA"] = sum(q[2] for q in yk) / len(yk) if yk else None; o["bN"] = sum(q[3] for q in yk) / len(yk) if yk else None
    o["kqA"] = sum(r["A"]["nkq"] for r in R) / len(R); o["kqN"] = sum(r["N"]["nkq"] for r in R) / len(R)
    o["kqsA"] = sum(r["A"]["nkqs"] for r in R) / len(R); o["kqsN"] = sum(r["N"]["nkqs"] for r in R) / len(R)
    o["nfr_kq"] = sum(1 for r in R if r["A"]["nkq"] + r["N"]["nkq"] > 0)
    o["com"] = {f: com(F, f) for f in (10, 20, 30, 40, 50)}
    o["R"] = R
    o["S"] = S
    return o


def main():
    d0 = sys.argv[1]
    groups = []
    for a in sys.argv[2:]:
        k, v = a.split("=", 1)
        groups.append((k, [q for q in v.split(",") if q]))
    out = []
    P = lambda *x: out.append(" ".join(str(q) for q in x))
    allo = {}
    for g, names in groups:
        allo[g] = [summarize(n, os.path.join(d0, n)) for n in names]
    P("### 표 1 -- 판별 요약 (A = 장벽 활성 칸, N = 나머지 점유 칸; 손실 = 첫 프레임 가우스 점 부피 ΣV·J0 대비 누적, 프레임 안 풀이 + 반환 사상)")
    P("")
    P("| 판 | 발산 · 끝 프레임 | 바닥 위 질량중심 f30 · f50 mm | 손실 A · N (% V0) | A 몫 (손실 · 부피) | 첫 손실 > 0.5 % 프레임 A · N | 첫 뒤집힌 점 A · N (합) | 첫 J −10 % 점 A · N |")
    P("|---|---|---|---|---|---|---|---|")
    for g, L in allo.items():
        for o in L:
            if not o.get("n"):
                P(f"| {g} {o['name']} | 기록 없음 | | | | | | |"); continue
            P(f"| {g} {o['name']} | {'발산' if o['div'] else '완주'} · {o['last']} | {fmt(o['com'][30], 0)} · {fmt(o['com'][50], 0)} | {fmt(100 * o['lossA'], 1)} · {fmt(100 * o['lossN'], 1)} | "
              f"{fmt(o['shareA'], 2)} · {fmt(o['volA'], 2)} | {fmt(o['f_lossA'])} · {fmt(o['f_lossN'])} | {fmt(o['f_invA'])} · {fmt(o['f_invN'])} ({o['invA']} · {o['invN']}) | {fmt(o['f_c09A'])} · {fmt(o['f_c09N'])} |")
    P("")
    P("### 표 2 -- 높이 띠별 누적 손실 (% V0) [부피 몫] (띠 = 걸음 끝 가우스 점 높이: 바닥 아래 · [0, d̂) · [d̂, 2 cm) · [2, 5) · [5, 10) · ≥ 10 cm)")
    P("")
    P("| 판 | <0 | [0,d̂) | [d̂,2) | [2,5) | [5,10) | ≥10 |")
    P("|---|---|---|---|---|---|---|")
    for g, L in allo.items():
        for o in L:
            if not o.get("n") or not o.get("band_loss"):
                continue
            P(f"| {g} {o['name']} | " + " | ".join(f"{fmt(100 * a, 1)} [{fmt(b, 2)}]" for a, b in zip(o["band_loss"], o["band_vol"])) + " |")
    P("")
    P("### 표 3 -- 풀이·장벽 (프레임 평균; 잔차 몫 A = 마지막 뉴턴 반복 시작 Σ|g|² 중 A 칸 노드 몫; 덜 풀린 = fem uf.fail; 장벽 힘 / A 무게; 시작점 β<1 노드 수 · 질량 가중 β; 보폭 자름 점 = 뉴턴 · 시작점 실현 가능 k > 0 인 점)")
    P("")
    P("| 판 | 덜 풀린 프레임 | 걸음 끝 잔차 비 중앙 | 잔차 몫 A (전체 · 덜 풀린 걸음) | A 노드 몫 | 장벽 힘/무게 중앙 · 최대 (d≤0 활성 점 합) | 시작점 β<1 노드 A · N | 질량 가중 β A · N | 보폭 자름 점 A · N (프레임) | 시작점 실현 가능 자름 점 A · N |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for g, L in allo.items():
        for o in L:
            if not o.get("n"):
                continue
            P(f"| {g} {o['name']} | {o['nfail']}/{o['nfr']} | {fmt(o['g2med'], 2)} | {fmt(o['r2A'], 2)} · {fmt(o['r2A_fail'], 2)} | {fmt(o['nodeA'], 2)} | {fmt(o['fbW_med'], 2)} · {fmt(o['fbW_max'], 1)} ({o['nfbin']}) | "
              f"{fmt(o['ykA'], 0)} · {fmt(o['ykN'], 0)} | {fmt(o['bA'], 2)} · {fmt(o['bN'], 2)} | {fmt(o['kqA'], 0)} · {fmt(o['kqN'], 0)} ({o['nfr_kq']}) | {fmt(o['kqsA'], 0)} · {fmt(o['kqsN'], 0)} |")
    P("")
    P("### 표 4 -- 프레임별 (각 판무리 첫 판; 프레임 부피 변화 % V0 A · N, 뒤집힌 점 A · N, 잔차 몫 A, 덜 풀림, 장벽 활성 칸 수)")
    P("")
    for g, L in allo.items():
        o = L[0] if L else None
        if not o or not o.get("n"):
            continue
        V0 = o["V0"]
        P(f"{g} {o['name']}:")
        P("")
        P("| 프레임 | K | ΔV A · N % | 뒤집힌 A · N | J −10 % A · N | 잔차 비 · 몫 A | 덜 풀림 | 장벽 칸 · 나머지 칸 | 장벽 힘/무게 A | 질량중심 mm |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for r in o["R"]:
            if r["fr"] % 5 and r["fr"] > 3 and not (r["A"]["inv"] or r["N"]["inv"]) and r["fr"] not in (o.get("f_lossA"), o.get("f_lossN")):
                continue
            ra = (r["r2"][0] / r["r2"][2]) if (r.get("r2") and r["r2"][2] > 0) else None
            fw = r["A"]["fb"] / r["A"]["W"] if r["A"]["W"] > 0 else None
            P(f"| {r['fr']} | {r['K']} | {fmt(100 * r['A']['dV'] / V0, 2)} · {fmt(100 * r['N']['dV'] / V0, 2)} | {r['A']['inv']} · {r['N']['inv']} | {r['A']['c09']} · {r['N']['c09']} | {fmt(r['g2'], 2)} · {fmt(ra, 2)} | {r['fail']} | "
              f"{(r.get('ncell') or ['-', '-'])[0]} · {(r.get('ncell') or ['-', '-'])[1]} | {fmt(fw, 2)} | {fmt(com(load(os.path.join(d0, o['name']))[1], r['fr']), 0)} |")
        P("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
