"""절 CM48 묶음 ghost2 검사 요약: python exe/ghost2_chk.py <결과 디렉토리> -> markdown (stdout).
g2fd.jsonl (커널 대 torch 참값·유한차분·훅 자리), g2sys.jsonl (뉴턴 반복별 선탐색 기록 -> 전체 목적 함수 기울기 유한차분), 결정적 해시 (fem.jsonl xh / summary x_sha1), g2fs.jsonl (면 집합 바뀜).
"""
import json
import math
import os
import sys

O = sys.argv[1]


def jl(n, f):
    p = os.path.join(O, n, f)
    if not os.path.exists(p):
        return None
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def runs():
    return sorted(d for d in os.listdir(O) if os.path.isdir(os.path.join(O, d)))


def g(x, f="%.1e"):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f % x)


print("## 검사 A~F: 커널 대 torch f64 참값 (상대 차; k = 그 판 k, g = 0 판은 k = 1)\n")
print("| 판 | 프레임 | 경계 칸 커널 · 참값 | 면 커널 · 참값 · 같은 집합 | E | f (참값) · 참값 autograd | Gv | 유한차분 E 비 · H | 대각 | 선탐색 후보 (i0·gate 포함) | Φ |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
HK = []
for n in runs():
    for r in jl(n, "g2fd.jsonl") or []:
        print(f"| {n} | {r['f']} | {r['ncut_k']} · {r['ncut_r']} | {r['nface_k']} · {r['nface_r']} · {'예' if r['face_same'] else '**아니오**'} | {g(r['E_rel'])} | {g(r['f_rel'])} · {g(r['f_auto_rel'])} | {g(r['mv_rel'])} | "
              f"{g(r.get('fdE_ratio'), '%.6f')} · {g(r.get('fdH_rel'))} | {g(r.get('diag_rel'))} | {g(r.get('ls_rel_max'))} | {g(r.get('phi_rel'))} |")
        if r.get("hook"):
            HK.append((n, r["f"], r["hook"]))
print()
print("## 검사 H: 훅 자리 -- 기억한 실행기 커널 인자로 '훅 켬 − 훅 끔' 대 벌점 커널 (k 를 1e3 배로; 상대 차)\n")
print("| 판 | 프레임 | 커널 | 결과 |")
print("|---|---|---|---|")
for n, f, h in HK:
    for nm, d in sorted(h.items()):
        print(f"| {n} | {f} | {nm} | " + ", ".join(f"{k}={g(v) if isinstance(v, float) else v}" for k, v in d.items()) + " |")
print()

print("## 전체 목적 함수 선탐색 유한차분 (g2sys; --lsfast 뺌): 반복마다 s_i = (Φ(c_i) − Φ(0))/c_i, c_i = 2^-i, 리처드슨 g·δ_FD = 2 s_9 − s_8 대 커널 g·δ\n")
print("| 판 | 반복 수 (g·δ < 0, 겹침 뺌) | |g·δ_FD / g·δ − 1| 중앙 · 90 % · 최대 | > 1e-2 몫 | 보폭 1 몫 · 0 몫 |")
print("|---|---|---|---|---|")
for n in runs():
    L = jl(n, "g2sys.jsonl")
    if not L:
        continue
    seen, rr, st1, st0, tot = set(), [], 0, 0, 0
    for q in L:
        for row in q["rows"]:
            k_, it, g2, E0, gd, st = row[:6]
            Ec = row[6:]
            key = (round(E0, 9), round(gd, 12))
            if key in seen or not (gd < 0) or not all(map(math.isfinite, [E0, gd] + Ec)) or E0 == 0.0:
                continue
            seen.add(key)
            tot += 1
            st1 += st == 1.0
            st0 += st == 0.0
            c = [0.5 ** i for i in range(len(Ec))]
            s = [(Ec[i] - E0) / c[i] for i in range(len(Ec))]
            fd = 2 * s[-1] - s[-2]
            rr.append(abs(fd / gd - 1.0))
    if rr:
        rr.sort()
        print(f"| {n} | {tot} | {rr[len(rr) // 2]:.1e} · {rr[int(0.9 * (len(rr) - 1))]:.1e} · {rr[-1]:.1e} | {sum(1 for x in rr if x > 1e-2) / len(rr):.2f} | {st1 / tot:.2f} · {st0 / tot:.2f} |")
print()

print("## 결정적 해시 (예산 안 걸림 @tb 1000): E 대 E + ghost (g = 0)\n")
print("| 씬 | 판 | 프레임 | 끝 해시 | 첫 다른 프레임 (대 E) | g2info 벌점 커널 넣은 수 |")
print("|---|---|---|---|---|---|")
for sc in sorted({n.split("_")[0] for n in runs() if n.startswith("d") and "_" in n}):
    ns = [n for n in runs() if n.split("_")[0] == sc]
    if not ns:
        continue
    base = None
    for n in ns:
        r = jl(n, "fem.jsonl") or []
        hs = [x.get("xh") for x in r]
        try:
            sm = json.load(open(os.path.join(O, n, "summary.json")))
        except Exception:
            sm = {}
        last = sm.get("x_sha1", hs[-1] if hs else None)
        try:
            info = json.load(open(os.path.join(O, n, "g2info.json")))
            nk = sum(info.get("nk", {}).values())
        except Exception:
            nk = "-"
        if n.endswith("_E"):
            base = (hs, last)
        fd = "-"
        if base and not n.endswith("_E"):
            if any(hs) and any(base[0]):
                fd = next((i + 1 for i, (a, b) in enumerate(zip(hs, base[0])) if a != b), "없음 (모두 같음)")
            else:
                fd = "같음" if last == base[1] else "**다름**"
        print(f"| {sc} | {n} | {len(r)} | {(last or '-')[:10]} | {fd} | {nk} |")
print()

print("## 경계 칸·면 집합 바뀜 (g2fs; 프레임 사이 절대 칸 좌표 Jaccard)\n")
print("| 판 | 프레임 수 | 경계 칸 수 범위 | Jaccard 경계 칸 중앙 · 최소 | Jaccard 면 중앙 · 최소 | 찬 칸 Jaccard 중앙 |")
print("|---|---|---|---|---|---|")
for n in runs():
    L = jl(n, "g2fs.jsonl")
    if not L:
        continue
    jc = sorted(q["jac_cut"] for q in L if "jac_cut" in q)
    jf = sorted(q["jac_face"] for q in L if "jac_face" in q)
    ja = sorted(q["jac_act"] for q in L if "jac_act" in q)
    nc = [q["ncut"] for q in L]
    if jc:
        print(f"| {n} | {len(L)} | {min(nc)}~{max(nc)} | {jc[len(jc) // 2]:.3f} · {jc[0]:.3f} | {jf[len(jf) // 2]:.3f} · {jf[0]:.3f} | {ja[len(ja) // 2]:.3f} |")
print()
