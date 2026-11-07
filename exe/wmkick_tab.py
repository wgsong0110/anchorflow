"""절 CM23.4 묶음 wmkick 표: 수박 기본 판 K 1 첫 접촉 뒤 위쪽 운동량의 출처.
사용: python exe/wmkick_tab.py <출력 디렉토리>
힘 몫 (N, 위쪽 = 바닥 법선 성분 합): 탄성 −f_el, 장벽 −G, 관성 −M(y − ŷ)/τ² (ŷ 에 중력 포함), 중력 m g, 잔차 g = 기울기 (= 관성 + 탄성 + 장벽 의 음 -- 수렴이면 0).
속도 변화 분해 (m/s, 질량 가중 평균): Δv = 가속도 기억 dt(1−γ)a0 + 중력 γ dt g + 탄성 γ dt (−f_el)/m + 장벽 γ dt (−G)/m + 잔차 γ dt g/m (노드마다 정확한 HHT 식).
튐 = 질량중심 z 가 그때까지 최저점보다 오른 최대 (m). 프레임 5 위쪽 속도 = (질량중심 z f5 − f4)/h (+ 진단 판은 f5 끝 입자 질량 가중 법선 속도).
"""
import json
import os
import sys

O = sys.argv[1]
VAR = [("base", "기본 판 (장벽 κ, d̂ = ½ dx)"), ("k01", "(a) κ x 0.1"), ("k10", "(a) κ x 10"), ("dh2", "(b) d̂ x 2 (κ 정의대로 = x 1/4)"), ("dh2k", "(b') d̂ x 2, κ 고정"),
       ("slip", "(c) 예전 미끄럼 바닥 (장벽 없음)"), ("n8p20", "(d) 뉴턴 8 x PCG 20 (wmk1conv d_N8P20 재사용)")]


def load(n):
    p = os.path.join(O, n, "fem.jsonl") if n != "n8p20" else os.path.join(O, "n8p20_fem.jsonl")
    if not os.path.exists(p):
        return None, None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    sp = os.path.join(O, n, "summary.json")
    s = json.load(open(sp)) if os.path.exists(sp) else None
    if n == "n8p20":
        s = dict(frames_done=len(rows), diverged=None)
    return rows, s


def bounce(zs):
    lo, b, at = float("inf"), 0.0, None
    for i, z in enumerate(zs):
        if z is None:
            continue
        lo = min(lo, z)
        if z - lo > b:
            b, at = z - lo, i + 1
    return b, at


def fm(x, f="{:.3g}"):
    return "-" if x is None else f.format(x)


base, bs = load("base")
H = None
if base:
    for r in base:
        if "fr" in r and "dt" in r["fr"]:
            H = r["fr"]["dt"]; break
print("# 절 CM23.4 수박 K 1 튐 원인 -- 묶음 wmkick (exe/stab_wmkick.sh)\n")
print(__doc__.split("\n", 2)[2])
if base is None:
    print("(기본 판 아직 없음)"); sys.exit(0)

# 결과 불변
b0, s0 = load("base0")
if b0:
    print("## 0. 결과 불변 (비결정 판이라 해시 대신 프레임별 오차 %, 진단 판 · 진단 없음)\n")
    print("| 프레임 | " + " | ".join(str(i + 1) for i in range(min(len(base), len(b0)))) + " |")
    print("|---|" + "---|" * min(len(base), len(b0)))
    print("| 진단 판 | " + " | ".join(f"{100 * r['err']:.4f}" for r in base[:len(b0)]) + " |")
    print("| 진단 없음 | " + " | ".join(f"{100 * r['err']:.4f}" for r in b0[:len(base)]) + " |\n")

# 1. 기본 판 프레임 3~6 뉴턴 반복별 힘 몫
print(f"## 1. 기본 판 (2 x 5) 프레임 3~6 뉴턴 반복별 위쪽 힘 합 (N) -- 몸 전체 / 바닥 근처 (걸음 시작 노드 높이 < {fm(base[0].get('fr', {}).get('band_m'), '{:.4f}')} m = 3 칸)\n")
print("반복 i = 뉴턴 i 의 기울기 위치 (i = 끝 = 마지막 뉴턴 뒤 다시 잰 값). 장벽 표본점: 활성 / 걸림 (활성 & d < d̂) · 에너지 J · 최소 d/d̂ (활성) · 최소 d/d̂ (모든 표본점).\n")
print("| 프레임 | 반복 | 몸 전체: 탄성 · 장벽 · 관성 · 중력 · 잔차 g | 바닥 근처: 탄성 · 장벽 · 관성 · 중력 · 잔차 g | 근처 탄성 |·| 합 | 장벽 표본점 활성 / 걸림 · E · d/d̂ 최소 (활성 · 전체) | ‖g‖/F |")
print("|---|---|---|---|---|---|---|")
for r in base:
    t = r["t"] + 1
    if t < 3 or t > 6 or "fr" not in r or "it" not in r["fr"]:
        continue
    nr = r.get("nr", {})
    gF = [q.get("gF") for q in nr.get("it", [])] + nr.get("gend", [None])
    for q in r["fr"]["it"]:
        A, N, B = q["all"], q["near"], q.get("barq")
        bq = "-" if B is None else f"{B['nact']} / {B['hit']} · {B['E']:.3g} · {fm(B['dmin_dh'])} · {fm(B['dmin_all_dh'])}"
        it = "끝" if q["end"] else str(q["it"] + 1) + ("" if q["ran"] else " (안 돎)")
        g_ = gF[q["it"]] if q["it"] < len(gF) else None
        print(f"| {t} | {it} | {A['el']:.4g} · {A['bar']:.4g} · {A['inert']:.4g} · {A['grav']:.4g} · {A['res']:.4g} | {N['el']:.4g} · {N['bar']:.4g} · {N['inert']:.4g} · {N['grav']:.4g} · {N['res']:.4g} | {N['el_abs']:.4g} | {bq} | {fm(g_)} |")
print()

# 2. 속도 변화 분해
print("## 2. 걸음 (= 프레임) 속도 변화 분해, 위쪽 m/s (질량 가중 노드 평균) -- 기본 판 프레임 1~8\n")
print("| 프레임 | 영역 | v 시작 → 끝 | 측정 Δv | = 가속도 기억 | + 중력 | + 탄성 | + 장벽 | + 잔차 | 대조 차 | 끝 가속도 a1 (m/s²) = 중력 + 탄성 + 장벽 + 잔차 몫 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in base:
    t = r["t"] + 1
    if t > 8 or "fr" not in r or "dv" not in r["fr"]:
        continue
    for k, nm in (("all", "몸 전체"), ("near", "바닥 근처")):
        d = r["fr"]["dv"].get(k)
        if d is None:
            continue
        g = d["grav"] / (r["fr"]["gm"] * r["fr"]["dt"])
        print(f"| {t} | {nm} | {d['v0']:.3f} → {d['v1']:.3f} | {d['meas']:.3f} | {d['mem']:.3f} | {d['grav']:.3f} | {d['el']:.3f} | {d['bar']:.3f} | {d['res']:.3f} | {d['chk']:.1e} | {d['a1']:.4g} = {g:.3g} + {d['a1_el']:.4g} + {d['a1_bar']:.4g} + {d['a1_res']:.4g} |")
print()

# 3. 프레임 끝 det F·탄성 에너지
print("## 3. 프레임 끝 입자 det F 분포·SNH 탄성 에너지 (같은 식으로 잰 정답 F 와) -- 기본 판\n")
print("| 프레임 | 우리 J 최소 · 1 % · 5 % · 중앙 · 95 % · 99 % · 최대 | J < 0.95 · < 0.9 · > 1.05 몫 | 정답 J 같은 순서 | 정답 몫 | 탄성 에너지 J 우리 · 정답 | 입자 위쪽 속도 m/s | 운동 에너지 J |")
print("|---|---|---|---|---|---|---|---|")
for r in base:
    t = r["t"] + 1
    if t > 8 or "fr" not in r or "p" not in r["fr"]:
        continue
    p, g = r["fr"]["p"], r["fr"].get("gt")
    js = lambda J: "-" if J is None else " · ".join(f"{J[i]:.4f}" for i in (0, 1, 2, 4, 6, 7, 8))
    gfr = f"{g['J_lt095']:.3f} · {g['J_lt09']:.3f} · {g['J_gt105']:.3f}" if g else "-"
    gE = fm(g["E"] if g else None, "{:.4g}")
    print(f"| {t} | {js(p['J'])} | {p['J_lt095']:.3f} · {p['J_lt09']:.3f} · {p['J_gt105']:.3f} | {js(g['J']) if g else '-'} | {gfr} | {p['E']:.4g} · {gE} | {p['vz']:.3f} | {p['KE']:.4g} |")
print()

# 4. 대조
print("## 4. 대조 판 (프레임 1~15)\n")
print("| 판 | 결과 | 오차 % f4 · f5 · f8 · f12 · f15 | 튐 m (프레임) | 프레임 5 위쪽 속도 m/s (질량중심 차 · 입자) | f4 Δv 분해 몸 전체 (기억 · 중력 · 탄성 · 장벽 · 잔차) | f5 Δv 분해 | f4 장벽 걸림 표본점 · 최소 d/d̂ (끝) |")
print("|---|---|---|---|---|---|---|---|")
for n, nm in VAR:
    rows, s = load(n)
    if rows is None:
        print(f"| {nm} | 아직 없음 | | | | | | |"); continue
    rows = rows[:15]
    res = "진행 중" if s is None else (f"발산 f{s['diverged']}" if s.get("diverged") else f"완주 {min(s['frames_done'], 15)}")
    er = {r["t"] + 1: r["err"] for r in rows}
    zs = [r.get("com", {}).get("z") for r in rows]
    b, bat = bounce(zs)
    v5 = (zs[4] - zs[3]) / H if (len(zs) > 4 and zs[4] is not None and zs[3] is not None and H) else None
    fr = {r["t"] + 1: r.get("fr") for r in rows}
    vp = fr.get(5, {}).get("p", {}).get("vz") if fr.get(5) else None
    def dvs(f):
        d = (fr.get(f) or {}).get("dv", {}).get("all")
        return "-" if not d else f"{d['mem']:.2f} · {d['grav']:.2f} · {d['el']:.2f} · {d['bar']:.2f} · {d['res']:.2f} (Δv {d['meas']:.2f})"
    e4 = (fr.get(4) or {}).get("it", [])
    bq = e4[-1].get("barq") if e4 else None
    hit = max([q["barq"]["hit"] for q in e4 if q.get("barq")], default=None)
    print(f"| {nm} | {res} | {' · '.join(fm(100 * er[f] if f in er else None, '{:.1f}') for f in (4, 5, 8, 12, 15))} | {b:.3f} (f{bat}) | {fm(v5, '{:.2f}')} · {fm(vp, '{:.2f}')} | {dvs(4)} | {dvs(5)} | {fm(hit)} · {fm(bq['dmin_dh'] if bq else None)} |")
gz = [r.get("com", {}).get("gz") for r in base[:15]]
gb, gat = bounce(gz)
gv5 = (gz[4] - gz[3]) / H if (len(gz) > 4 and H) else None
print(f"| 정답 | - | - | {gb:.3f} (f{gat}) | {fm(gv5, '{:.2f}')} | - | - | - |")
print(f"\nh (프레임 시간) = {fm(H, '{:.5f}')} s")
