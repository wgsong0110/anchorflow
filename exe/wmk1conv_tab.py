"""절 CM23.3 묶음 wmk1conv 표: 수박 K 1 반복 상한 2x5 · 4x10 · 8x20 · 16x40 (허용치 εN 1e-2 · εP 1e-1 그대로).
사용: python exe/wmk1conv_tab.py <출력 디렉토리>
뉴턴 잔차 = ‖g‖/F (F = 걸음 시작점 탄성·접촉 힘 몫 크기, 허용치 규칙 |g| <= εN F 와 같은 척도). g_i = 뉴턴 반복 i 시작 (= 반복 i-1 끝) 의 값, 끝 = 마지막 뉴턴 뒤.
튐 = 질량중심 z 가 그때까지의 최저점보다 올라간 높이의 최대 (m).
"""
import json
import os
import sys

O = sys.argv[1]
V = [("N2P5", 2, 5), ("N4P10", 4, 10), ("N8P20", 8, 20), ("N16P40", 16, 40)]
TOL = 1e-2


def load(n):
    p = os.path.join(O, n, "fem.jsonl")
    if not os.path.exists(p):
        return None, None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    sp = os.path.join(O, n, "summary.json")
    s = json.load(open(sp)) if os.path.exists(sp) else None
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


def res(s, rows):
    if s is None:
        return f"진행 중 ({len(rows)} 프레임)" if rows else "-"
    return f"발산 f{s['diverged']}" if s.get("diverged") else f"완주 {s['frames_done']}"


def gstr(r):
    nr = r.get("nr")
    if not nr:
        return "-", "-", "-", "-", "-", "-"
    it = nr["it"]
    g = [x["gF"] for x in it if x["gF"] is not None]
    nN = sum(1 for x in it if x["ran"])
    nP = [x.get("nP") for x in it if x["ran"]]
    cp = sum(x.get("capP", 0) for x in it if x["ran"])
    st = [x.get("st") for x in it if x["ran"]]
    return (" → ".join(f"{q:.2g}" for q in g), f"{nr['gend'][0]:.3g}", ("수렴" if nr["conv"][0] else "**미수렴**"),
            f"{nN} ({'·'.join(str(q) for q in nP)})", f"{cp}/{nN}", "·".join(f"{q:.3g}" for q in st))


print("# 절 CM23.3 수박 K 1 수렴 -- 묶음 wmk1conv (exe/stab_wmk1conv.sh)\n")
print("같은 명령 (절 CM23 기본 판 K 1: 장벽 바닥 + --lj_interp --fp_interp, SNH, --gq_mass, 셀 1.5, 허용치 εN 1e-2 · εP 1e-1) 에서 반복 상한 (뉴턴 x PCG) 만 바꿈. "
      "진단 판 = + --nrec (결과 불변) --qjd --jdiag --bar_cnt, 비결정. ‖g‖/F = 뉴턴 잔차 (F = 걸음 시작 탄성·접촉 힘 몫; 수렴 = 허용치에서 멈췄거나 마지막 뉴턴 뒤 ‖g‖/F ≤ 1e-2). "
      "튐 = 질량중심 z 가 그때까지 최저점보다 올라간 높이 최대.\n")
sm = os.path.join(O, "smoke", "inv.txt")
if os.path.exists(sm):
    print("결과 불변 확인 (6 프레임): " + " / ".join(l.strip() for l in open(sm) if l.strip()) + "\n")

print("## 1. 판별 요약\n")
print("| 상한 뉴턴 x PCG | 진단 판 결과 | 평균 오차 % | 오차 % f4 · f8 · f12 · f20 | 튐 m (프레임) | 정답 튐 m | 미수렴 프레임 / 끝난 프레임 | 뉴턴 평균 · 최대 | PCG 상한 도달 몫 | 단독 판 결과 · 오차 % | 단독 ms 평균 · 최대 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for nm, n, c in V:
    rows, s = load("d_" + nm)
    zr, zs = load("z_" + nm)
    if rows is None:
        print(f"| {n} x {c} | - |  |  |  |  |  |  |  | {res(zs, zr) if zr else '-'} |  |")
        continue
    e = [100 * r["err"] for r in rows]
    ef = lambda k: fm(e[k - 1], "{:.1f}") if len(e) >= k else "-"
    zz = [r.get("com", {}).get("z") for r in rows]; gz = [r.get("com", {}).get("gz") for r in rows]
    b, bat = bounce(zz); gb, _ = bounce(gz)
    nrs = [r["nr"] for r in rows if r.get("nr")]
    ncv = sum(1 for q in nrs if not q["conv"][0])
    nN = [sum(1 for x in q["it"] if x["ran"]) for q in nrs]
    ran = [x for q in nrs for x in q["it"] if x["ran"]]
    cpf = (sum(x.get("capP", 0) for x in ran) / len(ran)) if ran else None
    zt = "-"
    if zs:
        zt = f"{res(zs, zr)} · {100 * zs['err_mean_done']:.2f} · {fm(zs.get('ms_mean'), '{:.1f}')} · {fm(max([q['ms'] for q in zr[2:]] or [None]) if zr else None, '{:.1f}')}"
    ze = zt.split(" · ")
    print(f"| {n} x {c} | {res(s, rows)} | {fm(s['err_mean_done'] * 100 if s else sum(e) / len(e), '{:.2f}')} | {ef(4)} · {ef(8)} · {ef(12)} · {ef(20)} | {b:.4f} (f{bat}) | {gb:.4f} | {ncv} / {len(nrs)} | "
          f"{(sum(nN) / len(nN)) if nN else 0:.2f} · {max(nN) if nN else '-'} | {fm(cpf, '{:.2f}')} | {' · '.join(ze[:2]) if zs else '-'} | {' · '.join(ze[2:]) if zs else '-'} |")

for nm, n, c in V:
    rows, s = load("d_" + nm)
    if not rows:
        continue
    nmax = 20
    print(f"\n## 2. {n} x {c} 프레임별 (1~{min(nmax, len(rows))})\n")
    print("| 프레임 | 오차 % | 질량중심 z 우리 · 정답 | 뉴턴 시작 ‖g‖/F → … (반복 i 시작) | 마지막 뉴턴 뒤 ‖g‖/F | 판정 | 뉴턴 수 (반복별 PCG) | PCG 상한 도달 / 뉴턴 | 보폭 (반복별) | 바닥 장벽 표본점 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows[:nmax]:
        g, ge, cv, nn, cp, st = gstr(r)
        cm = r.get("com", {})
        print(f"| {r['t'] + 1} | {100 * r['err']:.2f} | {fm(cm.get('z'), '{:.4f}')} · {fm(cm.get('gz'), '{:.4f}')} | {g} | {ge} | {cv} | {nn} | {cp} | {st} | {r.get('bc', {}).get('hit', '-')} |")
