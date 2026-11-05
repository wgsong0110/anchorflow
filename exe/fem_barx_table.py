"""절 CM10 표: python exe/fem_barx_table.py <묶음 디렉토리 barx> <절 CM8 판정 디렉토리 cm8j (b4_L000·b4_drop1·b5_wolf)>
판: o_<씬> = 채택 판 (B) 그대로 (이 인스턴스), d_<씬> = (B) + --bar_cnt 진단 (결과 불변이면 o 와 해시 같음), x_<씬> = (B) + --bar_cross (+ --bar_cnt)."""
import json
import os
import re
import sys

B, J = sys.argv[1], sys.argv[2]


def ld(d):
    p = os.path.join(d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fr(d):
    p = os.path.join(d, "fem.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def state(d):
    s = ld(d)
    if s is None:
        ri = os.path.join(d, "run_info.txt")
        t = open(ri).read() if os.path.exists(ri) else ""
        return ("실패 (" + t.split("rc=")[-1].split()[0] + ")") if "rc=" in t else ("없음" if not t else "도는 중")
    dv = s.get("diverged")
    e = 100 * (s["err_mean_done"] if dv else s["err_mean"])
    return ("**x%d** (%.2f)" % (dv, e)) if dv else "%.2f" % e


def drop(d):
    q = os.path.join(d, "drop_ana.txt")
    if not os.path.exists(q):
        return ""
    m = re.search(r"\n[^\n]*?:[^\n]*반발 계수 ([0-9.\-]+), 튀어 오름 ([0-9.\-]+) mm", "\n" + "\n".join(l for l in open(q).read().splitlines() if not l.startswith("mpm:")))
    return (" / 반발 %s · %s mm" % m.groups()) if m else ""


print("# 절 CM10 barx -- 채택 판 (B) (장벽 + --y0eb, 절 CM8 b 판 명령: HHT ρ 0.8, K 2, 뉴턴 2, εN 1e-2 · εP 1e-1, PCG 상한 lego·낙하 4 · wolf 5, 결정적) 에 관통 칸 장벽 --bar_cross")
print()
print("## 1. 장벽이 실제로 걸리는 표본점 (진단 --bar_cnt, 결과 불변)")
print("정의: 프레임 안 어느 걸음·어느 뉴턴 반복에서든 활성 (걸음 시작 d > 0) 이고 d < d̂ (= ½ dx_정답) 인 표본점 = 장벽 걸림. 전체 = 입자 있는 팔분 표본점. 칸 = 그 점들이 속한 B-스플라인 칸 수 (전체 = 점유 칸 수).")
print("관통 칸 = 어느 걸음 시작에서든 칸 꼭짓점 8 개의 평면 부호 거리가 양·음 둘 다인 칸 (절 CM10 정의), 그 칸의 표본점 수, 장벽 걸린 점 중 관통 칸 점.")
for v in ("d", "x"):
    for t in ("wolf", "L000", "drop1"):
        rows = [q for q in fr(os.path.join(B, f"{v}_{t}")) if q.get("bc")]
        if not rows:
            continue
        lab = {"d": "채택 판 (B) 그대로", "x": "(B) + --bar_cross"}[v]
        ne = [q["bc"]["neng"] for q in rows]
        rt = [100.0 * q["bc"]["neng"] / max(q["bc"]["nocc"], 1) for q in rows]
        ce = [q["bc"]["ceng"] for q in rows]
        print()
        print(f"### {t} -- {lab} ({v}_{t}, 프레임 {len(rows)})")
        print(f"- 장벽 걸린 표본점: 프레임 평균 {sum(ne) / len(ne):.1f} · 최대 {max(ne)} (프레임 {ne.index(max(ne)) + 1}), 전체 대비 평균 {sum(rt) / len(rt):.3f} % · 최대 {max(rt):.3f} %, "
              f"걸린 프레임 {sum(1 for q in ne if q > 0)} / {len(ne)}; 칸 평균 {sum(ce) / len(ce):.1f} · 최대 {max(ce)}")
        print(f"- 전체 표본점 {rows[0]['bc']['nocc']} (프레임 1) ~ {rows[-1]['bc']['nocc']} (끝), 점유 칸 {rows[0]['bc']['nc']} ~ {rows[-1]['bc']['nc']}")
        print()
        print("| 프레임 | 걸린 점 | 전체 점 | 비율 % | 걸린 칸 | 점유 칸 | 관통 칸 | 관통 칸 점 | 걸린 점 중 관통 칸 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for q in rows:
            b = q["bc"]
            print(f"| {q['t'] + 1} | {b['neng']} | {b['nocc']} | {100.0 * b['neng'] / max(b['nocc'], 1):.3f} | {b['ceng']} | {b['nc']} | {b['ccr']} | {b['ncr']} | {b['engcr']} |")
print()
print("## 2. 결과 불변 확인 (끝 위치 해시)")
print("| 씬 | 절 CM8 cm8j (클러스터, aa5a7ef) | o (이 인스턴스, 새 코드 플래그 없음) | d (+ --bar_cnt) |")
print("|---|---|---|---|")
for t, jn in (("L000", "b4_L000"), ("wolf", "b5_wolf"), ("drop1", "b4_drop1")):
    h = lambda d: (ld(d) or {}).get("x_sha1", "")
    print(f"| {t} | {h(os.path.join(J, jn))} | {h(os.path.join(B, 'o_' + t))} | {h(os.path.join(B, 'd_' + t))} |")
print()
print("## 3. 판정 (결정적; 칸 = 평균 오차 % 또는 x발산 프레임 (발산 전 평균); 낙하 정답 반발 0.750 / 83.4 mm)")
print("| 씬 | (B) cm8j | (B) o | (B) + --bar_cross x |")
print("|---|---|---|---|")
for t, jn in (("L000", "b4_L000"), ("drop1", "b4_drop1"), ("wolf", "b5_wolf")):
    cs = []
    for d in (os.path.join(J, jn), os.path.join(B, "o_" + t), os.path.join(B, "x_" + t)):
        c = state(d)
        if t == "drop1":
            c += drop(d)
        cs.append(c)
    print(f"| {t} | " + " | ".join(cs) + " |")
print()
print("## 4. wolf 높이 곡선 (바닥 법선 z, 질량 가중 질량 중심 · 최대; 바닥 z 0.48)")
H = {}
for nm, d in (("cm8j", os.path.join(J, "b5_wolf")), ("o", os.path.join(B, "o_wolf")), ("x", os.path.join(B, "x_wolf"))):
    H[nm] = {q["t"] + 1: q for q in fr(d) if q.get("hp") and q.get("hg")}
FR_ = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
print("| 프레임 | 정답 질량중심 | (B) cm8j | (B) o | x | 정답 최대 | (B) cm8j | (B) o | x |")
print("|---|---|---|---|---|---|---|---|---|")
src = H["x"] or H["o"] or H["cm8j"]
for f in FR_:
    if f not in src:
        continue
    g = src[f]["hg"]
    row = [f"{g[0]:.3f}"] + [f"{H[n][f]['hp'][0]:.3f}" if f in H[n] else "" for n in ("cm8j", "o", "x")] + [f"{g[1]:.3f}"] + [f"{H[n][f]['hp'][1]:.3f}" if f in H[n] else "" for n in ("cm8j", "o", "x")]
    print(f"| {f} | " + " | ".join(row) + " |")
print()
print("## 5. 프레임당 반복 (뉴턴 합 N · PCG 합 P 평균, 최대 P)")
for v in ("o", "x"):
    for t in ("L000", "drop1", "wolf"):
        s = ld(os.path.join(B, f"{v}_{t}"))
        if s and "it_nN" in s:
            print(f"- {v}_{t}: N {s['it_nN']['mean']:.1f} · P {s['it_nP']['mean']:.1f} (최대 {s['it_nP']['max']})")
