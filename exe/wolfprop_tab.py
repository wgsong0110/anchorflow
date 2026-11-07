"""절 CM26 표: wolf 기본 판 (B) 과 (A) 의 --prec 진단 (prec.jsonl) -> 마크다운 (표준 출력).
사용: python exe/wolfprop_tab.py <묶음 출력 디렉토리> <wolf.pt> [<wolfwhy 디렉토리 (해시 대조)>]
판 디렉토리: B (기본 판 + --nrec --prec), A (가우스 점 고착 바닥 + --nrec --prec)."""
import json
import os
import sys

O = sys.argv[1]
WPT = sys.argv[2] if len(sys.argv) > 2 else None
WW = sys.argv[3] if len(sys.argv) > 3 else None
RUNS = [("B", "기본 판 (B) 장벽 바닥 + 보간 교체", "base"), ("A", "(A) 가우스 점 고착 바닥", "A_stick")]
REG = [("all", "몸 전체"), ("near1", "바닥 위 1 칸"), ("near3", "바닥 위 3 칸"), ("pin", "구속 노드 (k_구속 < ½)"), ("rest", "나머지 노드")]


def rd(p):
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def fm(v, d=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != 0 and (abs(v) >= 1e4 or abs(v) < 10 ** -d):
            return f"{v:.2e}"
        return f"{v:.{d}f}"
    return str(v)


P = {k: rd(os.path.join(O, k, "prec.jsonl")) for k, _, _ in RUNS}
F = {k: rd(os.path.join(O, k, "fem.jsonl")) for k, _, _ in RUNS}
print("# 절 CM26 wolf 기본 판 (장벽 바닥) 은 무엇으로 몸을 받치는가 -- 묶음 wolfprop (exe/stab_wolfprop.sh)\n")
print("모두 결정적 --det, 50 프레임, K 1, 뉴턴 2 x PCG 5 (절 CM25 base · A_stick 명령 + --nrec --prec 10:30). 프레임 f = 1 부터. 힘 N (법선 = 바닥 법선 = 위쪽 +), 속도 m/s.")
print("노드마다 정확히 Mi (y − Ũ) [관성 = m (a1 − g)] = 탄성·소성 (−f_el) + 장벽 (−G n) + 구속 반력 C (순차 방출 구속 노드) + 고착 반력 C (바닥 고착 노드) + 고정 노드 C + 잔차 T g;"
      " C = 날 잔차 R − T g = 구속·고착이 흡수한 힘. 몸 전체 탄성 합은 내력이라 ≈ 0 -> 정지한 몸이면 무게 (음수) 를 장벽·구속·고착·잔차가 든다.\n")

# 0. 결과 불변
print("## 0. 결과 불변 (--prec 켬 대 절 CM25 같은 명령 프레임 해시)")
for k, nm, ref in RUNS:
    h1 = [r.get("xh") for r in F[k]]
    h0 = [r.get("xh") for r in rd(os.path.join(WW, ref, "fem.jsonl"))] if WW else []
    same = sum(1 for a, b in zip(h0, h1) if a and a == b)
    print(f"- {nm} [{k}]: 프레임 해시 같음 {same} / {len(h1)} (절 CM25 {ref} {len(h0)} 프레임; 끝 {h0[-1] if h0 else '-'} · {h1[-1] if h1 else '-'})")
    errs = [r["f"] for r in P[k] if "err" in r]
    if errs:
        print(f"  - 진단 계산 실패 프레임 {errs}: {P[k][0].get('err', '')[-400:] if 'err' in P[k][0] else ''}")
print()

# 1. 순차 방출: 정답 config 와 우리 구속
print("## 1. 순차 방출 구속 -- 정답 config 와 우리 구현")
if WPT and os.path.exists(WPT):
    try:
        import torch
        d = torch.load(WPT, map_location="cpu", weights_only=False)
        cfg = d["cfg"]
        for bc in cfg.get("boundary_conditions") or []:
            print(f"- 정답 config 경계: {json.dumps(bc, ensure_ascii=False)}")
        print(f"- frame_dt {cfg.get('frame_dt')} · substep_dt {cfg.get('substep_dt')} · 격자 {cfg.get('n_grid')} (grid_lim {cfg.get('grid_lim')})")
        x0 = d["x"][0].double()
        print(f"- 처음 입자 범위 x {float(x0[:, 0].min()):.3f}~{float(x0[:, 0].max()):.3f}, y {float(x0[:, 1].min()):.3f}~{float(x0[:, 1].max()):.3f}, z {float(x0[:, 2].min()):.3f}~{float(x0[:, 2].max()):.3f}, 입자 {x0.shape[0]}")
    except Exception as ex:
        print(f"- config 읽기 실패 {ex}")
for k, nm, _ in RUNS:
    rows = [r for r in P[k] if "p" in r and r["p"]]
    if not rows:
        continue
    print(f"\n### {nm} [{k}] -- 프레임별 입자 무리 (묶임 = trel > 프레임 끝, 이번 = 이 프레임 안에 풀림, 풀림 = 프레임 시작 전에 풀림); 높이 = 바닥 위 (m), 내려감 = 처음 높이 − 지금 (평균)")
    print("| f | 정답 질량중심 · 우리 (hg·hp) | 묶임 수 · 묶인 x 범위 | 묶임 이동 우리 · 정답 (mm) | 풀림 수 | 풀림 내려감 우리 · 정답 (mm) | 이번 풀림 수 · 내려감 우리 · 정답 (mm) | 풀린 입자 중 구속 노드 가중 > ½ 수 · 평균 가중 | 구속 노드 수 (k<1 · <½) · 질량 kg |")
    print("|---|---|---|---|---|---|---|---|---|")
    fj = {r["t"] + 1: r for r in F[k]}
    for r in rows:
        p = r["p"]; f = r["f"]; b, n_, rl = p.get("bound", {}), p.get("now", {}), p.get("rel", {})
        hp = fj.get(f, {}).get("hp"); hg = fj.get(f, {}).get("hg")
        mm = lambda v: fm(v * 1000, 1) if v is not None else "-"
        bx = p.get("bound_x")
        print(f"| {f} | {fm(hg[0] if hg else None)} · {fm(hp[0] if hp else None)} | {b.get('n', 0)} · {fm(bx[0]) if bx else '-'}~{fm(bx[1]) if bx else '-'} | {mm(b.get('disp'))} · {mm(b.get('disp_gt'))} | {rl.get('n', 0)} | "
              f"{mm(rl.get('drop'))} · {mm(rl.get('drop_gt'))} | {n_.get('n', 0)} · {mm(n_.get('drop'))} · {mm(n_.get('drop_gt'))} | "
              f"{(rl.get('n_pinw_half', 0) or 0) + (n_.get('n_pinw_half', 0) or 0)} · {fm(rl.get('pinw_mean'))} | {r['nkind']['pin']} · {r['nkind']['pin_half']} · {fm(r['mkind']['pin'])} |")
    p = rows[0]["p"]
    print(f"\n처음 입자 x 범위 {p.get('all_x')}, 처음 바닥 위 높이 범위 {p.get('all_h0')}")
print()

# 2. 힘 몫 (걸음 끝)
for k, nm, _ in RUNS:
    rows = [r for r in P[k] if "it" in r]
    if not rows:
        continue
    print(f"## 2. 걸음 끝 노드 힘 몫 법선 합 (N) -- {nm} [{k}]")
    for rg, rn in REG:
        print(f"\n### {rn}")
        print("| f | 노드 · 질량 kg | 무게 m g | 관성 m(a1−g) | 탄성·소성 | 장벽 | 구속 반력 | 고착 반력 | 고정 노드 | 잔차 | 대조 | 구속 반력 접선 ‖·‖ 합 | 고착 반력 접선 합 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        acc = {}
        for r in rows:
            e = r["it"][-1][rg]
            print(f"| {r['f']} | {r['nnode'][rg]} · {fm(r['mass'][rg])} | {fm(e['grav'], 2)} | {fm(e['inert'], 2)} | {fm(e['el'], 2)} | {fm(e['bar'], 2)} | {fm(e['pinC'], 2)} | {fm(e['stkC'], 2)} | {fm(e['fixC'], 2)} | {fm(e['res'], 2)} | {fm(e['chk'], 2)} | {fm(e['pinC_tan'], 2)} | {fm(e['stkC_tan'], 2)} |")
            if 15 <= r["f"] <= 22:
                for q in ("grav", "inert", "el", "bar", "pinC", "stkC", "fixC", "res"):
                    acc.setdefault(q, []).append(e[q])
        if acc:
            av = {q: sum(v) / len(v) for q, v in acc.items()}
            print(f"| **f15~22 평균** | | {fm(av['grav'], 2)} | {fm(av['inert'], 2)} | {fm(av['el'], 2)} | {fm(av['bar'], 2)} | {fm(av['pinC'], 2)} | {fm(av['stkC'], 2)} | {fm(av['fixC'], 2)} | {fm(av['res'], 2)} | | | |")
    print()

# 3. 속도 변화 분해
for k, nm, _ in RUNS:
    rows = [r for r in P[k] if "dv" in r]
    if not rows:
        continue
    print(f"## 3. 걸음 속도 변화 분해 (질량 가중 평균 법선, m/s) -- {nm} [{k}]")
    for rg, rn in REG:
        print(f"\n### {rn}")
        print("| f | v0 → v1 | 측정 Δv | 기억 dt(1−γ)a0 | 중력 | 탄성·소성 | 장벽 | 구속 | 고착 | 고정 | 잔차 | 대조 | a1 식 · 저장 (지운 몫) m/s² |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            q = r["dv"].get(rg)
            if not q:
                continue
            print(f"| {r['f']} | {fm(q['v0'], 4)} → {fm(q['v1'], 4)} | {fm(q['meas'], 4)} | {fm(q['mem'], 4)} | {fm(q['grav'], 4)} | {fm(q['el'], 4)} | {fm(q['bar'], 4)} | {fm(q['pinC'], 4)} | {fm(q['stkC'], 4)} | {fm(q['fixC'], 4)} | {fm(q['res'], 4)} | {fm(q['chk'], 4)} | {fm(q['a1_formula'], 3)} · {fm(q['a1_stored'], 3)} ({fm(q['a1_wipe'], 3)}) |")
    print()

# 4. 바닥 근처: 장벽 거리·노드 법선 속도
for k, nm, _ in RUNS:
    rows = [r for r in P[k] if "it" in r]
    if not rows:
        continue
    print(f"## 4. 바닥 근처 -- {nm} [{k}]")
    print("장벽 표본점 d/d̂ (걸음 끝 y; 점유 표본점 = 입자 있는 팔분 점, 활성 = 걸음 시작 d > 0) 구간 수: <0, 0~1 (걸림), 1~2, 2~4, 4~8, ≥8; 바닥 위 3 칸 표본점 d/d̂ 분위 (0·1·5·25·50 %); 노드 법선 속도 v1 분위 (0·1·5·25·50·75·95·99·100 %)")
    print("| f | 걸린 점 | 활성 구간 수 | 점유 구간 수 | 3 칸 안 표본점 수 · d/d̂ 분위 | 1 칸 노드 수 · pm1 · pm0 · 고착 표시 | 1 칸 노드 v1 분위 | 3 칸 자유 노드 v1 분위 | 입자 바닥 3 칸 안 수 우리 · 정답 | 입자 높이 분위 우리 (0·1·5 %) · 정답 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        b = r.get("bar") or {}
        v1 = r.get("vn_near1", {}); v3 = r.get("vn_near3", {})
        p = r.get("p") or {}
        hp_, hg_ = p.get("h_pct") or [], p.get("h_pct_gt") or []
        rn_ = b.get("r_near") or []
        print(f"| {r['f']} | {b.get('hit', '-')} | {b.get('hist_act', '-')} | {b.get('hist_occ', '-')} | {b.get('n_near', '-')} · {[round(q, 2) for q in rn_[:5]] if rn_ else '-'} | "
              f"{r['nnode']['near1']} · {v1.get('n_pm1')} · {v1.get('n_pm0')} · {v1.get('n_an')} | {[round(q, 3) for q in (v1.get('v1') or [])]} | {[round(q, 3) for q in (v3.get('v1_free') or [])]} | "
              f"{p.get('n_low', '-')} · {p.get('n_low_gt', '-')} | {[round(q, 4) for q in hp_[:3]]} · {[round(q, 4) for q in hg_[:3]]} |")
    print()

# 5. 뉴턴 반복별 (f15·18·22, 몸 전체)
for k, nm, _ in RUNS:
    rows = [r for r in P[k] if "it" in r and r["f"] in (15, 18, 22)]
    if not rows:
        continue
    print(f"## 5. 뉴턴 반복별 힘 몫 (몸 전체·바닥 위 3 칸, N) -- {nm} [{k}]")
    print("| f | 반복 | 돌았나 | 영역 | 관성 | 탄성·소성 | 장벽 | 구속 | 고착 | 고정 | 잔차 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for e in r["it"]:
            for rg in ("all", "near3"):
                q = e[rg]
                print(f"| {r['f']} | {'끝' if e['end'] else e['it']} | {e['ran']} | {rg} | {fm(q['inert'], 2)} | {fm(q['el'], 2)} | {fm(q['bar'], 2)} | {fm(q['pinC'], 2)} | {fm(q['stkC'], 2)} | {fm(q['fixC'], 2)} | {fm(q['res'], 2)} |")
    print()
if P["B"] and "am" in P["B"][0]:
    r = P["B"][0]
    print(f"HHT 계수: αm {r['am']} · αf {r['af']} · γ {r['gm']} · τ² {r['tau2']:.3e}; 칸 {r['cell']:.4f} m; meff = m: {r['meff_eq']}")
