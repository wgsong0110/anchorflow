"""절 CM13 표 (수박 J 장벽 / Stable Neo-Hookean, 묶음 wmJ): python exe/fem_wmj_table.py <wmJ 디렉토리> <cm9 기준 디렉토리> [기준 입자 det json]
판 디렉토리: j_a·j_b (판 (1) --elx jbar), s_a·s_b (판 (2) --elx snh), zj·zs (단독 시간), 기준 = cm9 w1a_wm·w1b_wm (--gq_mass 켬, 장벽·확장 없음).
기준 입자 det json (선택) = 기준 덤프 pred.pt 에서 잰 프레임별 입자 F det 최소·≤ 0 수 (vast_wmj.sh 가 만듦)."""
import json
import math
import os
import sys

B, C = sys.argv[1], sys.argv[2]
BD = json.load(open(sys.argv[3])) if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else {}


def rows(d):
    p = os.path.join(d, "fem.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for ln in open(p):
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def summ(d):
    p = os.path.join(d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, k=2):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f"{x:.{k}f}" if isinstance(x, float) else str(x))


def g(x):
    return "-" if x is None else f"{x:.2e}"


RUNS = [("기준 w1a (cm9)", os.path.join(C, "w1a_wm"), "w1a"), ("기준 w1b (cm9)", os.path.join(C, "w1b_wm"), "w1b"),
        ("(1) J 장벽 a", os.path.join(B, "j_a"), None), ("(1) J 장벽 b", os.path.join(B, "j_b"), None),
        ("(2) SNH a", os.path.join(B, "s_a"), None), ("(2) SNH b", os.path.join(B, "s_b"), None)]
FR = list(range(1, 11)) + list(range(20, 101, 10))
print("## 수박 판 (셀 1.5, cm9 w1a 명령 그대로 = 절 AS 전체 + --gq_mass, 비결정 -- 같은 명령 두 번)")
print("| 판 | 끝난 프레임 | 발산 프레임 | 평균 오차 % (끝난 프레임) | 시험 X J ≤ 0 점-회 합 (q_inv) · 최대 프레임 | 걸음 시작 뒤집힌 점-회 합 (ls_inv) | 프레임 끝 F_q det ≤ 0 점 최대 | 입자 F det 최소 (프레임) | 입자 det ≤ 0 수 최대 | 평균 ms (비측정) |")
print("|---|---|---|---|---|---|---|---|---|---|")
CUR = {}
for nm, d, bk in RUNS:
    r, s = rows(d), summ(d)
    CUR[nm] = r
    if not r and not s:
        print(f"| {nm} | 도는 중 / 없음 | | | | | | | | |")
        continue
    nd = s["frames_done"] if s else len(r)
    dv = s.get("diverged") if s else None
    em = 100 * s["err_mean_done"] if s else None
    qi = [q.get("q_inv", 0) for q in r]
    ls = sum((q.get("ls_inv") or [0, 0, 0, 0])[2] for q in r)
    jd = [q["jd"] for q in r if "jd" in q]
    if jd:
        qle = max(q["q_le0"] for q in jd)
        pm = [(q["p_min"], i + 1) for i, q in enumerate(jd) if q["p_min"] is not None]
        pmin = min(pm) if pm else (None, None)
        ple = max(q["p_le0"] for q in jd)
    elif bk and BD.get(bk):
        qle = "-"
        pm = [(v, i + 1) for i, v in enumerate(BD[bk]["p_min"]) if v is not None]
        pmin = min(pm) if pm else (None, None)
        ple = max(BD[bk]["p_le0"])
    else:
        qle, pmin, ple = "-", (None, None), "-"
    ms = [q["ms"] for q in r[2:] if q.get("ms") is not None and math.isfinite(q["ms"])]
    pms = g(pmin[0]) + (" (f%d)" % pmin[1] if pmin[1] else "")
    print(f"| {nm} | {nd} | {'**x%d**' % dv if dv else '완주'} | {f(em)} | {sum(qi)} · {max(qi) if qi else '-'} | {ls} | {qle} | {pms} | {ple} | {f(sum(ms) / len(ms), 1) if ms else '-'} |")
print()
print("프레임별 오차 % (1..10, 그 뒤 10 프레임마다; nan = 그 프레임 발산):")
for nm, d, _ in RUNS:
    r = CUR.get(nm) or []
    if not r:
        continue
    e = {q["t"] + 1: q["err"] for q in r}
    print(f"- {nm}: " + " ".join(f"f{k} {100 * e[k]:.2f}" if (k in e and e[k] == e[k]) else (f"f{k} nan" if k in e else "") for k in FR if k in e))
print()
print("프레임별 시험 X J ≤ 0 점-회 (q_inv) / 프레임 끝 F_q det ≤ 0 점 / 입자 F det 최소 (1..10, 그 뒤 10 프레임마다; 기준 판은 F_q 열 없음, 입자 det 은 덤프 pred.pt 에서):")
for nm, d, bk in RUNS:
    r = CUR.get(nm) or []
    if not r:
        continue
    it = []
    for q in r:
        k = q["t"] + 1
        if k not in FR and k != len(r):
            continue
        if "jd" in q:
            it.append(f"f{k} {q.get('q_inv', 0)}/{q['jd']['q_le0']}/{g(q['jd']['p_min'])}")
        elif bk and BD.get(bk) and k - 1 < len(BD[bk]["p_min"]):
            it.append(f"f{k} {q.get('q_inv', 0)}/-/{g(BD[bk]['p_min'][k - 1])}")
        else:
            it.append(f"f{k} {q.get('q_inv', 0)}/-/-")
    print(f"- {nm}: " + " ".join(it))
print()
print("발산 프레임 --nan_step 진단 (프레임 안 처음 비유한 개수: nN 노드 변위, nF 노드 기울기, nV 노드 속도·가속도, nJ 가우스 점 시험 J ≤ 0, nS 응력 비유한, nQ 점 상태 비유한, gsc PCG 스칼라; jmax 걸음 사이 최대 증분 칸, nj1 1 칸 넘는 노드):")
for nm, d, _ in RUNS[2:]:
    r = CUR.get(nm) or []
    s = summ(d)
    if not r or not s or not s.get("diverged"):
        continue
    nz = sorted(x for x in os.listdir(d) if x.startswith("nan_step_f"))
    last = [q for q in r if "nsd" in q][-3:]
    print(f"- {nm} (발산 {s['diverged']}): " + "; ".join(f"f{q['t'] + 1} {json.dumps(q['nsd'])}" for q in last) + f" 덤프 {' '.join(nz) or '없음'}")
print()
print("## 단독 시간 (판 하나만 그 카드에, 시작·실행 중 그 카드 다른 계산 프로세스 0; 재생 GPU 시간, 첫 2 프레임 뺌; 비결정이라 발산 프레임은 위와 다를 수 있음)")
print("| 판 | 끝난 프레임 | 발산 | 평균 · 최대 ms | 평균 오차 % | 실행 중 그 카드 다른 프로세스 |")
print("|---|---|---|---|---|---|")
for nm, v in (("(1) J 장벽 zj", "zj"), ("(2) SNH zs", "zs")):
    d = os.path.join(B, v)
    r, s = rows(d), summ(d)
    if not r:
        print(f"| {nm} | 도는 중 / 없음 | | | | |")
        continue
    ms = [q["ms"] for q in r[2:] if q.get("ms") is not None and math.isfinite(q["ms"])]
    oth = open(os.path.join(d, "apps_during.txt")).read().strip() if os.path.exists(os.path.join(d, "apps_during.txt")) else "?"
    print(f"| {nm} | {s['frames_done'] if s else len(r)} | {('x%d' % s['diverged']) if s and s.get('diverged') else ('완주' if s else '도는 중')} | {f(sum(ms) / len(ms), 1) if ms else '-'} · {f(max(ms), 1) if ms else '-'} | {f(100 * s['err_mean_done']) if s else '-'} | {oth} |")
print()
print("기준 단독 시간 (절 CM9 cm9z, 켬 판): zw1a 100.6 ms (x41) · zw1b 105.6 ms (x22).")
