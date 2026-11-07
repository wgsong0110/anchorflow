"""절 CM23 묶음 k1new 표 (exe/stab_k1new.sh): 새 기본 판 프레임당 1 걸음 (K 1 = 기본 판) 대 예전 걸음 수 (비교), 전 씬.
사용: python exe/k1new_tab.py <출력 폴더> > k1new_table.md"""
import json
import math
import os
import sys

O = sys.argv[1]


def rd(n):
    d = os.path.join(O, n)
    s = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else None
    fj = []
    p = os.path.join(d, "fem.jsonl")
    if os.path.exists(p):
        for l_ in open(p):
            try:
                fj.append(json.loads(l_))
            except Exception:
                pass
    return s, fj


def fin(fj):
    return [q for q in fj if isinstance(q.get("err"), float) and math.isfinite(q["err"])]


def res(s, fj):
    if s is None:
        return ("도는 중 (%d 프레임)" % len(fj)) if fj else "-", "-"
    dv = s.get("diverged")
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    es = "%.2f" % (100 * e) if e is not None and math.isfinite(e) else "nan"
    return ("발산 f%d" % dv) if dv else "완주 %d" % s.get("frames_done", 0), es


def flips(fj):
    ff = fin(fj)
    qn = [(q["t"] + 1, q["qj"]["nneg"] if q.get("qj") else q["jd"]["q_le0"]) for q in ff if q.get("qj") or q.get("jd")]
    pn = [(q["t"] + 1, q["pd"]["nJneg"] if q.get("pd") else q["jd"]["p_le0"]) for q in ff if q.get("pd") or q.get("jd")]

    def c(L):
        if not L:
            return "-"
        nz = [(f, v) for f, v in L if v > 0]
        if not nz:
            return "0"
        f, v = max(nz, key=lambda z: z[1])
        return "%d 프레임 · 최대 %d (f%d)" % (len(nz), v, f)
    return c(qn), c(pn)


def per(n):
    p = os.path.join(O, n, "period.json")
    if not os.path.exists(p):
        return "-"
    j = json.load(open(p))
    return "%.2f / %.2f" % (j["ratio"], j["amp_ratio"])


def tm(n):
    s, fj = rd(n)
    ms = [q["ms"] for q in fin(fj) if q.get("t", 0) >= 2 and q.get("ms") is not None]
    if not ms:
        return None
    pc = os.path.join(O, n, "gpu_procs.txt")
    cm = nm_ = "-"
    if os.path.exists(pc):
        L = [l_.split() for l_ in open(pc) if l_.strip()]
        try:
            cm = max(int(q[2]) for q in L); nm_ = max(int(q[4]) for q in L)
        except Exception:
            pass
    r, e = res(s, fj)
    return sum(ms) / len(ms), max(ms), cm, nm_, r, e


SC = [("레고", "L", 2, ["L_K1"], ["L_K2"]), ("ficus (바닥 없음)", "F", 2, ["F_K1"], ["F_K2"]), ("wolf", "W", 2, ["W_K1"], ["W_K2"]),
      ("bread (바닥 없음)", "B", 2, ["B_K1"], ["B_K2"]), ("수박 (셀 1.5)", "M", 3, ["M_K1_a", "M_K1_b"], ["M_K3_a", "M_K3_b"])]
print("# 절 CM23 새 기본 판 프레임당 1 걸음 (K 1 = 기본 판) -- 묶음 k1new (exe/stab_k1new.sh)\n")
print("새 기본 판 = 장벽 바닥 (B) + F·재질 상태 가우스 점 보간 교체 (--fp_interp); ficus·bread 는 바닥 없음 (보간 교체만); 수박 = 절 CM21 (3) (장벽 바닥 + --lj_interp --fp_interp, SNH, --gq_mass). "
      "반복 상한은 씬 기본 값 그대로 (뉴턴 x PCG: 레고 2x4 · ficus 2x24 · wolf 2x5 · bread 2x2 · 수박 2x5), K 만 다름. 판정 = 결정적 (수박만 비결정 a·b). "
      "오차 = 끝난 프레임 평균 %. 뒤집힌 가우스 점 = 프레임 끝 X_q J ≤ 0 (--qjd; 수박 K3 재사용 판은 --jdiag 커밋 F_q det ≤ 0), 뒤집힌 입자 = 프레임 끝 입자 det F ≤ 0; 발산 전 프레임만. "
      "ficus = 주기 비 / 진폭 비 (정답 대비).\n")
print("## 1. 판정 판\n")
print("| 씬 | 판 | 실행 | 결과 | 평균 오차 % | ficus 주기 비 / 진폭 비 | 뒤집힌 가우스 점 | 뒤집힌 입자 | 출처 |")
print("|---|---|---|---|---|---|---|---|---|")
for nm, s_, kd, k1, kc in SC:
    for lab, L in (("**K 1 (기본 판)**", k1), ("K %d (비교)" % kd, kc)):
        for n in L:
            s, fj = rd(n)
            r, e = res(s, fj)
            q, p = flips(fj)
            src = open(os.path.join(O, n, "reuse.txt")).read().strip() if os.path.exists(os.path.join(O, n, "reuse.txt")) else "이 묶음"
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (nm, lab, n[-1] if n[-2] == "_" else "-", r, e, per(n) if s_ == "F" else "", q, p, src))
rf = os.path.join(O, "ref", "L_K2_fint", "fem.jsonl")
if os.path.exists(rf):
    a = [q.get("xh") for q in rd("L_K2")[1]]
    b = [json.loads(l_).get("xh") for l_ in open(rf)]
    m = min(len(a), len(b))
    print("\n레고 K 2 이 묶음 대 절 CM20 fint K2_on 프레임 해시 같음: %d / %d (합친 가지 코드 확인)" % (sum(1 for i in range(m) if a[i] and a[i] == b[i]), m))
print("\n## 2. 단독 시간 (비결정, 진단·덤프 없음; 카드 하나에 차례로 -- 씬마다 K1 a · 비교 a · K1 b · 비교 b; 프레임당 재생 GPU 시간, 첫 2 프레임 뺌, 발산 판은 발산 전)\n")
print("| 씬 | 판 | 실행 | ms 평균 | ms 최대 | 결과 · 오차 % | 그 카드 계산 프로세스 최대 (자기 포함) | 노드 전체 최대 |")
print("|---|---|---|---|---|---|---|---|")
for nm, s_, kd, k1, kc in SC:
    for k, lab in ((1, "**K 1 (기본 판)**"), (kd, "K %d (비교)" % kd)):
        for r_ in "ab":
            n = "z_%s_K%d_%s" % (s_, k, r_)
            t = tm(n)
            print("| %s | %s | %s | %s | %s | %s | %s | %s |" % ((nm, lab, r_, "%.2f" % t[0], "%.2f" % t[1], "%s · %s" % (t[4], t[5]), t[2], t[3]) if t else (nm, lab, r_, "-", "-", "-", "-", "-")))
v = os.path.join(O, "vid", "dmgtop.json")
if os.path.exists(v):
    J = json.load(open(v))
    print("\n## 3. 수박 손상 상위 표시 (logJp, 그 프레임 자기 분포 백분위; 비유한 따로) -- 실행 %s\n" % open(os.path.join(O, "vid", "run.txt")).read().strip())
    print("| 프레임 | 정답 문턱 q90 · q80 · q70 | 정답 실제 몫 상위 10·20·30 % 칸 | 정답 비유한 | 우리 문턱 | 우리 실제 몫 | 우리 비유한 | 우리에 정답 문턱 적용 몫 |")
    print("|---|---|---|---|---|---|---|---|")
    for t in (1, 4, 5, 6, 10, 20, 40, 70, 100):
        g = next((r for r in J["gt"] if r["t"] == t), None); o = next((r for r in J["ours"] if r["t"] == t), None); x = next((r for r in J["ours_gtthr"] if r["t"] == t), None)
        if not (g and o and x):
            continue
        f3 = lambda L: " · ".join("%.3g" % z for z in L)
        p3 = lambda L: " · ".join("%.1f" % (100 * z) for z in L)
        print("| %d | %s | %s | %.1f %% | %s | %s | %.1f %% | %s |" % (t, f3(g["thr"]), p3(g["frac"]), 100 * g["nf"], f3(o["thr"]), p3(o["frac"]), 100 * o["nf"], p3(x["frac"])))
