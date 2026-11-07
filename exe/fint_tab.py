"""절 CM20 묶음 fint 표 (exe/stab_fint.sh): 입자 F 교체 (--fp_interp) 끔 | 켬 -- 레고 L000 판들·--gq_mass 함께·K 2 단독 시간·수박 SNH.
사용: python exe/fint_tab.py <출력 폴더> > fint_table.md"""
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


def refj(n):
    p = os.path.join(O, "ref", n, "fem.jsonl")
    return [json.loads(l_) for l_ in open(p)] if os.path.exists(p) else []


def res(s, fj):
    if s is None:
        return "도는 중 (%d 프레임)" % len(fj) if fj else "-"
    dv = s.get("diverged")
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    es = "%.2f" % (100 * e) if e is not None and math.isfinite(e) else "nan"
    return ("발산 %d · %s" % (dv, es)) if dv else ("**완주 · %s**" % es)


def fin_frames(fj):
    return [q for q in fj if q.get("err") is not None and isinstance(q.get("err"), float) and math.isfinite(q["err"])]


def stats(fj):
    """발산 전 (오차 유한) 프레임만: 가우스 점 J ≤ 0 (qj.nneg)·입자 det ≤ 0 (pd.nJneg 또는 jd.p_le0) 있는 프레임 수·최대 (프레임)·처음, 입자 det 최소, 점 J 최소."""
    ff = fin_frames(fj)
    qn = [(q["t"] + 1, q["qj"]["nneg"]) for q in ff if q.get("qj")]
    pn = [(q["t"] + 1, (q["pd"]["nJneg"] if q.get("pd") else q["jd"]["p_le0"])) for q in ff if q.get("pd") or q.get("jd")]
    pm = [(q["pd"]["Jmin"] if q.get("pd") else q["jd"]["p_min"]) for q in ff if q.get("pd") or q.get("jd")]
    qm = [q["qj"].get("Jmin") for q in ff if q.get("qj") and q["qj"].get("Jmin") is not None]

    def cnt(L):
        nz = [(f, v) for f, v in L if v > 0]
        if not L:
            return "-"
        if not nz:
            return "0"
        f, v = max(nz, key=lambda z: z[1])
        return "%d 프레임 · 최대 %d (f%d) · 처음 f%d" % (len(nz), v, f, nz[0][0])
    pm = [q for q in pm if q is not None and math.isfinite(q)]
    return dict(q=cnt(qn), p=cnt(pn), pmin=("%.3g" % min(pm)) if pm else "-", qmin=("%.3g" % min(qm)) if qm else "-", qn=qn, pn=pn)


def hsame(fj, rj):
    a = [q.get("xh") for q in fj]
    b = [q.get("xh") for q in rj]
    if not a or not b:
        return "-"
    n = min(len(a), len(b))
    same = sum(1 for i in range(n) if a[i] == b[i] and a[i] is not None)
    first = next((i + 1 for i in range(n) if a[i] != b[i]), None)
    return "%d / %d%s" % (same, n, "" if first is None else " (처음 다름 f%d)" % first)


print("# 절 CM20 입자 F 교체 (--fp_interp) 끔 | 켬 -- 묶음 fint (exe/stab_fint.sh)\n")
print("오차 = 끝난 프레임 평균 %; 발산 = 그 프레임 비유한·메모리 부족. 가우스 점 J = 프레임 끝 U 로 X_q = F_q + Σ U ⊗ G_qn 의 det (부피 > 0 점, --qjd), 입자 det = 프레임 끝 (G2P·교체 뒤) 입자 F 의 det (--pdiag; 수박은 --jdiag). "
      "통계는 발산 전 (오차 유한) 프레임만. '끔 해시 = 기록' = 프레임별 입자 위치 해시가 절 S.11 묶음 gqm 같은 노드 실행 (끔 = <판>_off_w, --gq_mass = <판>_on_w) 과 같은 프레임 수.\n")
ROWS = [("(B) 장벽 K 2 (채택)", "K2", "off", "on", "K2_off_w"), ("(B) 장벽 K 4", "K4", "off", "on", "K4_off_w"), ("(B) 장벽 K 8", "K8", "off", "on", "K8_off_w"),
        ("(B) 장벽 K 16", "K16", "off", "on", "K16_off_w"), ("(A) 원래 고착", "fa", "off", "on", "fa_off_w"), ("노드 −z 클램프", "nc", "off", "on", "nc_off_w"),
        ("−z 저항 강 (k 9)", "vs", "off", "on", "vs_off_w"), ("(B) K 2 + --gq_mass", "K2", "gq", "gqon", "K2_on_w"), ("(B) K 4 + --gq_mass", "K4", "gq", "gqon", "K4_on_w")]
print("## 1. 레고 판별 (모두 결정적 --det)\n")
print("| 판 | 끔: 결과 · % | 켬: 결과 · % | 끔 해시 = 기록 | 뒤집힌 가우스 점 (J ≤ 0) 끔 | 켬 | 뒤집힌 입자 (det F ≤ 0) 끔 | 켬 | 입자 det F 최소 끔 → 켬 | 점 J 최소 끔 → 켬 |")
print("|---|---|---|---|---|---|---|---|---|---|")
PF = []
for nm, r, m0, m1, ref in ROWS:
    s0, f0 = rd("%s_%s" % (r, m0)); s1, f1 = rd("%s_%s" % (r, m1))
    a, b = stats(f0), stats(f1)
    print("| %s | %s | %s | %s | %s | %s | %s | %s | %s → %s | %s → %s |" % (nm, res(s0, f0), res(s1, f1), hsame(f0, refj(ref)), a["q"], b["q"], a["p"], b["p"], a["pmin"], b["pmin"], a["qmin"], b["qmin"]))
    PF.append((nm, m0, a)); PF.append((nm, m1, b))
print("\n## 2. 프레임별 뒤집힌 가우스 점 / 입자 수 (0 아닌 프레임만, 'f프레임 점/입자')\n")
for nm, m, st in PF:
    dq, dp = dict(st["qn"]), dict(st["pn"])
    fr = sorted(set(f for f, v in st["qn"] if v > 0) | set(f for f, v in st["pn"] if v > 0))
    txt = " ".join("f%d %d/%d" % (f, dq.get(f, 0), dp.get(f, 0)) for f in fr)
    print("- %s %s: %s" % (nm, {"off": "끔", "on": "켬", "gq": "끔 (gq_mass)", "gqon": "켬 (gq_mass)"}[m], txt if txt else ("없음" if st["qn"] else "-")))
print("\n## 3. 레고 (B) K 2 단독 시간 (비결정 = 판정 명령에서 --det·덤프 뺌, 진단 없음; 카드 하나에 차례로, 끔·켬 번갈아 두 번; 프레임당 재생 GPU 시간, 첫 2 프레임 뺌)\n")
print("| 판 | ms 평균 | ms 최대 | 결과 · % | 실행 중 그 카드 계산 프로세스 수 최대 (자기 포함) | 노드 전체 최대 |")
print("|---|---|---|---|---|---|")
for n in ("z_off_a", "z_on_a", "z_off_b", "z_on_b"):
    s, fj = rd(n)
    pc = os.path.join(O, n, "gpu_procs.txt")
    cm = nm_ = "-"
    if os.path.exists(pc):
        L = [l_.split() for l_ in open(pc) if l_.strip()]
        try:
            cm = max(int(q[2]) for q in L); nm_ = max(int(q[4]) for q in L)
        except Exception:
            pass
    print("| %s | %s | %s | %s | %s | %s |" % (n, "%.2f" % s["ms_mean"] if s and s.get("ms_mean") else "-", "%.2f" % s["ms_max"] if s and s.get("ms_max") else "-", res(s, fj), cm, nm_))
print("\n## 4. 수박 SNH (절 CM13 s_a 명령 + --qjd 진단; 비결정이라 두 번씩)\n")
print("| 판 | 결과 · % | 뒤집힌 가우스 점 (프레임 끝 X_q J ≤ 0) | 커밋 F_q det ≤ 0 점 최대 | 뒤집힌 입자 (det ≤ 0) | 입자 det 최소 | 응력 비유한 점-회 합 | 재생 ms 평균 |")
print("|---|---|---|---|---|---|---|---|")
for n in ("wm_off_a", "wm_off_b", "wm_on_a", "wm_on_b"):
    s, fj = rd(n)
    st = stats(fj)
    ff = fin_frames(fj)
    qd = [q["jd"]["q_le0"] for q in ff if q.get("jd")]
    qnf = sum(int(q.get("q_nf") or 0) for q in fj)
    print("| %s | %s | %s | %s | %s | %s | %d | %s |" % (n, res(s, fj), st["q"], max(qd) if qd else "-", st["p"], st["pmin"], qnf, "%.1f" % s["ms_mean"] if s and s.get("ms_mean") else "-"))
for m in ("off", "on"):
    p = os.path.join(O, "wm_vid_%s.txt" % m)
    if os.path.exists(p):
        print("\n영상 %s 칸 = %s" % ({"off": "끔", "on": "켬"}[m], open(p).read().strip()), end="")
print("\n\n## 5. 수박 프레임별 뒤집힌 가우스 점 / 입자 수 (0 아닌 프레임만)\n")
for n in ("wm_off_a", "wm_off_b", "wm_on_a", "wm_on_b"):
    s, fj = rd(n)
    st = stats(fj)
    dq, dp = dict(st["qn"]), dict(st["pn"])
    fr = sorted(set(f for f, v in st["qn"] if v > 0) | set(f for f, v in st["pn"] if v > 0))
    print("- %s: %s" % (n, " ".join("f%d %d/%d" % (f, dq.get(f, 0), dp.get(f, 0)) for f in fr) or ("없음" if fj else "-")))
