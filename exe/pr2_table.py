"""[paper_remeasure2] 설정 격자 x 3 회 중앙값 표 (LaTeX): python exe/pr2_table.py <all.json (pr2_collect 출력)> <출력 tex>
오차 % = 롤아웃 평균 |x−x_gt|/EXT 3 회 중앙값 (최소–최대), ms = 재생 GPU 시간 3 회 중앙값 (최소–최대). 발산 = 3 회 모두 발산."""
import json
import sys

G = json.load(open(sys.argv[1]))
NM = dict(ours="우리 (현재 기본 판)", mpm="정답 MPM (PhysGaussian)", vrgs="VR-GS", bnv="명시적 B-스플라인", phystwin="PhysTwin", simplicits="Simplicits", springgaus="Spring-Gaus")
SC = [("ficus", "ficus"), ("wolf", "wolf"), ("bread", "bread"), ("vasedeck", "vasedeck"), ("plane", "plane"), ("pillow2sofa", "pillow2sofa")]
ORD = list(NM)
T = ["\\begin{center}\\footnotesize\\begin{tabular}{@{}llrrrr@{}}", "\\toprule", "장면 & 방법 & 설정 $N$ & 오차 \\% (최소--최대) & ms (최소--최대) & 완주/반복 \\\\ \\midrule"]
for j, (s, sn) in enumerate(SC):
    q = [g for g in G if g["scene"] == s]
    if not q:
        continue
    if j:
        T.append("\\midrule")
    for m in ORD:
        for g in sorted([g for g in q if g["method"] == m], key=lambda g: g["N"]):
            if g["err"] is None:
                e = t = ("발산" if any(d_ is not None for d_ in g.get("div", [])) else "실행 실패")
            else:
                e = "%.3g (%.3g--%.3g)" % (g["err"], min(g["err_all"]), max(g["err_all"]))
                t = "%.3g (%.3g--%.3g)" % (g["ms"], min(g["ms_all"]), max(g["ms_all"]))
            T.append(f"{sn} & {NM[m]} & {g['N']} & {e} & {t} & {g['ok']}/{g['reps']} \\\\")
T += ["\\bottomrule", "\\end{tabular}\\end{center}"]
open(sys.argv[2], "w").write("\n".join(T) + "\n")
print("저장", sys.argv[2])
