"""topo_results.tex 끝에 붙일 절 "공정 판: 감쇠 L2 기준·수렴 기준 (2026-11-01)" 를 만든다 (기존 내용은 건드리지 않고 \\end{document} 앞에 넣음).
  python exe/topo_fem31_tex.py --dir <결과> --tex writing/anchorflow/topo_results.tex
"""
import argparse
import subprocess
import sys
import os

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--tex", required=True)
ap.add_argument("--missing", default="(실행 중)")
a = ap.parse_args()
out = subprocess.run([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "topo_fem31_table.py"),
                      "--dir", a.dir, "--missing", a.missing], capture_output=True, text=True, check=True).stdout
summ, fr = out.split("%% 프레임별 RMSE")
summ = summ.replace("%% 요약 행\n", "")
COLS = [1, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 180]
frows = []
prev = None
for ln in fr.strip().splitlines():
    s, name, *cells = [c.strip() for c in ln.rstrip("\\ ").split("&")]
    d = dict(c.split(": ") for c in cells)
    if prev is not None and s != prev:
        frows.append("\\midrule")
    prev = s
    frows.append(f"{s} & {name} & " + " & ".join(d.get(str(t), "") for t in COLS) + " \\\\")
SEC = r"""
\section*{공정 판: 감쇠 L2 기준·수렴 기준 (2026-11-01)}
앞 절의 원인 가르기 결과 (오차는 표현이 아니라 반복 예산 안의 보폭) 에 따라 최적화기 규칙 둘을 바꿔 세 표현에 똑같이 다시 돌렸다
(커밋 e81a1cc, \texttt{exe/topo\_fem31\_main2.sh}, 결과 R2 \texttt{topo/fem31/main2}). 앞 절 결과는 그대로 둔다.
(1) 감쇠 $\varepsilon = k\,\lambda_{\max}(H_{L2})$: 입자 L2 의 가우스-뉴턴 몫만으로 정한다 (장벽 몫 제외, 계량 $G$ 자체는 그대로; \texttt{--riem\_eps\_ref l2}).
(2) 반복 수 고정 대신 프레임별 수렴 기준: 최근 10 반복의 상대 L2 감소 $(L2_{k-10}-L2_k)/L2_{k-10}\le 10^{-2}$ 이면 그 프레임을 멈춘다 (최소 20, 최대 2000).
$k$ 는 세 표현 모두에 프레임 1 L2 곡선 진단 (bread·ficus, $k$ 0.01·0.03·0.1·0.3·1, 600 반복; R2 \texttt{topo/fem31/kdiag}) 으로 골랐다: 400 반복 기준 최적이 우리 0.3 (ficus 는 1 이 근소하게 낫다), 사면체 메쉬 0.3 (ficus 0.1), 그리드는 $k$ 에 둔감 (CFL 계량이 지배) 이라 세 표현 같은 $k=0.3$.
입자 $\det F$ 로그 장벽·가능성 검사·씬 규칙 node\_h·목표 파일은 앞 절과 같다. bread 사면체 메쉬도 같은 규칙이다 (앞 절의 감쇠 고정판이 아님).
30 프레임 확인 ($k=0.1$, \texttt{exe/topo\_fem31\_fix\_t2.sh}): RMSE wolf 우리 0.415 \% (사면체 메쉬 4.415, 그리드 7.141; 셋 다 되돌린 판 1.106), bread 우리 0.083 \% (3.021, 1.484; 되돌린 판 0.454).

\noindent RMSE·CD·평균오차 ($|x-x_{gt}|/L$ 평균)·뒤집힘 최대는 \%, 멈춤 = 가능성 검사로 걷지 않은 반복 수.
\begin{center}\small
\begin{tabular}{llrrrrrrr}
\toprule
씬 & 표현 & RMSE & CD & 평균오차 & $\det F$ 최소 & 뒤집힘 & 자유도 & 멈춤 \\
\midrule
""" + summ.strip().replace("$^\\ddagger$", "") + r"""
\bottomrule
\end{tabular}
\end{center}

\noindent 프레임별 RMSE (\%):
\begin{center}\scriptsize
\begin{tabular}{ll""" + "r" * len(COLS) + r"""}
\toprule
씬 & 표현 & """ + " & ".join(str(t) for t in COLS) + r""" \\
\midrule
""" + "\n".join(frows) + r"""
\bottomrule
\end{tabular}
\end{center}
%FEM31_OBS
"""
T = open(a.tex).read()
mark = "\\section*{공정 판: 감쇠 L2 기준·수렴 기준 (2026-11-01)}"
if mark in T:                                   # 다시 만들 때는 이 절만 바꾼다
    T = T[:T.index(mark) - 1] + "\n\\end{document}\n"
obs = ""
if os.path.exists(a.tex + ".fem31b_obs"):
    obs = open(a.tex + ".fem31b_obs").read()
SEC = SEC.replace("%FEM31_OBS", obs)
i = T.rindex("\\end{document}")
T = T[:i].rstrip() + "\n" + SEC + "\n\\end{document}\n"
open(a.tex, "w").write(T)
print("절 넣음")
