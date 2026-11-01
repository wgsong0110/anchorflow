"""topo_results.tex 끝에 붙일 절 "FEM 과 같은 표현으로 다시 (2026-10-31)" 를 만든다 (기존 내용은 건드리지 않고 \\end{document} 앞에 넣음).
  python exe/topo_fem31_tex.py --dir <결과> --tex writing/anchorflow/topo_results.tex
"""
import argparse
import subprocess
import sys
import os

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--tex", required=True)
a = ap.parse_args()
out = subprocess.run([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "topo_fem31_table.py"),
                      "--dir", a.dir], capture_output=True, text=True, check=True).stdout
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
\section*{FEM 과 같은 표현으로 다시 (2026-10-31)}
위 결과는 그대로 두고 새로 돌렸다 (커밋 bee476b, \texttt{exe/topo\_fem31\_main.sh}, 결과 R2 \texttt{topo/fem31/main}).

\textbf{우리 표현 = 명시적 FEM 과 같은 것} (\texttt{rep\_track2.py --method fem}).
세계 고정 Kuhn 사면체 격자, node\_h $=$ 그 씬 MPM $dx\cdot(2\sqrt2)^{1/3}$ (시뮬 좌표; wolf $dx=0.01$, bread $2/150$, watermelon $2/300$,
ficus $2/50$, hotdog $2/100$), 격자 정수점의 원점 = 시뮬 좌표 원점. 매 프레임 직전 위치에 다시 깔고, 입자 위치는 든 사면체의 barycentric 보간,
$F_p \leftarrow A_T F_p$ (명시적 FEM 의 G2P 와 같은 식). Gregory·방사형 없음.
기존 ``Tet'' 과 보간 식은 같고 다른 점은 둘: node\_h 가 모든 씬 $0.028284$ 로 고정돼 있었고 (씬 규칙과 맞는 것은 hotdog 뿐),
격자 원점이 정규화 좌표 원점이었다. 명시적 FEM 격자 (\texttt{fem\_rt.Grid}) 와 대조 (\texttt{exe/topo\_fem\_equiv.py}):
다섯 씬 모두 활성 노드 수가 같고 사면체 꼭짓점 위치 차 $\le 4.4\times10^{-16}$, barycentric 차 $\le 2.1\times10^{-14}$.

\textbf{입자 $J>0$ 규칙 (모든 표현에 똑같이).} 최적화기는 위와 같은 riem ($\varepsilon=0.01\lambda_{\max}(G)$, CG 50, Armijo 되돌림 30,
첫 프레임 400·이후 200 반복) 이고, 계량 $G$ 의 잔차 = 표현 고유 제약의 로그 장벽 (우리·사면체 메쉬: 셀 $\det$, 그리드: CFL $|u_n|<dx$)
$+$ 입자 누적 $\det F$ 로그 장벽 $\sqrt{1/N}\log\det F_p$ (\texttt{--riem\_partF}). 선탐색은 가능성 검사를 한다 (\texttt{--feas}):
모든 입자의 누적 $\det F_p>0$ (보고 지표와 같은 float64 스칼라 곱) 과 표현 고유 제약을 모두 만족하는 보폭만 받고,
30 번 되돌려도 없으면 그 반복은 걷지 않는다. bread 사면체 메쉬는 앞의 각주대로 감쇠 고정판 (\texttt{--riem\_damp fix --riem\_eps0 0.02603914}, 같은 값).
목표 궤적: wolf·bread·watermelon 은 v2 목표 파일, ficus·hotdog 는 위와 같은 파일. 가우시안은 공식 필터 뒤 전부 (부분표본 없음).

\textbf{지표.} RMSE·CD 는 위와 같이 정지 경계상자 대각선 $L$ 로 나눔, 평균 $|x-x_{gt}|/L$ 은 명시적 FEM 과 같은 지표 ($L$ = EXT).
$\det F$ 최소와 뒤집힘 최대는 전 프레임·전 입자. ``멈춘 반복'' = 가능성 검사로 걷지 않은 반복 수. 자유도는 프레임 평균.

\noindent RMSE·CD·평균오차 ($|x-x_{gt}|/L$ 평균)·뒤집힘 최대는 \%, 멈춤 = 가능성 검사로 걷지 않은 반복 수.
\begin{center}\small
\begin{tabular}{llrrrrrrr}
\toprule
씬 & 표현 & RMSE & CD & 평균오차 & $\det F$ 최소 & 뒤집힘 & 자유도 & 멈춤 \\
\midrule
""" + summ.strip() + r"""
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
mark = "\\section*{FEM 과 같은 표현으로 다시 (2026-10-31)}"
if mark in T:                                   # 다시 만들 때는 이 절만 바꾼다
    T = T[:T.index(mark) - 1] + "\n\\end{document}\n"
obs = ""
if os.path.exists(a.tex + ".fem31_obs"):
    obs = open(a.tex + ".fem31_obs").read()
SEC = SEC.replace("%FEM31_OBS", obs)
i = T.rindex("\\end{document}")
T = T[:i].rstrip() + "\n" + SEC + "\n\\end{document}\n"
open(a.tex, "w").write(T)
print("절 넣음")
