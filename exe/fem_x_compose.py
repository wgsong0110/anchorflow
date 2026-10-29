"""절 X 영상 합성 (lego): 정답 | 절 R 최고 (셀 안 입자 m_T/4, 제한 box 0.3, N192) | 절 X 새 판 (Σλm, 제한 box 0.3, N1024) -- exe/fem_ex_compose.py
(칸마다 ms·fps·오차, % 는 textfile + expansion=none). 로컬 (R2 마운트) 에서."""
import os
import subprocess
import sys

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
here = os.path.dirname(os.path.abspath(__file__))
title = ("lego L000 80 프레임 · A6000 단독 · 명시적 lag (프레임당 P2G 1 회 → 사면체 메시 서브스텝 N → G2P 1 회) · "
         "노드 질량: 절 R 셀 안 입자 m_T/4 | 절 X barycentric Σλm (적분점 = 셀 질량점)")
cells = [f"정답 (MPM)|-|{L}/fem2_lego/gt_lego.mp4",
         f"절 R 최고: m_T/4 · 제한 0.3 · N192|{L}/fem_r_lego/f_l_b03_n192_F1/summary.json|{L}/fem_r_lego/f_l_b03_n192_F1/vj/fem_lego.mp4",
         f"절 X: Σλm · 제한 0.3 · N1024|{L}/fem_x_lego/{sys.argv[1]}/summary.json|{L}/fem_x_lego/f_l_lL_n1024_F2/vj/fem_lego.mp4"]
o = f"{OUT}/fem_x_lego_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o, title, *cells], check=True)
print("[저장]", o)
