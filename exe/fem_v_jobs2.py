"""절 V 2 차 (β 0 순수 노드 적분이 lego 0/48 완주 -- 제한 끄면 프레임 16~24 에 사면체 뒤집힘 수백만 (영에너지 모드) 으로 발산) -> 문헌 안정화 선택지:
셀 적분 혼합 E = (1-β) Σ_n V_n ψ(F_n) + β Σ_T V_T ψ(F_T) (--nint_beta β). β 0.05·0.1·0.2·0.3·0.5 x 1 차와 같은 후보·N·반복 (exe/fem_v_jobs.py 와 같은 줄)."""
import subprocess
import sys
import os
here = os.path.dirname(os.path.abspath(__file__))
for bt in (0.05, 0.1, 0.2, 0.3, 0.5):
    sys.stdout.write(subprocess.run([sys.executable, f"{here}/fem_v_jobs.py", str(bt)], capture_output=True, text=True, check=True).stdout)
