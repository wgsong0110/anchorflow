"""절 AB 영상 합성 (로컬, R2 마운트): 정답 | 절 Z 최고 | 절 AB 새 최고 -- exe/fem_ex_compose.py (칸마다 ms·fps·오차, % 는 textfile + expansion=none).
사용: python exe/fem_ab_compose.py lego|ficus <새 판 태그 (최종 디렉토리 이름)> "<새 판 이름>" """
import os
import subprocess
import sys

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
here = os.path.dirname(os.path.abspath(__file__))
shape, tag, name = sys.argv[1], sys.argv[2], sys.argv[3]
new = f"{L}/fem_ab_{shape}/{tag}"
if not os.path.exists(f"{new}/vj/RENDER_DONE"):
    raise SystemExit(f"렌더 없음: {new}")
if shape == "lego":
    title = ("lego L000 80 프레임 · A6000 단독 · 명시적 lag (셀 집중 질량 m_T/4, 프레임당 P2G 1 회 → 메시 서브스텝 N → G2P 1 회) · "
             "G2P 입자 F 갱신: 절 Z 기존 F ← A_T F | 절 AB 노드 평균 속도 기울기 F ← Ã F")
    rows = [("정답 (MPM)", None, f"{L}/fem2_lego/gt_lego.mp4"),
            ("절 Z 최고: 클램프 σ1/σ3 ≤ 5 · N256 (기존 F 갱신)", f"{L}/fem_z_lego/f_l_c5_n256_F1", None),
            (name, new, None)]
else:
    title = ("ficus F_pg_v0 125 프레임 · A6000 단독 · 명시적 lag (lego 와 같은 코드, 바닥 없음) · "
             "G2P 입자 F 갱신: 절 Z 기존 F ← A_T F | 절 AB 노드 평균 속도 기울기 F ← Ã F")
    rows = [("정답 (PG)", None, f"{L}/fem3_ficus/gt_ficus.mp4"),
            ("절 Z 최고: 클램프 σ1/σ3 ≤ 2 · N640 (기존 F 갱신)", f"{L}/fem_z_ficus/f_f_c2_n640_F1", None),
            (name, new, None)]
cells = [f"{nm}|-|{v}" if d is None else f"{nm}|{d}/summary.json|{d}/vj/fem_{shape}.mp4" for nm, d, v in rows]
o = f"{OUT}/fem_ab_{shape}_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o, title, *cells], check=True)
print("[저장]", o)
