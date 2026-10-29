"""절 Z 영상 합성 (로컬, R2 마운트): 정답 | 절 R 최고 (det A ≥ 0.3 상자 제한) | 절 Z 새 판들 -- exe/fem_ex_compose.py (칸마다 ms·fps·오차, % 는 textfile + expansion=none).
사용: python exe/fem_z_compose.py lego|ficus      (최종 태그는 RENDER_DONE 이 있는 첫 _F 시도)"""
import glob
import os
import subprocess
import sys

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
here = os.path.dirname(os.path.abspath(__file__))
shape = sys.argv[1]


def fin(root, tag):
    for d in sorted(glob.glob(f"{L}/{root}/{tag}_F*")):
        if os.path.exists(f"{d}/vj/RENDER_DONE"):
            return d
    raise SystemExit(f"렌더 없음: {root}/{tag}")


if shape == "lego":
    title = ("lego L000 80 프레임 · A6000 단독 · 명시적 lag (셀 집중 질량 m_T/4, 프레임당 P2G 1 회 → 메시 서브스텝 N → G2P 1 회) · "
             "노드 이동 제한: 절 R det A_T ≥ 0.3 | 절 Z 특이값 비율 σ1/σ3(A_T) ≤ c 클램프")
    rows = [("정답 (MPM)", None, f"{L}/fem2_lego/gt_lego.mp4"),
            ("절 R 최고: det A ≥ 0.3 · N192", f"{L}/fem_r_lego/f_l_b03_n192_F1", None),
            ("절 Z 클램프 σ1/σ3 ≤ 5 · N256", fin("fem_z_lego", "f_l_c5_n256"), None),
            ("절 Z 클램프 σ1/σ3 ≤ 2 · N192", fin("fem_z_lego", "f_l_c2_n192"), None)]
else:
    title = ("ficus F_pg_v0 125 프레임 · A6000 단독 · 명시적 lag (lego 와 같은 코드, 바닥 없음) · "
             "절 R det A_T ≥ 0.3 | 절 Z σ1/σ3(A_T) ≤ 2 클램프 | 절 Z 비율 로그 장벽")
    rows = [("정답 (PG)", None, f"{L}/fem3_ficus/gt_ficus.mp4"),
            ("절 R 최고: det A ≥ 0.3 · N640", f"{L}/fem_r_ficus/f_f_b03_n640_F1", None),
            ("절 Z 클램프 σ1/σ3 ≤ 2 · N640", fin("fem_z_ficus", "f_f_c2_n640"), None),
            ("절 Z 장벽 κ1 c2 · N640", fin("fem_z_ficus", "f_f_sb1c2_n640"), None)]
cells = [f"{nm}|-|{v}" if d is None else f"{nm}|{d}/summary.json|{d}/vj/fem_{shape}.mp4" for nm, d, v in rows]
o = f"{OUT}/fem_z_{shape}_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o, title, *cells], check=True)
print("[저장]", o)
