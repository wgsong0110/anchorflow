"""절 AA 영상 합성 (로컬, R2 마운트): 정답 | 절 R 최고 (det A ≥ 0.3 상자 제한) | 절 AA 새 판 (셀 에너지 2 차 근사) -- exe/fem_ex_compose.py (칸마다 ms·fps·오차, % 는 textfile + expansion=none).
사용: python exe/fem_aa_compose.py lego|ficus <새 판 태그> <새 판 이름>      (최종 태그는 RENDER_DONE 이 있는 첫 _F 시도)"""
import glob
import os
import subprocess
import sys

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
here = os.path.dirname(os.path.abspath(__file__))
shape, tag, name = sys.argv[1], sys.argv[2], sys.argv[3]


def fin(root, tag):
    for d in sorted(glob.glob(f"{L}/{root}/{tag}_F*")):
        if os.path.exists(f"{d}/vj/RENDER_DONE"):
            return d
    raise SystemExit(f"렌더 없음: {root}/{tag}")


if shape == "lego":
    title = ("lego L000 80 프레임 · A6000 단독 · 명시적 lag (셀 집중 질량 m_T/4, 프레임당 P2G 1 회 → 메시 서브스텝 N → G2P 1 회) · "
             "셀 에너지: 절 R ψ(A·셀 평균 F) | 절 AA 셀 안 입자 에너지 평균의 2 차 근사 (셀 평균 F + 공분산)")
    rows = [("정답 (MPM)", None, f"{L}/fem2_lego/gt_lego.mp4"),
            ("절 R 최고: det A ≥ 0.3 · N192", f"{L}/fem_r_lego/f_l_b03_n192_F1", None),
            (name, fin("fem_aa_lego", tag), None)]
else:
    title = ("ficus F_pg_v0 125 프레임 · A6000 단독 · 명시적 lag (lego 와 같은 코드, 바닥 없음) · "
             "셀 에너지: 절 R ψ(A·셀 평균 F) | 절 AA 셀 안 입자 에너지 평균의 2 차 근사")
    rows = [("정답 (PG)", None, f"{L}/fem3_ficus/gt_ficus.mp4"),
            ("절 R 최고: det A ≥ 0.3 · N640", f"{L}/fem_r_ficus/f_f_b03_n640_F1", None),
            (name, fin("fem_aa_ficus", tag), None)]
cells = [f"{nm}|-|{v}" if d is None else f"{nm}|{d}/summary.json|{d}/vj/fem_{shape}.mp4" for nm, d, v in rows]
o = f"{OUT}/fem_aa_{shape}_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o, title, *cells], check=True)
print("[저장]", o)
