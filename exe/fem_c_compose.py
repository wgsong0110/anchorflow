"""절 C 영상 합성 (로컬, R2 마운트): 정답 | 절 Z 최고 (클램프, 최적화 전) | 절 C 최적화 판 -- exe/fem_ex_compose.py (칸마다 ms·fps·오차, % 는 textfile + expansion=none).
사용: python exe/fem_c_compose.py lego|ficus <최적화 태그> [<최적화 태그 2>]      (최종 태그는 RENDER_DONE 이 있는 첫 _F 시도)"""
import glob
import json
import os
import subprocess
import sys

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
here = os.path.dirname(os.path.abspath(__file__))
shape = sys.argv[1]
tags = sys.argv[2:]


def fin(root, tag):
    for d in sorted(glob.glob(f"{L}/{root}/{tag}_F*")):
        if os.path.exists(f"{d}/vj/RENDER_DONE"):
            return d
    raise SystemExit(f"렌더 없음: {root}/{tag}")


def lab(d):
    a = json.load(open(f"{d}/summary.json"))["args"]
    return f"절 C 최적화 · 클램프 σ1/σ3 ≤ {a['lim_sr']:g} · N{a['N']}"


if shape == "lego":
    title = ("lego L000 80 프레임 · A6000 단독 · 명시적 lag (셀 집중 질량 m_T/4) · 특이값 비율 클램프 -- "
             "절 C 속도 최적화 (바닥 접촉 판정 생략·클램프 판정 목록·점유 사면체 목록, 같은 식)")
    rows = [("정답 (MPM)", None, f"{L}/fem2_lego/gt_lego.mp4"),
            ("절 Z 최고 (최적화 전) · σ1/σ3 ≤ 5 · N256", fin("fem_z_lego", "f_l_c5_n256"), None)]
else:
    title = ("ficus F_pg_v0 125 프레임 · A6000 단독 · 명시적 lag (lego 와 같은 코드, 바닥 없음) · 특이값 비율 클램프 -- 절 C 속도 최적화")
    rows = [("정답 (PG)", None, f"{L}/fem3_ficus/gt_ficus.mp4"),
            ("절 Z 최고 (최적화 전) · σ1/σ3 ≤ 2 · N640", fin("fem_z_ficus", "f_f_c2_n640"), None)]
for t in tags:
    d = fin(f"fem_c_{shape}", t)
    rows.append((lab(d), d, None))
cells = [f"{nm}|-|{v}" if d is None else f"{nm}|{d}/summary.json|{d}/vj/fem_{shape}.mp4" for nm, d, v in rows]
o = f"{OUT}/fem_c_{shape}_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o, title, *cells], check=True)
print("[저장]", o)
