"""절 AE 영상 합성 (로컬): 정답 | 절 AB 최고 | 절 AD 최고 | 절 AE 새 최고 -- exe/fem_ex_compose.py (칸마다 ms·fps·오차, % 는 textfile + expansion=none).
R2 원본을 rclone 으로 /tmp/fem_ae_src 에 받아 쓴다 (마운트 캐시 지연 회피). 사용: python exe/fem_ae_compose.py lego|ficus <새 판 최종 태그 (예 f_l_nb025vc05_c5_F1)>"""
import os
import subprocess
import sys

R = "r2:storage/result/anchorflow/lbours"
SRC = "/tmp/fem_ae_src"
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
here = os.path.dirname(os.path.abspath(__file__))
shape, tag = sys.argv[1], sys.argv[2]


def get(rel, files):
    d = f"{SRC}/{rel}"
    os.makedirs(d, exist_ok=True)
    for f in files:
        if not os.path.exists(f"{d}/{f}"):
            subprocess.run(["rclone", "copyto", f"{R}/{rel}/{f}", f"{d}/{f}"], check=True)
    return d


def cell(nm, rel):
    d = get(rel, ["summary.json", f"vj/fem_{shape}.mp4"])
    return f"{nm}|{d}/summary.json|{d}/vj/fem_{shape}.mp4"


if shape == "lego":
    title = ("lego L000 80 프레임 · A6000 단독 · 명시적 lag (셀 집중 질량 m_T/4, 프레임당 P2G 1 회 → 메시 서브스텝 256 → G2P 1 회) · "
             "절 AB 매끈한 F 갱신 | 절 AD 탄성 부피 상한 | 절 AE 둘을 합침")
    g = get("fem2_lego", ["gt_lego.mp4"])
    cells = [f"정답 (MPM)|-|{g}/gt_lego.mp4",
             cell("절 AB: F 갱신 노드 평균 A β1 + 클램프 3", "fem_ab_lego/f_l_navg1c3_n256_F3"),
             cell("절 AD: 탄성 부피 상한 θ0.5 + 클램프 5", "fem_ad_lego/f_l_vc05_n256_F1"),
             cell("절 AE: F 갱신 β0.25 + 상한 θ0.5 + 클램프 5", f"fem_ae_lego/{tag}")]
else:
    title = ("ficus F_pg_v0 125 프레임 · A6000 단독 · 명시적 lag (lego 와 같은 코드, 바닥 없음) · "
             "절 AB 매끈한 F 갱신 | 절 AD 탄성 부피 상한 | 절 AE 둘을 합침")
    g = get("fem3_ficus", ["gt_ficus.mp4"])
    cells = [f"정답 (PG)|-|{g}/gt_ficus.mp4",
             cell("절 AB: F 갱신 노드 평균 A β1 + 클램프 3", "fem_ab_ficus/f_f_navg1c3_n640_F1"),
             cell("절 AD: 탄성 부피 상한 θ0.5 + 클램프 5", "fem_ad_ficus/f_f_vc05_n640_F1"),
             cell("절 AE: F 갱신 β0.25 + 상한 θ0.5 + 클램프 5", f"fem_ae_ficus/{tag}")]
o = f"{OUT}/fem_ae_{shape}_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o, title, *cells], check=True)
print("[저장]", o)
