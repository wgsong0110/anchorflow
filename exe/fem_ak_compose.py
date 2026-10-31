"""절 AK 영상 합성 (로컬, R2 마운트): 정답 | 확정 설정 (절 AD/AF, 클램프 σ1/σ3 ≤ 5) | 절 AI 새 판 (A1+A2+B1, 클램프 끔) -- exe/fem_ex_compose.py (칸마다 ms·fps·오차, % 는 textfile + expansion=none).
확정 칸 ms·오차: L000·ficus 는 절 AD 최종 판 summary, 홀드아웃은 같은 확정 설정 이번 단독 측정 (fem_ai_lego/<t>_a0b0_v_c5_t1) summary (영상은 절 AF 회차 1 렌더 -- 같은 설정).
새 판 칸: 영상 = fem_ak_<형상>/<씬>_pqf3[_z]_F1 (f32 덤프 렌더), ms·오차 = 깨끗한 단독 측정 _t1 summary. 사용: python exe/fem_ak_compose.py L000|<홀드아웃 0000 에서 0019>|ficus (fem_ai 단독 측정이 없으면 af_<t>_c267_t1)"""
import os
import subprocess
import sys

L = os.environ.get("AI_L", os.path.expanduser("~/r2/result/anchorflow/lbours"))
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
here = os.path.dirname(os.path.abspath(__file__))
sc = sys.argv[1]
NEW = "절 AK 새 판: 입자 적분점 셀 탄성 + 원래 F 갱신 + 상한 없음 · 띠 0 바닥 · 클램프 없음"
if sc == "ficus":
    title = "ficus F_pg_v0 125 프레임 · A6000 단독 · 명시적 lag · 공통 h_s 6.25e-5 (N640) · 확정 설정 | 절 AK (클램프 없이)"
    rows = [("정답 (PG)", "-", f"{L}/fem3_ficus/gt_ficus.mp4"),
            ("확정 설정 (절 AD): 클램프 σ1/σ3 ≤ 5", f"{L}/fem_ad_ficus/f_f_vc05_n640_F1/summary.json", f"{L}/fem_ad_ficus/f_f_vc05_n640_F1/vj/fem_ficus.mp4"),
            (NEW, f"{L}/fem_ak_ficus/ficus_pqf3_t1/summary.json", f"{L}/fem_ak_ficus/ficus_pqf3_F1/vj/fem_ficus.mp4")]
    shape = "ficus"
elif sc == "L000":
    title = "lego L000 80 프레임 · A6000 단독 · 명시적 lag · 확정 설정 (절 AD, N256) | 절 AK (공통 h_s, N267, 클램프 없이)"
    rows = [("정답 (MPM)", "-", f"{L}/fem2_lego/gt_lego.mp4"),
            ("확정 설정 (절 AD): 클램프 σ1/σ3 ≤ 5", f"{L}/fem_ad_lego/f_l_vc05_n256_F1/summary.json", f"{L}/fem_ad_lego/f_l_vc05_n256_F1/vj/fem_lego.mp4"),
            (NEW, f"{L}/fem_ak_lego/L000_pqf3_z_t1/summary.json", f"{L}/fem_ak_lego/L000_pqf3_z_F1/vj/fem_lego.mp4")]
    shape = "lego"
else:
    title = f"lego 홀드아웃 {sc} 80 프레임 · A6000 단독 · 명시적 lag · 공통 h_s 6.25e-5 (N267) · 확정 설정 | 절 AK (클램프 없이)"
    cs = f"{L}/fem_ai_lego/{sc}_a0b0_v_c5_t1/summary.json"
    if not os.path.exists(cs):                                                      # 0016·0018 등 절 AI 단독 측정이 없는 궤적: 같은 확정 설정의 절 AF 단독 측정 (af_<t>_c267_t1)
        cs = f"{L}/fem_af_lego/af_{sc}_c267_t1/summary.json"
    rows = [("정답 (MPM)", "-", f"{L}/fem_af_lego/gt_{sc}/gt_lego.mp4"),
            ("확정 설정 (절 AF): 클램프 σ1/σ3 ≤ 5", cs, f"{L}/fem_af_lego/af_{sc}_c267_r1/vj/fem_lego.mp4"),
            (NEW, f"{L}/fem_ak_lego/{sc}_pqf3_z_t1/summary.json", f"{L}/fem_ak_lego/{sc}_pqf3_z_F1/vj/fem_lego.mp4")]
    shape = "lego"
cells = [f"{nm}|{sj}|{v}" for nm, sj, v in rows]
o = f"{OUT}/fem_ak_{'lego' + sc if shape == 'lego' else 'ficus'}_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o, title, *cells], check=True)
print("[저장]", o)
