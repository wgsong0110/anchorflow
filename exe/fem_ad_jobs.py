"""절 AD 스윕 (비측정 병렬, GPU 3·4): 희소 셀 처리 변형 x 클램프 (σ1/σ3 ≤ 5 켬/끔).
lego L000 80 프레임 N 192·256 x 5 회, ficus F_pg_v0 125 프레임 N 640 x 2 회. 변형 목록은 환경변수 AD_V (기본 전부), AD_SET=2 면 2 차 목록.
줄 형식: <EX_ROOT> <태그> <fem_ex_sim 인자...>"""
import os

LB = "--mode lag --opt --mass tet --no_dump --floor vel --rstat"
FD = "/home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --mass tet --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat"
CHK = "--sp_chk 10,30,60"
V1 = [("ref", ""), ("mg1t", f"--sp merge --sp_a tet --sp_k 1 {CHK}"), ("mg2t", f"--sp merge --sp_a tet --sp_k 2 {CHK}"), ("cb2t", f"--sp cube --sp_a tet --sp_k 2 {CHK}"),
      ("mg2h1", f"--sp merge --sp_a avg --sp_hg 1 --sp_k 2 {CHK}"), ("bl2", f"--sp blend --sp_k 2 {CHK}"), ("st2", f"--sp stiff --sp_k 2 {CHK}"),
      ("ml01", f"--sp mlow --sp_mlow 0.1 {CHK}")]
V = V1
if os.environ.get("AD_SET") == "2":
    V = [eval(q) for q in os.environ["AD_V2"].split(";")] if os.environ.get("AD_V2") else []
CL = [("c5", "--lim_sr 5"), ("c0", "")]
J = []
for r in range(1, 6):
    for N in (192, 256):
        for nm, ar in V:
            for cn, ca in CL:
                J.append(f"fem_ad_lego ad_l_{nm}_{cn}_n{N}_r{r} {LB} {ar} {ca} --N {N}")
    if r <= 2:
        for nm, ar in V:
            for cn, ca in CL:
                J.append(f"fem_ad_ficus ad_f_{nm}_{cn}_n640_r{r} {FB} {ar.replace(CHK, '--sp_chk 10,60,120') if ar else ''} {ca} --N 640")
for j in J:
    print(" ".join(j.split()))
