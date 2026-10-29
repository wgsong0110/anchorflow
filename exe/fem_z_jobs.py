"""절 Z 2 차 스윕 (비측정 병렬): A_T 특이값 비율 σ1/σ3 -- (1) 클램프 (비율만 / 비율 + det ≥ 0.3), (2) 로그 장벽 κ 1 (N 스윕), 기준 (제한 없음)·절 R det 0.3 상자 판을 같은 횟수로.
lego L000 80 프레임 x N 192·256 (장벽은 384 도) x 5 회, ficus F_pg_v0 125 프레임 N 640 (장벽은 512·640·768) x 2 회.
c 값은 환경변수 Z_C (기본 "1.5 2 3 5"). 줄 형식: <EX_ROOT> <태그> <fem_ex_sim 인자...>"""
import os

C = os.environ.get("Z_C", "1.5 2 3 5").split()
LB = "--mode lag --opt --mass tet --no_dump --floor vel --rstat"
FD = "/home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --mass tet --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat"
cn = lambda c: c.replace(".", "p")
J = []
for r in range(1, 6):
    for N in (192, 256):
        J.append(f"fem_z_lego z_l_ref_n{N}_r{r} {LB} --N {N}")
        J.append(f"fem_z_lego z_l_d03_n{N}_r{r} {LB} --lim 0.3 --lim_mode box --N {N}")
        for c in C:
            J.append(f"fem_z_lego z_l_c{cn(c)}_n{N}_r{r} {LB} --lim_sr {c} --lim_sr_chk --N {N}")
            J.append(f"fem_z_lego z_l_c{cn(c)}d03_n{N}_r{r} {LB} --lim_sr {c} --lim 0.3 --lim_sr_chk --N {N}")
    for N in (192, 256, 384):
        for c in C:
            J.append(f"fem_z_lego z_l_sb1c{cn(c)}_n{N}_r{r} {LB} --sb 1 --sb_c {c} --N {N}")
for r in (1, 2):
    J.append(f"fem_z_ficus z_f_ref_n640_r{r} {FB} --N 640")
    J.append(f"fem_z_ficus z_f_d03_n640_r{r} {FB} --lim 0.3 --lim_mode box --N 640")
    for c in C:
        J.append(f"fem_z_ficus z_f_c{cn(c)}_n640_r{r} {FB} --lim_sr {c} --lim_sr_chk --N 640")
        J.append(f"fem_z_ficus z_f_c{cn(c)}d03_n640_r{r} {FB} --lim_sr {c} --lim 0.3 --lim_sr_chk --N 640")
        for N in (512, 640, 768):
            J.append(f"fem_z_ficus z_f_sb1c{cn(c)}_n{N}_r{r} {FB} --sb 1 --sb_c {c} --N {N}")
for j in J:
    print(j)
