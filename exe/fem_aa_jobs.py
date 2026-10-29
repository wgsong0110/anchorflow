"""절 AA 스윕 (비측정 병렬, GPU 1~6): 셀 에너지 2 차 근사 (--cq, 헤시안 kmanual) 이동 제한 끔 / 켬 (det A ≥ 0.3 상자),
lego L000 80 프레임 x N 192·256 x 5 회, ficus F_pg_v0 125 프레임 N 640 x 2 회; 이웃 벌어짐 비교용 기준 (절 R 판, --cq 없음) 도 같은 지표로.
줄 형식: <EX_ROOT> <태그> <fem_ex_sim 인자...>"""
LB = "--mode lag --opt --mass tet --no_dump --floor vel --rstat --fspread 10"
FD = "/home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --mass tet --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --fspread 25"
J = []
for r in range(1, 6):
    for N in (192, 256):
        J.append(f"fem_aa_lego aa_l_cq_n{N}_r{r} {LB} --cq --N {N}")
        J.append(f"fem_aa_lego aa_l_cqd03_n{N}_r{r} {LB} --cq --lim 0.3 --lim_mode box --N {N}")
J.append(f"fem_aa_lego aa_l_ref_n192_r1 {LB} --N 192")
J.append(f"fem_aa_lego aa_l_refd03_n192_r1 {LB} --lim 0.3 --lim_mode box --N 192")
for r in (1, 2):
    J.append(f"fem_aa_ficus aa_f_cq_n640_r{r} {FB} --cq --N 640")
    J.append(f"fem_aa_ficus aa_f_cqd03_n640_r{r} {FB} --cq --lim 0.3 --lim_mode box --N 640")
J.append(f"fem_aa_ficus aa_f_ref_n640_r1 {FB} --N 640")
for j in J:
    print(j)
