"""절 AB 2b 차 (비측정, 카드 1·2): navg β1 + 클램프 5 는 오차 9.5 % 대지만 일부 회가 80 프레임 끝 무렵 폭주 시작 (끝 퍼짐 1.9) -- 클램프 c 3·2 로 조여 안정·오차를 본다.
절 AC 빠른 경로 (--cfuse --cfuse_mode 2). 줄 형식: <EX_ROOT> <태그> <fem_ex_sim 인자...>"""
B = "--mode lag --opt --mass tet --no_dump --floor vel --rstat --cfuse --cfuse_mode 2"
FD = "/home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --mass tet --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --cfuse --cfuse_mode 2"
NV = "--fupd navg --fupd_b 1"
J = []
for r in (1, 2, 3, 4):
    J.append(f"fem_ab_lego ab2_l_navg1c3_n256_r{r} {B} {NV} --lim_sr 3 --N 256")
    J.append(f"fem_ab_lego ab2_l_navg1c2_n256_r{r} {B} {NV} --lim_sr 2 --N 256")
    if r <= 2:
        J.append(f"fem_ab_lego ab2_l_Tc3_n256_r{r} {B} --lim_sr 3 --N 256")
        J.append(f"fem_ab_lego ab2_l_navg1c3_n224_r{r} {B} {NV} --lim_sr 3 --N 224")
        J.append(f"fem_ab_lego ab2_l_navg1c2_n224_r{r} {B} {NV} --lim_sr 2 --N 224")
J += [f"fem_ab_ficus ab2_f_navg1c{c}_n640_r1 {FB} {NV} --lim_sr {c} --N 640" for c in (3, 2)]
J += [f"fem_ab_lego ab2_l_navg1_n256_r{r} {B} {NV} --lim_sr 5 --N 256" for r in (6, 7, 8, 9)]
for j in J:
    print(" ".join(j.split()))
