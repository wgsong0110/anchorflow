"""절 AB 2 차 스윕 (비측정, 카드 1·2): 1 차 (N192, 클램프 5) 에서 고른 F 갱신 방식을 절 AC 빠른 경로 (--cfuse --cfuse_mode 2, 결과 불변 구현) 위에서 -- lego N256 x 5·N224 x 3·N192 x 3, ficus N640.
이웃 벌어짐은 회 1 에만. 줄 형식: <EX_ROOT> <태그> <fem_ex_sim 인자...>"""
LB = "--mode lag --opt --mass tet --no_dump --floor vel --rstat --lim_sr 5 --cfuse --cfuse_mode 2"
FD = "/home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --mass tet --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --lim_sr 5 --cfuse --cfuse_mode 2 --fspread 62"
V = {"T": "", "navg1": "--fupd navg --fupd_b 1", "nab1b0p2": "--fupd navgbar --fupd_b 1 --fupd_b2 0.2", "nbar0p2": "--fupd nbar --fupd_b 0.2",
     "bar1": "--fupd bar --fupd_b 1", "nbar0p05": "--fupd nbar --fupd_b 0.05", "nab1b0p05": "--fupd navgbar --fupd_b 1 --fupd_b2 0.05"}
L = lambda k, N, r: f"fem_ab_lego ab2_l_{k}_n{N}_r{r} {LB} {V[k]} {'--fspread 40' if r == 1 else ''} --N {N}"
J = ["fem_ab_lego s2_navg1_n192 " + LB + " --frames 5 --fupd navg --fupd_b 1 --N 192",
     "fem_ab_lego s2_nab1b0_n192 " + LB + " --frames 5 --fupd navgbar --fupd_b 1 --fupd_b2 0 --N 192"]          # 연기: β2 0 = navg 1 과 같아야
J += [L(k, 256, 1) for k in V]
J += [f"fem_ab_ficus ab2_f_{k}_n640_r1 {FB} {V[k]} --N 640" for k in ("T", "navg1", "nab1b0p2", "nbar0p2", "nab1b0p05")]
J += [L(k, 256, 2) for k in V]
J += [L(k, N, r) for r in (1, 2) for N in (224, 192) for k in ("navg1", "nab1b0p2", "nbar0p2", "nab1b0p05")]
J += [L(k, 256, r) for r in (3, 4, 5) for k in V]
J += [L(k, N, 3) for N in (224, 192) for k in ("navg1", "nab1b0p2", "nbar0p2", "nab1b0p05")]
for j in J:
    print(" ".join(j.split()))
