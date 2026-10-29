"""절 T 2 차 스윕: 1 차 연기 시험의 국소 안정 비 (lego N192 최대 r 2.67 -> 필요 N ≈ 513, ficus N640 최대 r 3.98 -> 필요 N ≈ 2548) 를 넘는 N.
같은 판 (--bs_mesh occ), 같은 후보 4 개. lego 80 프레임 N 448~1024 x 5 회, ficus 125 프레임 N 1536~4096 x 2 회."""
W = "/home/dkta/work"
V = {"gb": "--g2p bary", "gs": "--g2p bs", "gbL": "--g2p bary --lim 0.3 --lim_mode box", "gsL": "--g2p bs --lim 0.3 --lim_mode box"}
LB = "--mode lag --opt --p2g bs --bs_mesh occ --no_dump --floor vel --rstat --bsdiag -1"
FD = f"{W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --p2g bs --bs_mesh occ --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --bsdiag -1"
J = []
for r in (1, 2, 3, 4, 5):
    for N in (448, 512, 640, 768, 1024):
        for v in V:
            J.append(f"fem_t_lego t_l_{v}_n{N}_r{r} {LB} {V[v]} --N {N}")
for r in (1, 2):
    for N in (1536, 2048, 2560, 3072, 4096):
        for v in V:
            J.append(f"fem_t_ficus t_f_{v}_n{N}_r{r} {FB} {V[v]} --N {N}")
for j in J:
    print(j)
