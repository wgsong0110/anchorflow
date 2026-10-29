"""절 T (사용자 10-29 "메시 넓히지 말고 내가 시킨 대로만 해서 다시 해 봐") 스윕 작업 목록 -- 한 줄 = "<EX_ROOT> <태그> <fem_ex_sim 인자...>".
절 S 와 같은 틀·같은 후보, 바뀐 것은 메시뿐: --bs_mesh occ = 절 R 판과 같은 메시 (입자가 든 셀과 그 꼭짓점만, 팽창 없음),
B-스플라인 가중은 메시에 있는 노드·셀에만 쓰고 입자마다 메시 안 가중 합으로 나눠 정규화 (Σ_n w = 1·Σ_c w = 1 -- 총 질량·운동량·부피 보존).
후보: G2P {bary, bs} x 이동 제한 {끔 (주력), 켬 (box det A_T ≥ 0.3)}. lego L000 80 프레임 N 64~384 x 5 회, ficus F_pg_v0 125 프레임 N 384~1024 x 2 회. 진단 _dg."""
W = "/home/dkta/work"
V = {"gb": "--g2p bary", "gs": "--g2p bs", "gbL": "--g2p bary --lim 0.3 --lim_mode box", "gsL": "--g2p bs --lim 0.3 --lim_mode box"}
LB = "--mode lag --opt --p2g bs --bs_mesh occ --no_dump --floor vel --rstat --bsdiag -1"
FD = f"{W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --p2g bs --bs_mesh occ --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --bsdiag -1"
J = []
for v in ("gb", "gs"):
    J.append(f"fem_t_lego t_l_{v}_n192_dg {LB.replace('--bsdiag -1', '--bsdiag 2')} {V[v]} --N 192")
    J.append(f"fem_t_ficus t_f_{v}_n640_dg {FB.replace('--bsdiag -1', '--bsdiag 10')} {V[v]} --N 640")
for r in (1, 2, 3, 4, 5):
    for N in (64, 96, 128, 160, 192, 224, 256, 384):
        for v in V:
            J.append(f"fem_t_lego t_l_{v}_n{N}_r{r} {LB} {V[v]} --N {N}")
for r in (1, 2):
    for N in (384, 448, 512, 640, 768, 1024):
        for v in V:
            J.append(f"fem_t_ficus t_f_{v}_n{N}_r{r} {FB} {V[v]} --N {N}")
for j in J:
    print(j)
