"""절 W (사용자 10-29: 질량·운동량도 셀 중심 커널로 셀에, 노드는 m_T/4 집중) 스윕 작업 목록 -- 한 줄 = "<EX_ROOT> <태그> <fem_ex_sim 인자...>".
절 T 와 같은 틀·메시·셀 가중 (--p2g bs --bs_mesh occ), 바뀐 것은 --bs_mass cell 뿐. 후보: G2P {bary, bs} x 이동 제한 {끔, 켬 (box det A_T ≥ 0.3)}.
lego L000 80 프레임 N 192·256·448 x 3 회, ficus F_pg_v0 125 프레임 N 640·1024·1536 x 2 회. 진단 _dg (노드 질량 분포·국소 안정 비, 10 프레임마다)."""
W = "/home/dkta/work"
V = {"gb": "--g2p bary", "gs": "--g2p bs", "gbL": "--g2p bary --lim 0.3 --lim_mode box", "gsL": "--g2p bs --lim 0.3 --lim_mode box"}
LB = "--mode lag --opt --p2g bs --bs_mesh occ --bs_mass cell --no_dump --floor vel --rstat --bsdiag -1"
FD = f"{W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = (f"--mode lag --opt --p2g bs --bs_mesh occ --bs_mass cell --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 "
      f"--floor none --cfg_bc --region_density --pg_vdamp --rstat --bsdiag -1")
J = []
for v in ("gb", "gs"):
    J.append(f"fem_w_lego w_l_{v}_n192_dg {LB.replace('--bsdiag -1', '--bsdiag 10')} {V[v]} --N 192")
    J.append(f"fem_w_ficus w_f_{v}_n640_dg {FB.replace('--bsdiag -1', '--bsdiag 25')} {V[v]} --N 640")
for r in (1, 2, 3):
    for N in (192, 256, 448):
        for v in V:
            J.append(f"fem_w_lego w_l_{v}_n{N}_r{r} {LB} {V[v]} --N {N}")
for r in (1, 2):
    for N in (640, 1024, 1536):
        for v in V:
            J.append(f"fem_w_ficus w_f_{v}_n{N}_r{r} {FB} {V[v]} --N {N}")
for j in J:
    print(j)
