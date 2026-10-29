"""절 Q 2 차 스윕 (1 차에서 K 를 키울수록 lego 오차가 내려감 -- K 32 에서 11.2~11.8 %): 큰 K 쪽을 촘촘히.
std (절 P 판) 도 같은 칸을 돌려 (a) 의 효과와 K 효과를 가른다. 비측정 병렬. 형식은 fem_q_jobs.py 와 같다."""
W = "/home/dkta/work"
V = {"a4": "--mass tet --rep_w tet4", "aL": "--mass tet --rep_w lam", "std": "--mass tet"}
LB = "--mode lag --opt --no_dump --floor vel"
FD = f"{W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
J = []
for v in V:
    for N in (128, 160, 192, 208, 224, 240, 256):
        for K in (24, 32, 48, 64, 96, 128):
            if K < N:
                J.append(f"fem_q_lego q_l_{v}_vel_n{N}_k{K} {LB} {V[v]} --N {N} --rep {K} --rep_mode vel")
    for N in (192, 240):
        for K in (32, 48, 64):
            J.append(f"fem_q_lego q_l_{v}_full_n{N}_k{K} {LB} {V[v]} --N {N} --rep {K} --rep_mode full")
    for N in (576, 640, 768):
        for K in (192, 256, 384):
            J.append(f"fem_q_ficus q_f_{v}_vel_n{N}_k{K} {FB} {V[v]} --N {N} --rep {K} --rep_mode vel")
for j in J:
    print(j)
