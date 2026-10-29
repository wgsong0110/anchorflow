"""절 S (사용자 10-29 "속도 질량 변형구배 나눠주는 방법을 그렇게 해서 실행해 봐") 스윕 작업 목록 -- 한 줄 = "<EX_ROOT> <태그> <fem_ex_sim 인자...>".
틀 = 프레임마다 P2G 1 회 (2 차 B-스플라인: 노드 질량·운동량 27 노드, 셀 F̄·V 셀 중심 27 셀) -> 사면체 메시 명시적 서브스텝 N -> G2P 1 회.
lego 바닥 = 절 O 속도 바닥 (frac + 비관통, 접촉 분배는 B-스플라인 질량 가중), ficus 바닥 없음.
후보: G2P {bary (사면체 barycentric), bs (B-스플라인)} x 이동 제한 {끔 (주력 -- 사용자 질문 "가벼운 노드가 없어지면 제한 없이도 안 터지나"), 켬 (box det A_T ≥ 0.3)}.
lego L000 80 프레임 N 64~384 x 5 회 (원자 합 순서로 발산 여부가 갈린다), ficus F_pg_v0 125 프레임 N 384~1024 x 2 회. 진단 (질량 분포 프레임마다) 은 _dg 태그."""
W = "/home/dkta/work"
V = {"gb": "--g2p bary", "gs": "--g2p bs", "gbL": "--g2p bary --lim 0.3 --lim_mode box", "gsL": "--g2p bs --lim 0.3 --lim_mode box"}
LB = "--mode lag --opt --p2g bs --no_dump --floor vel --rstat --bsdiag -1"
FD = f"{W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --p2g bs --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --bsdiag -1"
J = []
for v in ("gb", "gs"):
    J.append(f"fem_s_lego s_l_{v}_n192_dg {LB.replace('--bsdiag -1', '--bsdiag 2')} {V[v]} --N 192")
    J.append(f"fem_s_ficus s_f_{v}_n640_dg {FB.replace('--bsdiag -1', '--bsdiag 10')} {V[v]} --N 640")
for r in (1, 2, 3, 4, 5):
    for N in (64, 96, 128, 160, 192, 224, 256, 384):
        for v in V:
            J.append(f"fem_s_lego s_l_{v}_n{N}_r{r} {LB} {V[v]} --N {N}")
for r in (1, 2):
    for N in (384, 448, 512, 640, 768, 1024):
        for v in V:
            J.append(f"fem_s_ficus s_f_{v}_n{N}_r{r} {FB} {V[v]} --N {N}")
for j in J:
    print(j)
