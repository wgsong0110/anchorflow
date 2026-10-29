"""절 R (사용자 10-29 "두 가지 다 해 봐") 1 차 스윕 작업 목록 -- 한 줄 = "<EX_ROOT> <태그> <fem_ex_sim 인자...>".
틀 = 프레임마다 P2G 1 회 -> 사면체 메시 명시적 서브스텝 N (lag, 재평균 없음, 노드 질량 m_T/4) -> G2P 1 회. lego 바닥 = 절 O 속도 바닥 (frac + 비관통), ficus 바닥 없음.
후보 (모두 --rstat 통계 경로 ex_tet_force_R):
  base    = 장벽·제한 없음 (같은 커널, κ 0 -- 기준)
  fX1/fX10 = (1) FCR + κ μ b(det(A_T F̄_T); Ĵ 0.3)  κ 1 (기존 기본) / 10
  fA1     = (1) 같은 장벽을 det A_T 에 (κ 1, Ĵ 0.3)
  l01/l03 = (2) 서브스텝마다 노드 이동 제한 det A_T ≥ 0.1 / 0.3 (8 색 Gauss-Seidel, 속도도 α 배)
  l01k    = (2) det A_T ≥ 0.1, 속도는 그대로 (이동만 줄임)
  fX1l01 / fA1l01 = (1)+(2)
lego L000 80 프레임 N 64~384 x 3 회 (원자 합 순서로 발산 여부가 갈린다), ficus F_pg_v0 125 프레임 N 512·640·768 x 1 회."""
W = "/home/dkta/work"
V = {"base": "", "fX1": "--fb 1", "fX10": "--fb 10", "fA1": "--fb 1 --fb_on A",
     "l01": "--lim 0.1", "l03": "--lim 0.3", "l01k": "--lim 0.1 --lim_v keep",
     "fX1l01": "--fb 1 --lim 0.1", "fA1l01": "--fb 1 --fb_on A --lim 0.1"}
LB = "--mode lag --opt --mass tet --no_dump --floor vel --rstat"
FD = f"{W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --mass tet --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat"
J = []
for r in (1, 2, 3):
    for v in V:
        for N in (64, 96, 128, 160, 192, 256, 384):
            J.append(f"fem_r_lego r_l_{v}_n{N}_r{r} {LB} {V[v]} --N {N}")
for v in V:
    if v in ("fX10", "l01k"):
        continue
    for N in (512, 640, 768):
        J.append(f"fem_r_ficus r_f_{v}_n{N} {FB} {V[v]} --N {N}")
for j in J:
    print(j)
