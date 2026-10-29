"""절 X (사용자 10-29 "해봐"): lag 판 P2G -- 질량·운동량 barycentric Σλm (자기 사면체 네 노드, --mass bary), 셀 F̄_T 부피 가중·V_T 부피 합 (기존 ex_split 그대로),
적분점 = 셀 안 입자 질량점 (선형 사면체라 ∇λ·힘 불변 -- exe/fem_x_qpt.py 로 수치 확인). 나머지 (셀 FCR 응력, 절 O 속도 바닥 (접촉 분배도 λ m_p), G2P barycentric,
이동 제한 끔/켬 box 0.3) 는 절 R 판 그대로. 한 줄 = "<EX_ROOT> <태그> <fem_ex_sim 인자...>".
lego L000 80 프레임: 제한 끔/켬 x N 192·448·1024 x 3 회. ficus F_pg_v0 125 프레임: N 640·1536 x 제한 끔/켬 x 2 회.
진단 _dg (--lagdiag --lagdump 3: 발산 노드 질량 vs 힘 (절 U 방식, fem_dv_stats.py), 적분점·질량 검산 (fem_x_qpt.py))."""
W = "/home/dkta/work"
V = {"l0": "", "lL": "--lim 0.3 --lim_mode box"}
LB = "--mode lag --opt --mass bary --no_dump --floor vel --rstat"
FD = f"{W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = (f"--mode lag --opt --mass bary --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 "
      f"--floor none --cfg_bc --region_density --pg_vdamp --rstat")
DG = "--lagdiag --lagdump 3"
J = [f"fem_x_lego x_l_l0_n192_dg {LB} {DG} --N 192",
     f"fem_x_lego x_l_lL_n192_dg {LB} {DG} {V['lL']} --N 192",
     f"fem_x_lego x_l_l0_n1024_dg {LB} {DG} --N 1024",
     f"fem_x_ficus x_f_l0_n640_dg {FB} {DG} --N 640",
     f"fem_x_ficus x_f_l0_n1536_dg {FB} {DG} --N 1536"]
for r in (1, 2, 3):
    for N in (192, 448, 1024):
        for v in V:
            J.append(f"fem_x_lego x_l_{v}_n{N}_r{r} {LB} {V[v]} --N {N}")
for r in (1, 2):
    for N in (640, 1536):
        for v in V:
            J.append(f"fem_x_ficus x_f_{v}_n{N}_r{r} {FB} {V[v]} --N {N}")
for j in J:
    print(" ".join(j.split()))
