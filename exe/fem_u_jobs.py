"""절 U (사용자 10-29 "그것도 일단 해 보라고 해") 스윕 작업 목록 -- 한 줄 = "<EX_ROOT> <태그> <fem_ex_sim 인자...>".
노드 적분 FEM (--nint, β 0 = 순수 노드 적분): 틀·질량 (m_T/4)·F̄ 상태·바닥 (lego 절 O 속도 바닥, ficus 없음)·G2P barycentric 은 절 R 판 그대로, 내부력만 노드 응력.
후보: 이동 제한 {끔 (o), 켬 box det A_T ≥ 0.3 (L)}. lego L000 80 프레임 N 128~448 x 3 회, ficus F_pg_v0 125 프레임 N 384~1536 x 2 회. 비측정 (GPU 0~6 나눠 동시에)."""
import sys
W = "/home/dkta/work"
BT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0                               # 안정화 β (기본 0 = 순수 노드 적분)
SFX = "" if BT == 0 else f"_b{int(round(BT * 100)):02d}"
V = {"o": "", "L": "--lim 0.3 --lim_mode box"}
NB = f"--nint --nint_beta {BT:g}"
LB = f"--mode lag --opt --mass tet --no_dump --floor vel --rstat {NB}"
FD = f"{W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --mass tet --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat {NB}"
J = []
for r in (1, 2, 3):
    for N in (128, 160, 192, 224, 256, 320, 384, 448):
        for v in V:
            J.append(f"fem_u_lego u_l{SFX}_{v}_n{N}_r{r} {LB} {V[v]} --N {N}")
for r in (1, 2):
    for N in (384, 448, 512, 640, 768, 1024, 1536):
        for v in V:
            J.append(f"fem_u_ficus u_f{SFX}_{v}_n{N}_r{r} {FB} {V[v]} --N {N}")
for j in J:
    print(j)
