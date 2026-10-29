"""절 Q (사용자 10-29 "GPU 남잖아. 그냥 세 가지 다 해 보면 안 돼?") 1 차 스윕 작업 목록 -- 한 줄 = "<EX_ROOT> <태그> <fem_ex_sim 인자...>".
후보 (모두 lag 명시적 + 서브스텝 속도 바닥 + K 서브스텝마다 재평균, 절 P 코드 경로):
  a4 = (a) G2P 속도도 사면체 등분 ¼ Σ v_n (P2G m_p/4 와 같은 가중, 노드 질량 m_T/4 그대로)
  aL = (a) P2G 속도 가중도 λ m_p (정규화 Σλm, 동역학 노드 질량 m_T/4 그대로)
  aB = (a) 노드 질량 정의까지 Σλm (--mass bary: P2G·G2P·질량 모두 λ)
  b  = (b) APIC (--affine: G2P 가 C = Σ v_n ⊗ ∇λ_n A⁻¹ 저장, P2G m_p (v_p + C_p (x_n - x_p)))
  bL / bB = (a)·(b) 조합 (APIC + λ 가중 / APIC + Σλm 질량)
  std = 절 P 판 그대로 (회귀 확인)
lego L000 80 프레임 N 64~240 x K 1~32 (vel·full; b* 의 vel 은 항등이라 확인 몇 개), ficus 125 프레임 N 512·640·768 x K 끔·256·128·64·32."""
W = "/home/dkta/work"
V = {"a4": "--mass tet --rep_w tet4", "aL": "--mass tet --rep_w lam", "aB": "--mass bary",
     "b": "--mass tet --affine", "bL": "--mass tet --affine --rep_w lam", "bB": "--mass bary --affine", "std": "--mass tet"}
LB = "--mode lag --opt --no_dump --floor vel"
FD = f"{W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
J = []
for v in ("a4", "aL", "aB", "b", "bL", "bB"):
    modes = ("vel", "full") if not v.startswith("b") else ("full",)
    for md in modes:
        for N in (64, 96, 128, 160, 192, 240):
            for K in (1, 2, 4, 8, 16, 32):
                J.append(f"fem_q_lego q_l_{v}_{md}_n{N}_k{K} {LB} {V[v]} --N {N} --rep {K} --rep_mode {md}")
    J.append(f"fem_q_lego q_l_{v}_n192_k0 {LB} {V[v]} --N 192")
    if v.startswith("b"):                                                           # vel 재평균 APIC = 항등 (확인)
        J.append(f"fem_q_lego q_l_{v}_vel_n192_k16 {LB} {V[v]} --N 192 --rep 16 --rep_mode vel")
    for N in (512, 640, 768):
        J.append(f"fem_q_ficus q_f_{v}_n{N}_k0 {FB} {V[v]} --N {N}")
        for md in modes:
            for K in (256, 128, 64, 32):
                J.append(f"fem_q_ficus q_f_{v}_{md}_n{N}_k{K} {FB} {V[v]} --N {N} --rep {K} --rep_mode {md}")
    if v.startswith("b"):
        J.append(f"fem_q_ficus q_f_{v}_vel_n768_k128 {FB} {V[v]} --N 768 --rep 128 --rep_mode vel")
# 회귀 확인 (절 P 판 그대로, 이 커밋에서)
J.append(f"fem_q_lego q_l_std_vel_n192_k16 {LB} {V['std']} --N 192 --rep 16 --rep_mode vel")
J.append(f"fem_q_lego q_l_std_full_n240_k16 {LB} {V['std']} --N 240 --rep 16 --rep_mode full")
J.append(f"fem_q_ficus q_f_std_n768_k0 {FB} {V['std']} --N 768")
J.append(f"fem_q_ficus q_f_std_full_n768_k256 {FB} {V['std']} --N 768 --rep 256 --rep_mode full")
for j in J:
    print(j)
